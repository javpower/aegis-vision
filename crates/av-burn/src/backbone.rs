//! CSP-ELAN 骨干（ultralytics YOLOv8 backbone 层 0-9 同构）的 burn 移植。
//!
//! tch 参考实现：`crates/av-tasks/src/backbone_cspelan.rs`（先读后写，结构/
//! 缩放/抽头约定逐条对齐）。层表：
//!
//! | 层 | 类型 | 输入→输出通道（基准） | stride 累计 |
//! |----|------|----------------------|-------------|
//! | 0  | Conv 3×3 s2 | 3 → 64   | P1/2  |
//! | 1  | Conv 3×3 s2 | 64 → 128 | P2/4  |
//! | 2  | C2f (n=3)   | 128 → 128 |      |
//! | 3  | Conv 3×3 s2 | 128 → 256 | P3/8 |
//! | 4  | C2f (n=6)   | 256 → 256 | → 抽头 P3 |
//! | 5  | Conv 3×3 s2 | 256 → 512 | P4/16 |
//! | 6  | C2f (n=6)   | 512 → 512 | → 抽头 P4 |
//! | 7  | Conv 3×3 s2 | 512 → 1024 | P5/32 |
//! | 8  | C2f (n=3)   | 1024 → 1024 |    |
//! | 9  | SPPF 5×5    | 1024 → 1024 | → 抽头 P5 |
//!
//! - 通道缩放 `make_divisible(ch × width, 8)`（下限 8），深度缩放
//!   `max(round(n × depth), 1)`——与 ultralytics 约定一致（手算对照见测试）。
//! - `Conv = conv(bias=false) + BN(eps 1e-3) + SiLU`；C2f 内 Bottleneck
//!   e=1.0 同宽 + shortcut；SPPF 三级串联 5×5 maxpool（ceil=false，对齐 tch）。
//! - **差异（诚实边界）**：burn 无 VarStore，Param 名由字段路径生成，不做
//!   `model.N.*` 命名对齐（预训练导入不在 spike 范围）；BN 训练/推理语义由
//!   burn 后端 `ad_enabled` 自动切换（训练 = batch 统计 + running 更新）。

use av_core::error::{AvError, AvResult};
// burn 的 #[derive(Module)] 展开引用 `burn::` 路径，需要该别名（burn 自身
// 源码的同款约定）。
use burn_core as burn;
use burn_core::module::Module;
use burn_core::tensor::backend::Backend;
use burn_core::tensor::{Tensor, activation::silu, module::max_pool2d};
use burn_nn::modules::conv::{Conv2d, Conv2dConfig};
use burn_nn::modules::norm::{BatchNorm, BatchNormConfig};
use burn_nn::PaddingConfig2d;

/// 注册表族名（与 av-tasks::backbone_cspelan::FAMILY_NAME 一致）。
pub const FAMILY_NAME: &str = "csp-elan";

/// yaml 基准通道（ultralytics yolov8.yaml backbone 列）。
const BASE_CHANNELS: [usize; 5] = [64, 128, 256, 512, 1024];
/// yaml C2f 重复数基准（乘 depth）。
const BASE_REPEATS: [usize; 4] = [3, 6, 6, 3];
/// SPPF 池化核（5×5，pad 2，stride 1）。
const SPPF_POOL: usize = 5;

/// ultralytics `make_divisible(ch × width, 8)`，下限 8。
pub fn scale_channels(ch: usize, width: f32) -> usize {
    let v = (ch as f32 * width / 8.0).ceil() as i64 * 8;
    v.max(8) as usize
}

/// ultralytics `max(round(n × depth), 1)`。
pub fn scale_repeats(n: usize, depth: f32) -> usize {
    ((n as f32 * depth).round() as i64).max(1) as usize
}

/// ultralytics `Conv` 配置：conv(bias=False) + k/2 对称 padding。
fn conv_cfg(in_ch: usize, out_ch: usize, k: usize, stride: usize) -> Conv2dConfig {
    let p = k / 2;
    Conv2dConfig::new([in_ch, out_ch], [k, k])
        .with_stride([stride, stride])
        .with_padding(PaddingConfig2d::Explicit(p, p, p, p))
        .with_bias(false)
}

/// conv(bias=false) + BN(eps 1e-3) + SiLU。
#[derive(Module, Debug)]
pub struct ConvBnSilu<B: Backend> {
    conv: Conv2d<B>,
    bn: BatchNorm<B>,
}

impl<B: Backend> ConvBnSilu<B> {
    pub fn new(in_ch: usize, out_ch: usize, k: usize, stride: usize, device: &B::Device) -> Self {
        Self {
            conv: conv_cfg(in_ch, out_ch, k, stride).init(device),
            // eps 对齐 ultralytics（1e-3）；momentum 保持默认 0.1。
            bn: BatchNormConfig::new(out_ch).with_epsilon(1e-3).init(device),
        }
    }

    /// forward：burn 的 BatchNorm 依 ad_enabled 自动选 batch-stat（训练）/
    /// running-stat（推理），无需显式 train 标志。
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        silu(self.bn.forward(self.conv.forward(x)))
    }
}

/// ultralytics `Bottleneck`（C2f 内 e=1.0：不降通道）：两个 3×3 同宽卷积 +
/// shortcut 相加。
#[derive(Module, Debug)]
pub struct Bottleneck<B: Backend> {
    cv1: ConvBnSilu<B>,
    cv2: ConvBnSilu<B>,
    shortcut: bool,
}

impl<B: Backend> Bottleneck<B> {
    fn new(c: usize, shortcut: bool, device: &B::Device) -> Self {
        Self {
            cv1: ConvBnSilu::new(c, c, 3, 1, device),
            cv2: ConvBnSilu::new(c, c, 3, 1, device),
            shortcut,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let y = self.cv2.forward(self.cv1.forward(x.clone()));
        if self.shortcut {
            x + y
        } else {
            y
        }
    }
}

/// ultralytics `C2f`：cv1 → chunk(2) → n 级 Bottleneck 串联 → concat → cv2。
#[derive(Module, Debug)]
pub struct C2f<B: Backend> {
    cv1: ConvBnSilu<B>,
    cv2: ConvBnSilu<B>,
    m: Vec<Bottleneck<B>>,
}

impl<B: Backend> C2f<B> {
    fn new(c1: usize, c2: usize, n: usize, shortcut: bool, device: &B::Device) -> Self {
        let hidden = c2 / 2;
        Self {
            cv1: ConvBnSilu::new(c1, 2 * hidden, 1, 1, device),
            cv2: ConvBnSilu::new((2 + n) * hidden, c2, 1, 1, device),
            m: (0..n).map(|_| Bottleneck::new(hidden, shortcut, device)).collect(),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let ys = self.cv1.forward(x).chunk(2, 1);
        let mut parts = vec![ys[0].clone(), ys[1].clone()];
        for b in &self.m {
            let prev = parts.last().expect("chunk 非空").clone();
            parts.push(b.forward(prev));
        }
        self.cv2.forward(Tensor::cat(parts, 1))
    }
}

/// ultralytics `SPPF`：cv1 → 三级串联 5×5 maxpool → concat → cv2。
#[derive(Module, Debug)]
pub struct Sppf<B: Backend> {
    cv1: ConvBnSilu<B>,
    cv2: ConvBnSilu<B>,
}

impl<B: Backend> Sppf<B> {
    fn new(c1: usize, c2: usize, device: &B::Device) -> Self {
        let half = c1 / 2;
        Self {
            cv1: ConvBnSilu::new(c1, half, 1, 1, device),
            cv2: ConvBnSilu::new(4 * half, c2, 1, 1, device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let y = self.cv1.forward(x);
        // ceil_mode=false 对齐 tch max_pool2d(..., ceil_mode=false)。
        let y1 = max_pool2d(y.clone(), [SPPF_POOL; 2], [1, 1], [2, 2], [1, 1], false);
        let y2 = max_pool2d(y1.clone(), [SPPF_POOL; 2], [1, 1], [2, 2], [1, 1], false);
        let y3 = max_pool2d(y2.clone(), [SPPF_POOL; 2], [1, 1], [2, 2], [1, 1], false);
        self.cv2.forward(Tensor::cat(vec![y, y1, y2, y3], 1))
    }
}

/// 层 0-9 的枚举（burn Module derive 支持枚举 + Vec）。
#[derive(Module, Debug)]
pub enum CspLayer<B: Backend> {
    Conv(ConvBnSilu<B>),
    C2f(C2f<B>),
    Sppf(Sppf<B>),
}

impl<B: Backend> CspLayer<B> {
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        match self {
            CspLayer::Conv(c) => c.forward(x),
            CspLayer::C2f(c) => c.forward(x),
            CspLayer::Sppf(s) => s.forward(x),
        }
    }
}

/// 骨干装配参数（width/depth 缩放，同 tch 版 `BackboneCfg` 的语义子集）。
#[derive(Debug, Clone)]
pub struct BackboneCfg {
    pub width: f32,
    pub depth: f32,
}

/// CSP-ELAN（YOLOv8 backbone 层 0-9）：stem×2 → C2f×4（级间 s2 下采样）→
/// SPPF。抽头 P3（layer4 后，stride 8）/ P4（layer6 后，stride 16）/
/// P5（SPPF 后，stride 32）。
#[derive(Module, Debug)]
pub struct CspElanBackbone<B: Backend> {
    layers: Vec<CspLayer<B>>,
    p3_ch: usize,
    p4_ch: usize,
    p5_ch: usize,
}

impl<B: Backend> CspElanBackbone<B> {
    /// 装配 csp-elan；width/depth 必须 > 0（与 tch 版一致显式拒绝非法配置）。
    pub fn new(cfg: &BackboneCfg, device: &B::Device) -> AvResult<Self> {
        if cfg.width <= 0.0 || cfg.depth <= 0.0 {
            return Err(AvError::config(format!(
                "{FAMILY_NAME} 需要 width > 0 且 depth > 0（得到 width={} depth={}）；\
                 导入 yolov8n 权重须 width=0.25 depth=0.33",
                cfg.width, cfg.depth
            )));
        }
        let ch: Vec<usize> =
            BASE_CHANNELS.iter().map(|&c| scale_channels(c, cfg.width)).collect();
        let repeats: Vec<usize> =
            BASE_REPEATS.iter().map(|&n| scale_repeats(n, cfg.depth)).collect();
        let conv = |i: usize, o: usize, k: usize, s: usize| {
            CspLayer::Conv(ConvBnSilu::new(i, o, k, s, device))
        };
        let c2f = |ci: usize, co: usize, n: usize| {
            CspLayer::C2f(C2f::new(ci, co, n, true, device))
        };
        // 层 0-9（ultralytics yolov8.yaml backbone 列），索引即层号。
        let layers = vec![
            conv(3, ch[0], 3, 2),         // 0：P1/2
            conv(ch[0], ch[1], 3, 2),     // 1：P2/4
            c2f(ch[1], ch[1], repeats[0]), // 2
            conv(ch[1], ch[2], 3, 2),     // 3：P3/8
            c2f(ch[2], ch[2], repeats[1]), // 4 → P3 抽头
            conv(ch[2], ch[3], 3, 2),     // 5：P4/16
            c2f(ch[3], ch[3], repeats[2]), // 6 → P4 抽头
            conv(ch[3], ch[4], 3, 2),     // 7：P5/32
            c2f(ch[4], ch[4], repeats[3]), // 8
            CspLayer::Sppf(Sppf::new(ch[4], ch[4], device)), // 9 → P5 抽头
        ];
        Ok(Self { layers, p3_ch: ch[2], p4_ch: ch[3], p5_ch: ch[4] })
    }

    /// 全层前向，返回 (P3, P4, P5)。
    pub fn forward_features(
        &self,
        x: Tensor<B, 4>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        let mut cur = x;
        let mut p3 = None;
        let mut p4 = None;
        let mut p5 = None;
        for (i, layer) in self.layers.iter().enumerate() {
            cur = layer.forward(cur);
            match i {
                4 => p3 = Some(cur.clone()),
                6 => p4 = Some(cur.clone()),
                9 => p5 = Some(cur.clone()),
                _ => {}
            }
        }
        (
            p3.expect("P3（layer4 之后）"),
            p4.expect("P4（layer6 之后）"),
            p5.expect("P5（SPPF 之后）"),
        )
    }

    /// 金字塔通道 (P3, P4, P5)（已按 width 缩放）。
    pub fn pyramid_channels(&self) -> (usize, usize, usize) {
        (self.p3_ch, self.p4_ch, self.p5_ch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NdArrayB;

    fn nano() -> BackboneCfg {
        BackboneCfg { width: 0.25, depth: 0.33 }
    }

    /// 通道缩放手算对照：ultralytics make_divisible(ch × width, 8)，下限 8。
    #[test]
    fn channel_scaling_matches_ultralytics() {
        assert_eq!(scale_channels(64, 0.25), 16);
        assert_eq!(scale_channels(128, 0.25), 32);
        assert_eq!(scale_channels(256, 0.25), 64);
        assert_eq!(scale_channels(512, 0.25), 128);
        assert_eq!(scale_channels(1024, 0.25), 256);
        // width=1.0 → 基准原样
        assert_eq!(scale_channels(1024, 1.0), 1024);
        // 小 width 触底 8
        assert_eq!(scale_channels(64, 0.01), 8, "下限 8");
    }

    /// depth 缩放手算对照：max(round(n × depth), 1)。
    #[test]
    fn depth_scaling_matches_ultralytics() {
        assert_eq!(scale_repeats(3, 0.33), 1);
        assert_eq!(scale_repeats(6, 0.33), 2);
        assert_eq!(scale_repeats(6, 1.0), 6);
        assert_eq!(scale_repeats(3, 0.1), 1, "round 后下限 1");
    }

    /// 装配契约：nano 尺寸通道 16/32/64/128/256；非法 width 显式拒绝。
    #[test]
    fn assembly_contract() {
        let device = burn_ndarray::NdArrayDevice::Cpu;
        let b = CspElanBackbone::<NdArrayB>::new(&nano(), &device).unwrap();
        assert_eq!(b.pyramid_channels(), (64, 128, 256));
        let bad = BackboneCfg { width: 0.0, depth: 0.33 };
        assert!(CspElanBackbone::<NdArrayB>::new(&bad, &device).is_err());
    }

    /// 金字塔契约（对照 tch 版 feature_pyramid_contract）：输入 64×64 →
    /// P3 [2,64,8,8] / P4 [2,128,4,4] / P5 [2,256,2,2]，stride 8/16/32。
    #[test]
    fn feature_pyramid_contract() {
        let device = burn_ndarray::NdArrayDevice::Cpu;
        let b = CspElanBackbone::<NdArrayB>::new(&nano(), &device).unwrap();
        let x = Tensor::<NdArrayB, 4>::ones([2, 3, 64, 64], &device);
        let (p3, p4, p5) = b.forward_features(x);
        assert_eq!(p3.dims(), [2, 64, 8, 8]);
        assert_eq!(p4.dims(), [2, 128, 4, 4]);
        assert_eq!(p5.dims(), [2, 256, 2, 2]);
    }
}
