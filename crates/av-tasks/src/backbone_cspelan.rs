//! CSP-ELAN 骨干（YOLOv8 backbone 逐层同构），供 [`crate::models`] 的
//! 分类 / 检测 / 分割骨干枚举（`family = "csp-elan"`）装配。
//!
//! 结构与层名严格对齐 ultralytics YOLOv8 backbone（yaml 层 0-9：stem×2 +
//! C2f×4 + SPPF，层名 `model.N.*`），预训练权重兼容是硬要求：
//!
//! | ultralytics 层名                  | AV 变量名（VarStore）             | nano 形状      |
//! |-----------------------------------|-----------------------------------|----------------|
//! | `model.0.conv.weight`             | `backbone.0.conv.weight`          | [16,3,3,3]     |
//! | `model.0.bn.running_mean`         | `backbone.0.bn.running_mean`      | [16]           |
//! | `model.2.cv1.conv.weight`         | `backbone.2.cv1.conv.weight`      | [32,32,1,1]    |
//! | `model.2.m.0.cv2.bn.weight`       | `backbone.2.m.0.cv2.bn.weight`    | [16]           |
//! | `model.9.cv2.bn.bias`             | `backbone.9.cv2.bn.bias`          | [256]          |
//!
//! 即 AV 名 = `model.N.` → `backbone.N.`，conv/bn 参数与 BN 统计量**全部同名
//! 对齐**，[`av_pretrain::weight_adapter`] 用一条前缀映射
//! （`configs/cspelan_yolov8n_map.toml`：`^model\.` → `backbone.`）即可导入
//! 官方 `yolov8n.pt` 的**完整骨干**（层 0-9，COCO 预训练）。
//!
//! # 宽度 / 深度缩放（ultralytics 约定）
//!
//! - `width`：`make_divisible(ch × width, 8)`。nano = 0.25 → 16/32/64/128/256；
//!   1.0 → 64/128/256/512/1024（YOLOv8l 量级，约 25M 参数）。
//! - `depth`：C2f 重复数 `max(round(n × depth), 1)`。nano = 0.33 → 1/2/2/1
//!   （yaml n = 3/6/6/3）。
//!
//! 要导入 yolov8n 权重必须 `width = 0.25, depth = 0.33`（形状兼容硬约束，
//! 见 `configs/seg_harness_cspelan.toml`）。
//!
//! # BN 统计量与 train 标志
//!
//! 与 [`crate::backbone_resnet`] 同款方案：tch `nn::batch_norm2d` 的
//! running_mean/running_var 是 VarStore 命名变量（no_train），与普通权重同
//! 一条导入链路；`Cell<bool>` 训练标志默认 false（FrozenBN 推理语义），
//! 引擎经 `TaskModel::set_train` 切换。eps 对齐 ultralytics（1e-3）。
//!
//! # 特征金字塔
//!
//! P3 = layer4 之后（stride 8）、P4 = layer6 之后（stride 16）、
//! P5 = SPPF 之后（stride 32）——与 YOLOv8 neck 的输入约定一致。
//! `forward_pooled` = P5 GAP（分类路径）。

use std::cell::Cell;

use tch::nn;
use tch::nn::{Module, ModuleT};
use tch::Tensor;

use av_core::config::BackboneCfg;
use av_core::error::{AvError, AvResult};
use av_core::traits::{BaseBackbone, BackboneSpec, FeatureMap, FeaturePyramid, LevelSpec};

/// 注册表族名（av-tasks::register_builtin 登记；config.rs 默认值即此）。
pub const FAMILY_NAME: &str = "csp-elan";

/// yaml 基准通道（ultralytics yolov8.yaml backbone 列）。
const BASE_CHANNELS: [i64; 5] = [64, 128, 256, 512, 1024];
/// yaml C2f 重复数基准（乘 depth）。
const BASE_REPEATS: [i64; 4] = [3, 6, 6, 3];
/// SPPF 隐藏通道减半系数 / 池化核。
const SPPF_POOL: i64 = 5;

/// ultralytics `make_divisible(ch × width, 8)`，下限 8。
fn scale_channels(ch: i64, width: f32) -> i64 {
    let v = (ch as f32 * width / 8.0).ceil() as i64 * 8;
    v.max(8)
}

/// ultralytics `max(round(n × depth), 1)`。
fn scale_repeats(n: i64, depth: f32) -> i64 {
    ((n as f32 * depth).round() as i64).max(1)
}

/// ultralytics `Conv`：conv(bias=False) + BN(eps 1e-3) + SiLU。
struct ConvBnSilu {
    conv: nn::Conv2D,
    bn: nn::BatchNorm,
}

impl ConvBnSilu {
    fn new(p: &nn::Path, in_ch: i64, out_ch: i64, k: i64, stride: i64) -> Self {
        Self {
            conv: nn::conv2d(
                p / "conv",
                in_ch,
                out_ch,
                k,
                nn::ConvConfig {
                    stride,
                    padding: (k / 2) as i64,
                    bias: false,
                    ..Default::default()
                },
            ),
            bn: nn::batch_norm2d(
                p / "bn",
                out_ch,
                nn::BatchNormConfig { eps: 1e-3, ..Default::default() },
            ),
        }
    }

    fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        self.bn
            .forward_t(&self.conv.forward(x), train)
            .silu()
    }
}

/// ultralytics `Bottleneck`（C2f 内 e=1.0：不降通道）：两个 3×3 同宽卷积，
/// shortcut 相加。
struct Bottleneck {
    cv1: ConvBnSilu,
    cv2: ConvBnSilu,
    shortcut: bool,
}

impl Bottleneck {
    fn new(p: &nn::Path, c: i64, shortcut: bool) -> Self {
        Self {
            cv1: ConvBnSilu::new(&(p / "cv1"), c, c, 3, 1),
            cv2: ConvBnSilu::new(&(p / "cv2"), c, c, 3, 1),
            shortcut,
        }
    }

    fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        let y = self.cv2.forward(&self.cv1.forward(x, train), train);
        if self.shortcut {
            x + y
        } else {
            y
        }
    }
}

/// ultralytics `C2f`：cv1 → chunk(2) → n 级 Bottleneck 串联 → concat → cv2。
struct C2f {
    cv1: ConvBnSilu,
    cv2: ConvBnSilu,
    m: Vec<Bottleneck>,
}

impl C2f {
    fn new(p: &nn::Path, c1: i64, c2: i64, n: i64, shortcut: bool) -> Self {
        let hidden = c2 / 2;
        let mp = p / "m";
        let m = (0..n)
            .map(|i| {
                let bp = &mp / &i.to_string();
                Bottleneck::new(&bp, hidden, shortcut)
            })
            .collect();
        Self {
            cv1: ConvBnSilu::new(&(p / "cv1"), c1, 2 * hidden, 1, 1),
            cv2: ConvBnSilu::new(&(p / "cv2"), (2 + n) * hidden, c2, 1, 1),
            m,
        }
    }

    fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        let ys = self.cv1.forward(x, train).chunk(2, 1);
        let mut parts = vec![ys[0].shallow_clone(), ys[1].shallow_clone()];
        for b in &self.m {
            let prev = parts.last().expect("非空");
            parts.push(b.forward(prev, train));
        }
        self.cv2.forward(&Tensor::cat(&parts, 1), train)
    }
}

/// ultralytics `SPPF`：cv1 → 三级串联 5×5 maxpool → concat → cv2。
struct Sppf {
    cv1: ConvBnSilu,
    cv2: ConvBnSilu,
}

impl Sppf {
    fn new(p: &nn::Path, c1: i64, c2: i64) -> Self {
        let half = c1 / 2;
        Self {
            cv1: ConvBnSilu::new(&(p / "cv1"), c1, half, 1, 1),
            cv2: ConvBnSilu::new(&(p / "cv2"), 4 * half, c2, 1, 1),
        }
    }

    fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        let y = self.cv1.forward(x, train);
        let y1 = y.max_pool2d([SPPF_POOL, SPPF_POOL], [1, 1], [2, 2], [1, 1], false);
        let y2 = y1.max_pool2d([SPPF_POOL, SPPF_POOL], [1, 1], [2, 2], [1, 1], false);
        let y3 = y2.max_pool2d([SPPF_POOL, SPPF_POOL], [1, 1], [2, 2], [1, 1], false);
        self.cv2.forward(&Tensor::cat(&[y, y1, y2, y3], 1), train)
    }
}

/// CSP-ELAN（YOLOv8 backbone 层 0-9）：stem×2 → C2f×4（级间 s2 卷积下采样）
/// → SPPF。层名 `backbone.N.*` 与 ultralytics `model.N.*` 逐字对齐。
pub struct CspElanBackbone {
    layers: Vec<CspLayer>,
    /// 金字塔通道（P3/P4/P5，已按 width 缩放）。
    p3_ch: i64,
    p4_ch: i64,
    p5_ch: i64,
    /// 训练标志（见模块注释；默认 false = FrozenBN 推理语义）。
    train: Cell<bool>,
}

enum CspLayer {
    Conv(ConvBnSilu),
    C2f(C2f),
    Sppf(Sppf),
}

impl CspLayer {
    fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        match self {
            CspLayer::Conv(c) => c.forward(x, train),
            CspLayer::C2f(c) => c.forward(x, train),
            CspLayer::Sppf(s) => s.forward(x, train),
        }
    }
}

impl CspElanBackbone {
    /// 装配 csp-elan。变量名 = ultralytics 层名 `model.N.*` → `backbone.N.*`
    /// （p 为 build_model 传入的 `root / "backbone"`）。
    pub fn new(p: &nn::Path, cfg: &BackboneCfg) -> AvResult<Self> {
        if cfg.width <= 0.0 || cfg.depth <= 0.0 {
            return Err(AvError::config(format!(
                "{FAMILY_NAME} 需要 width > 0 且 depth > 0（得到 width={} depth={}）；\
                 导入 yolov8n 权重须 width=0.25 depth=0.33",
                cfg.width, cfg.depth
            )));
        }
        let ch: Vec<i64> = BASE_CHANNELS.iter().map(|&c| scale_channels(c, cfg.width)).collect();
        let repeats: Vec<i64> = BASE_REPEATS.iter().map(|&n| scale_repeats(n, cfg.depth)).collect();
        let conv = |pp: nn::Path, i: i64, o: i64, k: i64, s: i64| {
            CspLayer::Conv(ConvBnSilu::new(&pp, i, o, k, s))
        };
        // 层 0-9（ultralytics yolov8.yaml backbone 列），索引即变量名数字。
        let c2f = |pp: nn::Path, ci: i64, co: i64, n: i64| {
            CspLayer::C2f(C2f::new(&pp, ci, co, n, true))
        };
        let layers = vec![
            conv(p / "0", 3, ch[0], 3, 2),                              // P1/2
            conv(p / "1", ch[0], ch[1], 3, 2),                          // P2/4
            c2f(p / "2", ch[1], ch[1], repeats[0]),
            conv(p / "3", ch[1], ch[2], 3, 2),                          // P3/8
            c2f(p / "4", ch[2], ch[2], repeats[1]),
            conv(p / "5", ch[2], ch[3], 3, 2),                          // P4/16
            c2f(p / "6", ch[3], ch[3], repeats[2]),
            conv(p / "7", ch[3], ch[4], 3, 2),                          // P5/32
            c2f(p / "8", ch[4], ch[4], repeats[3]),
            CspLayer::Sppf(Sppf::new(&(p / "9"), ch[4], ch[4])),
        ];
        Ok(Self {
            layers,
            p3_ch: ch[2],
            p4_ch: ch[3],
            p5_ch: ch[4],
            train: Cell::new(false),
        })
    }

    /// 切换训练/推理语义（见模块注释）。
    pub fn set_train(&self, train: bool) {
        self.train.set(train);
    }

    pub fn stride_channels(&self, stride: u32) -> AvResult<i64> {
        match stride {
            8 => Ok(self.p3_ch),
            16 => Ok(self.p4_ch),
            32 => Ok(self.p5_ch),
            other => Err(AvError::shape(format!(
                "{FAMILY_NAME} 不存在 stride {other} 的特征层（金字塔 = P3/P4/P5 = 8/16/32）"
            ))),
        }
    }

    pub fn pooled_channels(&self) -> i64 {
        self.p5_ch
    }

    /// 全层前向，返回 (P3, P4, P5)。
    fn forward_all(&self, x: &Tensor, train: bool) -> (Tensor, Tensor, Tensor) {
        let mut cur = x.shallow_clone();
        let mut p3 = None;
        let mut p4 = None;
        let mut p5 = None;
        for (i, layer) in self.layers.iter().enumerate() {
            cur = layer.forward(&cur, train);
            match i {
                4 => p3 = Some(cur.shallow_clone()),
                6 => p4 = Some(cur.shallow_clone()),
                9 => p5 = Some(cur.shallow_clone()),
                _ => {}
            }
        }
        (
            p3.expect("P3（layer4 之后）"),
            p4.expect("P4（layer6 之后）"),
            p5.expect("P5（SPPF 之后）"),
        )
    }
}

impl BaseBackbone for CspElanBackbone {
    fn forward_features(&self, x: &Tensor) -> AvResult<FeaturePyramid> {
        let (p3, p4, p5) = self.forward_all(x, self.train.get());
        let mut pyramid = FeaturePyramid::default();
        pyramid.levels.push(FeatureMap::new(p3, 8)?);
        pyramid.levels.push(FeatureMap::new(p4, 16)?);
        pyramid.levels.push(FeatureMap::new(p5, 32)?);
        pyramid.validate_ascending()?;
        Ok(pyramid)
    }

    fn forward_pooled(&self, x: &Tensor) -> AvResult<Tensor> {
        let (_p3, _p4, p5) = self.forward_all(x, self.train.get());
        let pooled = p5.adaptive_avg_pool2d([1, 1]).reshape([-1, self.p5_ch]);
        Ok(pooled)
    }

    fn spec(&self) -> BackboneSpec {
        BackboneSpec {
            levels: vec![
                LevelSpec { stride: 8, channels: self.p3_ch as usize },
                LevelSpec { stride: 16, channels: self.p4_ch as usize },
                LevelSpec { stride: 32, channels: self.p5_ch as usize },
            ],
        }
    }
}

#[cfg(all(test, feature = "torch"))]
mod tests {
    use super::*;
    use tch::{Device, Kind};

    /// nano 尺寸（yolov8n 权重兼容配置）装配。
    fn nano(vs: &nn::VarStore) -> CspElanBackbone {
        CspElanBackbone::new(
            &(vs.root() / "backbone"),
            &BackboneCfg { width: 0.25, depth: 0.33, ..Default::default() },
        )
        .expect("csp-elan nano 装配应成功")
    }

    /// 通道缩放：nano（width 0.25）= 16/32/64/128/256；非法配置被拒。
    #[test]
    fn channel_scaling_matches_ultralytics() {
        let vs = nn::VarStore::new(Device::Cpu);
        let b = nano(&vs);
        // nano P3/P4/P5 = 64/128/256（scale(256/512/1024)）
        assert_eq!(b.stride_channels(8).unwrap(), 64);
        assert_eq!(b.stride_channels(16).unwrap(), 128);
        assert_eq!(b.stride_channels(32).unwrap(), 256);
        assert!(b.stride_channels(4).is_err(), "P2（stride 4）不在金字塔");
        let bad = BackboneCfg { width: 0.0, depth: 0.33, ..Default::default() };
        assert!(CspElanBackbone::new(&(vs.root() / "backbone"), &bad).is_err());
    }

    /// 金字塔契约：stride 8/16/32、通道按 width 缩放；pooled [N,P5]；spec 同款。
    #[test]
    fn feature_pyramid_contract() {
        let vs = nn::VarStore::new(Device::Cpu);
        let b = nano(&vs);
        let x = Tensor::randn([2, 3, 64, 64], (Kind::Float, Device::Cpu));
        let pyramid = b.forward_features(&x).unwrap();
        let strides: Vec<u32> = pyramid.levels.iter().map(|l| l.stride).collect();
        assert_eq!(strides, vec![8, 16, 32]);
        let shapes: Vec<Vec<i64>> = pyramid.levels.iter().map(|l| l.tensor.size()).collect();
        assert_eq!(shapes[0], vec![2, 64, 8, 8]);
        assert_eq!(shapes[1], vec![2, 128, 4, 4]);
        assert_eq!(shapes[2], vec![2, 256, 2, 2]);
        let pooled = b.forward_pooled(&x).unwrap();
        assert_eq!(pooled.size(), vec![2, 256]);
        // spec 与实际一致
        let spec = b.spec();
        assert_eq!(spec.levels.len(), 3);
        assert_eq!(spec.levels[2], LevelSpec { stride: 32, channels: 256 });
    }

    /// 变量名与 ultralytics 逐字对齐（`backbone.` 前缀 + 同名层）——预训练
    /// 一条 `^model\.` → `backbone.` 映射即可全量导入的结构前提。
    #[test]
    fn variable_names_match_ultralytics() {
        let vs = nn::VarStore::new(Device::Cpu);
        let _b = nano(&vs);
        let names: Vec<String> = vs.variables().keys().cloned().collect();
        for expect in [
            "backbone.0.conv.weight",
            "backbone.0.bn.running_mean",
            "backbone.1.bn.bias",
            "backbone.2.cv1.conv.weight",
            "backbone.2.m.0.cv2.bn.weight",
            "backbone.4.m.1.cv1.conv.weight",
            "backbone.7.bn.running_var",
            "backbone.9.cv2.bn.bias",
        ] {
            assert!(names.iter().any(|n| n == expect), "缺变量 {expect}");
        }
        assert!(
            !names.iter().any(|n| n.contains("num_batches_tracked")),
            "tch 无 num_batches_tracked 变量（导入时落 unexpected）"
        );
    }

    /// C2f 重复数随 depth 缩放：0.33 → max(round(n×0.33),1)。
    #[test]
    fn depth_scaling_matches_ultralytics() {
        assert_eq!(scale_repeats(3, 0.33), 1);
        assert_eq!(scale_repeats(6, 0.33), 2);
        assert_eq!(scale_repeats(6, 1.0), 6);
        assert_eq!(scale_repeats(3, 0.1), 1, "round 后下限 1");
    }
}
