//! v0.2 任务头：分类（GAP + Linear）与检测（多尺度解耦卷积头 + DFL）。
//!
//! v0.2 检测 box 分支按 YOLOv8 做 DFL 参数化：每边离散 [`REG_MAX`] = 16 bins，
//! 头部卷积输出 4*REG_MAX 通道；由 `models::dfl_project` 做 softmax + 积分期望
//! 还原为 [N,4,H,W]（对外 RawPred 结构不变，engine 只经 model.loss / model.predict
//! 两个入口调用）。解码约定不变：
//! cx,cy = (cell + 0.5 + tanh(t)) * s，w,h = exp(t) * s。
//!
//! 层级由配置驱动（detect.head_levels，stride 升序）：默认 [8,16] 两档；
//! 工业小缺陷场景配置 [4,8,16] 启用 P2（stride 4）高分辨率层。每层独立
//! [`LevelHead`]，变量名 `s{stride}`（默认两档与历史 checkpoint 命名兼容）。
//!
//! OBB（PLAN §4.2 双头并存，detect.obb_mode = true）：box 分支保持 DFL 4 维不变，
//! 额外增加独立角度分支——1×1 conv 从 reg 分支的共享特征（box2 输出）回归单通道
//! tθ（YOLOv8-OBB 同款拓扑：角度与 (tx,ty,tw,th) 同源）。角度解码在 models.rs：
//! θ = normalize(tanh(tθ)·π/2)，域取配置 AngleDomain。
//! TAL 标签分配见 `assigner.rs`；CIoU + DFL 损失见 `models.rs`；KFIoU 见 `kfiou.rs`。

use tch::nn;
use tch::nn::Module;
use tch::Tensor;

/// DFL 每边离散 bin 数（YOLOv8 reg_max=16）。
pub const REG_MAX: i64 = 16;

pub struct ClassifyHead {
    fc: nn::Linear,
    pub num_classes: i64,
}

impl ClassifyHead {
    pub fn new(p: &nn::Path, in_c: i64, num_classes: i64) -> Self {
        Self {
            fc: nn::linear(p / "fc", in_c, num_classes, Default::default()),
            num_classes,
        }
    }

    pub fn logits(&self, pooled: &Tensor) -> Tensor {
        self.fc.forward(pooled)
    }
}

struct LevelHead {
    cls1: nn::Conv2D,
    cls2: nn::Conv2D,
    cls_out: nn::Conv2D,
    box1: nn::Conv2D,
    box2: nn::Conv2D,
    box_out: nn::Conv2D,
    /// OBB 角度分支（1×1 conv → tθ），与 reg 分支共享 box2 特征；非 OBB 为 None。
    theta_out: Option<nn::Conv2D>,
}

impl LevelHead {
    fn new(p: &nn::Path, in_c: i64, mid: i64, num_classes: i64, obb: bool) -> Self {
        let cc = nn::ConvConfig {
            padding: 1,
            ..Default::default()
        };
        Self {
            cls1: nn::conv2d(p / "cls1", in_c, mid, 3, cc),
            cls2: nn::conv2d(p / "cls2", mid, mid, 3, cc),
            cls_out: nn::conv2d(p / "cls3", mid, num_classes, 1, Default::default()),
            box1: nn::conv2d(p / "box1", in_c, mid, 3, cc),
            box2: nn::conv2d(p / "box2", mid, mid, 3, cc),
            box_out: nn::conv2d(p / "box3", mid, 4 * REG_MAX, 1, Default::default()),
            theta_out: obb.then(|| nn::conv2d(p / "theta", mid, 1, 1, Default::default())),
        }
    }

    /// 返回 (cls logits [N,C,H,W], box DFL 分布 [N, 4*REG_MAX, H,W], tθ raw [N,1,H,W])。
    /// 通道布局：边 k 的 bin b 在通道 k*REG_MAX + b；非 OBB 模式 tθ 为 None。
    fn forward(&self, feat: &Tensor) -> (Tensor, Tensor, Option<Tensor>) {
        let cls = self
            .cls_out
            .forward(&self.cls2.forward(&self.cls1.forward(feat).relu()).relu());
        let box_hidden = self.box2.forward(&self.box1.forward(feat).relu()).relu();
        let box_dist = self.box_out.forward(&box_hidden);
        let theta = self.theta_out.as_ref().map(|t| t.forward(&box_hidden));
        (cls, box_dist, theta)
    }
}

pub struct DetectHead {
    /// 每层独立解耦头，与 [`DetectHead::strides`] 一一对应（升序）。
    levels: Vec<LevelHead>,
    /// 各层 stride（升序；默认 [8,16]，小缺陷场景 [4,8,16] 启用 P2）。
    pub strides: Vec<u32>,
    pub num_classes: i64,
}

impl DetectHead {
    /// 普通检测头（无角度分支）。`strides`/`channels` 一一对应且升序
    /// （如 strides = [8,16] 或小缺陷 [4,8,16]，channels 取
    /// `backbone.stride_channels(s)`）。
    pub fn new(p: &nn::Path, strides: &[u32], channels: &[i64], num_classes: i64) -> Self {
        Self::with_mode(p, strides, channels, num_classes, false)
    }

    /// OBB 检测头：每层额外角度分支（PLAN §4.2 obb_mode）。
    pub fn new_obb(p: &nn::Path, strides: &[u32], channels: &[i64], num_classes: i64) -> Self {
        Self::with_mode(p, strides, channels, num_classes, true)
    }

    /// 按 obb 开关装配（build_model 直通入口）。
    pub fn with_mode(
        p: &nn::Path,
        strides: &[u32],
        channels: &[i64],
        num_classes: i64,
        obb: bool,
    ) -> Self {
        assert_eq!(
            strides.len(),
            channels.len(),
            "head_levels 与 channels 数量必须一致"
        );
        let levels = strides
            .iter()
            .zip(channels.iter())
            .map(|(&s, &c)| LevelHead::new(&(p / format!("s{s}")), c, 64, num_classes, obb))
            .collect();
        Self {
            levels,
            strides: strides.to_vec(),
            num_classes,
        }
    }

    /// 入参 `feats` 与 `strides` 一一对应（如 (stride 4, 8, 16) 特征）。
    /// 返回逐层 [(cls logits, box DFL 分布, tθ raw)]（非 OBB 模式第三项为 None）。
    pub fn forward(&self, feats: &[&Tensor]) -> Vec<(Tensor, Tensor, Option<Tensor>)> {
        assert_eq!(
            feats.len(),
            self.levels.len(),
            "特征层数与检测头层数必须一致"
        );
        self.levels
            .iter()
            .zip(feats.iter())
            .map(|(head, feat)| head.forward(feat))
            .collect()
    }
}
