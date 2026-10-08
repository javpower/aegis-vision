//! 关键点头 KeypointHead（PLAN §4.4）：stride 8 单尺度解耦卷积头，
//! **直接回归档**（center 回归式，decode = "direct"）。
//!
//! **设计取舍（诚实边界，为何不是热图 argmax / SimDR）**：
//! - PLAN §4.4 规划了「热图回归（精度档）/ SimDR 分类解码（实时档）」两档。
//!   热图 argmax 方案在自底向上解码时每个通道整图取一个峰，多人图会混合
//!   不同实例的关键点（无关联分组嵌入，拼不成每实例骨架），且 argmax 坐标
//!   不可微（OKS 无法作为训练损失，只能退化成 MSE）；SimDR 需要按框裁剪的
//!   top-down 前置（本骨干无检测器串联）。
//! - 因此选 **直接回归**（任务规范允许的简化项）：每 cell 预测
//!   `1 类别 + 4 框 (tx,ty,tw,th) + K×(dx,dy) 偏移 + K 可见性 logit`，
//!   实例 = 框中心所在 cell（v0.1 检测 center 分配同款，见 models.rs
//!   `loss_center_l1` 的先例）。每实例天然一个预测，PCK 评测按实例诚实计数；
//!   偏移坐标线性可微，OKS（[`crate::oks`]）可直接作为训练损失。
//! - 头输出通道布局（[`kp_out_channels`] = 5 + 3K）：
//!   `ch0` 类别 logit；`ch1..5` 框 (tx,ty,tw,th)（解码 cx,cy=(cell+0.5+tanh t)·s，
//!   w,h=exp t·s，与检测 v0.1 同式）；`ch5..5+2K` 关键点偏移 (dx,dy)
//!   （cell 单位、线性不限幅——关键点可离中心半个框高，tanh 会截断远点）；
//!   `ch5+2K..5+3K` 可见性 logit（BCE，目标 v>0）。
//! - 尺度取 stride 8（320 输入 → 40×40 网格）：比 stride 16 更靠近关键点、
//!   偏移量程小一半，小模型回归更稳；stride 4 网格过密、正负失衡更重。

use tch::nn;
use tch::nn::Module;
use tch::Tensor;

/// 关键点头工作 stride（网格 = img/8）。
pub const KP_HEAD_STRIDE: u32 = 8;

/// 类别通道下标。
pub const KP_CLS_CH: i64 = 0;
/// 框分支起始通道（tx,ty,tw,th 共 4 通道）。
pub const KP_BOX_CH: i64 = 1;
/// 关键点偏移分支起始通道（2K 通道：k 点 dx = 2k，dy = 2k+1）。
pub const KP_OFF_CH: i64 = 5;

/// 头输出通道数：1 cls + 4 box + 2K offset + K vis = 5 + 3K。
pub fn kp_out_channels(num_keypoints: i64) -> i64 {
    5 + 3 * num_keypoints
}

pub struct KeypointHead {
    c1: nn::Conv2D,
    c2: nn::Conv2D,
    out: nn::Conv2D,
    pub num_keypoints: i64,
}

impl KeypointHead {
    pub fn new(p: &nn::Path, in_c: i64, num_keypoints: i64) -> Self {
        let cc = nn::ConvConfig {
            padding: 1,
            ..Default::default()
        };
        Self {
            c1: nn::conv2d(p / "c1", in_c, 64, 3, cc),
            c2: nn::conv2d(p / "c2", 64, 64, 3, cc),
            out: nn::conv2d(
                p / "out",
                64,
                kp_out_channels(num_keypoints),
                1,
                Default::default(),
            ),
            num_keypoints,
        }
    }

    /// 入参 stride 8 特征 [N, in_c, H/8, W/8] → [N, 5+3K, H/8, W/8] 原始输出
    /// （通道布局见模块文档；解码/损失在 [`crate::models::KeypointModel`]）。
    pub fn forward(&self, f8: &Tensor) -> Tensor {
        self.out
            .forward(&self.c2.forward(&self.c1.forward(f8).relu()).relu())
    }
}

#[cfg(all(test, feature = "torch"))]
mod tests {
    use super::*;
    use tch::Device;

    #[test]
    fn head_output_channel_layout() {
        let vs = tch::nn::VarStore::new(Device::Cpu);
        let head = KeypointHead::new(&vs.root(), 64, 3);
        assert_eq!(kp_out_channels(3), 14);
        let x = Tensor::randn([2, 64, 8, 8], (tch::Kind::Float, Device::Cpu));
        let out = head.forward(&x);
        assert_eq!(out.size(), vec![2, kp_out_channels(3), 8, 8]);
        assert_eq!(head.num_keypoints, 3);
    }
}
