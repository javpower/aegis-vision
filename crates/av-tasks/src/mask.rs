//! 实例分割 MaskBranch（PLAN §4.3，实时档：YOLACT 式原型掩码 + 每实例系数）。
//!
//! **设计取舍（简化说明，诚实边界）**：
//! - 选型：实时档（原型 + 线性组合）而非 `direct` 精度档（RoIAlign 逐实例头）。
//!   理由：免 RoIAlign 与二阶段实例分配器，工程量小、CPU 可训、吞吐优先；
//!   `head = "direct"` 在 build_model 处显式报未支持（PLAN 允许配置双档，M4 补齐）。
//! - 原型分支：stride 16 特征（320 输入下 20×20）→ 3×3 conv → 双线性 ×4 上采样到
//!   img/4（80×80@320）→ 3×3 conv → 1×1 conv → K 个原型（sigmoid 后作「基掩码」）。
//!   YOLACT 原版在 P3（stride 4）上做原型；本骨干无 FPN，从 stride 16 上采样等价
//!   地给每个原型全图视野，且 img/4 的低分辨率监督本就容不下更细的结构。
//! - 系数分支：检测式解耦卷积头（同 LevelHead 拓扑），每 cell 输出
//!   [num_classes 类别分数 ⊕ K 个系数]；实例掩码 = sigmoid(Σ_k coef_k · proto_k)。
//!   **系数没有显式回归目标**——梯度经掩码损失端到端回传（YOLACT 同款训练方式；
//!   其系数最小二乘求解只服务于 fast-NMS 加速技巧，本实现不做该技巧故不需要）。
//! - 分配：实例掩码质心所在 cell（v0.1 center 单点分配同款），不引入 TAL——理由与
//!   OBB 官的外接框简化一致：分配只决定「哪个 cell 学哪个实例」，掩码精度由
//!   BCE+Dice 保证；拥挤场景同格冲突时后写覆盖先写（记录在案，coco8 规模无碰撞）。

use tch::nn;
use tch::nn::Module;
use tch::{Kind, Tensor};

/// 原型掩码数（YOLACT 默认 K=32）。
pub const NUM_PROTOS: i64 = 32;

/// Dice 损失（pred 为 sigmoid 概率图、gt 为 0/1 目标，同形任意维）：逐图标量。
/// dice = 2|p∩g| / (|p|+|g|)；写成 (denom − 2·inter + ε)/(denom + ε)
/// （分子即 1−dice 的分母同倍形式），ε 防空掩码除零。
pub fn dice_loss(pred: &Tensor, gt: &Tensor) -> Tensor {
    let eps = 1e-5f64;
    let inter = (pred * gt).sum(Kind::Float);
    let denom = pred.sum(Kind::Float) + gt.sum(Kind::Float);
    (&denom - &inter * 2.0 + eps) / (&denom + eps)
}

/// 两个等长 flat 0/1 掩码的 IoU（纯标量，供评测/掩码 NMS；不等长返回 0）。
pub fn mask_iou(a: &[u8], b: &[u8]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut inter = 0usize;
    let mut union = 0usize;
    for (&x, &y) in a.iter().zip(b) {
        inter += (x != 0 && y != 0) as usize;
        union += (x != 0 || y != 0) as usize;
    }
    if union == 0 {
        return 0.0;
    }
    inter as f32 / union as f32
}

pub struct MaskBranch {
    proto1: nn::Conv2D,
    proto2: nn::Conv2D,
    proto_out: nn::Conv2D,
    coef1: nn::Conv2D,
    coef2: nn::Conv2D,
    coef_out: nn::Conv2D,
    pub num_classes: i64,
    pub num_protos: i64,
}

impl MaskBranch {
    pub fn new(p: &nn::Path, in_c: i64, num_classes: i64, num_protos: i64) -> Self {
        let cc = nn::ConvConfig {
            padding: 1,
            ..Default::default()
        };
        Self {
            proto1: nn::conv2d(p / "proto1", in_c, 64, 3, cc),
            proto2: nn::conv2d(p / "proto2", 64, 64, 3, cc),
            proto_out: nn::conv2d(p / "proto_out", 64, num_protos, 1, Default::default()),
            coef1: nn::conv2d(p / "coef1", in_c, 64, 3, cc),
            coef2: nn::conv2d(p / "coef2", 64, 64, 3, cc),
            coef_out: nn::conv2d(
                p / "coef_out",
                64,
                num_classes + num_protos,
                1,
                Default::default(),
            ),
            num_classes,
            num_protos,
        }
    }

    /// 入参 f16：stride 16 特征 [N, in_c, H/16, W/16]；mask_hw = 掩码画布 (H/4, W/4)。
    /// 返回 (原型 [N,K,H/4,W/4]（未过 sigmoid），系数+类别 [N, C+K, H/16, W/16]
    /// ——通道布局：前 C 通道为类别 logits，后 K 通道为原型系数）。
    pub fn forward(&self, f16: &Tensor, mask_hw: (i64, i64)) -> (Tensor, Tensor) {
        // 原型：低分辨率特征 → 上采样到掩码分辨率（可导，梯度经插值回传）
        let hidden = self.proto1.forward(f16).relu();
        let up =
            Tensor::upsample_bilinear2d(&hidden, &[mask_hw.0, mask_hw.1], false, None, None);
        let proto = self.proto_out.forward(&self.proto2.forward(&up).relu());
        // 系数 + 类别：检测式解耦卷积头
        let coef_hidden = self.coef2.forward(&self.coef1.forward(f16).relu()).relu();
        let coef_cls = self.coef_out.forward(&coef_hidden);
        (proto, coef_cls)
    }
}

#[cfg(all(test, feature = "torch"))]
mod tests {
    use super::*;
    use tch::Device;

    #[test]
    fn dice_loss_identical_masks_near_zero() {
        // 预测 logit 大梯度（sigmoid≈1）处 gt=1、大负处 gt=0 → dice 损失 ≈ 0
        let gt = Tensor::from_slice(&[1.0f32, 1.0, 0.0, 0.0]).reshape([2i64, 2]);
        let pred = Tensor::from_slice(&[20.0f32, 20.0, -20.0, -20.0])
            .reshape([2i64, 2])
            .sigmoid();
        let l = dice_loss(&pred, &gt);
        assert!(l.double_value(&[]) < 1e-4, "got {}", l.double_value(&[]));
    }

    #[test]
    fn dice_loss_hand_computed_constant_prediction() {
        // p ≡ 0.5（logit 0），gt 单像素 1 @2×2：p_sum=2, g_sum=1, inter=0.5
        // loss = (2+1−1+ε)/(2+1+ε) ≈ 0.666663
        let gt = Tensor::from_slice(&[1.0f32, 0.0, 0.0, 0.0]).reshape([2i64, 2]);
        let pred = Tensor::zeros([2i64, 2], (Kind::Float, Device::Cpu)).sigmoid();
        let l = dice_loss(&pred, &gt);
        assert!(
            (l.double_value(&[]) - 2.0 / 3.0).abs() < 1e-4,
            "got {}",
            l.double_value(&[])
        );
    }

    #[test]
    fn mask_iou_basic_cases() {
        let a = vec![1u8, 1, 0, 0];
        let b = vec![1u8, 0, 1, 0];
        assert!((mask_iou(&a, &b) - 1.0 / 3.0).abs() < 1e-6);
        assert!((mask_iou(&a, &a) - 1.0).abs() < 1e-6);
        assert_eq!(mask_iou(&a, &[0u8; 4]), 0.0);
        assert_eq!(mask_iou(&a, &[1u8]), 0.0, "长度不等应返回 0");
        assert_eq!(mask_iou(&[0u8; 4], &[0u8; 4]), 0.0, "双空掩码应返回 0");
    }
}
