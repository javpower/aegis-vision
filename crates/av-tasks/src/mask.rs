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

/// 掩码几何摘要：n×m IoU 矩阵 / 掩码 NMS 的 O(1) 预筛依据。
///
/// 每实例构建一次（单遍扫描求前景外接框）；配对求 IoU 时先比外接框——
/// **严格分离的框交集必空、IoU 恒为 0**，免去整画布逐字节扫描（拥挤图上
/// O(n²) 对全画布 popcount 是评测/NMS 的隐形大头）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaskSummary {
    /// 方画布宽（掩码长度非完全平方数时 None，预筛退化为全量路径）
    width: Option<usize>,
    /// 前景外接框 [x1, y1, x2, y2]（方画布且含前景时 Some）
    bbox: Option<[usize; 4]>,
    len: usize,
}

impl MaskSummary {
    pub fn of(mask: &[u8]) -> Self {
        let w = (mask.len() as f64).sqrt() as usize;
        let width = (w > 0 && w * w == mask.len()).then_some(w);
        let bbox = width.and_then(|w| {
            let mut bbox: Option<[usize; 4]> = None;
            for (i, &v) in mask.iter().enumerate() {
                if v == 0 {
                    continue;
                }
                let (x, y) = (i % w, i / w);
                bbox = Some(match bbox {
                    None => [x, y, x, y],
                    Some([x1, y1, x2, y2]) => [x1.min(x), y1.min(y), x2.max(x), y2.max(y)],
                });
            }
            bbox
        });
        Self {
            width,
            bbox,
            len: mask.len(),
        }
    }

    /// `self` 对应掩码 `a` 与 `other` 对应掩码 `b` 的 IoU。语义与
    /// [`mask_iou`] 逐位一致：预筛只短路「必为 0」的对（外接框严格分离、
    /// 同尺寸方画布下含空掩码、长度不等），其余（含非方画布）走全量路径。
    pub fn iou(&self, a: &[u8], other: &Self, b: &[u8]) -> f32 {
        if self.len != other.len {
            return 0.0; // mask_iou 对不等长恒 0
        }
        match (self.width, other.width) {
            (Some(_), Some(_)) => match (self.bbox, other.bbox) {
                (Some(ba), Some(bb)) => {
                    if ba[2] < bb[0] || bb[2] < ba[0] || ba[3] < bb[1] || bb[3] < ba[1] {
                        0.0 // 框严格分离 → 交集必空
                    } else {
                        mask_iou(a, b)
                    }
                }
                // 同尺寸且至少一方全空：交集或并集为 0 → IoU = 0
                _ => 0.0,
            },
            // 非方画布无法定位外接框：退回全量路径（含不等长已在上面拦截）
            _ => mask_iou(a, b),
        }
    }
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
        let up = Tensor::upsample_bilinear2d(&hidden, [mask_hw.0, mask_hw.1], false, None, None);
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

    /// 4×4 画布上的掩码构造助手（行主序）。
    fn mask4(cells: &[(usize, usize)]) -> Vec<u8> {
        let mut m = vec![0u8; 16];
        for &(x, y) in cells {
            m[y * 4 + x] = 1;
        }
        m
    }

    #[test]
    fn mask_summary_iou_matches_mask_iou() {
        // 重叠对：预筛路径与全量 mask_iou 逐位一致
        let a = mask4(&[(0, 0), (1, 0), (1, 1)]);
        let b = mask4(&[(1, 0), (1, 1), (2, 1)]);
        let (sa, sb) = (MaskSummary::of(&a), MaskSummary::of(&b));
        assert_eq!(sa.iou(&a, &sb, &b), mask_iou(&a, &b));
        assert_eq!(sa.iou(&a, &sa, &a), 1.0);
    }

    #[test]
    fn mask_summary_short_circuits_separated_boxes() {
        // 框严格分离 → 必为 0；相邻（touching）框不分离 → 走全量仍为 0
        let left = mask4(&[(0, 1), (0, 2)]);
        let right = mask4(&[(3, 1), (3, 2)]);
        let adjacent = mask4(&[(1, 1), (1, 2)]);
        let (sl, sr, sadj) = (
            MaskSummary::of(&left),
            MaskSummary::of(&right),
            MaskSummary::of(&adjacent),
        );
        assert_eq!(sl.iou(&left, &sr, &right), 0.0);
        assert_eq!(sl.iou(&left, &sadj, &adjacent), 0.0);
        // bbox 记录正确（左列掩码 x∈{0}, y∈{1,2}）
        assert_eq!(sl.bbox, Some([0, 1, 0, 2]));
    }

    #[test]
    fn mask_summary_empty_and_unequal_masks() {
        let a = mask4(&[(0, 0)]);
        let (sa, sempty) = (MaskSummary::of(&a), MaskSummary::of(&[0u8; 16]));
        assert_eq!(sempty.bbox, None);
        assert_eq!(sa.iou(&a, &sempty, &[0u8; 16]), 0.0, "空+非空 → 0");
        assert_eq!(sempty.iou(&[0u8; 16], &sempty, &[0u8; 16]), 0.0, "双空 → 0");
        // 长度不等 → 0；非方画布 → 回退全量路径，结果与 mask_iou 一致
        assert_eq!(sa.iou(&a, &MaskSummary::of(&[1u8]), &[1u8]), 0.0);
        let nonsquare = vec![1u8, 0, 1];
        let sn = MaskSummary::of(&nonsquare);
        assert_eq!(sn.width, None);
        assert_eq!(
            sn.iou(&nonsquare, &sn, &nonsquare),
            mask_iou(&nonsquare, &nonsquare)
        );
    }
}
