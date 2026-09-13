//! YOLACT 式分割头（简化版，spike）：原型掩码分支 + 每网格系数/类别分支。
//!
//! tch 参考实现：`crates/av-tasks/src/mask.rs` 的 `MaskBranch`（拓扑与通道
//! 逐层对齐）：
//!
//! - **原型分支**：stride 16 特征 → 3×3 conv + ReLU → ×4 上采样到 img/4
//!   （掩码画布）→ 3×3 conv + ReLU → 1×1 conv → K 个原型 logits（训练时在
//!   loss 内过 sigmoid 作「基掩码」，与 predict 组合方式一致）。
//!   **与 tch 参考实现的差异（诚实边界）**：tch 版用双线性上采样；
//!   burn-ndarray 0.21 只实现了 nearest 插值的反传（`interpolate_backward`
//!   对 Bilinear/Bicubic/Lanczos3 显式 panic），本 spike 训练链路统一用
//!   nearest（梯度为 scatter 语义，链路仍可导可训）；换 wgpu 后端时改回
//!   `InterpolateMode::Linear` 即可（cubecl 有双线性反传）。
//! - **系数分支**：检测式解耦卷积头（3×3×2 + 1×1），每 cell 输出
//!   `[C 类别 logits ⊕ K 系数]`；实例掩码 = sigmoid(Σ_k coef_k · proto_k)。
//!   系数无显式回归目标，梯度只经掩码损失端到端回传（YOLACT 同款）。

// burn 的 #[derive(Module)] 展开引用 `burn::` 路径，需要该别名（burn 自身
// 源码的同款约定）。
use burn_core as burn;
use burn_core::module::Module;
use burn_core::tensor::backend::Backend;
use burn_core::tensor::{Tensor, activation::relu};
use burn_nn::modules::conv::{Conv2d, Conv2dConfig};
use burn_nn::modules::interpolate::{Interpolate2d, Interpolate2dConfig, InterpolateMode};
use burn_nn::PaddingConfig2d;

/// 原型掩码分支的隐藏通道（tch 版同款 64）。
const HEAD_HIDDEN: usize = 64;

/// 3×3 conv（pad 1，带 bias——tch 版 nn::conv2d 默认带 bias）。
fn conv3x3<B: Backend>(in_c: usize, out_c: usize, device: &B::Device) -> Conv2d<B> {
    Conv2dConfig::new([in_c, out_c], [3, 3])
        .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
        .init(device)
}

/// 1×1 conv（无 padding，带 bias）。
fn conv1x1<B: Backend>(in_c: usize, out_c: usize, device: &B::Device) -> Conv2d<B> {
    Conv2dConfig::new([in_c, out_c], [1, 1]).init(device)
}

/// YOLACT 式分割头。
#[derive(Module, Debug)]
pub struct MaskHead<B: Backend> {
    proto1: Conv2d<B>,
    proto2: Conv2d<B>,
    proto_out: Conv2d<B>,
    coef1: Conv2d<B>,
    coef2: Conv2d<B>,
    coef_out: Conv2d<B>,
    upsample: Interpolate2d,
    /// 类别数（C）。
    pub num_classes: usize,
    /// 原型数（K，YOLACT 默认 32）。
    pub num_protos: usize,
}

impl<B: Backend> MaskHead<B> {
    /// `in_c`：stride 16 特征通道数。
    pub fn new(in_c: usize, num_classes: usize, num_protos: usize, device: &B::Device) -> Self {
        Self {
            proto1: conv3x3(in_c, HEAD_HIDDEN, device),
            proto2: conv3x3(HEAD_HIDDEN, HEAD_HIDDEN, device),
            proto_out: conv1x1(HEAD_HIDDEN, num_protos, device),
            coef1: conv3x3(in_c, HEAD_HIDDEN, device),
            coef2: conv3x3(HEAD_HIDDEN, HEAD_HIDDEN, device),
            coef_out: conv1x1(HEAD_HIDDEN, num_classes + num_protos, device),
            // stride 16 → img/4 恰为 ×4。nearest 而非 tch 版的双线性：ndarray
            // 后端只有 nearest 反传（见模块注释的诚实边界说明）。
            upsample: Interpolate2dConfig::new()
                .with_scale_factor(Some([4.0, 4.0]))
                .with_mode(InterpolateMode::Nearest)
                .init(),
            num_classes,
            num_protos,
        }
    }

    /// 入参 f16：stride 16 特征 [N, in_c, H/16, W/16]。
    /// 返回 (原型 logits [N, K, H/4, W/4]，系数+类别 [N, C+K, H/16, W/16]
    /// ——通道布局：前 C 通道类别 logits，后 K 通道原型系数)。
    pub fn forward(&self, f16: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        // 原型：低分辨率特征 → 上采样到掩码分辨率（可导，梯度经插值回传）
        let hidden = relu(self.proto1.forward(f16.clone()));
        let up = self.upsample.forward(hidden);
        let proto = self.proto_out.forward(relu(self.proto2.forward(up)));
        // 系数 + 类别：检测式解耦卷积头
        let coef_hidden = relu(self.coef2.forward(relu(self.coef1.forward(f16))));
        let coef_cls = self.coef_out.forward(coef_hidden);
        (proto, coef_cls)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NdArrayB;

    /// 头部 shape 契约（对照 tch MaskBranch）：输入 stride16 [1,8,4,4] →
    /// 原型 [1,K,16,16]、系数+类别 [1,C+K,4,4]（掩码画布 = 16 = 4×4 上采样）。
    #[test]
    fn head_shapes_match_yolact_contract() {
        let device = burn_ndarray::NdArrayDevice::Cpu;
        let (c, k) = (5usize, 8usize);
        let head = MaskHead::<NdArrayB>::new(8, c, k, &device);
        let f16 = Tensor::<NdArrayB, 4>::ones([1, 8, 4, 4], &device);
        let (proto, coef_cls) = head.forward(f16);
        assert_eq!(proto.dims(), [1, k, 16, 16]);
        assert_eq!(coef_cls.dims(), [1, c + k, 4, 4]);
    }
}
