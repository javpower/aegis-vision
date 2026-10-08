//! # av-burn
//!
//! AegisVision 的 burn 框架技术验证 spike：用 burn 0.21（默认 ndarray 后端做
//! 测试、可选 wgpu 后端做编译验证）复刻 tch 版的两条核心链路——
//!
//! 1. [`backbone`]：CSP-ELAN 骨干（ultralytics YOLOv8 backbone 层 0-9 同构），
//!    移植自 `crates/av-tasks/src/backbone_cspelan.rs`；
//! 2. [`seg`]：YOLACT 式实例分割链路「原型掩码 × 每网格系数」，语义对齐
//!    `crates/av-tasks/src/models.rs` 的 `SegModel`（掩码画布 img/4、系数网格
//!    img/16、质心单点分配、BCE+Dice 监督）。
//!
//! 数据解码（[`data`]）按 `crates/av-runtime/src/dataset.rs` 的
//! letterbox / load_cocoseg_dir_raw / rasterize_polygon 语义自实现（本 crate
//! 禁止依赖 av-runtime/av-tasks——它们绑 tch），letterbox 几何直接复用
//! `av_core::geometry::letterbox`。
//!
//! # 与 tch 版的诚实差异（spike 边界，逐条记录）
//!
//! - **无权重命名对齐**：burn 的 Param 命名由模块字段决定，与 ultralytics
//!   `model.N.*` / VarStore `backbone.N.*` 不一一对应；预训练导入适配器
//!   （av-pretrain）不在本 spike 范围。
//! - **无 predict/NMS**：只做训练/损失链路 + 单实例掩码组合辅助
//!   （[`seg`] 内 `combine_proto_coef`），推理解耦头、掩码 NMS 留待正式移植。
//! - **无 6 大核心 trait 接入**：不实现 av_core::traits（该模块绑 tch）；
//!   金字塔契约以单元测试对照 tch 版断言（通道缩放/stride/shape）。
//! - **BN 训练语义**：tch 版手控 `Cell<bool>` 训练标志；burn 的
//!   `BatchNorm::forward` 依后端 `ad_enabled` 自动切换 batch-stat（训练，且
//!   按 momentum 更新 running stats）/ running-stat（推理）语义，等价。
//!
//! # 后端选择
//!
//! - 测试/冒烟：[`NdArrayB`]（纯 Rust CPU，`burn-ndarray`），
//!   训练后端 [`TrainB`] = `Autodiff<NdArrayB>`（`burn-autodiff`）。
//! - wgpu：`--features wgpu` 打开，仅编译验证（[`wgpu_check`]），**不运行**
//!   GPU 训练（显卡被实验占用）。

pub mod backbone;
pub mod data;
pub mod head;
pub mod seg;
pub mod train;

pub use seg::{SegNet, SegNetCfg};

/// ndarray f32 后端（单测 / coco8-seg 过拟合冒烟测试）。
pub type NdArrayB = burn_ndarray::NdArray<f32>;
/// 训练后端 = ndarray + 自动微分（burn-autodiff 装饰器）。
pub type TrainB = burn_autodiff::Autodiff<NdArrayB>;

/// wgpu 后端编译验证模块。
///
/// 只在 `--features wgpu` 下编译：装配 `SegNet<Autodiff<Wgpu>>` + AdamW +
/// 前向 shape 自检，证明模型/优化器/损失对 wgpu 后端类型检查通过。
/// **没有任何测试会运行它**——禁止 GPU 训练，wgpu 仅编译期验证。
#[cfg(feature = "wgpu")]
pub mod wgpu_check {
    use crate::seg::{SegNet, SegNetCfg};
    use burn_autodiff::Autodiff;
    use burn_core::tensor::Tensor;
    use burn_wgpu::{Wgpu, WgpuDevice};

    /// wgpu + 自动微分训练后端。
    pub type WgpuTrainB = Autodiff<Wgpu>;

    /// nano 配置装配（编译期验证模块/BN/conv 在 Wgpu 后端可初始化）。
    pub fn build_model(device: &WgpuDevice) -> av_core::AvResult<SegNet<WgpuTrainB>> {
        SegNet::new(
            &SegNetCfg {
                width: 0.25,
                depth: 0.33,
                num_classes: 80,
                num_protos: 32,
                loss_w_bce: 1.0,
                loss_w_dice: 1.0,
            },
            device,
        )
    }

    /// 装配 AdamW（含梯度范数裁剪），验证优化器对 Wgpu 后端泛型检查。
    pub fn build_optimizer() -> impl burn_optim::Optimizer<SegNet<WgpuTrainB>, WgpuTrainB> {
        crate::train::make_optimizer(&crate::train::TrainCfg {
            lr: 0.01,
            lr_min: 0.001,
            weight_decay: 0.0,
            max_grad_norm: 10.0,
            total_steps: 1,
        })
    }

    /// 前向 shape 自检（无测试调用；供人工 GPU 冒烟，当前环境禁止执行）。
    pub fn forward_shapes(device: &WgpuDevice) -> ([usize; 4], [usize; 4]) {
        let model = build_model(device).expect("nano 装配应成功");
        let x = Tensor::<WgpuTrainB, 4>::ones([1, 3, 64, 64], device);
        let (proto, coefcls) = model.forward_seg(x);
        (proto.dims(), coefcls.dims())
    }
}
