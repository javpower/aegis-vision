//! # av-burn
//!
//! AegisVision 的 **burn 框架后端**（crate 名 `aegisvision-burn`，lib 名
//! `av_burn`，与本工作区其余 crate 的命名约定一致）：用 burn 0.21（默认
//! ndarray 后端、可选 wgpu 后端）复刻 tch 版的核心链路，**纯 Rust、零
//! libtorch/C++ 工具链依赖**（av-core 关闭 default features）。
//!
//! # 四条链路（训练 → 保存 → 加载 → 推理，零参数闭环）
//!
//! 1. [`backbone`]：CSP-ELAN 骨干（ultralytics YOLOv8 backbone 层 0-9 同构），
//!    移植自 `crates/av-tasks/src/backbone_cspelan.rs`；
//! 2. [`seg`]：YOLACT 式实例分割「原型掩码 × 每网格系数」，语义对齐
//!    `crates/av-tasks/src/models.rs` 的 `SegModel`（掩码画布 img/4、系数网格
//!    img/16、质心单点分配、BCE+Dice 监督）；
//! 3. [`checkpoint`]：`model.bp` 权重 + `config.snapshot.toml` 快照
//!    （约定与 av-runtime 一致），预测侧从快照自动重建超参；
//! 4. [`infer`]：推理解码（conf 截取 → 批量原型合成 → 同类掩码 NMS →
//!    [`infer::SegInstance`]，字段与 tch 版同名同型）+ 掩码 → 原图坐标映射。
//!
//! 数据解码（[`data`]）按 `crates/av-runtime/src/dataset.rs` 的
//! letterbox / load_cocoseg_dir_raw / rasterize_polygon 语义自实现（本 crate
//! 禁止依赖 av-runtime/av-tasks——它们绑 tch）；data.yaml 解析直接复用
//! `av_core::config::parse_data_yaml`，letterbox 几何复用
//! `av_core::geometry::letterbox`。
//!
//! # CLI（与主 CLI `av` 体验对齐）
//!
//! ```text
//! cargo install aegisvision-burn           # CPU 版（ndarray）
//! cargo install aegisvision-burn --features wgpu   # GPU 版（wgpu）
//! avb train   --data <目录|data.yaml> --epochs 100
//! avb predict --weights runs-avb/<name> --input img.jpg --save-viz out/
//! ```
//!
//! # 与 tch 版的诚实差异（逐条记录）
//!
//! - **无权重命名对齐**：burn 的 Param 命名由模块字段决定，与 ultralytics
//!   `model.N.*` / VarStore `backbone.N.*` 不一一对应；预训练导入适配器
//!   （av-pretrain）不在本 crate 范围（checkpoint 为 burn 原生格式）。
//! - **上采样 nearest**：burn-ndarray 0.21 无双线性插值反传（对
//!   Bilinear/Bicubic/Lanczos3 显式 panic），头内 ×4 上采样用 nearest；
//!   wgpu 后端有双线性反传，head.rs 一行可切（cubecl 支持）。
//! - **无 6 大核心 trait 接入**：不实现 av_core::traits（该模块绑 tch）；
//!   金字塔契约以单元测试对照 tch 版断言（通道缩放/stride/shape）。
//! - **BN 训练语义**：tch 版手控 `Cell<bool>` 训练标志；burn 的
//!   `BatchNorm::forward` 依后端 `ad_enabled` 自动切换 batch-stat（训练，且
//!   按 momentum 更新 running stats）/ running-stat（推理）语义，等价。
//! - **性能**：ndarray 后端矩阵乘走 matrixmultiply（无 BLAS），适合冒烟/
//!   小模型；正式训练/推理用 `--features wgpu`（GPU）或主后端 tch。
//!
//! # 后端选择
//!
//! - 测试/冒烟：[`NdArrayB`]（纯 Rust CPU，`burn-ndarray`），
//!   训练后端 [`TrainB`] = `Autodiff<NdArrayB>`（`burn-autodiff`）。
//! - wgpu：`--features wgpu` 打开（[`wgpu_check`]），GPU 训练/推理可用。

pub mod backbone;
pub mod checkpoint;
pub mod data;
pub mod head;
pub mod infer;
pub mod seg;
pub mod train;

pub use infer::SegInstance;
pub use seg::{SegNet, SegNetCfg};

/// ndarray f32 后端（单测 / CPU 推理 / 冒烟训练）。
pub type NdArrayB = burn_ndarray::NdArray<f32>;
/// 训练后端 = ndarray + 自动微分（burn-autodiff 装饰器）。
pub type TrainB = burn_autodiff::Autodiff<NdArrayB>;

/// wgpu 后端入口（`--features wgpu` 下编译）。
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
