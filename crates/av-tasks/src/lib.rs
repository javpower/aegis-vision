//! # av-tasks
//!
//! 五任务内置实现（PLAN §4）：classify / detect 已在 v0.1 引擎落地；
//! obb / seg / keypoint 按 PLAN §8 里程碑（M3–M5）逐步补齐。
//! 四类任务头共用同一骨干，靠 `model.tasks` 配置组合（附录 A）。

pub mod assigner;
/// 数据增强原语（纯 Rust 像素/坐标变换，无 tch / image 依赖，torch feature 外也可用）。
pub mod augment;
pub mod rng;
/// 旋转 NMS 与角度工具（纯标量，无 tch 依赖）。
pub mod rot_nms;

#[cfg(feature = "torch")]
pub mod backbone;
/// DINOv2 ViT 骨干族（ViT-S/14 起步）：官方预训练权重导入 + pos_embed 网格插值。
#[cfg(feature = "torch")]
pub mod backbone_cspelan;
pub mod backbone_dino;
/// torchvision ResNet18 骨干族：层名逐字对齐 torchvision，ImageNet 权重可全量导入。
#[cfg(feature = "torch")]
pub mod backbone_resnet;
#[cfg(feature = "torch")]
pub mod heads;
/// 关键点头 KeypointHead（PLAN §4.4：直接回归档，cls+box+K×(dx,dy)+K vis）。
#[cfg(feature = "torch")]
pub mod keypoint;
/// KFIoU / ProbIoU 旋转框回归损失（可微张量运算）。
#[cfg(feature = "torch")]
pub mod kfiou;
/// 实例分割 MaskBranch（PLAN §4.3：原型掩码 + 每实例系数）。
#[cfg(feature = "torch")]
pub mod mask;
#[cfg(feature = "torch")]
pub mod models;
/// OKS 关键点相似度：常数表 + 标量评测版 + 可微张量损失版（PLAN §4.4）。
#[cfg(feature = "torch")]
pub mod oks;

#[cfg(feature = "torch")]
pub mod prelude {
    //! 任务实现共用的 re-export。
    pub use av_core::traits::{
        BaseBackbone, Configurable, FeatureMap, FeatureNeck, FeaturePyramid, LossAggregator,
        LossDict, PostProcessor, TaskHead, WeightedSumAggregator,
    };
    pub use av_core::types::{DetTarget, Detection, ImageMeta, ObbTarget};
}

/// 把本 crate 内置实现登记进全局注册表（runtime 启动时调用，PLAN §6.0）。
#[cfg(feature = "torch")]
pub fn register_builtin() -> av_core::AvResult<()> {
    use av_core::registry::{register, Category};
    register(Category::Backbone, crate::backbone::FAMILY_NAME)?;
    register(Category::Backbone, crate::backbone_dino::FAMILY_NAME)?;
    register(Category::Backbone, crate::backbone_resnet::FAMILY_NAME)?;
    register(Category::Head, "classify")?;
    register(Category::Head, "detect")?;
    register(Category::Head, "obb")?;
    register(Category::Head, "seg")?;
    register(Category::Head, "keypoint")?;
    register(Category::Loss, "kfiou")?;
    register(Category::Loss, "probiou")?;
    register(Category::Loss, "dice")?;
    register(Category::PostProc, "rotate_nms")?;
    Ok(())
}
