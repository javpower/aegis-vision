//! # av-pretrain
//!
//! AegisVision 预训练权重层（预训练权重方案第一层）：
//! - [`av_weight`]：AV 原生权重格式（avpretrain 目录）：每变量一文件 +
//!   manifest.json（元信息 + 每张量 blake3 `hash`）+ 全量哈希校验；
//! - [`weight_adapter`]：外部 PyTorch 权重导入适配——读取 `.safetensors`
//!   （tch 0.17 原生 API），按 [`weight_adapter::LayerMap`]（正则/前缀 →
//!   改名、可选转置）映射到目标模型变量并比对形状，输出四类完整清单。
//!
//! feature 门控与 av-core 一致：`torch`（默认）启用张量层（`Tensor::save/load`、
//! safetensors 读取）；`--no-default-features` 时 manifest + 哈希校验纯逻辑可用。

pub mod av_weight;
#[cfg(feature = "torch")]
pub mod weight_adapter;

pub use av_weight::{WeightManifest, WeightMeta};
