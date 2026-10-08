//! 面向嵌入方的稳定门面（crates.io 库面）：**在线训练 → 加载 → 推理 → 导出**
//! 的最短路径。CLI（`src/main.rs`）是本模块同一引擎的薄封装；嵌入方（完整
//! 应用内训练模型的场景）只依赖这里列出的入口，内部模块的签名变化不算破坏。
//!
//! # 一分钟上手（应用内训练）
//!
//! ```no_run
//! use av_core::config::RunConfig;
//! use std::path::Path;
//!
//! # fn main() -> av_core::error::AvResult<()> {
//! // 配置 = TOML 文本（等价 configs/*.toml；`av init` 可生成完整模板）。
//! // serde(default)：只写需要覆盖的字段，其余走内置默认。
//! let cfg: RunConfig = toml::from_str(r#"
//! run_id = "in-app-run"
//! [[model.tasks]]
//! kind = "detect"
//! num_classes = 3
//! img_size = 640
//! [data.sources.train]
//! dir = "dataset/train"
//! [data.sources.val]
//! dir = "dataset/val"
//! "#)?;
//!
//! // ① 阻塞训练（长任务；进度走 println + runs/<id>/metrics.jsonl）
//! let report = av_runtime::api::train(&cfg)?;
//! println!("{} = {}", report.metric, report.metric_value);
//!
//! // ② 推理：bbox 已还原到原图坐标，结构与 CLI `av infer` 完全一致
//! let out = av_runtime::api::infer(
//!     &cfg,
//!     Path::new("runs/in-app-run/best.ckpt"),
//!     Path::new("sample.jpg"),
//! )?;
//! println!("{}", out["detections"]);
//!
//! // ③ 跨生态出口：safetensors（Python 生态直接可读）
//! av_runtime::api::export_safetensors(
//!     &cfg,
//!     Path::new("runs/in-app-run/best.ckpt"),
//!     Path::new("best.safetensors"),
//! )?;
//! # Ok(())
//! # }
//! ```
//!
//! # 特性说明
//!
//! `default = ["torch"]`：训练/推理需要 libtorch（torch-sys 构建期自动下载
//! CPU 版；GPU 配置见 docs/USAGE.md §6）。纯数据场景可
//! `default-features = false` 只取 [`crate::avpack`] 与指标层。

use std::path::Path;

use av_core::config::RunConfig;
use av_core::error::AvResult;

use crate::engine::TrainReport;

/// 训练一个任务（阻塞直到全部 epoch 完成）。进度经 `println!` 与
/// `runs/<run_id>/metrics.jsonl`（[`crate::metrics`] 可增量读取）输出，
/// 供宿主应用做 UI 展示。
pub fn train(cfg: &RunConfig) -> AvResult<TrainReport> {
    crate::engine::train(cfg)
}

/// 断点续训（`runs/<id>/last.ckpt` 存在时从上次 epoch 继续；当前仅 seg 路径
/// 支持周期存盘，见 docs/USAGE.md §8 已知边界）。
pub fn train_resumed(cfg: &RunConfig) -> AvResult<TrainReport> {
    crate::engine::train_resumed(cfg)
}

/// 单图推理。返回 serde_json::Value：`task` + `detections`（含还原到原图坐标
/// 的 bbox、score、class_id）或 `predictions`（分类），结构与 CLI `av infer`
/// 完全一致。
pub fn infer(cfg: &RunConfig, weights: &Path, input: &Path) -> AvResult<serde_json::Value> {
    crate::engine::infer(cfg, weights, input)
}

/// 超大图切片推理（SAHI 式：原图分辨率滑窗，小目标召回优先）。
pub fn infer_sliced(
    cfg: &RunConfig,
    weights: &Path,
    input: &Path,
    window: u32,
    overlap: f32,
) -> AvResult<serde_json::Value> {
    crate::engine::infer_sliced(cfg, weights, input, window, overlap)
}

/// 在配置指定的验证集上评测权重（COCO 风格 mAP 等指标 JSON）。
pub fn eval(cfg: &RunConfig, weights: &Path) -> AvResult<serde_json::Value> {
    crate::engine::eval(cfg, weights)
}

/// 权重导出：`fmt = "safetensors"`（跨生态）或 `"torch"`（本框架目录格式）。
/// `cfg` 仅用于按配置重建变量表（名字/形状对齐），权重来自 `weights_dir`。
pub fn export(cfg: &RunConfig, weights_dir: &Path, out: &Path, fmt: &str) -> AvResult<()> {
    let mut vs = tch::nn::VarStore::new(tch::Device::Cpu);
    let _model = av_tasks::models::build_model(&vs.root(), cfg)?;
    crate::engine::load_checkpoint_dir(&mut vs, weights_dir)?;
    crate::engine::export_checkpoint(&vs, out, fmt)
}

/// safetensors 导出捷径（等价 [`export`] 的 `fmt = "safetensors"`）。
pub fn export_safetensors(cfg: &RunConfig, weights_dir: &Path, out: &Path) -> AvResult<()> {
    export(cfg, weights_dir, out, "safetensors")
}
