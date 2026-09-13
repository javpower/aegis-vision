//! # av-core
//!
//! AegisVision 公共抽象层（PLAN §3）：
//! - [`error`]：全框架统一错误类型
//! - [`conventions`]：数据约定（归一化常数、角度域）
//! - [`geometry`]：纯几何工具（AABB / 旋转框 IoU / letterbox 映射）
//! - [`types`]：serde 化数据结构与推理产物
//! - [`config`]：强类型配置 schema（`deny_unknown_fields`）
//! - [`registry`]：插件注册表（v1 静态注册，[`av_plugin!`] 宏）
//! - [`traits`]：六大核心 trait（`torch` feature；无 libtorch 环境可关闭）
//!
//! 无 `torch` feature 时本 crate 为纯 Rust，可在无 libtorch 的环境编译，
//! 用于 CI 快速回归与 docs.rs 构建（PLAN §2.3 特性门控）。

pub mod config;
pub mod conventions;
pub mod error;
pub mod geometry;
pub mod registry;
#[cfg(feature = "torch")]
pub mod traits;
pub mod types;

pub use error::{AvError, AvResult};

/// 注册插件（v1 静态注册，PLAN §6.0）。
///
/// 在插件 crate 的 `register_all` 装配函数中调用；重名注册会 panic
/// （注册表不一致属于编程错误，应尽早暴露而非静默忽略）。
///
/// ```ignore
/// av_plugin!(Backbone, "csp-elan");
/// ```
#[macro_export]
macro_rules! av_plugin {
    ($category:ident, $name:literal) => {
        $crate::registry::register($crate::registry::Category::$category, $name)
            .expect("av_plugin! 注册失败：重名或非法类别")
    };
}
