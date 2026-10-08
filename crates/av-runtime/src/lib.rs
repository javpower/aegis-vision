//! av-runtime 库层：训练 / 推理引擎（PLAN §2.1「库与二进制分离」，CLI 只是薄封装）。

/// `.avpack` 数据集打包容器（PLAN 附录 B；纯逻辑，无 torch 依赖）。
pub mod avpack;

#[cfg(feature = "torch")]
pub mod api;
#[cfg(feature = "torch")]
pub mod cuda_link;
#[cfg(feature = "torch")]
pub mod dataset;
#[cfg(feature = "torch")]
pub mod engine;
#[cfg(feature = "torch")]
pub mod eval_map;
/// 训练指标增量落盘（面板实时化）：runs/<id>/metrics.jsonl 追加/增量读。
pub mod metrics;
/// 观测面板（PLAN §7.1，feature = "panel"）。
#[cfg(all(feature = "panel", feature = "torch"))]
pub mod panel;

/// 集成测试辅助入口（复用 engine 内部构造；不属于公开 API 承诺）。
#[cfg(feature = "torch")]
pub mod testing {
    use std::path::Path;

    use tch::Tensor;

    use av_core::config::RunConfig;
    use av_core::error::AvResult;
    use av_tasks::models::TaskModel;
    use av_tasks::rng::XorShift;

    pub fn load_model(cfg: &RunConfig, weights: &Path) -> AvResult<TaskModel> {
        crate::engine::testing_load_model(cfg, weights)
    }

    pub fn synthetic_detect(
        rng: &mut XorShift,
        n: i64,
        num_classes: u32,
        img_size: u32,
    ) -> (Tensor, Vec<[f32; 4]>, Vec<u32>) {
        crate::engine::testing_synthetic_detect(rng, n, num_classes, img_size)
    }

    /// 关键点 PCK/OKS 评测（集成测试桥接；CPU 设备固定）。
    pub fn eval_kp_samples(
        m: &av_tasks::models::KeypointModel,
        samples: &[crate::dataset::KeypointSample],
    ) -> AvResult<(f32, f32, usize, usize)> {
        crate::engine::testing_eval_kp_samples(m, samples)
    }
}
