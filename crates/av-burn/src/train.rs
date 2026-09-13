//! 训练循环原语：AdamW + cosine 退火 + 梯度范数裁剪。
//!
//! 对齐任务要求的三件套（对应 av-runtime 训练引擎语义的子集）：
//! - AdamW（`burn-optim`，解耦权重衰减）；
//! - cosine 退火：`lr(t) = lr_min + (lr0 - lr_min) · ½(1 + cos(πt/T))`
//!   （无 warm restart，t∈[0,T]，端点/单调性有单测）；
//! - 梯度裁剪：经 `AdamWConfig::with_grad_clipping(Norm(max_norm))` 在
//!   step 内逐参数按 L2 范数缩放（burn 0.21 的裁剪挂点在优化器侧；
//!   与 torch 的全局范数裁剪的差异记录在 PROGRESS.md 遗留问题）。

use burn_core::module::AutodiffModule;
use burn_core::tensor::ElementConversion;
use burn_core::tensor::Tensor;
use burn_core::tensor::backend::AutodiffBackend;
use burn_optim::grad_clipping::GradientClippingConfig;
use burn_optim::{AdamWConfig, GradientsParams, Optimizer};

/// 训练超参。
#[derive(Debug, Clone)]
pub struct TrainCfg {
    /// 初始学习率。
    pub lr: f64,
    /// cosine 退火终点学习率。
    pub lr_min: f64,
    /// AdamW 解耦权重衰减。
    pub weight_decay: f32,
    /// 梯度 L2 范数裁剪上限（逐参数）。
    pub max_grad_norm: f32,
    /// 退火总步数 T。
    pub total_steps: usize,
}

/// cosine 退火学习率（第 `step` 步，step ∈ [0, total_steps]）。
pub fn cosine_lr(cfg: &TrainCfg, step: usize) -> f64 {
    let total = cfg.total_steps.max(1) as f64;
    let t = (step.min(cfg.total_steps) as f64 / total).clamp(0.0, 1.0);
    cfg.lr_min + (cfg.lr - cfg.lr_min) * 0.5 * (1.0 + (std::f64::consts::PI * t).cos())
}

/// 装配 AdamW 优化器（内置梯度范数裁剪）。
/// 返回 `impl Optimizer`（0.21 的 `OptimizerAdaptor` 类型不在公开路径上，
/// 用 `impl Trait` 避免暴露私有路径）。
pub fn make_optimizer<B, M>(cfg: &TrainCfg) -> impl Optimizer<M, B>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    AdamWConfig::new()
        .with_weight_decay(cfg.weight_decay)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(cfg.max_grad_norm)))
        .init::<B, M>()
}

/// 单步训练：forward → loss → backward → optimizer.step。
/// 返回 (更新后的模型，loss 标量值)。
pub fn train_step<B, M, F>(
    model: M,
    optim: &mut impl Optimizer<M, B>,
    lr: f64,
    loss_fn: F,
) -> (M, f32)
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
    F: FnOnce(&M) -> Tensor<B, 1>,
{
    let loss = loss_fn(&model);
    // 先读标量再反传（backward 消耗 loss 的计算图）；泛型后端的标量类型经
    // Element::elem 统一转 f32。
    let value = loss.clone().into_scalar().elem::<f32>();
    let grads = GradientsParams::from_grads(loss.backward(), &model);
    let model = optim.step(lr, model, grads);
    (model, value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> TrainCfg {
        TrainCfg {
            lr: 0.01,
            lr_min: 0.001,
            weight_decay: 0.0,
            max_grad_norm: 10.0,
            total_steps: 100,
        }
    }

    /// cosine 退火端点与单调性手算对照。
    #[test]
    fn cosine_lr_endpoints_and_monotonic() {
        let c = cfg();
        assert!((cosine_lr(&c, 0) - 0.01).abs() < 1e-12, "step 0 = lr0");
        assert!((cosine_lr(&c, 100) - 0.001).abs() < 1e-12, "step T = lr_min");
        // 中点：½(1+cos(π/2)) = ½ → lr = lr_min + (lr0-lr_min)/2
        let mid = cosine_lr(&c, 50);
        assert!((mid - (0.001 + 0.009 / 2.0)).abs() < 1e-12);
        // 单调不增（越界 step 钳位）
        let mut prev = cosine_lr(&c, 0);
        for s in (0..=100).step_by(10) {
            let v = cosine_lr(&c, s);
            assert!(v <= prev + 1e-15);
            prev = v;
        }
        assert_eq!(cosine_lr(&c, 999), cosine_lr(&c, 100), "越界钳位");
    }
}
