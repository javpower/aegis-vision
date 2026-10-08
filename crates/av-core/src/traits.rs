//! 六大核心 trait（PLAN §3；[`FeatureNeck`] 为 v1.1 新增第 6 个）。
//!
//! 维度/类型安全策略（替代原方案"生命周期锁死"）：newtype 契约
//! （[`FeatureMap`] 携带 stride/通道元信息并在构造时校验）+ 数据边界断言。
//! trait 只约束 `Send`（多卡数据并行的搬运需求）；`Sync` 待 v1.1
//! 通信方案定型后评估，避免过早锁死实现。

use std::ops::{Add, Mul};

use tch::Tensor;

use crate::error::{AvError, AvResult};
use crate::types::ImageMeta;

/// 损失快照：名称 → 数值（面板 / 日志用，PLAN §7.1）。
pub type LossDict = Vec<(&'static str, f32)>;

/// 带 stride/通道元信息的特征图 newtype（PLAN §3 维度契约）。
#[derive(Debug)]
pub struct FeatureMap {
    pub tensor: Tensor,
    pub stride: u32,
    pub channels: usize,
}

impl FeatureMap {
    /// 形状契约：必须 [N, C, H, W]；通道数在构造期确定，下游装配期即可校验。
    pub fn new(tensor: Tensor, stride: u32) -> AvResult<Self> {
        let size = tensor.size();
        if size.len() != 4 {
            return Err(AvError::shape(format!(
                "FeatureMap 需要 4 维 [N,C,H,W]，实际 {size:?}（stride={stride}）"
            )));
        }
        Ok(Self {
            tensor,
            stride,
            channels: size[1] as usize,
        })
    }

    pub fn height(&self) -> i64 {
        self.tensor.size()[2]
    }

    pub fn width(&self) -> i64 {
        self.tensor.size()[3]
    }
}

/// 多尺度特征金字塔，levels 按 stride 升序（shape 全运行时动态，任意分辨率可用）。
#[derive(Debug, Default)]
pub struct FeaturePyramid {
    pub levels: Vec<FeatureMap>,
}

impl FeaturePyramid {
    /// stride 严格升序契约。
    pub fn validate_ascending(&self) -> AvResult<()> {
        for w in self.levels.windows(2) {
            if w[0].stride >= w[1].stride {
                return Err(AvError::shape(format!(
                    "金字塔 stride 必须严格升序：{} -> {}",
                    w[0].stride, w[1].stride
                )));
            }
        }
        Ok(())
    }
}

/// 骨干结构自述：neck/head 装配时做一次性校验（通道对不上在装配时报错，
/// 而不是等到前向时张量形状炸掉）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackboneSpec {
    pub levels: Vec<LevelSpec>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LevelSpec {
    pub stride: u32,
    pub channels: usize,
}

/// 统一配置入口：TOML 片段 → 具体类型，装配后自检。
pub trait Configurable: Sized {
    fn from_config(cfg: &toml::Value) -> AvResult<Self>;

    fn validate(&self) -> AvResult<()> {
        Ok(())
    }
}

/// 骨干：多尺度特征 + 全局池化 + 结构自述。
pub trait BaseBackbone: Send {
    fn forward_features(&self, x: &Tensor) -> AvResult<FeaturePyramid>;

    /// [N, C] 全局池化特征（分类头复用）。
    fn forward_pooled(&self, x: &Tensor) -> AvResult<Tensor>;

    fn spec(&self) -> BackboneSpec;
}

/// 特征 Neck：多尺度对齐。实现：FPN / PAN；RF-DETR 头自管对齐时配 Identity。
pub trait FeatureNeck: Send {
    fn forward(&self, feats: FeaturePyramid) -> AvResult<FeaturePyramid>;
}

/// 任务头：关联类型把「任务 ↔ 标签 ↔ 原始预测」在类型层面绑死。
pub trait TaskHead: Send {
    type Target;
    type RawPred;

    fn forward(&self, feats: &FeaturePyramid) -> AvResult<Self::RawPred>;
}

/// 损失聚合器：单任务分支损失注册 + 多任务加权合并（PLAN §3）。
pub trait LossAggregator: Default {
    fn add(&mut self, name: &'static str, value: Tensor, weight: f64);

    /// backward 目标（加权合并结果）。
    fn total(&self) -> Tensor;

    fn snapshot(&self) -> LossDict;
}

/// 内置实现：加权求和聚合器（多任务 / 多分支通用）。
#[derive(Default)]
pub struct WeightedSumAggregator {
    parts: Vec<(&'static str, Tensor, f64)>,
}

impl LossAggregator for WeightedSumAggregator {
    fn add(&mut self, name: &'static str, value: Tensor, weight: f64) {
        self.parts.push((name, value, weight));
    }

    fn total(&self) -> Tensor {
        let mut acc: Option<Tensor> = None;
        for (_, v, w) in &self.parts {
            let term = v.mul(&Tensor::from(*w as f32));
            acc = Some(match acc {
                None => term,
                Some(a) => a.add(&term),
            });
        }
        acc.unwrap_or_else(|| Tensor::from(0f32))
    }

    fn snapshot(&self) -> LossDict {
        self.parts
            .iter()
            .map(|(name, v, w)| {
                let scalar = if v.size().is_empty() {
                    v.double_value(&[])
                } else {
                    v.mean_dim(&[0i64][..], true, v.kind()).double_value(&[])
                };
                (*name, (scalar * *w) as f32)
            })
            .collect()
    }
}

/// 后处理器：按任务类型调度（NMS / 旋转 NMS / 掩码解码 / 关键点还原）。
pub trait PostProcessor {
    type RawPred;
    type Output;

    fn process(&self, pred: &Self::RawPred, meta: &ImageMeta) -> AvResult<Self::Output>;
}

#[cfg(all(test, feature = "torch"))]
mod tests {
    use super::*;

    #[test]
    fn weighted_sum_total_and_snapshot() {
        let mut agg = WeightedSumAggregator::default();
        agg.add("cls", Tensor::from(2f32), 1.0);
        agg.add("box", Tensor::from(4f32), 0.5);
        let total = agg.total();
        assert!((total.double_value(&[]) - 4.0).abs() < 1e-6);
        let snap = agg.snapshot();
        assert_eq!(snap.len(), 2);
        assert!((snap[1].1 - 2.0).abs() < 1e-5);
    }

    #[test]
    fn empty_total_is_zero() {
        let agg = WeightedSumAggregator::default();
        assert!(agg.total().double_value(&[]).abs() < 1e-7);
    }

    #[test]
    fn feature_map_shape_contract() {
        let t = Tensor::randn([1, 8, 16, 16], (tch::Kind::Float, tch::Device::Cpu));
        let fm = FeatureMap::new(t, 8).unwrap();
        assert_eq!(fm.channels, 8);
        assert_eq!(fm.height(), 16);

        let bad = Tensor::randn([8, 16, 16], (tch::Kind::Float, tch::Device::Cpu));
        assert!(FeatureMap::new(bad, 8).is_err());
    }

    #[test]
    fn pyramid_stride_ascending_contract() {
        let mut p = FeaturePyramid::default();
        p.levels.push(
            FeatureMap::new(
                Tensor::randn([1, 8, 32, 32], (tch::Kind::Float, tch::Device::Cpu)),
                8,
            )
            .unwrap(),
        );
        p.levels.push(
            FeatureMap::new(
                Tensor::randn([1, 8, 16, 16], (tch::Kind::Float, tch::Device::Cpu)),
                16,
            )
            .unwrap(),
        );
        assert!(p.validate_ascending().is_ok());

        p.levels[1].stride = 8;
        assert!(p.validate_ascending().is_err());
    }
}
