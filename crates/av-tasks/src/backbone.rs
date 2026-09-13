//! v0.1 内置骨干：4 段 conv+relu（无 BN），输出 stride 4/8/16 三级特征。
//!
//! 小巧、CPU 可训，服务合成数据端到端冒烟链路；正式骨干族
//! （csp-elan / vit-hybrid 等）按 PLAN §8 M2 落地，通过注册表插拔。

use tch::nn;
use tch::nn::Module;
use tch::Tensor;

use av_core::config::BackboneCfg;
use av_core::error::{AvError, AvResult};
use av_core::traits::{BaseBackbone, BackboneSpec, FeatureMap, FeaturePyramid, LevelSpec};

pub const FAMILY_NAME: &str = "simple-cnn";

pub struct SimpleCnnBackbone {
    conv1: nn::Conv2D,
    conv2: nn::Conv2D,
    conv3: nn::Conv2D,
    conv4: nn::Conv2D,
    width: i64,
}

impl SimpleCnnBackbone {
    pub fn new(p: &nn::Path, cfg: &BackboneCfg) -> Self {
        let width = ((16.0 * cfg.width).round() as i64).max(4);
        let cc = nn::ConvConfig {
            stride: 2,
            padding: 1,
            ..Default::default()
        };
        Self {
            conv1: nn::conv2d(p / "c1", 3, width, 3, cc),
            conv2: nn::conv2d(p / "c2", width, width * 2, 3, cc),
            conv3: nn::conv2d(p / "c3", width * 2, width * 4, 3, cc),
            conv4: nn::conv2d(p / "c4", width * 4, width * 8, 3, cc),
            width,
        }
    }

    pub fn pooled_channels(&self) -> i64 {
        self.width * 8
    }

    pub fn stride_channels(&self, stride: u32) -> AvResult<i64> {
        match stride {
            4 => Ok(self.width * 2),
            8 => Ok(self.width * 4),
            16 => Ok(self.width * 8),
            other => Err(AvError::shape(format!(
                "{FAMILY_NAME} 不存在 stride {other} 的特征层"
            ))),
        }
    }

    fn forward_all(&self, x: &Tensor) -> (Tensor, Tensor, Tensor, Tensor) {
        let a = self.conv1.forward(x).relu();
        let b = self.conv2.forward(&a).relu();
        let c = self.conv3.forward(&b).relu();
        let d = self.conv4.forward(&c).relu();
        (a, b, c, d)
    }
}

impl BaseBackbone for SimpleCnnBackbone {
    fn forward_features(&self, x: &Tensor) -> AvResult<FeaturePyramid> {
        let (_, b, c, d) = self.forward_all(x);
        let mut pyramid = FeaturePyramid::default();
        pyramid.levels.push(FeatureMap::new(b, 4)?);
        pyramid.levels.push(FeatureMap::new(c, 8)?);
        pyramid.levels.push(FeatureMap::new(d, 16)?);
        pyramid.validate_ascending()?;
        Ok(pyramid)
    }

    fn forward_pooled(&self, x: &Tensor) -> AvResult<Tensor> {
        let (_, _, _, d) = self.forward_all(x);
        let pooled = d
            .adaptive_avg_pool2d([1, 1])
            .reshape([-1, self.width * 8]);
        Ok(pooled)
    }

    fn spec(&self) -> BackboneSpec {
        BackboneSpec {
            levels: vec![
                LevelSpec {
                    stride: 4,
                    channels: (self.width * 2) as usize,
                },
                LevelSpec {
                    stride: 8,
                    channels: (self.width * 4) as usize,
                },
                LevelSpec {
                    stride: 16,
                    channels: (self.width * 8) as usize,
                },
            ],
        }
    }
}

#[cfg(all(test, feature = "torch"))]
mod tests {
    use super::*;

    #[test]
    fn feature_pyramid_contract() {
        let vs = tch::nn::VarStore::new(tch::Device::Cpu);
        let cfg = BackboneCfg::default();
        let backbone = SimpleCnnBackbone::new(&vs.root(), &cfg);
        let x = Tensor::randn([2, 3, 64, 64], (tch::Kind::Float, tch::Device::Cpu));
        let pyramid = backbone.forward_features(&x).unwrap();
        let strides: Vec<u32> = pyramid.levels.iter().map(|l| l.stride).collect();
        assert_eq!(strides, vec![4, 8, 16]);
        let pooled = backbone.forward_pooled(&x).unwrap();
        assert_eq!(pooled.size(), vec![2, backbone.pooled_channels()]);
        assert_eq!(backbone.spec().levels.len(), 3);
    }
}
