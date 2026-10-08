//! ResNet18 骨干（torchvision 逐层同构），供 [`crate::models`] 的分类骨干
//! 枚举（`ClassifyBackbone`，`family = "resnet18"`）装配。
//!
//! 结构与层名严格对齐 torchvision.models.resnet18（预训练权重兼容是硬要求）：
//!
//! | torchvision 层名                     | AV 变量名（VarStore）                       | 形状        |
//! |--------------------------------------|---------------------------------------------|-------------|
//! | `conv1.weight`                       | `backbone.conv1.weight`                     | [64,3,7,7]  |
//! | `bn1.running_mean`                   | `backbone.bn1.running_mean`                 | [64]        |
//! | `layer1.0.conv1.weight`              | `backbone.layer1.0.conv1.weight`            | [64,64,3,3] |
//! | `layer2.0.downsample.1.running_var`  | `backbone.layer2.0.downsample.1.running_var`| [128]       |
//! | `layer4.1.bn2.bias`                  | `backbone.layer4.1.bn2.bias`                | [512]       |
//!
//! 即 AV 名 = torchvision 名 + `backbone.` 前缀，conv/bn 参数与 BN 统计量
//! **全部同名对齐**，[`av_pretrain::weight_adapter`] 用一条前缀映射
//! （[`av_pretrain::weight_adapter::resnet18_preset`] / configs/resnet18_map.toml）
//! 即可导入 torchvision ImageNet 权重。
//!
//! # BN 统计量（running_mean/running_var）处理方案
//!
//! 不需要手工 buffer 加载：tch 0.24 的 `nn::batch_norm2d` 用
//! `Path::zeros_no_train` / `ones_no_train` 创建 running_mean/running_var
//! （var_store.rs `add(name, tensor, trainable=false)`），它们**就在
//! `VarStore::variables()` 的命名变量集合里**（requires_grad=false，不进
//! trainable_variables、优化器天然跳过）。因此 BN 统计量与普通权重走完全
//! 相同的链路：weight_adapter 导入 → engine 写回 → checkpoint 保存/读回。
//! 单测 [`tests::bn_buffers_live_in_varstore_and_drive_eval`] 锁定这一前提，
//! [`tests::bn_eval_matches_manual_formula`] 锁定推理模式 BN 数值公式。
//!
//! # train 标志
//!
//! `BaseBackbone::forward_features/forward_pooled` 签名不含 train（引擎经
//! `TaskModel::set_train` 装配）。结构体存 `Cell<bool>` 训练标志（默认
//! **false**）+ [`ResNetBackbone::set_train`]。默认 false 即 FrozenBN 微调
//! 语义（推理式 BN：始终用预训练 running 统计量归一化，梯度照常流向
//! conv/γ/β）——检测微调惯用配方（Detectron/mmdet FrozenBN）；引擎训练循环
//! 每 epoch 开始调用 `TaskModel::set_train(true)`、评测前 `set_train(false)`
//! 切到 torch 原生训练态（批统计 + running 更新，由 `Tensor::batch_norm`
//! 原生实现）。`[pretrain].freeze_backbone = true` 时引擎恒保持 false
//! （冻结骨干 = FrozenBN 微调，统计量不随目标域重估）。
//!
//! # 特征金字塔
//!
//! 标准 torchvision 结构下原生 stride：layer1→4、layer2→8、layer3→16、
//! layer4→32。金字塔取 layer1/2/3（stride 4/8/16，与 SimpleCnnBackbone
//! 三级约定一致）；layer4 专供 [`BaseBackbone::forward_pooled`]（GAP 后
//! [N,512]，分类路径，同 torchvision avgpool→fc）。注：ResNet18 真实宽度
//! 为 64/128/256/512（早期任务简报中的 256/512/1024 是 ResNet50 数值）。
//! cfg.width/depth 不参与缩放（改宽将破坏 ImageNet 权重形状兼容，width≠1
//! 直接报错而非静默忽略）。

use std::cell::Cell;

use tch::nn;
use tch::nn::{Module, ModuleT};
use tch::Tensor;

use av_core::config::BackboneCfg;
use av_core::error::{AvError, AvResult};
use av_core::traits::{BackboneSpec, BaseBackbone, FeatureMap, FeaturePyramid, LevelSpec};

/// 注册表族名（av-tasks::register_builtin 登记）。
pub const FAMILY_NAME: &str = "resnet18";

/// ResNet18 各 stage 通道数（torchvision 固定值，不随 cfg 缩放）。
pub const STAGE_CHANNELS: [i64; 4] = [64, 128, 256, 512];
/// 各 stage 第一个块的 stride。
const STAGE_STRIDES: [i64; 4] = [1, 2, 2, 2];
/// 每个 stage 的 BasicBlock 数（resnet18 = [2, 2, 2, 2]）。
const BLOCKS_PER_STAGE: usize = 2;

/// 池化特征通道（layer4 通道数，forward_pooled 输出 [N,512]）。
pub const POOLED_CHANNELS: i64 = STAGE_CHANNELS[3];

// ---------------------------------------------------------------------------
// 基础块（torchvision BasicBlock 同构）
// ---------------------------------------------------------------------------

/// torchvision `BasicBlock`：conv1-bn1-relu-conv2-bn2 (+downsample) 残差相加后 relu。
struct BasicBlock {
    conv1: nn::Conv2D,
    bn1: nn::BatchNorm,
    conv2: nn::Conv2D,
    bn2: nn::BatchNorm,
    /// stride≠1 或通道变化时的投影支路：`downsample.0` conv1x1 + `downsample.1` bn。
    /// 显式存两级路径（而非 nn::Sequential），层名 `downsample.0/1` 仍与
    /// torchvision Sequential 命名逐字一致。
    downsample: Option<(nn::Conv2D, nn::BatchNorm)>,
}

impl BasicBlock {
    fn new(p: &nn::Path, in_ch: i64, out_ch: i64, stride: i64) -> Self {
        let conv1 = nn::conv2d(
            p / "conv1",
            in_ch,
            out_ch,
            3,
            nn::ConvConfig {
                stride,
                padding: 1,
                bias: false,
                ..Default::default()
            },
        );
        let bn1 = nn::batch_norm2d(p / "bn1", out_ch, Default::default());
        let conv2 = nn::conv2d(
            p / "conv2",
            out_ch,
            out_ch,
            3,
            nn::ConvConfig {
                padding: 1,
                bias: false,
                ..Default::default()
            },
        );
        let bn2 = nn::batch_norm2d(p / "bn2", out_ch, Default::default());
        let downsample = if stride != 1 || in_ch != out_ch {
            let ds = p / "downsample";
            let conv = nn::conv2d(
                &ds / "0",
                in_ch,
                out_ch,
                1,
                nn::ConvConfig {
                    stride,
                    bias: false,
                    ..Default::default()
                },
            );
            let bn = nn::batch_norm2d(&ds / "1", out_ch, Default::default());
            Some((conv, bn))
        } else {
            None
        };
        Self {
            conv1,
            bn1,
            conv2,
            bn2,
            downsample,
        }
    }

    fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        let mut out = self.conv1.forward(x);
        out = self.bn1.forward_t(&out, train).relu();
        out = self.conv2.forward(&out);
        out = self.bn2.forward_t(&out, train);
        let identity = match &self.downsample {
            Some((conv, bn)) => bn.forward_t(&conv.forward(x), train),
            None => x.shallow_clone(),
        };
        (out + identity).relu()
    }
}

// ---------------------------------------------------------------------------
// ResNet18 骨干
// ---------------------------------------------------------------------------

/// torchvision ResNet18 同构骨干（conv1/bn1/relu/maxpool/layer1-4/avgpool）。
pub struct ResNetBackbone {
    conv1: nn::Conv2D,
    bn1: nn::BatchNorm,
    /// layer1-4（各 [`BLOCKS_PER_STAGE`] 个 BasicBlock）。
    layers: [Vec<BasicBlock>; 4],
    /// 训练标志（见模块注释「train 标志」；默认 false = FrozenBN 推理语义）。
    train: Cell<bool>,
}

impl ResNetBackbone {
    /// 装配 resnet18。变量名 = torchvision 名 + `backbone.` 前缀（p 为
    /// build_model 传入的 `root / "backbone"`）。
    pub fn new(p: &nn::Path, cfg: &BackboneCfg) -> AvResult<Self> {
        if (cfg.width - 1.0).abs() > 1e-6 {
            return Err(AvError::config(format!(
                "{FAMILY_NAME} 固定 torchvision 宽度（64/128/256/512），width = {} 不支持：\
                 ImageNet 预训练权重形状兼容优先，改宽请用 simple-cnn 或后续 resnet 变宽档",
                cfg.width
            )));
        }
        let stem = nn::ConvConfig {
            stride: 2,
            padding: 3,
            bias: false,
            ..Default::default()
        };
        let conv1 = nn::conv2d(p / "conv1", 3, STAGE_CHANNELS[0], 7, stem);
        let bn1 = nn::batch_norm2d(p / "bn1", STAGE_CHANNELS[0], Default::default());
        let mut layers: [Vec<BasicBlock>; 4] = Default::default();
        let mut in_ch = STAGE_CHANNELS[0];
        for (stage, &out_ch) in STAGE_CHANNELS.iter().enumerate() {
            let lp = p / &format!("layer{}", stage + 1);
            let stride = STAGE_STRIDES[stage];
            let mut blocks = Vec::with_capacity(BLOCKS_PER_STAGE);
            for b in 0..BLOCKS_PER_STAGE {
                let bp = &lp / &b.to_string();
                // 每 stage 首块下采样（stage0 stride=1 无空间下采样）
                blocks.push(BasicBlock::new(
                    &bp,
                    in_ch,
                    out_ch,
                    if b == 0 { stride } else { 1 },
                ));
                in_ch = out_ch;
            }
            layers[stage] = blocks;
        }
        Ok(Self {
            conv1,
            bn1,
            layers,
            train: Cell::new(false),
        })
    }

    /// 切换训练/推理语义（见模块注释「train 标志」）。
    pub fn set_train(&self, train: bool) {
        self.train.set(train);
    }

    pub fn pooled_channels(&self) -> i64 {
        POOLED_CHANNELS
    }

    pub fn stride_channels(&self, stride: u32) -> AvResult<i64> {
        match stride {
            4 => Ok(STAGE_CHANNELS[0]),  // layer1
            8 => Ok(STAGE_CHANNELS[1]),  // layer2
            16 => Ok(STAGE_CHANNELS[2]), // layer3
            other => Err(AvError::shape(format!(
                "{FAMILY_NAME} 不存在 stride {other} 的特征层（金字塔 = layer1/2/3，\
                 layer4 专供 forward_pooled）"
            ))),
        }
    }

    /// 全层前向：返回 (layer1, layer2, layer3, layer4)。
    fn forward_all(&self, x: &Tensor, train: bool) -> (Tensor, Tensor, Tensor, Tensor) {
        let mut x = self.conv1.forward(x);
        x = self.bn1.forward_t(&x, train).relu();
        x = x.max_pool2d([3i64, 3], [2i64, 2], [1i64, 1], [1i64, 1], false);
        let mut feats = Vec::with_capacity(4);
        for stage in &self.layers {
            let mut out = feats.last().unwrap_or(&x).shallow_clone();
            for block in stage {
                out = block.forward(&out, train);
            }
            feats.push(out);
        }
        let mut it = feats.into_iter();
        (
            it.next().expect("layer1"),
            it.next().expect("layer2"),
            it.next().expect("layer3"),
            it.next().expect("layer4"),
        )
    }
}

impl BaseBackbone for ResNetBackbone {
    fn forward_features(&self, x: &Tensor) -> AvResult<FeaturePyramid> {
        let (l1, l2, l3, _l4) = self.forward_all(x, self.train.get());
        let mut pyramid = FeaturePyramid::default();
        pyramid.levels.push(FeatureMap::new(l1, 4)?);
        pyramid.levels.push(FeatureMap::new(l2, 8)?);
        pyramid.levels.push(FeatureMap::new(l3, 16)?);
        pyramid.validate_ascending()?;
        Ok(pyramid)
    }

    fn forward_pooled(&self, x: &Tensor) -> AvResult<Tensor> {
        let (_l1, _l2, _l3, l4) = self.forward_all(x, self.train.get());
        let pooled = l4
            .adaptive_avg_pool2d([1, 1])
            .reshape([-1, POOLED_CHANNELS]);
        Ok(pooled)
    }

    fn spec(&self) -> BackboneSpec {
        BackboneSpec {
            levels: vec![
                LevelSpec {
                    stride: 4,
                    channels: STAGE_CHANNELS[0] as usize,
                },
                LevelSpec {
                    stride: 8,
                    channels: STAGE_CHANNELS[1] as usize,
                },
                LevelSpec {
                    stride: 16,
                    channels: STAGE_CHANNELS[2] as usize,
                },
            ],
        }
    }
}

#[cfg(all(test, feature = "torch"))]
mod tests {
    use super::*;
    use tch::{Device, Kind};

    fn resnet18_backbone(vs: &nn::VarStore) -> ResNetBackbone {
        ResNetBackbone::new(&(vs.root() / "backbone"), &BackboneCfg::default())
            .expect("resnet18 默认配置（width=1.0）应可装配")
    }

    /// 宽度缩放被拒绝（ImageNet 权重形状兼容优先）。
    #[test]
    fn rejects_width_scaling() {
        let vs = nn::VarStore::new(Device::Cpu);
        let cfg = BackboneCfg {
            width: 0.5,
            ..Default::default()
        };
        let err = match ResNetBackbone::new(&(vs.root() / "backbone"), &cfg) {
            Err(e) => e,
            Ok(_) => panic!("width = 0.5 应被拒绝"),
        };
        assert!(err.to_string().contains("width"), "{err}");
    }

    /// 金字塔契约：stride 4/8/16、通道 64/128/256；pooled [N,512]；spec 同款约定。
    #[test]
    fn feature_pyramid_contract() {
        let vs = nn::VarStore::new(Device::Cpu);
        let backbone = resnet18_backbone(&vs);
        let x = Tensor::randn([2, 3, 64, 64], (Kind::Float, Device::Cpu));
        let pyramid = backbone.forward_features(&x).unwrap();
        let strides: Vec<u32> = pyramid.levels.iter().map(|l| l.stride).collect();
        assert_eq!(strides, vec![4, 8, 16]);
        let channels: Vec<usize> = pyramid.levels.iter().map(|l| l.channels).collect();
        assert_eq!(channels, vec![64, 128, 256], "ResNet18 layer1/2/3 真实宽度");
        // stride 契约对输入分辨率成立：64/4=16, 64/8=8, 64/16=4
        for (level, expect) in pyramid.levels.iter().zip([16i64, 8, 4]) {
            assert_eq!(
                level.tensor.size()[2],
                expect,
                "stride {} 空间尺寸",
                level.stride
            );
        }
        let pooled = backbone.forward_pooled(&x).unwrap();
        assert_eq!(pooled.size(), vec![2, backbone.pooled_channels()]);
        assert_eq!(backbone.pooled_channels(), POOLED_CHANNELS);
        // spec 与 stride_channels 一致（装配期校验依据）
        for lv in backbone.spec().levels {
            assert_eq!(
                backbone.stride_channels(lv.stride).unwrap(),
                lv.channels as i64
            );
        }
    }

    /// 320 输入（检测常用分辨率）空间契约：conv1 stride2 + maxpool stride2 后
    /// layer1 出 80（stride4）、layer2 出 40（stride8）、layer3 出 20（stride16），
    /// 通道 64/128/256 = layer1/2/3 真实宽度（检测头装配依据）。
    #[test]
    fn feature_pyramid_spatial_contract_at_320() {
        let vs = nn::VarStore::new(Device::Cpu);
        let backbone = resnet18_backbone(&vs);
        let x = Tensor::randn([2, 3, 320, 320], (Kind::Float, Device::Cpu));
        let pyramid = backbone.forward_features(&x).unwrap();
        for (level, (ch, hw)) in pyramid
            .levels
            .iter()
            .zip([(64i64, 80i64), (128, 40), (256, 20)])
        {
            assert_eq!(
                level.tensor.size(),
                vec![2, ch, hw, hw],
                "stride {}",
                level.stride
            );
        }
    }

    /// 层名清单与 torchvision resnet18 state_dict 逐一同名（+`backbone.` 前缀）：
    /// 100 个变量 = 20 conv 权重 + 40 BN 参数 + 40 BN 统计量。
    #[test]
    fn torchvision_layer_names_inventory() {
        let vs = nn::VarStore::new(Device::Cpu);
        let _ = resnet18_backbone(&vs);

        let mut expected: Vec<String> = vec!["backbone.conv1.weight".into()];
        let bn = ["weight", "bias", "running_mean", "running_var"];
        expected.extend(bn.iter().map(|s| format!("backbone.bn1.{s}")));
        for stage in 1..=4usize {
            for b in 0..BLOCKS_PER_STAGE {
                let base = format!("backbone.layer{stage}.{b}");
                expected.push(format!("{base}.conv1.weight"));
                expected.extend(bn.iter().map(|s| format!("{base}.bn1.{s}")));
                expected.push(format!("{base}.conv2.weight"));
                expected.extend(bn.iter().map(|s| format!("{base}.bn2.{s}")));
                if stage > 1 && b == 0 {
                    expected.push(format!("{base}.downsample.0.weight"));
                    expected.extend(bn.iter().map(|s| format!("{base}.downsample.1.{s}")));
                }
            }
        }
        assert_eq!(expected.len(), 100, "20 conv + 40 BN 参数 + 40 BN 统计量");

        let mut got: Vec<String> = vs
            .variables()
            .into_keys()
            .filter(|n| n.starts_with("backbone."))
            .collect();
        got.sort();
        expected.sort();
        assert_eq!(
            got, expected,
            "变量名必须与 torchvision 层名逐一同名（+前缀）"
        );
    }

    /// BN 统计量是 VarStore 命名变量（no_train）：weight_adapter/engine 写回
    /// 同款 copy_ 即可更新，且推理前向真实消费它（eval 数值随统计量变化）。
    #[test]
    fn bn_buffers_live_in_varstore_and_drive_eval() {
        let vs = nn::VarStore::new(Device::Cpu);
        let backbone = resnet18_backbone(&vs);
        assert!(vs.variables().contains_key("backbone.bn1.running_mean"));
        assert!(vs
            .variables()
            .contains_key("backbone.layer2.0.downsample.1.running_var"));

        let x = Tensor::randn([2, 3, 64, 64], (Kind::Float, Device::Cpu));
        let out1 = backbone.forward_pooled(&x).unwrap();

        // engine apply_pretrain 同款写回路径：变量 shallow_clone 上 copy_
        let shift = Tensor::ones([64i64], (Kind::Float, Device::Cpu)) * 5.0;
        tch::no_grad(|| {
            let mut vars = vs.variables();
            let mut rm = vars
                .remove("backbone.bn1.running_mean")
                .expect("统计量应存在");
            rm.copy_(&shift);
        });
        let out2 = backbone.forward_pooled(&x).unwrap();
        let diff = (&out2 - &out1).abs().max().double_value(&[]);
        assert!(
            diff > 1e-3,
            "改写 running_mean 必须改变推理输出（diff={diff}）"
        );

        // eval 确定性：同输入两次前向逐位一致
        let out3 = backbone.forward_pooled(&x).unwrap();
        assert_eq!((&out3 - &out2).abs().max().double_value(&[]), 0.0);
    }

    /// train 标志切换 BN 语义（引擎 set_train 接线的行为锁定）：train = true
    /// 时前向用**批统计**（同输入两次前向输出一致——批统计量相同，running 不进
    /// 训练态输出），但**更新 running 统计量**（train 前向后 running_mean 漂移）；
    /// train = false 时 FrozenBN 推理语义（确定性、走 running）。默认 false；
    /// set_train(false) 后回到 eval，且 eval 数值消费更新后的 running。
    #[test]
    fn bn_train_flag_switches_batch_stats_and_updates_running() {
        let vs = nn::VarStore::new(Device::Cpu);
        let backbone = resnet18_backbone(&vs);
        let x = Tensor::randn([4, 3, 64, 64], (Kind::Float, Device::Cpu));

        // 默认 eval：确定性
        let e1 = tch::no_grad(|| backbone.forward_pooled(&x).unwrap());
        let e2 = tch::no_grad(|| backbone.forward_pooled(&x).unwrap());
        assert_eq!(
            (&e1 - &e2).abs().max().double_value(&[]),
            0.0,
            "默认 FrozenBN（eval）两次前向应逐位一致"
        );
        let rm_sum_before = vs.variables()["backbone.bn1.running_mean"]
            .sum(Kind::Float)
            .double_value(&[]);

        // train：批统计归一化（输出 ≠ eval 态：running 初值 mean=0/var=1，批统计
        // 是真实批均值/方差），并推进 running 统计量
        backbone.set_train(true);
        let t1 = tch::no_grad(|| backbone.forward_pooled(&x).unwrap());
        let t2 = tch::no_grad(|| backbone.forward_pooled(&x).unwrap());
        assert_eq!(
            (&t1 - &t2).abs().max().double_value(&[]),
            0.0,
            "train 态输出 = 批统计归一化，同批两次前向应一致（running 不进训练态输出）"
        );
        let t_vs_e = tch::no_grad(|| (&t1 - &e1).abs().max().double_value(&[]));
        assert!(t_vs_e > 1e-4, "train 态应用批统计（与 eval 差 {t_vs_e}）");
        let rm_sum_after = vs.variables()["backbone.bn1.running_mean"]
            .sum(Kind::Float)
            .double_value(&[]);
        assert!(
            (rm_sum_after - rm_sum_before).abs() > 1e-6,
            "train 前向应更新 running_mean（sum {rm_sum_before} → {rm_sum_after}）"
        );

        // 切回 eval：恢复确定性，且数值随更新后的 running 统计量改变（统计量被真实消费）
        backbone.set_train(false);
        let e3 = tch::no_grad(|| backbone.forward_pooled(&x).unwrap());
        let e4 = tch::no_grad(|| backbone.forward_pooled(&x).unwrap());
        assert_eq!(
            (&e3 - &e4).abs().max().double_value(&[]),
            0.0,
            "回 eval 应恢复确定性"
        );
        let e3_vs_e1 = tch::no_grad(|| (&e3 - &e1).abs().max().double_value(&[]));
        assert!(
            e3_vs_e1 > 1e-6,
            "eval 前向应消费更新后的 running 统计量（diff={e3_vs_e1}）"
        );
    }

    /// 推理模式 BN 数值 = 手工公式 (x-mean)/sqrt(var+eps)*γ+β（锁定 train=false
    /// 语义走 running 统计量，与 torchvision eval 行为一致）。
    #[test]
    fn bn_eval_matches_manual_formula() {
        let vs = nn::VarStore::new(Device::Cpu);
        let mut bn = nn::batch_norm2d(&(vs.root() / "bn"), 2, Default::default());
        tch::no_grad(|| {
            bn.running_mean.copy_(&Tensor::from_slice(&[1.0f32, -2.0]));
            bn.running_var.copy_(&Tensor::from_slice(&[4.0f32, 0.25]));
            bn.ws
                .as_mut()
                .unwrap()
                .copy_(&Tensor::from_slice(&[2.0f32, 3.0]));
            bn.bs
                .as_mut()
                .unwrap()
                .copy_(&Tensor::from_slice(&[0.5f32, -1.0]));
        });
        let x = Tensor::randn([2, 2, 3, 3], (Kind::Float, Device::Cpu));
        let y = bn.forward_t(&x, false);
        let eps = 1e-5f64;
        let mean = Tensor::from_slice(&[1.0f32, -2.0]).reshape([1i64, 2, 1, 1]);
        let var = Tensor::from_slice(&[4.0f32, 0.25]).reshape([1i64, 2, 1, 1]);
        let gamma = Tensor::from_slice(&[2.0f32, 3.0]).reshape([1i64, 2, 1, 1]);
        let beta = Tensor::from_slice(&[0.5f32, -1.0]).reshape([1i64, 2, 1, 1]);
        let manual = &(&(&x - &mean) / &(&var + eps).sqrt() * &gamma) + &beta;
        let diff = (&y - &manual).abs().max().double_value(&[]);
        assert!(diff < 1e-5, "推理 BN 应等于手工公式（diff={diff}）");
    }
}
