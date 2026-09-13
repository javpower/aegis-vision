//! v0.2 任务模型：骨干 + 任务头的组合，带单任务损失与推理解码。
//!
//! 检测训练管线（PLAN §8 M2 后半，YOLOv8 风格），按 `model.tasks.assigner` 切换：
//! - `"tal"`（默认）：TAL 标签分配（[`crate::assigner`]）→
//!   BCE(cls，正样本按归一化对齐指标软标签加权) + CIoU(reg) + DFL(reg 分布)，
//!   权重 [`LOSS_W_CLS`] / [`LOSS_W_CIOU`] / [`LOSS_W_DFL`] = 1.0 / 5.0 / 1.5；
//! - 其他值（如 `"center"`）：保留 v0.1 单点 cell 分配 + 掩码 L1 路径（回退用）。
//!
//! box 分支为 DFL 参数化（每边 [`REG_MAX`] = 16 bins，头输出 4*16 通道），
//! 经 [`dfl_project`] softmax+积分期望还原为 [N,4,H,W]；
//! 张量域解码 cx,cy = (cell+0.5+tanh(t))*s，w,h = exp(t)*s。
//! `raw_preds` / `loss` / `predict` 对外结构不变（engine 只经 model.loss 与
//! model.predict 两个入口调用）。
//!
//! OBB 旋转框（PLAN §4.2，detect.obb_mode = true，M3）：
//! - 头：box DFL 4 维不变 + 独立角度分支 tθ（见 heads.rs）；
//!   解码 θ = normalize(tanh(tθ)·π/2)，域取 [`AngleDomain`]（obb_mode 默认 Le90）；
//! - 分配：gt 旋转框的外接 Aabb 走普通 TAL（简化及理由见 [`DetectModel::loss_obb`]）；
//! - 损失：BCE(cls) + KFIoU(reg，`crate::kfiou`) + DFL（DFL 继续监督
//!   (tx,ty,tw,th) 分布，保持 DFL 参数化有直接梯度；置 loss_dfl_weight=0 即纯
//!   BCE+KFIoU）；
//! - 预测：Detection 带 angle=Some(θ)（bbox = 未旋转参数框 (cx±w/2, cy±h/2)，
//!   配合 angle 可精确重建 RotBox），NMS 切旋转版（`crate::rot_nms`）。
//!
//! 实例分割（PLAN §4.3，TaskCfg::Seg，head = "yolact" 实时档）：
//! - 头：[`crate::mask::MaskBranch`]——stride 16 特征产出 K=32 原型 [K,img/4,img/4]，
//!   每 cell 系数+类别 [C+K]；实例掩码 = sigmoid(Σ coef_k·proto_k)（设计取舍见 mask.rs）；
//! - 分配：实例掩码质心所在 cell（center 单点分配，拥挤同格后写覆盖先写）；
//! - 损失：BCE(cls，正 cell boost) + BCE(mask) + Dice(mask)（gt 已在 img/4 掩码画布）；
//! - 预测：SegInstance { label, score, mask }，同类别掩码 IoU NMS 去重。
//!
//! 关键点（PLAN §4.4，TaskCfg::Keypoint，head = "direct" 直接回归档）：
//! - 头：[`crate::keypoint::KeypointHead`]——stride 8 解耦卷积，每 cell 输出
//!   [1 cls ⊕ 4 box ⊕ K×(dx,dy) 偏移 ⊕ K 可见性]（选型取舍见 keypoint.rs 模块文档）；
//! - 分配：实例框中心所在 cell（center 单点分配，同 OBB/分割的简化先例）；
//! - 损失：BCE(cls) + 掩码 L1(box) + 可见性加权 L1(偏移，cell 域) + BCE(可见性)
//!   + (1 − mean OKS)（[`crate::oks::oks_loss`]，对偏移坐标可微）；
//! - 预测：Detection { bbox, keypoints: Option<Vec<Kp3>> }——每实例一套关键点
//!   （画布像素 [x,y,v]，v 沿用 COCO 0/1/2 语义，由可见性 logit 阈值 0.5 二值化），
//!   框 IoU NMS 去重。

use std::cell::RefCell;

use tch::nn;
use tch::{Device, Kind, Tensor};

use av_core::config::{RunConfig, TaskCfg};
use av_core::conventions::AngleDomain;
use av_core::error::{AvError, AvResult};
use av_core::geometry::Aabb;
use av_core::traits::BaseBackbone;
use av_core::types::{nms, Detection};

use crate::assigner::{self, TalConfig};
use crate::backbone::SimpleCnnBackbone;
use crate::backbone_dino::{DinoV2Backbone, FAMILY_NAME as DINO_FAMILY};
use crate::backbone_cspelan::{CspElanBackbone, FAMILY_NAME as CSP_FAMILY};
use crate::backbone_resnet::{FAMILY_NAME as RESNET_FAMILY, ResNetBackbone};
use crate::heads::{ClassifyHead, DetectHead, REG_MAX};
use crate::keypoint::KeypointHead;
use crate::kfiou::{kfiou_element, KfTransform};
use crate::mask::{dice_loss, mask_iou, MaskBranch};
use crate::oks::sigma_table;
use crate::rot_nms::{envelope_half_extents, rotate_nms, RotNmsMetric};

/// 训练批数据（v0.1 合成数据源 / YOLO 目录数据源产出）。
pub enum TrainBatch {
    Classify { labels: Tensor },
    /// 每图若干绝对像素 xyxy 框 + 对应类别（真实数据一图多框）
    Detect {
        boxes: Vec<Vec<[f32; 4]>>,
        labels: Vec<Vec<u32>>,
    },
    /// OBB：每图若干旋转框 + 对应类别。
    /// 框格式 [cx, cy, w, h, θ]——绝对像素中心/宽高 + 弧度角
    /// （θ 建议已按 AngleDomain 规范化；loss 内会再做防御性 normalize）。
    Obb {
        boxes: Vec<Vec<[f32; 5]>>,
        labels: Vec<Vec<u32>>,
    },
    /// 实例分割：每图若干二值掩码（img/4 × img/4，0/1 u8，flat）+ 对应类别。
    /// 掩码分辨率与 [`SegModel`] 的掩码画布一致（loader 栅格化时确定）。
    Seg {
        masks: Vec<Vec<Vec<u8>>>,
        labels: Vec<Vec<u32>>,
    },
    /// 关键点（PLAN §4.4）：每图若干实例。
    /// - `boxes`：[cx, cy, w, h] 画布像素（供 center 分配与 OKS 尺度 sqrt(w·h)）；
    /// - `kpts`：每实例 K 个 [x, y, v]（画布像素 + COCO 可见性标志 0/1/2）；
    /// - `labels`：实例类别（当前关键点头是单通道前景，多类仅作记录）。
    Keypoint {
        boxes: Vec<Vec<[f32; 4]>>,
        kpts: Vec<Vec<Vec<[f32; 3]>>>,
        labels: Vec<Vec<u32>>,
    },
}

/// 分割推理单实例：类别 + 分数 + 二值掩码（img/4 × img/4，flat，0/1）。
#[derive(Debug, Clone)]
pub struct SegInstance {
    pub label: u32,
    pub score: f32,
    pub mask: Vec<u8>,
}

/// 推理产物。
pub enum PredictOutput {
    Classify {
        labels: Vec<u32>,
        confs: Vec<f32>,
    },
    Detect {
        per_image: Vec<Vec<Detection>>,
    },
    Seg {
        per_image: Vec<Vec<SegInstance>>,
    },
    /// 关键点：每实例一个 Detection（bbox = 人员框，keypoints = K×[x,y,v] 画布像素）。
    Keypoint {
        per_image: Vec<Vec<Detection>>,
    },
}

pub enum TaskModel {
    Classify(ClassifyModel),
    Detect(DetectModel),
    Seg(SegModel),
    Keypoint(KeypointModel),
}

pub struct ClassifyModel {
    backbone: ClassifyBackbone,
    head: ClassifyHead,
    img_size: u32,
}

/// 分类骨干封装：simple-cnn（既有默认）、resnet18（torchvision 同构，ImageNet
/// 权重经通用 [pretrain] 通道导入）与 dinov2（RF-DETR 同款 DINOv2 预训练配方）。
/// 不用 `Box<dyn BaseBackbone>`：`pooled_channels` 不在 trait 契约里，小枚举直取零开销；
/// 分割/关键点仍走 SimpleCnnBackbone（检测已有 [`DetectBackbone`] 金字塔级枚举）。
enum ClassifyBackbone {
    SimpleCnn(SimpleCnnBackbone),
    ResNet18(ResNetBackbone),
    DinoV2(DinoV2Backbone),
    CspElan(CspElanBackbone),
}

impl ClassifyBackbone {
    fn pooled_channels(&self) -> i64 {
        match self {
            Self::SimpleCnn(b) => b.pooled_channels(),
            Self::ResNet18(b) => b.pooled_channels(),
            Self::DinoV2(b) => b.pooled_channels(),
            Self::CspElan(b) => b.pooled_channels(),
        }
    }

    fn forward_pooled(&self, x: &Tensor) -> AvResult<Tensor> {
        match self {
            Self::SimpleCnn(b) => b.forward_pooled(x),
            Self::ResNet18(b) => b.forward_pooled(x),
            Self::DinoV2(b) => b.forward_pooled(x),
            Self::CspElan(b) => b.forward_pooled(x),
        }
    }

    /// BN train/eval 装配开关：resnet18 → BatchNorm train 语义（批统计 +
    /// running 更新）；simple-cnn / dinov2 无 train 态 BN，no-op。
    fn set_train(&self, train: bool) {
        match self {
            Self::SimpleCnn(_) | Self::DinoV2(_) => {}
            Self::ResNet18(b) => b.set_train(train),
            Self::CspElan(b) => b.set_train(train),
        }
    }
}

/// 分类骨干装配：按 `backbone.family` 分发。`backbone.pretrained` 非 "none" 时
/// 作为 DINOv2 官方 safetensors 权重路径直接导入（含 pos_embed 网格插值，
/// 见 [`DinoV2Backbone::load_dinov2_weights`]；引擎通用 [pretrain] 通道做不了
/// 插值，故 DINOv2 走骨干自己的导入口）。
fn build_classify_backbone(
    p: &nn::Path,
    cfg: &av_core::config::BackboneCfg,
    img_size: u32,
) -> AvResult<ClassifyBackbone> {
    match cfg.family.as_str() {
        DINO_FAMILY => {
            let mut b = DinoV2Backbone::new(p, cfg, img_size)?;
            if !matches!(cfg.pretrained.as_str(), "" | "none") {
                let path = std::path::Path::new(&cfg.pretrained);
                if !path.exists() {
                    return Err(AvError::config(format!(
                        "backbone.pretrained = {:?} 指向的权重文件不存在（下载见 \
                         tools/export/export_dinov2.py）",
                        cfg.pretrained
                    )));
                }
                let stats = b.load_dinov2_weights(path)?;
                println!("[dinov2] 预训练导入 {}", stats.summary());
            }
            Ok(ClassifyBackbone::DinoV2(b))
        }
        RESNET_FAMILY => Ok(ClassifyBackbone::ResNet18(ResNetBackbone::new(p, cfg)?)),
        CSP_FAMILY => Ok(ClassifyBackbone::CspElan(CspElanBackbone::new(p, cfg)?)),
        _ => Ok(ClassifyBackbone::SimpleCnn(SimpleCnnBackbone::new(p, cfg))),
    }
}

/// 检测骨干封装（金字塔级）：simple-cnn（既有默认）与 resnet18（torchvision
/// 同构，ImageNet 权重经通用 [pretrain] + resnet18_map 通道导入）。两者天然
/// 输出 stride 4/8/16 三级金字塔，[`DetectHead`] 的每层输入通道由
/// [`DetectBackbone::stride_channels`] 给出（resnet18 = 64/128/256，即
/// layer1/2/3 真实宽度）。同 [`ClassifyBackbone`]：小枚举直取，不走 dyn。
enum DetectBackbone {
    SimpleCnn(SimpleCnnBackbone),
    ResNet18(ResNetBackbone),
    CspElan(CspElanBackbone),
}

impl DetectBackbone {
    fn forward_features(&self, x: &Tensor) -> AvResult<av_core::traits::FeaturePyramid> {
        match self {
            Self::SimpleCnn(b) => b.forward_features(x),
            Self::ResNet18(b) => b.forward_features(x),
            Self::CspElan(b) => b.forward_features(x),
        }
    }

    fn stride_channels(&self, stride: u32) -> AvResult<i64> {
        match self {
            Self::SimpleCnn(b) => b.stride_channels(stride),
            Self::ResNet18(b) => b.stride_channels(stride),
            Self::CspElan(b) => b.stride_channels(stride),
        }
    }

    /// BN train/eval 装配开关：resnet18 / csp-elan → BatchNorm train 语义
    /// （批统计 + running 更新）；simple-cnn 无 train 态 BN，no-op。
    fn set_train(&self, train: bool) {
        match self {
            Self::SimpleCnn(_) => {}
            Self::ResNet18(b) => b.set_train(train),
            Self::CspElan(b) => b.set_train(train),
        }
    }
}

/// 检测骨干装配：按 `backbone.family` 分发（"resnet18" → ResNetBackbone，
/// 其余维持 simple-cnn 既有行为）。层名 = torchvision 名 + `backbone.` 前缀，
/// 预训练导入走引擎通用 [pretrain] 通道（load_only_backbone 同语义命中）。
fn build_detect_backbone(p: &nn::Path, cfg: &av_core::config::BackboneCfg) -> AvResult<DetectBackbone> {
    match cfg.family.as_str() {
        RESNET_FAMILY => Ok(DetectBackbone::ResNet18(ResNetBackbone::new(p, cfg)?)),
        CSP_FAMILY => Ok(DetectBackbone::CspElan(CspElanBackbone::new(p, cfg)?)),
        _ => Ok(DetectBackbone::SimpleCnn(SimpleCnnBackbone::new(p, cfg))),
    }
}

/// 分割骨干封装：simple-cnn（既有默认）与 resnet18（ImageNet 预训练经通用
/// [pretrain] + resnet18_map 通道导入）。此前 seg 装配硬编码 simple-cnn、
/// 无视 `backbone.family`，导致 seg + resnet18 配置**静默回退**（pretrain
/// 报告 loaded=0 可查）。YOLACT 分支只消费 stride 16 特征，接口同检测侧。
enum SegBackbone {
    SimpleCnn(SimpleCnnBackbone),
    ResNet18(ResNetBackbone),
    CspElan(CspElanBackbone),
    /// DINOv2 ViT-S/14：ViTDet 式 stride 8/16/32 金字塔（×256），预训练权重经
    /// 骨干自己的 load_dinov2_weights 导入（pos_embed 插值 + QKV 融合，通用
    /// [pretrain] 通道做不了），不走引擎 [pretrain] 段。
    DinoV2(DinoV2Backbone),
}

impl SegBackbone {
    fn forward_features(&self, x: &Tensor) -> AvResult<av_core::traits::FeaturePyramid> {
        match self {
            Self::SimpleCnn(b) => b.forward_features(x),
            Self::ResNet18(b) => b.forward_features(x),
            Self::DinoV2(b) => b.forward_features(x),
            Self::CspElan(b) => b.forward_features(x),
        }
    }

    fn stride_channels(&self, stride: u32) -> AvResult<i64> {
        match self {
            Self::SimpleCnn(b) => b.stride_channels(stride),
            Self::ResNet18(b) => b.stride_channels(stride),
            Self::DinoV2(b) => b.stride_channels(stride),
            Self::CspElan(b) => b.stride_channels(stride),
        }
    }

    /// BN train/eval 装配开关：resnet18 → BatchNorm train 语义；simple-cnn /
    /// DINOv2（LayerNorm 无 running 统计量）no-op。
    fn set_train(&self, train: bool) {
        match self {
            Self::SimpleCnn(_) | Self::DinoV2(_) => {}
            Self::ResNet18(b) => b.set_train(train),
            Self::CspElan(b) => b.set_train(train),
        }
    }
}

/// 分割骨干装配：按 `backbone.family` 分发（"resnet18" / "dinov2" /
/// "csp-elan" / 其余 simple-cnn）。DINOv2 需要 img_size（patch 网格与
/// 金字塔尺寸推导）。
fn build_seg_backbone(
    p: &nn::Path,
    cfg: &av_core::config::BackboneCfg,
    img_size: u32,
) -> AvResult<SegBackbone> {
    match cfg.family.as_str() {
        DINO_FAMILY => {
            let mut b = DinoV2Backbone::new(p, cfg, img_size)?;
            if !matches!(cfg.pretrained.as_str(), "" | "none") {
                let path = std::path::Path::new(&cfg.pretrained);
                if !path.exists() {
                    return Err(AvError::config(format!(
                        "backbone.pretrained = {:?} 指向的权重文件不存在（下载见                          tools/export/export_dinov2.py）",
                        cfg.pretrained
                    )));
                }
                let stats = b.load_dinov2_weights(path)?;
                println!("[dinov2] 预训练导入 {}", stats.summary());
            }
            Ok(SegBackbone::DinoV2(b))
        }
        RESNET_FAMILY => Ok(SegBackbone::ResNet18(ResNetBackbone::new(p, cfg)?)),
        CSP_FAMILY => Ok(SegBackbone::CspElan(CspElanBackbone::new(p, cfg)?)),
        _ => Ok(SegBackbone::SimpleCnn(SimpleCnnBackbone::new(p, cfg))),
    }
}

/// OBB 模式参数（detect.obb_mode = true 时装配；PLAN §4.2）。
#[derive(Debug, Clone, Copy)]
pub struct ObbParams {
    /// θ 解码后规范化到该角度域（进 KFIoU / 旋转 NMS 前统一域，PLAN §4.0）
    pub angle_domain: AngleDomain,
}

pub struct DetectModel {
    backbone: DetectBackbone,
    head: DetectHead,
    img_size: u32,
    /// 检测头层级（stride 升序，来自 DetectCfg.head_levels；默认 [8,16]，
    /// 小缺陷场景 [4,8,16] 启用 P2）。分配/损失/解码的层级循环都以它为准。
    head_levels: Vec<u32>,
    /// 标签分配/损失路径："tal" → TAL+CIoU+DFL；其他（如 "center"）→ v0.1 单点+掩码 L1。
    assigner: String,
    /// 损失权重（来自 DetectCfg，默认 = LOSS_W_* 常量）
    loss_w_cls: f64,
    loss_w_ciou: f64,
    loss_w_dfl: f64,
    /// OBB 模式参数；None = 普通检测（水平框）
    obb: Option<ObbParams>,
}

/// 实例分割模型（PLAN §4.3）：骨干 + MaskBranch（原型掩码 + 每实例系数）。
pub struct SegModel {
    backbone: SegBackbone,
    mask_branch: MaskBranch,
    num_classes: i64,
    img_size: u32,
    /// 任务级损失总权重（SegCfg.loss_weight）
    loss_weight: f64,
    /// 掩码 BCE 权重（SegCfg.loss_bce_weight）
    loss_w_bce: f64,
    /// 掩码 Dice 权重（SegCfg.loss_dice_weight）
    loss_w_dice: f64,
}

/// 关键点模型（PLAN §4.4，直接回归档）：骨干 + KeypointHead（stride 8）。
pub struct KeypointModel {
    backbone: SimpleCnnBackbone,
    head: KeypointHead,
    img_size: u32,
    /// 任务级损失总权重（KeypointCfg.loss_weight）
    loss_weight: f64,
    /// OKS 损失项权重（KeypointCfg.loss_oks_weight）
    loss_w_oks: f64,
}

/// 单图关键点实例数上限（conf 过低时防解码爆炸，同 MAX_SEGS_PER_IMAGE）。
pub const MAX_KPS_PER_IMAGE: usize = 100;

/// 关键点路径损失权重（cls 1 / box L1 5，沿用检测 v0.1 惯例；
/// 偏移 L1 2——关键点定位是主任务；可见性 BCE 0.5——辅助输出）。
pub const LOSS_W_KP_CLS: f64 = 1.0;
pub const LOSS_W_KP_BOX: f64 = 5.0;
pub const LOSS_W_KP_OFF: f64 = 2.0;
pub const LOSS_W_KP_VIS: f64 = 0.5;

/// 单图掩码 NMS 前的实例数上限（防 conf 过低时解码爆炸）。
pub const MAX_SEGS_PER_IMAGE: usize = 100;

/// 分割掩码 NMS：按分数降序贪心保留，同类别且与已保留实例掩码 IoU ≥
/// `nms_iou` 的候选被抑制（不同类别互不抑制）。抽出为独立函数以便
/// 手工实例的确定性单测（predict 内的随机权重路径不可复现）。
pub fn mask_nms(mut insts: Vec<SegInstance>, nms_iou: f32) -> Vec<SegInstance> {
    insts.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    insts.truncate(MAX_SEGS_PER_IMAGE);
    let mut kept: Vec<SegInstance> = Vec::new();
    for cand in insts {
        let suppressed = kept.iter().any(|kp| {
            kp.label == cand.label && mask_iou(&kp.mask, &cand.mask) >= nms_iou
        });
        if !suppressed {
            kept.push(cand);
        }
    }
    kept
}

/// 按 run 配置装配模型（单任务；多任务联合按 PLAN §4.5 落地）。
pub fn build_model(p: &nn::Path, cfg: &RunConfig) -> AvResult<TaskModel> {
    let task = cfg
        .model
        .tasks
        .first()
        .ok_or_else(|| AvError::config("model.tasks 不能为空"))?;
    let backbone_cfg = cfg.model.backbone.clone();
    match task {
        TaskCfg::Classify(c) => {
            // 骨干按 family 分发（"dinov2" → DINOv2 ViT-S/14 + 官方预训练导入，
            // 其余维持 simple-cnn 既有行为）
            let backbone =
                build_classify_backbone(&(p / "backbone"), &backbone_cfg, c.img_size)?;
            let head = ClassifyHead::new(
                &(p / "head"),
                backbone.pooled_channels(),
                c.num_classes as i64,
            );
            Ok(TaskModel::Classify(ClassifyModel {
                backbone,
                head,
                img_size: c.img_size,
            }))
        }
        TaskCfg::Detect(d) => {
            if !matches!(d.head.as_str(), "yolo") {
                return Err(AvError::config("rfdetr 头在 M6 落地（PLAN §8）"));
            }
            let head_levels = validate_head_levels(&d.head_levels)?;
            // 骨干按 family 分发（"resnet18" → ResNetBackbone；ImageNet 权重经
            // 引擎 [pretrain] + resnet18_map 通道导入，见 build_detect_backbone）
            let backbone = build_detect_backbone(&(p / "backbone"), &backbone_cfg)?;
            // 每层通道数由骨干 stride → 通道映射给出（P2 = stride_channels(4)；
            // resnet18 = 64/128/256，simple-cnn 按宽度缩放）
            let channels: AvResult<Vec<i64>> =
                head_levels.iter().map(|&s| backbone.stride_channels(s)).collect();
            let channels = channels?;
            // OBB 与普通检测共用检测头代码，只在角度分支与损失/后处理处分叉（PLAN §4.2）
            let head = DetectHead::with_mode(
                &(p / "head"),
                &head_levels,
                &channels,
                d.num_classes as i64,
                d.obb_mode,
            );
            Ok(TaskModel::Detect(DetectModel {
                backbone,
                head,
                img_size: d.img_size,
                head_levels,
                assigner: d.assigner.clone(),
                loss_w_cls: d.loss_cls_weight as f64,
                loss_w_ciou: d.loss_ciou_weight as f64,
                loss_w_dfl: d.loss_dfl_weight as f64,
                obb: d.obb_mode.then(|| ObbParams {
                    // DetectCfg 无 angle 字段，obb_mode 走默认 le90（[-π/2, π/2)，长边域）
                    angle_domain: AngleDomain::Le90,
                }),
            }))
        }
        TaskCfg::Seg(s) => {
            // 实时档（原型掩码）已落地；精度档（RoIAlign 逐实例头）M4 补齐（PLAN §4.3）
            if !matches!(s.head.as_str(), "yolact") {
                return Err(AvError::config(
                    "seg.head = \"direct\"（精度档 RoIAlign 逐实例掩码头）按 M4 落地，\
                     当前请使用 head = \"yolact\"（实时档）",
                ));
            }
            let backbone = build_seg_backbone(&(p / "backbone"), &backbone_cfg, s.img_size)?;
            let mask_branch = MaskBranch::new(
                &(p / "mask"),
                backbone.stride_channels(16)?,
                s.num_classes as i64,
                s.num_protos as i64,
            );
            Ok(TaskModel::Seg(SegModel {
                backbone,
                mask_branch,
                num_classes: s.num_classes as i64,
                img_size: s.img_size,
                loss_weight: s.loss_weight as f64,
                loss_w_bce: s.loss_bce_weight as f64,
                loss_w_dice: s.loss_dice_weight as f64,
            }))
        }
        TaskCfg::Keypoint(k) => {
            // 只落地直接回归档（head = "direct"）；热图/SimDR 未实现（选型取舍
            // 见 keypoint.rs 模块文档：argmax 不可微 + 多实例分组缺失）
            if !matches!(k.decode.as_str(), "direct") {
                return Err(AvError::config(format!(
                    "keypoint.decode = \"{}\" 未落地（PLAN §4.4 精度/实时档按里程碑补齐），\
                     当前请使用 decode = \"direct\"（直接回归档：stride 8 每 cell 回归 \
                     cls+box+K×(dx,dy)+可见性，OKS 可微损失可直连）",
                    k.decode
                )));
            }
            let backbone = SimpleCnnBackbone::new(&(p / "backbone"), &backbone_cfg);
            let head = KeypointHead::new(
                &(p / "head"),
                backbone.stride_channels(8)?,
                k.num_keypoints as i64,
            );
            Ok(TaskModel::Keypoint(KeypointModel {
                backbone,
                head,
                img_size: k.img_size,
                loss_weight: k.loss_weight as f64,
                loss_w_oks: k.loss_oks_weight as f64,
            }))
        }
        TaskCfg::Obb(_) => Err(AvError::config(
            "独立 kind = \"obb\" 任务暂不可装配：ObbCfg 缺少 num_classes/img_size 字段\
             （av-core 本期只读）。请使用 detect 任务 + obb_mode = true（PLAN §4.2：\
             OBB 与普通检测共用检测头，仅角度分支与损失分叉），模型侧能力已完备；\
             如需独立 obb 任务，请在 av-core 的 ObbCfg 增补 num_classes/img_size \
             （及损失权重）字段后再接入",
        )),
    }
}

fn atanh_clamp(x: f32) -> f32 {
    let x = x.clamp(-0.95, 0.95);
    0.5 * ((1.0 + x) / (1.0 - x)).ln()
}

thread_local! {
    /// 诊断通道：训练外注入的调试文本，随下一次 loss() 打印（仅调试用）。
    pub static LOSS_DEBUG: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// 张量 → Vec<f32>（CPU）。
/// copy_data 一次性拷贝（单次 FFI）；此前逐元素 double_value 在 TAL 分配的
/// 每 cell 读取上是实测训练耗时瓶颈（coco128 300 epochs 数小时级）。
fn tensor_to_vec_f32(t: &Tensor) -> Vec<f32> {
    let t = t
        .to_device(Device::Cpu)
        .to_kind(Kind::Float)
        .contiguous()
        .reshape([-1]);
    let n = t.size()[0] as usize;
    let mut dst = vec![0f32; n];
    t.copy_data(&mut dst, n);
    dst
}

/// 张量 → Vec<i64>（CPU）。注意 Kind::Int64 才是 i64（Kind::Int 是 i32，
/// copy_data 的元素类型与张量 dtype 不匹配会 panic）。
fn tensor_to_vec_i64(t: &Tensor) -> Vec<i64> {
    let t = t
        .to_device(Device::Cpu)
        .to_kind(Kind::Int64)
        .contiguous()
        .reshape([-1]);
    let n = t.size()[0] as usize;
    let mut dst = vec![0i64; n];
    t.copy_data(&mut dst, n);
    dst
}

fn pyramid_level(
    py: &av_core::traits::FeaturePyramid,
    stride: i64,
) -> AvResult<&av_core::traits::FeatureMap> {
    py.levels
        .iter()
        .find(|l| l.stride as i64 == stride)
        .ok_or_else(|| AvError::shape(format!("骨干缺少 stride {stride} 特征层")))
}

/// 检测头层级配置校验（build_model 装配期）：非空、严格升序（无重复）、
/// 骨干支持该 stride（simple-cnn：4/8/16）。错误在配置加载后立刻暴露。
fn validate_head_levels(head_levels: &[u32]) -> AvResult<Vec<u32>> {
    if head_levels.is_empty() {
        return Err(AvError::config("detect.head_levels 不能为空"));
    }
    if !head_levels.windows(2).all(|w| w[0] < w[1]) {
        return Err(AvError::config(format!(
            "detect.head_levels 必须严格升序且无重复，got {head_levels:?}"
        )));
    }
    Ok(head_levels.to_vec())
}

/// 层级尺寸区间的单调切分边界：break_i = img_size × level_i / 32。
/// 层 i（升序）负责目标边长 ∈ (break_{i-1}, break_i]——默认 [8,16] 时
/// break = [img/4, img/2]，与历史 `s_break = img_size × 0.25` 两档一致。
fn level_breaks(img_size: u32, head_levels: &[u32]) -> Vec<f32> {
    head_levels
        .iter()
        .map(|&l| img_size as f32 * l as f32 / 32.0)
        .collect()
}

/// 目标边长 → 落层 stride：第一个 break ≥ side 的层；超出全部 break（目标
/// 大于 img_size，理论上不出现）回落最大 stride 层。
fn select_level(head_levels: &[u32], breaks: &[f32], side: f32) -> u32 {
    debug_assert_eq!(head_levels.len(), breaks.len());
    head_levels
        .iter()
        .zip(breaks.iter())
        .find(|(_s, &b)| side <= b)
        .map(|(&s, _)| s)
        .unwrap_or(*head_levels.last().expect("head_levels 非空"))
}

// ---------------------------------------------------------------------------
// 检测：DFL / CIoU / TAL 训练管线（YOLOv8 风格）
// ---------------------------------------------------------------------------

/// DFL bin 中心平移量：softmax 期望域 [0, REG_MAX-1] 平移到 ±(REG_MAX-1)/2，
/// 与 tanh/exp 解码的取值域（0 附近）对齐。
pub const DFL_SHIFT: f32 = (REG_MAX as f32 - 1.0) / 2.0;

/// 损失权重（任务规范默认，YOLOv8 风格）：cls 1.0 / ciou 5.0 / dfl 1.5。
pub const LOSS_W_CLS: f64 = 1.0;
pub const LOSS_W_CIOU: f64 = 5.0;
pub const LOSS_W_DFL: f64 = 1.5;

/// KFIoU 是 KL 域损失（匹配良好时量级 ≪ 1；相同框 → 0），与 CIoU（IoU 域，
/// ≈1 量级）直接同权会让回归梯度偏弱。OBB 路径把 reg 权重放大该系数
/// （经验值：与 CIoU 损失量级对齐；最终权重 = loss_ciou_weight × 本系数）。
pub const LOSS_W_KFIOU_SCALE: f64 = 10.0;

/// DFL 分布 [N, 4*REG_MAX, H, W] → softmax + 积分期望 → [N,4,H,W]。
/// 再平移 -[`DFL_SHIFT`]，得到与 v0.1 同语义的 (tx,ty,tw,th)。
fn dfl_project(dist: &Tensor) -> Tensor {
    let size = dist.size();
    let (n, h, w) = (size[0], size[2], size[3]);
    let device = dist.device();
    let prob = dist.reshape([n, 4, REG_MAX, h, w]).softmax(2, Kind::Float);
    let bins = Tensor::arange(REG_MAX, (Kind::Float, device))
        .reshape([1i64, 1, REG_MAX, 1, 1]);
    let expect = (&prob * &bins).sum_dim_intlist(&[2i64][..], false, Kind::Float);
    expect - (DFL_SHIFT as f64)
}

/// 张量域解码预测框（cxcywh）：[N,4,H,W] 的 (tx,ty,tw,th) → (cx,cy,w,h) [N,4,H,W]。
/// cx=(cell+0.5+tanh(tx))*s，cy 同理；w=exp(tw)*s，h=exp(th)*s。
fn decode_pred_cxcywh(box_raw: &Tensor, s: f32) -> Tensor {
    let size = box_raw.size();
    let (_n, h, w) = (size[0], size[2], size[3]);
    let device = box_raw.device();
    let xs = Tensor::arange(w, (Kind::Float, device)).reshape([1i64, 1, 1, w]);
    let ys = Tensor::arange(h, (Kind::Float, device)).reshape([1i64, 1, h, 1]);
    let tx = box_raw.select(1, 0).unsqueeze(1);
    let ty = box_raw.select(1, 1).unsqueeze(1);
    let tw = box_raw.select(1, 2).unsqueeze(1);
    let th = box_raw.select(1, 3).unsqueeze(1);
    let cx = (&xs + 0.5 + tx.tanh()) * (s as f64);
    let cy = (&ys + 0.5 + ty.tanh()) * (s as f64);
    let bw = tw.exp() * (s as f64);
    let bh = th.exp() * (s as f64);
    Tensor::cat(&[&cx, &cy, &bw, &bh], 1)
}

/// 张量域解码预测框：[N,4,H,W] 的 (tx,ty,tw,th) → xyxy [N,4,H,W]。
fn decode_pred_xyxy(box_raw: &Tensor, s: f32) -> Tensor {
    let b = decode_pred_cxcywh(box_raw, s);
    let cx = b.select(1, 0).unsqueeze(1);
    let cy = b.select(1, 1).unsqueeze(1);
    let bw = b.select(1, 2).unsqueeze(1);
    let bh = b.select(1, 3).unsqueeze(1);
    Tensor::cat(
        &[&(&cx - &bw / 2.0), &(&cy - &bh / 2.0), &(&cx + &bw / 2.0), &(&cy + &bh / 2.0)],
        1,
    )
}

/// 张量版角度规范化（与 av_core::conventions 的 normalize_window 同语义：
/// 周期 π 折叠进 [lo, lo+π)），floor 平移法对任意实数域安全。
/// OBB 解码输入 tθ ∈ (−π/2, π/2)（tanh·π/2），一次折叠必落窗口内
/// （窗口宽恰为 π，端点为开区间），无需再做边界 where 折回。
fn normalize_theta_tensor(t: &Tensor, domain: AngleDomain) -> Tensor {
    let pi = std::f64::consts::PI;
    let (lo, _hi) = domain.range();
    let folds = ((t - (lo as f64)) / pi).floor();
    t - folds * pi
}

/// OBB 角度解码：tθ [N,1,H,W] → θ（弧度，已规范化到 angle_domain）。
/// θ = normalize(tanh(tθ)·π/2)——tanh 保证有界（±π/2 开区间），避免 atanh 型
/// 解码在角度域端点（如 le90 的 ±π/2）发散；角度回归目标经 KFIoU 直接监督。
fn decode_theta(t_theta: &Tensor, domain: AngleDomain) -> Tensor {
    let half_pi = std::f64::consts::FRAC_PI_2;
    let th = t_theta.tanh() * half_pi;
    normalize_theta_tensor(&th, domain)
}

/// CIoU（paperswithcode 公式：IoU − 中心距离项 − 长宽比一致性项）。
/// 入参 pred/gt 均为 xyxy [N,4,H,W]；返回逐 cell 的 1−CIoU [N,H,W]。
fn ciou_element(pred: &Tensor, gt: &Tensor) -> Tensor {
    let px1 = pred.select(1, 0);
    let py1 = pred.select(1, 1);
    let px2 = pred.select(1, 2);
    let py2 = pred.select(1, 3);
    let gx1 = gt.select(1, 0);
    let gy1 = gt.select(1, 1);
    let gx2 = gt.select(1, 2);
    let gy2 = gt.select(1, 3);

    // IoU
    let ix1 = px1.maximum(&gx1);
    let iy1 = py1.maximum(&gy1);
    let ix2 = px2.minimum(&gx2);
    let iy2 = py2.minimum(&gy2);
    let inter = (ix2 - ix1).clamp_min(0.0) * (iy2 - iy1).clamp_min(0.0);
    let parea = (&px2 - &px1) * (&py2 - &py1);
    let garea = (&gx2 - &gx1) * (&gy2 - &gy1);
    let union = (&parea + &garea - &inter).clamp_min(1e-7);
    let iou = &inter / &union;

    // 中心距离项：ρ² / c²（最小包络框对角线）
    let pcx = (&px1 + &px2) / 2.0;
    let pcy = (&py1 + &py2) / 2.0;
    let gcx = (&gx1 + &gx2) / 2.0;
    let gcy = (&gy1 + &gy2) / 2.0;
    let dx = &pcx - &gcx;
    let dy = &pcy - &gcy;
    let rho2 = &dx * &dx + &dy * &dy;
    let ex1 = px1.minimum(&gx1);
    let ey1 = py1.minimum(&gy1);
    let ex2 = px2.maximum(&gx2);
    let ey2 = py2.maximum(&gy2);
    let cdx = ex2 - ex1;
    let cdy = ey2 - ey1;
    let c2 = &(&cdx * &cdx + &cdy * &cdy) + 1e-7;

    // 长宽比一致性项：v = (4/π²)(atan(w_gt/h_gt) − atan(w_p/h_p))²，α = v/(1−IoU+v)
    let pw = (&px2 - &px1).clamp_min(1e-4);
    let ph = (&py2 - &py1).clamp_min(1e-4);
    let gw = (&gx2 - &gx1).clamp_min(1e-4);
    let gh = (&gy2 - &gy1).clamp_min(1e-4);
    let ratio_diff = (&gw / &gh).atan() - (&pw / &ph).atan();
    let v = &ratio_diff * &ratio_diff * (4.0 / (std::f64::consts::PI * std::f64::consts::PI));
    let alpha = &v / ((1.0 - &iou) + &v + 1e-7);

    // CIoU = IoU − ρ²/c² − αv；损失 = 1 − CIoU
    1.0 - &iou + (&rho2 / &c2) + (&alpha * &v)
}

/// DFL 损失（YOLOv8 公式：目标值左右两个 bin 的交叉熵加权）。
/// dist [N,4R,H,W]；target [N,4,H,W] 为 bin 域目标（已平移、须落在 [ε, R−1−ε]）；
/// pos_w [N,1,H,W] 为正样本权重，返回按 Σpos_w 归一的标量损失。
fn dfl_element(dist: &Tensor, target: &Tensor, pos_w: &Tensor) -> Tensor {
    let size = dist.size();
    let (n, h, w) = (size[0], size[2], size[3]);
    let logp = dist
        .reshape([n, 4, REG_MAX, h, w])
        .log_softmax(2, Kind::Float);
    let tl = target.floor(); // ≤ R-2（由目标的 clamp 保证）→ tr ≤ R-1，gather 不越界
    let trf = &tl + 1.0;
    let wl = &trf - target; // 左 bin 权重 ∈ [0,1]
    let wr = target - &tl; // 右 bin 权重 ∈ [0,1]
    let mut total: Option<Tensor> = None;
    for k in 0..4i64 {
        let side = logp.select(1, k); // [N,R,H,W]
        let tl_idx = tl.select(1, k).to_kind(Kind::Int64).unsqueeze(1);
        let tr_idx = trf.select(1, k).to_kind(Kind::Int64).unsqueeze(1);
        let ce_l = side.gather(1, &tl_idx, false).neg(); // −log p(tl)
        let ce_r = side.gather(1, &tr_idx, false).neg(); // −log p(tr)
        let term = (wl.select(1, k).unsqueeze(1) * &ce_l + wr.select(1, k).unsqueeze(1) * &ce_r)
            * pos_w;
        total = Some(match total {
            None => term,
            Some(t) => t + term,
        });
    }
    let num = total.expect("至少一条边").sum(Kind::Float);
    let den = pos_w.sum(Kind::Float).clamp_min(1.0);
    num / den
}

/// 单层 CPU 快照（no_grad 常数侧）：每图每 cell 的解码预测框 + 每 gt 的类别分数行。
struct LevelSnap {
    stride: f32,
    h: usize,
    w: usize,
    /// [n][cells] 解码预测框 xyxy
    pred_boxes: Vec<Vec<[f32; 4]>>,
    /// [n][g][cells] gt 类别 sigmoid 分数
    scores: Vec<Vec<Vec<f32>>>,
}

fn snapshot_level(
    cls: &Tensor,
    box_raw: &Tensor,
    stride: f32,
    gts: &[Vec<[f32; 4]>],
    gt_labels: &[Vec<u32>],
) -> LevelSnap {
    let size = cls.size();
    let h = size[2] as usize;
    let w = size[3] as usize;
    let n = gts.len();
    let cells = h * w;
    let (sig, bx) = tch::no_grad(|| {
        (
            cls.sigmoid().to_device(Device::Cpu),
            box_raw.to_device(Device::Cpu),
        )
    });
    // 向量化读取：整平面一次性拷入 CPU（copy_data 单次 FFI），
    // 替代每 cell 4+1 次 double_value 的逐元素读取（实测瓶颈）。
    let c_len = sig.size()[1] as usize;
    let bx_v = tensor_to_vec_f32(&bx);
    let sig_v = tensor_to_vec_f32(&sig);
    let bidx = |ni: usize, k: usize, hi: usize, wi: usize| ((ni * 4 + k) * h + hi) * w + wi;
    let sidx = |ni: usize, ci: usize, hi: usize, wi: usize| {
        ((ni * c_len + ci) * h + hi) * w + wi
    };
    let mut pred_boxes = vec![vec![[0f32; 4]; cells]; n];
    let mut scores: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n];
    for ni in 0..n {
        for hi in 0..h {
            for wi in 0..w {
                let cell = hi * w + wi;
                let tx = bx_v[bidx(ni, 0, hi, wi)];
                let ty = bx_v[bidx(ni, 1, hi, wi)];
                let tw = bx_v[bidx(ni, 2, hi, wi)];
                let th = bx_v[bidx(ni, 3, hi, wi)];
                let cx = (wi as f32 + 0.5 + tx.tanh()) * stride;
                let cy = (hi as f32 + 0.5 + ty.tanh()) * stride;
                let bw = tw.exp() * stride;
                let bh = th.exp() * stride;
                pred_boxes[ni][cell] = [
                    cx - bw / 2.0,
                    cy - bh / 2.0,
                    cx + bw / 2.0,
                    cy + bh / 2.0,
                ];
            }
        }
        for g in 0..gt_labels[ni].len() {
            let label = gt_labels[ni][g] as usize;
            let mut row = vec![0f32; cells];
            for hi in 0..h {
                for wi in 0..w {
                    row[hi * w + wi] = sig_v[sidx(ni, label, hi, wi)];
                }
            }
            scores[ni].push(row);
        }
    }
    LevelSnap {
        stride,
        h,
        w,
        pred_boxes,
        scores,
    }
}

impl ClassifyModel {
    pub fn logits(&self, x: &Tensor) -> AvResult<Tensor> {
        Ok(self.head.logits(&self.backbone.forward_pooled(x)?))
    }

    pub fn loss(&self, x: &Tensor, batch: &TrainBatch) -> AvResult<Tensor> {
        let TrainBatch::Classify { labels } = batch else {
            return Err(AvError::train("分类模型收到非分类批数据"));
        };
        Ok(self.logits(x)?.cross_entropy_for_logits(labels))
    }

    pub fn predict(&self, x: &Tensor) -> AvResult<(Vec<u32>, Vec<f32>)> {
        tch::no_grad(|| {
            let logits = self.logits(x)?;
            let labels = logits.argmax(-1, false);
            let probs = logits.softmax(-1, Kind::Float);
            let conf = probs.gather(-1, &labels.reshape([-1, 1]), false).reshape([-1]);
            let labels = tensor_to_vec_i64(&labels);
            Ok((
                labels.into_iter().map(|v| v as u32).collect(),
                tensor_to_vec_f32(&conf),
            ))
        })
    }
}

impl DetectModel {
    /// 每层 (stride, cls logits [N,C,H,W], box DFL 分布 [N,4R,H,W], 积分框 [N,4,H,W])。
    /// 层级循环以 head_levels 为准（默认 [8,16]，P2 场景 [4,8,16]）。
    fn forward_levels(&self, x: &Tensor) -> AvResult<Vec<(i64, Tensor, Tensor, Tensor)>> {
        let py = self.backbone.forward_features(x)?;
        let feats: Vec<&Tensor> = self
            .head_levels
            .iter()
            .map(|&s| pyramid_level(&py, s as i64).map(|l| &l.tensor))
            .collect::<AvResult<Vec<&Tensor>>>()?;
        let outs = self.head.forward(&feats);
        Ok(outs
            .into_iter()
            .zip(self.head_levels.iter())
            .map(|((cls, dist, _theta), &s)| {
                let box_raw = dfl_project(&dist);
                (s as i64, cls, dist, box_raw)
            })
            .collect())
    }

    /// OBB 前向：每层 (stride, cls [N,C,H,W], box 积分 [N,4,H,W], tθ raw [N,1,H,W], DFL 分布)。
    fn forward_levels_obb(
        &self,
        x: &Tensor,
    ) -> AvResult<Vec<(i64, Tensor, Tensor, Tensor, Tensor)>> {
        let py = self.backbone.forward_features(x)?;
        let feats: Vec<&Tensor> = self
            .head_levels
            .iter()
            .map(|&s| pyramid_level(&py, s as i64).map(|l| &l.tensor))
            .collect::<AvResult<Vec<&Tensor>>>()?;
        let outs = self.head.forward(&feats);
        let mut out = Vec::with_capacity(self.head_levels.len());
        for ((cls, dist, theta), &s) in outs.into_iter().zip(self.head_levels.iter()) {
            let theta =
                theta.ok_or_else(|| AvError::shape("OBB 模型缺少角度分支输出"))?;
            let box_raw = dfl_project(&dist);
            out.push((s as i64, cls, box_raw, theta, dist));
        }
        Ok(out)
    }

    /// 每层 (stride, cls logits [N,C,H,W], box 积分回归 [N,4,H,W])。
    /// 对外结构保持 v0.1 不变（box 已由 DFL 分布积分还原）。
    pub(crate) fn raw_preds(&self, x: &Tensor) -> AvResult<Vec<(i64, Tensor, Tensor)>> {
        Ok(self
            .forward_levels(x)?
            .into_iter()
            .map(|(s, cls, _dist, box_raw)| (s, cls, box_raw))
            .collect())
    }

    pub fn loss(&self, x: &Tensor, batch: &TrainBatch) -> AvResult<Tensor> {
        match batch {
            TrainBatch::Obb { boxes, labels } => self.loss_obb(x, boxes, labels),
            TrainBatch::Detect { boxes, labels } => {
                if self.obb.is_some() {
                    return Err(AvError::train(
                        "OBB 模型（obb_mode=true）需要 TrainBatch::Obb（含角度 gt）",
                    ));
                }
                if self.assigner.eq_ignore_ascii_case("tal") {
                    self.loss_tal(x, boxes, labels)
                } else {
                    self.loss_center_l1(x, boxes, labels)
                }
            }
            TrainBatch::Classify { .. } => Err(AvError::train("检测模型收到非检测批数据")),
            TrainBatch::Seg { .. } => Err(AvError::train(
                "检测模型收到分割批数据（TrainBatch::Seg 需要 TaskCfg::Seg 模型）",
            )),
            TrainBatch::Keypoint { .. } => Err(AvError::train(
                "检测模型收到关键点批数据（TrainBatch::Keypoint 需要 TaskCfg::Keypoint 模型）",
            )),
        }
    }

    /// YOLOv8 风格训练损失：TAL 分配 → BCE(cls) + CIoU(reg) + DFL(reg 分布)。
    fn loss_tal(
        &self,
        x: &Tensor,
        boxes: &[Vec<[f32; 4]>],
        labels: &[Vec<u32>],
    ) -> AvResult<Tensor> {
        let n = x.size()[0] as usize;
        let device = x.device();
        let tal_cfg = TalConfig::default();
        let c_len_total = self.head.num_classes as usize;

        // 过滤标注类别越界的 gt（数据脏，整条跳过）
        let mut gts: Vec<Vec<[f32; 4]>> = vec![Vec::new(); n];
        let mut gt_labels: Vec<Vec<u32>> = vec![Vec::new(); n];
        for i in 0..n.min(boxes.len()).min(labels.len()) {
            for (b, &l) in boxes[i].iter().zip(&labels[i]) {
                if (l as usize) < c_len_total {
                    gts[i].push(*b);
                    gt_labels[i].push(l);
                }
            }
        }

        let levels = self.forward_levels(x)?;

        // ---- 常数侧：逐层 CPU 快照 + 跨层 TAL 分配（不参与梯度）----
        let snaps: Vec<LevelSnap> = levels
            .iter()
            .map(|(stride, cls, _dist, box_raw)| {
                snapshot_level(cls, box_raw, *stride as f32, &gts, &gt_labels)
            })
            .collect();
        // 层级 cell 前缀偏移：全局 cell 索引 = cell_offsets[li] + 层内索引
        //（N 层通用；两档时退化为历史 if li == 0 { 0 } else { 层0 cells }）
        let cell_offsets: Vec<usize> = {
            let mut offs = Vec::with_capacity(snaps.len() + 1);
            let mut acc = 0usize;
            for snap in &snaps {
                offs.push(acc);
                acc += snap.h * snap.w;
            }
            offs
        };
        let n_cells_total = *cell_offsets.last().expect("至少一层");

        // cell 中心（像素坐标，层 0 在前）
        let mut all_centers: Vec<[f32; 2]> = Vec::with_capacity(n_cells_total * 2);
        for snap in &snaps {
            for hi in 0..snap.h {
                for wi in 0..snap.w {
                    all_centers.push([
                        (wi as f32 + 0.5) * snap.stride,
                        (hi as f32 + 0.5) * snap.stride,
                    ]);
                }
            }
        }
        let mut assignments: Vec<Vec<Option<assigner::PosCell>>> = Vec::with_capacity(n);
        for ni in 0..n {
            let mut all_boxes: Vec<[f32; 4]> = Vec::with_capacity(all_centers.len());
            let mut rows: Vec<Vec<f32>> = Vec::new();
            for snap in &snaps {
                all_boxes.extend_from_slice(&snap.pred_boxes[ni]);
            }
            let g_cnt = snaps[0].scores[ni].len();
            for g in 0..g_cnt {
                let mut row = Vec::with_capacity(all_centers.len());
                for snap in &snaps {
                    row.extend_from_slice(&snap.scores[ni][g]);
                }
                rows.push(row);
            }
            let row_slices: Vec<&[f32]> = rows.iter().map(|r| r.as_slice()).collect();
            assignments.push(assigner::assign_single_image(
                &all_boxes,
                &all_centers,
                &row_slices,
                &gts[ni],
                &tal_cfg,
            ));
        }

        // ---- 梯度侧：逐层构建目标张量并计算三类损失 ----
        let mut total: Option<Tensor> = None;
        for (li, (_stride, cls, dist, box_raw)) in levels.iter().enumerate() {
            let size = cls.size();
            let (c_len, h, w) = (size[1] as usize, size[2] as usize, size[3] as usize);
            let s = snaps[li].stride;
            let cells = h * w;
            let cell_base = cell_offsets[li];

            let mut cls_t = vec![0f32; n * c_len * cells];
            let mut cls_w = vec![1f32; n * c_len * cells];
            let mut gt_box = vec![0f32; n * 4 * cells];
            let mut pos_w = vec![0f32; n * cells];
            // 非 pos cell 的 bin 目标落在合法域内即可（被掩码剔除，但保证 gather 不越界）
            let mut dfl_t = vec![DFL_SHIFT; n * 4 * cells];
            let mut pos_cnt = 0usize;
            let mut pos_cls_idx: Vec<usize> = Vec::new();
            let clamp_dfl =
                |v: f32| v.clamp(1e-4, REG_MAX as f32 - 1.0 - 1e-4);

            for (gi, asg) in assignments.iter().enumerate() {
                for (cell, pos) in asg.iter().enumerate() {
                    let Some(p) = pos else { continue };
                    let local = cell as isize - cell_base as isize;
                    if local < 0 || local as usize >= cells {
                        continue; // 不属于本层
                    }
                    let local = local as usize;
                    let (hi, wi) = (local / w, local % w);
                    let flat2 = gi * cells + hi * w + wi;
                    let gt = gts[gi][p.gt];
                    let (gcx, gcy) = ((gt[0] + gt[2]) / 2.0, (gt[1] + gt[3]) / 2.0);
                    let gw = (gt[2] - gt[0]).max(1e-3);
                    let gh = (gt[3] - gt[1]).max(1e-3);
                    // cls：软标签 = 归一化对齐指标（YOLOv8 norm_align_metric）
                    let cls_flat =
                        (gi * c_len + gt_labels[gi][p.gt] as usize) * cells + hi * w + wi;
                    cls_t[cls_flat] = p.weight;
                    pos_cls_idx.push(cls_flat);
                    pos_w[flat2] = p.weight;
                    pos_cnt += 1;
                    // CIoU 目标（像素 xyxy）
                    for (k, v) in gt.iter().enumerate() {
                        gt_box[(gi * 4 + k) * cells + hi * w + wi] = *v;
                    }
                    // DFL 目标（bin 域）：t = 原始目标 + DFL_SHIFT
                    let offx = gcx / s - (wi as f32 + 0.5);
                    let offy = gcy / s - (hi as f32 + 0.5);
                    let t = [
                        clamp_dfl(atanh_clamp(offx) + DFL_SHIFT),
                        clamp_dfl(atanh_clamp(offy) + DFL_SHIFT),
                        clamp_dfl((gw / s).ln() + DFL_SHIFT),
                        clamp_dfl((gh / s).ln() + DFL_SHIFT),
                    ];
                    for (k, tv) in t.iter().enumerate() {
                        dfl_t[(gi * 4 + k) * cells + hi * w + wi] = *tv;
                    }
                }
            }

            // 正负失衡加权（正样本 class 通道额外加权，上限 50）
            if pos_cnt > 0 {
                let total_elems = n * c_len * cells;
                let boost =
                    (((total_elems - pos_cnt) as f32) / (pos_cnt as f32)).clamp(1.0, 50.0);
                for &idx in &pos_cls_idx {
                    cls_w[idx] = boost;
                }
            }

            let shape_c = [n as i64, c_len as i64, h as i64, w as i64];
            let cls_t_t = Tensor::from_slice(&cls_t).to_device(device).reshape(shape_c);
            let cls_w_t = Tensor::from_slice(&cls_w).to_device(device).reshape(shape_c);
            let cls_loss = cls.binary_cross_entropy_with_logits(
                &cls_t_t,
                Some(&cls_w_t),
                None::<&Tensor>,
                tch::Reduction::Mean,
            );

            let shape_b = [n as i64, 4i64, h as i64, w as i64];
            let gt_box_t = Tensor::from_slice(&gt_box).to_device(device).reshape(shape_b);
            let pred_xyxy = decode_pred_xyxy(box_raw, s);
            let ciou_elem = ciou_element(&pred_xyxy, &gt_box_t); // [N,H,W]
            let pos_w_t = Tensor::from_slice(&pos_w)
                .to_device(device)
                .reshape([n as i64, 1i64, h as i64, w as i64]);
            let ciou_loss = (&ciou_elem * &pos_w_t).sum(Kind::Float)
                / pos_w_t.sum(Kind::Float).clamp_min(1.0);

            let dfl_t_t = Tensor::from_slice(&dfl_t).to_device(device).reshape(shape_b);
            let dfl_loss = dfl_element(dist, &dfl_t_t, &pos_w_t);

            let level_loss = &cls_loss * self.loss_w_cls
                + &(&ciou_loss * self.loss_w_ciou)
                + &(&dfl_loss * self.loss_w_dfl);

            if let Some(dbg) = LOSS_DEBUG.with(|d| d.take()) {
                eprintln!(
                    "[loss-debug] s={s} cls={:.4} ciou={:.4} dfl={:.4} pos={pos_cnt} {dbg}",
                    cls_loss.double_value(&[]),
                    ciou_loss.double_value(&[]),
                    dfl_loss.double_value(&[]),
                );
            }

            total = Some(match total {
                None => level_loss,
                Some(t) => t + level_loss,
            });
        }
        total.ok_or_else(|| AvError::train("检测损失为空：无特征层"))
    }

    /// OBB 训练损失（detect.obb_mode = true）：
    /// 分配 = gt 外接 Aabb 的 TAL（与普通检测同一分配器）；损失 = BCE(cls) + KFIoU(reg) + DFL。
    ///
    /// **分配简化说明**：正样本分配没有把旋转 IoU 加入对齐指标，而是用 gt 旋转框的
    /// 外接 Aabb（envelope）走普通 [`assigner::assign_single_image`]。理由：
    /// 1. 对齐指标需要 CPU 侧逐 cell × 逐 gt 的 IoU 热路径计算，旋转 IoU 是
    ///    多边形裁剪级的标量运算，开销高一个数量级（TAL 快照会成新瓶颈）；
    /// 2. 分配只决定「哪些 cell 学哪个 gt」；回归精度由旋转感知的 KFIoU 保证，
    ///    外接框 over-approximate 只是候选集略宽（正样本略多），不会错配监督；
    /// 3. YOLOv8-OBB 官方实现的 assigner 同样不含角度项（旋转只在回归损失与
    ///    后处理 NMS 处生效），本简化与其一致。
    ///
    /// KFIoU 权重 = `loss_ciou_weight` × [`LOSS_W_KFIOU_SCALE`]（复用 reg 权重槽位）。
    fn loss_obb(
        &self,
        x: &Tensor,
        boxes: &[Vec<[f32; 5]>],
        labels: &[Vec<u32>],
    ) -> AvResult<Tensor> {
        let obb = self
            .obb
            .ok_or_else(|| AvError::train("非 OBB 模型收到 TrainBatch::Obb"))?;
        let n = x.size()[0] as usize;
        let device = x.device();
        let tal_cfg = TalConfig::default();
        let c_len_total = self.head.num_classes as usize;

        // gt (cx,cy,w,h,θ)：θ 防御性规范化；退化 w/h 抬下限；外接 Aabb 供分配
        let mut gts5: Vec<Vec<[f32; 5]>> = vec![Vec::new(); n];
        let mut gts_env: Vec<Vec<[f32; 4]>> = vec![Vec::new(); n];
        let mut gt_labels: Vec<Vec<u32>> = vec![Vec::new(); n];
        for i in 0..n.min(boxes.len()).min(labels.len()) {
            for (b, &l) in boxes[i].iter().zip(&labels[i]) {
                if (l as usize) >= c_len_total {
                    continue; // 标注类别越界（数据脏）整条跳过
                }
                let (cx, cy, w, h) = (b[0], b[1], b[2].max(1e-3), b[3].max(1e-3));
                let th = obb.angle_domain.normalize(b[4]);
                gts5[i].push([cx, cy, w, h, th]);
                let (hw, hh) = envelope_half_extents(w, h, th);
                gts_env[i].push([cx - hw, cy - hh, cx + hw, cy + hh]);
                gt_labels[i].push(l);
            }
        }

        let levels = self.forward_levels_obb(x)?;

        // ---- 常数侧：逐层 CPU 快照 + TAL 分配（与普通检测同构；不参与梯度）----
        // 快照里的 pred_boxes 是未旋转 (tx,ty,tw,th) 的解码——分配简化即建立在其上
        let snaps: Vec<LevelSnap> = levels
            .iter()
            .map(|(stride, cls, box_raw, _theta, _dist)| {
                snapshot_level(cls, box_raw, *stride as f32, &gts_env, &gt_labels)
            })
            .collect();
        // 层级 cell 前缀偏移（N 层通用，同 loss_tal）
        let cell_offsets: Vec<usize> = {
            let mut offs = Vec::with_capacity(snaps.len() + 1);
            let mut acc = 0usize;
            for snap in &snaps {
                offs.push(acc);
                acc += snap.h * snap.w;
            }
            offs
        };
        let n_cells_total = *cell_offsets.last().expect("至少一层");

        let mut all_centers: Vec<[f32; 2]> = Vec::with_capacity(n_cells_total * 2);
        for snap in &snaps {
            for hi in 0..snap.h {
                for wi in 0..snap.w {
                    all_centers.push([
                        (wi as f32 + 0.5) * snap.stride,
                        (hi as f32 + 0.5) * snap.stride,
                    ]);
                }
            }
        }
        let mut assignments: Vec<Vec<Option<assigner::PosCell>>> = Vec::with_capacity(n);
        for ni in 0..n {
            let mut all_boxes: Vec<[f32; 4]> = Vec::with_capacity(all_centers.len());
            let mut rows: Vec<Vec<f32>> = Vec::new();
            for snap in &snaps {
                all_boxes.extend_from_slice(&snap.pred_boxes[ni]);
            }
            let g_cnt = snaps[0].scores[ni].len();
            for g in 0..g_cnt {
                let mut row = Vec::with_capacity(all_centers.len());
                for snap in &snaps {
                    row.extend_from_slice(&snap.scores[ni][g]);
                }
                rows.push(row);
            }
            let row_slices: Vec<&[f32]> = rows.iter().map(|r| r.as_slice()).collect();
            assignments.push(assigner::assign_single_image(
                &all_boxes,
                &all_centers,
                &row_slices,
                &gts_env[ni],
                &tal_cfg,
            ));
        }

        // ---- 梯度侧：逐层构建目标张量并计算 BCE(cls) + KFIoU(reg) + DFL ----
        let mut total: Option<Tensor> = None;
        for (li, (_stride, cls, box_raw, t_theta, dist)) in levels.iter().enumerate() {
            let size = cls.size();
            let (c_len, h, w) = (size[1] as usize, size[2] as usize, size[3] as usize);
            let s = snaps[li].stride;
            let cells = h * w;
            let cell_base = cell_offsets[li];

            let mut cls_t = vec![0f32; n * c_len * cells];
            let mut cls_w = vec![1f32; n * c_len * cells];
            // 非 pos cell 填良性合法框（cx=cell 中心、w=h=1、θ=0）：被掩码剔除，
            // 但保证协方差正定（det>0），KL/log 数值全程有限、无 NaN 风险
            let mut gt5 = vec![0f32; n * 5 * cells];
            for gi in 0..n {
                for hi in 0..h {
                    for wi in 0..w {
                        let base = gi * 5 * cells;
                        gt5[base + hi * w + wi] = (wi as f32 + 0.5) * s;
                        gt5[base + cells + hi * w + wi] = (hi as f32 + 0.5) * s;
                        gt5[base + 2 * cells + hi * w + wi] = 1.0;
                        gt5[base + 3 * cells + hi * w + wi] = 1.0;
                        gt5[base + 4 * cells + hi * w + wi] = 0.0;
                    }
                }
            }
            let mut pos_w = vec![0f32; n * cells];
            // 非 pos cell 的 bin 目标落在合法域内即可（被掩码剔除，但保证 gather 不越界）
            let mut dfl_t = vec![DFL_SHIFT; n * 4 * cells];
            let mut pos_cnt = 0usize;
            let mut pos_cls_idx: Vec<usize> = Vec::new();
            let clamp_dfl = |v: f32| v.clamp(1e-4, REG_MAX as f32 - 1.0 - 1e-4);

            for (gi, asg) in assignments.iter().enumerate() {
                for (cell, pos) in asg.iter().enumerate() {
                    let Some(p) = pos else { continue };
                    let local = cell as isize - cell_base as isize;
                    if local < 0 || local as usize >= cells {
                        continue; // 不属于本层
                    }
                    let local = local as usize;
                    let (hi, wi) = (local / w, local % w);
                    let flat = hi * w + wi;
                    let gt = gts5[gi][p.gt];
                    let (gcx, gcy, gw, gh) = (gt[0], gt[1], gt[2], gt[3]);
                    // cls：软标签 = 归一化对齐指标（YOLOv8 norm_align_metric）
                    let cls_flat =
                        (gi * c_len + gt_labels[gi][p.gt] as usize) * cells + flat;
                    cls_t[cls_flat] = p.weight;
                    pos_cls_idx.push(cls_flat);
                    // pos_w 布局 [N,H,W]：必须带批索引 gi（对齐 loss_tal 的 flat2）。
                    // 漏写 gi 时 bs>1 全部图的正样本权重塌进第 0 张图的平面——
                    // 其余图 KFIoU/DFL 失去监督、第 0 图被跨图权重污染（OBB bs4 精度为 0 的根因）。
                    pos_w[gi * cells + flat] = p.weight;
                    pos_cnt += 1;
                    // KFIoU 目标（cx,cy,w,h,θ，像素域）
                    for (k, v) in gt.iter().enumerate() {
                        gt5[(gi * 5 + k) * cells + flat] = *v;
                    }
                    // DFL 目标（bin 域）：t = 原始目标 + DFL_SHIFT（仅 (tx,ty,tw,th)）
                    let offx = gcx / s - (wi as f32 + 0.5);
                    let offy = gcy / s - (hi as f32 + 0.5);
                    let t = [
                        clamp_dfl(atanh_clamp(offx) + DFL_SHIFT),
                        clamp_dfl(atanh_clamp(offy) + DFL_SHIFT),
                        clamp_dfl((gw / s).ln() + DFL_SHIFT),
                        clamp_dfl((gh / s).ln() + DFL_SHIFT),
                    ];
                    for (k, tv) in t.iter().enumerate() {
                        dfl_t[(gi * 4 + k) * cells + flat] = *tv;
                    }
                }
            }

            // 正负失衡加权（正样本 class 通道额外加权，上限 50）
            if pos_cnt > 0 {
                let total_elems = n * c_len * cells;
                let boost =
                    (((total_elems - pos_cnt) as f32) / (pos_cnt as f32)).clamp(1.0, 50.0);
                for &idx in &pos_cls_idx {
                    cls_w[idx] = boost;
                }
            }

            let shape_c = [n as i64, c_len as i64, h as i64, w as i64];
            let cls_t_t = Tensor::from_slice(&cls_t).to_device(device).reshape(shape_c);
            let cls_w_t = Tensor::from_slice(&cls_w).to_device(device).reshape(shape_c);
            let cls_loss = cls.binary_cross_entropy_with_logits(
                &cls_t_t,
                Some(&cls_w_t),
                None::<&Tensor>,
                tch::Reduction::Mean,
            );

            // KFIoU：解码旋转预测 [N,5,H,W] = (cx,cy,w,h) ⊕ θ
            let gt5_t = Tensor::from_slice(&gt5)
                .to_device(device)
                .reshape([n as i64, 5i64, h as i64, w as i64]);
            let pred_cxcywh = decode_pred_cxcywh(box_raw, s);
            let pred_theta = decode_theta(t_theta, obb.angle_domain);
            let pred5 = Tensor::cat(&[&pred_cxcywh, &pred_theta], 1);
            let kf_elem = kfiou_element(&pred5, &gt5_t, KfTransform::Log1p); // [N,H,W]
            let pos_w_hw = Tensor::from_slice(&pos_w)
                .to_device(device)
                .reshape([n as i64, h as i64, w as i64]);
            let kf_loss = (&kf_elem * &pos_w_hw).sum(Kind::Float)
                / pos_w_hw.sum(Kind::Float).clamp_min(1.0);

            // DFL：与普通 TAL 路径一致，继续监督 (tx,ty,tw,th) 分布
            let shape_b = [n as i64, 4i64, h as i64, w as i64];
            let dfl_t_t = Tensor::from_slice(&dfl_t).to_device(device).reshape(shape_b);
            let dfl_loss = dfl_element(dist, &dfl_t_t, &pos_w_hw.unsqueeze(1));

            let level_loss = &cls_loss * self.loss_w_cls
                + &(&kf_loss * (self.loss_w_ciou * LOSS_W_KFIOU_SCALE))
                + &(&dfl_loss * self.loss_w_dfl);

            if let Some(dbg) = LOSS_DEBUG.with(|d| d.take()) {
                eprintln!(
                    "[loss-debug][obb] s={s} cls={:.4} kfiou={:.4} dfl={:.4} pos={pos_cnt} {dbg}",
                    cls_loss.double_value(&[]),
                    kf_loss.double_value(&[]),
                    dfl_loss.double_value(&[]),
                );
            }

            total = Some(match total {
                None => level_loss,
                Some(t) => t + level_loss,
            });
        }
        total.ok_or_else(|| AvError::train("OBB 损失为空：无特征层"))
    }

    /// v0.1 保留路径：单点 cell 分配（gt 中心所在格）+ 掩码 L1 回归
    /// （assigner = "center" 等非 "tal" 值时启用；回归输入为 DFL 积分后的 [N,4,H,W]）。
    fn loss_center_l1(
        &self,
        x: &Tensor,
        boxes: &[Vec<[f32; 4]>],
        labels: &[Vec<u32>],
    ) -> AvResult<Tensor> {
        let n = x.size()[0] as usize;
        let device = x.device();
        // 层级分配区间（模型常量）：break_i = img_size × level_i / 32 的单调序列，
        // 层 i 负责目标边长 ∈ (break_{i-1}, break_i]；默认 [8,16] 与历史
        // `s_break = img_size × 0.25` 两档行为一致
        let head_levels = self.head_levels.clone();
        let breaks = level_breaks(self.img_size, &head_levels);
        let mut total: Option<Tensor> = None;

        for (s, cls, box_raw) in self.forward_levels(x)?.iter().map(|(s, c, _d, b)| (s, c, b)) {
            let size = cls.size();
            let (c_len, h, w) = (size[1] as usize, size[2] as usize, size[3] as usize);
            let s = *s as f32;
            let mut cls_target = vec![0f32; n * c_len * h * w];
            let mut cls_weight = vec![1f32; n * c_len * h * w];
            let mut box_target = vec![0f32; n * 4 * h * w];
            let mut pos_cnt = 0usize;
            for (gi, (img_boxes, img_labels)) in boxes.iter().zip(labels.iter()).enumerate().take(n)
            {
                for (gt, &label) in img_boxes.iter().zip(img_labels.iter()) {
                    let (x1, y1, x2, y2) = (gt[0], gt[1], gt[2], gt[3]);
                    let (cx, cy) = ((x1 + x2) / 2.0, (y1 + y2) / 2.0);
                    let side = (x2 - x1).max(y2 - y1);
                    let s_sel = select_level(&head_levels, &breaks, side) as f32;
                    if s_sel != s {
                        continue;
                    }
                    if label as usize >= c_len {
                        continue; // 标注类别越界（数据脏）跳过
                    }
                    pos_cnt += 1;
                    let wi = ((cx / s) as usize).min(w - 1);
                    let hi = ((cy / s) as usize).min(h - 1);
                    let cw = ((gi * c_len + label as usize) * h + hi) * w + wi;
                    cls_target[cw] = 1.0;
                    cls_weight[cw] = 50.0; // 正负样本失衡加权
                    let offx = cx / s - (wi as f32 + 0.5);
                    let offy = cy / s - (hi as f32 + 0.5);
                    let tbox = [
                        atanh_clamp(offx),
                        atanh_clamp(offy),
                        ((x2 - x1) / s).ln(),
                        ((y2 - y1) / s).ln(),
                    ];
                    for (k, tv) in tbox.into_iter().enumerate() {
                        box_target[((gi * 4 + k) * h + hi) * w + wi] = tv;
                    }
                }
            }
            let shape = [n as i64, c_len as i64, h as i64, w as i64];
            let cls_t = Tensor::from_slice(&cls_target)
                .to_device(device)
                .reshape(shape);
            let cls_w = Tensor::from_slice(&cls_weight)
                .to_device(device)
                .reshape(shape);
            let cls_loss = cls.binary_cross_entropy_with_logits(
                &cls_t,
                Some(&cls_w),
                None::<&Tensor>,
                tch::Reduction::Mean,
            );
            let shape4 = [n as i64, 4i64, h as i64, w as i64];
            let box_t = Tensor::from_slice(&box_target)
                .to_device(device)
                .reshape(shape4);
            let box_mask = Tensor::from_slice(&{
                let mut m = vec![0f32; n * 4 * h * w];
                for (gi, (img_boxes, _)) in boxes.iter().zip(labels.iter()).enumerate().take(n) {
                    for gt in img_boxes {
                        let (cx, cy) = ((gt[0] + gt[2]) / 2.0, (gt[1] + gt[3]) / 2.0);
                        let side = (gt[2] - gt[0]).max(gt[3] - gt[1]);
                        let s_sel = select_level(&head_levels, &breaks, side) as f32;
                        if s_sel != s {
                            continue;
                        }
                        let wi = ((cx / s) as usize).min(w - 1);
                        let hi = ((cy / s) as usize).min(h - 1);
                        for k in 0..4 {
                            m[((gi * 4 + k) * h + hi) * w + wi] = 1.0;
                        }
                    }
                }
                m
            })
            .to_device(device)
            .reshape(shape4);
            // 回归用掩码 L1（只对正样本 cell 计损失，均值按正样本数重归一）
            let scale = if pos_cnt == 0 {
                0.0
            } else {
                (n * 4 * h * w) as f32 / (pos_cnt * 4) as f32
            };
            let box_loss = (box_raw - &box_t)
                .abs()
                * &box_mask;
            let box_loss = box_loss
                .mean_dim(&[0i64, 1, 2, 3][..], false, Kind::Float)
                * &Tensor::from(scale);
            let level_loss = &cls_loss + &(&box_loss * &Tensor::from(5f32));
            if let Some(dbg) = LOSS_DEBUG.with(|d| d.take()) {
                eprintln!(
                    "[loss-debug] s={s} cls={:.4} box={:.4} (scale={scale:.0}) pos={pos_cnt} {}",
                    cls_loss.double_value(&[]),
                    box_loss.double_value(&[]),
                    dbg
                );
            }
            total = Some(match total {
                None => level_loss,
                Some(t) => &t + &level_loss,
            });
        }
        total.ok_or_else(|| AvError::train("检测损失为空：无特征层"))
    }

    pub fn predict(&self, x: &Tensor, conf: f32, iou: f32) -> AvResult<Vec<Vec<Detection>>> {
        tch::no_grad(|| {
            // OBB：解码含 θ 的候选 → 旋转 NMS（角度感知抑制，PLAN §4.2）
            if let Some(obb) = self.obb {
                let n = x.size()[0] as usize;
                let mut all: Vec<Vec<Detection>> = vec![Vec::new(); n];
                for (s, cls, box_raw, t_theta, _dist) in self.forward_levels_obb(x)? {
                    for (i, dets) in decode_level_obb(
                        s,
                        &cls,
                        &box_raw,
                        &t_theta,
                        conf,
                        obb.angle_domain,
                    )?
                    .into_iter()
                    .enumerate()
                    {
                        all[i].extend(dets);
                    }
                }
                for dets in all.iter_mut() {
                    // 多边形 IoU 为精确度量；ProbIou 近似档留待 TaskCfg::Obb 接入时按
                    // rot_nms 配置切换（当前 obb_mode 走精确档）
                    *dets = rotate_nms(std::mem::take(dets), iou, RotNmsMetric::Polygon);
                }
                return Ok(all);
            }
            let n = x.size()[0] as usize;
            let mut all: Vec<Vec<Detection>> = vec![Vec::new(); n];
            for (s, cls, box_raw) in self.raw_preds(x)? {
                for (i, dets) in decode_level(s, &cls, &box_raw, conf)?
                    .into_iter()
                    .enumerate()
                {
                    all[i].extend(dets);
                }
            }
            for dets in all.iter_mut() {
                *dets = nms(std::mem::take(dets), iou);
            }
            Ok(all)
        })
    }
}

/// OBB 层解码：cls/box/tθ → 逐图候选（CPU 标量循环，同 decode_level 规模）。
///
/// **OBB 的 Detection.bbox 语义**：旋转框参数 (cx, cy, w, h) 的轴对齐形式
/// （即未旋转框 x1y1x2y2），配合 `angle = Some(θ)` 可精确重建 RotBox——这是
/// 旋转 NMS 与坐标还原（letterbox 逆映射）所需的完备信息。需要轴对齐外接
/// 足迹时用 [`rot_nms::envelope_half_extents`] 由 (w, h, θ) 计算。
fn decode_level_obb(
    s: i64,
    cls: &Tensor,
    box_raw: &Tensor,
    t_theta: &Tensor,
    conf: f32,
    domain: AngleDomain,
) -> AvResult<Vec<Vec<Detection>>> {
    let size = cls.size();
    let (n, c, h, w) = (
        size[0] as usize,
        size[1] as usize,
        size[2] as usize,
        size[3] as usize,
    );
    let probs_v = tensor_to_vec_f32(&cls.sigmoid());
    let boxes_v = tensor_to_vec_f32(box_raw);
    let theta_v = tensor_to_vec_f32(t_theta);
    let pidx = |ni: usize, ci: usize, hi: usize, wi: usize| ((ni * c + ci) * h + hi) * w + wi;
    let bidx = |ni: usize, k: usize, hi: usize, wi: usize| ((ni * 4 + k) * h + hi) * w + wi;
    let tidx = |ni: usize, hi: usize, wi: usize| (ni * h + hi) * w + wi;
    let sf = s as f32;
    let half_pi = std::f32::consts::FRAC_PI_2;
    let mut out = vec![Vec::new(); n];
    for ni in 0..n {
        for hi in 0..h {
            for wi in 0..w {
                let (best_ci, best_p) = (0..c)
                    .map(|ci| (ci, probs_v[pidx(ni, ci, hi, wi)]))
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .unwrap_or((0, 0.0));
                if best_p < conf {
                    continue;
                }
                let tx = boxes_v[bidx(ni, 0, hi, wi)];
                let ty = boxes_v[bidx(ni, 1, hi, wi)];
                let tw = boxes_v[bidx(ni, 2, hi, wi)];
                let th = boxes_v[bidx(ni, 3, hi, wi)];
                let cx = (wi as f32 + 0.5 + tx.tanh()) * sf;
                let cy = (hi as f32 + 0.5 + ty.tanh()) * sf;
                let bw = tw.exp() * sf;
                let bh = th.exp() * sf;
                let theta = domain.normalize(theta_v[tidx(ni, hi, wi)].tanh() * half_pi);
                out[ni].push(Detection {
                    // 未旋转参数框（cx±w/2, cy±h/2）+ angle：OBB 重建 RotBox 的精确载体
                    bbox: Aabb::new(cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0),
                    score: best_p,
                    class_id: best_ci as u32,
                    angle: Some(theta),
                    keypoints: None,
                });
            }
        }
    }
    Ok(out)
}

/// 张量解码为逐图候选（CPU、纯标量循环；v0.1 规模足够，批量解码 M2 优化）。
fn decode_level(s: i64, cls: &Tensor, box_raw: &Tensor, conf: f32) -> AvResult<Vec<Vec<Detection>>> {
    let cls = cls.to_device(Device::Cpu).to_kind(Kind::Float);
    let box_raw = box_raw.to_device(Device::Cpu).to_kind(Kind::Float);
    let size = cls.size();
    let (n, c, h, w) = (size[0] as usize, size[1] as usize, size[2] as usize, size[3] as usize);
    let probs_v = tensor_to_vec_f32(&cls.sigmoid());
    let boxes_v = tensor_to_vec_f32(&box_raw);
    let pidx = |ni: usize, ci: usize, hi: usize, wi: usize| {
        ((ni * c + ci) * h + hi) * w + wi
    };
    let bidx = |ni: usize, k: usize, hi: usize, wi: usize| {
        ((ni * 4 + k) * h + hi) * w + wi
    };
    let sf = s as f32;
    let mut out = vec![Vec::new(); n];
    for ni in 0..n {
        for hi in 0..h {
            for wi in 0..w {
                let (best_ci, best_p) = (0..c)
                    .map(|ci| (ci, probs_v[pidx(ni, ci, hi, wi)]))
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .unwrap_or((0, 0.0));
                if best_p < conf {
                    continue;
                }
                let tx = boxes_v[bidx(ni, 0, hi, wi)];
                let ty = boxes_v[bidx(ni, 1, hi, wi)];
                let tw = boxes_v[bidx(ni, 2, hi, wi)];
                let th = boxes_v[bidx(ni, 3, hi, wi)];
                let cx = (wi as f32 + 0.5 + tx.tanh()) * sf;
                let cy = (hi as f32 + 0.5 + ty.tanh()) * sf;
                let bw = tw.exp() * sf;
                let bh = th.exp() * sf;
                out[ni].push(Detection {
                    bbox: Aabb::from_xywh(cx - bw / 2.0, cy - bh / 2.0, bw, bh),
                    score: best_p,
                    class_id: best_ci as u32,
                    angle: None,
                    keypoints: None,
                });
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// 实例分割：SegModel（原型掩码 + 每实例系数，PLAN §4.3）
// ---------------------------------------------------------------------------

impl SegModel {
    /// 掩码画布边长（img/4）。
    pub fn mask_size(&self) -> u32 {
        self.img_size / 4
    }

    /// 输入边长。
    pub fn img_size(&self) -> u32 {
        self.img_size
    }

    fn forward_branch(&self, x: &Tensor) -> AvResult<(Tensor, Tensor)> {
        let py = self.backbone.forward_features(x)?;
        let f16 = pyramid_level(&py, 16)?;
        let m = self.mask_size() as i64;
        Ok(self.mask_branch.forward(&f16.tensor, (m, m)))
    }

    pub fn loss(&self, x: &Tensor, batch: &TrainBatch) -> AvResult<Tensor> {
        let TrainBatch::Seg { masks, labels } = batch else {
            return Err(AvError::train("分割模型收到非分割批数据"));
        };
        let n = x.size()[0] as usize;
        let device = x.device();
        let (proto, coefcls) = self.forward_branch(x)?;
        let cs = coefcls.size();
        // coefcls: [N, C+K, gh, gw]（gh,gw = img/16 网格）
        let gh = cs[2];
        let gw = cs[3];
        let c = self.num_classes;
        let k = self.mask_branch.num_protos;
        let ps = proto.size();
        // proto: [N, K, mh, mw]（mh,mw = img/4 掩码画布）
        let (mh, mw) = (ps[2], ps[3]);
        let cells = (gh * gw) as usize;
        let total_elems = n * (c * gh * gw) as usize;

        let mut cls_t = vec![0f32; total_elems];
        let mut cls_w = vec![1f32; total_elems];
        let mut pos_idx: Vec<usize> = Vec::new();
        let mut mask_losses: Vec<Tensor> = Vec::new();

        for i in 0..n.min(masks.len()).min(labels.len()) {
            for (g, &label) in labels[i].iter().enumerate() {
                let Some(gt_mask) = masks[i].get(g) else { continue };
                if (label as i64) >= c {
                    continue; // 标注类别越界（数据脏）整条跳过
                }
                if gt_mask.len() != (mh * mw) as usize {
                    continue; // 掩码分辨率与模型画布不符（数据脏）跳过
                }
                // 实例质心（掩码画布像素单位）→ stride16 cell
                let (mut sx, mut sy, mut area) = (0f32, 0f32, 0usize);
                for (pi, &v) in gt_mask.iter().enumerate() {
                    if v != 0 {
                        sx += (pi % mw as usize) as f32;
                        sy += (pi / mw as usize) as f32;
                        area += 1;
                    }
                }
                if area == 0 {
                    continue;
                }
                let (cx, cy) = (sx / area as f32, sy / area as f32);
                let cell_w = mw as f32 / gw as f32;
                let cell_h = mh as f32 / gh as f32;
                let wi = ((cx / cell_w) as usize).min(gw as usize - 1);
                let hi = ((cy / cell_h) as usize).min(gh as usize - 1);
                let flat = hi * gw as usize + wi;
                let cls_flat = (i * c as usize + label as usize) * cells + flat;
                cls_t[cls_flat] = 1.0;
                pos_idx.push(cls_flat);

                // 该 cell 的系数向量 [K] → 与原型线性组合成实例掩码 logits [mh,mw]。
                // 原型先过 sigmoid（YOLACT 语义：原型是「基掩码」，系数线性组合后
                // 再过最终 sigmoid）——必须与 predict 的组合方式逐位一致，
                // 否则推理时的掩码与训练学到的组合不可比（实测会导致评测全零）。
                let coef_map = coefcls.select(0, i as i64).narrow(0, c, k); // [K,gh,gw]
                let coef = coef_map.select(1, hi as i64).select(1, wi as i64); // [K]
                let proto_i = proto.select(0, i as i64).sigmoid(); // [K,mh,mw]
                let logit = (&proto_i * &coef.reshape([k, 1i64, 1i64]))
                    .sum_dim_intlist(&[0i64][..], false, Kind::Float);
                let gt_t = Tensor::from_slice(gt_mask)
                    .to_device(device)
                    .to_kind(Kind::Float)
                    .reshape([mh, mw]);
                let bce = logit.binary_cross_entropy_with_logits(
                    &gt_t,
                    None::<&Tensor>,
                    None::<&Tensor>,
                    tch::Reduction::Mean,
                );
                let dl = dice_loss(&logit.sigmoid(), &gt_t);
                mask_losses.push(&bce * self.loss_w_bce + &dl * self.loss_w_dice);
            }
        }

        // cls：正 cell 加权（正负失衡，上限 50，同检测路径）
        if !pos_idx.is_empty() {
            let boost = (((total_elems - pos_idx.len()) as f32) / (pos_idx.len() as f32))
                .clamp(1.0, 50.0);
            for &idx in &pos_idx {
                cls_w[idx] = boost;
            }
        }
        // cls BCE 只作用于前 C 通道（类别分数）；后 K 通道是原型系数，
        // 不应被推向 0（系数没有显式目标，梯度只经掩码损失回传——YOLACT 同款）
        let shape_c = [n as i64, c, gh, gw];
        let cls_t_t = Tensor::from_slice(&cls_t).to_device(device).reshape(shape_c);
        let cls_w_t = Tensor::from_slice(&cls_w).to_device(device).reshape(shape_c);
        let cls_loss = coefcls
            .slice(1, 0, c, 1)
            .binary_cross_entropy_with_logits(
                &cls_t_t,
                Some(&cls_w_t),
                None::<&Tensor>,
                tch::Reduction::Mean,
            );

        // 掩码项：逐实例 BCE+Dice 求均值；批内无实例时退化为纯 cls 项
        // （cls 项恒连通梯度，backward 不会因常数张量断图）
        let mut total = cls_loss;
        if !mask_losses.is_empty() {
            let cnt = mask_losses.len() as f64;
            let mut acc: Option<Tensor> = None;
            for l in mask_losses {
                acc = Some(match acc {
                    None => l,
                    Some(t) => t + l,
                });
            }
            let mask_mean = acc.expect("mask_losses 非空") / cnt;
            total = total + mask_mean;
        }
        Ok(total * self.loss_weight)
    }

    /// 推理：逐 cell 类别分数过 conf → 系数与原型线性组合 → 0.5 阈值二值化
    /// → 同类别掩码 IoU NMS 去重（贪心，按分数降序）。
    pub fn predict(&self, x: &Tensor, conf: f32, nms_iou: f32) -> AvResult<Vec<Vec<SegInstance>>> {
        tch::no_grad(|| {
            let (proto, coefcls) = self.forward_branch(x)?;
            let n = x.size()[0] as usize;
            let cs = coefcls.size();
            let (c, k, gh, gw) = (self.num_classes, self.mask_branch.num_protos, cs[2], cs[3]);
            let ps = proto.size();
            let (mh, mw) = (ps[2] as usize, ps[3] as usize);
            // 整平面一次性拷入 CPU（copy_data 单次 FFI，同 decode_level 模式）
            let proto_v = tensor_to_vec_f32(&proto.sigmoid());
            let cls_v = tensor_to_vec_f32(
                &coefcls
                    .slice(1, 0, c, 1)
                    .sigmoid()
                    .to_device(Device::Cpu)
                    .to_kind(Kind::Float),
            );
            let coef_v = tensor_to_vec_f32(&coefcls.slice(1, c, c + k, 1));
            let c_len = c as usize;
            let k_len = k as usize;
            let pidx = |ni: usize, ci: usize, hi: usize, wi: usize| {
                ((ni * c_len + ci) * gh as usize + hi) * gw as usize + wi
            };
            let fidx = |ni: usize, ki: usize, hi: usize, wi: usize| {
                ((ni * k_len + ki) * gh as usize + hi) * gw as usize + wi
            };
            let pridx = |ni: usize, ki: usize, yi: usize, xi: usize| {
                ((ni * k_len + ki) * mh + yi) * mw + xi
            };

            let mut out: Vec<Vec<SegInstance>> = vec![Vec::new(); n];
            for ni in 0..n {
                for hi in 0..gh as usize {
                    for wi in 0..gw as usize {
                        // 类别分数（argmax over C）
                        let (best_ci, best_p) = (0..c_len)
                            .map(|ci| (ci, cls_v[pidx(ni, ci, hi, wi)]))
                            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                            .unwrap_or((0, 0.0));
                        if best_p < conf {
                            continue;
                        }
                        // 掩码 = sigmoid(Σ coef_k · proto_k)，0.5 阈值二值化
                        let mut coef = vec![0f32; k_len];
                        for ki in 0..k_len {
                            coef[ki] = coef_v[fidx(ni, ki, hi, wi)];
                        }
                        let mut mask = vec![0u8; mh * mw];
                        for yi in 0..mh {
                            for xi in 0..mw {
                                let mut s = 0f32;
                                for ki in 0..k_len {
                                    s += coef[ki] * proto_v[pridx(ni, ki, yi, xi)];
                                }
                                if s > 0.0 {
                                    mask[yi * mw + xi] = 1;
                                }
                            }
                        }
                        if !mask.iter().any(|&v| v == 1) {
                            continue; // 空掩码无意义，丢弃
                        }
                        out[ni].push(SegInstance {
                            label: best_ci as u32,
                            score: best_p,
                            mask,
                        });
                    }
                }
                // 掩码 NMS：分数降序贪心，同类掩码 IoU ≥ 阈值者抑制
                out[ni] = mask_nms(std::mem::take(&mut out[ni]), nms_iou);
            }
            Ok(out)
        })
    }
}

// ---------------------------------------------------------------------------
// 关键点：KeypointModel（直接回归档，PLAN §4.4）
// ---------------------------------------------------------------------------

impl KeypointModel {
    /// 输入边长。
    pub fn img_size(&self) -> u32 {
        self.img_size
    }

    /// 头工作 stride（网格 = img/8，见 [`crate::keypoint::KP_HEAD_STRIDE`]）。
    pub fn kp_stride(&self) -> u32 {
        crate::keypoint::KP_HEAD_STRIDE
    }

    fn forward_head(&self, x: &Tensor) -> AvResult<Tensor> {
        let py = self.backbone.forward_features(x)?;
        let f8 = pyramid_level(&py, self.kp_stride() as i64)?;
        Ok(self.head.forward(&f8.tensor))
    }

    /// 训练损失（decode = "direct"）：
    /// BCE(cls，正 cell boost) + 掩码 L1(box，×5) + 可见性加权 L1(偏移，cell 域)
    /// + BCE(可见性) + (1 − mean OKS)（[`oks_loss`]，对偏移坐标可微），
    /// 权重 [`LOSS_W_KP_CLS`] / [`LOSS_W_KP_BOX`] / [`LOSS_W_KP_OFF`] /
    /// [`LOSS_W_KP_VIS`]，总权重乘 KeypointCfg.loss_weight。
    ///
    /// **分配简化说明**：正样本 = 实例框中心所在 cell（v0.1 检测 center 单点
    /// 分配同款；拥挤同格后写覆盖先写，记录在案——coco8-pose 无碰撞）。与 OBB
    /// 官的外接框简化同理：分配只决定「哪个 cell 学哪个实例」，定位精度由
    /// L1 + OKS 直接监督保证。
    pub fn loss(&self, x: &Tensor, batch: &TrainBatch) -> AvResult<Tensor> {
        let TrainBatch::Keypoint {
            boxes,
            kpts,
            labels: _,
        } = batch
        else {
            return Err(AvError::train("关键点模型收到非关键点批数据"));
        };
        let n = x.size()[0] as usize;
        let device = x.device();
        let k = self.head.num_keypoints as usize;
        let out = self.forward_head(x)?;
        let size = out.size();
        let (h, w) = (size[2] as usize, size[3] as usize);
        let s = self.kp_stride() as f32;
        let cells = h * w;
        let ku = 2 * k; // 偏移通道数（dx,dy 交错）

        // ---- 常数侧：CPU 目标/掩码构建（写法同 loss_center_l1）----
        let mut cls_t = vec![0f32; n * cells];
        let mut cls_w = vec![1f32; n * cells];
        let mut box_t = vec![0f32; n * 4 * cells];
        let mut pos_w = vec![0f32; n * cells]; // box L1 掩码（正 cell）
        let mut off_t = vec![0f32; n * ku * cells];
        let mut off_w = vec![0f32; n * ku * cells]; // 偏移 L1 掩码（按点可见性 δ(v>0)）
        let mut vis_t = vec![0f32; n * k * cells];
        let mut vis_w = vec![0f32; n * k * cells]; // 可见性 BCE 掩码（正 cell 全 K 通道）
        let mut gtx = vec![0f32; n * k * cells]; // OKS gt 坐标图（画布像素）
        let mut gty = vec![0f32; n * k * cells];
        let mut vis_map = vec![0f32; n * k * cells];
        let mut scale_map = vec![1f32; n * cells]; // OKS 实例尺度（非正 cell 被掩掉）
        let mut oks_pos = vec![0f32; n * cells]; // OKS 计入掩码（有可见点的实例才计）
        let mut pos_cnt = 0usize;
        let mut pos_cls_idx: Vec<usize> = Vec::new();

        for (gi, (img_boxes, img_kpts)) in boxes.iter().zip(kpts.iter()).enumerate().take(n) {
            for (gt, gk) in img_boxes.iter().zip(img_kpts.iter()) {
                if gk.len() != k {
                    return Err(AvError::data(format!(
                        "关键点实例点数 {} 与模型 num_keypoints = {k} 不一致（检查数据与配置）",
                        gk.len()
                    )));
                }
                let (gcx, gcy) = (gt[0], gt[1]);
                let (gw, gh) = (gt[2].max(1e-3), gt[3].max(1e-3));
                let wi = ((gcx / s) as usize).min(w - 1);
                let hi = ((gcy / s) as usize).min(h - 1);
                let flat = hi * w + wi;
                let cell = gi * cells + flat;
                cls_t[cell] = 1.0;
                pos_cls_idx.push(cell);
                pos_cnt += 1;
                pos_w[cell] = 1.0;
                scale_map[cell] = (gw * gh).sqrt().max(1e-3);
                let tb = [
                    atanh_clamp(gcx / s - (wi as f32 + 0.5)),
                    atanh_clamp(gcy / s - (hi as f32 + 0.5)),
                    (gw / s).ln(),
                    (gh / s).ln(),
                ];
                for (c, tv) in tb.iter().enumerate() {
                    box_t[(gi * 4 + c) * cells + flat] = *tv;
                }
                let mut any_vis = false;
                for (j, kp) in gk.iter().enumerate() {
                    let vis = kp[2] > 0.0;
                    any_vis |= vis;
                    let vw = if vis { 1.0 } else { 0.0 };
                    let off_base = (gi * ku + 2 * j) * cells + flat;
                    off_t[off_base] = kp[0] / s - (wi as f32 + 0.5);
                    off_w[off_base] = vw;
                    let off_base = off_base + cells;
                    off_t[off_base] = kp[1] / s - (hi as f32 + 0.5);
                    off_w[off_base] = vw;
                    let vi = (gi * k + j) * cells + flat;
                    vis_t[vi] = vw;
                    vis_w[vi] = 1.0; // 可见性监督覆盖正 cell 全部 K 通道（学「点是否标注」）
                    gtx[vi] = kp[0];
                    gty[vi] = kp[1];
                    vis_map[vi] = vw;
                }
                if any_vis {
                    oks_pos[cell] = 1.0;
                }
            }
        }

        // 正负失衡加权（正 cell 的 class 通道额外加权，上限 50，同检测/分割路径）
        if pos_cnt > 0 {
            let total_elems = n * cells;
            let boost = (((total_elems - pos_cnt) as f32) / (pos_cnt as f32)).clamp(1.0, 50.0);
            for &idx in &pos_cls_idx {
                cls_w[idx] = boost;
            }
        }

        let shape_c = [n as i64, 1i64, h as i64, w as i64];
        let cls_t_t = Tensor::from_slice(&cls_t).to_device(device).reshape(shape_c);
        let cls_w_t = Tensor::from_slice(&cls_w).to_device(device).reshape(shape_c);
        let loss_cls = out.narrow(1, crate::keypoint::KP_CLS_CH, 1)
            .binary_cross_entropy_with_logits(
                &cls_t_t,
                Some(&cls_w_t),
                None::<&Tensor>,
                tch::Reduction::Mean,
            );

        // 空批（无实例）兜底：cls 项恒连通梯度，backward 不会因常数张量断图
        // （同 SegModel::loss 的处理），box/offset/vis/OKS 项此时无监督意义
        if pos_cnt == 0 {
            return Ok(loss_cls * self.loss_weight);
        }

        let shape_b = [n as i64, 4i64, h as i64, w as i64];
        let box_t_t = Tensor::from_slice(&box_t).to_device(device).reshape(shape_b);
        let pos_w_t = Tensor::from_slice(&pos_w)
            .to_device(device)
            .reshape([n as i64, 1i64, h as i64, w as i64]);
        let box_l1 = (out.narrow(1, crate::keypoint::KP_BOX_CH, 4) - &box_t_t).abs() * &pos_w_t;
        let box_l1 = box_l1.sum(Kind::Float) / (&pos_w_t * 4.0).sum(Kind::Float).clamp_min(1.0);

        let shape_o = [n as i64, ku as i64, h as i64, w as i64];
        let off_t_t = Tensor::from_slice(&off_t).to_device(device).reshape(shape_o);
        let off_w_t = Tensor::from_slice(&off_w).to_device(device).reshape(shape_o);
        let off_l1 = (out.narrow(1, crate::keypoint::KP_OFF_CH, ku as i64) - &off_t_t).abs()
            * &off_w_t;
        let off_l1 = off_l1.sum(Kind::Float) / off_w_t.sum(Kind::Float).clamp_min(1.0);

        let shape_v = [n as i64, k as i64, h as i64, w as i64];
        let vis_t_t = Tensor::from_slice(&vis_t).to_device(device).reshape(shape_v);
        let vis_w_t = Tensor::from_slice(&vis_w).to_device(device).reshape(shape_v);
        let loss_vis = out
            .narrow(1, crate::keypoint::KP_OFF_CH + ku as i64, k as i64)
            .binary_cross_entropy_with_logits(
                &vis_t_t,
                Some(&vis_w_t),
                None::<&Tensor>,
                tch::Reduction::Mean,
            );

        // ---- OKS 项：解码坐标图（对偏移线性可微）逐点 exp(−d²/(2·s²·σ²)) ----
        // 逐元素广播实现，无 gather；不可见点的坐标图无意义但被 vis_map 掩掉，
        // 其 (0,0) 退化目标不产生损失路径（exp 项有界，无 NaN 风险）
        let gtx_t = Tensor::from_slice(&gtx).to_device(device).reshape(shape_v);
        let gty_t = Tensor::from_slice(&gty).to_device(device).reshape(shape_v);
        let vis_map_t = Tensor::from_slice(&vis_map).to_device(device).reshape(shape_v);
        let scale_map_t = Tensor::from_slice(&scale_map)
            .to_device(device)
            .reshape([n as i64, 1i64, h as i64, w as i64]);
        let oks_pos_t = Tensor::from_slice(&oks_pos)
            .to_device(device)
            .reshape([n as i64, 1i64, h as i64, w as i64]);
        let sigma = Tensor::from_slice(&sigma_table(k))
            .to_device(device)
            .reshape([1i64, k as i64, 1i64, 1i64]);
        let xs = Tensor::arange(w as i64, (Kind::Float, device))
            .reshape([1i64, 1i64, 1i64, w as i64]);
        let ys = Tensor::arange(h as i64, (Kind::Float, device))
            .reshape([1i64, 1i64, h as i64, 1i64]);
        let off_raw = out
            .narrow(1, crate::keypoint::KP_OFF_CH, ku as i64)
            .reshape([n as i64, k as i64, 2i64, h as i64, w as i64]);
        let dx = off_raw.select(2, 0); // [N,K,H,W]
        let dy = off_raw.select(2, 1);
        let px = (&xs + 0.5) * (s as f64) + &dx * (s as f64);
        let py = (&ys + 0.5) * (s as f64) + &dy * (s as f64);
        let ex = &px - &gtx_t;
        let ey = &py - &gty_t;
        let d2 = &ex * &ex + &ey * &ey;
        let denom = &(&(&scale_map_t * &scale_map_t) * 2.0) * &sigma * &sigma; // [N,K,H,W] = 2·s²·σ²
        let e = (&d2 / &denom).neg().exp();
        let num = (&e * &vis_map_t).sum_dim_intlist(&[1i64][..], false, Kind::Float); // [N,1,H,W]
        let den = vis_map_t
            .sum_dim_intlist(&[1i64][..], false, Kind::Float)
            .clamp_min(1.0);
        let ok_map = &num / &den;
        let ok_mean = (&ok_map * &oks_pos_t).sum(Kind::Float)
            / oks_pos_t.sum(Kind::Float).clamp_min(1.0);
        // 损失 = 1 − mean OKS（标量在左的 `1.0 - &t` 会踩 E0282，改写为右乘加）
        let loss_oks = ok_mean * -1.0 + 1.0;

        if let Some(dbg) = LOSS_DEBUG.with(|d| d.take()) {
            eprintln!(
                "[loss-debug][kp] cls={:.4} box={:.4} off={:.4} vis={:.4} oks={:.4} pos={pos_cnt} {dbg}",
                loss_cls.double_value(&[]),
                box_l1.double_value(&[]),
                off_l1.double_value(&[]),
                loss_vis.double_value(&[]),
                loss_oks.double_value(&[]),
            );
        }

        Ok((&loss_cls * LOSS_W_KP_CLS
            + &(&box_l1 * LOSS_W_KP_BOX)
            + &(&off_l1 * LOSS_W_KP_OFF)
            + &(&loss_vis * LOSS_W_KP_VIS)
            + &(&loss_oks * self.loss_w_oks))
            * self.loss_weight)
    }

    /// 推理：每 cell 类别分数过 conf → 解码实例框 + K×[x,y,v] → 分数降序截断
    /// → 框 IoU NMS 去重。关键点坐标 clamp 到画布内；v = 可见性概率 > 0.5 ?
    /// 2 : 0（COCO 标志语义，预测侧不输出 1=遮挡的中间态）。
    pub fn predict(&self, x: &Tensor, conf: f32, iou: f32) -> AvResult<Vec<Vec<Detection>>> {
        tch::no_grad(|| {
            let out = self.forward_head(x)?;
            let k = self.head.num_keypoints as usize;
            let size = out.size();
            let (n, h, w) = (size[0] as usize, size[2] as usize, size[3] as usize);
            let sf = self.kp_stride() as f32;
            let lim = self.img_size as f32;
            // 整平面一次性拷入 CPU（copy_data 单次 FFI，同 decode_level 模式）
            let cls_v = tensor_to_vec_f32(&out.narrow(1, crate::keypoint::KP_CLS_CH, 1).sigmoid());
            let box_v = tensor_to_vec_f32(&out.narrow(1, crate::keypoint::KP_BOX_CH, 4));
            let off_v =
                tensor_to_vec_f32(&out.narrow(1, crate::keypoint::KP_OFF_CH, ku_of(k)));
            let vis_v = tensor_to_vec_f32(
                &out.narrow(1, crate::keypoint::KP_OFF_CH + 2 * k as i64, k as i64).sigmoid(),
            );
            let cidx = |ni: usize, hi: usize, wi: usize| (ni * h + hi) * w + wi;
            let bidx =
                |ni: usize, c: usize, hi: usize, wi: usize| ((ni * 4 + c) * h + hi) * w + wi;
            let oidx =
                |ni: usize, c: usize, hi: usize, wi: usize| ((ni * 2 * k + c) * h + hi) * w + wi;
            let vidx =
                |ni: usize, j: usize, hi: usize, wi: usize| ((ni * k + j) * h + hi) * w + wi;

            let mut out_imgs: Vec<Vec<Detection>> = vec![Vec::new(); n];
            for ni in 0..n {
                for hi in 0..h {
                    for wi in 0..w {
                        let p = cls_v[cidx(ni, hi, wi)];
                        if p < conf {
                            continue;
                        }
                        let tx = box_v[bidx(ni, 0, hi, wi)];
                        let ty = box_v[bidx(ni, 1, hi, wi)];
                        let tw = box_v[bidx(ni, 2, hi, wi)];
                        let th = box_v[bidx(ni, 3, hi, wi)];
                        let cx = (wi as f32 + 0.5 + tx.tanh()) * sf;
                        let cy = (hi as f32 + 0.5 + ty.tanh()) * sf;
                        let bw = tw.exp() * sf;
                        let bh = th.exp() * sf;
                        let mut kps = Vec::with_capacity(k);
                        for j in 0..k {
                            let kx = (wi as f32 + 0.5 + off_v[oidx(ni, 2 * j, hi, wi)]) * sf;
                            let ky = (hi as f32 + 0.5 + off_v[oidx(ni, 2 * j + 1, hi, wi)]) * sf;
                            let v = if vis_v[vidx(ni, j, hi, wi)] > 0.5 { 2.0 } else { 0.0 };
                            kps.push([kx.clamp(0.0, lim), ky.clamp(0.0, lim), v]);
                        }
                        out_imgs[ni].push(Detection {
                            bbox: Aabb::new(
                                cx - bw / 2.0,
                                cy - bh / 2.0,
                                cx + bw / 2.0,
                                cy + bh / 2.0,
                            ),
                            score: p,
                            class_id: 0,
                            angle: None,
                            keypoints: Some(kps),
                        });
                    }
                }
                // 分数降序 + 上限截断 + 框 IoU NMS（贪心）
                let mut dets = std::mem::take(&mut out_imgs[ni]);
                dets.sort_by(|a, b| {
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                dets.truncate(MAX_KPS_PER_IMAGE);
                out_imgs[ni] = nms(dets, iou);
            }
            Ok(out_imgs)
        })
    }
}

/// 头偏移分支通道数（K 点 × 2）。
fn ku_of(k: usize) -> i64 {
    (2 * k) as i64
}

impl TaskModel {
    pub fn num_classes(&self) -> i64 {
        match self {
            TaskModel::Classify(m) => m.head.num_classes,
            TaskModel::Detect(m) => m.head.num_classes,
            TaskModel::Seg(m) => m.num_classes,
            // 关键点头是单通道前景（人员），无多类概念
            TaskModel::Keypoint(_) => 1,
        }
    }

    pub fn img_size(&self) -> u32 {
        match self {
            TaskModel::Classify(m) => m.img_size,
            TaskModel::Detect(m) => m.img_size,
            TaskModel::Seg(m) => m.img_size,
            TaskModel::Keypoint(m) => m.img_size,
        }
    }

    pub fn loss(&self, x: &Tensor, batch: &TrainBatch) -> AvResult<Tensor> {
        match self {
            TaskModel::Classify(m) => m.loss(x, batch),
            TaskModel::Detect(m) => m.loss(x, batch),
            TaskModel::Seg(m) => m.loss(x, batch),
            TaskModel::Keypoint(m) => m.loss(x, batch),
        }
    }

    pub fn predict(&self, x: &Tensor, conf: f32, iou: f32) -> AvResult<PredictOutput> {
        match self {
            TaskModel::Classify(m) => {
                let (labels, confs) = m.predict(x)?;
                Ok(PredictOutput::Classify { labels, confs })
            }
            TaskModel::Detect(m) => Ok(PredictOutput::Detect {
                per_image: m.predict(x, conf, iou)?,
            }),
            TaskModel::Seg(m) => Ok(PredictOutput::Seg {
                per_image: m.predict(x, conf, iou)?,
            }),
            TaskModel::Keypoint(m) => Ok(PredictOutput::Keypoint {
                per_image: m.predict(x, conf, iou)?,
            }),
        }
    }

    /// BN train/eval 装配开关（引擎训练循环每 epoch 开始 / 评测前后调用）：
    /// resnet18 骨干切 BatchNorm train 语义（train = true 用批统计并更新
    /// running 统计量；false = FrozenBN 推理语义），其余模型 no-op（无 train
    /// 态 BN）。`[pretrain].freeze_backbone = true` 时引擎恒传 false（冻结
    /// 骨干 = Detectron/mmdet FrozenBN 微调配方，统计量不随目标域重估）。
    pub fn set_train(&self, train: bool) {
        match self {
            TaskModel::Classify(m) => m.backbone.set_train(train),
            TaskModel::Detect(m) => m.backbone.set_train(train),
            TaskModel::Seg(m) => m.backbone.set_train(train),
            // 关键点走 SimpleCnnBackbone（无 train 态 BN），no-op
            TaskModel::Keypoint(_) => {}
        }
    }
}

#[cfg(all(test, feature = "torch"))]
mod tests {
    use super::*;

    fn detect_cfg_toml(assigner: &str) -> av_core::config::RunConfig {
        let toml = format!(
            concat!(
                "[model]
backbone = {{ family = \"simple-cnn\", depth = 1.0, width = 1.0, pretrained = \"none\" }}
",
                "[data.sources.train]\ndir = \"d\"\n",
                "[[model.tasks]]\nkind = \"detect\"\nnum_classes = 2\nimg_size = 64\n",
                "assigner = \"{a}\"\n"
            ),
            a = assigner
        );
        av_core::config::RunConfig::from_toml_str(&toml).expect("smoke 配置必须合法")
    }

    fn sample_batch() -> TrainBatch {
        TrainBatch::Detect {
            boxes: vec![vec![[10.0, 12.0, 30.0, 34.0]], vec![[20.0, 20.0, 44.0, 41.0]]],
            labels: vec![vec![0], vec![1]],
        }
    }

    /// seg 批：img_size 96 → 24×24 掩码画布，单实例方块掩码。
    fn sample_seg_batch() -> TrainBatch {
        let mh = 24usize;
        let mut m = vec![0u8; mh * mh];
        for y in 4..12 {
            for x in 4..12 {
                m[y * mh + x] = 1;
            }
        }
        TrainBatch::Seg {
            masks: vec![vec![m]],
            labels: vec![vec![0]],
        }
    }

    #[test]
    fn dfl_project_expectation_matches_onehot() {
        // 每边近似独热 bin 8（logit 20 压倒其余 0）→ 期望 ≈ 8 → 平移后 ≈ 0.5
        let mut v = vec![0f32; (4 * REG_MAX) as usize];
        for k in 0..4usize {
            v[k * REG_MAX as usize + 8] = 20.0;
        }
        let dist = Tensor::from_slice(&v).reshape([1i64, 4 * REG_MAX, 1, 1]);
        let proj = dfl_project(&dist);
        assert!(
            (proj.double_value(&[0, 0, 0, 0]) - (8.0 - DFL_SHIFT as f64)).abs() < 1e-3,
            "got {}",
            proj.double_value(&[0, 0, 0, 0])
        );
    }

    #[test]
    fn decode_pred_xyxy_zero_offset_gives_cell_center_box() {
        let box_raw = Tensor::zeros([1i64, 4, 2, 3], (Kind::Float, Device::Cpu));
        let xy = decode_pred_xyxy(&box_raw, 8.0);
        // cell (hi=0,wi=0)：中心 (4,4)，w=h=8 → x1=0, x2=8
        assert!((xy.double_value(&[0, 0, 0, 0]) - 0.0).abs() < 1e-4);
        assert!((xy.double_value(&[0, 2, 0, 0]) - 8.0).abs() < 1e-4);
        // cell (hi=1,wi=2)：中心 (20,12) → x1=16, y2=16
        assert!((xy.double_value(&[0, 0, 1, 2]) - 16.0).abs() < 1e-4);
        assert!((xy.double_value(&[0, 3, 1, 2]) - 16.0).abs() < 1e-4);
    }

    #[test]
    fn ciou_matches_hand_computed() {
        // 情形 1：完全重合 → 1−CIoU = 0
        let pred = Tensor::from_slice(&[0.0f32, 0.0, 10.0, 10.0]).reshape([1i64, 4, 1, 1]);
        let gt = Tensor::from_slice(&[0.0f32, 0.0, 10.0, 10.0]).reshape([1i64, 4, 1, 1]);
        let l = ciou_element(&pred, &gt);
        assert!((l.double_value(&[0, 0])).abs() < 1e-4, "got {}", l.double_value(&[0, 0]));

        // 情形 2：pred [0,0,10,10] vs gt [5,5,15,15]
        // IoU = 25/175 = 0.142857；ρ² = 50，c² = 15²+15² = 450 → 50/450 = 0.111111
        // 两框均方形 → v = 0 → loss = 1 − 0.142857 + 0.111111 = 0.968254
        let pred = Tensor::from_slice(&[0.0f32, 0.0, 10.0, 10.0]).reshape([1i64, 4, 1, 1]);
        let gt = Tensor::from_slice(&[5.0f32, 5.0, 15.0, 15.0]).reshape([1i64, 4, 1, 1]);
        let l = ciou_element(&pred, &gt);
        assert!(
            (l.double_value(&[0, 0]) - 0.968254).abs() < 1e-3,
            "got {}",
            l.double_value(&[0, 0])
        );

        // 情形 3：pred [0,0,20,10] vs gt [0,0,10,10]
        // IoU = 100/200 = 0.5；ρ² = 25，c² = 20²+10² = 500 → 0.05
        // v = (4/π²)(atan2 − atan1)² = 0.0419576，α = v/(1−0.5+v) = 0.0774518
        // loss = 1 − 0.5 + 0.05 + 0.0774518×0.0419576 = 0.553250
        let pred = Tensor::from_slice(&[0.0f32, 0.0, 20.0, 10.0]).reshape([1i64, 4, 1, 1]);
        let gt = Tensor::from_slice(&[0.0f32, 0.0, 10.0, 10.0]).reshape([1i64, 4, 1, 1]);
        let l = ciou_element(&pred, &gt);
        assert!(
            (l.double_value(&[0, 0]) - 0.553250).abs() < 1e-3,
            "got {}",
            l.double_value(&[0, 0])
        );
    }

    #[test]
    fn dfl_loss_left_right_cross_entropy_hand_computed() {
        // 分布：每边独热 bin 8（logit 10，其余 0）；目标 t = 7.75
        // tl=7, tr=8；wl=0.25, wr=0.75；Z = e^10 + 15
        // loss(每边) = 0.25·lnZ + 0.75·(lnZ − 10) = lnZ − 7.5；四边求和 ×4
        let mut v = vec![0f32; (4 * REG_MAX) as usize];
        for k in 0..4usize {
            v[k * REG_MAX as usize + 8] = 10.0;
        }
        let dist = Tensor::from_slice(&v).reshape([1i64, 4 * REG_MAX, 1, 1]);
        let target = Tensor::from_slice(&[7.75f32; 4]).reshape([1i64, 4, 1, 1]);
        let pos_w = Tensor::ones([1i64, 1, 1, 1], (Kind::Float, Device::Cpu));
        let loss = dfl_element(&dist, &target, &pos_w);
        let z = 10f64.exp() + 15.0;
        let expected = 4.0 * (z.ln() - 7.5);
        assert!(
            (loss.double_value(&[]) - expected).abs() < 1e-3,
            "got {} expected {expected}",
            loss.double_value(&[])
        );
    }

    #[test]
    fn tal_loss_smoke_forward_backward() {
        let cfg = detect_cfg_toml("tal");
        let vs = tch::nn::VarStore::new(tch::Device::Cpu);
        let model = build_model(&vs.root(), &cfg).expect("装配应成功");
        let x = Tensor::randn([2, 3, 64, 64], (Kind::Float, Device::Cpu));
        let loss = model.loss(&x, &sample_batch()).expect("TAL 损失应成功");
        assert!(loss.double_value(&[]).is_finite());
        loss.backward(); // 反传通路不应 panic
    }

    #[test]
    fn legacy_l1_loss_smoke_forward_backward() {
        let cfg = detect_cfg_toml("center");
        let vs = tch::nn::VarStore::new(tch::Device::Cpu);
        let model = build_model(&vs.root(), &cfg).expect("装配应成功");
        let x = Tensor::randn([2, 3, 64, 64], (Kind::Float, Device::Cpu));
        let loss = model.loss(&x, &sample_batch()).expect("L1 旧路径损失应成功");
        assert!(loss.double_value(&[]).is_finite());
        loss.backward();
    }

    // -----------------------------------------------------------------
    // P2 层级开关（detect.head_levels，小缺陷场景 [4,8,16]）
    // -----------------------------------------------------------------

    fn detect_cfg_toml_levels(assigner: &str, head_levels: &[u32]) -> av_core::config::RunConfig {
        let levels = head_levels
            .iter()
            .map(|l| l.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let toml = format!(
            concat!(
                "[model]
backbone = {{ family = \"simple-cnn\", depth = 1.0, width = 1.0, pretrained = \"none\" }}
",
                "[data.sources.train]\ndir = \"d\"\n",
                "[[model.tasks]]\nkind = \"detect\"\nnum_classes = 2\nimg_size = 64\n",
                "assigner = \"{a}\"\nhead_levels = [{levels}]\n"
            ),
            a = assigner,
            levels = levels
        );
        av_core::config::RunConfig::from_toml_str(&toml).expect("P2 配置必须合法")
    }

    #[test]
    fn detect_head_levels_default_is_two_tier() {
        // 缺省 head_levels = [8,16]：既有配置/checkpoint 行为零变化
        let cfg = detect_cfg_toml("tal");
        let Some(av_core::config::TaskCfg::Detect(d)) = cfg.model.tasks.first() else {
            panic!("应为 detect 任务");
        };
        assert_eq!(d.head_levels, vec![8, 16]);
    }

    #[test]
    fn p2_feature_pyramid_and_head_shape_contract() {
        let cfg = detect_cfg_toml_levels("tal", &[4, 8, 16]);
        let vs = tch::nn::VarStore::new(tch::Device::Cpu);
        let model = build_model(&vs.root(), &cfg).expect("装配应成功");
        let TaskModel::Detect(m) = &model else {
            panic!("应为检测模型");
        };
        assert_eq!(m.head.strides, vec![4, 8, 16], "头层级应为 P2/P3/P4");

        // 骨干金字塔契约：forward 后 3 层、stride 升序 4/8/16
        let x = Tensor::randn([2, 3, 64, 64], (Kind::Float, Device::Cpu));
        let py = m.backbone.forward_features(&x).expect("骨干前向应成功");
        assert_eq!(py.levels.len(), 3);
        let strides: Vec<u32> = py.levels.iter().map(|l| l.stride).collect();
        assert_eq!(strides, vec![4, 8, 16]);

        // P2 层 in_channels = backbone.stride_channels(4)（变量名 head.s4.*）
        let vars = vs.variables();
        let s4_key = vars
            .keys()
            .find(|k| k.contains("head.s4.cls1.weight"))
            .expect("P2 头变量 head.s4.cls1.weight 应存在")
            .clone();
        assert_eq!(
            vars[&s4_key].size()[1],
            m.backbone.stride_channels(4).expect("stride 4 应存在"),
            "P2 层输入通道须与骨干 stride 4 特征一致"
        );

        // 前向契约：3 层 raw_preds，每层网格 = img/stride
        let raw = m.raw_preds(&x).expect("检测前向应成功");
        assert_eq!(raw.len(), 3);
        for (li, (s, cls, box_raw)) in raw.iter().enumerate() {
            let stride = *s as i64;
            assert_eq!(stride as u32, [4u32, 8, 16][li]);
            let grid = 64 / stride;
            assert_eq!(
                cls.size(),
                vec![2, 2, grid, grid],
                "s{stride} cls 形状应为 [N,C,img/s,img/s]"
            );
            assert_eq!(
                box_raw.size(),
                vec![2, 4, grid, grid],
                "s{stride} box 形状应为 [N,4,img/s,img/s]"
            );
        }
        // 解码遍历全部配置层（conf=0 时 P2 层也应产出候选）
        let per_image = m.predict(&x, 0.0, 0.5).expect("P2 推理应成功");
        assert_eq!(per_image.len(), 2);
    }

    #[test]
    fn default_head_variable_names_unchanged() {
        // 默认 [8,16] 的变量名保持 head.s8.*/head.s16.*（历史 checkpoint 兼容）
        let cfg = detect_cfg_toml("tal");
        let vs = tch::nn::VarStore::new(tch::Device::Cpu);
        build_model(&vs.root(), &cfg).expect("装配应成功");
        let names: Vec<String> = vs.variables().keys().cloned().collect();
        assert!(names.iter().any(|n| n.contains("head.s8.cls1.weight")));
        assert!(names.iter().any(|n| n.contains("head.s16.cls1.weight")));
        assert!(!names.iter().any(|n| n.contains("head.s4")), "默认无 P2 层");
    }

    #[test]
    fn level_assignment_boundaries_two_tier_matches_legacy() {
        // [8,16] img=64：breaks=[16,32]，side ≤ 16 → s8，其余 → s16
        //（与历史 s_break = img_size×0.25 两档完全一致）
        let levels = vec![8u32, 16];
        let breaks = level_breaks(64, &levels);
        assert_eq!(breaks, vec![16.0, 32.0]);
        assert_eq!(select_level(&levels, &breaks, 0.5), 8);
        assert_eq!(select_level(&levels, &breaks, 16.0), 8, "边界值（含）归低层");
        assert_eq!(select_level(&levels, &breaks, 16.1), 16);
        assert_eq!(select_level(&levels, &breaks, 64.0), 16);
    }

    #[test]
    fn level_assignment_boundaries_three_tier_p2() {
        // [4,8,16] img=640：breaks=[80,160,320]，区间 (0,80] / (80,160] / (160,320]
        let levels = vec![4u32, 8, 16];
        let breaks = level_breaks(640, &levels);
        assert_eq!(breaks, vec![80.0, 160.0, 320.0]);
        assert_eq!(select_level(&levels, &breaks, 1.0), 4);
        assert_eq!(select_level(&levels, &breaks, 80.0), 4, "边界值（含）归 P2");
        assert_eq!(select_level(&levels, &breaks, 80.5), 8);
        assert_eq!(select_level(&levels, &breaks, 160.0), 8, "边界值（含）归中层");
        assert_eq!(select_level(&levels, &breaks, 160.5), 16);
        assert_eq!(select_level(&levels, &breaks, 320.0), 16);
        assert_eq!(select_level(&levels, &breaks, 640.0), 16, "超出回落最大 stride 层");
    }

    #[test]
    fn head_levels_rejects_empty_unsorted_or_duplicated() {
        let vs = tch::nn::VarStore::new(tch::Device::Cpu);
        for (levels, expect_msg) in [
            (&[0u32; 0][..], "不能为空"),
            (&[8u32, 8][..], "严格升序"),
            (&[16u32, 8][..], "严格升序"),
        ] {
            let cfg = detect_cfg_toml_levels("tal", levels);
            let err = match build_model(&vs.root(), &cfg) {
                Err(e) => e,
                Ok(_) => panic!("head_levels = {levels:?} 应被拒绝"),
            };
            assert!(
                err.to_string().contains(expect_msg),
                "head_levels = {levels:?} 应报「{expect_msg}」，got: {err}"
            );
        }
        // 骨干不支持的 stride 在装配期报错（stride_channels 校验）
        let cfg = detect_cfg_toml_levels("tal", &[4, 32]);
        let err = match build_model(&vs.root(), &cfg) {
            Err(e) => e,
            Ok(_) => panic!("不支持的 stride 应被拒绝"),
        };
        assert!(
            err.to_string().contains("stride 32"),
            "不支持的 stride 应报骨干形状错，got: {err}"
        );
    }

    #[test]
    fn p2_tal_loss_smoke_forward_backward() {
        let cfg = detect_cfg_toml_levels("tal", &[4, 8, 16]);
        let vs = tch::nn::VarStore::new(tch::Device::Cpu);
        let model = build_model(&vs.root(), &cfg).expect("装配应成功");
        let x = Tensor::randn([2, 3, 64, 64], (Kind::Float, Device::Cpu));
        let loss = model.loss(&x, &sample_batch()).expect("P2 TAL 损失应成功");
        assert!(loss.double_value(&[]).is_finite());
        loss.backward(); // 三层 TAL（BCE+CIoU+DFL）反传不应 panic
    }

    #[test]
    fn p2_center_l1_loss_smoke_forward_backward() {
        // 旧路径的尺寸切分分配在三层下可跑（P2 小目标层参与监督）
        let cfg = detect_cfg_toml_levels("center", &[4, 8, 16]);
        let vs = tch::nn::VarStore::new(tch::Device::Cpu);
        let model = build_model(&vs.root(), &cfg).expect("装配应成功");
        let x = Tensor::randn([2, 3, 64, 64], (Kind::Float, Device::Cpu));
        let loss = model.loss(&x, &sample_batch()).expect("P2 L1 损失应成功");
        assert!(loss.double_value(&[]).is_finite());
        loss.backward();
    }

    // -----------------------------------------------------------------
    // 检测骨干接入 resnet18（DetectBackbone：金字塔级骨干 → DetectHead）
    // -----------------------------------------------------------------

    #[test]
    fn resnet18_detect_backbone_wiring() {
        let toml = concat!(
            "[model]\n",
            "backbone = { family = \"resnet18\", depth = 1.0, width = 1.0, pretrained = \"none\" }\n",
            "neck = { type = \"identity\", channels = [] }\n",
            "[data.sources.train]\ndir = \"d\"\n",
            "[[model.tasks]]\nkind = \"detect\"\nhead = \"yolo\"\n",
            "num_classes = 2\nimg_size = 96\nassigner = \"tal\"\n"
        );
        let cfg = av_core::config::RunConfig::from_toml_str(toml).expect("resnet18 检测配置应合法");
        let vs = tch::nn::VarStore::new(tch::Device::Cpu);
        let model = build_model(&vs.root(), &cfg).expect("resnet18 检测装配应成功");
        let TaskModel::Detect(m) = &model else {
            panic!("应为检测模型");
        };
        // stride → 通道映射（DetectHead 每层输入通道装配依据）
        assert_eq!(m.backbone.stride_channels(4).unwrap(), 64, "layer1");
        assert_eq!(m.backbone.stride_channels(8).unwrap(), 128, "layer2");
        assert_eq!(m.backbone.stride_channels(16).unwrap(), 256, "layer3");
        // 装配的是 resnet18（torchvision 同名层）而非 simple-cnn（backbone.c1.*）
        let vars = vs.variables();
        assert!(vars.contains_key("backbone.layer1.0.conv1.weight"));
        assert!(vars.contains_key("backbone.layer4.1.bn2.bias"));
        assert!(
            !vars.keys().any(|n| n.starts_with("backbone.c1")),
            "检测 resnet18 不应出现 simple-cnn 变量名"
        );
        // 头通道装配：s8 层头输入通道 = 128（DetectHead conv 输入维）
        let s8 = vars
            .iter()
            .find(|(n, _)| n.contains("head.s8.cls1.weight"))
            .expect("s8 头应存在");
        assert_eq!(s8.1.size()[1], 128, "s8 头输入通道应等于骨干 stride 8 通道");
        // 前向 + 反传全链路（默认 [8,16] 两级头）
        let x = Tensor::randn([2, 3, 96, 96], (Kind::Float, Device::Cpu));
        let loss = model.loss(&x, &sample_batch()).expect("resnet18 检测损失应成功");
        assert!(loss.double_value(&[]).is_finite());
        loss.backward();
    }

    /// seg 骨干 family 分发回归锁：resnet18 必须装配 torchvision 同名层
    /// （此前 seg 硬编码 simple-cnn 无视 family，导致预训练静默空载）。
    #[test]
    fn resnet18_seg_backbone_wiring() {
        let toml = concat!(
            "[model]
",
            "backbone = { family = \"resnet18\", depth = 1.0, width = 1.0, pretrained = \"none\" }
",
            "[[model.tasks]]
kind = \"seg\"
head = \"yolact\"
",
            "num_classes = 2
img_size = 96
",
            "[data.sources.train]
dir = \"d\"
"
        );
        let cfg = av_core::config::RunConfig::from_toml_str(toml).expect("resnet18 seg 配置应合法");
        let vs = tch::nn::VarStore::new(tch::Device::Cpu);
        let model = build_model(&vs.root(), &cfg).expect("resnet18 seg 装配应成功");
        let TaskModel::Seg(m) = &model else {
            panic!("应为分割模型");
        };
        assert_eq!(m.backbone.stride_channels(16).unwrap(), 256, "layer3");
        let vars = vs.variables();
        assert!(vars.contains_key("backbone.layer1.0.conv1.weight"));
        assert!(
            !vars.keys().any(|n| n.starts_with("backbone.c1")),
            "seg resnet18 不应出现 simple-cnn 变量名"
        );
        // 前向 + 反传全链路（YOLACT 分支，输入须为 32 倍数）
        let x = Tensor::randn([2, 3, 96, 96], (Kind::Float, Device::Cpu));
        let batch = sample_seg_batch();
        let loss = model.loss(&x, &batch).expect("resnet18 seg 损失应成功");
        assert!(loss.double_value(&[]).is_finite());
        loss.backward();
    }

    // -----------------------------------------------------------------
    // OBB（detect.obb_mode = true）
    // -----------------------------------------------------------------

    fn obb_cfg_toml() -> av_core::config::RunConfig {
        let toml = concat!(
                "[model]
backbone = { family = \"simple-cnn\", depth = 1.0, width = 1.0, pretrained = \"none\" }
",
            "[data.sources.train]\ndir = \"d\"\n",
            "[[model.tasks]]\nkind = \"detect\"\nnum_classes = 2\nimg_size = 64\n",
            "assigner = \"tal\"\nobb_mode = true\n"
        );
        av_core::config::RunConfig::from_toml_str(toml).expect("obb smoke 配置必须合法")
    }

    fn obb_sample_batch() -> TrainBatch {
        // [cx, cy, w, h, θ]（像素 / 弧度）
        TrainBatch::Obb {
            boxes: vec![
                vec![[20.0, 20.0, 16.0, 8.0, 0.5]],
                vec![[32.0, 30.0, 12.0, 12.0, -0.6]],
            ],
            labels: vec![vec![0], vec![1]],
        }
    }

    #[test]
    fn obb_loss_smoke_forward_backward() {
        let cfg = obb_cfg_toml();
        let vs = tch::nn::VarStore::new(tch::Device::Cpu);
        let model = build_model(&vs.root(), &cfg).expect("OBB 模型装配应成功");
        let x = Tensor::randn([2, 3, 64, 64], (Kind::Float, Device::Cpu));
        let loss = model.loss(&x, &obb_sample_batch()).expect("OBB 损失应成功");
        assert!(loss.double_value(&[]).is_finite());
        loss.backward(); // BCE + KFIoU + DFL 全链路反传不应 panic
    }

    #[test]
    fn obb_p2_loss_smoke_forward_backward() {
        // OBB 角度分支 + P2 三层组合：forward_levels_obb 的层级循环参数化后可跑
        let toml = concat!(
                "[model]
backbone = { family = \"simple-cnn\", depth = 1.0, width = 1.0, pretrained = \"none\" }
",
            "[data.sources.train]\ndir = \"d\"\n",
            "[[model.tasks]]\nkind = \"detect\"\nnum_classes = 2\nimg_size = 64\n",
            "assigner = \"tal\"\nobb_mode = true\nhead_levels = [4, 8, 16]\n"
        );
        let cfg = av_core::config::RunConfig::from_toml_str(toml).expect("配置必须合法");
        let vs = tch::nn::VarStore::new(tch::Device::Cpu);
        let model = build_model(&vs.root(), &cfg).expect("OBB P2 模型装配应成功");
        let TaskModel::Detect(m) = &model else {
            panic!("应为检测模型");
        };
        assert_eq!(m.head.strides, vec![4, 8, 16]);
        let x = Tensor::randn([2, 3, 64, 64], (Kind::Float, Device::Cpu));
        let loss = model.loss(&x, &obb_sample_batch()).expect("OBB P2 损失应成功");
        assert!(loss.double_value(&[]).is_finite());
        loss.backward();
    }

    #[test]
    fn obb_batch_type_mismatch_is_rejected() {
        let cfg = obb_cfg_toml();
        let vs = tch::nn::VarStore::new(tch::Device::Cpu);
        let obb_model = build_model(&vs.root(), &cfg).expect("装配应成功");
        // OBB 模型 + 水平框批 → 明确报错（避免静默丢角度监督）
        assert!(obb_model.loss(&Tensor::randn([2, 3, 64, 64], (Kind::Float, Device::Cpu)), &sample_batch()).is_err());
        // 普通检测模型 + OBB 批 → 明确报错
        let vs2 = tch::nn::VarStore::new(tch::Device::Cpu);
        let det_model = build_model(&vs2.root(), &detect_cfg_toml("tal")).expect("装配应成功");
        assert!(
            det_model
                .loss(&Tensor::randn([2, 3, 64, 64], (Kind::Float, Device::Cpu)), &obb_sample_batch())
                .is_err()
        );
    }

    #[test]
    fn obb_predict_outputs_angle_and_rot_nms_runs() {
        let cfg = obb_cfg_toml();
        let vs = tch::nn::VarStore::new(tch::Device::Cpu);
        let model = build_model(&vs.root(), &cfg).expect("OBB 模型装配应成功");
        let x = Tensor::randn([2, 3, 64, 64], (Kind::Float, Device::Cpu));
        let TaskModel::Detect(m) = &model else {
            panic!("OBB 配置应装配出检测模型");
        };
        let per_image = m.predict(&x, 0.0, 0.5).expect("OBB 推理应成功");
        assert_eq!(per_image.len(), 2);
        assert!(per_image.iter().any(|d| !d.is_empty()), "conf=0 应给出候选");
        let (lo, hi) = av_core::conventions::AngleDomain::Le90.range();
        for dets in &per_image {
            for d in dets {
                let th = d.angle.expect("OBB 候选必须带角度");
                assert!(th >= lo - 1e-4 && th < hi, "θ={th} 越出 le90 域");
                assert!(d.bbox.w() > 0.0 && d.bbox.h() > 0.0, "外接框须合法");
            }
        }
    }

    #[test]
    fn obb_bs2_overfit_both_images_regression() {
        // 回归保护（bs>1 批索引 bug 复现器）：loss_obb 的 pos_w 曾漏写批索引 gi，
        // bs=2 时两图正样本权重塌进第 0 图平面——第 1 图回归无监督、第 0 图被
        // 跨图权重污染。修复后两图都应把各自的旋转框学到 IoU > 0.5。
        let cfg = obb_cfg_toml();
        let vs = tch::nn::VarStore::new(tch::Device::Cpu);
        let model = build_model(&vs.root(), &cfg).expect("OBB 模型装配应成功");
        let TaskModel::Detect(m) = &model else {
            panic!("OBB 配置应装配出检测模型");
        };
        let mut opt = {
            use tch::nn::OptimizerConfig;
            tch::nn::Adam::default().build(&vs, 3e-3).expect("优化器")
        };
        // 两图各 1 个 gt：位置/尺寸/角度/类别全不同（第 1 图 gt 不与第 0 图重叠）
        let boxes: Vec<Vec<[f32; 5]>> = vec![
            vec![[20.0, 20.0, 16.0, 8.0, 0.5]],
            vec![[44.0, 40.0, 10.0, 18.0, -0.9]],
        ];
        let labels = vec![vec![0u32], vec![1u32]];
        // 固定输入（两图不同的低幅噪声，训练动力学确定可复现）
        let x0 = Tensor::randn([3, 64, 64], (Kind::Float, Device::Cpu)) * 0.1;
        let x1 = Tensor::randn([3, 64, 64], (Kind::Float, Device::Cpu)) * 0.1 + 0.5;
        let x = Tensor::stack(&[x0, x1], 0);
        let batch = TrainBatch::Obb {
            boxes: boxes.clone(),
            labels: labels.clone(),
        };
        for _ in 0..250 {
            let loss = m.loss(&x, &batch).expect("OBB 损失应成功");
            opt.zero_grad();
            loss.backward();
            opt.clip_grad_norm(10.0);
            opt.step();
        }
        // 两图分别推理：各自最优检测须命中各自 gt（RotBox IoU > 0.5）
        use av_core::geometry::RotBox;
        tch::no_grad(|| {
            for (gi, gts) in boxes.iter().enumerate() {
                let xi = x.copy().slice(0, gi as i64, gi as i64 + 1, 1);
                let dets = m.predict(&xi, 0.25, 0.5).expect("OBB 推理应成功");
                let gr = RotBox {
                    cx: gts[0][0],
                    cy: gts[0][1],
                    w: gts[0][2],
                    h: gts[0][3],
                    theta: gts[0][4],
                };
                let best = dets[0]
                    .iter()
                    .map(|d| {
                        gr.iou(&RotBox {
                            cx: (d.bbox.x1 + d.bbox.x2) / 2.0,
                            cy: (d.bbox.y1 + d.bbox.y2) / 2.0,
                            w: d.bbox.x2 - d.bbox.x1,
                            h: d.bbox.y2 - d.bbox.y1,
                            theta: d.angle.unwrap_or(0.0),
                        })
                    })
                    .fold(0.0f32, f32::max);
                assert!(
                    best > 0.5,
                    "图 {gi} 过拟合后最优旋转 IoU = {best}（bs>1 批索引回归？）"
                );
            }
        });
    }

    #[test]
    fn standalone_obb_task_reports_guidance() {
        let toml = concat!(
                "[model]
backbone = { family = \"simple-cnn\", depth = 1.0, width = 1.0, pretrained = \"none\" }
",
            "[data.sources.train]\ndir = \"d\"\n",
            "[[model.tasks]]\nkind = \"obb\"\n"
        );
        let cfg = av_core::config::RunConfig::from_toml_str(toml).expect("配置应合法");
        let vs = tch::nn::VarStore::new(tch::Device::Cpu);
        let err = match build_model(&vs.root(), &cfg) {
            Err(e) => e,
            Ok(_) => panic!("独立 obb 任务应报接入指引错误"),
        };
        assert!(
            err.to_string().contains("obb_mode"),
            "错误信息应指向 detect + obb_mode，got: {err}"
        );
    }

    // -----------------------------------------------------------------
    // 实例分割（TaskCfg::Seg，head = "yolact"）
    // -----------------------------------------------------------------

    fn seg_cfg_toml(head: &str) -> av_core::config::RunConfig {
        let toml = format!(
            concat!(
                "[model]
backbone = {{ family = \"simple-cnn\", depth = 1.0, width = 1.0, pretrained = \"none\" }}
",
                "[data.sources.train]\ndir = \"d\"\n",
                "[[model.tasks]]\nkind = \"seg\"\nhead = \"{h}\"\n",
                "num_classes = 3\nimg_size = 64\nnum_protos = 8\n"
            ),
            h = head
        );
        av_core::config::RunConfig::from_toml_str(&toml).expect("seg smoke 配置必须合法")
    }

    fn seg_sample_batch() -> TrainBatch {
        // img=64 → 掩码画布 16×16；图 0 一个方块（类 0），图 1 两个方块（类 1/2）
        let fill_rect = |m: &mut Vec<u8>, (x0, y0, x1, y1): (usize, usize, usize, usize)| {
            for y in y0..y1 {
                for x in x0..x1 {
                    m[y * 16 + x] = 1;
                }
            }
        };
        let mut m0 = vec![0u8; 256];
        fill_rect(&mut m0, (3, 3, 9, 9));
        let mut m1 = vec![0u8; 256];
        fill_rect(&mut m1, (1, 1, 7, 7));
        let mut m2 = vec![0u8; 256];
        fill_rect(&mut m2, (8, 8, 14, 14));
        TrainBatch::Seg {
            masks: vec![vec![m0], vec![m1, m2]],
            labels: vec![vec![0], vec![1, 2]],
        }
    }

    #[test]
    fn seg_loss_smoke_forward_backward() {
        let cfg = seg_cfg_toml("yolact");
        let vs = tch::nn::VarStore::new(tch::Device::Cpu);
        let model = build_model(&vs.root(), &cfg).expect("Seg 模型装配应成功");
        let x = Tensor::randn([2, 3, 64, 64], (Kind::Float, Device::Cpu));
        let loss = model.loss(&x, &seg_sample_batch()).expect("Seg 损失应成功");
        assert!(loss.double_value(&[]).is_finite());
        loss.backward(); // BCE(cls) + BCE(mask) + Dice 全链路反传不应 panic
    }

    #[test]
    fn seg_loss_empty_batch_stays_connected() {
        // 批内无实例（全空 masks/labels）→ 损失仍有限且可反传（cls 项恒连通）
        let cfg = seg_cfg_toml("yolact");
        let vs = tch::nn::VarStore::new(tch::Device::Cpu);
        let model = build_model(&vs.root(), &cfg).expect("Seg 模型装配应成功");
        let x = Tensor::randn([2, 3, 64, 64], (Kind::Float, Device::Cpu));
        let batch = TrainBatch::Seg {
            masks: vec![Vec::new(), Vec::new()],
            labels: vec![Vec::new(), Vec::new()],
        };
        let loss = model.loss(&x, &batch).expect("空批损失应成功");
        assert!(loss.double_value(&[]).is_finite());
        loss.backward();
    }

    #[test]
    fn seg_predict_outputs_binary_masks_with_nms() {
        let cfg = seg_cfg_toml("yolact");
        let vs = tch::nn::VarStore::new(tch::Device::Cpu);
        let model = build_model(&vs.root(), &cfg).expect("Seg 模型装配应成功");
        let TaskModel::Seg(m) = &model else {
            panic!("seg 配置应装配出 SegModel");
        };
        assert_eq!(m.mask_size(), 16);
        let x = Tensor::randn([1, 3, 64, 64], (Kind::Float, Device::Cpu));
        // 随机权重下候选数不定（所有 cell 组合掩码全空时合法为 0），
        // 只断言结构不变式：二值掩码、分辨率 img/4、非空掩码、NMS 契约
        let out = m.predict(&x, 0.0, 0.5).expect("Seg 推理应成功");
        assert_eq!(out.len(), 1);
        for inst in &out[0] {
            assert_eq!(inst.mask.len(), 256);
            assert!(inst.mask.iter().all(|&v| v <= 1), "掩码应二值");
            assert!(inst.mask.iter().any(|&v| v == 1), "空掩码应被丢弃");
            assert!((0.0..=1.0).contains(&inst.score), "score 应在 [0,1]");
            assert!((inst.label as i64) < 3, "label 应在类数内");
        }
        // NMS 契约：保留集内同类别掩码 IoU 两两 < 阈值（近重复被抑制）
        let kept = &out[0];
        for (a, b) in kept.iter().enumerate().flat_map(|(i, a)| {
            kept[i + 1..].iter().map(move |b| (a, b))
        }) {
            if a.label == b.label {
                assert!(
                    mask_iou(&a.mask, &b.mask) < 0.5,
                    "NMS 后同类掩码 IoU 应 < 0.5"
                );
            }
        }
    }

    #[test]
    fn mask_nms_suppresses_same_class_near_duplicates() {
        // 手工实例（确定性）：类 5 的两个高重叠掩码（IoU=2/4）+ 类 5 一个远离的
        // （IoU=0）+ 类 6 一个与最高分重叠的（跨类不抑制）
        let inst = |label: u32, score: f32, m: Vec<u8>| SegInstance {
            label,
            score,
            mask: m,
        };
        let a = inst(5, 0.9, vec![1, 1, 1, 1]); // 与 b 同类 IoU=2/4 ≥ 0.5 → b 被抑制
        let b = inst(5, 0.8, vec![1, 1, 0, 0]);
        let c = inst(5, 0.7, vec![0, 0, 0, 1]); // 与 a 同类 IoU=1/5 < 0.5 → 保留
        let d = inst(6, 0.6, vec![1, 1, 1, 1]); // 与 a 异类 → 不抑制
        let kept = mask_nms(vec![b, c, d, a], 0.5);
        assert_eq!(kept.len(), 3, "b 应被 a 抑制，c/d 保留");
        assert_eq!(kept[0].score, 0.9);
        assert!(kept.iter().any(|k| k.label == 6), "异类不应被抑制");
        assert!(!kept.iter().any(|k| k.score == 0.8), "近重复同类应被抑制");
        // 空输入与单实例
        assert!(mask_nms(Vec::new(), 0.5).is_empty());
        let solo = inst(5, 0.9, vec![1, 1, 1, 1]);
        assert_eq!(mask_nms(vec![solo], 0.5).len(), 1);
    }

    #[test]
    fn seg_batch_type_mismatch_is_rejected() {
        let cfg = seg_cfg_toml("yolact");
        let vs = tch::nn::VarStore::new(tch::Device::Cpu);
        let seg_model = build_model(&vs.root(), &cfg).expect("装配应成功");
        // Seg 模型 + 检测批 → 明确报错
        assert!(
            seg_model
                .loss(&Tensor::randn([2, 3, 64, 64], (Kind::Float, Device::Cpu)), &sample_batch())
                .is_err()
        );
        // 检测模型 + Seg 批 → 明确报错
        let vs2 = tch::nn::VarStore::new(tch::Device::Cpu);
        let det_model = build_model(&vs2.root(), &detect_cfg_toml("tal")).expect("装配应成功");
        assert!(
            det_model
                .loss(&Tensor::randn([2, 3, 64, 64], (Kind::Float, Device::Cpu)), &seg_sample_batch())
                .is_err()
        );
    }

    #[test]
    fn seg_direct_head_reports_not_supported() {
        let cfg = seg_cfg_toml("direct");
        let vs = tch::nn::VarStore::new(tch::Device::Cpu);
        let err = match build_model(&vs.root(), &cfg) {
            Err(e) => e,
            Ok(_) => panic!("direct 精度档应报未支持"),
        };
        assert!(err.to_string().contains("direct"), "got: {err}");
    }

    #[test]
    fn seg_overfit_tiny_batch_reduces_loss() {
        use tch::nn::OptimizerConfig;
        // 过拟合冒烟：同一小批反复训练，损失应显著下降且 predict 产出掩码候选。
        // 输入用与 gt 掩码对齐（掩码坐标 ×4 = 像素域）的通道化亮块而非纯噪声：
        // 纯噪声下所有 cell 特征相同，优化易陷入「全空掩码」对称盆地
        // （偶发、与真实数据无关——真实图像特征各异，见 coco8-seg 实测）。
        let cfg = seg_cfg_toml("yolact");
        let vs = tch::nn::VarStore::new(tch::Device::Cpu);
        let model = build_model(&vs.root(), &cfg).expect("装配应成功");
        let s = 64usize;
        let mut buf = vec![0f32; 2 * 3 * s * s];
        let mut paint = |img: usize, ch: usize, (x0, y0, x1, y1): (usize, usize, usize, usize)| {
            for y in y0..y1 {
                for x in x0..x1 {
                    buf[((img * 3 + ch) * s + y) * s + x] += 1.5;
                }
            }
        };
        // 与 seg_sample_batch 的掩码逐像素对齐（掩码 16×16 × 4 = 像素 64×64）
        paint(0, 0, (12, 12, 36, 36)); // m0 rect (3,3)-(9,9)
        paint(1, 1, (4, 4, 28, 28)); // m1 rect (1,1)-(7,7)
        paint(1, 2, (32, 32, 56, 56)); // m2 rect (8,8)-(14,14)
        let x = Tensor::from_slice(&buf)
            .to_kind(Kind::Float)
            .reshape([2i64, 3, s as i64, s as i64]);
        let batch = seg_sample_batch();
        let mut opt = tch::nn::Adam::default().build(&vs, 1e-2).expect("优化器应构建");
        let first = model.loss(&x, &batch).expect("损失应成功").double_value(&[]);
        // 自适应预算：每 40 步检查一次 predict 是否已产出掩码候选（上限 400 步，
        // 与 coco8-seg 实测训练同量级）。不绑损失比值——Adam 对坏 init 的早期
        // 收敛速度随种子波动，数值阈值必然偶发误报。
        let TaskModel::Seg(sm) = &model else {
            panic!("seg 配置应装配出 SegModel");
        };
        let mut last = first;
        let mut candidates_ok = false;
        for step in 0..400 {
            opt.zero_grad();
            let loss = model.loss(&x, &batch).expect("损失应成功");
            loss.backward();
            opt.step();
            last = loss.double_value(&[]);
            if (step + 1) % 40 == 0 {
                let preds = sm.predict(&x, 0.0, 0.5).expect("过拟合推理应成功");
                candidates_ok = preds.iter().all(|dets| !dets.is_empty());
                if candidates_ok {
                    break;
                }
            }
        }
        assert!(last.is_finite(), "损失必须有限");
        assert!(
            candidates_ok && last < first,
            "过拟合后 predict 应每图产出掩码候选且损失下降：first={first} last={last}"
        );
    }

    // -----------------------------------------------------------------
    // 关键点（TaskCfg::Keypoint，decode = "direct" 直接回归档）
    // -----------------------------------------------------------------

    fn kp_cfg_toml() -> av_core::config::RunConfig {
        let toml = concat!(
                "[model]
backbone = { family = \"simple-cnn\", depth = 1.0, width = 1.0, pretrained = \"none\" }
",
            "[data.sources.train]\ndir = \"d\"\n",
            "[[model.tasks]]\nkind = \"keypoint\"\ndecode = \"direct\"\n",
            "num_keypoints = 3\nimg_size = 64\n"
        );
        av_core::config::RunConfig::from_toml_str(toml).expect("kp smoke 配置必须合法")
    }

    fn kp_sample_batch() -> TrainBatch {
        // img=64 → 网格 8×8（stride 8）；框 cxcywh 画布像素，kpts [x,y,v] 画布像素
        TrainBatch::Keypoint {
            boxes: vec![
                vec![[32.0, 32.0, 16.0, 16.0]],
                vec![[20.0, 40.0, 12.0, 20.0], [44.0, 16.0, 12.0, 12.0]],
            ],
            kpts: vec![
                vec![vec![[30.0, 28.0, 2.0], [34.0, 28.0, 2.0], [32.0, 36.0, 1.0]]],
                vec![
                    vec![[18.0, 32.0, 2.0], [22.0, 32.0, 0.0], [20.0, 44.0, 2.0]],
                    vec![[42.0, 12.0, 2.0], [46.0, 12.0, 2.0], [44.0, 20.0, 2.0]],
                ],
            ],
            labels: vec![vec![0], vec![0, 0]],
        }
    }

    #[test]
    fn kp_loss_smoke_forward_backward() {
        let cfg = kp_cfg_toml();
        let vs = tch::nn::VarStore::new(tch::Device::Cpu);
        let model = build_model(&vs.root(), &cfg).expect("Keypoint 模型装配应成功");
        let x = Tensor::randn([2, 3, 64, 64], (Kind::Float, Device::Cpu));
        let loss = model.loss(&x, &kp_sample_batch()).expect("关键点损失应成功");
        assert!(loss.double_value(&[]).is_finite());
        loss.backward(); // BCE(cls) + L1(box/off) + BCE(vis) + OKS 全链路反传不应 panic
    }

    #[test]
    fn kp_empty_batch_stays_connected() {
        // 批内无实例 → cls 项恒连通，损失有限可反传（同分割路径兜底）
        let cfg = kp_cfg_toml();
        let vs = tch::nn::VarStore::new(tch::Device::Cpu);
        let model = build_model(&vs.root(), &cfg).expect("装配应成功");
        let x = Tensor::randn([2, 3, 64, 64], (Kind::Float, Device::Cpu));
        let batch = TrainBatch::Keypoint {
            boxes: vec![Vec::new(), Vec::new()],
            kpts: vec![Vec::new(), Vec::new()],
            labels: vec![Vec::new(), Vec::new()],
        };
        let loss = model.loss(&x, &batch).expect("空批损失应成功");
        assert!(loss.double_value(&[]).is_finite());
        loss.backward();
    }

    #[test]
    fn kp_point_count_mismatch_is_rejected() {
        let cfg = kp_cfg_toml();
        let vs = tch::nn::VarStore::new(tch::Device::Cpu);
        let model = build_model(&vs.root(), &cfg).expect("装配应成功");
        let batch = TrainBatch::Keypoint {
            boxes: vec![vec![[32.0, 32.0, 16.0, 16.0]]],
            kpts: vec![vec![vec![[30.0, 28.0, 2.0]]]], // 1 点 ≠ num_keypoints = 3
            labels: vec![vec![0]],
        };
        let err = model
            .loss(
                &Tensor::randn([1, 3, 64, 64], (Kind::Float, Device::Cpu)),
                &batch,
            )
            .unwrap_err();
        assert!(err.to_string().contains("num_keypoints"), "got: {err}");
    }

    #[test]
    fn kp_batch_type_mismatch_is_rejected() {
        let cfg = kp_cfg_toml();
        let vs = tch::nn::VarStore::new(tch::Device::Cpu);
        let kp_model = build_model(&vs.root(), &cfg).expect("装配应成功");
        // 关键点模型 + 检测批 → 明确报错
        assert!(
            kp_model
                .loss(&Tensor::randn([2, 3, 64, 64], (Kind::Float, Device::Cpu)), &sample_batch())
                .is_err()
        );
        // 检测模型 + 关键点批 → 明确报错
        let vs2 = tch::nn::VarStore::new(tch::Device::Cpu);
        let det_model = build_model(&vs2.root(), &detect_cfg_toml("tal")).expect("装配应成功");
        assert!(
            det_model
                .loss(&Tensor::randn([2, 3, 64, 64], (Kind::Float, Device::Cpu)), &kp_sample_batch())
                .is_err()
        );
    }

    #[test]
    fn kp_predict_outputs_detections_with_keypoints() {
        let cfg = kp_cfg_toml();
        let vs = tch::nn::VarStore::new(tch::Device::Cpu);
        let model = build_model(&vs.root(), &cfg).expect("装配应成功");
        let TaskModel::Keypoint(m) = &model else {
            panic!("keypoint 配置应装配出 KeypointModel");
        };
        assert_eq!(m.kp_stride(), 8);
        let x = Tensor::randn([1, 3, 64, 64], (Kind::Float, Device::Cpu));
        // 随机权重下候选数不定，只断言结构不变式：每实例带 K 个画布内关键点
        let per_image = m.predict(&x, 0.0, 0.5).expect("关键点推理应成功");
        assert_eq!(per_image.len(), 1);
        for d in &per_image[0] {
            let kps = d.keypoints.as_ref().expect("实例必须带关键点");
            assert_eq!(kps.len(), 3);
            for kp in kps {
                assert!(
                    (0.0..=64.0).contains(&kp[0]) && (0.0..=64.0).contains(&kp[1]),
                    "关键点应 clamp 在画布内: {kp:?}"
                );
                assert!(kp[2] == 0.0 || kp[2] == 2.0, "v 应二值化为 0/2，got {}", kp[2]);
            }
            assert!(d.bbox.x2 >= d.bbox.x1 && d.bbox.y2 >= d.bbox.y1, "框应合法");
            assert!((0.0..=1.0).contains(&d.score), "score 应在 [0,1]");
        }
        // NMS 契约：保留集框两两 IoU ≤ 阈值
        let kept = &per_image[0];
        for (a, b) in kept.iter().enumerate().flat_map(|(i, a)| {
            kept[i + 1..].iter().map(move |b| (a, b))
        }) {
            assert!(
                a.bbox.iou(&b.bbox) <= 0.5 + 1e-6,
                "NMS 后框 IoU 应 ≤ 0.5，got {}",
                a.bbox.iou(&b.bbox)
            );
        }
    }

    #[test]
    fn kp_overfit_tiny_batch_reduces_loss() {
        use tch::nn::OptimizerConfig;
        // 过拟合冒烟：与 seg 同思路，输入用与 gt 对齐的亮块（框区通道 0 +
        // 可见关键点亮点通道 1）——纯噪声下所有 cell 特征相同，回归无从学起
        let cfg = kp_cfg_toml();
        let vs = tch::nn::VarStore::new(tch::Device::Cpu);
        let model = build_model(&vs.root(), &cfg).expect("装配应成功");
        let s = 64usize;
        let batch = kp_sample_batch();
        let TrainBatch::Keypoint { boxes, kpts, .. } = &batch else {
            unreachable!()
        };
        let mut buf = vec![0f32; 2 * 3 * s * s];
        for (img, img_boxes) in boxes.iter().enumerate() {
            for (b, gk) in img_boxes.iter().zip(gk_of(&kpts[img])) {
                let (x0, y0) = ((b[0] - b[2] / 2.0) as usize, (b[1] - b[3] / 2.0) as usize);
                for y in y0..y0 + b[3] as usize {
                    for x in x0..x0 + b[2] as usize {
                        buf[((img * 3) * s + y) * s + x] += 1.5;
                    }
                }
                for kp in gk {
                    if kp[2] > 0.0 {
                        let (ky, kx) = (kp[1] as usize, kp[0] as usize);
                        buf[((img * 3 + 1) * s + ky) * s + kx] += 2.0;
                    }
                }
            }
        }
        let x = Tensor::from_slice(&buf)
            .to_kind(Kind::Float)
            .reshape([2i64, 3, s as i64, s as i64]);
        let mut opt = tch::nn::Adam::default().build(&vs, 1e-2).expect("优化器应构建");
        let first = model.loss(&x, &batch).expect("损失应成功").double_value(&[]);
        let mut last = first;
        for _ in 0..150 {
            opt.zero_grad();
            let loss = model.loss(&x, &batch).expect("损失应成功");
            loss.backward();
            opt.step();
            last = loss.double_value(&[]);
        }
        assert!(last.is_finite(), "损失必须有限");
        assert!(
            last < first * 0.8,
            "过拟合后损失应显著下降：first={first} last={last}"
        );
    }

    /// 借用切片辅助（过拟合测试里 zip 两个平行 Vec）。
    fn gk_of(kpts: &[Vec<[f32; 3]>]) -> &[Vec<[f32; 3]>] {
        kpts
    }
}
