//! 简化版 YOLACT 分割模型：CSP-ELAN 骨干 + 原型×系数掩码链路（burn 版）。
//!
//! 语义对齐 `crates/av-tasks/src/models.rs` 的 `SegModel`（先读后写）：
//!
//! - 抽头：P4（stride 16）进分割头；
//! - 原型 [N,K,img/4,img/4]（logits，组合前过 sigmoid）+ 每网格
//!   [N,C+K,img/16,img/16]（前 C 通道类别 logits，后 K 通道系数）；
//! - 分配：实例掩码质心（掩码画布像素）所在 stride16 cell，单点分配；
//! - 监督：cls = 加权 BCE-with-logits（正 cell 权重 = 负正比上限 50，
//!   torch weighted-mean 语义 Σw·ℓ/Σw）；mask = 逐实例 BCE+Dice 求均值；
//!   批内无实例时退化为纯 cls 项（梯度不断图）。
//!
//! **简化（诚实边界）**：分类只用单点分配 + 加权 BCE（无 TAL / anchor 匹配）；
//! 系数无显式回归目标，梯度仅经掩码损失回传（YOLACT 同款）。推理路径只提供
//! [`combine_proto_coef`] 单实例组合辅助（推理解耦头/NMS 不在 spike 范围）。

use av_core::error::AvResult;
// burn 的 #[derive(Module)] 展开引用 `burn::` 路径，需要该别名（burn 自身
// 源码的同款约定）。
use burn_core as burn;
use burn_core::module::Module;
use burn_core::tensor::backend::Backend;
use burn_core::tensor::{activation::log_sigmoid, activation::sigmoid, Tensor, TensorData};

use crate::backbone::{BackboneCfg, CspElanBackbone};
use crate::head::MaskHead;

/// Dice 损失（pred 为 sigmoid 概率图、gt 为 0/1 目标，同形任意维）：
/// dice = 2|p∩g| / (|p|+|g|)；写成 (denom − 2·inter + ε)/(denom + ε)，
/// ε 防空掩码除零。返回 rank-1 标量 [1]。
pub fn dice_loss<B: Backend, const D: usize>(pred: Tensor<B, D>, gt: Tensor<B, D>) -> Tensor<B, 1> {
    let eps = 1e-5f32;
    let inter = (pred.clone() * gt.clone()).sum();
    let denom = pred.sum() + gt.sum();
    (denom.clone() - inter.mul_scalar(2.0) + eps) / (denom + eps)
}

/// BCE-with-logits（数值稳定式 (1−t)·x − log σ(x)），元素均值。返回 [1]。
pub fn bce_with_logits_mean<B: Backend, const D: usize>(
    logits: Tensor<B, D>,
    targets: Tensor<B, D>,
) -> Tensor<B, 1> {
    let per_elem = (targets.neg().add_scalar(1.0)) * logits.clone() - log_sigmoid(logits);
    per_elem.mean()
}

/// 掩码扁平质心 → stride16 cell 索引 (hi, wi)（与 tch SegModel::loss 逐式对齐：
/// 质心以掩码画布像素为单位，cell 边 = 画布/网格数，末行/列钳位）。
pub fn centroid_cell(gt_mask: &[u8], mw: usize, gh: usize, gw: usize) -> Option<(usize, usize)> {
    let mh = mw;
    if gt_mask.len() != mh * mw {
        return None;
    }
    let (mut sx, mut sy, mut area) = (0f32, 0f32, 0usize);
    for (pi, &v) in gt_mask.iter().enumerate() {
        if v != 0 {
            sx += (pi % mw) as f32;
            sy += (pi / mw) as f32;
            area += 1;
        }
    }
    if area == 0 {
        return None;
    }
    let (cx, cy) = (sx / area as f32, sy / area as f32);
    let (cell_w, cell_h) = (mw as f32 / gw as f32, mh as f32 / gh as f32);
    let wi = ((cx / cell_w) as usize).min(gw - 1);
    let hi = ((cy / cell_h) as usize).min(gh - 1);
    Some((hi, wi))
}

/// YOLACT 组合：sigmoid 后的原型 [K,mh,mw] × 系数 [K] → 实例掩码 logits
/// [mh,mw]。训练与（未来的）推理必须共用本函数，保证组合方式逐位一致。
pub fn combine_proto_coef<B: Backend>(
    proto_sigmoid: Tensor<B, 3>,
    coef: Tensor<B, 1>,
    k: usize,
    mh: usize,
    mw: usize,
) -> Tensor<B, 2> {
    (proto_sigmoid * coef.reshape([k, 1, 1]))
        .sum_dim(0)
        .reshape([mh, mw])
}

/// 分割模型装配参数。
#[derive(Debug, Clone)]
pub struct SegNetCfg {
    pub width: f32,
    pub depth: f32,
    pub num_classes: usize,
    pub num_protos: usize,
    /// 掩码 BCE 权重（tch 版 loss_w_bce）。
    pub loss_w_bce: f64,
    /// Dice 权重（tch 版 loss_w_dice）。
    pub loss_w_dice: f64,
}

/// CSP-ELAN 骨干 + YOLACT 式分割头。
#[derive(Module, Debug)]
pub struct SegNet<B: Backend> {
    backbone: CspElanBackbone<B>,
    head: MaskHead<B>,
    num_classes: usize,
    num_protos: usize,
    loss_w_bce: f32,
    loss_w_dice: f32,
}

impl<B: Backend> SegNet<B> {
    /// 装配（width/depth 传给骨干；头挂 P4 通道）。
    pub fn new(cfg: &SegNetCfg, device: &B::Device) -> AvResult<Self> {
        let backbone = CspElanBackbone::new(
            &BackboneCfg {
                width: cfg.width,
                depth: cfg.depth,
            },
            device,
        )?;
        let (_p3, p4, _p5) = backbone.pyramid_channels();
        Ok(Self {
            head: MaskHead::new(p4, cfg.num_classes, cfg.num_protos, device),
            backbone,
            num_classes: cfg.num_classes,
            num_protos: cfg.num_protos,
            loss_w_bce: cfg.loss_w_bce as f32,
            loss_w_dice: cfg.loss_w_dice as f32,
        })
    }

    /// 前向：输入 [N,3,S,S] → (原型 logits [N,K,S/4,S/4]，
    /// 系数+类别 [N,C+K,S/16,S/16])。
    pub fn forward_seg(&self, x: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let (_p3, p4, _p5) = self.backbone.forward_features(x);
        self.head.forward(p4)
    }

    /// 类别数 C。
    pub fn num_classes(&self) -> usize {
        self.num_classes
    }

    /// 原型数 K。
    pub fn num_protos(&self) -> usize {
        self.num_protos
    }

    /// 分割损失（语义对齐 tch SegModel::loss）。
    ///
    /// `masks[i][g]`：第 i 图第 g 实例的 0/1 掩码（img/4 画布扁平）；
    /// `labels[i][g]`：对应类别。
    pub fn loss(
        &self,
        x: Tensor<B, 4>,
        masks: &[Vec<Vec<u8>>],
        labels: &[Vec<u32>],
    ) -> Tensor<B, 1> {
        let device = x.device();
        let n = x.dims()[0];
        let (proto, coefcls) = self.forward_seg(x);
        let [_nc, _nc_k, gh, gw] = coefcls.dims();
        let [_np, _nk, mh, mw] = proto.dims();
        let c = self.num_classes;
        let k = self.num_protos;
        let cells = gh * gw;
        let total = n * c * cells;

        let mut cls_t = vec![0f32; total];
        let mut cls_w = vec![1f32; total];
        let mut pos_idx: Vec<usize> = Vec::new();
        // 掩码项累加器（恒连图的零张量，避免批内无实例时构建空 sum）。
        let mut mask_sum = Tensor::<B, 1>::zeros([1], &device);
        let mut mask_cnt = 0usize;
        // 原型整批一次 sigmoid（逐元素，与逐实例 sigmoid 逐位一致；同 tch 路径优化）
        let proto_sig = sigmoid(proto.clone());

        for i in 0..n.min(masks.len()).min(labels.len()) {
            for (g, &label) in labels[i].iter().enumerate() {
                let Some(gt_mask) = masks[i].get(g) else {
                    continue;
                };
                if label as usize >= c {
                    continue; // 标注类别越界（数据脏）整条跳过
                }
                let Some((hi, wi)) = centroid_cell(gt_mask, mw, gh, gw) else {
                    continue; // 掩码分辨率不符或空掩码（数据脏）跳过
                };
                let flat = hi * gw + wi;
                let cls_flat = (i * c + label as usize) * cells + flat;
                cls_t[cls_flat] = 1.0;
                pos_idx.push(cls_flat);

                // 该 cell 的系数 [K] → 与原型线性组合成实例掩码 logits [mh,mw]。
                // 原型先过 sigmoid（YOLACT「基掩码」语义），组合方式与
                // combine_proto_coef 共用一份逻辑。
                // （burn 0.21 的 select 需要 Int 张量索引，单点抽取统一走
                //   多维 slice + reshape。）
                let coef = coefcls
                    .clone()
                    .slice([i..(i + 1), c..(c + k), hi..(hi + 1), wi..(wi + 1)])
                    .reshape([k]); // [K]
                let proto_i = proto_sig
                    .clone()
                    .slice([i..(i + 1), 0..k])
                    .reshape([k, mh, mw]);
                let logit = combine_proto_coef(proto_i, coef, k, mh, mw);
                let gt_t = Tensor::<B, 2>::from_data(
                    TensorData::new(
                        gt_mask.iter().map(|&v| v as f32).collect::<Vec<_>>(),
                        [mh, mw],
                    ),
                    &device,
                );
                let bce = bce_with_logits_mean(logit.clone(), gt_t.clone());
                let dl = dice_loss(sigmoid(logit), gt_t);
                mask_sum =
                    mask_sum + (bce.mul_scalar(self.loss_w_bce) + dl.mul_scalar(self.loss_w_dice));
                mask_cnt += 1;
            }
        }

        // cls：正 cell 加权（负正失衡上限 50，同检测路径）
        if !pos_idx.is_empty() {
            let boost =
                (((total - pos_idx.len()) as f32) / (pos_idx.len() as f32)).clamp(1.0, 50.0);
            for &idx in &pos_idx {
                cls_w[idx] = boost;
            }
        }
        // cls BCE 只作用于前 C 通道（类别分数）；后 K 通道是原型系数，
        // 不应被推向 0（系数没有显式目标，梯度只经掩码损失回传——YOLACT 同款）。
        let cls_logits = coefcls.slice([0..n, 0..c]); // [N,C,gh,gw]
        let targets = Tensor::<B, 4>::from_data(TensorData::new(cls_t, [n, c, gh, gw]), &device);
        let weights = Tensor::<B, 4>::from_data(TensorData::new(cls_w, [n, c, gh, gw]), &device);
        // torch weighted-mean 语义：Σ w·ℓ / Σ w
        let per_elem = ((targets.clone().neg().add_scalar(1.0)) * cls_logits.clone()
            - log_sigmoid(cls_logits))
            * weights.clone();
        let cls_loss = per_elem.sum() / weights.sum();

        // 掩码项：逐实例 BCE+Dice 求均值；批内无实例时退化为纯 cls 项
        if mask_cnt > 0 {
            cls_loss + mask_sum.div_scalar(mask_cnt as f32)
        } else {
            cls_loss
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::load_cocoseg_dir;
    use crate::train::{cosine_lr, make_optimizer, train_step, TrainCfg};
    use crate::NdArrayB;
    use burn_core::module::AutodiffModule;
    use burn_core::tensor::activation::sigmoid;
    use std::path::Path;

    fn device() -> burn_ndarray::NdArrayDevice {
        Default::default()
    }

    fn micro_cfg() -> SegNetCfg {
        // micro 配置（width 0.0625 → 全通道触底 8）：ndarray CPU 上冒烟可承受
        SegNetCfg {
            width: 0.0625,
            depth: 0.33,
            num_classes: 80,
            num_protos: 8,
            loss_w_bce: 1.0,
            loss_w_dice: 1.0,
        }
    }

    /// dice 手算对照（对齐 tch mask.rs 测试）：
    /// p ≡ 0.5、gt 单像素 1 @2×2 → (2+1−1+ε)/(2+1+ε) ≈ 0.666663。
    #[test]
    fn dice_loss_hand_computed_constant_prediction() {
        let device = device();
        let gt = Tensor::<NdArrayB, 2>::from_data(
            TensorData::new(vec![1.0f32, 0.0, 0.0, 0.0], [2, 2]),
            &device,
        );
        let pred = Tensor::<NdArrayB, 2>::zeros([2, 2], &device);
        let l = dice_loss(sigmoid(pred), gt).into_scalar();
        assert!((l - 2.0 / 3.0).abs() < 1e-4, "got {l}");
    }

    /// dice 恒等近零：pred 大梯度处与 gt 一致 → dice 损失 < 1e-4。
    #[test]
    fn dice_loss_identical_masks_near_zero() {
        let device = device();
        let gt = Tensor::<NdArrayB, 2>::from_data(
            TensorData::new(vec![1.0f32, 1.0, 0.0, 0.0], [2, 2]),
            &device,
        );
        let pred = Tensor::<NdArrayB, 2>::from_data(
            TensorData::new(vec![20.0f32, 20.0, -20.0, -20.0], [2, 2]),
            &device,
        );
        let l = dice_loss(sigmoid(pred), gt).into_scalar();
        assert!(l < 1e-4, "got {l}");
    }

    /// BCE-with-logits 手算对照：x=0,t=1 → −log 0.5 ≈ 0.6931；
    /// x=20,t=0 → 20 − log σ(20) ≈ 20；均值语义。
    #[test]
    fn bce_with_logits_hand_computed() {
        let device = device();
        let logits =
            Tensor::<NdArrayB, 1>::from_data(TensorData::new(vec![0.0f32, 20.0], [2]), &device);
        let targets =
            Tensor::<NdArrayB, 1>::from_data(TensorData::new(vec![1.0f32, 0.0], [2]), &device);
        let l = bce_with_logits_mean(logits, targets).into_scalar();
        // −log σ(0) = −ln 0.5 = ln 2；x=20,t=0 → 20 − ln σ(20) ≈ 20
        let expect = (0.5f64.ln().abs() + 20.0) / 2.0;
        assert!((l as f64 - expect).abs() < 1e-4, "got {l} expect {expect}");
    }

    /// 质心分配手算对照：8×4 掩码（mw=4，2×2 网格）中央 2×2 块 → 质心
    /// (col=2.5? 按像素索引均值) → cell 钳位。
    #[test]
    fn centroid_cell_hand_computed() {
        // 4×4 掩码（mw=4），块覆盖 (row1,col1..3)（第 1 行 col 1、2）
        let mut m = vec![0u8; 16];
        m[4 + 1] = 1;
        m[4 + 2] = 1;
        // 质心 col=(1+2)/2=1.5, row=1.0；gw=gh=2，cell=2.0 → (hi=0, wi=0)
        assert_eq!(centroid_cell(&m, 4, 2, 2), Some((0, 0)));
        // 右下块（3,3）单点：质心 (3.0,3.0) → cell (1,1)
        let mut m2 = vec![0u8; 16];
        m2[3 * 4 + 3] = 1;
        assert_eq!(centroid_cell(&m2, 4, 2, 2), Some((1, 1)));
        // 全零与尺寸不符 → None
        assert_eq!(centroid_cell(&[0u8; 16], 4, 2, 2), None);
        assert_eq!(centroid_cell(&[0u8; 5], 4, 2, 2), None);
    }

    /// 前向 shape 契约（对照 tch SegModel）：128 输入 → 原型 [1,K,32,32]、
    /// 系数+类别 [1,C+K,8,8]。
    #[test]
    fn segnet_forward_shapes() {
        let device = device();
        let cfg = micro_cfg();
        let model = SegNet::<NdArrayB>::new(&cfg, &device).unwrap();
        let x = Tensor::<NdArrayB, 4>::ones([1, 3, 128, 128], &device);
        let (proto, coefcls) = model.forward_seg(x);
        assert_eq!(proto.dims(), [1, 8, 32, 32]);
        assert_eq!(coefcls.dims(), [1, 88, 8, 8]);
    }

    /// coco8-seg 过拟合冒烟（验收标准 1）：单图多步训练，loss 显著下降，
    /// 且「原型×系数」组合出的实例掩码 IoU 随训练显著上升。
    #[test]
    fn coco8_seg_overfit_smoke() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/coco8-seg");
        // 数据集不入库（.gitignore /data/）：缺失时跳过（CI 无数据集也能全绿）
        let Ok(samples) = load_cocoseg_dir(&root, "train", 128) else {
            println!(
                "跳过 coco8-seg 冒烟：数据集不存在（{}）；获取方式见 docs/USAGE.md",
                root.display()
            );
            return;
        };
        // 取实例最多的图（单图过拟合，BN batch=1 的方差由小画布缓解）。
        let sample = samples
            .iter()
            .max_by_key(|s| s.masks.len())
            .expect("至少一张含实例")
            .clone();
        assert!(!sample.masks.is_empty());

        let device = device();
        let x = Tensor::<crate::TrainB, 4>::from_data(
            TensorData::new(sample.pixels.clone(), [1, 3, 128, 128]),
            &device,
        );
        let masks = vec![sample.masks.clone()];
        let labels = vec![sample.labels.clone()];

        let mut model = SegNet::<crate::TrainB>::new(&micro_cfg(), &device).unwrap();
        let tcfg = TrainCfg {
            lr: 0.01,
            lr_min: 0.0005,
            weight_decay: 0.0,
            max_grad_norm: 10.0,
            total_steps: 60,
        };
        let mut optim = make_optimizer(&tcfg);
        let mut first = 0.0f32;
        let mut last = 0.0f32;
        for step in 0..tcfg.total_steps {
            let lr = cosine_lr(&tcfg, step);
            let (m, v) = train_step(model, &mut optim, lr, |m| {
                m.loss(x.clone(), &masks, &labels)
            });
            model = m;
            if step == 0 {
                first = v;
            }
            last = v;
        }
        assert!(
            last < first * 0.35,
            "loss 应显著下降：first={first} last={last}"
        );
        println!(
            "coco8-seg 过拟合冒烟：loss first={first:.4} last={last:.4}（降到初值 {:.1}%）",
            100.0 * last / first
        );

        // 掩码链路证据：最大实例在训练后的掩码 IoU 应明显优于随机/未训练水平。
        let eval = model.valid();
        let (proto, coefcls) = eval.forward_seg(x.inner());
        let (k, mh, mw) = (8usize, 32usize, 32usize);
        let (gh, gw) = (8usize, 8usize);
        // 最大实例
        let (gi, gt_mask) = sample
            .masks
            .iter()
            .enumerate()
            .max_by_key(|(_, m)| m.iter().filter(|&&v| v != 0).count())
            .unwrap();
        let proto_data = proto.into_data().convert::<f32>();
        let coef_data = coefcls.into_data().convert::<f32>();
        let pd = proto_data.to_vec::<f32>().unwrap();
        let cd = coef_data.to_vec::<f32>().unwrap();
        let p = pd
            .iter()
            .map(|&v| 1.0 / (1.0 + (-v).exp()))
            .collect::<Vec<_>>();
        let (hi, wi) = centroid_cell(gt_mask, mw, gh, gw).unwrap();
        let c = 80usize;
        // coef[i][c..c+k][hi][wi]
        let coef_base = gi * (c + k) * gh * gw + c * gh * gw + hi * gw + wi;
        let mut pred_mask = vec![0u8; mh * mw];
        for y in 0..mh {
            for xx in 0..mw {
                let mut acc = 0.0f32;
                for j in 0..k {
                    acc += p[j * mh * mw + y * mw + xx] * cd[coef_base + j * gh * gw];
                }
                if (1.0 / (1.0 + (-acc).exp())) > 0.5 {
                    pred_mask[y * mw + xx] = 1;
                }
            }
        }
        let iou = |a: &[u8], b: &[u8]| -> f32 {
            let inter = a.iter().zip(b).filter(|(&u, &v)| u != 0 && v != 0).count();
            let union = a.iter().zip(b).filter(|(&u, &v)| u != 0 || v != 0).count();
            if union == 0 {
                0.0
            } else {
                inter as f32 / union as f32
            }
        };
        let iou_final = iou(&pred_mask, gt_mask);
        println!("coco8-seg 过拟合冒烟：最大实例掩码 IoU={iou_final:.3}");
        assert!(
            iou_final > 0.4,
            "过拟合后最大实例掩码 IoU 应 > 0.4（got {iou_final}）"
        );
    }
}
