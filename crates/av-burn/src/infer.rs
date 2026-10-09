//! 推理解码链路：类别 argmax → conf 截取 → 批量原型合成 → 掩码 NMS →
//! [`SegInstance`]。
//!
//! 语义对齐 `crates/av-tasks/src/models.rs` 的 `SegModel::predict`（先读后写）：
//!
//! - 每 cell 对 C 类取 argmax 分数与类别下标，候选 = score ≥ conf；
//! - 候选按分数降序（稳定，平局保持 cell 行序）截到 [`MAX_SEGS_PER_IMAGE`]；
//! - 全部候选掩码 logits 一次批量合成：`[M,K] × [K,mh·mw]`（与训练共用
//!   「sigmoid 原型 × 系数」组合语义，见 [`crate::seg::combine_proto_coef`]；
//!   批量求和次序与逐实例循环可在 logit 极近 0 处产生 ±1ulp 差异，阈值化后
//!   理论可能翻转个别像素，掩码语义不变——与 tch 版同款口径）；
//! - `logit > 0`（⟺ sigmoid > 0.5）二值化，空掩码候选丢弃；
//! - [`mask_nms`]：分数降序贪心，同类掩码 IoU ≥ 阈值者抑制（外接框不相交
//!   对免整画布扫描，与 tch 版 MaskSummary 同思路）。
//!
//! 输出 [`SegInstance`] 字段与 tch 版逐一同名同型（label/score/mask，
//! mask 为 img/4 画布行优先扁平 0/1）；[`upmask_to_original`] 提供掩码 →
//! 原图坐标的最近邻上采样（letterbox 逆映射）。

use av_core::error::AvResult;
use av_core::geometry::Letterbox;
use burn_core::tensor::backend::Backend;
use burn_core::tensor::{activation::sigmoid, Tensor, TensorData};

use crate::data::decode_letterbox_chw;
use crate::seg::SegNet;

/// 单图实例数上限（conf 过低时防解码爆炸；tch 版同款常量）。
pub const MAX_SEGS_PER_IMAGE: usize = 100;

/// 单实例分割结果（字段与 tch 版 `av_tasks::models::SegInstance` 同名同型）。
#[derive(Debug, Clone)]
pub struct SegInstance {
    /// 类别下标（快照 classes 里的行号）。
    pub label: u32,
    /// 类别置信度（argmax 分数，0~1）。
    pub score: f32,
    /// 0/1 掩码，img/4 画布（letterbox 画布空间），行优先扁平。
    pub mask: Vec<u8>,
}

/// 两掩码 IoU（同长扁平 0/1 画布；任一为空且另一非空 → 0）。
pub fn mask_iou(a: &[u8], b: &[u8]) -> f32 {
    assert_eq!(a.len(), b.len(), "掩码画布尺寸须一致");
    let mut inter = 0usize;
    let mut union = 0usize;
    for (&u, &v) in a.iter().zip(b) {
        if u != 0 || v != 0 {
            union += 1;
        }
        if u != 0 && v != 0 {
            inter += 1;
        }
    }
    if union == 0 {
        0.0
    } else {
        inter as f32 / union as f32
    }
}

/// 掩码外接框（含端点；空掩码 None）——不相交对免整画布 IoU 扫描。
fn bbox_of(mask: &[u8], w: usize) -> Option<(usize, usize, usize, usize)> {
    let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0, 0);
    for (i, &v) in mask.iter().enumerate() {
        if v != 0 {
            let (x, y) = (i % w, i / w);
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
        }
    }
    if x1 < x0 {
        None
    } else {
        Some((x0, y0, x1, y1))
    }
}

/// 掩码 NMS：分数降序贪心，**同类**掩码 IoU ≥ `nms_iou` 者抑制。
/// total_cmp 全序比较（NaN 分数不 panic）；截到 [`MAX_SEGS_PER_IMAGE`]。
pub fn mask_nms(mut insts: Vec<SegInstance>, nms_iou: f32) -> Vec<SegInstance> {
    insts.sort_by(|a, b| b.score.total_cmp(&a.score));
    insts.truncate(MAX_SEGS_PER_IMAGE);
    let mut kept: Vec<SegInstance> = Vec::new();
    let mut kept_box: Vec<(usize, usize, usize, usize)> = Vec::new();
    for cand in insts {
        if cand.mask.iter().all(|&v| v == 0) {
            continue; // 空掩码无意义，丢弃
        }
        let w = (cand.mask.len() as f64).sqrt().round() as usize;
        debug_assert_eq!(w * w, cand.mask.len(), "掩码画布应为正方形");
        let cb = bbox_of(&cand.mask, w).expect("非空掩码必有外接框");
        let suppressed = kept.iter().zip(&kept_box).any(|(kp, kb)| {
            kp.label == cand.label
                && !(cb.0 > kb.2 || kb.0 > cb.2 || cb.1 > kb.3 || kb.1 > cb.3)
                && mask_iou(&kp.mask, &cand.mask) >= nms_iou
        });
        if !suppressed {
            kept_box.push(cb);
            kept.push(cand);
        }
    }
    kept
}

impl<B: Backend> SegNet<B> {
    /// 推理解码（语义对齐 tch `SegModel::predict`；请传 `model.valid()` 的
    /// 推理模型，避免 Autodiff 模型上建无谓计算图）。
    ///
    /// 输入 `[N,3,S,S]`（letterbox 预处理后的 [0,1] CHW）；返回每图的
    /// [`SegInstance`] 列表（mask 为 img/4 画布空间）。
    pub fn predict(
        &self,
        x: Tensor<B, 4>,
        conf: f32,
        nms_iou: f32,
    ) -> AvResult<Vec<Vec<SegInstance>>> {
        let device = x.device();
        let (proto, coefcls) = self.forward_seg(x);
        let n = proto.dims()[0];
        let c = self.num_classes();
        let k = self.num_protos();
        let [_np, _nk, mh, mw] = proto.dims();
        let [_nc, _nck, gh, gw] = coefcls.dims();
        let cells = gh * gw;
        let plane = mh * mw;

        // 原型整批一次 sigmoid（「基掩码」，与训练/combine_proto_coef 同语义）
        let proto_sig = sigmoid(proto);
        let proto_n0 = proto_sig.clone().reshape([n, k, plane]);
        // 类别分数图（前 C 通道）与系数图（后 K 通道）
        let cls_sig = sigmoid(coefcls.clone().slice([0..n, 0..c])); // [N,C,gh,gw]
        let coef_map = coefcls.slice([0..n, c..(c + k)]).reshape([n, k, cells]);

        // argmax 一次算完：[N,cells] 分数 + 类别下标，整批下行一次
        let (scores_t, cls_idx_t) = cls_sig.max_dim_with_indices(1);
        let scores_v: Vec<f32> = scores_t
            .reshape([n, cells])
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("f32 分数下行");
        let cls_idx_v: Vec<i32> = cls_idx_t
            .reshape([n, cells])
            .into_data()
            .convert::<i32>()
            .to_vec::<i32>()
            .expect("类别下标下行");

        let mut out: Vec<Vec<SegInstance>> = Vec::with_capacity(n);
        for i in 0..n {
            let row = &scores_v[i * cells..(i + 1) * cells];
            // 候选 = score ≥ conf（NaN 比较为 false 自动排除）
            let cand: Vec<usize> = (0..cells).filter(|&p| row[p] >= conf).collect();
            if cand.is_empty() {
                out.push(Vec::new());
                continue;
            }
            // 分数降序稳定排序（平局保持 cell 行序，与 tch decode 同款），
            // 截到上限——截断后喂 NMS 与全量合成逐实例一致。
            let mut order = cand;
            order.sort_by(|&a, &b| row[b].total_cmp(&row[a]));
            order.truncate(MAX_SEGS_PER_IMAGE);
            let m = order.len();

            // 系数按候选下标批量上设备 → 一次批量合成全部掩码 logits
            let idx = Tensor::<B, 1, burn_core::tensor::Int>::from_data(
                TensorData::new(order.iter().map(|&p| p as u32).collect::<Vec<_>>(), [m]),
                &device,
            );
            let coef_i = coef_map
                .clone()
                .slice([i..(i + 1), 0..k, 0..cells])
                .reshape([k, cells])
                .select(1, idx)
                .transpose(); // [M,K]
            let proto_i = proto_n0
                .clone()
                .slice([i..(i + 1), 0..k, 0..plane])
                .reshape([k, plane]);
            let logits = coef_i.matmul(proto_i); // [M, plane]
            let masks_v: Vec<u8> = logits
                .greater_elem(0.0f32)
                .int()
                .into_data()
                .convert::<u8>()
                .to_vec::<u8>()
                .expect("掩码下行");

            let mut insts: Vec<SegInstance> = Vec::with_capacity(m);
            for (j, mask) in masks_v.chunks_exact(plane).enumerate() {
                insts.push(SegInstance {
                    label: cls_idx_v[i * cells + order[j]] as u32,
                    score: row[order[j]],
                    mask: mask.to_vec(),
                });
            }
            out.push(mask_nms(insts, nms_iou));
        }
        Ok(out)
    }
}

/// 掩码画布 → 原图坐标的最近邻上采样（letterbox 逆映射）。
///
/// 原图像素 (ox, oy) → 画布 (ox·scale+pad_left, oy·scale+pad_top) →
/// 掩码格 (÷(dst/mw))；越界（内容区外）记 0。`mw·4 == lb.dst_w`、
/// `mh·4 == lb.dst_h`（掩码画布恒为 letterbox 画布的 1/4）。
pub fn upmask_to_original(
    mask: &[u8],
    mw: usize,
    mh: usize,
    lb: &Letterbox,
    orig_w: u32,
    orig_h: u32,
) -> Vec<u8> {
    debug_assert_eq!(mw * 4, lb.dst_w as usize, "掩码画布宽应为 dst_w/4");
    debug_assert_eq!(mh * 4, lb.dst_h as usize, "掩码画布高应为 dst_h/4");
    let sx = mw as f32 / lb.dst_w as f32;
    let sy = mh as f32 / lb.dst_h as f32;
    let mut out = vec![0u8; (orig_w as usize) * (orig_h as usize)];
    for oy in 0..orig_h {
        let cy = (oy as f32 * lb.scale + lb.pad_top) * sy;
        let my = (cy.floor() as isize).clamp(0, mh as isize - 1);
        for ox in 0..orig_w {
            let cx = (ox as f32 * lb.scale + lb.pad_left) * sx;
            let mx = (cx.floor() as isize).clamp(0, mw as isize - 1);
            out[(oy as usize) * orig_w as usize + ox as usize] =
                mask[my as usize * mw + mx as usize];
        }
    }
    out
}

/// 单图推理便捷入口：读图 → letterbox → [`SegNet::predict`] → 掩码上采样回
/// 原图。返回 `(实例列表（mask 已映射到原图尺寸）, letterbox 元数据)`。
pub fn predict_image<B: Backend>(
    model: &SegNet<B>,
    device: &B::Device,
    rgb: &image::RgbImage,
    img_size: u32,
    conf: f32,
    nms_iou: f32,
) -> AvResult<(Vec<SegInstance>, Letterbox)> {
    let (pixels, lb) = decode_letterbox_chw(rgb, img_size);
    let s = img_size as usize;
    let x = Tensor::<B, 4>::from_data(TensorData::new(pixels, [1, 3, s, s]), device);
    let raw = model.predict(x, conf, nms_iou)?;
    let (mw, mh) = (lb.dst_w as usize / 4, lb.dst_h as usize / 4);
    let mut out = Vec::with_capacity(raw[0].len());
    for inst in &raw[0] {
        out.push(SegInstance {
            label: inst.label,
            score: inst.score,
            mask: upmask_to_original(&inst.mask, mw, mh, &lb, rgb.width(), rgb.height()),
        });
    }
    Ok((out, lb))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NdArrayB;
    use burn_ndarray::NdArrayDevice;

    #[test]
    fn mask_iou_hand_computed() {
        // 4 像素 vs 1 像素，交 1 → 1/4
        let a = vec![1u8, 1, 1, 1];
        let b = vec![0u8, 1, 0, 0];
        assert!((mask_iou(&a, &b) - 0.25).abs() < 1e-6);
        assert_eq!(mask_iou(&a, &a), 1.0);
        assert_eq!(mask_iou(&a, &[0u8; 4]), 0.0);
        // 双空画布 union=0 → 0（与 tch 版语义一致）
        assert_eq!(mask_iou(&[0u8; 4], &[0u8; 4]), 0.0);
    }

    fn inst(label: u32, score: f32, mask: Vec<u8>) -> SegInstance {
        SegInstance { label, score, mask }
    }

    #[test]
    fn mask_nms_suppresses_same_label_keeps_other_labels() {
        // 6×6 画布，两同标签掩码交 4/并 8（IoU 0.5）+ 一异标签重叠掩码
        let full = vec![1u8; 36];
        let mut half = vec![0u8; 36];
        for y in 0..6 {
            for x in 0..4 {
                half[y * 6 + x] = 1;
            }
        } // 前 4 列
        let insts = vec![
            inst(0, 0.9, full.clone()),
            inst(0, 0.8, half),         // 与 full IoU = 24/36 ≈ 0.667 ≥ 0.5 → 抑制
            inst(1, 0.7, full.clone()), // 异标签 → 保留
        ];
        let kept = mask_nms(insts, 0.5);
        assert_eq!(kept.len(), 2);
        assert_eq!(kept[0].score, 0.9);
        assert_eq!(kept[0].label, 0);
        assert_eq!(kept[1].label, 1);
        // 阈值 0.7 时不抑制（IoU 0.667 < 0.7）
        let insts = vec![
            inst(0, 0.9, full.clone()),
            inst(0, 0.8, {
                let mut h = vec![0u8; 36];
                for y in 0..6 {
                    for x in 0..4 {
                        h[y * 6 + x] = 1;
                    }
                }
                h
            }),
        ];
        assert_eq!(mask_nms(insts, 0.7).len(), 2);
    }

    #[test]
    fn mask_nms_drops_empty_masks_and_respects_cap() {
        let kept = mask_nms(
            vec![inst(0, 0.9, vec![0u8; 4]), inst(1, 0.8, vec![1u8; 4])],
            0.5,
        );
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].label, 1);
        // 上限 100：构造 150 个互不相交的单像素实例（160×160 方画布对角布点）
        let insts: Vec<SegInstance> = (0..150)
            .map(|i| {
                let mut m = vec![0u8; 160 * 160];
                m[i * 160 + i] = 1; // 沿对角线放单像素，同类但 IoU=0
                inst(0, 1.0 - i as f32 / 1000.0, m)
            })
            .collect();
        assert_eq!(mask_nms(insts, 0.5).len(), MAX_SEGS_PER_IMAGE);
    }

    #[test]
    fn segnet_predict_smoke_and_shapes() {
        let device = NdArrayDevice::default();
        let cfg = crate::seg::SegNetCfg {
            width: 0.0625,
            depth: 0.33,
            num_classes: 3,
            num_protos: 8,
            loss_w_bce: 1.0,
            loss_w_dice: 1.0,
        };
        let model = SegNet::<NdArrayB>::new(&cfg, &device).unwrap();
        let x = Tensor::<NdArrayB, 4>::ones([1, 3, 128, 128], &device);
        // 阈值极端：conf=1.1 → 全空；conf=0 → 解码不 panic，掩码画布 32×32
        let empty = model.predict(x.clone(), 1.1, 0.7).unwrap();
        assert!(empty[0].is_empty());
        let all = model.predict(x, 0.0, 0.7).unwrap();
        for inst in &all[0] {
            assert_eq!(inst.mask.len(), 32 * 32);
            assert!(inst.score >= 0.0);
            assert!((inst.label as usize) < 3);
        }
        assert!(all[0].len() <= MAX_SEGS_PER_IMAGE);
    }

    #[test]
    fn upmask_to_original_hand_computed() {
        // 无缩放无补边（scale 1，pad 0）：8×8 画布、2×2 掩码 → 8×8 原图按
        // 4×4 象限最近邻放大。mask = [0,1 / 1,0] → 右上/左下象限为 1，各 16 像素
        let lb = Letterbox {
            scale: 1.0,
            pad_left: 0.0,
            pad_top: 0.0,
            dst_w: 8,
            dst_h: 8,
        };
        let mask = vec![0u8, 1, 1, 0];
        let up = upmask_to_original(&mask, 2, 2, &lb, 8, 8);
        let expect = [
            0u8, 0, 0, 0, 1, 1, 1, 1, //
            0, 0, 0, 0, 1, 1, 1, 1, //
            0, 0, 0, 0, 1, 1, 1, 1, //
            0, 0, 0, 0, 1, 1, 1, 1, //
            1, 1, 1, 1, 0, 0, 0, 0, //
            1, 1, 1, 1, 0, 0, 0, 0, //
            1, 1, 1, 1, 0, 0, 0, 0, //
            1, 1, 1, 1, 0, 0, 0, 0,
        ];
        assert_eq!(up, expect);
        // 带补边（scale 0.5，pad_left 4）：原图 4×2 → 画布 cx∈[4,5.5] 全落
        // mx=1（mask[1]=1）、cy∈[0,0.5] 落 my=0
        let lb2 = Letterbox {
            scale: 0.5,
            pad_left: 4.0,
            pad_top: 0.0,
            dst_w: 8,
            dst_h: 8,
        };
        let up2 = upmask_to_original(&mask, 2, 2, &lb2, 4, 2);
        assert!(up2.iter().all(|&v| v == 1), "逆映射应全落 mask[1]");
    }
}
