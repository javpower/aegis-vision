//! COCO 风格检测评测协议（真实基准，M8 前置）：IoU 0.50:0.05:0.95 十档、
//! 按 score 降序贪心匹配、101 点插值 AP、按类别求平均得 mAP50 与 mAP50:95。
//!
//! 与 pycocotools/COCO 对齐的协议要点：
//! - 按类别独立评估，仅对「存在 gt 的类别」求平均（无 gt 的类别上的预测直接忽略）；
//! - 每个 IoU 档位独立做匹配：该类全部预测跨图合并后按 score 降序贪心，
//!   每个 gt 至多匹配一次（同图同类内取 IoU 最大且 ≥ 阈值的未匹配 gt）；
//! - AP 用 101 点插值：对 recall ∈ {0.00, 0.01, …, 1.00} 取 recall ≥ r 的
//!   最大 precision（precision 包络），对 101 个点求均值。
//!
//! 纯 Rust 无第三方依赖。预测与真值必须在同一像素空间（输入画布或原图均可，
//! 只要同空间即可直接比较；letterbox 预处理后两者都在 img_size 画布空间）。

use std::collections::BTreeMap;

use rayon::prelude::*;

use av_core::geometry::Aabb;
use av_core::types::Detection;

/// IoU 阈值档位：0.50:0.05:0.95 共 10 档（COCO 惯例）。
pub const IOU_THRESHOLDS: [f64; 10] = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95];

/// 真值框（与预测同一像素空间）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GtBox {
    pub bbox: Aabb,
    pub class_id: u32,
}

impl GtBox {
    pub fn new(bbox: Aabb, class_id: u32) -> Self {
        Self { bbox, class_id }
    }
}

/// mAP 报告。空评测集 / 无 gt 时指标为 0（不产生 NaN）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MapReport {
    /// COCO 主指标：十档 IoU AP 的类别均值（mAP50:95）。
    pub map50_95: f32,
    /// IoU=0.50 单档 AP 的类别均值（mAP50）。
    pub map50: f32,
    /// Ultralytics parity 的 precision：逐类在 IoU=0.5 档的 PR 曲线上取
    /// max-F1 点的 precision，对存在 gt 的类别求平均（`metrics/precision(B)`）。
    pub precision: f32,
    /// 同 [`MapReport::precision`] 的 recall（`metrics/recall(B)`）。
    pub recall: f32,
    /// 参与平均的 gt 类别数。
    pub num_classes: usize,
    /// 已记录的图片数。
    pub num_images: usize,
    /// 真值框总数。
    pub num_gts: usize,
    /// 预测框总数。
    pub num_dets: usize,
}

#[derive(Debug)]
struct ImageRec {
    dets: Vec<Detection>,
    gts: Vec<GtBox>,
}

/// 增量式 COCO 风格评测器：逐图 `update`，结束后 `finalize` 出报告。
#[derive(Debug, Default)]
pub struct CocoEvaluator {
    images: BTreeMap<u32, ImageRec>,
}

impl CocoEvaluator {
    pub fn new() -> Self {
        Self::default()
    }

    /// 记录一张图的预测与真值（同一 image_id 重复 update 时覆盖旧记录）。
    pub fn update(&mut self, image_id: u32, dets: &[Detection], gts: &[GtBox]) {
        self.images.insert(
            image_id,
            ImageRec {
                dets: dets.to_vec(),
                gts: gts.to_vec(),
            },
        );
    }

    /// 汇总所有已记录图片，输出 mAP 报告（评测器本身不可变，可重复调用）。
    pub fn finalize(&self) -> MapReport {
        let num_gts: usize = self.images.values().map(|r| r.gts.len()).sum();
        let num_dets: usize = self.images.values().map(|r| r.dets.len()).sum();

        // gt 中出现过的类别（BTreeMap 键序保证升序去重，求平均只算有 gt 的类别）
        let mut classes: Vec<u32> = self
            .images
            .values()
            .flat_map(|r| r.gts.iter().map(|g| g.class_id))
            .collect();
        classes.sort_unstable();
        classes.dedup();

        let report = MapReport {
            map50_95: 0.0,
            map50: 0.0,
            precision: 0.0,
            recall: 0.0,
            num_classes: classes.len(),
            num_images: self.images.len(),
            num_gts,
            num_dets,
        };
        if classes.is_empty() {
            return report;
        }

        // 逐类独立评测，rayon 按类并行：每类只构建一次上下文（det 列表 +
        // det×gt IoU 矩阵），10 个 IoU 档位复用同一份 IoU——旧实现每档重算
        // 全部 IoU，评测是 10 倍重复计算。保序 collect 后仍按类 id 升序、
        // 档位序累加，f64 求和顺序与串行逐位一致。
        let per_class: Vec<([f64; 10], (f64, f64))> = classes
            .par_iter()
            .map(|&c| {
                let ctx = self.class_context(c);
                let (prec, rec) = class_pr_curve_at(&ctx, IOU_THRESHOLDS[0]);
                let pr = pr_at_max_f1(&prec, &rec);
                let mut aps = [0.0f64; 10];
                for (ti, &thr) in IOU_THRESHOLDS.iter().enumerate() {
                    aps[ti] = class_ap_at(&ctx, thr);
                }
                (aps, pr)
            })
            .collect();
        let mut sum_all = 0.0f64; // Σ AP over (class, 10 thresholds)
        let mut sum_50 = 0.0f64; // Σ AP at IoU=0.50
        let mut sum_p = 0.0f64; // Σ per-class precision@max-F1
        let mut sum_r = 0.0f64; // Σ per-class recall@max-F1
        for (aps, (p, r)) in &per_class {
            for (ti, &ap) in aps.iter().enumerate() {
                sum_all += ap;
                if ti == 0 {
                    sum_50 += ap;
                }
            }
            sum_p += p;
            sum_r += r;
        }
        let n = classes.len() as f64;
        MapReport {
            map50_95: (sum_all / (10.0 * n)) as f32,
            map50: (sum_50 / n) as f32,
            precision: (sum_p / n) as f32,
            recall: (sum_r / n) as f32,
            ..report
        }
    }

    /// 单类别的评测上下文：按图归组的 gt、跨图合并后按 score 降序的 det，
    /// 以及每个 det 与同图全部 gt 的 IoU（阈值无关，只算一次）。
    fn class_context(&self, cls: u32) -> ClassContext {
        let mut gt_boxes: Vec<Vec<Aabb>> = Vec::with_capacity(self.images.len());
        // (图下标, score, 框)——收集后统一按 score 降序
        let mut dets: Vec<(usize, f64, Aabb)> = Vec::new();
        let mut n_gt = 0usize;
        for (idx, rec) in self.images.values().enumerate() {
            let g: Vec<Aabb> = rec
                .gts
                .iter()
                .filter(|gt| gt.class_id == cls)
                .map(|gt| gt.bbox)
                .collect();
            n_gt += g.len();
            gt_boxes.push(g);
            for d in &rec.dets {
                if d.class_id == cls {
                    dets.push((idx, d.score as f64, d.bbox));
                }
            }
        }

        // 跨图合并后按 score 降序（稳定排序：同分按图序/插入序，保证确定性）
        dets.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // det×gt IoU 矩阵（与排序后的 det 对齐；阈值无关，只算一次）
        let ious: Vec<Vec<f64>> = dets
            .iter()
            .map(|(img, _, bbox)| gt_boxes[*img].iter().map(|g| g.iou(bbox) as f64).collect())
            .collect();

        ClassContext {
            gt_boxes,
            dets_img: dets.iter().map(|(img, _, _)| *img).collect(),
            ious,
            n_gt,
        }
    }
}

/// 单类别评测上下文（IoU 矩阵跨 10 个阈值档位复用）。
struct ClassContext {
    /// 每图该类的 gt 框。
    gt_boxes: Vec<Vec<Aabb>>,
    /// 按 score 降序排列的预测所属图下标。
    dets_img: Vec<usize>,
    /// `ious[i][g]` = 第 i 个预测与其同图第 g 个 gt 的 IoU。
    ious: Vec<Vec<f64>>,
    n_gt: usize,
}

/// 在给定 IoU 阈值下对缓存上下文做贪心匹配 → TP/FP 序列 → 101 点插值 AP。
fn class_ap_at(ctx: &ClassContext, thr: f64) -> f64 {
    let (prec, rec) = class_pr_curve_at(ctx, thr);
    ap_101(&rec, &prec)
}

/// 单类在给定 IoU 阈值下的累积 PR 序列（按 score 降序逐检出的累积
/// precision/recall，未做包络平滑——Ultralytics `ap_per_class` 同语义）。
fn class_pr_curve_at(ctx: &ClassContext, thr: f64) -> (Vec<f64>, Vec<f64>) {
    let mut gt_matched: Vec<Vec<bool>> =
        ctx.gt_boxes.iter().map(|g| vec![false; g.len()]).collect();
    let mut rec = Vec::with_capacity(ctx.dets_img.len());
    let mut prec = Vec::with_capacity(ctx.dets_img.len());
    let (mut tp, mut fp) = (0usize, 0usize);
    for (di, &img) in ctx.dets_img.iter().enumerate() {
        let mut best = (0.0f64, usize::MAX); // (IoU, 未匹配 gt 下标)
        for (gi, _) in ctx.gt_boxes[img].iter().enumerate() {
            if gt_matched[img][gi] {
                continue;
            }
            let iou = ctx.ious[di][gi];
            if iou >= thr && iou > best.0 {
                best = (iou, gi);
            }
        }
        if best.1 != usize::MAX {
            gt_matched[img][best.1] = true;
            tp += 1;
        } else {
            fp += 1;
        }
        rec.push(tp as f64 / ctx.n_gt as f64);
        prec.push(tp as f64 / (tp + fp) as f64);
    }
    (prec, rec)
}

/// PR 序列上 F1 最大点的 (precision, recall)。Ultralytics 报告的
/// `metrics/precision(B)` / `metrics/recall(B)` 即各自类别曲线该点的
/// 类均值（此处 argmax 不做 smooth 预处理，同分取更早点，确定性一致）。
fn pr_at_max_f1(prec: &[f64], rec: &[f64]) -> (f64, f64) {
    let mut best = (0.0f64, 0.0f64, 0.0f64); // (f1, p, r)
    for i in 0..prec.len() {
        let (p, r) = (prec[i], rec[i]);
        let f1 = if p + r > 0.0 { 2.0 * p * r / (p + r) } else { 0.0 };
        if f1 > best.0 {
            best = (f1, p, r);
        }
    }
    (best.1, best.2)
}

/// 101 点插值 AP：对 r ∈ {0.00, 0.01, …, 1.00} 取 recall ≥ r 的最大 precision，
/// 对 101 个插值点求均值（等价 pycocotools `recThrs` 插值）。`rec` 单调不减。
fn ap_101(rec: &[f64], prec: &[f64]) -> f64 {
    // precision 包络：后缀最大值，env[i] = max(prec[i..])
    let mut env = vec![0.0f64; prec.len() + 1];
    for i in (0..prec.len()).rev() {
        env[i] = env[i + 1].max(prec[i]);
    }
    let mut ap = 0.0f64;
    let mut ri = 0usize; // 首个满足 rec >= r 的下标（r 单调递增，指针只前进）
    for k in 0..=100 {
        let r = k as f64 / 100.0;
        while ri < rec.len() && rec[ri] < r {
            ri += 1;
        }
        ap += env[ri];
    }
    ap / 101.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(x1: f32, y1: f32, x2: f32, y2: f32, score: f32, class_id: u32) -> Detection {
        Detection {
            bbox: Aabb::new(x1, y1, x2, y2),
            score,
            class_id,
            angle: None,
            keypoints: None,
        }
    }

    /// 手算对照例：2 图 2 类。
    ///
    /// 图 1 gt：c0 的 A=(0,0,10,10)；c1 的 B1=(0,0,20,20)、B2=(30,30,40,40)
    /// 图 1 预测：d1=c1 精确命中 B2 (0.95)；d2=c0 精确命中 A (0.9)；
    ///            d3=c1 半宽框 (0,0,10,20)，与 B1 的 IoU=0.5（只在 0.50 档算 TP）
    /// 图 2 gt：c0 的 C=(5,5,15,15)
    /// 图 2 预测：d4=c0 精确命中 C (0.7)；d5=c0 离谱框 (0.6) 恒 FP
    ///
    /// c0：全部档 AP=1.0；c1：0.50 档 AP=1.0，0.55..0.95 档 AP=51/101。
    /// mAP50=1.0；mAP50:95=(11 + 9*51/101)/20 ≈ 0.777228。
    #[test]
    fn hand_computed_two_images_two_classes() {
        let mut ev = CocoEvaluator::new();
        ev.update(
            1,
            &[
                det(30.0, 30.0, 40.0, 40.0, 0.95, 1), // d1 精确命中 B2
                det(0.0, 0.0, 10.0, 10.0, 0.9, 0),    // d2 精确命中 A
                det(0.0, 0.0, 10.0, 20.0, 0.8, 1),    // d3 IoU=0.5 with B1
            ],
            &[
                GtBox::new(Aabb::new(0.0, 0.0, 10.0, 10.0), 0),
                GtBox::new(Aabb::new(0.0, 0.0, 20.0, 20.0), 1),
                GtBox::new(Aabb::new(30.0, 30.0, 40.0, 40.0), 1),
            ],
        );
        ev.update(
            2,
            &[
                det(5.0, 5.0, 15.0, 15.0, 0.7, 0),       // d4 精确命中 C
                det(100.0, 100.0, 110.0, 110.0, 0.6, 0), // d5 恒 FP
            ],
            &[GtBox::new(Aabb::new(5.0, 5.0, 15.0, 15.0), 0)],
        );

        let rep = ev.finalize();
        assert_eq!(rep.num_images, 2);
        assert_eq!(rep.num_gts, 4);
        assert_eq!(rep.num_dets, 5);
        assert_eq!(rep.num_classes, 2);
        assert!((rep.map50 - 1.0).abs() < 1e-3, "mAP50={}", rep.map50);
        let expect = (11.0 + 9.0 * 51.0 / 101.0) / 20.0;
        assert!(
            (rep.map50_95 - expect as f32).abs() < 1e-3,
            "mAP50:95={} 期望 {expect}",
            rep.map50_95
        );
    }

    /// 空评测器 / 只有预测没有 gt：指标为 0 且无 NaN。
    #[test]
    fn empty_or_gtless_is_zero_not_nan() {
        let rep = CocoEvaluator::new().finalize();
        assert_eq!(rep.map50, 0.0);
        assert_eq!(rep.map50_95, 0.0);
        assert!(!rep.map50.is_nan() && !rep.map50_95.is_nan());

        let mut ev = CocoEvaluator::new();
        ev.update(
            7,
            &[det(0.0, 0.0, 10.0, 10.0, 0.9, 0)],
            &[], // 无 gt → 不参与平均
        );
        let rep = ev.finalize();
        assert_eq!(rep.num_classes, 0);
        assert_eq!(rep.num_images, 1);
        assert_eq!(rep.num_dets, 1);
        assert_eq!(rep.map50, 0.0);
        assert!(!rep.map50_95.is_nan());
    }

    /// gt 存在但预测全部不匹配：AP 应为 0。
    #[test]
    fn unmatched_gt_gives_zero_ap() {
        let mut ev = CocoEvaluator::new();
        ev.update(
            1,
            &[det(50.0, 50.0, 60.0, 60.0, 0.9, 0)],
            &[GtBox::new(Aabb::new(0.0, 0.0, 10.0, 10.0), 0)],
        );
        let rep = ev.finalize();
        assert_eq!(rep.num_classes, 1);
        assert_eq!(rep.map50, 0.0);
        assert_eq!(rep.map50_95, 0.0);
    }

    /// 高分重复框抢占 gt 后，低分精确框按 FP 计（贪心语义），且每个 gt 只匹配一次。
    #[test]
    fn greedy_takes_best_iou_among_unmatched() {
        let mut ev = CocoEvaluator::new();
        // 两个预测都命中同一 gt：高分者 TP，低分者 FP
        ev.update(
            1,
            &[
                det(0.0, 0.0, 10.0, 10.0, 0.8, 0),
                det(0.0, 0.0, 10.0, 10.0, 0.6, 0),
            ],
            &[GtBox::new(Aabb::new(0.0, 0.0, 10.0, 10.0), 0)],
        );
        let rep = ev.finalize();
        // tp=1, fp=1：rec=[1,1]，prec=[1,0.5] → 包络 [1,1] → 全部档 AP=1.0
        assert!((rep.map50 - 1.0).abs() < 1e-3, "mAP50={}", rep.map50);
    }

    /// P/R 取 max-F1 点（Ultralytics parity）：精确命中 + 一个低分 FP。
    /// PR 曲线：prec=[1, 0.5]，rec=[1, 1]；F1=[1, 2/3] → max-F1 点 P=1, R=1。
    #[test]
    fn precision_recall_at_max_f1() {
        let mut ev = CocoEvaluator::new();
        ev.update(
            1,
            &[
                det(0.0, 0.0, 10.0, 10.0, 0.9, 0),  // TP（精确）
                det(50.0, 50.0, 60.0, 60.0, 0.8, 0), // FP
            ],
            &[GtBox::new(Aabb::new(0.0, 0.0, 10.0, 10.0), 0)],
        );
        let rep = ev.finalize();
        assert!((rep.precision - 1.0).abs() < 1e-3, "P={}", rep.precision);
        assert!((rep.recall - 1.0).abs() < 1e-3, "R={}", rep.recall);

        // 2 个 gt 只命中 1 个：prec=[1]，rec=[0.5]，F1=2/3 → P=1, R=0.5
        let mut ev = CocoEvaluator::new();
        ev.update(
            1,
            &[det(0.0, 0.0, 10.0, 10.0, 0.9, 0)],
            &[
                GtBox::new(Aabb::new(0.0, 0.0, 10.0, 10.0), 0),
                GtBox::new(Aabb::new(30.0, 30.0, 40.0, 40.0), 0),
            ],
        );
        let rep = ev.finalize();
        assert!((rep.precision - 1.0).abs() < 1e-3, "P={}", rep.precision);
        assert!((rep.recall - 0.5).abs() < 1e-3, "R={}", rep.recall);
    }
}
