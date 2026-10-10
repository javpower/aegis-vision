//! TAL（Task-Aligned Assigner，YOLOv8 风格）标签分配器：纯逻辑实现，不依赖 tch。
//!
//! 对齐指标 t = s_cls^α × IoU^β（默认 α=0.5 / β=6.0，可经 [`TalConfig`] 配置）。
//! 对每个 gt：
//! 1. 在全部候选 cell（两层 stride 8/16 拼接后的扁平序列）上计算 t；
//! 2. 候选过滤（YOLOv8 `select_candidates_in_gts` 同款）：cell 中心须落在 gt 框内，
//!    且预测框与 gt 有正重叠（零重叠的 cell 对回归无意义）；
//! 3. 取 top-k（k 默认 10，且不超过候选 cell 总数）为正样本；
//! 4. 若某 gt 过滤后没有任何候选（极小目标恰跨在格点间隙），回退为
//!    「距 gt 中心最近的 cell」，保证每个 gt 至少 1 个正样本。
//!
//! 冲突消解：一个 cell 被多个 gt 争夺时取 t 更大者；t 相同取 gt 序号小者
//! （先到先得，保证确定性）。
//!
//! 正样本权重 = t / 该 gt 的最大 t ∈ (0, 1]，供 cls 损失做软标签 / 加权
//! （YOLOv8 的 norm_align_metric 同款）。

/// 默认 top-k（YOLOv8 同款）。
pub const DEFAULT_TOPK: usize = 10;
/// 对齐指标 cls 分数项指数 α。
pub const DEFAULT_ALPHA: f32 = 0.5;
/// 对齐指标 IoU 项指数 β。
pub const DEFAULT_BETA: f32 = 6.0;

/// TAL 配置（α / β / top-k 可调）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TalConfig {
    /// 每个 gt 最多分配的正样本 cell 数（实际再与 cell 总数取 min）。
    pub topk: usize,
    pub alpha: f32,
    pub beta: f32,
}

impl Default for TalConfig {
    fn default() -> Self {
        Self {
            topk: DEFAULT_TOPK,
            alpha: DEFAULT_ALPHA,
            beta: DEFAULT_BETA,
        }
    }
}

/// 一个正样本 cell 的分配结果。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PosCell {
    /// 全图扁平 cell 序号（层 0 在前、层 1 接后，与损失侧布局一致）。
    pub cell: usize,
    /// 归属的 gt 序号。
    pub gt: usize,
    /// 对齐指标 t = s_cls^α × IoU^β。
    pub metric: f32,
    /// 归一化权重 = (t × IoU) / 该 gt 候选内的 max(t × IoU)。
    /// Ultralytics `norm_align_metric` 同构：对齐指标再乘候选自身预测 IoU——
    /// 定位差的候选在 cls 软标签与 box/DFL 损失权重中被进一步降权。
    /// 本数据集受控对照 ±0.3 中性，语义对齐保留。
    pub weight: f32,
}

/// 对齐指标 t = s_cls^α × IoU^β（输入截断到 [0,1] 防脏数据）。
pub fn alignment_metric(cls_score: f32, iou: f32, alpha: f32, beta: f32) -> f32 {
    cls_score.clamp(0.0, 1.0).powf(alpha) * iou.clamp(0.0, 1.0).powf(beta)
}

/// xyxy 域 IoU（任一框退化面积为 0 时返回 0）。
pub fn iou_xyxy(a: [f32; 4], b: [f32; 4]) -> f32 {
    let iw = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let ih = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let inter = iw * ih;
    let union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
    if union <= 0.0 {
        0.0
    } else {
        (inter / union).clamp(0.0, 1.0)
    }
}

fn center_in_box(c: [f32; 2], b: [f32; 4]) -> bool {
    c[0] >= b[0] && c[0] <= b[2] && c[1] >= b[1] && c[1] <= b[3]
}

/// 对一张图做 TAL 分配。
///
/// - `pred_boxes`：每个 cell 的解码预测框 xyxy（长度 = 全部候选 cell 数，
///   两层特征拼扁平，层 0 在前）。
/// - `cell_centers`：每个 cell 中心（像素坐标）。
/// - `gt_scores`：每个 gt 一份「该 gt 类别下各 cell 的分类分数」（长度同 cell 数）。
/// - `gt_boxes`：gt 框 xyxy。
///
/// 返回与 cell 等长的数组：`Some(PosCell)` 表示该 cell 被分配给对应 gt。
pub fn assign_single_image(
    pred_boxes: &[[f32; 4]],
    cell_centers: &[[f32; 2]],
    gt_scores: &[&[f32]],
    gt_boxes: &[[f32; 4]],
    cfg: &TalConfig,
) -> Vec<Option<PosCell>> {
    let n_cells = pred_boxes.len();
    let mut out: Vec<Option<PosCell>> = vec![None; n_cells];
    if n_cells == 0 || gt_boxes.is_empty() {
        return out;
    }
    // top-k 不超过候选 anchor（cell）总数上限
    let k = cfg.topk.clamp(1, n_cells);
    let mut best: Vec<Option<PosCell>> = vec![None; n_cells];
    // 候选缓冲跨 gt 复用（640 输入下每 gt 扫 8 千 cell，逐 gt 重新分配是纯浪费）
    let mut cand: Vec<(f32, usize)> = Vec::new();

    for (g, gt) in gt_boxes.iter().enumerate() {
        let scores = gt_scores.get(g).copied().unwrap_or(&[]);
        let score = |c: usize| scores.get(c).copied().unwrap_or(0.0);

        // 候选过滤：中心在 gt 内 且 预测框与 gt 有正重叠。
        // 先跑 4 次比较的 center_in_box（绝大多数 cell 直接淘汰），
        // 再算贵的 IoU——顺序交换语义不变，每步省数千次多边形面积乘加。
        cand.clear();
        for (c, pb) in pred_boxes.iter().enumerate() {
            if !center_in_box(cell_centers[c], *gt) {
                continue;
            }
            let iou = iou_xyxy(*pb, *gt);
            if iou <= 0.0 {
                continue;
            }
            let m = alignment_metric(score(c), iou, cfg.alpha, cfg.beta);
            cand.push((m, c));
        }
        if cand.is_empty() {
            // 回退：距 gt 中心最近的 cell（平局取序号小者），保证 ≥1 正样本
            let gcx = (gt[0] + gt[2]) / 2.0;
            let gcy = (gt[1] + gt[3]) / 2.0;
            let mut best_c = 0usize;
            let mut best_d = f32::INFINITY;
            for (c, cc) in cell_centers.iter().enumerate() {
                let d = (cc[0] - gcx).powi(2) + (cc[1] - gcy).powi(2);
                if d < best_d {
                    best_d = d;
                    best_c = c;
                }
            }
            let iou = iou_xyxy(pred_boxes[best_c], *gt);
            let m = alignment_metric(score(best_c), iou, cfg.alpha, cfg.beta);
            cand.push((m, best_c));
        }

        // top-k：t 降序，平局取 cell 序号小者（确定性）
        cand.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.1.cmp(&b.1))
        });
        cand.truncate(k);
        // 归一化基准与权重：Ultralytics norm_align_metric——对齐指标再乘候选
        // 自身 IoU 后按该 gt 候选内最大值归一
        let max_w = cand
            .iter()
            .map(|&(m, c)| m * iou_xyxy(pred_boxes[c], *gt))
            .fold(0.0f32, f32::max);
        for &(m, c) in &cand {
            let iou = iou_xyxy(pred_boxes[c], *gt);
            let w = if max_w > 1e-12 {
                (m * iou / max_w).clamp(0.0, 1.0)
            } else {
                1.0
            };
            // 冲突消解：t 更大者赢；平局先到先得（gt 序号小者赢）
            let take = match &best[c] {
                Some(prev) => m > prev.metric,
                None => true,
            };
            if take {
                best[c] = Some(PosCell {
                    cell: c,
                    gt: g,
                    metric: m,
                    weight: w,
                });
            }
        }
    }

    for (c, b) in best.into_iter().enumerate() {
        out[c] = b;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    /// 4×2 网格、stride 8 的 cell 中心：
    /// c0(4,4)  c1(12,4)  c2(20,4)  c3(28,4)
    /// c4(4,12) c5(12,12) c6(20,12) c7(28,12)
    fn grid8() -> Vec<[f32; 2]> {
        let mut v = Vec::new();
        for hi in 0..2usize {
            for wi in 0..4usize {
                v.push([((2 * wi + 1) * 4) as f32, ((2 * hi + 1) * 4) as f32]);
            }
        }
        v
    }

    #[test]
    fn alignment_metric_hand_computed() {
        // t = 0.8^0.5 × 0.5^6 = 0.8944272 × 0.015625 = 0.0139754
        let m = alignment_metric(0.8, 0.5, 0.5, 6.0);
        assert!(approx(m, 0.8f32.sqrt() * 0.5f32.powi(6), 1e-6), "m={m}");
        assert!(approx(m, 0.013975424, 1e-6), "m={m}");
    }

    #[test]
    fn iou_hand_computed() {
        // inter=1, union=4+4-1=7
        assert!(approx(
            iou_xyxy([0., 0., 2., 2.], [1., 1., 3., 3.]),
            1.0 / 7.0,
            1e-6
        ));
        assert!(approx(
            iou_xyxy([0., 0., 4., 4.], [0., 0., 4., 4.]),
            1.0,
            1e-6
        ));
        assert_eq!(iou_xyxy([0., 0., 1., 1.], [2., 2., 3., 3.]), 0.0);
        // 退化框
        assert_eq!(iou_xyxy([0., 0., 0., 0.], [0., 0., 4., 4.]), 0.0);
    }

    #[test]
    fn topk_weights_and_center_filter_hand_computed() {
        // gt A = [8,0,24,16]；中心在 A 内的 cell：c1 c2 c5 c6。
        // c0 预测框与 A 完全重合（IoU=1）但中心 (4,4) 在 A 外 → 必须被中心过滤排除。
        let centers = grid8();
        let gt = [[8.0f32, 0.0, 24.0, 16.0]];
        let pred = [
            [8., 0., 24., 16.],  // c0：IoU=1 但中心在外 → 排除
            [8., 0., 24., 16.],  // c1：IoU=1
            [8., 0., 24., 16.],  // c2：IoU=1
            [8., 0., 24., 16.],  // c3：中心在外 → 排除
            [8., 0., 24., 16.],  // c4：中心在外 → 排除
            [12., 4., 20., 12.], // c5：IoU=64/256=0.25
            [12., 4., 20., 12.], // c6：IoU=0.25
            [8., 0., 24., 16.],  // c7：中心在外 → 排除
        ];
        let scores: &[&[f32]] = &[&[0.99, 0.9, 0.8, 0.0, 0.0, 0.9, 0.8, 0.0]];

        let out = assign_single_image(&pred, &centers, scores, &gt, &TalConfig::default());

        // 中心过滤：c0/c3/c4/c7 尽管可能 IoU 高也不得入选
        for c in [0usize, 3, 4, 7] {
            assert!(out[c].is_none(), "c{c} 不应入选");
        }
        // 指标：t = s^0.5 × IoU^6
        let t1 = 0.9f32.sqrt() * 1.0f32.powi(6); // c1 = 0.9486833
        let t2 = 0.8f32.sqrt() * 1.0f32.powi(6); // c2 = 0.8944272
        let t5 = 0.9f32.sqrt() * 0.25f32.powi(6); // c5 = 2.316065e-4
        let t6 = 0.8f32.sqrt() * 0.25f32.powi(6); // c6 = 2.1836596e-4
        let a = out[1].expect("c1 应入选");
        assert_eq!(a.gt, 0);
        assert!(approx(a.metric, t1, 1e-6));
        assert!(approx(a.weight, 1.0, 1e-6), "最大 t 的权重应为 1");
        let b = out[2].expect("c2 应入选");
        assert!(approx(b.metric, t2, 1e-6));
        assert!(approx(b.weight, t2 / t1, 1e-5));
        let e = out[5].expect("c5 应入选");
        assert!(approx(e.metric, t5, 1e-10));
        // norm_align_metric：权重 = (t × IoU) / max(t × IoU)，c5 的 IoU=0.25 二次计入
        assert!(approx(e.weight, t5 * 0.25 / t1, 1e-6));
        let f = out[6].expect("c6 应入选");
        assert!(approx(f.metric, t6, 1e-10));
        assert!(approx(f.weight, t6 * 0.25 / t1, 1e-6));
    }

    #[test]
    fn topk_limits_positive_count() {
        // 与上一测同布局，topk=2 → 只保留 t 最大的 c1、c2
        let centers = grid8();
        let gt = [[8.0f32, 0.0, 24.0, 16.0]];
        let pred = [[8., 0., 24., 16.]; 8];
        let scores: &[&[f32]] = &[&[0.0, 0.9, 0.8, 0.0, 0.0, 0.9, 0.8, 0.0]];
        let cfg = TalConfig {
            topk: 2,
            ..TalConfig::default()
        };
        let out = assign_single_image(&pred, &centers, scores, &gt, &cfg);
        let picked: Vec<usize> = out
            .iter()
            .enumerate()
            .filter(|(_, p)| p.is_some())
            .map(|(c, _)| c)
            .collect();
        // c1 与 c5 的 t 同为 0.9^0.5（IoU 均 1）→ 平局按 cell 序号取 c1、c5
        assert_eq!(picked, vec![1, 5], "topk=2 应按 t 降序取前两个");
    }

    #[test]
    fn conflict_larger_metric_wins() {
        // gt A=[8,0,24,16]、gt B=[16,8,32,24]；c6 (20,12) 同时在两框内（争夺 cell）。
        // c6 对 A：IoU=0.25, s=0.9 → t_A=2.316065e-4；对 B：IoU=0.25, s=0.8 → t_B=2.1836596e-4
        // t_A > t_B → c6 归 A。
        let centers = grid8();
        let gts = [[8.0f32, 0.0, 24.0, 16.0], [16.0, 8.0, 32.0, 24.0]];
        let pred = [
            [0., 0., 1., 1.],
            [8., 0., 24., 16.], // c1：A 内，IoU_A=1
            [8., 0., 24., 16.], // c2：A 内，IoU_A=1
            [0., 0., 1., 1.],
            [0., 0., 1., 1.],
            [12., 4., 20., 12.], // c5：A 内，IoU_A=0.25
            [16., 8., 24., 16.], // c6：A、B 内，IoU 均为 0.25
            [16., 8., 32., 24.], // c7：B 内，IoU_B=1
        ];
        let row_a = [0.0f32, 0.9, 0.8, 0.0, 0.0, 0.9, 0.9, 0.0];
        let row_b = [0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.95];
        let scores: &[&[f32]] = &[&row_a, &row_b];

        let out = assign_single_image(&pred, &centers, scores, &gts, &TalConfig::default());
        assert_eq!(out[1].unwrap().gt, 0);
        assert_eq!(out[2].unwrap().gt, 0);
        assert_eq!(out[5].unwrap().gt, 0);
        assert_eq!(out[6].unwrap().gt, 0, "t_A>t_B，争夺 cell 应归 A");
        assert_eq!(out[7].unwrap().gt, 1);
        for c in [0usize, 3, 4] {
            assert!(out[c].is_none());
        }
    }

    #[test]
    fn conflict_tie_breaks_to_lower_gt_index() {
        // c6 两边分数同为 0.8 → t 相同 → 取 gt 序号小者（A）
        let centers = grid8();
        let gts = [[8.0f32, 0.0, 24.0, 16.0], [16.0, 8.0, 32.0, 24.0]];
        let mut pred = [[0.0f32, 0., 1., 1.]; 8];
        pred[6] = [16., 8., 24., 16.];
        let row_a = [0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0];
        let row_b = [0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0];
        let scores: &[&[f32]] = &[&row_a, &row_b];
        let out = assign_single_image(&pred, &centers, scores, &gts, &TalConfig::default());
        assert_eq!(out[6].unwrap().gt, 0, "平局应归 gt 序号小者");
    }

    #[test]
    fn tiny_gt_falls_back_to_nearest_center_cell() {
        // gt [25,9,27,11]（2×2 极小框）不含任何 cell 中心，且预测框零重叠
        // → 回退：距 gt 中心 (26,10) 最近的 c7 (28,12)。
        let centers = grid8();
        let gt = [[25.0f32, 9.0, 27.0, 11.0]];
        let pred = [[0.0f32, 0.0, 1.0, 1.0]; 8];
        let scores: &[&[f32]] = &[&[0.5; 8]];
        let out = assign_single_image(&pred, &centers, scores, &gt, &TalConfig::default());
        let picked: Vec<usize> = out
            .iter()
            .enumerate()
            .filter(|(_, p)| p.is_some())
            .map(|(c, _)| c)
            .collect();
        assert_eq!(picked, vec![7], "应恰好回退到最近中心 cell c7");
        assert_eq!(out[7].unwrap().gt, 0);
    }

    #[test]
    fn empty_gts_yields_no_positives() {
        let centers = grid8();
        let pred = [[8.0f32, 0.0, 24.0, 16.0]; 8];
        let scores: &[&[f32]] = &[&[0.9; 8]];
        let out = assign_single_image(&pred, &centers, scores, &[], &TalConfig::default());
        assert!(out.iter().all(|p| p.is_none()));
    }
}
