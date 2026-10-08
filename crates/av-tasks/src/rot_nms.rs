//! 旋转 NMS 与角度工具（PLAN §4.2 OBB 后处理；纯标量，无 tch 依赖）。
//!
//! [`av_core::types::nms`] 只支持 Aabb（角度盲）：十字交叉放置的两个细长旋转框
//! （外接矩形几乎完全重叠、真实重叠很小）会被错误抑制。本模块按
//! [`av_core::geometry::RotBox::iou`]（Sutherland–Hodgman 凸多边形裁剪，av-core
//! 已有）做角度感知的贪心抑制，另提供高斯 Bhattacharyya（ProbIoU）度量供选择。
//!
//! 贪心流程与 av_core::nms 同款：分数降序 → 逐一与保留集合比重叠度 → 超阈值丢弃。
//! 与 av_core::nms 一致采用类别无关抑制（跨类比较），保持整库后处理语义统一。

use av_core::geometry::{Aabb, RotBox};
use av_core::types::Detection;

/// 旋转 NMS 的重叠度量。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotNmsMetric {
    /// 多边形 IoU（精确值，默认；Sutherland–Hodgman 裁剪）
    Polygon,
    /// 高斯 Bhattacharyya 系数（KFIoU 同族的 ProbIoU；阈值语义与多边形 IoU
    /// 不完全等价——取值偏「分布相似度」而非严格面积比，作近似档）
    ProbIou,
}

/// 旋转框 (w, h, θ) 的外接轴对齐半宽 / 半高（与 RotBox::corners 的包络一致）。
pub fn envelope_half_extents(w: f32, h: f32, theta: f32) -> (f32, f32) {
    let (sin, cos) = theta.sin_cos();
    let hw = (w * cos.abs() + h * sin.abs()) / 2.0;
    let hh = (w * sin.abs() + h * cos.abs()) / 2.0;
    (hw, hh)
}

/// Detection → RotBox：OBB 的 bbox 语义为旋转框参数 (cx, cy, w, h) 的轴对齐形式
/// （models::decode_level_obb 产出），配合 angle 精确重建；angle 缺失按 0
/// （退化为水平框，与普通 NMS 行为一致）。
fn det_rotbox(d: &Detection) -> RotBox {
    let (cx, cy) = d.bbox.center();
    RotBox {
        cx,
        cy,
        w: d.bbox.w(),
        h: d.bbox.h(),
        theta: d.angle.unwrap_or(0.0),
    }
}

/// 旋转框的轴对齐包络框（顶点包络，与 [`envelope_half_extents`] 一致）。
fn envelope_aabb(r: &RotBox) -> Aabb {
    let (hw, hh) = envelope_half_extents(r.w, r.h, r.theta);
    Aabb::new(r.cx - hw, r.cy - hh, r.cx + hw, r.cy + hh)
}

/// 角度感知贪心 NMS：分数降序保留，抑制与已保留框旋转重叠超阈值的候选。
///
/// - `iou_thr`：抑制阈值（Polygon 度量即普通 IoU 阈值；ProbIou 为近似档）；
/// - 类别无关（与 [`av_core::types::nms`] 语义一致）。
///
/// 保留集合的 RotBox / 包络框一次预计算（旧实现每对比较重建 RotBox）；Polygon
/// 度量先做包络框相交测试——包络不相交则多边形 IoU 必为 0，密集候选下把昂贵的
/// 多边形裁剪只在包络重叠的对上执行。
pub fn rotate_nms(mut dets: Vec<Detection>, iou_thr: f32, metric: RotNmsMetric) -> Vec<Detection> {
    dets.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut kept: Vec<(Detection, RotBox, Aabb)> = Vec::new();
    for d in dets {
        let rb = det_rotbox(&d);
        let env = envelope_aabb(&rb);
        let suppressed = kept.iter().any(|(_, rk, renv)| {
            if matches!(metric, RotNmsMetric::Polygon) && renv.intersection(&env).area() <= 0.0 {
                return false; // 包络不相交 → 多边形交面积 = 0 → 不抑制
            }
            let ov = match metric {
                RotNmsMetric::Polygon => rk.iou(&rb),
                RotNmsMetric::ProbIou => probiou_scalar(*rk, rb),
            };
            ov > iou_thr
        });
        if !suppressed {
            kept.push((d, rb, env));
        }
    }
    kept.into_iter().map(|(d, _, _)| d).collect()
}

/// 标量版 ProbIoU：两旋转框高斯拟合的 Bhattacharyya 系数 ∈ (0, 1]（相同框 → 1）。
/// 数学与 [`crate::kfiou::probiou_element`] 一致（张量版供损失、本版供后处理）。
pub fn probiou_scalar(a: RotBox, b: RotBox) -> f32 {
    fn cov(r: &RotBox) -> [f32; 3] {
        let (sin, cos) = r.theta.sin_cos();
        let m = r.w * r.w / 12.0;
        let n = r.h * r.h / 12.0;
        [
            m * cos * cos + n * sin * sin,
            (m - n) * sin * cos,
            m * sin * sin + n * cos * cos,
        ]
    }
    let [axx, axy, ayy] = cov(&a);
    let [bxx, bxy, byy] = cov(&b);
    let (dx, dy) = (b.cx - a.cx, b.cy - a.cy);
    let sxx = (axx + bxx) * 0.5;
    let sxy = (axy + bxy) * 0.5;
    let syy = (ayy + byy) * 0.5;
    let det_s = (sxx * syy - sxy * sxy).max(1e-12);
    let det_a = (axx * ayy - axy * axy).max(1e-12);
    let det_b = (bxx * byy - bxy * bxy).max(1e-12);
    let q = (syy * dx * dx + sxx * dy * dy - sxy * dx * dy * 2.0) / det_s;
    ((det_a * det_b).sqrt() / det_s).sqrt() * (-0.125 * q).exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use av_core::geometry::Aabb;
    use std::f32::consts::PI;

    /// 构造 OBB 解码形状的候选：bbox = 未旋转参数框 (cx±w/2, cy±h/2)，angle = Some(θ)。
    fn det(cx: f32, cy: f32, w: f32, h: f32, theta: f32, score: f32) -> Detection {
        Detection {
            bbox: Aabb::new(cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0),
            score,
            class_id: 0,
            angle: Some(theta),
            keypoints: None,
        }
    }

    /// 角度规范化往返：幂等 + 周期 π（矩形几何等价）+ 落在域窗口内。
    /// （OBB 解码 θ = normalize(tanh(tθ)·π/2) 依赖该性质，PLAN §4.0。）
    #[test]
    fn angle_normalize_roundtrip_and_period_pi() {
        for d in [
            av_core::conventions::AngleDomain::Le90,
            av_core::conventions::AngleDomain::Le135,
            av_core::conventions::AngleDomain::OpenCv,
        ] {
            for &t in &[-4.0f32, -2.8, -0.9, 0.0, 0.3, 1.2, 2.5] {
                let n1 = d.normalize(t);
                let n2 = d.normalize(n1); // 幂等（往返）
                assert!((n1 - n2).abs() < 1e-5, "{d:?}: {t} -> {n1} -> {n2}");
                let n3 = d.normalize(t + PI); // 周期 π：同一几何框
                assert!((n1 - n3).abs() < 1e-4, "{d:?}: period π {t}: {n1} vs {n3}");
                let (lo, hi) = d.range();
                assert!(n1 >= lo - 1e-4 && n1 < hi, "{d:?}: {t} -> {n1} 越界");
            }
        }
    }

    #[test]
    fn rotated_boxes_avoid_angle_blind_suppression() {
        // 斜向密集场景（旋转 NMS 的存在理由）：两条平行的 40×8 条带，θ=45°，
        // 沿条带法向错开整条短边（8px）——两者几何上零重叠，都必须保留。
        let s = std::f32::consts::FRAC_1_SQRT_2;
        let a = det(50.0, 50.0, 40.0, 8.0, std::f32::consts::FRAC_PI_4, 0.9);
        // b = a 沿法向 (−sin45°, cos45°) 平移 8px
        let b = det(
            50.0 - 8.0 * s,
            50.0 + 8.0 * s,
            40.0,
            8.0,
            std::f32::consts::FRAC_PI_4,
            0.7,
        );
        // 前置：旋转 IoU ≈ 0（法向间距 = 短边宽 → 交叠面积归零）
        let (ra, rb) = (det_rotbox(&a), det_rotbox(&b));
        let rot_iou = ra.iou(&rb);
        assert!(
            rot_iou < 0.01,
            "前置：平行错位条带旋转 IoU 应≈0，got {rot_iou}"
        );
        // 对照：同样的两个目标若按外接 Aabb 走角度盲 NMS（envelope IoU ≈ 0.53 > 0.5）
        // 会被错杀——这正是旋转 NMS 要修的回归
        let (hwa, hha) = envelope_half_extents(40.0, 8.0, std::f32::consts::FRAC_PI_4);
        let env_a = Aabb::new(50.0 - hwa, 50.0 - hha, 50.0 + hwa, 50.0 + hha);
        let (cxb, cyb) = b.bbox.center();
        let env_b = Aabb::new(cxb - hwa, cyb - hha, cxb + hwa, cyb + hha);
        assert!(env_a.iou(&env_b) > 0.5, "前置：外接框角度盲抑制会误杀");
        // 旋转 NMS：两者都保留
        let kept = rotate_nms(vec![a, b], 0.5, RotNmsMetric::Polygon);
        assert_eq!(kept.len(), 2, "平行错位旋转框不得被抑制");
        assert!((kept[0].score - 0.9).abs() < 1e-6);
        assert!((kept[1].score - 0.7).abs() < 1e-6);
    }

    #[test]
    fn cross_placed_rotated_boxes_not_suppressed() {
        // 十字交叉：40×8 横条 vs 竖条（同一 w/h、θ=90°）同中心。
        // 旋转 IoU = 64/576 ≈ 0.111 < 0.5 → 两者都保留。
        let a = det(50.0, 50.0, 40.0, 8.0, 0.0, 0.9);
        let b = det(50.0, 50.0, 40.0, 8.0, std::f32::consts::FRAC_PI_2, 0.7);
        let kept = rotate_nms(vec![a, b], 0.5, RotNmsMetric::Polygon);
        assert_eq!(kept.len(), 2, "交叉放置的旋转框不得被抑制");
        assert!((kept[0].score - 0.9).abs() < 1e-6);
        assert!((kept[1].score - 0.7).abs() < 1e-6);
    }

    #[test]
    fn rotated_duplicate_is_suppressed() {
        // 同角度近重复（1px 平移，IoU ≈ 39/41 ≈ 0.95）→ 抑制；远处框保留
        let a = det(50.0, 50.0, 40.0, 8.0, 0.0, 0.9);
        let dup = det(51.0, 50.0, 40.0, 8.0, 0.0, 0.8);
        let far = det(200.0, 200.0, 40.0, 8.0, 0.0, 0.5);
        let kept = rotate_nms(vec![a, dup, far], 0.5, RotNmsMetric::Polygon);
        assert_eq!(kept.len(), 2);
        assert!((kept[0].score - 0.9).abs() < 1e-6);
        assert!((kept[1].score - 0.5).abs() < 1e-6);
    }

    #[test]
    fn angle_none_degenerates_to_plain_nms() {
        // angle = None → 按水平框处理，结果应与 av_core::nms 完全一致
        let base = |x1: f32, score: f32| Detection {
            bbox: Aabb::new(x1, 0.0, x1 + 10.0, 10.0),
            score,
            class_id: 0,
            angle: None,
            keypoints: None,
        };
        let dets = vec![base(0.0, 0.7), base(1.0, 0.9), base(50.0, 0.5)];
        let got = rotate_nms(dets.clone(), 0.5, RotNmsMetric::Polygon);
        let want = av_core::types::nms(dets, 0.5);
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(&want) {
            assert_eq!(g.score.to_bits(), w.score.to_bits());
        }
    }

    #[test]
    fn probiou_metric_hand_check_and_nms() {
        // 手算：a = (0,0,40,8,0)，b = (0,0,8,40,0)（十字交叉，同中心）
        //   Σa = diag(1600,64)/12，Σb = diag(64,1600)/12，det 均为 711.111
        //   Σ* = diag(69.3333,69.3333)，det* = 4807.11，q = 0
        //   BC = √(711.111/4807.11) ≈ 0.384616 < 0.5 → 交叉框不互抑
        let a = RotBox {
            cx: 0.0,
            cy: 0.0,
            w: 40.0,
            h: 8.0,
            theta: 0.0,
        };
        let b = RotBox {
            cx: 0.0,
            cy: 0.0,
            w: 8.0,
            h: 40.0,
            theta: 0.0,
        };
        let bc_cross = probiou_scalar(a, b);
        assert!((bc_cross - 0.384616).abs() < 1e-3, "BC={bc_cross}");
        // 1px 平移的近重复：q = dx²·(h²/12)⁻¹·… = 12/1600 → BC ≈ 0.9991 > 0.99
        let dup = RotBox {
            cx: 1.0,
            cy: 0.0,
            w: 40.0,
            h: 8.0,
            theta: 0.0,
        };
        let bc_dup = probiou_scalar(a, dup);
        assert!(bc_dup > 0.99, "BC dup={bc_dup}");
        assert!(probiou_scalar(a, a) > 0.9999);

        // NMS（ProbIou 度量）：同向近重复抑制；法向错开整条短边的平行条带保留
        // （q = 8²·12/8² = 12 → BC = exp(−1.5) ≈ 0.223 < 0.5）
        let s = std::f32::consts::FRAC_1_SQRT_2;
        let ka = det(50.0, 50.0, 40.0, 8.0, std::f32::consts::FRAC_PI_4, 0.9);
        let kdup = det(
            50.0 + s,
            50.0 + s,
            40.0,
            8.0,
            std::f32::consts::FRAC_PI_4,
            0.8,
        );
        let kb = det(
            50.0 - 8.0 * s,
            50.0 + 8.0 * s,
            40.0,
            8.0,
            std::f32::consts::FRAC_PI_4,
            0.7,
        );
        let kept = rotate_nms(vec![ka, kdup, kb], 0.5, RotNmsMetric::ProbIou);
        assert_eq!(kept.len(), 2, "ProbIou 度量：重复框抑制、错位条带保留");
    }

    #[test]
    fn envelope_matches_rotbox_corners() {
        // 外接半宽高 = 顶点包络（旋转 30°（π/6）的 10×6 框逐顶点验证）
        let (w, h, th) = (10.0f32, 6.0f32, std::f32::consts::FRAC_PI_6);
        let rb = RotBox {
            cx: 0.0,
            cy: 0.0,
            w,
            h,
            theta: th,
        };
        let mut xmax = 0f32;
        let mut ymax = 0f32;
        for c in rb.corners() {
            xmax = xmax.max(c[0].abs());
            ymax = ymax.max(c[1].abs());
        }
        let (hw, hh) = envelope_half_extents(w, h, th);
        assert!((hw - xmax).abs() < 1e-5);
        assert!((hh - ymax).abs() < 1e-5);
    }
}
