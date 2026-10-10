//! serde 化数据结构与推理产物（字段对齐 COCO/DOTA 习惯，PLAN §3）。

use serde::{Deserialize, Serialize};

use crate::geometry::{Aabb, Letterbox, RotBox};

/// 单个关键点 [x, y, v]（v 为可见性标志）。
pub type Kp3 = [f32; 3];

/// 预处理元数据：后处理把画布坐标还原回原图坐标的依据（PLAN §5.1）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageMeta {
    pub orig_w: u32,
    pub orig_h: u32,
    pub letterbox: Option<Letterbox>,
    pub path: Option<String>,
}

/// 统一检测产物：普通框；OBB 附带 `angle`；关键点任务附带 `keypoints`（可叠加）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Detection {
    pub bbox: Aabb,
    pub score: f32,
    pub class_id: u32,
    pub angle: Option<f32>,
    pub keypoints: Option<Vec<Kp3>>,
}

impl Detection {
    /// 依据 letterbox 元数据把框与关键点还原回原图坐标系。
    pub fn restore(&mut self, lb: &Letterbox) {
        self.bbox = lb.restore_box(self.bbox, u32::MAX, u32::MAX);
        if let Some(kps) = &mut self.keypoints {
            for kp in kps.iter_mut() {
                kp[0] = (kp[0] - lb.pad_left) / lb.scale;
                kp[1] = (kp[1] - lb.pad_top) / lb.scale;
            }
        }
    }
}

/// 检测任务标签。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DetTarget {
    pub boxes: Vec<Aabb>,
    pub labels: Vec<u32>,
}

/// OBB 任务标签。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ObbTarget {
    pub boxes: Vec<RotBox>,
    pub labels: Vec<u32>,
}

/// 关键点任务标签。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KpTarget {
    pub boxes: Vec<Aabb>,
    pub keypoints: Vec<Vec<Kp3>>,
    pub labels: Vec<u32>,
}

/// 贪心 NMS + 分数加权融合（Ultralytics `non_max_suppression` 同语义：按分数
/// 降序、仅同类互相抑制，类间从不互斥），被抑制框按分数加权**融进**保留框
/// 而非丢弃——小目标上同一物体的多个偏移预测（IoU 低于阈值的不完全重叠
/// 重复框）融合后框更紧、重复 FP 消失；无重叠时与普通 NMS 行为一致。
pub fn nms(mut dets: Vec<Detection>, iou_thr: f32) -> Vec<Detection> {
    dets.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut kept: Vec<Detection> = Vec::new();
    'candidate: for d in dets {
        for k in &mut kept {
            if k.class_id == d.class_id && k.bbox.iou(&d.bbox) > iou_thr {
                // 分数加权融合（只融合几何；score/class 沿用保留框）
                let w = k.score / (k.score + d.score).max(1e-9);
                let m = |a: f32, b: f32| a * w + b * (1.0 - w);
                k.bbox = Aabb::new(
                    m(k.bbox.x1, d.bbox.x1),
                    m(k.bbox.y1, d.bbox.y1),
                    m(k.bbox.x2, d.bbox.x2),
                    m(k.bbox.y2, d.bbox.y2),
                );
                continue 'candidate;
            }
        }
        kept.push(d);
    }
    kept
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::letterbox;

    #[test]
    fn detection_restore_maps_back() {
        let lb = letterbox(1000, 1000, 500, 32);
        // 关键点先做正向映射（模拟预处理），再断言还原回原图坐标
        let mapped_kp = [
            80.0 * lb.scale + lb.pad_left,
            90.0 * lb.scale + lb.pad_top,
            2.0,
        ];
        let mut det = Detection {
            bbox: lb.map_box(Aabb::new(50.0, 60.0, 150.0, 160.0)),
            score: 0.9,
            class_id: 1,
            angle: None,
            keypoints: Some(vec![mapped_kp]),
        };
        det.restore(&lb);
        assert!((det.bbox.x1 - 50.0).abs() < 1e-4);
        assert!((det.bbox.y1 - 60.0).abs() < 1e-4);
        let kp = det.keypoints.unwrap()[0];
        assert!((kp[0] - 80.0).abs() < 1e-3, "kp={kp:?}");
        assert!((kp[1] - 90.0).abs() < 1e-3, "kp={kp:?}");
    }

    #[test]
    fn nms_suppresses_overlapping_keeps_best() {
        let base = |x1: f32, score: f32| Detection {
            bbox: Aabb::new(x1, 0.0, x1 + 10.0, 10.0),
            score,
            class_id: 0,
            angle: None,
            keypoints: None,
        };
        let dets = vec![base(0.0, 0.7), base(1.0, 0.9), base(50.0, 0.5)];
        let kept = nms(dets, 0.5);
        assert_eq!(kept.len(), 2);
        assert!((kept[0].score - 0.9).abs() < 1e-6);
        assert!((kept[1].score - 0.5).abs() < 1e-6);
    }

    /// 类感知：同位置不同类别的框不互相抑制（Ultralytics max_wh 偏移等价），
    /// 同类高重叠才被抑制。
    #[test]
    fn nms_is_class_aware() {
        let base = |x1: f32, score: f32, class_id: u32| Detection {
            bbox: Aabb::new(x1, 0.0, x1 + 10.0, 10.0),
            score,
            class_id,
            angle: None,
            keypoints: None,
        };
        // c0 高分框与 c1 高分框完全重叠 → 都保留；c0 低分重复框被同类抑制
        let dets = vec![
            base(0.0, 0.9, 0),
            base(0.0, 0.8, 1),
            base(1.0, 0.7, 0),
        ];
        let kept = nms(dets, 0.5);
        assert_eq!(kept.len(), 2);
        assert_eq!(kept[0].class_id, 0);
        assert_eq!(kept[1].class_id, 1);
    }
}
