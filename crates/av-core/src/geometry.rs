//! 纯几何工具：轴对齐框、旋转框（Sutherland–Hodgman 旋转 IoU）、letterbox 映射。
//!
//! 无张量依赖，全部可单元测试（PLAN §9 的数值回归基础）。

use serde::{Deserialize, Serialize};

/// 轴对齐边界框，绝对像素 xyxy（PLAN §4.0 约定）。
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Aabb {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

impl Aabb {
    pub fn new(x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
        Self { x1, y1, x2, y2 }
    }

    pub fn from_xywh(x: f32, y: f32, w: f32, h: f32) -> Self {
        Self {
            x1: x,
            y1: y,
            x2: x + w,
            y2: y + h,
        }
    }

    pub fn w(&self) -> f32 {
        (self.x2 - self.x1).max(0.0)
    }

    pub fn h(&self) -> f32 {
        (self.y2 - self.y1).max(0.0)
    }

    pub fn area(&self) -> f32 {
        self.w() * self.h()
    }

    pub fn center(&self) -> (f32, f32) {
        ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)
    }

    /// 头输出侧 cxcywh 表示（PLAN §4.0）。
    pub fn to_xywh(&self) -> [f32; 4] {
        let (cx, cy) = self.center();
        [cx, cy, self.w(), self.h()]
    }

    pub fn intersection(&self, o: &Aabb) -> Aabb {
        Aabb::new(
            self.x1.max(o.x1),
            self.y1.max(o.y1),
            self.x2.min(o.x2),
            self.y2.min(o.y2),
        )
    }

    pub fn iou(&self, o: &Aabb) -> f32 {
        let inter = self.intersection(o).area();
        let union = self.area() + o.area() - inter;
        if union <= 0.0 {
            0.0
        } else {
            inter / union
        }
    }
}

/// 旋转框 (cx, cy, w, h, θ)；θ 语义由 [`crate::conventions::AngleDomain`] 规范化。
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RotBox {
    pub cx: f32,
    pub cy: f32,
    pub w: f32,
    pub h: f32,
    pub theta: f32,
}

impl RotBox {
    /// 四角点，按逆时针序（数学坐标，y 向上）。
    pub fn corners(&self) -> [[f32; 2]; 4] {
        let (sin, cos) = self.theta.sin_cos();
        let (hw, hh) = (self.w / 2.0, self.h / 2.0);
        [
            [self.cx + hw * cos - hh * sin, self.cy + hw * sin + hh * cos],
            [self.cx - hw * cos - hh * sin, self.cy - hw * sin + hh * cos],
            [self.cx - hw * cos + hh * sin, self.cy - hw * sin - hh * cos],
            [self.cx + hw * cos + hh * sin, self.cy + hw * sin - hh * cos],
        ]
    }

    /// 旋转 IoU：凸多边形裁剪求交面积。
    ///
    /// 面积直接用 w·h（旋转矩形，省两次鞋带公式）；裁剪在栈上完成
    /// （两四边形凸交顶点 ≤ 8，零堆分配——旋转 NMS 每图要做 O(n²) 次本函数）。
    pub fn iou(&self, o: &RotBox) -> f32 {
        let a = self.corners();
        let b = o.corners();
        let inter = quad_intersection_area(&a, &b);
        let union = self.w * self.h + o.w * o.h - inter;
        if union <= 1e-12 {
            0.0
        } else {
            inter / union
        }
    }
}

/// 鞋带公式面积（顶点按序，方向不限）。
pub fn polygon_area(pts: &[[f32; 2]]) -> f32 {
    let n = pts.len();
    let mut s = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        s += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1];
    }
    (s / 2.0).abs()
}

fn cross(o: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
}

fn line_intersect(a: [f32; 2], b: [f32; 2], c: [f32; 2], d: [f32; 2]) -> [f32; 2] {
    let ab = [b[0] - a[0], b[1] - a[1]];
    let cd = [d[0] - c[0], d[1] - c[1]];
    let denom = ab[0] * cd[1] - ab[1] * cd[0];
    if denom.abs() < 1e-12 {
        return c; // 平行退化
    }
    let t = ((c[0] - a[0]) * cd[1] - (c[1] - a[1]) * cd[0]) / denom;
    [a[0] + t * ab[0], a[1] + t * ab[1]]
}

/// Sutherland–Hodgman 凸多边形裁剪：返回交多边形面积（顶点须为逆时针/顺时针一致序）。
///
/// 通用入口（任意顶点数）。`RotBox::iou` 走 [`quad_intersection_area`] 栈上特化路径。
pub fn convex_polygon_intersection_area(subject: &[[f32; 2]], clip: &[[f32; 2]]) -> f32 {
    let cap = subject.len() + clip.len();
    let mut output: Vec<[f32; 2]> = Vec::with_capacity(cap);
    let mut scratch: Vec<[f32; 2]> = Vec::with_capacity(cap);
    output.extend_from_slice(subject);
    let n = clip.len();
    for i in 0..n {
        if output.is_empty() {
            return 0.0;
        }
        let a = clip[i];
        let b = clip[(i + 1) % n];
        // 双缓冲交换：两块缓冲容量全程复用（旧实现 mem::take 后从零扩容，
        // 每条裁剪边都触发一次 realloc）
        let input = std::mem::take(&mut output);
        scratch.clear();
        let m = input.len();
        for j in 0..m {
            let cur = input[j];
            let nxt = input[(j + 1) % m];
            let cur_in = cross(a, b, cur) >= 0.0;
            let nxt_in = cross(a, b, nxt) >= 0.0;
            if cur_in {
                scratch.push(cur);
                if !nxt_in {
                    scratch.push(line_intersect(a, b, cur, nxt));
                }
            } else if nxt_in {
                scratch.push(line_intersect(a, b, cur, nxt));
            }
        }
        output = std::mem::take(&mut scratch);
        scratch = input;
        scratch.clear();
    }
    if output.len() < 3 {
        0.0
    } else {
        polygon_area(&output)
    }
}

/// 四边形 × 四边形裁剪特化：交多边形顶点 ≤ 8（每条裁剪边至多 +1 顶点），
/// 全程栈上定长数组，零堆分配。
fn quad_intersection_area(subject: &[[f32; 2]; 4], clip: &[[f32; 2]; 4]) -> f32 {
    let mut out = [[0.0f32; 2]; 8];
    out[..4].copy_from_slice(subject);
    let mut out_len = 4usize;
    let mut tmp = [[0.0f32; 2]; 8];
    for i in 0..clip.len() {
        if out_len == 0 {
            return 0.0;
        }
        let a = clip[i];
        let b = clip[(i + 1) % clip.len()];
        let mut tmp_len = 0usize;
        for j in 0..out_len {
            let cur = out[j];
            let nxt = out[(j + 1) % out_len];
            let cur_in = cross(a, b, cur) >= 0.0;
            let nxt_in = cross(a, b, nxt) >= 0.0;
            if cur_in {
                tmp[tmp_len] = cur;
                tmp_len += 1;
                if !nxt_in {
                    tmp[tmp_len] = line_intersect(a, b, cur, nxt);
                    tmp_len += 1;
                }
            } else if nxt_in {
                tmp[tmp_len] = line_intersect(a, b, cur, nxt);
                tmp_len += 1;
            }
        }
        std::mem::swap(&mut out, &mut tmp);
        out_len = tmp_len;
    }
    if out_len < 3 {
        0.0
    } else {
        polygon_area(&out[..out_len])
    }
}

/// letterbox 映射参数（PLAN §5.1：pad 元数据随 `ImageMeta` 走，后处理还原坐标）。
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Letterbox {
    pub scale: f32,
    pub pad_left: f32,
    pub pad_top: f32,
    pub dst_w: u32,
    pub dst_h: u32,
}

/// 计算 letterbox：等比缩放到 target 内切，画布对齐到 align 的倍数，居中补边。
pub fn letterbox(orig_w: u32, orig_h: u32, target: u32, align: u32) -> Letterbox {
    assert!(orig_w > 0 && orig_h > 0 && target > 0 && align > 0);
    let scale = (target as f32 / orig_w as f32).min(target as f32 / orig_h as f32);
    let new_w = ((orig_w as f32 * scale).round() as u32).max(1);
    let new_h = ((orig_h as f32 * scale).round() as u32).max(1);
    let dst_w = new_w.div_ceil(align) * align;
    let dst_h = new_h.div_ceil(align) * align;
    Letterbox {
        scale,
        pad_left: (dst_w - new_w) as f32 / 2.0,
        pad_top: (dst_h - new_h) as f32 / 2.0,
        dst_w,
        dst_h,
    }
}

impl Letterbox {
    /// 原图坐标 → 画布坐标。
    pub fn map_box(&self, b: Aabb) -> Aabb {
        Aabb::new(
            b.x1 * self.scale + self.pad_left,
            b.y1 * self.scale + self.pad_top,
            b.x2 * self.scale + self.pad_left,
            b.y2 * self.scale + self.pad_top,
        )
    }

    /// 画布坐标 → 原图坐标（裁剪回原图范围）。
    pub fn restore_box(&self, b: Aabb, orig_w: u32, orig_h: u32) -> Aabb {
        let inv = if self.scale > 0.0 {
            1.0 / self.scale
        } else {
            0.0
        };
        Aabb::new(
            ((b.x1 - self.pad_left) * inv).clamp(0.0, orig_w as f32),
            ((b.y1 - self.pad_top) * inv).clamp(0.0, orig_h as f32),
            ((b.x2 - self.pad_left) * inv).clamp(0.0, orig_w as f32),
            ((b.y2 - self.pad_top) * inv).clamp(0.0, orig_h as f32),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::{FRAC_PI_2, PI};

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn aabb_iou_one_third_offset() {
        let a = Aabb::new(0.0, 0.0, 10.0, 10.0);
        let b = Aabb::new(5.0, 0.0, 15.0, 10.0);
        assert!(close(a.iou(&b), 1.0 / 3.0));
    }

    #[test]
    fn rot_iou_matches_aabb_at_zero_angle() {
        let a = RotBox {
            cx: 5.0,
            cy: 5.0,
            w: 10.0,
            h: 10.0,
            theta: 0.0,
        };
        let b = RotBox {
            cx: 10.0,
            cy: 5.0,
            w: 10.0,
            h: 10.0,
            theta: 0.0,
        };
        assert!(close(a.iou(&b), 1.0 / 3.0));
    }

    #[test]
    fn rot_iou_identical_is_one() {
        let a = RotBox {
            cx: 3.0,
            cy: 7.0,
            w: 8.0,
            h: 5.0,
            theta: 0.7,
        };
        assert!(close(a.iou(&a), 1.0));
    }

    #[test]
    fn rot_iou_quarter_turn_with_wh_swap_is_one() {
        let a = RotBox {
            cx: 5.0,
            cy: 5.0,
            w: 10.0,
            h: 5.0,
            theta: 0.0,
        };
        let b = RotBox {
            cx: 5.0,
            cy: 5.0,
            w: 5.0,
            h: 10.0,
            theta: FRAC_PI_2,
        };
        assert!(close(a.iou(&b), 1.0));
    }

    #[test]
    fn rot_iou_period_pi() {
        let a = RotBox {
            cx: 0.0,
            cy: 0.0,
            w: 4.0,
            h: 3.0,
            theta: 0.2,
        };
        let b = RotBox {
            cx: 0.0,
            cy: 0.0,
            w: 4.0,
            h: 3.0,
            theta: 0.2 + PI,
        };
        assert!(close(a.iou(&b), 1.0));
    }

    #[test]
    fn disjoint_is_zero() {
        let a = RotBox {
            cx: 0.0,
            cy: 0.0,
            w: 2.0,
            h: 2.0,
            theta: 0.3,
        };
        let b = RotBox {
            cx: 100.0,
            cy: 100.0,
            w: 2.0,
            h: 2.0,
            theta: 1.1,
        };
        assert!(a.iou(&b) < 1e-6);
    }

    #[test]
    fn letterbox_roundtrip() {
        let lb = letterbox(1920, 1080, 640, crate::conventions::STRIDE_ALIGN);
        assert_eq!(lb.dst_w, 640);
        assert_eq!(lb.dst_h, 384);
        let b = Aabb::new(100.0, 100.0, 300.0, 200.0);
        let restored = lb.restore_box(lb.map_box(b), 1920, 1080);
        assert!(
            close(restored.x1, b.x1)
                && close(restored.y1, b.y1)
                && close(restored.x2, b.x2)
                && close(restored.y2, b.y2),
            "restored={restored:?}"
        );
    }
}
