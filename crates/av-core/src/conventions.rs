//! 数据约定（PLAN §4.0：全框架统一约定，固化后由 golden 测试锁定）。

use serde::{Deserialize, Serialize};
use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};

/// ImageNet 归一化常数（RGB、[0,1] 线性域下的 mean/std）。
pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// 输入对齐的公共 stride（batch 内补齐画布按此对齐）。
pub const STRIDE_ALIGN: u32 = 32;

pub fn deg2rad(d: f32) -> f32 {
    d * PI / 180.0
}

pub fn rad2deg(r: f32) -> f32 {
    r * 180.0 / PI
}

/// 旋转框角度域（PLAN §4.0：进 KFIoU / 旋转 NMS 前必须先规范化到统一域）。
///
/// 注：原方案写「le90，[-π/2, 0)」，该区间实为 OpenCV 域且对固定 (w, h) 不闭合
/// （矩形角度真实周期是 π）。实现期已修正：le90 = [-π/2, π/2)（mmrotate 定义）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AngleDomain {
    /// long-edge-90：[-π/2, π/2)，width 定义为长边（默认）
    #[default]
    Le90,
    /// long-edge-135：[-π/4, 3π/4)
    Le135,
    /// OpenCV 风格：[-π/2, π/2)，width 定义为短边
    OpenCv,
}

impl AngleDomain {
    /// 把任意角度按周期 π 规范化进本域窗口。
    pub fn normalize(self, theta: f32) -> f32 {
        match self {
            Self::Le90 | Self::OpenCv => normalize_window(theta, -FRAC_PI_2),
            Self::Le135 => normalize_window(theta, -FRAC_PI_4),
        }
    }

    /// 本域窗口 [lo, hi)。
    pub fn range(self) -> (f32, f32) {
        match self {
            Self::Le90 | Self::OpenCv => (-FRAC_PI_2, FRAC_PI_2),
            Self::Le135 => (-FRAC_PI_4, 3.0 * FRAC_PI_4),
        }
    }
}

/// 折叠 θ 到 [start, start + π)。
fn normalize_window(theta: f32, start: f32) -> f32 {
    let mut t = (theta - start).rem_euclid(PI) + start;
    // 浮点边界保护：rem_euclid 在极端输入下可能恰好返回模长 π
    if t - start >= PI {
        t -= PI;
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    #[test]
    fn le90_wraps_quarter_turn() {
        assert!(close(AngleDomain::Le90.normalize(FRAC_PI_2), -FRAC_PI_2));
        assert!(close(AngleDomain::Le90.normalize(-PI), 0.0));
    }

    #[test]
    fn le90_keeps_in_domain_values() {
        for t in [0.0f32, -1.0, 1.0, -FRAC_PI_2] {
            let n = AngleDomain::Le90.normalize(t);
            assert!(close(n, t), "t={t} -> n={n}");
        }
    }

    #[test]
    fn normalize_respects_domain_window() {
        for d in [AngleDomain::Le90, AngleDomain::Le135, AngleDomain::OpenCv] {
            for t in [-3.0f32, -1.57, 0.0, 1.0, 2.5, 5.0, 10.0] {
                let n = d.normalize(t);
                let (lo, hi) = d.range();
                assert!(n >= lo - 1e-5 && n < hi + 1e-5, "{d:?} t={t} -> {n}");
            }
        }
    }

    #[test]
    fn radian_degree_roundtrip() {
        assert!(close(rad2deg(deg2rad(37.5)), 37.5));
    }
}
