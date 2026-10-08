//! coco8-seg（YOLO-seg 目录格式）数据解码：letterbox 预处理 + 多边形栅格化。
//!
//! 按 `crates/av-runtime/src/dataset.rs` 的语义自实现（本 crate 禁止依赖
//! av-runtime/av-tasks）：
//!
//! - [`decode_letterbox_chw`] ≙ `decode_rgb_with_meta(Letterbox, imagenet_norm=false)`：
//!   等比缩放（Triangle 滤镜）+ 114/255 灰居中补边 → CHW f32 [0,1]；
//!   letterbox 几何（scale/pad）直接复用 `av_core::geometry::letterbox`。
//! - [`rasterize_polygon`] ≙ 同名函数：偶奇扫描线填充，像素中心 (x+0.5, y+0.5)
//!   采样约定（可手算对照）。
//! - [`load_cocoseg_dir`] ≙ `load_cocoseg_dir`：标注行 `cls x1 y1 x2 y2 ...`，
//!   **恰 5 值的纯检测框行跳过**；多边形坐标经 letterbox 映射后栅格化到
//!   img/4 × img/4 掩码画布；栅格化为空的退化实例整条跳过。

use std::path::{Path, PathBuf};

use av_core::error::{AvError, AvResult};
use av_core::geometry::letterbox;

/// 单图分割样本（预解码，标量存储；堆批由调用方转 burn 张量）。
#[derive(Clone)]
pub struct SegImageSample {
    /// letterbox 画布像素，CHW、RGB、[0,1]，长度 3·S·S。
    pub pixels: Vec<f32>,
    /// 每实例 0/1 掩码（img/4 画布，行优先扁平）。
    pub masks: Vec<Vec<u8>>,
    /// 每实例类别（与 masks 一一对应）。
    pub labels: Vec<u32>,
    /// 输入画布边长 S。
    pub img_size: u32,
}

/// 多边形 → 二值掩码（纯 Rust 偶奇扫描线填充，逐行求边与扫描线的交点）。
///
/// 覆盖约定：像素 (x, y) 由其中心 (x+0.5, y+0.5) 是否落在多边形内决定
/// （与 OpenCV/Matplotlib 的「像素中心」采样约定一致，可手算对照）。
/// 坐标可为任意浮点（画布外部分自动被裁掉）；点数 < 3 返回全零。
pub fn rasterize_polygon(points: &[[f32; 2]], w: usize, h: usize) -> Vec<u8> {
    let mut mask = vec![0u8; w * h];
    let n = points.len();
    if n < 3 || w == 0 || h == 0 {
        return mask;
    }
    let min_y = points.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min);
    let max_y = points
        .iter()
        .map(|p| p[1])
        .fold(f32::NEG_INFINITY, f32::max);
    for y in 0..h {
        let cy = y as f32 + 0.5;
        if cy < min_y || cy > max_y {
            continue;
        }
        // 收集所有边与扫描线 y=cy 的交点 x（偶奇规则，容忍自交多边形）
        let mut xs: Vec<f32> = Vec::new();
        let mut j = n - 1;
        for i in 0..n {
            let (p, q) = (points[i], points[j]);
            if (p[1] <= cy && q[1] > cy) || (q[1] <= cy && p[1] > cy) {
                let t = (cy - p[1]) / (q[1] - p[1]);
                xs.push(p[0] + t * (q[0] - p[0]));
            }
            j = i;
        }
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        for pair in xs.chunks(2) {
            if pair.len() < 2 {
                continue;
            }
            // 中心 ∈ [x_enter, x_exit) 的像素被覆盖：x ∈ [ceil(x_enter-0.5), ceil(x_exit-0.5))
            let x0 = ((pair[0] - 0.5).ceil() as isize).max(0);
            let x1 = ((pair[1] - 0.5).ceil() as isize).min(w as isize);
            for x in x0..x1 {
                mask[y * w + x as usize] = 1;
            }
        }
    }
    mask
}

/// letterbox 预解码：等比缩放 + 114 灰补边 → CHW f32 [0,1]（RGB）。
/// 返回 (像素，letterbox 元数据)（元数据供坐标映射与测试对照）。
pub fn decode_letterbox_chw(
    rgb: &image::RgbImage,
    img_size: u32,
) -> (Vec<f32>, av_core::geometry::Letterbox) {
    let lb = letterbox(rgb.width(), rgb.height(), img_size, img_size);
    let nw = ((rgb.width() as f32 * lb.scale).round() as u32).clamp(1, img_size);
    let nh = ((rgb.height() as f32 * lb.scale).round() as u32).clamp(1, img_size);
    let resized = image::imageops::resize(rgb, nw, nh, image::imageops::FilterType::Triangle);
    let mut canvas = image::RgbImage::from_pixel(img_size, img_size, image::Rgb([114, 114, 114]));
    // 粘贴偏移取整（与 av-runtime rgb_to_input_tensor 同款）。
    image::imageops::overlay(
        &mut canvas,
        &resized,
        lb.pad_left.round() as i64,
        lb.pad_top.round() as i64,
    );
    let n = (img_size * img_size) as usize;
    let mut pixels = Vec::with_capacity(3 * n);
    // CHW 通道优先展开，[0,1]（对应 imagenet_norm=false 的历史语义）。
    for ch in 0..3 {
        for p in canvas.pixels() {
            pixels.push(p[ch] as f32 / 255.0);
        }
    }
    (pixels, lb)
}

/// 加载 COCO 分割格式数据集（Ultralytics coco8-seg 目录布局：
/// `images/<split>` + `labels/<split>/*.txt`）。
///
/// 标注行为 `cls x1 y1 x2 y2 ... xn yn`（归一化多边形，n >= 3）；**恰 5 个值
/// 的纯检测框行被跳过**。多边形点经 letterbox 等比映射到画布后，栅格化到
/// img/4 × img/4 的低分辨率掩码；栅格化后为空的实例（退化标注）整条跳过。
pub fn load_cocoseg_dir(root: &Path, split: &str, img_size: u32) -> AvResult<Vec<SegImageSample>> {
    let img_dir = root.join("images").join(split);
    let lbl_dir = root.join("labels").join(split);
    if !img_dir.is_dir() {
        return Err(AvError::data(format!(
            "数据集图片目录不存在: {}",
            img_dir.display()
        )));
    }
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&img_dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            matches!(
                p.extension().and_then(|e| e.to_str()),
                Some("jpg") | Some("jpeg") | Some("png") | Some("bmp")
            )
        })
        .collect();
    paths.sort();
    if paths.is_empty() {
        return Err(AvError::data(format!(
            "数据集图片目录为空: {}",
            img_dir.display()
        )));
    }

    let mw = (img_size / 4) as usize;
    let mh = (img_size / 4) as usize;
    let k = mw as f32 / img_size as f32; // 画布 → 掩码画布 缩放
    let mut out = Vec::new();
    for p in paths {
        let stem = p
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| AvError::data("文件名非法"))?
            .to_string();
        let lbl_path = lbl_dir.join(format!("{stem}.txt"));
        let img = image::open(&p).map_err(|e| AvError::data(format!("读图失败 {p:?}: {e}")))?;
        let rgb = img.to_rgb8();
        let (ow, oh) = (rgb.width() as f32, rgb.height() as f32);
        let (pixels, lb) = decode_letterbox_chw(&rgb, img_size);

        let mut masks = Vec::new();
        let mut labels = Vec::new();
        if lbl_path.exists() {
            for line in std::fs::read_to_string(&lbl_path)?.lines() {
                let vals: Vec<f32> = line
                    .split_whitespace()
                    .filter_map(|t| t.parse().ok())
                    .collect();
                // 多边形 = 1 类别 + 2n 坐标，n >= 3 → 至少 7 个值；5 值行为纯检测框，跳过
                if vals.len() < 7 {
                    continue;
                }
                let n_pts = (vals.len() - 1) / 2;
                // 归一化 → 原图像素 → letterbox 画布 → 掩码画布（×k）
                let pts: Vec<[f32; 2]> = (0..n_pts)
                    .map(|i| {
                        [
                            (vals[1 + 2 * i] * ow * lb.scale + lb.pad_left) * k,
                            (vals[2 + 2 * i] * oh * lb.scale + lb.pad_top) * k,
                        ]
                    })
                    .collect();
                let mask = rasterize_polygon(&pts, mw, mh);
                if mask.iter().all(|&v| v == 0) {
                    continue; // 退化标注（画布外/面积 0）整条跳过
                }
                masks.push(mask);
                labels.push(vals[0] as u32);
            }
        }
        out.push(SegImageSample {
            pixels,
            masks,
            labels,
            img_size,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 矩形多边形栅格化手算对照：[1,1]-[4,4] 方形 → 像素中心采样覆盖
    /// x∈[1,3]、y∈[1,3]（中心 1.5/2.5/3.5 落入，开区间右端除外）。
    #[test]
    fn rasterize_polygon_rectangle_hand_computed() {
        let pts = [[1.0, 1.0], [4.0, 1.0], [4.0, 4.0], [1.0, 4.0]];
        let mask = rasterize_polygon(&pts, 5, 5);
        let expect = [
            0u8, 0, 0, 0, 0, //
            0, 1, 1, 1, 0, //
            0, 1, 1, 1, 0, //
            0, 1, 1, 1, 0, //
            0, 0, 0, 0, 0,
        ];
        assert_eq!(mask, expect);
    }

    /// 三角形手算对照：顶点 (1,1),(3,1),(2,4)。扫描线交点：
    /// y=1.5 → x∈[1.167,2.833) 覆盖 x=1,2；y=2.5 → x∈[1.5,2.5) 覆盖 x=1；
    /// y=3.5 → x∈[1.833,2.167) 中心采样为空（宽 <1 像素）——共 3 像素。
    #[test]
    fn rasterize_polygon_triangle_half_open() {
        let pts = [[1.0, 1.0], [3.0, 1.0], [2.0, 4.0]];
        let mask = rasterize_polygon(&pts, 5, 5);
        let flat = |x: usize, y: usize| y * 5 + x;
        assert_eq!(mask[flat(1, 1)], 1);
        assert_eq!(mask[flat(2, 1)], 1);
        assert_eq!(mask[flat(3, 1)], 0, "半开区间右端不含");
        assert_eq!(mask[flat(1, 2)], 1);
        assert_eq!(mask[flat(2, 2)], 0, "右边界像素中心不落入");
        assert_eq!(mask[flat(1, 3)], 0, "尖端行窄于 1 像素，中心采样为空");
        assert_eq!(
            mask.iter().map(|&v| v as usize).sum::<usize>(),
            3,
            "恰好 3 像素"
        );
    }

    /// 退化输入：点数 < 3 返回全零；画布外多边形返回全零。
    #[test]
    fn rasterize_polygon_degenerate_inputs() {
        assert!(rasterize_polygon(&[[0.0, 0.0], [1.0, 1.0]], 4, 4)
            .iter()
            .all(|&v| v == 0));
        let outside = [[100.0, 100.0], [120.0, 100.0], [110.0, 120.0]];
        assert!(rasterize_polygon(&outside, 4, 4).iter().all(|&v| v == 0));
    }

    /// letterbox 解码对照：64×32 全红图 → 32 画布（scale=0.5，内容 32×16，
    /// pad_top=8）：补边为 114/255 灰、内容为 1.0，且 CHW 布局正确。
    #[test]
    fn letterbox_decode_pads_gray_keeps_content() {
        let red = image::RgbImage::from_pixel(64, 32, image::Rgb([255, 0, 0]));
        let (pixels, lb) = decode_letterbox_chw(&red, 32);
        assert!((lb.scale - 0.5).abs() < 1e-6);
        assert_eq!((lb.dst_w, lb.dst_h), (32, 32));
        let n = (32 * 32) as usize;
        assert_eq!(pixels.len(), 3 * n);
        // R 通道：第 0 行（补边）= 114/255；第 16 行（内容）= 1.0
        let r_row0 = &pixels[0..32];
        let r_row16 = &pixels[16 * 32..16 * 32 + 32];
        assert!((r_row0[0] - 114.0 / 255.0).abs() < 1e-6);
        assert!((r_row16[16] - 1.0).abs() < 1e-6);
        // G 通道：内容区为 0（红色），补边区为 114/255（灰）
        let g_plane = &pixels[n..2 * n];
        assert!((g_plane[16 * 32 + 16] - 0.0).abs() < 1e-6, "内容区 G=0");
        assert!(
            (g_plane[0] - 114.0 / 255.0).abs() < 1e-6,
            "补边区 G=114/255"
        );
        // B 通道同 G（灰与红均无蓝差异？灰含蓝：同 G 断言）
        let b_plane = &pixels[2 * n..3 * n];
        assert!((b_plane[16 * 32 + 16] - 0.0).abs() < 1e-6, "内容区 B=0");
        // CHW：R 平面在前
        assert!((pixels[16 * 32 + 16] - 1.0).abs() < 1e-6);
    }

    /// 端到端：真实 coco8-seg 目录可加载，掩码画布 = img/4，类别合法
    /// （COCO 80 类），且至少一张图含实例。
    #[test]
    fn load_cocoseg_dir_smoke() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/coco8-seg");
        let samples = load_cocoseg_dir(&root, "train", 128).expect("coco8-seg 应存在");
        assert_eq!(samples.len(), 4, "train split 4 图");
        assert!(samples.iter().all(|s| s.pixels.len() == 3 * 128 * 128));
        assert!(samples.iter().all(|s| s.masks.len() == s.labels.len()));
        assert!(
            samples.iter().all(|s| s.labels.iter().all(|&c| c < 80)),
            "COCO 类别应 < 80"
        );
        let with_inst = samples.iter().filter(|s| !s.masks.is_empty()).count();
        assert!(with_inst >= 1, "至少一张图含实例");
        let s = samples.iter().find(|s| !s.masks.is_empty()).unwrap();
        assert_eq!(s.masks[0].len(), 32 * 32, "掩码画布 = img/4");
    }
}
