//! YOLO 目录格式数据集加载（`images/<split>` + `labels/<split>/*.txt`）与
//! ImageNet ImageFolder 分类目录加载（`<split>/<wnid>/*.JPEG`）。
//!
//! 标注格式：`class cx cy w h`（相对原图归一化）。预处理默认 letterbox
//! （等比缩放 + 114 灰居中补边，YOLO 惯例），标注框经 [`av_core::geometry::Letterbox::map_box`]
//! 映射到输入画布像素空间；拉伸模式（v0.1 旧行为）保留可选。
//! 预测与 gt 同处画布空间，评测协议（mAP）直接可比。

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use rayon::prelude::*;
use tch::{Device, Kind, Tensor};

use av_core::error::{AvError, AvResult};
use av_core::geometry::{Aabb, Letterbox, letterbox};
use av_tasks::augment::{AugmentPlan, scaled_dims};

/// 单样本：图片张量 + 输入画布空间绝对像素 xyxy 框 + 类别（与框一一对应）。
pub struct SampleTensor {
    pub x: Tensor,
    pub boxes: Vec<[f32; 4]>,
    pub labels: Vec<u32>,
}

// tch 0.17 的 Tensor 未实现 Clone，用 copy()（引用计数共享存储）手写
impl Clone for SampleTensor {
    fn clone(&self) -> Self {
        Self {
            x: self.x.copy(),
            boxes: self.boxes.clone(),
            labels: self.labels.clone(),
        }
    }
}

/// 预处理模式：letterbox（等比缩放补边，默认，评测协议可比性前提）或
/// stretch（拉伸到方形，v0.1 旧行为，保留作对照开关）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResizeMode {
    /// 等比缩放 + 114/255 灰居中补边（YOLO 惯例）。
    #[default]
    Letterbox,
    /// 拉伸到 img_size 方形（旧行为，坐标线性映射）。
    Stretch,
}

/// 加载 YOLO 目录数据集并预解码为张量（默认 letterbox 预处理；
/// v0.1 规模数据集整集预载，流式加载/多线程解码按 M2 数据管线落地）。
///
/// `imagenet_norm`：true 时输入做 ImageNet mean/std 归一化（预训练骨干域），
/// false 保持 [0,1] RGB（历史语义）。
pub fn load_yolo_dir(
    root: &Path,
    split: &str,
    img_size: u32,
    device: Device,
    imagenet_norm: bool,
) -> AvResult<Vec<SampleTensor>> {
    load_yolo_dir_with_mode(root, split, img_size, device, ResizeMode::Letterbox, imagenet_norm)
}

/// 同 [`load_yolo_dir`]，可显式选择预处理模式（letterbox / stretch）。
pub fn load_yolo_dir_with_mode(
    root: &Path,
    split: &str,
    img_size: u32,
    device: Device,
    mode: ResizeMode,
    imagenet_norm: bool,
) -> AvResult<Vec<SampleTensor>> {
    let img_dir = root.join("images").join(split);
    let lbl_dir = root.join("labels").join(split);
    if !img_dir.is_dir() {
        return Err(AvError::data(format!(
            "数据集图片目录不存在: {}",
            img_dir.display()
        )));
    }

    let mut entries: Vec<PathBuf> = std::fs::read_dir(&img_dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            matches!(
                p.extension().and_then(|e| e.to_str()),
                Some("jpg") | Some("jpeg") | Some("png") | Some("bmp")
            )
        })
        .collect();
    entries.sort();
    if entries.is_empty() {
        return Err(AvError::data(format!(
            "数据集图片目录为空: {}",
            img_dir.display()
        )));
    }

    // rayon 多核并行解码（PLAN 第一部分）：每图一个独立任务（读图 + 标注解析 +
    // 张量编码），par_iter 保持原顺序 collect
    let samples: Vec<SampleTensor> = entries
        .par_iter()
        .map(|img_path| -> AvResult<SampleTensor> {
            let stem = img_path
                .file_stem()
                .and_then(|s| s.to_str())
                .ok_or_else(|| AvError::data("文件名非法"))?
                .to_string();
            let lbl_path = lbl_dir.join(format!("{stem}.txt"));
            let rgb = image::open(img_path)
                .map_err(|e| AvError::data(format!("读图失败 {}: {e}", img_path.display())))?
                .to_rgb8();
            let (ow, oh) = (rgb.width(), rgb.height());
            let lb = match mode {
                ResizeMode::Letterbox => Some(letterbox(ow, oh, img_size, img_size)),
                ResizeMode::Stretch => None,
            };

            let (mut boxes, mut labels) = (Vec::new(), Vec::new());
            if lbl_path.exists() {
                let text = std::fs::read_to_string(&lbl_path)?;
                for line in text.lines() {
                    let mut it = line.split_whitespace();
                    let (Some(cls), Some(cx), Some(cy), Some(w), Some(h)) =
                        (it.next(), it.next(), it.next(), it.next(), it.next())
                    else {
                        continue;
                    };
                    let (cls, cx, cy, w, h): (f32, f32, f32, f32, f32) = (
                        cls.parse().map_err(|_| bad_label(&lbl_path))?,
                        cx.parse().map_err(|_| bad_label(&lbl_path))?,
                        cy.parse().map_err(|_| bad_label(&lbl_path))?,
                        w.parse().map_err(|_| bad_label(&lbl_path))?,
                        h.parse().map_err(|_| bad_label(&lbl_path))?,
                    );
                    boxes.push(match lb {
                        // 归一化 cxcywh → 原图像素 xyxy → letterbox 画布 xyxy
                        Some(lb) => {
                            let (bw, bh) = (w * ow as f32, h * oh as f32);
                            let (bcx, bcy) = (cx * ow as f32, cy * oh as f32);
                            let g = Aabb::new(
                                bcx - bw / 2.0,
                                bcy - bh / 2.0,
                                bcx + bw / 2.0,
                                bcy + bh / 2.0,
                            );
                            let m = lb.map_box(g);
                            [m.x1, m.y1, m.x2, m.y2]
                        }
                        // 归一化 cxcywh → 拉伸 resize 后的绝对像素 xyxy
                        None => {
                            let s = img_size as f32;
                            let (bw, bh) = (w * s, h * s);
                            let (bcx, bcy) = (cx * s, cy * s);
                            [bcx - bw / 2.0, bcy - bh / 2.0, bcx + bw / 2.0, bcy + bh / 2.0]
                        }
                    });
                    labels.push(cls as u32);
                }
            }
            let x = rgb_to_input_tensor(&rgb, img_size, lb, device, imagenet_norm)?;
            Ok(SampleTensor { x, boxes, labels })
        })
        .collect::<Result<Vec<_>, AvError>>()?;
    Ok(samples)
}

// ---------------------------------------------------------------------------
// `.avpack` 容器数据源（PLAN 附录 B）：容器名约定 = 打包时的相对路径
// ---------------------------------------------------------------------------

/// 容器条目名是否为图片（扩展名 jpg/jpeg/png/bmp，与目录加载器同一白名单）。
fn is_image_entry_name(name: &str) -> bool {
    match name.rsplit_once('.') {
        Some((_, ext)) => matches!(ext, "jpg" | "jpeg" | "png" | "bmp"),
        None => false,
    }
}

/// 图片条目名 `images/<split>/<stem>.<ext>` → 标注条目名 `labels/<split>/<stem>.txt`。
fn label_entry_name(img_name: &str) -> Option<String> {
    let rest = img_name.strip_prefix("images/")?;
    let (stem, _) = rest.rsplit_once('.')?;
    Some(format!("labels/{stem}.txt"))
}

/// 解析 YOLO txt 标注全文（每行 `cls cx cy w h`，归一化）→ (原图像素 xyxy 框, 类别)。
/// 空行跳过、坏行报错（与 [`load_yolo_dir`] 同款严格解析与换算公式）；供
/// avpack 容器加载器复用（容器里没有文件路径，只有字节与条目名）。
fn parse_yolo_label_text(
    text: &str,
    ow: u32,
    oh: u32,
    lbl_disp: &str,
) -> AvResult<(Vec<[f32; 4]>, Vec<u32>)> {
    let mut boxes = Vec::new();
    let mut labels = Vec::new();
    for line in text.lines() {
        let mut it = line.split_whitespace();
        let (Some(cls), Some(cx), Some(cy), Some(w), Some(h)) =
            (it.next(), it.next(), it.next(), it.next(), it.next())
        else {
            continue;
        };
        let (cls, cx, cy, w, h): (f32, f32, f32, f32, f32) = (
            cls.parse()
                .map_err(|_| AvError::data(format!("标注解析失败: {lbl_disp}")))?,
            cx.parse()
                .map_err(|_| AvError::data(format!("标注解析失败: {lbl_disp}")))?,
            cy.parse()
                .map_err(|_| AvError::data(format!("标注解析失败: {lbl_disp}")))?,
            w.parse()
                .map_err(|_| AvError::data(format!("标注解析失败: {lbl_disp}")))?,
            h.parse()
                .map_err(|_| AvError::data(format!("标注解析失败: {lbl_disp}")))?,
        );
        // 归一化 cxcywh → 原图像素 xyxy（letterbox 映射由调用方按需叠加）
        let (bw, bh) = (w * ow as f32, h * oh as f32);
        let (bcx, bcy) = (cx * ow as f32, cy * oh as f32);
        boxes.push([bcx - bw / 2.0, bcy - bh / 2.0, bcx + bw / 2.0, bcy + bh / 2.0]);
        labels.push(cls as u32);
    }
    Ok((boxes, labels))
}

/// 从 `.avpack` 容器加载 YOLO 格式数据集 split 并预解码为张量（letterbox 预处理，
/// 与 [`load_yolo_dir`] 同一条编码/映射路径）。容器名约定 = 打包时的相对路径：
/// 图片在 `images/<split>/` 下（jpg/jpeg/png/bmp），标注为对应
/// `labels/<split>/<stem>.txt`（缺标注文件的图片按空标注处理，与目录加载一致）。
pub fn load_yolo_avpack(
    pack: &Path,
    split: &str,
    img_size: u32,
    device: Device,
    imagenet_norm: bool,
) -> AvResult<Vec<SampleTensor>> {
    Ok(load_yolo_avpack_named(pack, split, img_size, device, imagenet_norm)?
        .into_iter()
        .map(|(_, s)| s)
        .collect())
}

/// 同 [`load_yolo_avpack`]，附带每个样本的容器条目名（`images/<split>/...`，
/// 按名字排序，供测试/诊断对账）。
pub fn load_yolo_avpack_named(
    pack: &Path,
    split: &str,
    img_size: u32,
    device: Device,
    imagenet_norm: bool,
) -> AvResult<Vec<(String, SampleTensor)>> {
    let prefix = format!("images/{split}/");
    let reader = crate::avpack::AvPackReader::open(pack)?;
    let mut names: Vec<String> = reader
        .entries()
        .iter()
        .filter(|e| e.name.starts_with(&prefix) && is_image_entry_name(&e.name))
        .map(|e| e.name.clone())
        .collect();
    names.sort();
    if names.is_empty() {
        return Err(AvError::data(format!(
            "avpack 容器 {} 无 images/{split}/ 下图片（布局须为 images/<split> + labels/<split>）",
            pack.display()
        )));
    }

    // rayon 并行解码（与 load_yolo_dir 同策略：par_iter 保序 collect）
    let samples: Vec<(String, SampleTensor)> = names
        .par_iter()
        .map(|name| -> AvResult<(String, SampleTensor)> {
            let bytes = reader.read(name)?;
            let rgb = image::load_from_memory(&bytes)
                .map_err(|e| AvError::data(format!("读图失败（容器条目 {name}）: {e}")))?
                .to_rgb8();
            let (ow, oh) = (rgb.width(), rgb.height());
            let lb = letterbox(ow, oh, img_size, img_size);

            let (mut boxes, mut labels) = (Vec::new(), Vec::new());
            if let Some(lbl_name) = label_entry_name(name) {
                if reader.entries().iter().any(|e| e.name == lbl_name) {
                    let bytes = reader.read(&lbl_name)?;
                    let text = String::from_utf8(bytes)
                        .map_err(|_| AvError::data(format!("标注非 UTF-8: {lbl_name}")))?;
                    let (px_boxes, cls) = parse_yolo_label_text(&text, ow, oh, &lbl_name)?;
                    // 原图像素 xyxy → letterbox 画布 xyxy（load_yolo_dir 同款 map_box 路径）
                    boxes = px_boxes
                        .iter()
                        .map(|b| {
                            let m = lb.map_box(Aabb::new(b[0], b[1], b[2], b[3]));
                            [m.x1, m.y1, m.x2, m.y2]
                        })
                        .collect();
                    labels = cls;
                }
            }
            let x = rgb_to_input_tensor(&rgb, img_size, Some(lb), device, imagenet_norm)?;
            Ok((name.clone(), SampleTensor { x, boxes, labels }))
        })
        .collect::<Result<Vec<_>, AvError>>()?;
    Ok(samples)
}

/// `.avpack` 容器 → 检测原始样本（标注解析与 [`load_yolo_dir_raw`] 同款，仅省去
/// letterbox 映射——供训练增强的「raw 样本 + 逐 epoch 编码」路径使用）。
pub fn load_yolo_avpack_raw(pack: &Path, split: &str) -> AvResult<Vec<RawDetectSample>> {
    let prefix = format!("images/{split}/");
    let reader = crate::avpack::AvPackReader::open(pack)?;
    let mut names: Vec<String> = reader
        .entries()
        .iter()
        .filter(|e| e.name.starts_with(&prefix) && is_image_entry_name(&e.name))
        .map(|e| e.name.clone())
        .collect();
    names.sort();
    if names.is_empty() {
        return Err(AvError::data(format!(
            "avpack 容器 {} 无 images/{split}/ 下图片",
            pack.display()
        )));
    }

    let mut out = Vec::new();
    for name in &names {
        let bytes = reader.read(name)?;
        let rgb = image::load_from_memory(&bytes)
            .map_err(|e| AvError::data(format!("读图失败（容器条目 {name}）: {e}")))?
            .to_rgb8();
        let (ow, oh) = (rgb.width(), rgb.height());
        let (mut boxes, mut labels) = (Vec::new(), Vec::new());
        if let Some(lbl_name) = label_entry_name(name) {
            if reader.entries().iter().any(|e| e.name == lbl_name) {
                let bytes = reader.read(&lbl_name)?;
                let text = String::from_utf8(bytes)
                    .map_err(|_| AvError::data(format!("标注非 UTF-8: {lbl_name}")))?;
                (boxes, labels) = parse_yolo_label_text(&text, ow, oh, &lbl_name)?;
            }
        }
        out.push(RawDetectSample {
            w: ow,
            h: oh,
            rgb: rgb.into_raw(),
            boxes,
            labels,
        });
    }
    Ok(out)
}

/// 图片文件 → [3,S,S] 单样本张量（默认 letterbox，RGB [0,1]；堆批用 stack_samples）。
/// `imagenet_norm = true` 时输出为 ImageNet mean/std 归一化域（预训练骨干）。
pub fn decode_image_tensor(
    path: &Path,
    img_size: u32,
    device: Device,
    imagenet_norm: bool,
) -> AvResult<Tensor> {
    decode_image_tensor_with_mode(path, img_size, device, ResizeMode::Letterbox, imagenet_norm)
}

/// 同 [`decode_image_tensor`]，可显式选择预处理模式。
pub fn decode_image_tensor_with_mode(
    path: &Path,
    img_size: u32,
    device: Device,
    mode: ResizeMode,
    imagenet_norm: bool,
) -> AvResult<Tensor> {
    let (x, _, _, _) = decode_image_with_meta(path, img_size, device, mode, imagenet_norm)?;
    Ok(x)
}

/// 解码并返回预处理元数据：张量 + letterbox 参数 + 原图尺寸。
/// 推理产物坐标还原（画布 → 原图）必需。
pub fn decode_image_with_meta(
    path: &Path,
    img_size: u32,
    device: Device,
    mode: ResizeMode,
    imagenet_norm: bool,
) -> AvResult<(Tensor, Option<Letterbox>, u32, u32)> {
    let img = image::open(path).map_err(|e| AvError::data(format!("读图失败 {}: {e}", path.display())))?;
    let rgb = img.to_rgb8();
    let (ow, oh) = (rgb.width(), rgb.height());
    let lb = match mode {
        ResizeMode::Letterbox => Some(letterbox(ow, oh, img_size, img_size)),
        ResizeMode::Stretch => None,
    };
    let x = rgb_to_input_tensor(&rgb, img_size, lb, device, imagenet_norm)?;
    Ok((x, lb, ow, oh))
}

/// 已在内存的 RGB 图（切片推理的每个窗口裁剪）→ 张量 + letterbox 元数据。
pub fn decode_rgb_with_meta(
    rgb: &image::RgbImage,
    img_size: u32,
    device: Device,
    mode: ResizeMode,
    imagenet_norm: bool,
) -> AvResult<(Tensor, Option<Letterbox>)> {
    let lb = match mode {
        ResizeMode::Letterbox => Some(letterbox(rgb.width(), rgb.height(), img_size, img_size)),
        ResizeMode::Stretch => None,
    };
    let x = rgb_to_input_tensor(rgb, img_size, lb, device, imagenet_norm)?;
    Ok((x, lb))
}

/// RGB 图 → [3,S,S] 张量。letterbox 模式下等比缩放后贴到 114 灰画布
/// （对齐参数取 img_size，保证画布恰为 img_size 方形、内容居中）。
///
/// `imagenet_norm = true` 时做 ImageNet mean/std 归一化（(x/255 − mean)/std），
/// 供 ImageNet 预训练骨干（BN running 统计量在 ImageNet 域）消费；默认 false
/// 保持 [0,1] RGB 历史语义（合成数据与非预训练路径零变化）。
fn rgb_to_input_tensor(
    rgb: &image::RgbImage,
    img_size: u32,
    lb: Option<Letterbox>,
    device: Device,
    imagenet_norm: bool,
) -> AvResult<Tensor> {
    let canvas: image::RgbImage = match lb {
        Some(lb) => {
            let nw = ((rgb.width() as f32 * lb.scale).round() as u32).clamp(1, img_size);
            let nh = ((rgb.height() as f32 * lb.scale).round() as u32).clamp(1, img_size);
            let resized =
                image::imageops::resize(rgb, nw, nh, image::imageops::FilterType::Triangle);
            let mut c =
                image::RgbImage::from_pixel(img_size, img_size, image::Rgb([114, 114, 114]));
            // 粘贴偏移取整（与 map_box 的 pad 差 ≤0.5px，框映射仍统一走 map_box）
            image::imageops::overlay(
                &mut c,
                &resized,
                lb.pad_left.round() as i64,
                lb.pad_top.round() as i64,
            );
            c
        }
        None => image::imageops::resize(
            rgb,
            img_size,
            img_size,
            image::imageops::FilterType::Triangle,
        ),
    };
    canvas_to_input_tensor(canvas, device, imagenet_norm)
}

/// 已合成好的 RGB 画布 → [3,H,W] 张量（逐像素 [0,1] 或 ImageNet 域）。
/// letterbox 合成（缩放 + 114 灰补边）与归一化解耦：缓存编码路径复用同一归一化。
/// 支持非方形画布（内容贴片 W×H）；通道面按行主序像素划分（与既有布局一致）。
fn canvas_to_input_tensor(
    canvas: image::RgbImage,
    device: Device,
    imagenet_norm: bool,
) -> AvResult<Tensor> {
    let (w, h) = (canvas.width() as usize, canvas.height() as usize);
    let n = w * h;
    // ImageNet 归一化常数（torchvision 预训练域，RGB 通道序）
    const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];
    let mut buf = vec![0f32; 3 * n];
    for (i, px) in canvas.pixels().enumerate() {
        let [r, g, b] = px.0;
        if imagenet_norm {
            buf[i] = (r as f32 / 255.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
            buf[n + i] = (g as f32 / 255.0 - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
            buf[2 * n + i] = (b as f32 / 255.0 - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
        } else {
            buf[i] = r as f32 / 255.0;
            buf[n + i] = g as f32 / 255.0;
            buf[2 * n + i] = b as f32 / 255.0;
        }
    }
    Ok(Tensor::from_slice(&buf)
        .to_kind(Kind::Float)
        .to_device(device)
        .reshape([3, h as i64, w as i64]))
}

/// 把一批样本堆成训练张量 [B,3,S,S]。
pub fn stack_samples(samples: &[SampleTensor]) -> AvResult<Tensor> {
    let xs: Vec<Tensor> = samples.iter().map(|s| s.x.copy()).collect();
    Ok(Tensor::stack(&xs, 0))
}

// ---------------------------------------------------------------------------
// ImageNet ImageFolder（分类）：<root>/<split>/<wnid>/*.JPEG
// ---------------------------------------------------------------------------

/// 分类单样本：预解码输入张量 [3,S,S]（标签在 [`load_imagefolder`] 返回的平行
/// `Vec<u32>` 里，洗牌时按同一下标联动，避免每样本一份标签的冗余）。
#[derive(Debug)]
pub struct ClassifySample {
    pub x: Tensor,
}

// tch 0.17 的 Tensor 未实现 Clone，用 copy()（引用计数共享存储）手写
impl Clone for ClassifySample {
    fn clone(&self) -> Self {
        Self { x: self.x.copy() }
    }
}

/// 把一批分类样本堆成 [B,3,S,S] 张量。
pub fn stack_classify(samples: &[ClassifySample]) -> AvResult<Tensor> {
    let xs: Vec<Tensor> = samples.iter().map(|s| s.x.copy()).collect();
    Ok(Tensor::stack(&xs, 0))
}

/// ImageFolder 加载（类 id 由 wnid 目录名排序推导，ImageFolder 惯例）。
///
/// - `num_classes_from_dir = true`：类数 = `<split>` 下 wnid 子目录数（排序后映射
///   0..N，确定性）。当前唯一实现路径即从目录推导；`false` 同样推导（固定外部
///   词表映射按 M2 落地），参数保留作调用方意图表达与前向兼容。
/// - 返回 `(样本, 平行标签, wnid→类id 映射)`；调用方需校验映射长度与
///   `classify.num_classes` 一致（分类头维度在建模期已固定）。
pub fn load_imagefolder(
    root: &Path,
    split: &str,
    img_size: u32,
    num_classes_from_dir: bool,
    device: Device,
    imagenet_norm: bool,
) -> AvResult<(Vec<ClassifySample>, Vec<u32>, HashMap<String, u32>)> {
    let _ = num_classes_from_dir; // 语义见 doc；行为恒为「从目录推导」
    load_imagefolder_with_classes(root, split, img_size, None, device, imagenet_norm)
}

/// 同 [`load_imagefolder`]，可用已有 `wnid→类id` 映射（例如 train split 推导的
/// 映射）加载另一 split，保证 val 与 train 类 id 一致——val 缺某类时按目录排序
/// 推导的 id 会错位，必须复用 train 映射。`classes = None` 时从本 split 目录推导。
pub fn load_imagefolder_with_classes(
    root: &Path,
    split: &str,
    img_size: u32,
    classes: Option<&HashMap<String, u32>>,
    device: Device,
    imagenet_norm: bool,
) -> AvResult<(Vec<ClassifySample>, Vec<u32>, HashMap<String, u32>)> {
    let split_dir = root.join(split);
    if !split_dir.is_dir() {
        return Err(AvError::data(format!(
            "ImageFolder split 目录不存在: {}",
            split_dir.display()
        )));
    }

    // wnid 子目录排序（确定性类 id 的前提）
    let mut wnids: Vec<String> = std::fs::read_dir(&split_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .filter_map(|e| e.file_name().into_string().ok())
        .collect();
    wnids.sort();

    let class_map: HashMap<String, u32> = match classes {
        Some(m) => {
            for w in &wnids {
                if !m.contains_key(w) {
                    return Err(AvError::data(format!(
                        "split {split} 含未知类别 {w}（不在给定词表中）"
                    )));
                }
            }
            m.clone()
        }
        None => wnids
            .iter()
            .enumerate()
            .map(|(i, w)| (w.clone(), i as u32))
            .collect(),
    };
    if wnids.is_empty() {
        return Err(AvError::data(format!(
            "ImageFolder split 目录为空（无 wnid 子目录）: {}",
            split_dir.display()
        )));
    }

    // rayon 并行解码：先收集 (路径, 类别) 任务对，再多核解码（顺序保持）
    let mut jobs: Vec<(PathBuf, u32)> = Vec::new();
    for wnid in &wnids {
        let label = class_map[wnid];
        let cls_dir = split_dir.join(wnid);
        let mut files: Vec<PathBuf> = std::fs::read_dir(&cls_dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                matches!(
                    p.extension().and_then(|e| e.to_str()),
                    Some("jpg") | Some("jpeg") | Some("JPEG") | Some("png") | Some("bmp")
                )
            })
            .collect();
        files.sort();
        jobs.extend(files.into_iter().map(|p| (p, label)));
    }
    if jobs.is_empty() {
        return Err(AvError::data(format!(
            "ImageFolder split 无图片: {}",
            split_dir.display()
        )));
    }

    let decoded: Vec<(ClassifySample, u32)> = jobs
        .par_iter()
        .map(|(img_path, label)| -> AvResult<(ClassifySample, u32)> {
            let img = image::open(img_path).map_err(|e| {
                AvError::data(format!("读图失败 {}: {e}", img_path.display()))
            })?;
            // 分类预处理：拉伸 resize 到 img_size 方形（rgb_to_input_tensor 的
            // lb=None 路径），RGB [0,1]（imagenet_norm = true 时 ImageNet 域）
            let x = rgb_to_input_tensor(&img.to_rgb8(), img_size, None, device, imagenet_norm)?;
            Ok((ClassifySample { x }, *label))
        })
        .collect::<Result<Vec<_>, AvError>>()?;
    let (samples, labels): (Vec<ClassifySample>, Vec<u32>) = decoded
        .into_iter()
        .map(|(s, l)| (s, l))
        .unzip();
    if samples.is_empty() {
        return Err(AvError::data(format!(
            "ImageFolder split 无图片: {}",
            split_dir.display()
        )));
    }
    Ok((samples, labels, class_map))
}

fn bad_label(p: &Path) -> AvError {
    AvError::data(format!("标注解析失败: {}", p.display()))
}

// ---------------------------------------------------------------------------
// OBB：DOTA 格式（Ultralytics OBB：class + 归一化 4 角点）
// ---------------------------------------------------------------------------

/// OBB 单样本：图片张量 + [cx,cy,w,h,θ]（le90 域，画布像素）+ 类别。
pub struct ObbSample {
    pub x: Tensor,
    pub boxes: Vec<[f32; 5]>,
    pub labels: Vec<u32>,
}

// tch 的 Tensor 未实现 Clone，用 copy()（引用计数共享存储）手写
impl Clone for ObbSample {
    fn clone(&self) -> Self {
        Self {
            x: self.x.copy(),
            boxes: self.boxes.clone(),
            labels: self.labels.clone(),
        }
    }
}

/// 加载 DOTA 格式 OBB 数据集。4 角点经 letterbox 等比映射（角度保持不变），
/// 转为 le90 域的 [cx,cy,w,h,θ]。
pub fn load_dota_dir(
    root: &Path,
    split: &str,
    img_size: u32,
    device: Device,
    imagenet_norm: bool,
) -> AvResult<Vec<ObbSample>> {
    use av_core::conventions::AngleDomain;

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

    let mut out = Vec::new();
    for p in paths {
        let stem = p
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| AvError::data("文件名非法"))?
            .to_string();
        let lbl_path = lbl_dir.join(format!("{stem}.txt"));
        let img = image::open(&p)
            .map_err(|e| AvError::data(format!("读图失败 {p:?}: {e}")))?;
        let rgb = img.to_rgb8();
        let (ow, oh) = (rgb.width() as f32, rgb.height() as f32);
        let (x, lb) = decode_rgb_with_meta(&rgb, img_size, device, ResizeMode::Letterbox, imagenet_norm)?;
        let lb = lb.ok_or_else(|| AvError::data("dota 加载要求 letterbox 模式"))?;

        let mut boxes = Vec::new();
        let mut labels = Vec::new();
        if lbl_path.exists() {
            for line in std::fs::read_to_string(&lbl_path)?.lines() {
                let vals: Vec<f32> =
                    line.split_whitespace().filter_map(|t| t.parse().ok()).collect();
                if vals.len() < 9 {
                    continue;
                }
                // 4 角点：归一化 → 原图像素 → letterbox 映射（等比+平移，角度不变）
                let mut pts = [[0f32; 2]; 4];
                for k in 0..4 {
                    pts[k] = [
                        vals[1 + 2 * k] * ow * lb.scale + lb.pad_left,
                        vals[2 + 2 * k] * oh * lb.scale + lb.pad_top,
                    ];
                }
                let cx = (pts[0][0] + pts[2][0]) / 2.0;
                let cy = (pts[0][1] + pts[2][1]) / 2.0;
                let dx = pts[1][0] - pts[0][0];
                let dy = pts[1][1] - pts[0][1];
                let w = (dx * dx + dy * dy).sqrt();
                let h = {
                    let ex = pts[3][0] - pts[0][0];
                    let ey = pts[3][1] - pts[0][1];
                    (ex * ex + ey * ey).sqrt()
                };
                if w < 1.0 || h < 1.0 {
                    continue;
                }
                let theta = dy.atan2(dx);
                boxes.push([cx, cy, w, h, AngleDomain::Le90.normalize(theta)]);
                labels.push(vals[0] as u32);
            }
        }
        out.push(ObbSample { x, boxes, labels });
    }
    Ok(out)
}

/// 把一批 OBB 样本堆成训练张量 [B,3,S,S]。
pub fn stack_obb_samples(samples: &[ObbSample]) -> AvResult<Tensor> {
    let xs: Vec<Tensor> = samples.iter().map(|s| s.x.copy()).collect();
    Ok(Tensor::stack(&xs, 0))
}

// ---------------------------------------------------------------------------
// 实例分割：COCO 分割格式（Ultralytics coco8-seg：labels 行为
// `cls x1 y1 x2 y2 ... xn yn`，归一化多边形点，n >= 3）
// ---------------------------------------------------------------------------

/// 分割单样本：图片张量 + 每实例二值掩码（img/4 × img/4，0/1 u8，flat）+ 类别。
///
/// 掩码监督取低分辨率（img_size/4，320 输入下 80×80）：多边形 gt 本就是粗粒度
/// 标注，低分辨率画布让扫描线栅格化与掩码损失的计算量都缩 16 倍（对齐 YOLACT
/// 原型掩码分辨率）。
pub struct SegSample {
    pub x: Tensor,
    pub masks: Vec<Vec<u8>>,
    pub labels: Vec<u32>,
}

// tch 的 Tensor 未实现 Clone，用 copy()（引用计数共享存储）手写
impl Clone for SegSample {
    fn clone(&self) -> Self {
        Self {
            x: self.x.copy(),
            masks: self.masks.clone(),
            labels: self.labels.clone(),
        }
    }
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
    let max_y = points.iter().map(|p| p[1]).fold(f32::NEG_INFINITY, f32::max);
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

/// 加载 COCO 分割格式数据集（Ultralytics coco8-seg 目录布局：
/// `images/<split>` + `labels/<split>/*.txt`）。
///
/// 标注行为 `cls x1 y1 x2 y2 ... xn yn`（归一化多边形，n >= 3）；
/// **纯检测框行（恰 5 个值：cls + cxcywh）被跳过**——coco8-seg 的 label 文件
/// 可能混有检测任务写出的框行。多边形点经 letterbox 等比映射到画布后，
/// 栅格化到 img/4 × img/4 的低分辨率掩码；栅格化后为空的实例（退化标注）
/// 整条跳过。
pub fn load_cocoseg_dir(
    root: &Path,
    split: &str,
    img_size: u32,
    device: Device,
    imagenet_norm: bool,
) -> AvResult<Vec<SegSample>> {
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
        let (x, lb) = decode_rgb_with_meta(&rgb, img_size, device, ResizeMode::Letterbox, imagenet_norm)?;
        let lb = lb.ok_or_else(|| AvError::data("coco seg 加载要求 letterbox 模式"))?;

        let mut masks = Vec::new();
        let mut labels = Vec::new();
        if lbl_path.exists() {
            for line in std::fs::read_to_string(&lbl_path)?.lines() {
                let vals: Vec<f32> =
                    line.split_whitespace().filter_map(|t| t.parse().ok()).collect();
                // 多边形 = 1 类别 + 2n 坐标，n >= 3 → 至少 7 个值；5 值行为纯检测框，跳过
                if vals.len() < 7 {
                    continue;
                }
                let n_pts = (vals.len() - 1) / 2;
                // 归一化 → 原图像素 → letterbox 画布 → 掩码画布（÷4）
                let k = mw as f32 / img_size as f32;
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
        out.push(SegSample { x, masks, labels });
    }
    Ok(out)
}

/// 把一批分割样本堆成训练张量 [B,3,S,S]。
pub fn stack_seg_samples(samples: &[SegSample]) -> AvResult<Tensor> {
    let xs: Vec<Tensor> = samples.iter().map(|s| s.x.copy()).collect();
    Ok(Tensor::stack(&xs, 0))
}

// ---------------------------------------------------------------------------
// 关键点：COCO 姿态格式（Ultralytics coco8-pose：labels 行为
// `cls cx cy w h (x y v)×K`，坐标相对原图归一化，v 为 COCO 可见性标志 0/1/2）
// ---------------------------------------------------------------------------

/// 关键点单样本：图片张量 + 每实例框（cxcywh，画布像素，供分配与 OKS 尺度）
/// + 每实例 K 个关键点 [x, y, v]（画布像素 + 可见性标志）+ 类别。
pub struct KeypointSample {
    pub x: Tensor,
    pub boxes: Vec<[f32; 4]>,
    pub kpts: Vec<Vec<[f32; 3]>>,
    pub labels: Vec<u32>,
}

// tch 的 Tensor 未实现 Clone，用 copy()（引用计数共享存储）手写
impl Clone for KeypointSample {
    fn clone(&self) -> Self {
        Self {
            x: self.x.copy(),
            boxes: self.boxes.clone(),
            kpts: self.kpts.clone(),
            labels: self.labels.clone(),
        }
    }
}

/// 加载 COCO 姿态格式数据集（Ultralytics coco8-pose 目录布局：
/// `images/<split>` + `labels/<split>/*.txt`）。
///
/// 标注行为 `cls cx cy w h (x y v)×K`；K 由行内 token 数推导（每行可不同，
/// 模型侧按 num_keypoints 一致性校验）。关键点经 letterbox 等比映射：
/// `x' = x·orig_w·scale + pad_left`，`y'` 同理（与框映射同一坐标系，预测与
/// gt 同处画布空间）；v 标志原样保留（0/1/2，v=0 的退化 (0,0) 坐标不参与
/// 损失/评测）。实例框映射为 cxcywh 画布像素：中心加 pad、宽高乘 scale。
pub fn load_cocopose_dir(
    root: &Path,
    split: &str,
    img_size: u32,
    device: Device,
    imagenet_norm: bool,
) -> AvResult<Vec<KeypointSample>> {
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
        let (x, lb) = decode_rgb_with_meta(&rgb, img_size, device, ResizeMode::Letterbox, imagenet_norm)?;
        let lb = lb.ok_or_else(|| AvError::data("cocopose 加载要求 letterbox 模式"))?;

        let mut boxes = Vec::new();
        let mut kpts = Vec::new();
        let mut labels = Vec::new();
        if lbl_path.exists() {
            for line in std::fs::read_to_string(&lbl_path)?.lines() {
                let vals: Vec<f32> =
                    line.split_whitespace().filter_map(|t| t.parse().ok()).collect();
                // 行 = 1 类别 + 4 框 + 3K 关键点；K >= 1 → 至少 8 个值
                if vals.len() < 8 || (vals.len() - 5) % 3 != 0 {
                    continue;
                }
                let n_k = (vals.len() - 5) / 3;
                let kp: Vec<[f32; 3]> = (0..n_k)
                    .map(|j| {
                        [
                            vals[5 + 3 * j] * ow * lb.scale + lb.pad_left,
                            vals[6 + 3 * j] * oh * lb.scale + lb.pad_top,
                            vals[7 + 3 * j],
                        ]
                    })
                    .collect();
                // 归一化 cxcywh → 画布 cxcywh（中心经 letterbox 平移，宽高乘 scale）
                boxes.push([
                    vals[1] * ow * lb.scale + lb.pad_left,
                    vals[2] * oh * lb.scale + lb.pad_top,
                    (vals[3] * ow * lb.scale).max(1e-3),
                    (vals[4] * oh * lb.scale).max(1e-3),
                ]);
                kpts.push(kp);
                labels.push(vals[0] as u32);
            }
        }
        out.push(KeypointSample {
            x,
            boxes,
            kpts,
            labels,
        });
    }
    Ok(out)
}

/// 把一批关键点样本堆成训练张量 [B,3,S,S]。
pub fn stack_kp_samples(samples: &[KeypointSample]) -> AvResult<Tensor> {
    let xs: Vec<Tensor> = samples.iter().map(|s| s.x.copy()).collect();
    Ok(Tensor::stack(&xs, 0))
}

// ---------------------------------------------------------------------------
// 训练期增强（数据增强官任务 §1/§2）：原始空间样本 + 逐 epoch 随机编码
// ---------------------------------------------------------------------------

// 既有 *_dir 加载器走「整集预解码」：每张图只编码一次，同一权重反复见到逐位
// 相同的输入。增强要求每个 epoch 独立抽样（翻转 / HSV / 缩放），因此训练侧改用
// 「raw 加载器 + encode_*」：raw 加载器只做图片解码与**原图像素空间**标注解析，
// encode_* 在增强后调用与既有加载器完全相同的编码函数（rgb_to_input_tensor）
// 与坐标映射公式（letterbox::map_box / 同款点映射表达式）。坐标同步由
// 「同一份 AugmentPlan、同一条线性映射链 flip → scale → letterbox」保证；
// AugmentPlan::none() 时输出与既有加载器逐位一致（dataset 单测锁定）。

/// 检测原始样本：原图 RGB8（w*h*3 字节）+ 原图像素 xyxy 框 + 类别。
pub struct RawDetectSample {
    pub w: u32,
    pub h: u32,
    pub rgb: Vec<u8>,
    pub boxes: Vec<[f32; 4]>,
    pub labels: Vec<u32>,
}

/// 关键点原始样本：cxcywh 原图像素框 + [x, y, v] 原图像素关键点。
pub struct RawKeypointSample {
    pub w: u32,
    pub h: u32,
    pub rgb: Vec<u8>,
    pub boxes: Vec<[f32; 4]>,
    pub kpts: Vec<Vec<[f32; 3]>>,
    pub labels: Vec<u32>,
}

/// 分割原始样本：原图像素多边形（每实例一条）。
pub struct RawSegSample {
    pub w: u32,
    pub h: u32,
    pub rgb: Vec<u8>,
    pub polys: Vec<Vec<[f32; 2]>>,
    pub labels: Vec<u32>,
}

/// OBB 原始样本：4 角点原图像素（与 DOTA 标注同点序）。
pub struct RawObbSample {
    pub w: u32,
    pub h: u32,
    pub rgb: Vec<u8>,
    pub corners: Vec<[[f32; 2]; 4]>,
    pub labels: Vec<u32>,
}

/// 列出 split 图片目录下的图片文件（排序，保证确定性）。
fn list_image_files(img_dir: &Path) -> AvResult<Vec<PathBuf>> {
    if !img_dir.is_dir() {
        return Err(AvError::data(format!(
            "数据集图片目录不存在: {}",
            img_dir.display()
        )));
    }
    let mut paths: Vec<PathBuf> = std::fs::read_dir(img_dir)?
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
    Ok(paths)
}

/// 解码一张图为原始样本公共部分（RGB8 + 宽高）。
fn decode_raw_rgb(p: &Path) -> AvResult<(u32, u32, Vec<u8>)> {
    let rgb = image::open(p)
        .map_err(|e| AvError::data(format!("读图失败 {}: {e}", p.display())))?
        .to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    Ok((w, h, rgb.into_raw()))
}

/// YOLO 目录 → 检测原始样本（标注解析与 [`load_yolo_dir`] 同款，仅省去
/// letterbox 映射——映射延迟到 encode 期，flip/scale 在原图空间先做）。
pub fn load_yolo_dir_raw(root: &Path, split: &str) -> AvResult<Vec<RawDetectSample>> {
    let img_dir = root.join("images").join(split);
    let lbl_dir = root.join("labels").join(split);
    let mut out = Vec::new();
    for p in list_image_files(&img_dir)? {
        let stem = p
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| AvError::data("文件名非法"))?
            .to_string();
        let lbl_path = lbl_dir.join(format!("{stem}.txt"));
        let (ow, oh, rgb) = decode_raw_rgb(&p)?;
        let (mut boxes, mut labels) = (Vec::new(), Vec::new());
        if lbl_path.exists() {
            let text = std::fs::read_to_string(&lbl_path)?;
            for line in text.lines() {
                let mut it = line.split_whitespace();
                let (Some(cls), Some(cx), Some(cy), Some(w), Some(h)) =
                    (it.next(), it.next(), it.next(), it.next(), it.next())
                else {
                    continue;
                };
                let (cls, cx, cy, w, h): (f32, f32, f32, f32, f32) = (
                    cls.parse().map_err(|_| bad_label(&lbl_path))?,
                    cx.parse().map_err(|_| bad_label(&lbl_path))?,
                    cy.parse().map_err(|_| bad_label(&lbl_path))?,
                    w.parse().map_err(|_| bad_label(&lbl_path))?,
                    h.parse().map_err(|_| bad_label(&lbl_path))?,
                );
                // 归一化 cxcywh → 原图像素 xyxy（letterbox 延迟到 encode）
                let (bw, bh) = (w * ow as f32, h * oh as f32);
                let (bcx, bcy) = (cx * ow as f32, cy * oh as f32);
                boxes.push([bcx - bw / 2.0, bcy - bh / 2.0, bcx + bw / 2.0, bcy + bh / 2.0]);
                labels.push(cls as u32);
            }
        }
        out.push(RawDetectSample {
            w: ow,
            h: oh,
            rgb,
            boxes,
            labels,
        });
    }
    Ok(out)
}

/// COCO 姿态目录 → 关键点原始样本（解析规则与 [`load_cocopose_dir`] 一致：
/// 行 = cls cx cy w h (x y v)×K，坏行跳过，K 按行内 token 数推导）。
pub fn load_cocopose_dir_raw(root: &Path, split: &str) -> AvResult<Vec<RawKeypointSample>> {
    let img_dir = root.join("images").join(split);
    let lbl_dir = root.join("labels").join(split);
    let mut out = Vec::new();
    for p in list_image_files(&img_dir)? {
        let stem = p
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| AvError::data("文件名非法"))?
            .to_string();
        let lbl_path = lbl_dir.join(format!("{stem}.txt"));
        let (ow, oh, rgb) = decode_raw_rgb(&p)?;
        let (mut boxes, mut kpts, mut labels) = (Vec::new(), Vec::new(), Vec::new());
        if lbl_path.exists() {
            for line in std::fs::read_to_string(&lbl_path)?.lines() {
                let vals: Vec<f32> =
                    line.split_whitespace().filter_map(|t| t.parse().ok()).collect();
                if vals.len() < 8 || (vals.len() - 5) % 3 != 0 {
                    continue;
                }
                let n_k = (vals.len() - 5) / 3;
                let kp: Vec<[f32; 3]> = (0..n_k)
                    .map(|j| [vals[5 + 3 * j] * ow as f32, vals[6 + 3 * j] * oh as f32, vals[7 + 3 * j]])
                    .collect();
                // 框存原图像素（1e-3 下限在 encode 映射后施加，与既有加载器同序）
                boxes.push([
                    vals[1] * ow as f32,
                    vals[2] * oh as f32,
                    vals[3] * ow as f32,
                    vals[4] * oh as f32,
                ]);
                kpts.push(kp);
                labels.push(vals[0] as u32);
            }
        }
        out.push(RawKeypointSample {
            w: ow,
            h: oh,
            rgb,
            boxes,
            kpts,
            labels,
        });
    }
    Ok(out)
}

/// COCO 分割目录 → 分割原始样本（多边形归一化 → 原图像素；退化实例的
/// 判定依赖最终画布，延迟到 [`encode_seg_sample`] 栅格化时做）。
/// JPEG 解码 rayon 并行（与 load_yolo_dir 同策略：par_iter 保序 collect）。
pub fn load_cocoseg_dir_raw(root: &Path, split: &str) -> AvResult<Vec<RawSegSample>> {
    let img_dir = root.join("images").join(split);
    let lbl_dir = root.join("labels").join(split);
    list_image_files(&img_dir)?
        .par_iter()
        .map(|p| -> AvResult<RawSegSample> {
            let stem = p
                .file_stem()
                .and_then(|s| s.to_str())
                .ok_or_else(|| AvError::data("文件名非法"))?
                .to_string();
            let lbl_path = lbl_dir.join(format!("{stem}.txt"));
            let (ow, oh, rgb) = decode_raw_rgb(p)?;
            let (mut polys, mut labels) = (Vec::new(), Vec::new());
            if lbl_path.exists() {
                for line in std::fs::read_to_string(&lbl_path)?.lines() {
                    let vals: Vec<f32> =
                        line.split_whitespace().filter_map(|t| t.parse().ok()).collect();
                    // 与 load_cocoseg_dir 同规则：≥7 值（1 类 + 3 点）才算多边形行
                    if vals.len() < 7 {
                        continue;
                    }
                    let n_pts = (vals.len() - 1) / 2;
                    polys.push(
                        (0..n_pts)
                            .map(|i| [vals[1 + 2 * i] * ow as f32, vals[2 + 2 * i] * oh as f32])
                            .collect(),
                    );
                    labels.push(vals[0] as u32);
                }
            }
            Ok(RawSegSample {
                w: ow,
                h: oh,
                rgb,
                polys,
                labels,
            })
        })
        .collect()
}

/// DOTA 目录 → OBB 原始样本（4 角点归一化 → 原图像素；cxcywhθ 推导延迟到
/// [`encode_obb_sample`]，翻转后角度需从镜像角点重新推导）。
pub fn load_dota_dir_raw(root: &Path, split: &str) -> AvResult<Vec<RawObbSample>> {
    let img_dir = root.join("images").join(split);
    let lbl_dir = root.join("labels").join(split);
    let mut out = Vec::new();
    for p in list_image_files(&img_dir)? {
        let stem = p
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| AvError::data("文件名非法"))?
            .to_string();
        let lbl_path = lbl_dir.join(format!("{stem}.txt"));
        let (ow, oh, rgb) = decode_raw_rgb(&p)?;
        let (mut corners, mut labels) = (Vec::new(), Vec::new());
        if lbl_path.exists() {
            for line in std::fs::read_to_string(&lbl_path)?.lines() {
                let vals: Vec<f32> =
                    line.split_whitespace().filter_map(|t| t.parse().ok()).collect();
                if vals.len() < 9 {
                    continue;
                }
                let mut pts = [[0f32; 2]; 4];
                for k in 0..4 {
                    pts[k] = [vals[1 + 2 * k] * ow as f32, vals[2 + 2 * k] * oh as f32];
                }
                corners.push(pts);
                labels.push(vals[0] as u32);
            }
        }
        out.push(RawObbSample {
            w: ow,
            h: oh,
            rgb,
            corners,
            labels,
        });
    }
    Ok(out)
}

/// 原始 RGB8 缓冲按 plan 做像素增强（flip → HSV 通道增益 → 缩放 resize），
/// 返回增强后图像与其尺寸（坐标侧由各 encode_* 用同一 plan 走同一步骤）。
fn augment_rgb_image(
    w: u32,
    h: u32,
    rgb: &[u8],
    plan: &AugmentPlan,
) -> AvResult<image::RgbImage> {
    let mut buf = rgb.to_vec();
    if plan.flip {
        av_tasks::augment::hflip_rgb(w as usize, h as usize, &mut buf);
    }
    av_tasks::augment::mul_rgb(&mut buf, plan.rgb_gains);
    let mut img = image::RgbImage::from_raw(w, h, buf)
        .ok_or_else(|| AvError::data("RGB8 缓冲长度与宽高不符"))?;
    if plan.scale != 1.0 {
        let (aw, ah) = scaled_dims(w, h, plan.scale);
        if (aw, ah) != (w, h) {
            img = image::imageops::resize(&img, aw, ah, image::imageops::FilterType::Triangle);
        }
    }
    Ok(img)
}

// ---------------------------------------------------------------------------
// 组合增强 raw 域封装（数据增强官二波）：mosaic / mixup
// 像素缩放用与 augment_rgb_image 同款 Triangle stretch，框换算走
// av_tasks::augment 纯函数（拼接线裁剪），坐标约定与单图增强同源。
// ---------------------------------------------------------------------------

/// 4 图 mosaic → 合成原始检测样本：象限 = 锚点 `items[0]` 的原始尺寸，其余项
/// stretch 到同尺寸后按 2×2 网格拼接（TL→TR→BL→BR）；输出画布 = 2×锚点尺寸，
/// 框已换算到画布像素域并按拼接线裁剪（退化目标丢弃）。四图不足时由调用方
/// 重复采样填充（输入是普通引用，同一样本可重复出现）。
pub fn mosaic4_raw(items: [&RawDetectSample; 4]) -> AvResult<RawDetectSample> {
    let (qw, qh) = (items[0].w, items[0].h);
    // 像素先各自缩放到象限尺寸（恒等时直接借用原缓冲，避免多余拷贝）
    let mut resized: [Option<Vec<u8>>; 4] = [None, None, None, None];
    for (slot, it) in resized.iter_mut().zip(items.iter()) {
        if (it.w, it.h) != (qw, qh) {
            let img = image::RgbImage::from_raw(it.w, it.h, it.rgb.clone())
                .ok_or_else(|| AvError::data("RGB8 缓冲长度与宽高不符"))?;
            *slot = Some(
                image::imageops::resize(&img, qw, qh, image::imageops::FilterType::Triangle)
                    .into_raw(),
            );
        }
    }
    let mitems: [av_tasks::augment::MosaicItem<'_>; 4] = std::array::from_fn(|k| {
        let it = items[k];
        av_tasks::augment::MosaicItem {
            rgb: match &resized[k] {
                Some(buf) => buf.as_slice(),
                None => &it.rgb,
            },
            src_w: it.w,
            src_h: it.h,
            boxes: &it.boxes,
            labels: &it.labels,
        }
    });
    let (rgb, boxes, labels) = av_tasks::augment::mosaic_compose(qw, qh, &mitems);
    let (w, h) = av_tasks::augment::mosaic_canvas_dims(qw, qh);
    Ok(RawDetectSample { w, h, rgb, boxes, labels })
}

/// mixup 双样本融合（检测惯例）：b stretch 到 a 的尺寸后像素加权
/// `λ·a + (1−λ)·b`，框与类别取**并集**（两张图的 gt 都保留，YOLO 惯例）。
/// 关键点任务不适用（两套人体拓扑叠加后关键点无语义）——引擎只在检测路径调用。
pub fn mixup_raw(a: &RawDetectSample, b: &RawDetectSample, lam: f32) -> AvResult<RawDetectSample> {
    let brgb: Vec<u8> = if (a.w, a.h) == (b.w, b.h) {
        b.rgb.clone()
    } else {
        let img = image::RgbImage::from_raw(b.w, b.h, b.rgb.clone())
            .ok_or_else(|| AvError::data("RGB8 缓冲长度与宽高不符"))?;
        image::imageops::resize(&img, a.w, a.h, image::imageops::FilterType::Triangle).into_raw()
    };
    Ok(RawDetectSample {
        w: a.w,
        h: a.h,
        rgb: av_tasks::augment::mixup_rgb(&a.rgb, &brgb, lam),
        boxes: a
            .boxes
            .iter()
            .copied()
            .chain(b.boxes.iter().copied())
            .collect(),
        labels: a
            .labels
            .iter()
            .copied()
            .chain(b.labels.iter().copied())
            .collect(),
    })
}

/// 检测原始样本 → 画布样本（先按 plan 增强原图与框，再走与
/// [`load_yolo_dir`] 完全相同的 letterbox/拉伸编码与 map_box 映射）。
pub fn encode_detect_sample(
    raw: &RawDetectSample,
    img_size: u32,
    device: Device,
    mode: ResizeMode,
    plan: &AugmentPlan,
    imagenet_norm: bool,
) -> AvResult<SampleTensor> {
    let mut boxes = raw.boxes.clone();
    for b in boxes.iter_mut() {
        if plan.flip {
            *b = av_tasks::augment::flip_box_xyxy(*b, raw.w as f32);
        }
        if plan.scale != 1.0 {
            for v in b.iter_mut() {
                *v *= plan.scale;
            }
        }
    }
    let (aw, ah) = scaled_dims(raw.w, raw.h, plan.scale);
    let lb = match mode {
        ResizeMode::Letterbox => Some(letterbox(aw, ah, img_size, img_size)),
        ResizeMode::Stretch => None,
    };
    let mapped: Vec<[f32; 4]> = boxes
        .iter()
        .map(|b| match lb {
            // 画布 xyxy：与 load_yolo_dir 同款 map_box 路径
            Some(lb) => {
                let m = lb.map_box(Aabb::new(b[0], b[1], b[2], b[3]));
                [m.x1, m.y1, m.x2, m.y2]
            }
            // 拉伸：增强后图 (aw, ah) → img_size 方形的线性缩放
            None => {
                let (sx, sy) = (img_size as f32 / aw as f32, img_size as f32 / ah as f32);
                [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]
            }
        })
        .collect();
    let img = augment_rgb_image(raw.w, raw.h, &raw.rgb, plan)?;
    let x = rgb_to_input_tensor(&img, img_size, lb, device, imagenet_norm)?;
    Ok(SampleTensor {
        x,
        boxes: mapped,
        labels: raw.labels.clone(),
    })
}

/// 关键点原始样本 → 画布样本（letterbox 固定，与 [`load_cocopose_dir`] 同款
/// 点映射表达式；翻转时 COCO 17 点交换索引，坐标镜像与图像翻转同一 plan）。
pub fn encode_keypoint_sample(
    raw: &RawKeypointSample,
    img_size: u32,
    device: Device,
    plan: &AugmentPlan,
    imagenet_norm: bool,
) -> AvResult<KeypointSample> {
    let (fw, s) = (raw.w as f32, plan.scale);
    let mut boxes = raw.boxes.clone();
    for b in boxes.iter_mut() {
        if plan.flip {
            b[0] = fw - b[0];
        }
        b[0] *= s;
        b[1] *= s;
        b[2] *= s;
        b[3] *= s;
    }
    let mut kpts = raw.kpts.clone();
    for g in kpts.iter_mut() {
        for p in g.iter_mut() {
            if plan.flip {
                p[0] = fw - p[0];
            }
            p[0] *= s;
            p[1] *= s;
        }
    }
    if plan.flip {
        // 坐标已镜像，这里只做 COCO 17 左右语义换位（v 随三元组整体换位）
        av_tasks::augment::swap_coco17_keypoints(&mut kpts);
    }
    let (aw, ah) = scaled_dims(raw.w, raw.h, plan.scale);
    let lb = letterbox(aw, ah, img_size, img_size);
    // 与 load_cocoseg 同款映射表达式（中心 × scale + pad，宽高 × scale）
    let boxes: Vec<[f32; 4]> = boxes
        .iter()
        .map(|b| {
            [
                b[0] * lb.scale + lb.pad_left,
                b[1] * lb.scale + lb.pad_top,
                (b[2] * lb.scale).max(1e-3),
                (b[3] * lb.scale).max(1e-3),
            ]
        })
        .collect();
    let kpts: Vec<Vec<[f32; 3]>> = kpts
        .iter()
        .map(|g| {
            g.iter()
                .map(|p| {
                    [
                        p[0] * lb.scale + lb.pad_left,
                        p[1] * lb.scale + lb.pad_top,
                        p[2],
                    ]
                })
                .collect()
        })
        .collect();
    let img = augment_rgb_image(raw.w, raw.h, &raw.rgb, plan)?;
    let x = rgb_to_input_tensor(&img, img_size, Some(lb), device, imagenet_norm)?;
    Ok(KeypointSample {
        x,
        boxes,
        kpts,
        labels: raw.labels.clone(),
    })
}

/// 分割原始样本 → 画布样本（多边形 → img/4 掩码栅格化与退化跳过规则与
/// [`load_cocoseg_dir`] 一致）。
pub fn encode_seg_sample(
    raw: &RawSegSample,
    img_size: u32,
    device: Device,
    plan: &AugmentPlan,
    imagenet_norm: bool,
) -> AvResult<SegSample> {
    let (mw, mh) = ((img_size / 4) as usize, (img_size / 4) as usize);
    let k = mw as f32 / img_size as f32;
    let (fw, s) = (raw.w as f32, plan.scale);
    let (aw, ah) = scaled_dims(raw.w, raw.h, plan.scale);
    let lb = letterbox(aw, ah, img_size, img_size);
    let (mut masks, mut labels) = (Vec::new(), Vec::new());
    for (poly, label) in raw.polys.iter().zip(&raw.labels) {
        let pts: Vec<[f32; 2]> = poly
            .iter()
            .map(|p| {
                let (mut x, y) = (p[0], p[1]);
                if plan.flip {
                    x = fw - x;
                }
                [
                    (x * s * lb.scale + lb.pad_left) * k,
                    (y * s * lb.scale + lb.pad_top) * k,
                ]
            })
            .collect();
        let mask = rasterize_polygon(&pts, mw, mh);
        if mask.iter().all(|&v| v == 0) {
            continue; // 退化标注（画布外/面积 0）整条跳过
        }
        masks.push(mask);
        labels.push(*label);
    }
    let img = augment_rgb_image(raw.w, raw.h, &raw.rgb, plan)?;
    let x = rgb_to_input_tensor(&img, img_size, Some(lb), device, imagenet_norm)?;
    Ok(SegSample {
        x,
        masks,
        labels,
    })
}

// ---------------------------------------------------------------------------
// 分割缓存样本（数据管线 v2）：letterbox 内容区缓存 + 逐 epoch 小图增强
//
// 旧路径每个 epoch 从原始全分辨率（如 2448×2048）重新 flip/gain/resize，
// 单样本 ~0.5s 且单线程串行——训练全程 GPU 利用率 <5%。缓存路径把「raw →
// img_size 内容贴片」的一次性缩放缓存下来，逐 epoch 只在小图上增强：
// 单样本 ~10ms（40 倍），rayon 并行 + 双缓冲预取后 GPU 不再等数据。
//
// 语义对齐（相对 encode_seg_sample 全分辨率路径，单测锁定）：
// - 掩码：多边形变换公式逐字相同（同一 plan + 同一 letterbox(scaled_dims)
//   映射，坐标不经过像素缓存）→ 逐位一致；
// - 像素：flip 与 Triangle 缩放可交换（核对称）⇒ 贴片翻转 = 全图翻转再
//   letterbox，逐位一致；s = 1 且无增益时整样本与旧路径逐位一致；
//   增益/缩放路径是 u8 定点运算次序差（clamp/混合先后，≤2 LSB、均值 <0.1），
//   以容差单测锁定。
// ---------------------------------------------------------------------------

/// 分割缓存样本：坐标域（原始 w×h 上的归一化多边形）不变；像素侧只保留
/// s=1 letterbox 的内容贴片（cw×ch RGB8）——画布 padding 恒为 114 灰、
/// 不占缓存，由编码期按需合成。
#[derive(Debug)]
pub struct CachedSegSample {
    pub w: u32,
    pub h: u32,
    pub polys: Vec<Vec<[f32; 2]>>,
    pub labels: Vec<u32>,
    /// s=1 letterbox 内容贴片（cw×ch×3 RGB8 交错）
    pub content: Vec<u8>,
    pub cw: u32,
    pub ch: u32,
}

/// raw 分割样本 → 缓存样本（一次性：全分辨率 → 内容贴片的唯一一次缩放）。
pub fn build_seg_cache_sample(raw: &RawSegSample, img_size: u32) -> AvResult<CachedSegSample> {
    let img = image::RgbImage::from_raw(raw.w, raw.h, raw.rgb.to_vec())
        .ok_or_else(|| AvError::data("RGB8 缓冲长度与宽高不符"))?;
    let lb = letterbox(raw.w, raw.h, img_size, 1); // align=1：贴片本身不补边
    let nw = ((raw.w as f32 * lb.scale).round() as u32).clamp(1, img_size);
    let nh = ((raw.h as f32 * lb.scale).round() as u32).clamp(1, img_size);
    let resized = image::imageops::resize(&img, nw, nh, image::imageops::FilterType::Triangle);
    Ok(CachedSegSample {
        w: raw.w,
        h: raw.h,
        polys: raw.polys.clone(),
        labels: raw.labels.clone(),
        content: resized.into_raw(),
        cw: nw,
        ch: nh,
    })
}

/// 并行构建整集缓存（rayon 保序）。返回 (缓存, 字节总量)。
pub fn build_seg_cache(
    raw: &[RawSegSample],
    img_size: u32,
) -> AvResult<(Vec<CachedSegSample>, u64)> {
    let mut bytes = 0u64;
    let out = raw
        .par_iter()
        .map(|r| -> AvResult<CachedSegSample> {
            let c = build_seg_cache_sample(r, img_size)?;
            Ok(c)
        })
        .collect::<AvResult<Vec<_>>>()?;
    for c in &out {
        bytes += c.content.len() as u64;
    }
    Ok((out, bytes))
}

/// 缓存样本 → 画布样本：增强在内容贴片上做，掩码公式与全分辨率路径逐字相同。
pub fn encode_seg_sample_cached(
    c: &CachedSegSample,
    img_size: u32,
    device: Device,
    plan: &AugmentPlan,
    imagenet_norm: bool,
) -> AvResult<SegSample> {
    let (masks, labels) = seg_masks_from_plan(c, img_size, plan);
    // 像素：内容贴片 flip → 增益 →（按需）缩放到与旧路径相同的目标尺寸
    let mut buf = c.content.clone();
    if plan.flip {
        av_tasks::augment::hflip_rgb(c.cw as usize, c.ch as usize, &mut buf);
    }
    av_tasks::augment::mul_rgb(&mut buf, plan.rgb_gains);
    let (nw_t, nh_t, lb) = scaled_content_target(c, img_size, plan.scale);
    let content = if plan.scale != 1.0 {
        let img = image::RgbImage::from_raw(c.cw, c.ch, buf)
            .ok_or_else(|| AvError::data("缓存贴片长度与宽高不符"))?;
        image::imageops::resize(&img, nw_t, nh_t, image::imageops::FilterType::Triangle)
    } else {
        image::RgbImage::from_raw(c.cw, c.ch, buf)
            .ok_or_else(|| AvError::data("缓存贴片长度与宽高不符"))?
    };
    let mut canvas = image::RgbImage::from_pixel(img_size, img_size, image::Rgb([114, 114, 114]));
    image::imageops::overlay(
        &mut canvas,
        &content,
        lb.pad_left.round() as i64,
        lb.pad_top.round() as i64,
    );
    let x = canvas_to_input_tensor(canvas, device, imagenet_norm)?;
    Ok(SegSample {
        x,
        masks,
        labels,
    })
}

/// plan 缩放下内容贴片的目标尺寸与 letterbox 参数（CPU/GPU 两条编码路径
/// 共用：nw/nh 与全分辨率路径「raw 缩放 → letterbox 再缩放」的目标一致，
/// 贴片粘贴偏移同样取 round(pad)）。
fn scaled_content_target(
    c: &CachedSegSample,
    img_size: u32,
    scale: f32,
) -> (u32, u32, Letterbox) {
    let (aw, ah) = scaled_dims(c.w, c.h, scale);
    let lb = letterbox(aw, ah, img_size, img_size);
    let nw = ((aw as f32 * lb.scale).round() as u32).clamp(1, img_size);
    let nh = ((ah as f32 * lb.scale).round() as u32).clamp(1, img_size);
    (nw, nh, lb)
}

/// 掩码侧：多边形按 plan 变换 + 栅格化（与 encode_seg_sample 全分辨率路径
/// 逐字同公式：raw 坐标域 + letterbox(scaled_dims)，不经过像素缓存）。
fn seg_masks_from_plan(
    c: &CachedSegSample,
    img_size: u32,
    plan: &AugmentPlan,
) -> (Vec<Vec<u8>>, Vec<u32>) {
    let (mw, mh) = ((img_size / 4) as usize, (img_size / 4) as usize);
    let k = mw as f32 / img_size as f32;
    let (fw, s) = (c.w as f32, plan.scale);
    let (aw, ah) = scaled_dims(c.w, c.h, plan.scale);
    let lb = letterbox(aw, ah, img_size, img_size);
    let (mut masks, mut labels) = (Vec::new(), Vec::new());
    for (poly, label) in c.polys.iter().zip(&c.labels) {
        let pts: Vec<[f32; 2]> = poly
            .iter()
            .map(|p| {
                let (mut x, y) = (p[0], p[1]);
                if plan.flip {
                    x = fw - x;
                }
                [
                    (x * s * lb.scale + lb.pad_left) * k,
                    (y * s * lb.scale + lb.pad_top) * k,
                ]
            })
            .collect();
        let mask = rasterize_polygon(&pts, mw, mh);
        if mask.iter().all(|&v| v == 0) {
            continue; // 退化标注（画布外/面积 0）整条跳过，与全分辨率路径同规则
        }
        masks.push(mask);
        labels.push(*label);
    }
    (masks, labels)
}

/// 并行编码一批缓存样本（下标与 plan 一一对应，rayon 保序）。
pub fn encode_seg_batch_cached(
    cache: &[CachedSegSample],
    idx: &[usize],
    plans: &[AugmentPlan],
    img_size: u32,
    device: Device,
    imagenet_norm: bool,
) -> AvResult<Vec<SegSample>> {
    idx.par_iter()
        .zip(plans)
        .map(|(&i, plan)| encode_seg_sample_cached(&cache[i], img_size, device, plan, imagenet_norm))
        .collect()
}

// --- 显存驻留（T0）：整集画布一次上传，逐 epoch 增强在 GPU 张量域 ---

/// 缓存样本 → [0,1] f32 画布（s=1，内容贴在 114/255 灰底上的整画布），
/// 堆成 [N,3,S,S]。显存占用 = N×3×S²×4B。
pub fn build_seg_canvas_stack(
    cache: &[CachedSegSample],
    img_size: u32,
    device: Device,
) -> AvResult<Tensor> {
    let s = img_size as i64;
    let gray = 114.0f32 / 255.0;
    let mut bufs: Vec<Tensor> = Vec::with_capacity(cache.len());
    for c in cache {
        let lb = letterbox(c.w, c.h, img_size, img_size);
        let (pl, pt) = (lb.pad_left.round() as i64, lb.pad_top.round() as i64);
        let img = image::RgbImage::from_raw(c.cw, c.ch, c.content.clone())
            .ok_or_else(|| AvError::data("缓存贴片长度与宽高不符"))?;
        let content = canvas_to_input_tensor(img, device, false)?; // [0,1]
        let canvas = Tensor::full([3, s, s], gray as f64, (Kind::Float, device));
        canvas
            .narrow(2, pl, c.cw as i64)
            .narrow(1, pt, c.ch as i64)
            .copy_(&content);
        bufs.push(canvas);
    }
    Ok(Tensor::stack(&bufs, 0))
}

/// 显存驻留编码：画布从堆里取（零 H2D），flip/增益/缩放在 GPU 张量域做。
/// 增益在 f32 [0,1] 域 clamp（≈ u8 饱和乘，差 ≤1/255）；缩放用 bilinear
/// （与 CPU Triangle 滤波差异 ≤2 LSB 量级）；掩码仍在 CPU 栅格化（公式同源）。
pub fn encode_seg_sample_gpu(
    stack: &Tensor,
    i: u32,
    c: &CachedSegSample,
    img_size: u32,
    plan: &AugmentPlan,
    imagenet_norm: bool,
) -> AvResult<SegSample> {
    let (masks, labels) = seg_masks_from_plan(c, img_size, plan);
    // s=1 内容区几何（flip/增益的窄区域）
    let lb1 = letterbox(c.w, c.h, img_size, img_size);
    let (pl1, pt1) = (lb1.pad_left.round() as i64, lb1.pad_top.round() as i64);
    let mut x = stack.select(0, i as i64).copy(); // [3,S,S] 视图共享存储，copy 脱离
    // flip：只翻内容区（padding 灰底对称不可见），与「全图翻转再 letterbox」等价
    if plan.flip {
        let region = x.narrow(2, pl1, c.cw as i64).copy();
        x.narrow(2, pl1, c.cw as i64)
            .copy_(&region.flip([2i64]));
    }
    // 增益：内容区逐通道乘 + clamp（画布 padding 保持 114 灰）
    if plan.rgb_gains != [1.0; 3] {
        let gains = Tensor::from_slice(&plan.rgb_gains)
            .to_kind(Kind::Float)
            .to_device(x.device())
            .reshape([3i64, 1, 1]);
        let region = x.narrow(2, pl1, c.cw as i64).copy() * gains;
        x.narrow(2, pl1, c.cw as i64).copy_(&region.clamp(0.0, 1.0));
    }
    // 缩放：内容区双线性重采样 → 合成到新画布的 round(pad) 偏移
    if plan.scale != 1.0 {
        let (nw, nh, lb) = scaled_content_target(c, img_size, plan.scale);
        let region = x
            .narrow(2, pl1, c.cw as i64)
            .narrow(1, pt1, c.ch as i64)
            .copy();
        let resized = region
            .unsqueeze(0)
            .upsample_bilinear2d([nh as i64, nw as i64], false, None, None)
            .squeeze_dim(0);
        let gray = 114.0f32 / 255.0;
        let s = img_size as i64;
        let canvas = Tensor::full([3, s, s], gray as f64, (Kind::Float, x.device()));
        canvas
            .narrow(2, lb.pad_left.round() as i64, nw as i64)
            .narrow(1, lb.pad_top.round() as i64, nh as i64)
            .copy_(&resized);
        x = canvas;
    }
    if imagenet_norm {
        const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
        const STD: [f32; 3] = [0.229, 0.224, 0.225];
        let mean = Tensor::from_slice(&MEAN)
            .to_kind(Kind::Float)
            .to_device(x.device())
            .reshape([3i64, 1, 1]);
        let std = Tensor::from_slice(&STD)
            .to_kind(Kind::Float)
            .to_device(x.device())
            .reshape([3i64, 1, 1]);
        x = (x - mean) / std;
    }
    Ok(SegSample {
        x,
        masks,
        labels,
    })
}

/// OBB 原始样本 → 画布样本（4 角点经增强 + letterbox 等比映射后推导
/// cxcywhθ，与 [`load_dota_dir`] 同款公式；退化（w/h < 1px）实例跳过）。
pub fn encode_obb_sample(
    raw: &RawObbSample,
    img_size: u32,
    device: Device,
    plan: &AugmentPlan,
    imagenet_norm: bool,
) -> AvResult<ObbSample> {
    use av_core::conventions::AngleDomain;

    let (fw, s) = (raw.w as f32, plan.scale);
    let (aw, ah) = scaled_dims(raw.w, raw.h, plan.scale);
    let lb = letterbox(aw, ah, img_size, img_size);
    let (mut boxes, mut labels) = (Vec::new(), Vec::new());
    for (corners, label) in raw.corners.iter().zip(&raw.labels) {
        let mut pts = [[0f32; 2]; 4];
        for (dst, src) in pts.iter_mut().zip(corners) {
            let (mut x, y) = (src[0], src[1]);
            if plan.flip {
                x = fw - x;
            }
            *dst = [
                x * s * lb.scale + lb.pad_left,
                y * s * lb.scale + lb.pad_top,
            ];
        }
        // 与 load_dota_dir 同款 cxcywhθ 推导（镜像后角度随角点自然更新）
        let cx = (pts[0][0] + pts[2][0]) / 2.0;
        let cy = (pts[0][1] + pts[2][1]) / 2.0;
        let dx = pts[1][0] - pts[0][0];
        let dy = pts[1][1] - pts[0][1];
        let w = (dx * dx + dy * dy).sqrt();
        let h = {
            let ex = pts[3][0] - pts[0][0];
            let ey = pts[3][1] - pts[0][1];
            (ex * ex + ey * ey).sqrt()
        };
        if w < 1.0 || h < 1.0 {
            continue;
        }
        let theta = dy.atan2(dx);
        boxes.push([cx, cy, w, h, AngleDomain::Le90.normalize(theta)]);
        labels.push(*label);
    }
    let img = augment_rgb_image(raw.w, raw.h, &raw.rgb, plan)?;
    let x = rgb_to_input_tensor(&img, img_size, Some(lb), device, imagenet_norm)?;
    Ok(ObbSample {
        x,
        boxes,
        labels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 已知宽高比（640x480 → 320 方形画布）下框映射的往返精度（1px 容差）。
    #[test]
    fn letterbox_label_mapping_roundtrip() {
        // 640x480 → scale=0.5，内容 320x240 居中：pad_left=0，pad_top=40
        let lb = letterbox(640, 480, 320, 320);
        assert_eq!((lb.dst_w, lb.dst_h), (320, 320));
        assert!((lb.scale - 0.5).abs() < 1e-6);

        let g = Aabb::new(10.0, 20.0, 100.0, 80.0);
        let m = lb.map_box(g);
        assert!((m.x1 - 5.0).abs() < 1e-4, "x1={}", m.x1);
        assert!((m.y1 - 50.0).abs() < 1e-4, "y1={}", m.y1); // 20*0.5 + 40
        assert!((m.x2 - 50.0).abs() < 1e-4, "x2={}", m.x2);
        assert!((m.y2 - 80.0).abs() < 1e-4, "y2={}", m.y2);

        // 往返：画布坐标还原图坐标，1px 内
        let back = lb.restore_box(m, 640, 480);
        assert!((back.x1 - g.x1).abs() <= 1.0, "back={back:?}");
        assert!((back.y1 - g.y1).abs() <= 1.0, "back={back:?}");
        assert!((back.x2 - g.x2).abs() <= 1.0, "back={back:?}");
        assert!((back.y2 - g.y2).abs() <= 1.0, "back={back:?}");

        // 非整除缩放（427x640 → 320）：内容 214 宽、pad_left=53，仍需往返闭合
        let lb2 = letterbox(427, 640, 320, 320);
        assert_eq!((lb2.dst_w, lb2.dst_h), (320, 320));
        let g2 = Aabb::new(3.0, 7.0, 424.0, 633.0);
        let back2 = lb2.restore_box(lb2.map_box(g2), 427, 640);
        assert!((back2.x1 - g2.x1).abs() <= 1.0, "back2={back2:?}");
        assert!((back2.x2 - g2.x2).abs() <= 1.0, "back2={back2:?}");
        assert!((back2.y2 - g2.y2).abs() <= 1.0, "back2={back2:?}");
    }

    /// letterbox 解码路径端到端：64x32 全红图 → 32 画布，补边区为 114 灰、内容区为红。
    #[test]
    fn letterbox_decode_pads_gray_keeps_content() {
        let dir = std::env::temp_dir().join(format!("av-ds-lb-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("red.png");
        image::RgbImage::from_pixel(64, 32, image::Rgb([255, 0, 0]))
            .save(&p)
            .unwrap();

        // 64x32 → scale=0.5，内容 32x16，pad_top=8：画布第 0 行是灰、第 16 行是红
        let x = decode_image_tensor_with_mode(&p, 32, Device::Cpu, ResizeMode::Letterbox, false).unwrap();
        let px = |c: usize, y: usize, xx: usize| x.double_value(&[c as i64, y as i64, xx as i64]);
        // 补边（画布 (0,0)）：114/255 灰，三通道一致
        for c in 0..3 {
            assert!(
                (px(c, 0, 0) - 114.0 / 255.0).abs() < 1e-6,
                "pad c{c}={}",
                px(c, 0, 0)
            );
        }
        // 内容（画布 (16,16) 在内容行 8..24 内）：红 (1,0,0)
        assert!((px(0, 16, 16) - 1.0).abs() < 1e-6, "r={}", px(0, 16, 16));
        assert!(px(1, 16, 16) < 1e-6, "g={}", px(1, 16, 16));
        assert!(px(2, 16, 16) < 1e-6, "b={}", px(2, 16, 16));

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// stretch 模式保留旧行为：全图拉伸，无补边（角落不是 114 灰）。
    #[test]
    fn stretch_mode_still_available() {
        let dir = std::env::temp_dir().join(format!("av-ds-st-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("red.png");
        image::RgbImage::from_pixel(64, 32, image::Rgb([255, 0, 0]))
            .save(&p)
            .unwrap();

        let x = decode_image_tensor_with_mode(&p, 32, Device::Cpu, ResizeMode::Stretch, false).unwrap();
        let px = |c: usize, y: usize, xx: usize| x.double_value(&[c as i64, y as i64, xx as i64]);
        // 32x32 全是红，无灰补边
        assert!((px(0, 0, 0) - 1.0).abs() < 1e-6);
        assert!(px(2, 31, 31) < 1e-6);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// imagenet_norm = true：红像素 (255,0,0) 映射到 ImageNet 域
    /// ((1−mean)/std)；false（默认）保持 [0,1]。锁定两域语义与预训练域修复。
    #[test]
    fn imagenet_norm_maps_red_pixel_to_imagenet_domain() {
        let dir = std::env::temp_dir().join(format!("av-ds-norm-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("red.png");
        image::RgbImage::from_pixel(8, 8, image::Rgb([255, 0, 0]))
            .save(&p)
            .unwrap();

        // 默认 false：[0,1] 历史语义（零变化）
        let x01 = decode_image_tensor_with_mode(&p, 8, Device::Cpu, ResizeMode::Stretch, false).unwrap();
        assert!((x01.double_value(&[0, 0, 0]) - 1.0).abs() < 1e-6);
        assert!(x01.double_value(&[1, 0, 0]).abs() < 1e-6);

        // true：ImageNet mean/std 域：(1−0.485)/0.229, (0−0.456)/0.224, (0−0.406)/0.225
        let xn = decode_image_tensor_with_mode(&p, 8, Device::Cpu, ResizeMode::Stretch, true).unwrap();
        let expect = [
            (1.0f32 - 0.485) / 0.229,
            (0.0f32 - 0.456) / 0.224,
            (0.0f32 - 0.406) / 0.225,
        ];
        for (c, &e) in expect.iter().enumerate() {
            assert!(
                (xn.double_value(&[c as i64, 4, 4]) - e as f64).abs() < 1e-6,
                "c{c}={}",
                xn.double_value(&[c as i64, 4, 4])
            );
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// 目录加载：letterbox 模式下标注框落在画布空间（手工构造小数据集）。
    #[test]
    fn load_yolo_dir_letterbox_maps_boxes_to_canvas() {
        let dir = std::env::temp_dir().join(format!("av-ds-dir-test-{}", std::process::id()));
        let (img_dir, lbl_dir) = (dir.join("images/train"), dir.join("labels/train"));
        std::fs::create_dir_all(&img_dir).unwrap();
        std::fs::create_dir_all(&lbl_dir).unwrap();
        // 64x32 全红图，标注归一化 cxcywh：整图框 (0.5, 0.5, 1.0, 1.0)
        image::RgbImage::from_pixel(64, 32, image::Rgb([255, 0, 0]))
            .save(img_dir.join("a.png"))
            .unwrap();
        std::fs::write(lbl_dir.join("a.txt"), "0 0.5 0.5 1.0 1.0\n").unwrap();

        let samples =
            load_yolo_dir_with_mode(&dir, "train", 32, Device::Cpu, ResizeMode::Letterbox, false)
                .unwrap();
        assert_eq!(samples.len(), 1);
        let b = samples[0].boxes[0];
        // 原图整图框 → 画布内容区 (0,8)-(32,24)
        assert!((b[0] - 0.0).abs() <= 1.0, "x1={}", b[0]);
        assert!((b[1] - 8.0).abs() <= 1.0, "y1={}", b[1]);
        assert!((b[2] - 32.0).abs() <= 1.0, "x2={}", b[2]);
        assert!((b[3] - 24.0).abs() <= 1.0, "y2={}", b[3]);
        assert_eq!(samples[0].labels, vec![0]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// avpack 打包 → 加载往返：2 图 2 标注（+1 val 图验证 split 过滤），
    /// 样本数 / 条目名排序 / letterbox 框映射手算对照；raw 路径给原图像素框。
    #[test]
    fn load_yolo_avpack_roundtrip_names_and_boxes() {
        let dir = std::env::temp_dir().join(format!("av-ds-avpack-{}", std::process::id()));
        let (img_dir, lbl_dir) = (dir.join("images/train"), dir.join("labels/train"));
        let (vimg, vlbl) = (dir.join("images/val"), dir.join("labels/val"));
        for d in [&img_dir, &lbl_dir, &vimg, &vlbl] {
            std::fs::create_dir_all(d).unwrap();
        }
        // 图 a 64x32 红（整图框 cls0）；图 b 32x32 蓝（1/4 框 cls3）；val 一张绿图
        image::RgbImage::from_pixel(64, 32, image::Rgb([255, 0, 0]))
            .save(img_dir.join("a.png"))
            .unwrap();
        std::fs::write(lbl_dir.join("a.txt"), "0 0.5 0.5 1.0 1.0\n").unwrap();
        image::RgbImage::from_pixel(32, 32, image::Rgb([0, 0, 255]))
            .save(img_dir.join("b.png"))
            .unwrap();
        std::fs::write(lbl_dir.join("b.txt"), "3 0.25 0.5 0.5 0.5\n").unwrap();
        image::RgbImage::from_pixel(16, 16, image::Rgb([0, 255, 0]))
            .save(vimg.join("c.png"))
            .unwrap();
        std::fs::write(vlbl.join("c.txt"), "1 0.5 0.5 0.5 0.5\n").unwrap();

        let out = dir.join("ds.avpack");
        let (count, _) = crate::avpack::pack_dir(&dir, &out).unwrap();
        assert_eq!(count, 6, "a/b 图 + a/b 标注 + c 图 + c 标注");

        let named = load_yolo_avpack_named(&out, "train", 32, Device::Cpu, false).unwrap();
        assert_eq!(
            named.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>(),
            vec!["images/train/a.png", "images/train/b.png"],
            "train split 只含 train 图，按名字排序"
        );

        // a：64x32 整图框 → 画布内容区 (0,8)-(32,24)（scale=0.5、pad_top=8）
        let a = &named[0].1;
        assert_eq!(a.labels, vec![0]);
        assert!((a.boxes[0][0] - 0.0).abs() <= 1.0, "a x1={}", a.boxes[0][0]);
        assert!((a.boxes[0][1] - 8.0).abs() <= 1.0, "a y1={}", a.boxes[0][1]);
        assert!((a.boxes[0][2] - 32.0).abs() <= 1.0, "a x2={}", a.boxes[0][2]);
        assert!((a.boxes[0][3] - 24.0).abs() <= 1.0, "a y2={}", a.boxes[0][3]);

        // b：32x32 等比无 pad，cxcywh (0.25,0.5,0.5,0.5) → xyxy (0,8)-(16,24)
        let b = &named[1].1;
        assert_eq!(b.labels, vec![3]);
        for (got, want) in b.boxes[0].iter().zip([0.0f32, 8.0, 16.0, 24.0]) {
            assert!((got - want).abs() < 1e-4, "b box {got} vs {want}");
        }

        // 批堆叠形状 [B,3,S,S]
        let samples: Vec<SampleTensor> = named.into_iter().map(|(_, s)| s).collect();
        assert_eq!(stack_samples(&samples).unwrap().size(), vec![2, 3, 32, 32]);

        // split 过滤：val 只剩绿图 1 张
        let val = load_yolo_avpack(&out, "val", 32, Device::Cpu, false).unwrap();
        assert_eq!(val.len(), 1);
        assert_eq!(val[0].labels, vec![1]);

        // raw 路径（增强用）：原图像素 xyxy
        let raw = load_yolo_avpack_raw(&out, "train").unwrap();
        assert_eq!(raw.len(), 2);
        assert_eq!(raw[0].boxes, vec![[0.0, 0.0, 64.0, 32.0]]);
        assert_eq!(raw[1].boxes, vec![[0.0, 8.0, 16.0, 24.0]]);

        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_file(&out);
    }

    /// 手工构造 2 类 ImageFolder（2 张小图）：wnid 排序 → 类 id 0/1，
    /// 拉伸 resize、RGB [0,1]、标签平行向量与映射内容正确。
    #[test]
    fn load_imagefolder_derives_sorted_class_ids() {
        let dir = std::env::temp_dir().join(format!("av-ds-if-test-{}", std::process::id()));
        let train = dir.join("train");
        std::fs::create_dir_all(train.join("n02102040")).unwrap(); // 排序在后 → 类 1
        std::fs::create_dir_all(train.join("n01440764")).unwrap(); // 排序在前 → 类 0
        // 16x8 全红图（非方形，验证拉伸 resize 到方形；用 PNG 避免 JPEG 有损量化）
        image::RgbImage::from_pixel(16, 8, image::Rgb([255, 0, 0]))
            .save(train.join("n01440764").join("a.png"))
            .unwrap();
        // 8x16 全蓝图
        image::RgbImage::from_pixel(8, 16, image::Rgb([0, 0, 255]))
            .save(train.join("n02102040").join("b.png"))
            .unwrap();

        let (samples, labels, map) =
            load_imagefolder(&dir, "train", 32, true, Device::Cpu, false)
                .expect("ImageFolder 应可加载");
        assert_eq!(samples.len(), 2);
        assert_eq!(labels, vec![0, 1]);
        assert_eq!(map.len(), 2);
        assert_eq!(map["n01440764"], 0);
        assert_eq!(map["n02102040"], 1);

        // 拉伸 resize：全 32x32 红 / 蓝，无 letterbox 灰补边
        let px = |s: &ClassifySample, c: usize, y: usize, x: usize| {
            s.x.double_value(&[c as i64, y as i64, x as i64])
        };
        assert!((px(&samples[0], 0, 0, 0) - 1.0).abs() < 1e-6, "应为红");
        assert!((px(&samples[0], 1, 31, 31) - 0.0).abs() < 1e-6);
        assert!((px(&samples[1], 2, 16, 16) - 1.0).abs() < 1e-6, "应为蓝");
        assert!((px(&samples[1], 0, 0, 0) - 0.0).abs() < 1e-6);

        // stack_classify 形状 [B,3,S,S]
        let batch = stack_classify(&samples).unwrap();
        assert_eq!(batch.size(), vec![2, 3, 32, 32]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// val split 复用 train 词表：给定映射加载时标签取映射值；
    /// 出现映射外的 wnid 时报可读错误。
    #[test]
    fn load_imagefolder_reuses_train_class_map() {
        let dir = std::env::temp_dir().join(format!("av-ds-ifmap-test-{}", std::process::id()));
        let (tr, va) = (dir.join("train"), dir.join("val"));
        for base in [&tr, &va] {
            std::fs::create_dir_all(base.join("n01440764")).unwrap();
            std::fs::create_dir_all(base.join("n02102040")).unwrap();
        }
        image::RgbImage::from_pixel(8, 8, image::Rgb([255, 0, 0]))
            .save(tr.join("n01440764").join("a.png"))
            .unwrap();
        image::RgbImage::from_pixel(8, 8, image::Rgb([0, 255, 0]))
            .save(tr.join("n02102040").join("b.png"))
            .unwrap();
        image::RgbImage::from_pixel(8, 8, image::Rgb([0, 0, 255]))
            .save(va.join("n02102040").join("c.png"))
            .unwrap();

        let (_, _, train_map) =
            load_imagefolder(&dir, "train", 16, true, Device::Cpu, false).unwrap();
        let (_, val_labels, _) =
            load_imagefolder_with_classes(&dir, "val", 16, Some(&train_map), Device::Cpu, false)
                .unwrap();
        assert_eq!(val_labels, vec![1], "val 复用 train 映射（n02102040 → 1）");

        // 映射外的 wnid → 报错
        let mut partial = HashMap::new();
        partial.insert("n01440764".to_string(), 0u32);
        let err =
            load_imagefolder_with_classes(&dir, "val", 16, Some(&partial), Device::Cpu, false)
                .unwrap_err();
        assert!(err.to_string().contains("n02102040"), "got: {err}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// 空 split / 缺目录给出可读错误。
    #[test]
    fn load_imagefolder_missing_split_errors() {
        let dir = std::env::temp_dir().join(format!("av-ds-ifmiss-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let err = load_imagefolder(&dir, "train", 16, true, Device::Cpu, false).unwrap_err();
        assert!(err.to_string().contains("不存在"), "got: {err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    // -----------------------------------------------------------------
    // 实例分割：多边形栅格化 + COCO seg 目录加载
    // -----------------------------------------------------------------

    /// 矩形多边形栅格化手算对照：[(2,2),(10,2),(10,8),(2,8)] @16×16。
    /// 像素中心 (x+0.5, y+0.5) 落在开区间 (2,10)×(2,8) → x∈[2,9]，y∈[2,7]，共 8×6=48。
    #[test]
    fn rasterize_polygon_rectangle_hand_computed() {
        let pts = [[2.0, 2.0], [10.0, 2.0], [10.0, 8.0], [2.0, 8.0]];
        let m = rasterize_polygon(&pts, 16, 16);
        assert_eq!(m.iter().filter(|&&v| v == 1).count(), 48);
        // 覆盖角点内部
        assert_eq!(m[2 * 16 + 2], 1, "(2,2) 应覆盖");
        assert_eq!(m[7 * 16 + 9], 1, "(9,7) 应覆盖");
        // 边界外沿不覆盖（像素中心在边界上/外）
        assert_eq!(m[2 * 16 + 10], 0, "(10,2) 中心 x=10.5 在界外");
        assert_eq!(m[8 * 16 + 2], 0, "(2,8) 中心 y=8.5 在界外");
        assert_eq!(m[1 * 16 + 2], 0);
        assert_eq!(m[2 * 16 + 1], 0);
    }

    /// 三角形：[(0,0),(4,0),(0,4)] —— 中心 (x+0.5,y+0.5) 满足 x+y+1 < 4 的像素覆盖。
    #[test]
    fn rasterize_polygon_triangle_half_open() {
        let pts = [[0.0, 0.0], [4.0, 0.0], [0.0, 4.0]];
        let m = rasterize_polygon(&pts, 8, 8);
        let cnt = m.iter().filter(|&&v| v == 1).count();
        // 逐像素手算：y 行覆盖 x = 0..(3-y)（中心 x+0.5 < 4-y-0.5+1 → x < 4-y-1+0.5）
        // y=0: x+0.5+y+0.5<4 → x<3 → {0,1,2}；y=1: x<2 → {0,1}；y=2: x<1 → {0}
        assert_eq!(cnt, 6);
        assert_eq!(m[0], 1);
        assert_eq!(m[2], 1); // (2,0)
        assert_eq!(m[1 * 8 + 1], 1); // (1,1)
        assert_eq!(m[2 * 8], 1); // (0,2)
        assert_eq!(m[3 * 8], 0); // (0,3) 中心 3.5+0.5=4 边界上 → 不覆盖
    }

    /// 退化输入（<3 点）与画布外多边形安全返回。
    #[test]
    fn rasterize_polygon_degenerate_inputs() {
        assert!(rasterize_polygon(&[], 4, 4).iter().all(|&v| v == 0));
        assert!(rasterize_polygon(&[[1.0, 1.0], [3.0, 3.0]], 4, 4)
            .iter()
            .all(|&v| v == 0));
        // 完全在画布外
        let out = [[100.0, 100.0], [120.0, 100.0], [120.0, 120.0], [100.0, 120.0]];
        assert!(rasterize_polygon(&out, 4, 4).iter().all(|&v| v == 0));
        // 部分越界：覆盖部分被裁进画布
        let half = [[-4.0, -4.0], [4.0, -4.0], [4.0, 4.0], [-4.0, 4.0]];
        let m = rasterize_polygon(&half, 4, 4);
        assert_eq!(m.iter().filter(|&&v| v == 1).count(), 16, "整画布被覆盖");
    }

    /// 目录加载：多边形行栅格化为 img/4 掩码；纯检测框行（5 值）被跳过。
    #[test]
    fn load_cocoseg_dir_rasterizes_and_skips_box_lines() {
        let dir = std::env::temp_dir().join(format!("av-ds-cseg-test-{}", std::process::id()));
        let (img_dir, lbl_dir) = (dir.join("images/train"), dir.join("labels/train"));
        std::fs::create_dir_all(&img_dir).unwrap();
        std::fs::create_dir_all(&lbl_dir).unwrap();
        // 64×32 全红图 → 32 画布（scale=0.5，pad_top=8）；img=32 → 掩码画布 8×8
        image::RgbImage::from_pixel(64, 32, image::Rgb([255, 0, 0]))
            .save(img_dir.join("a.png"))
            .unwrap();
        // 行 1：归一化多边形（画布内容区中一个方形：原图 x∈[16,48], y∈[8,24]
        // → 画布 (8,12)-(24,20) → 掩码 (2,3)-(6,5)）；行 2：纯检测框（5 值）必须跳过
        std::fs::write(
            lbl_dir.join("a.txt"),
            concat!(
                "7 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n",
                "3 0.5 0.5 0.5 0.5\n",
                "9 0.1 0.1 0.2 0.2 0.3 0.3 0.1 0.2 0.9 0.9 0.1 0.9\n",
            ),
        )
        .unwrap();

        let samples = load_cocoseg_dir(&dir, "train", 32, Device::Cpu, false).unwrap();
        assert_eq!(samples.len(), 1);
        let s = &samples[0];
        // 三行标注：多边形 + 纯检测框（5 值，跳过）+ 多边形 → 共 2 实例
        assert_eq!(s.labels, vec![7, 9]);
        assert_eq!(s.masks.len(), 2);
        // 第一条多边形：原图 (16,8)-(48,24) → 画布 (8,12)-(24,20) → 掩码 (2,3)-(6,5)
        // → 中心落在开区间的像素 x∈{2..5}, y∈{3,4}，恰 8 个
        let m0 = &s.masks[0];
        assert_eq!(m0.len(), 8 * 8);
        let cnt0 = m0.iter().filter(|&&v| v == 1).count();
        assert_eq!(cnt0, 8, "掩码 (2,3)-(6,5) 应覆盖 4×2=8 像素，实际 {cnt0}");
        assert_eq!(m0[3 * 8 + 3], 1, "中心点应覆盖");
        assert_eq!(m0[0], 0, "画布角落（灰边）不应覆盖");
        // 第二条（多点多边形）也应产出非空掩码
        assert!(s.masks[1].iter().any(|&v| v == 1));

        // stack_seg_samples 形状 [B,3,S,S]
        let x = stack_seg_samples(&samples).unwrap();
        assert_eq!(x.size(), vec![1, 3, 32, 32]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    // -----------------------------------------------------------------
    // 关键点：COCO 姿态目录加载（letterbox 关键点映射）
    // -----------------------------------------------------------------

    /// 手工小图 + 标注的映射对照：64×32 → 32 画布（scale=0.5，pad_top=8）。
    /// 关键点 x' = x·64·0.5 + pad_left、y' = y·32·0.5 + pad_top；框映射 cxcywh。
    #[test]
    fn load_cocopose_dir_maps_kpts_and_boxes_to_canvas() {
        let dir = std::env::temp_dir().join(format!("av-ds-kp-test-{}", std::process::id()));
        let (img_dir, lbl_dir) = (dir.join("images/train"), dir.join("labels/train"));
        std::fs::create_dir_all(&img_dir).unwrap();
        std::fs::create_dir_all(&lbl_dir).unwrap();
        // 64×32 全红图；图 a：K=2（一实例）；图 b：K=3（验证行内 K 可变推导）
        image::RgbImage::from_pixel(64, 32, image::Rgb([255, 0, 0]))
            .save(img_dir.join("a.png"))
            .unwrap();
        image::RgbImage::from_pixel(64, 32, image::Rgb([0, 255, 0]))
            .save(img_dir.join("b.png"))
            .unwrap();
        // 图 a：cls=0 cx=0.5 cy=0.5 w=0.5 h=0.5
        //   kpt0 (0.25,0.25) v=2 → 画布 (0.25·64·0.5, 0.25·32·0.5+8) = (8, 12)
        //   kpt1 (0.5,0.5)  v=0 → 画布 (16, 16)（不可见，坐标无意义仅记录）
        //   框 → 画布中心 (16·1+0, 8·1+8)=(16,16)，wh (16, 8)
        std::fs::write(
            lbl_dir.join("a.txt"),
            "0 0.5 0.5 0.5 0.5 0.25 0.25 2.0 0.5 0.5 0.0\n",
        )
        .unwrap();
        // 图 b：K=3 一实例（token 数推导行内点数）
        std::fs::write(
            lbl_dir.join("b.txt"),
            "0 0.5 0.5 1.0 1.0 0.1 0.2 2.0 0.3 0.4 1.0 0.5 0.6 2.0\n",
        )
        .unwrap();

        let samples = load_cocopose_dir(&dir, "train", 32, Device::Cpu, false).unwrap();
        assert_eq!(samples.len(), 2);

        let a = &samples[0];
        assert_eq!(a.labels, vec![0]);
        assert_eq!(a.boxes.len(), 1);
        // 框：中心 (16,16)，宽高 (16,8)
        assert!((a.boxes[0][0] - 16.0).abs() < 1e-4, "cx={}", a.boxes[0][0]);
        assert!((a.boxes[0][1] - 16.0).abs() < 1e-4, "cy={}", a.boxes[0][1]);
        assert!((a.boxes[0][2] - 16.0).abs() < 1e-4, "w={}", a.boxes[0][2]);
        assert!((a.boxes[0][3] - 8.0).abs() < 1e-4, "h={}", a.boxes[0][3]);
        // 关键点映射 + 可见性保留
        assert_eq!(a.kpts[0].len(), 2);
        assert!((a.kpts[0][0][0] - 8.0).abs() < 1e-4, "kx={}", a.kpts[0][0][0]);
        assert!((a.kpts[0][0][1] - 12.0).abs() < 1e-4, "ky={}", a.kpts[0][0][1]);
        assert_eq!(a.kpts[0][0][2], 2.0);
        assert!((a.kpts[0][1][0] - 16.0).abs() < 1e-4);
        assert!((a.kpts[0][1][1] - 16.0).abs() < 1e-4);
        assert_eq!(a.kpts[0][1][2], 0.0, "v=0 应原样保留");

        let b = &samples[1];
        assert_eq!(b.kpts[0].len(), 3, "K 应由行内 token 数推导");
        // kpt (0.1, 0.2) v=2 → (0.1·64·0.5, 0.2·32·0.5+8) = (3.2, 11.2)
        assert!((b.kpts[0][0][0] - 3.2).abs() < 1e-4);
        assert!((b.kpts[0][0][1] - 11.2).abs() < 1e-4);
        assert_eq!(b.kpts[0][1][2], 1.0, "v=1（遮挡）应保留");
        // 整图框 → 画布内容区 (0,8)-(32,24)
        assert!((b.boxes[0][3] - 16.0).abs() < 1e-4);

        // stack_kp_samples 形状 [B,3,S,S]
        let x = stack_kp_samples(&samples).unwrap();
        assert_eq!(x.size(), vec![2, 3, 32, 32]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// 非 3 倍数尾行 / 空 label 文件被安全跳过（不产生实例、不报错）。
    #[test]
    fn load_cocopose_dir_skips_malformed_lines() {
        let dir = std::env::temp_dir().join(format!("av-ds-kpbad-test-{}", std::process::id()));
        let (img_dir, lbl_dir) = (dir.join("images/train"), dir.join("labels/train"));
        std::fs::create_dir_all(&img_dir).unwrap();
        std::fs::create_dir_all(&lbl_dir).unwrap();
        image::RgbImage::from_pixel(32, 32, image::Rgb([255, 0, 0]))
            .save(img_dir.join("a.png"))
            .unwrap();
        std::fs::write(
            lbl_dir.join("a.txt"),
            concat!(
                "0 0.5 0.5 0.5 0.5 0.1 0.1\n",      // 尾部 2 值，凑不出 3K → 跳过
                "\n",                                 // 空行 → 跳过
                "0 0.5 0.5 0.5 0.5 0.2 0.2 2.0\n",   // 合法 K=1 → 保留
            ),
        )
        .unwrap();

        let samples = load_cocopose_dir(&dir, "train", 32, Device::Cpu, false).unwrap();
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].kpts.len(), 1);
        assert_eq!(samples[0].kpts[0].len(), 1);
        // (0.2,0.2) → 画布 (6.4, 6.4)（32×32 等比、无 pad）
        assert!((samples[0].kpts[0][0][0] - 6.4).abs() < 1e-4);

        let _ = std::fs::remove_dir_all(&dir);
    }

    // -----------------------------------------------------------------
    // 训练期增强：raw 加载器 + encode_*（坐标同步 / none 恒等）
    // -----------------------------------------------------------------

    fn tensor_max_diff(a: &Tensor, b: &Tensor) -> f64 {
        (a - b).abs().max().double_value(&[])
    }

    /// 关键点：raw + encode(none) 必须与既有 plain 加载器逐位一致
    /// （张量 / 框 / 关键点 / 可见性），保证「关增强 = 历史行为」。
    #[test]
    fn encode_keypoint_none_matches_plain_loader() {
        let dir = std::env::temp_dir().join(format!("av-ds-augkpn-{}", std::process::id()));
        let (img_dir, lbl_dir) = (dir.join("images/train"), dir.join("labels/train"));
        std::fs::create_dir_all(&img_dir).unwrap();
        std::fs::create_dir_all(&lbl_dir).unwrap();
        image::RgbImage::from_pixel(64, 32, image::Rgb([255, 0, 0]))
            .save(img_dir.join("a.png"))
            .unwrap();
        std::fs::write(
            lbl_dir.join("a.txt"),
            "0 0.5 0.5 0.5 0.5 0.25 0.25 2.0 0.5 0.5 0.0\n",
        )
        .unwrap();

        let plain = load_cocopose_dir(&dir, "train", 32, Device::Cpu, false).unwrap();
        let raw = load_cocopose_dir_raw(&dir, "train").unwrap();
        assert_eq!(raw.len(), 1);
        let enc = encode_keypoint_sample(&raw[0], 32, Device::Cpu, &AugmentPlan::none(), false)
            .unwrap();
        assert_eq!(tensor_max_diff(&plain[0].x, &enc.x), 0.0, "none() 张量应逐位一致");
        assert_eq!(plain[0].boxes, enc.boxes, "none() 框应逐位一致");
        assert_eq!(plain[0].kpts, enc.kpts, "none() 关键点应逐位一致");
        assert_eq!(plain[0].labels, enc.labels);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// 关键点翻转：画布 x 镜像 + COCO 17 索引交换（v 随三元组换位）+ 像素镜像。
    /// 32×32 图（letterbox 无 pad）下画布镜像 = `32 − x`，可手算。
    #[test]
    fn encode_keypoint_flip_mirrors_and_swaps_coco17() {
        use av_tasks::augment::COCO17_FLIP_SWAP;
        let dir = std::env::temp_dir().join(format!("av-ds-augkpf-{}", std::process::id()));
        let (img_dir, lbl_dir) = (dir.join("images/train"), dir.join("labels/train"));
        std::fs::create_dir_all(&img_dir).unwrap();
        std::fs::create_dir_all(&lbl_dir).unwrap();
        // 左半红右半蓝的 32×32 图：翻转后左半应为蓝
        let mut img = image::RgbImage::new(32, 32);
        for y in 0..32 {
            for x in 0..32 {
                img.put_pixel(x, y, if x < 16 { image::Rgb([255, 0, 0]) } else { image::Rgb([0, 0, 255]) });
            }
        }
        img.save(img_dir.join("a.png")).unwrap();
        // K=17 一实例：kpt i 归一化 x=(i+1)/19, y=0.5，v 交替 0/2
        let mut line = String::from("0 0.5 0.5 0.5 0.5");
        for i in 0..17 {
            let v = if i % 3 == 0 { 0.0 } else { 2.0 };
            line.push_str(&format!(" {} 0.5 {v}", (i as f32 + 1.0) / 19.0));
        }
        line.push('\n');
        std::fs::write(lbl_dir.join("a.txt"), line).unwrap();

        let plain = load_cocopose_dir(&dir, "train", 32, Device::Cpu, false).unwrap();
        let raw = load_cocopose_dir_raw(&dir, "train").unwrap();
        let plan = AugmentPlan {
            flip: true,
            ..AugmentPlan::none()
        };
        let enc = encode_keypoint_sample(&raw[0], 32, Device::Cpu, &plan, false).unwrap();
        let (p, e) = (&plain[0], &enc);

        // 像素：翻转后画布 (0,0) 是右半内容 → 蓝
        let px = |t: &Tensor, c: usize, y: usize, x: usize| {
            t.double_value(&[c as i64, y as i64, x as i64])
        };
        assert!(px(&e.x, 0, 0, 0) < 1e-6, "翻转后左上应为蓝的 R=0");
        assert!((px(&e.x, 2, 0, 0) - 1.0).abs() < 1e-6, "翻转后左上应为蓝的 B=1");
        assert!((px(&e.x, 0, 0, 31) - 1.0).abs() < 1e-6, "翻转后右上应为红");

        // 框中心镜像：cx' = 32 − cx（w/h/pad 无 pad 不变）
        assert!((e.boxes[0][0] - (32.0 - p.boxes[0][0])).abs() < 1e-4);
        assert!((e.boxes[0][2] - p.boxes[0][2]).abs() < 1e-4);

        // 关键点：new[i] = mirror(old[SWAP[i]])；y 与 v 随点换位
        for i in 0..17 {
            let src = &p.kpts[0][COCO17_FLIP_SWAP[i]];
            assert!(
                (e.kpts[0][i][0] - (32.0 - src[0])).abs() < 1e-4,
                "kpt{i}: {} != 32−{}",
                e.kpts[0][i][0],
                src[0]
            );
            assert!((e.kpts[0][i][1] - src[1]).abs() < 1e-4, "kpt{i} y 不应变");
            assert_eq!(e.kpts[0][i][2], src[2], "kpt{i} v 应随点换位");
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// 检测翻转 / 缩放的坐标手算：64×32 图，标注 cxcywh (0.75,0.25,0.25,0.25)。
    #[test]
    fn encode_detect_flip_and_scale_boxes_hand_computed() {
        let dir = std::env::temp_dir().join(format!("av-ds-augdet-{}", std::process::id()));
        let (img_dir, lbl_dir) = (dir.join("images/train"), dir.join("labels/train"));
        std::fs::create_dir_all(&img_dir).unwrap();
        std::fs::create_dir_all(&lbl_dir).unwrap();
        image::RgbImage::from_pixel(64, 32, image::Rgb([255, 0, 0]))
            .save(img_dir.join("a.png"))
            .unwrap();
        std::fs::write(lbl_dir.join("a.txt"), "3 0.75 0.25 0.25 0.25\n").unwrap();
        let raw = load_yolo_dir_raw(&dir, "train").unwrap();
        assert_eq!(raw[0].boxes, vec![[40.0, 4.0, 56.0, 12.0]], "原图像素 xyxy");

        // none：与 plain 加载器逐位一致
        let plain = load_yolo_dir_with_mode(&dir, "train", 32, Device::Cpu, ResizeMode::Letterbox, false)
            .unwrap();
        let enc = encode_detect_sample(
            &raw[0], 32, Device::Cpu, ResizeMode::Letterbox, &AugmentPlan::none(), false,
        )
        .unwrap();
        assert_eq!(tensor_max_diff(&plain[0].x, &enc.x), 0.0);
        assert_eq!(plain[0].boxes, enc.boxes);

        // flip：原图 x 镜像 (64−x)：(40,4,56,12) → (8,4,24,12)
        // → 画布（scale 0.5、pad_top 8）：(4,10)-(12,14)（= 未翻转画布框 (20,10)-(28,14) 的画布镜像）
        let fl = encode_detect_sample(
            &raw[0],
            32,
            Device::Cpu,
            ResizeMode::Letterbox,
            &AugmentPlan { flip: true, ..AugmentPlan::none() },
            false,
        )
        .unwrap();
        let b = fl.boxes[0];
        assert!((b[0] - 4.0).abs() < 1e-4 && (b[1] - 10.0).abs() < 1e-4, "b={b:?}");
        assert!((b[2] - 12.0).abs() < 1e-4 && (b[3] - 14.0).abs() < 1e-4, "b={b:?}");

        // scale 0.5：增强后图 32×16，letterbox scale=1、pad_top=8
        // → 框 (40,4,56,12)×0.5=(20,2,28,6) → 画布 (20,10)-(28,14)
        let sc = encode_detect_sample(
            &raw[0],
            32,
            Device::Cpu,
            ResizeMode::Letterbox,
            &AugmentPlan { scale: 0.5, ..AugmentPlan::none() },
            false,
        )
        .unwrap();
        let b = sc.boxes[0];
        assert!((b[0] - 20.0).abs() < 1e-4 && (b[1] - 10.0).abs() < 1e-4, "b={b:?}");
        assert!((b[2] - 28.0).abs() < 1e-4 && (b[3] - 14.0).abs() < 1e-4, "b={b:?}");
        assert_eq!(sc.x.size(), vec![3, 32, 32], "输出画布尺寸不变");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// 分割：raw + encode(none) 与 plain 加载器一致；翻转后掩码逐像素镜像
    /// （连续坐标镜像 x→W−x 把像素中心 (x+0.5) 映到 (W−1−x)+0.5，栅格化精确互镜）。
    #[test]
    fn encode_seg_none_matches_and_flip_mirrors_mask() {
        let dir = std::env::temp_dir().join(format!("av-ds-augseg-{}", std::process::id()));
        let (img_dir, lbl_dir) = (dir.join("images/train"), dir.join("labels/train"));
        std::fs::create_dir_all(&img_dir).unwrap();
        std::fs::create_dir_all(&lbl_dir).unwrap();
        image::RgbImage::from_pixel(32, 32, image::Rgb([255, 0, 0]))
            .save(img_dir.join("a.png"))
            .unwrap();
        std::fs::write(
            lbl_dir.join("a.txt"),
            "7 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5\n",
        )
        .unwrap();

        let plain = load_cocoseg_dir(&dir, "train", 32, Device::Cpu, false).unwrap();
        let raw = load_cocoseg_dir_raw(&dir, "train").unwrap();
        let none = encode_seg_sample(&raw[0], 32, Device::Cpu, &AugmentPlan::none(), false).unwrap();
        assert_eq!(plain[0].labels, none.labels);
        assert_eq!(plain[0].masks, none.masks, "none() 掩码应逐位一致");
        assert_eq!(tensor_max_diff(&plain[0].x, &none.x), 0.0);
        let (mw, mh) = (8usize, 8usize);
        assert_eq!(none.masks[0].iter().filter(|&&v| v == 1).count(), 9);

        let fl = encode_seg_sample(
            &raw[0],
            32,
            Device::Cpu,
            &AugmentPlan { flip: true, ..AugmentPlan::none() },
            false,
        )
        .unwrap();
        assert_eq!(fl.labels, plain[0].labels, "翻转不应丢实例");
        for y in 0..mh {
            for x in 0..mw {
                assert_eq!(
                    fl.masks[0][y * mw + x],
                    none.masks[0][y * mw + (mw - 1 - x)],
                    "翻转掩码应逐像素镜像 ({x},{y})"
                );
            }
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// OBB：翻转后 cxcywhθ 由镜像角点重推（水平镜像 → θ 取反），none 与 plain 一致。
    #[test]
    fn encode_obb_none_matches_and_flip_negates_theta() {
        use av_core::conventions::AngleDomain;
        let dir = std::env::temp_dir().join(format!("av-ds-augobb-{}", std::process::id()));
        let (img_dir, lbl_dir) = (dir.join("images/train"), dir.join("labels/train"));
        std::fs::create_dir_all(&img_dir).unwrap();
        std::fs::create_dir_all(&lbl_dir).unwrap();
        // 64×64 图；30° 斜框：归一化角点（cx=0.5, cy=0.5, w=0.5, h=0.25, θ=30°）
        let (th, cw, ch) = (30f32.to_radians(), 0.25f32, 0.125f32);
        let (c, s) = (th.cos(), th.sin());
        let corners: Vec<[f32; 2]> = [
            (-cw, -ch),
            (cw, -ch),
            (cw, ch),
            (-cw, ch),
        ]
        .iter()
        .map(|&(dx, dy)| [0.5 + dx * c - dy * s, 0.5 + dx * s + dy * c])
        .collect();
        let line = format!(
            "5 {}\n",
            corners
                .iter()
                .map(|p| format!("{:.6} {:.6}", p[0], p[1]))
                .collect::<Vec<_>>()
                .join(" ")
        );
        image::RgbImage::from_pixel(64, 64, image::Rgb([255, 0, 0]))
            .save(img_dir.join("a.png"))
            .unwrap();
        std::fs::write(lbl_dir.join("a.txt"), line).unwrap();

        let plain = load_dota_dir(&dir, "train", 32, Device::Cpu, false).unwrap();
        let raw = load_dota_dir_raw(&dir, "train").unwrap();
        let none = encode_obb_sample(&raw[0], 32, Device::Cpu, &AugmentPlan::none(), false).unwrap();
        assert_eq!(plain[0].boxes.len(), 1);
        assert_eq!(none.labels, plain[0].labels);
        for (a, b) in plain[0].boxes.iter().zip(&none.boxes) {
            for (va, vb) in a.iter().zip(b) {
                assert!((va - vb).abs() < 1e-4, "none() 应与 plain 一致: {va} vs {vb}");
            }
        }
        assert_eq!(tensor_max_diff(&plain[0].x, &none.x), 0.0);

        // 翻转：中心 x 镜像（画布 32），θ 取反（le90 域归一化后仍是取反关系）
        let fl = encode_obb_sample(
            &raw[0],
            32,
            Device::Cpu,
            &AugmentPlan { flip: true, ..AugmentPlan::none() },
            false,
        )
        .unwrap();
        let (p, e) = (&plain[0].boxes[0], &fl.boxes[0]);
        assert!((e[0] - (32.0 - p[0])).abs() < 1e-4, "cx 镜像: {} vs 32−{}", e[0], p[0]);
        assert!((e[1] - p[1]).abs() < 1e-4);
        assert!((e[2] - p[2]).abs() < 1e-4 && (e[3] - p[3]).abs() < 1e-4, "wh 不变");
        let (pn, en) = (AngleDomain::Le90.normalize(p[4]), AngleDomain::Le90.normalize(e[4]));
        assert!(
            (en + pn).abs() < 1e-3 || ((en - pn).abs() < 1e-3 && (p[2] - p[3]).abs() < 1e-3),
            "镜像应 θ→−θ（le90 归一化后）：{pn} vs {en}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    // -----------------------------------------------------------------
    // 组合增强 raw 封装（数据增强官二波）：mosaic4_raw / mixup_raw
    // -----------------------------------------------------------------

    /// 2×2 纯色 raw 检测样本。
    fn solid_raw(w: u32, h: u32, rgb: [u8; 3], boxes: Vec<[f32; 4]>, labels: Vec<u32>) -> RawDetectSample {
        RawDetectSample {
            w,
            h,
            rgb: vec![rgb; (w * h) as usize].into_iter().flatten().collect(),
            boxes,
            labels,
        }
    }

    /// mosaic4_raw：画布尺寸 / 象限像素 / 框换算手算对照；同一样本重复填充合法。
    #[test]
    fn mosaic4_raw_composes_quadrants_and_boxes() {
        let red = solid_raw(2, 2, [255, 0, 0], vec![[0.0, 0.0, 2.0, 2.0]], vec![7]);
        let green = solid_raw(2, 2, [0, 255, 0], vec![[0.0, 0.0, 1.0, 1.0]], vec![8]);
        let blue = solid_raw(2, 2, [0, 0, 255], vec![[1.0, 1.0, 2.0, 2.0]], vec![9]);
        let white = solid_raw(2, 2, [255, 255, 255], vec![[0.0, 1.0, 1.0, 2.0]], vec![10]);

        let m = mosaic4_raw([&red, &green, &blue, &white]).unwrap();
        // 画布 = 2×锚点尺寸 = 4×4
        assert_eq!((m.w, m.h), (4, 4));
        assert_eq!(m.rgb.len(), 4 * 4 * 3);
        let pixel = |x: usize, y: usize| &m.rgb[(y * 4 + x) * 3..(y * 4 + x) * 3 + 3];
        assert_eq!(pixel(0, 0), &[255, 0, 0], "左上=锚点红");
        assert_eq!(pixel(3, 0), &[0, 255, 0]);
        assert_eq!(pixel(0, 3), &[0, 0, 255]);
        assert_eq!(pixel(3, 3), &[255, 255, 255]);
        // 框逐一换算（画布像素域）+ 类别随框
        assert_eq!(
            m.boxes,
            vec![
                [0.0, 0.0, 2.0, 2.0],
                [2.0, 0.0, 3.0, 1.0],
                [1.0, 2.0 + 1.0, 2.0, 2.0 + 2.0],
                [2.0, 2.0 + 1.0, 2.0 + 1.0, 2.0 + 2.0],
            ]
        );
        assert_eq!(m.labels, vec![7, 8, 9, 10]);

        // 四图不足 → 重复采样填充（同一引用传 4 次）：满象限框 [0,0,2,2] 在每个
        // 象限各自落位（TL/TR/BL/BR），像素全同
        let rep = mosaic4_raw([&red, &red, &red, &red]).unwrap();
        assert_eq!(
            rep.boxes,
            vec![
                [0.0, 0.0, 2.0, 2.0],
                [2.0, 0.0, 4.0, 2.0],
                [0.0, 2.0, 2.0, 4.0],
                [2.0, 2.0, 4.0, 4.0],
            ]
        );
        assert_eq!(rep.labels, vec![7; 4]);
        assert!(rep.rgb.chunks(3).all(|px| px == [255, 0, 0]));
    }

    /// mosaic4_raw 缩放来源：4×4 源进 2×2 象限，框随像素同比例（0.5）换算。
    #[test]
    fn mosaic4_raw_scales_boxes_with_pixels() {
        let big = solid_raw(4, 4, [1, 2, 3], vec![[2.0, 2.0, 4.0, 4.0]], vec![3]);
        let anchor = solid_raw(2, 2, [9, 9, 9], vec![], vec![]);
        let m = mosaic4_raw([&anchor, &big, &big, &big]).unwrap();
        assert_eq!((m.w, m.h), (4, 4));
        // TR 象限：[2,2,4,4] × 0.5 + (2,0) = [3,1,4,2]；BL：+ (0,2) → [1,3,2,4]；BR：+ (2,2) → [3,3,4,4]
        assert_eq!(
            m.boxes,
            vec![[3.0, 1.0, 4.0, 2.0], [1.0, 3.0, 2.0, 4.0], [3.0, 3.0, 4.0, 4.0]]
        );
        assert_eq!(m.labels, vec![3, 3, 3]);
    }

    /// mixup_raw：像素 λ 加权手算 + 框/类别并集 + 异尺寸 partner 自动 stretch。
    #[test]
    fn mixup_raw_blends_and_merges_labels() {
        // a：2×1 双像素 [100,0,255 | 0,255,0]；b：2×1 [200,100,0 | 100,0,200]
        let mut a = solid_raw(2, 1, [0, 0, 0], vec![[0.0, 0.0, 1.0, 1.0]], vec![1]);
        a.rgb = vec![100, 0, 255, 0, 255, 0];
        let mut b = solid_raw(2, 1, [0, 0, 0], vec![[0.0, 0.0, 2.0, 1.0]], vec![2]);
        b.rgb = vec![200, 100, 0, 100, 0, 200];

        // λ=0.25：px1 = [100·0.25+200·0.75, 75, 63.75→64] = [175, 75, 64]；
        // px2 = [75, 63.75→64, 150]
        let m = mixup_raw(&a, &b, 0.25).unwrap();
        assert_eq!((m.w, m.h), (2, 1));
        assert_eq!(&m.rgb[..3], &[175, 75, 64]);
        assert_eq!(&m.rgb[3..], &[75, 64, 150]);
        // 框取并集（双份 gt），顺序 = a 后 b
        assert_eq!(m.boxes, vec![[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 2.0, 1.0]]);
        assert_eq!(m.labels, vec![1, 2]);

        // λ=1 → 逐位 a
        let m1 = mixup_raw(&a, &b, 1.0).unwrap();
        assert_eq!(m1.rgb, a.rgb);

        // 异尺寸 partner：b 4×2 stretch 到 a 的 2×1 后融合（尺寸 = a）
        let big = solid_raw(4, 2, [255, 255, 255], vec![[0.0, 0.0, 4.0, 2.0]], vec![5]);
        let m2 = mixup_raw(&a, &big, 0.5).unwrap();
        assert_eq!((m2.w, m2.h), (2, 1), "输出尺寸随 a");
        assert_eq!(m2.labels, vec![1, 5], "并集含 partner 标签");
        assert_eq!(m2.boxes.len(), 2);
    }
}

// ---------------------------------------------------------------------------
// 数据管线 v2 语义对齐单测：缓存编码 vs 全分辨率编码
// ---------------------------------------------------------------------------

/// 合成 raw 分割样本：低频平滑渐变（贴近真实影像的频谱；高频棋盘纹会让
/// 重采样相位差被病态放大，测不出语义差异）+ 矩形/三角多边形。
#[cfg(all(test, feature = "torch"))]
fn synthetic_raw_for_test(w: u32, h: u32) -> RawSegSample {
    let mut rgb = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let fx = x as f32 / w as f32;
            let fy = y as f32 / h as f32;
            rgb.push((40.0 + 190.0 * fx) as u8);
            rgb.push((60.0 + 160.0 * fy) as u8);
            rgb.push((70.0 + 150.0 * (fx * 0.5 + fy * 0.5)) as u8);
        }
    }
    let poly_rect = vec![
        [0.1 * w as f32, 0.1 * h as f32],
        [0.6 * w as f32, 0.12 * h as f32],
        [0.62 * w as f32, 0.55 * h as f32],
        [0.12 * w as f32, 0.5 * h as f32],
    ];
    let poly_tri = vec![
        [0.7 * w as f32, 0.6 * h as f32],
        [0.95 * w as f32, 0.65 * h as f32],
        [0.8 * w as f32, 0.92 * h as f32],
    ];
    RawSegSample {
        w,
        h,
        rgb,
        polys: vec![poly_rect, poly_tri],
        labels: vec![0, 1],
    }
}

#[cfg(all(test, feature = "torch"))]
mod seg_cache_tests {
    use super::*;
    use av_tasks::augment::AugmentPlan;

    /// 合成 raw 分割样本（低频平滑渐变 + 多边形），测试与基准共用。
    fn synthetic_raw(w: u32, h: u32) -> RawSegSample {
        super::synthetic_raw_for_test(w, h)
    }

    /// (最大绝对差, 平均绝对差)，张量逐元素。
    fn tensor_diff(a: &Tensor, b: &Tensor) -> (f32, f32) {
        let d = (a - b).abs();
        (
            d.max().double_value(&[]) as f32,
            d.mean(Kind::Float).double_value(&[]) as f32,
        )
    }

    fn assert_masks_eq(a: &SegSample, b: &SegSample) {
        assert_eq!(a.masks.len(), b.masks.len(), "实例数应一致");
        for (ma, mb) in a.masks.iter().zip(&b.masks) {
            assert_eq!(ma, mb, "掩码必须逐位一致");
        }
        assert_eq!(a.labels, b.labels);
    }

    /// none plan（收尾关增强 / 验收集路径）：缓存编码与全分辨率编码**逐位一致**。
    #[test]
    fn cached_none_plan_is_bit_exact() {
        let raw = synthetic_raw(160, 128);
        let img_size = 128;
        let reference =
            encode_seg_sample(&raw, img_size, Device::Cpu, &AugmentPlan::none(), false).unwrap();
        let cached = build_seg_cache_sample(&raw, img_size).unwrap();
        let fast =
            encode_seg_sample_cached(&cached, img_size, Device::Cpu, &AugmentPlan::none(), false)
                .unwrap();
        let (dmax, _) = tensor_diff(&reference.x, &fast.x);
        assert_eq!(dmax, 0.0, "none plan 必须逐位一致，实际最大差 {dmax}");
        assert_masks_eq(&reference, &fast);
    }

    /// flip-only plan：翻转与 Triangle 缩放可交换 ⇒ 逐位一致。
    #[test]
    fn cached_flip_plan_is_bit_exact() {
        let raw = synthetic_raw(160, 128);
        let img_size = 128;
        let plan = AugmentPlan {
            flip: true,
            scale: 1.0,
            rgb_gains: [1.0; 3],
        };
        let reference = encode_seg_sample(&raw, img_size, Device::Cpu, &plan, false).unwrap();
        let cached = build_seg_cache_sample(&raw, img_size).unwrap();
        let fast =
            encode_seg_sample_cached(&cached, img_size, Device::Cpu, &plan, false).unwrap();
        let (dmax, _) = tensor_diff(&reference.x, &fast.x);
        assert_eq!(dmax, 0.0, "flip 路径必须逐位一致，实际最大差 {dmax}");
        assert_masks_eq(&reference, &fast);
    }

    /// 增益路径：u8 定点乘的先后次序差（clamp/混合），允许 ≤2/255 容差；
    /// 掩码仍逐位一致。
    #[test]
    fn cached_gains_within_tolerance() {
        let raw = synthetic_raw(160, 128);
        let img_size = 128;
        let plan = AugmentPlan {
            flip: true,
            scale: 1.0,
            rgb_gains: [1.08, 0.92, 1.0],
        };
        let reference = encode_seg_sample(&raw, img_size, Device::Cpu, &plan, false).unwrap();
        let cached = build_seg_cache_sample(&raw, img_size).unwrap();
        let fast =
            encode_seg_sample_cached(&cached, img_size, Device::Cpu, &plan, false).unwrap();
        let (dmax, dmean) = tensor_diff(&reference.x, &fast.x);
        assert!(dmax <= 2.0 / 255.0, "增益路径最大差 {dmax} 超容差 2/255");
        assert!(dmean < 0.2 / 255.0, "增益路径平均差 {dmean} 偏大");
        assert_masks_eq(&reference, &fast);
    }

    /// 缩放路径：单次重采样替代旧路径两级重采样，允许 ≤4/255 容差；
    /// 掩码坐标公式同源 ⇒ 逐位一致。
    #[test]
    fn cached_scale_within_tolerance() {
        let raw = synthetic_raw(160, 128);
        let img_size = 128;
        let plan = AugmentPlan {
            flip: false,
            scale: 1.15,
            rgb_gains: [1.0; 3],
        };
        let reference = encode_seg_sample(&raw, img_size, Device::Cpu, &plan, false).unwrap();
        let cached = build_seg_cache_sample(&raw, img_size).unwrap();
        let fast =
            encode_seg_sample_cached(&cached, img_size, Device::Cpu, &plan, false).unwrap();
        let (dmax, dmean) = tensor_diff(&reference.x, &fast.x);
        assert!(dmax <= 4.0 / 255.0, "缩放路径最大差 {dmax} 超容差 4/255");
        assert!(dmean < 1.0 / 255.0, "缩放路径平均差 {dmean} 偏大");
        assert_masks_eq(&reference, &fast);
    }

    /// 缓存构建的几何：贴片尺寸 = round(原始 × letterbox scale)。
    #[test]
    fn cache_geometry_matches_letterbox() {
        let raw = synthetic_raw(244, 204); // 非方形
        let img_size = 128;
        let c = build_seg_cache_sample(&raw, img_size).unwrap();
        let lb = letterbox(raw.w, raw.h, img_size, 1);
        let nw = ((raw.w as f32 * lb.scale).round() as u32).max(1);
        let nh = ((raw.h as f32 * lb.scale).round() as u32).max(1);
        assert_eq!((c.cw, c.ch), (nw, nh));
        assert_eq!(c.content.len(), (nw * nh * 3) as usize);
    }

    /// GPU 显存驻留路径与 CPU 缓存路径的数值一致性（flip/gain ≤2/255、
    /// scale 双线性 vs Triangle ≤6/255）；无 CUDA 时跳过。
    #[test]
    fn gpu_stack_matches_cpu_cached() {
        if !matches!(Device::cuda_if_available(), Device::Cuda(_)) {
            return; // 无 CUDA 环境跳过（CI/CPU 构建友好）
        }
        let device = Device::Cuda(0);
        let raw = synthetic_raw(160, 128);
        let img_size = 128;
        let cached = build_seg_cache_sample(&raw, img_size).unwrap();
        let stack = build_seg_canvas_stack(std::slice::from_ref(&cached), img_size, device)
            .unwrap();
        // 回归锁：内容贴片是非方形（cw≠ch），画布堆必须是 [N,3,S,S] 且
        // reshape/窄拷贝路径不能静默走样（曾因方形假设引发运行期 panic）
        assert_eq!(
            stack.size(),
            vec![1i64, 3, img_size as i64, img_size as i64],
            "画布堆形状错误"
        );
        for (name, plan, tol_max, tol_mean) in [
            ("flip", AugmentPlan { flip: true, scale: 1.0, rgb_gains: [1.0; 3] }, 2.0f32, 0.2f32),
            ("gain", AugmentPlan { flip: false, scale: 1.0, rgb_gains: [1.06, 0.95, 1.0] }, 2.0, 0.2),
            ("scale", AugmentPlan { flip: false, scale: 1.1, rgb_gains: [1.0; 3] }, 8.0, 2.0),
        ] {
            let cpu = encode_seg_sample_cached(
                &cached, img_size, Device::Cpu, &plan, false,
            )
            .unwrap()
            .x
            .to_device(device);
            let gpu = encode_seg_sample_gpu(&stack, 0, &cached, img_size, &plan, false)
                .unwrap()
                .x;
            let (dmax, dmean) = tensor_diff(&cpu, &gpu);
            assert!(
                dmax <= tol_max / 255.0 && dmean <= tol_mean / 255.0,
                "{name}: GPU/CPU 差 max={dmax} mean={dmean} 超容差 ({tol_max},{tol_mean})/255"
            );
        }
    }
}

/// 编码吞吐基准（`cargo test seg_cache_bench -- --ignored --nocapture` 显式跑）：
/// 旧全分辨率单线程路径 vs 新缓存并行路径的每样本耗时与加速比。
#[test]
#[ignore = "基准：需要显式运行（--ignored --nocapture）"]
fn seg_cache_bench() {
    use std::time::Instant;
    let (w, h) = (2448u32, 2048u32);
    let img_size = 640;
    let n = 8;
    let raws: Vec<RawSegSample> = (0..n)
        .map(|k| {
            let mut r = {
                let mut s = synthetic_raw_for_test(w, h);
                // 逐样本微移相位，避免完全相同缓存被过度共享
                for v in s.rgb.iter_mut().step_by(97) {
                    *v = v.wrapping_add(k as u8 * 7);
                }
                s
            };
            r.polys = vec![vec![[0.1 * w as f32, 0.1 * h as f32], [0.6 * w as f32, 0.6 * h as f32]]];
            r.labels = vec![0];
            r
        })
        .collect();
    let plans = vec![AugmentPlan { flip: true, scale: 1.05, rgb_gains: [1.02, 0.98, 1.0] }; n];

    let t0 = Instant::now();
    let _old: Vec<_> = raws
        .iter()
        .zip(&plans)
        .map(|(r, p)| encode_seg_sample(r, img_size, Device::Cpu, p, false).unwrap())
        .collect();
    let old_ms = t0.elapsed().as_millis() as f64 / n as f64;

    let t1 = Instant::now();
    let (cache, _bytes) = build_seg_cache(&raws, img_size).unwrap();
    let build_ms = t1.elapsed().as_millis() as f64 / n as f64;

    let t2 = Instant::now();
    let idx: Vec<usize> = (0..n).collect();
    let _new = encode_seg_batch_cached(&cache, &idx, &plans, img_size, Device::Cpu, false).unwrap();
    let new_ms = t2.elapsed().as_millis() as f64 / n as f64;

    println!("旧路径（全分辨率单线程）: {old_ms:.1} ms/样本");
    println!("新路径（缓存+rayon 并行）: {new_ms:.1} ms/样本（缓存构建一次性 {build_ms:.1} ms/样本）");
    println!("稳态加速比: {:.0}x", old_ms / new_ms);
    assert!(old_ms > new_ms * 4.0, "新路径应显著快于旧路径");
}
