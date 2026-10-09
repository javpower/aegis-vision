//! avb —— AegisVision burn 后端 CLI（纯 Rust，**无 libtorch 依赖**）。
//!
//! 与主 CLI `av`（aegisvision-runtime，tch/libtorch）体验对齐：
//!
//! - `avb train --data <目录|data.yaml>`：一条命令训练分割模型
//!   （数据格式 = 同款 YOLO-seg 目录：images/<split> + labels/<split>）；
//! - `avb predict --weights <目录|model.bp> --input <图>`：单图推理，
//!   超参从权重旁 config.snapshot.toml 自动重建，掩码映射回原图坐标；
//! - `--save-viz` 叠色可视化 / `--save-masks` 导出 PNG 掩码 / `--json` 结构化输出。
//!
//! 后端为编译期选择：默认 ndarray（CPU）；`--features wgpu` 构建 GPU 版
//! （`--device gpu`）。训练引擎 = AdamW + cosine 退火 + 逐参数范数裁剪
//! （与 av-burn 训练链路一致，详见 crate 文档）。

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use av_burn::checkpoint::{load_model, save_model, SegSnapshot};
use av_burn::data::load_cocoseg_dir;
use av_burn::infer::predict_image;
use av_burn::seg::SegNet;
use av_burn::train::{cosine_lr, make_optimizer, train_step, TrainCfg};
use burn_core::module::AutodiffModule;
use burn_core::tensor::{Tensor, TensorData};
use clap::{Parser, Subcommand};

/// 推理后端（编译期选择：ndarray 默认 / wgpu feature）。
#[cfg(not(feature = "wgpu"))]
mod backend {
    use anyhow::{bail, Result};
    use av_burn::NdArrayB;
    use burn_autodiff::Autodiff;
    use burn_ndarray::NdArrayDevice;

    /// 推理后端 = ndarray（CPU）。
    pub type InferB = NdArrayB;
    /// 训练后端 = ndarray + autodiff。
    pub type TrainB = Autodiff<InferB>;
    pub type Dev = NdArrayDevice;

    pub fn select_device(want_gpu: bool) -> Result<Dev> {
        if want_gpu {
            bail!(
                "本构建未启用 wgpu 后端：GPU 需以 `cargo install aegisvision-burn --features wgpu` \
                 重新安装（或源码构建时加 --features wgpu）；当前请用 --device cpu"
            );
        }
        Ok(Default::default())
    }
}

#[cfg(feature = "wgpu")]
mod backend {
    use anyhow::Result;
    use burn_autodiff::Autodiff;
    use burn_wgpu::{Wgpu, WgpuDevice};

    /// 推理后端 = wgpu（fusion + autotune）。
    pub type InferB = Wgpu;
    /// 训练后端 = wgpu + autodiff。
    pub type TrainB = Autodiff<InferB>;
    pub type Dev = WgpuDevice;

    pub fn select_device(want_gpu: bool) -> Result<Dev> {
        Ok(if want_gpu {
            WgpuDevice::default()
        } else {
            WgpuDevice::Cpu
        })
    }
}

use backend::{select_device, InferB, TrainB};

#[derive(Parser)]
#[command(
    name = "avb",
    version,
    about = "AegisVision burn 后端：纯 Rust 分割训练/推理（无 libtorch）"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// 训练：给 --data 即一条命令训练（数据目录或 Ultralytics data.yaml，
    /// 类数自动探测）；数据格式 = YOLO-seg 目录 images/<split> + labels/<split>
    Train {
        /// 数据集：目录（images/train 布局）或 data.yaml
        #[arg(long)]
        data: PathBuf,
        /// 输入画布边长
        #[arg(long, default_value_t = 640)]
        imgsz: u32,
        /// 训练轮数
        #[arg(long, default_value_t = 100)]
        epochs: u32,
        /// 批大小
        #[arg(long, default_value_t = 8)]
        batch: usize,
        /// 骨干宽度乘子（0.25=nano 档；CPU 冒烟建议 0.0625）
        #[arg(long, default_value_t = 0.25)]
        width: f32,
        /// 骨干深度乘子
        #[arg(long, default_value_t = 0.33)]
        depth: f32,
        /// 原型掩码数 K
        #[arg(long, default_value_t = 32)]
        protos: usize,
        /// 初始学习率
        #[arg(long, default_value_t = 1e-3)]
        lr: f64,
        /// cosine 终点学习率
        #[arg(long, default_value_t = 1e-5)]
        lr_min: f64,
        /// AdamW 解耦权重衰减
        #[arg(long, default_value_t = 5e-4)]
        weight_decay: f32,
        /// 梯度范数裁剪上限
        #[arg(long, default_value_t = 10.0)]
        clip: f32,
        /// 计算设备：cpu | gpu（gpu 需 wgpu feature 构建）
        #[arg(long, default_value = "cpu")]
        device: String,
        /// 输出目录（缺省 runs-avb/<数据集名>）
        #[arg(short, long)]
        out: Option<PathBuf>,
        /// 随机种子（洗牌；默认固定 42 保证可复现）
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    /// 推理：超参从权重旁 config.snapshot.toml 自动重建（train 产物即权重目录）
    Predict {
        /// 权重目录（或 model.bp 文件）
        #[arg(short, long)]
        weights: PathBuf,
        /// 输入图片
        #[arg(long)]
        input: PathBuf,
        #[arg(long, default_value_t = 0.25)]
        conf: f32,
        /// 掩码 NMS IoU 阈值
        #[arg(long, default_value_t = 0.7)]
        iou: f32,
        /// 计算设备：cpu | gpu（gpu 需 wgpu feature 构建）
        #[arg(long, default_value = "cpu")]
        device: String,
        /// 可视化输出目录：掩码叠色画到原图，存 <DIR>/<输入名>.jpg
        #[arg(long = "save-viz", value_name = "DIR")]
        save_viz: Option<PathBuf>,
        /// 掩码 PNG 导出目录：每实例一张 <DIR>/<输入名>_inst<k>.png
        #[arg(long = "save-masks", value_name = "DIR")]
        save_masks: Option<PathBuf>,
        /// 结构化结果 JSON 输出路径
        #[arg(long)]
        json: Option<PathBuf>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Train {
            data,
            imgsz,
            epochs,
            batch,
            width,
            depth,
            protos,
            lr,
            lr_min,
            weight_decay,
            clip,
            device,
            out,
            seed,
        } => train(
            &data,
            imgsz,
            epochs,
            batch,
            width,
            depth,
            protos,
            lr,
            lr_min,
            weight_decay,
            clip,
            &device,
            out,
            seed,
        ),
        Command::Predict {
            weights,
            input,
            conf,
            iou,
            device,
            save_viz,
            save_masks,
            json,
        } => predict(
            &weights, &input, conf, iou, &device, save_viz, save_masks, json,
        ),
    }
}

/// data.yaml 的 train/val 值 → split 名（剥 "images/" 前缀，load_cocoseg_dir
/// 自行拼接 images/<split> + labels/<split>）。
fn yaml_split(v: &str) -> String {
    let v = v.trim_start_matches("./");
    v.strip_prefix("images/").unwrap_or(v).to_string()
}

/// 目录模式类数探测：labels/<split>/*.txt 标注首列最大 id + 1（与 av 的
/// scan_max_class_id 同语义）。
fn scan_num_classes(root: &Path, split: &str) -> Result<usize> {
    let lbl = root.join("labels").join(split);
    let mut max_id: Option<u32> = None;
    for entry in
        std::fs::read_dir(&lbl).with_context(|| format!("读标注目录失败: {}", lbl.display()))?
    {
        let p = entry?.path();
        if p.extension().and_then(|e| e.to_str()) != Some("txt") {
            continue;
        }
        let text = std::fs::read_to_string(&p)?;
        for line in text.lines() {
            if let Some(first) = line.split_whitespace().next() {
                if let Ok(id) = first.parse::<u32>() {
                    max_id = Some(max_id.map_or(id, |m: u32| m.max(id)));
                }
            }
        }
    }
    max_id
        .map(|m| m as usize + 1)
        .with_context(|| format!("未在 {} 下找到标注（labels/{split}）", root.display()))
}

/// 极简 xorshift 洗牌（与 bench 示例同款，避免引入 rand）。
fn shuffle(idx: &mut [usize], seed: u64) {
    let mut s = seed | 1;
    for i in (1..idx.len()).rev() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        idx.swap(i, (s as usize) % (i + 1));
    }
}

#[allow(clippy::too_many_arguments)]
fn train(
    data: &Path,
    imgsz: u32,
    epochs: u32,
    batch: usize,
    width: f32,
    depth: f32,
    protos: usize,
    lr: f64,
    lr_min: f64,
    weight_decay: f32,
    clip: f32,
    device: &str,
    out: Option<PathBuf>,
    seed: u64,
) -> Result<()> {
    let want_gpu = match device.to_lowercase().as_str() {
        "cpu" => false,
        "gpu" | "cuda" => true,
        other => bail!("未知设备 '{other}'（可选 cpu | gpu）"),
    };
    let dev = select_device(want_gpu)?;

    // 数据：yaml → (root, split, names)；目录 → root + "train" + 类数扫描
    let ext = data
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    let (root, split, names) = if matches!(ext.as_str(), "yaml" | "yml") {
        let (root, train, _val, names) = av_core::config::parse_data_yaml(data)?;
        (root, yaml_split(&train), Some(names))
    } else {
        (data.to_path_buf(), "train".to_string(), None)
    };
    let t0 = Instant::now();
    let samples = load_cocoseg_dir(&root, &split, imgsz)
        .with_context(|| format!("加载数据集 {}（split {split}）失败", root.display()))?;
    let num_classes = match &names {
        Some(n) => n.len(),
        None => scan_num_classes(&root, &split)?,
    };
    println!(
        "[avb] 数据 {} 张（{} 类，img={imgsz}，预解码 {:.1}s，设备 {}）",
        samples.len(),
        num_classes,
        t0.elapsed().as_secs_f32(),
        if want_gpu { "gpu" } else { "cpu" }
    );
    if !want_gpu && imgsz >= 640 && width >= 0.25 {
        println!(
            "[avb] 提示：ndarray CPU 后端无 BLAS，nano@640 训练会非常慢；\
             冒烟建议 --imgsz 320 --width 0.0625"
        );
    }

    let run_id = root
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("dataset")
        .to_string();
    let out_dir = out.unwrap_or_else(|| PathBuf::from("runs-avb").join(&run_id));

    let mut model = SegNet::<TrainB>::new(
        &av_burn::seg::SegNetCfg {
            width,
            depth,
            num_classes,
            num_protos: protos,
            loss_w_bce: 1.0,
            loss_w_dice: 1.0,
        },
        &dev,
    )?;
    let steps_per_epoch = (samples.len() / batch.max(1)).max(1);
    let tcfg = TrainCfg {
        lr,
        lr_min,
        weight_decay,
        max_grad_norm: clip,
        total_steps: steps_per_epoch * epochs as usize,
    };
    let mut optim = make_optimizer(&tcfg);

    let s = imgsz as usize;
    let mut order: Vec<usize> = (0..samples.len()).collect();
    for epoch in 1..=epochs {
        shuffle(&mut order, seed ^ (epoch as u64));
        let t = Instant::now();
        let mut loss_sum = 0f32;
        let mut steps = 0usize;
        for chunk in order.chunks(batch) {
            let n = chunk.len();
            let mut buf = Vec::with_capacity(n * 3 * s * s);
            for &i in chunk {
                buf.extend_from_slice(&samples[i].pixels);
            }
            let x = Tensor::<TrainB, 4>::from_data(TensorData::new(buf, [n, 3, s, s]), &dev);
            let masks: Vec<Vec<Vec<u8>>> =
                chunk.iter().map(|&i| samples[i].masks.clone()).collect();
            let labels: Vec<Vec<u32>> = chunk.iter().map(|&i| samples[i].labels.clone()).collect();
            let step_lr = cosine_lr(&tcfg, (epoch as usize - 1) * steps_per_epoch + steps);
            let (m, v) = train_step(model, &mut optim, step_lr, |m| {
                m.loss(x.clone(), &masks, &labels)
            });
            model = m;
            loss_sum += v;
            steps += 1;
        }
        println!(
            "[avb] epoch {epoch:>4}/{epochs}  loss={:.4}  lr={:.2e}  {:.1}s",
            loss_sum / steps.max(1) as f32,
            cosine_lr(&tcfg, epoch as usize * steps_per_epoch),
            t.elapsed().as_secs_f32()
        );
    }

    // 保存：推理模型（剥 autodiff）+ 快照（train → predict 零参数闭环）
    let snapshot = SegSnapshot {
        format: 1,
        imgsz,
        width,
        depth,
        num_classes,
        num_protos: protos,
        loss_w_bce: 1.0,
        loss_w_dice: 1.0,
        classes: names.unwrap_or_else(|| (0..num_classes).map(|i| format!("class_{i}")).collect()),
    };
    let bp = save_model(&model.valid(), &snapshot, &out_dir)?;
    println!(
        "[avb] 完成：权重 {}（快照 {}）\n[avb] 预测：avb predict --weights {} --input <图>",
        bp.display(),
        out_dir.join("config.snapshot.toml").display(),
        out_dir.display()
    );
    Ok(())
}

/// 类别调色板：golden-ratio 色相环（同标签恒定色）。
fn label_color(label: u32) -> [u8; 3] {
    let hue = (fmod(label as f32 * 47.0, 360.0)) / 360.0;
    let (sat, val) = (0.75f32, 0.95f32);
    let h6 = hue * 6.0;
    let i = h6.floor() as i32 % 6;
    let f = h6 - h6.floor();
    let (r, g, b) = match i.rem_euclid(6) {
        0 => (val, val * (1.0 - sat * (1.0 - f)), val * (1.0 - sat)),
        1 => (val * (1.0 - sat * f), val, val * (1.0 - sat)),
        2 => (val * (1.0 - sat), val, val * (1.0 - sat * (1.0 - f))),
        3 => (val * (1.0 - sat), val * (1.0 - sat * f), val),
        4 => (val * (1.0 - sat * (1.0 - f)), val * (1.0 - sat), val),
        _ => (val, val * (1.0 - sat), val * (1.0 - sat * f)),
    };
    [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]
}

fn fmod(a: f32, b: f32) -> f32 {
    a - b * (a / b).floor()
}

/// 掩码实例结构化摘要（打印 + JSON 共用）。
struct InstSummary {
    label: u32,
    name: String,
    score: f32,
    area: usize,
    bbox: [usize; 4], // x0,y0,x1,y1（原图坐标）
}

fn summarize(
    instances: &[av_burn::infer::SegInstance],
    snapshot: &SegSnapshot,
) -> Vec<InstSummary> {
    instances
        .iter()
        .map(|inst| {
            let w = (inst.mask.len() as f64).sqrt().round() as usize;
            let (mut x0, mut y0, mut x1, mut y1) = (w, w, 0, 0);
            for (i, &v) in inst.mask.iter().enumerate() {
                if v != 0 {
                    let (x, y) = (i % w, i / w);
                    x0 = x0.min(x);
                    y0 = y0.min(y);
                    x1 = x1.max(x);
                    y1 = y1.max(y);
                }
            }
            InstSummary {
                label: inst.label,
                name: snapshot.class_name(inst.label as usize),
                score: inst.score,
                area: inst.mask.iter().filter(|&&v| v != 0).count(),
                bbox: [x0, y0, x1, y1],
            }
        })
        .collect()
}

fn mask_to_rgb_image(
    base: &image::RgbImage,
    instances: &[av_burn::infer::SegInstance],
) -> image::RgbImage {
    let mut out = base.clone();
    for inst in instances {
        let [r, g, b] = label_color(inst.label);
        let a = 0.45f32;
        for (i, px) in out.pixels_mut().enumerate() {
            if inst.mask.get(i).copied().unwrap_or(0) != 0 {
                let blend = |c: u8, t: u8| (c as f32 * (1.0 - a) + t as f32 * a).round() as u8;
                *px = image::Rgb([blend(px[0], r), blend(px[1], g), blend(px[2], b)]);
            }
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn predict(
    weights: &Path,
    input: &Path,
    conf: f32,
    iou: f32,
    device: &str,
    save_viz: Option<PathBuf>,
    save_masks: Option<PathBuf>,
    json: Option<PathBuf>,
) -> Result<()> {
    let want_gpu = match device.to_lowercase().as_str() {
        "cpu" => false,
        "gpu" | "cuda" => true,
        other => bail!("未知设备 '{other}'（可选 cpu | gpu）"),
    };
    let dev = select_device(want_gpu)?;
    let (model, snapshot) = load_model::<InferB>(weights, &dev)
        .with_context(|| format!("加载权重失败: {}", weights.display()))?;
    let img = image::open(input)
        .with_context(|| format!("读图失败: {}", input.display()))?
        .to_rgb8();
    let (instances, _lb) = predict_image(&model, &dev, &img, snapshot.imgsz, conf, iou)?;

    let summaries = summarize(&instances, &snapshot);
    println!(
        "[avb] {}：{} 个实例（conf={conf} iou={iou}，掩码已映射原图 {}×{}）",
        input.display(),
        summaries.len(),
        img.width(),
        img.height()
    );
    for (k, s) in summaries.iter().enumerate() {
        println!(
            "  #{k:<3} {}（label {}）  score={:.3}  area={}px  bbox={:?}",
            s.name, s.label, s.score, s.area, s.bbox
        );
    }

    if let Some(dir) = save_viz {
        std::fs::create_dir_all(&dir)?;
        let viz = mask_to_rgb_image(&img, &instances);
        let out_p = dir.join(
            input
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("out")
                .to_string()
                + ".jpg",
        );
        image::DynamicImage::ImageRgb8(viz)
            .save(&out_p)
            .with_context(|| format!("写可视化失败: {}", out_p.display()))?;
        println!("[avb] 可视化 → {}", out_p.display());
    }
    if let Some(dir) = save_masks {
        std::fs::create_dir_all(&dir)?;
        let stem = input.file_stem().and_then(|s| s.to_str()).unwrap_or("out");
        for (k, inst) in instances.iter().enumerate() {
            let (w, h) = (img.width(), img.height());
            let mut g = image::GrayImage::new(w, h);
            for (i, px) in g.pixels_mut().enumerate() {
                *px = image::Luma([if inst.mask.get(i).copied().unwrap_or(0) != 0 {
                    255
                } else {
                    0
                }]);
            }
            let p = dir.join(format!("{stem}_inst{k}.png"));
            image::DynamicImage::ImageLuma8(g).save(&p)?;
            println!("[avb] 掩码 → {}", p.display());
        }
    }
    if let Some(p) = json {
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let val = serde_json::json!({
            "input": input.display().to_string(),
            "imgsz": snapshot.imgsz,
            "conf": conf,
            "iou": iou,
            "classes": snapshot.classes,
            "instances": summaries.iter().map(|s| serde_json::json!({
                "label": s.label,
                "name": s.name,
                "score": s.score,
                "area_px": s.area,
                "bbox": s.bbox,
            })).collect::<Vec<_>>(),
        });
        std::fs::write(&p, serde_json::to_string_pretty(&val)?)?;
        println!("[avb] JSON → {}", p.display());
    }
    Ok(())
}
