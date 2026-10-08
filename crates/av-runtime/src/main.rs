//! av-runtime：AegisVision 训练/推理入口与 CLI（PLAN §2.1 / 附录 C）。
//!
//! M0 阶段：`av train --dry-run` 打通「配置加载 → 校验 → 运行计划」；
//! 其余子命令按 PLAN §8 里程碑逐步启用。

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use av_core::config::{DataPipeline, RunConfig, TaskCfg};
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "av", version, about = "AegisVision：Rust 原生多任务检测框架")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// 生成最小可用训练配置（脚手架：探测数据格式与类别数，超参全走默认）
    Init {
        /// 任务类型：detect | classify | obb
        #[arg(long)]
        task: InitTask,
        /// 数据集根目录（缺省生成 synthetic 合成数据配置，开箱即训）
        #[arg(long)]
        data: Option<PathBuf>,
        /// 输出配置路径（缺省 configs/av_<task>.toml；无 configs/ 目录时写当前目录）
        #[arg(long)]
        out: Option<PathBuf>,
        /// 类别数（缺省自动统计：YOLO/DOTA 标签取最大类 id+1；ImageFolder 数 wnid 子目录）
        #[arg(long)]
        classes: Option<usize>,
    },
    /// 数据集打包为 .avpack 容器（附录 B；M2 实现）
    Pack {
        /// 源数据集目录
        #[arg(long)]
        src: PathBuf,
        /// 输出 .avpack 路径
        #[arg(long)]
        out: PathBuf,
    },
    /// 训练：给 --data 即一条命令训练（数据目录或 Ultralytics data.yaml，
    /// 类数自动探测，其余超参走内置默认）；给 -c 则按配置文件训练
    Train {
        /// TOML 配置文件（与 --data 二选一）
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// 数据集：目录（images/labels 布局）或 data.yaml
        #[arg(long)]
        data: Option<PathBuf>,
        /// 任务类型（--data 模式；当前 detect）
        #[arg(long, default_value = "detect")]
        task: String,
        /// 输入边长（--data 模式）
        #[arg(long, default_value_t = 640)]
        imgsz: u32,
        /// 训练轮数（--data 模式）
        #[arg(long, default_value_t = 100)]
        epochs: u32,
        /// 批大小（--data 模式；0 = 按设备自动：cuda 8 / cpu 4）
        #[arg(long, default_value_t = 0)]
        batch: u32,
        /// 从 runs/<run_id>/last.pt 续训（M7 生效）
        #[arg(long)]
        resume: bool,
        /// 覆盖配置项 key=value（当前支持 seed / device / output_dir）
        #[arg(long = "override", value_name = "KEY=VALUE")]
        overrides: Vec<String>,
        /// 只校验配置并打印运行计划
        #[arg(long)]
        dry_run: bool,
    },
    /// 推理（v0.1 已接通：分类/检测单图推理，模型配置取权重旁的 config.snapshot.toml）
    Infer {
        /// 权重文件
        #[arg(short, long)]
        weights: PathBuf,
        /// 模型配置（缺省取权重同目录 config.snapshot.toml）
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// 输入图片
        #[arg(long)]
        input: PathBuf,
        #[arg(long, default_value_t = 0.25)]
        conf: f32,
        #[arg(long, default_value_t = 0.7)]
        iou: f32,
        /// 超大图切片推理（SAHI 式，已实现：高分辨率图保持原分辨率滑窗检测）
        #[arg(long)]
        slice: bool,
        /// 切片窗口边长（原图像素，--slice 时生效）
        #[arg(long, default_value_t = 256)]
        slice_window: u32,
        /// 切片重叠比例 0~0.8（--slice 时生效）
        #[arg(long, default_value_t = 0.2)]
        slice_overlap: f32,
        /// 可视化输出目录：把检测框（按类着色 + 类别/分数标签）画在原图上，
        /// 存为 <DIR>/<输入文件名>.jpg
        #[arg(long = "save-viz", value_name = "DIR")]
        save_viz: Option<PathBuf>,
    },
    /// 基准 / 场景评测（v0.1 已接通：合成验证集冒烟指标；真实基准协议 M8 落地）
    Eval {
        #[arg(short, long)]
        weights: PathBuf,
        /// 数据集路径（v0.1 未使用，评测走合成验证集；保留参数给 M8）
        #[arg(long)]
        dataset: Option<PathBuf>,
        /// 扰动矩阵：rotate,shift,bg（M8 生效）
        #[arg(long)]
        perturb: Option<String>,
        #[arg(long)]
        report: Option<PathBuf>,
    },
    /// 权重导出：safetensors（已实现，单文件跨生态）| torch 目录格式
    Export {
        #[arg(short, long)]
        weights: PathBuf,
        #[arg(long, default_value = "safetensors")]
        format: String,
        /// 输出路径（safetensors 需 .safetensors 后缀；缺省 = weights 同目录 model.safetensors）
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// 教师蒸馏（M9 实现）
    Distill {
        #[arg(long)]
        teacher: PathBuf,
        #[arg(short, long)]
        config: PathBuf,
    },
    /// 骨干预算选型（M9 实现）
    Nas {
        #[arg(long)]
        max_latency_ms: Option<f32>,
        #[arg(long)]
        max_params_m: Option<f32>,
    },
    /// 观测面板（已接通：浏览 runs / 查看 config 快照与指标报告，5 秒自动刷新）
    Panel {
        #[arg(long, default_value_t = 8080)]
        port: u16,
        #[arg(long, default_value = "runs")]
        runs_dir: PathBuf,
    },
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .compact()
        .init();

    av_plugins::register_all()?;
    #[cfg(feature = "torch")]
    av_tasks::register_builtin()?;

    match Cli::parse().command {
        Command::Init {
            task,
            data,
            out,
            classes,
        } => cmd_init(task, data.as_deref(), out.as_deref(), classes),
        Command::Train {
            config,
            data,
            task,
            imgsz,
            epochs,
            batch,
            resume,
            overrides,
            dry_run,
        } => cmd_train(
            config.as_deref(),
            data.as_deref(),
            &task,
            imgsz,
            epochs,
            batch,
            resume,
            &overrides,
            dry_run,
        ),
        Command::Pack { src, out } => {
            let (count, bytes) =
                av_runtime::avpack::pack_dir(&src, &out).map_err(anyhow::Error::from)?;
            println!("打包完成 ✔ {count} 个文件 / {bytes} 字节 → {}", out.display());
            println!("下一步: data.sources.train.avpack 指向此文件（加载器按 M2 接入）");
            Ok(())
        }
        Command::Eval {
            weights,
            dataset: _dataset,
            perturb: _perturb,
            report,
        } => cmd_eval(&weights, report.as_deref()),
        Command::Panel { port, runs_dir } => cmd_panel(port, &runs_dir),
        Command::Export {
            weights,
            format,
            out,
        } => cmd_export(&weights, &format, out.as_deref()),
        other => bail!(
            "{}：尚未实现（按 PLAN §8 路线图排期）\n  下一步：当前可用子命令 av init / av train / av infer / av eval（av train --dry-run 可先校验配置）",
            milestone_of(&other)
        ),
    }
}

fn milestone_of(cmd: &Command) -> &'static str {
    match cmd {
        Command::Init { .. } => "av init",
        Command::Pack { .. } => "av pack（已接通：.avpack 容器打包）",
        Command::Eval { .. } => "av eval（v0.1 合成冒烟已接通）",
        Command::Export { .. } => "av export（safetensors 已接通）",
        Command::Distill { .. } => "av distill（M9）",
        Command::Nas { .. } => "av nas（M9）",
        Command::Panel { .. } => "av panel（已接通：runs 浏览 + 指标视图）",
        Command::Train { .. } => "av train",
        Command::Infer { .. } => "av infer",
    }
}

#[cfg(feature = "torch")]
fn cmd_train(
    config_path: Option<&Path>,
    data: Option<&Path>,
    task: &str,
    imgsz: u32,
    epochs: u32,
    batch: u32,
    resume: bool,
    overrides: &[String],
    dry_run: bool,
) -> Result<()> {
    let mut cfg = match (config_path, data) {
        (Some(p), _) => load_config(p)?,
        (None, Some(d)) => synthesize_config(d, task, imgsz, epochs, batch)?,
        (None, None) => bail!("--data 与 -c 至少提供一个（--data 走一条命令训练）"),
    };
    apply_overrides(&mut cfg, overrides)?;
    // 训练前的数据侧预检（CLI 层，库不改）：把「目录不存在 / 类数不匹配」拦在
    // 整集预解码之前，错误信息带下一步动作。
    preflight_data_check(&cfg)?;

    if dry_run {
        print_run_plan(&cfg);
        return Ok(());
    }
    run_train(&cfg, resume)
}

/// 一条命令训练的配置合成：目录或 data.yaml → RunConfig。
/// 默认装配 = csp-elan（YOLOv8n 同构）+ TAL 检测头 + mosaic/flip/hsv 增强 +
/// AdamW/余弦退火 + AMP + 内容贴片缓存；类数从标注自动探测。
#[cfg(feature = "torch")]
fn synthesize_config(
    data: &Path,
    task: &str,
    imgsz: u32,
    epochs: u32,
    batch: u32,
) -> Result<RunConfig> {
    use av_core::config::{
        parse_data_yaml, AugmentCfg, DataConfig, DataPipeline, DataSourceCfg, DataSources,
        ModelConfig, OptimizerCfg, SchedulerCfg, SourceTaskCfg, TaskCfg, TaskKind, TrainConfig,
    };
    if task != "detect" {
        bail!("--data 一条命令训练当前支持 detect（其余任务用 -c 配置文件）");
    }
    // run_id 取数据集目录名（data.yaml 时取其父目录，比文件名 "data" 更有辨识度）
    let run_id = data
        .parent()
        .and_then(|p| p.file_name())
        .or_else(|| data.file_stem())
        .and_then(|s| s.to_str())
        .unwrap_or("yolo")
        .to_string();
    let lower = data
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    let (train_src, val_src, num_classes) = if matches!(lower.as_str(), "yaml" | "yml") {
        let (root, train, val, names) = parse_data_yaml(data)?;
        let source = |split: String, aug: bool| DataSourceCfg {
            dir: Some(root.clone()),
            split: Some(split),
            tasks: if aug {
                vec![SourceTaskCfg {
                    kind: TaskKind::Detect,
                    format: Some("yolo".into()),
                    augment: AugmentCfg {
                        mosaic: 1.0,
                        flip: 0.5,
                        hsv: [0.1, 0.1, 0.1],
                        scale_jitter: Some([0.9, 1.1]),
                        close_last_epochs: 15,
                        ..AugmentCfg::default()
                    },
                }]
            } else {
                vec![]
            },
            ..Default::default()
        };
        (source(train, true), source(val, false), names.len() as u32)
    } else {
        let n = scan_max_class_id(&data.join("labels/train"))
            .or_else(|| scan_max_class_id(&data.join("labels")))
            .map(|m| m + 1)
            .ok_or_else(|| anyhow::anyhow!("未在 {} 下找到标注（labels/train）", data.display()))?;
        let source = |aug: bool| DataSourceCfg {
            dir: Some(data.to_path_buf()),
            tasks: if aug {
                vec![SourceTaskCfg {
                    kind: TaskKind::Detect,
                    format: Some("yolo".into()),
                    augment: AugmentCfg {
                        mosaic: 1.0,
                        flip: 0.5,
                        hsv: [0.1, 0.1, 0.1],
                        scale_jitter: Some([0.9, 1.1]),
                        close_last_epochs: 15,
                        ..AugmentCfg::default()
                    },
                }]
            } else {
                vec![]
            },
            ..Default::default()
        };
        (source(true), source(false), n)
    };
    let batch = if batch == 0 {
        if tch::Device::cuda_if_available() == tch::Device::Cpu {
            4
        } else {
            8
        }
    } else {
        batch
    };
    Ok(RunConfig {
        run_id: format!("av-{run_id}"),
        model: ModelConfig {
            tasks: vec![TaskCfg::Detect(av_core::config::DetectCfg {
                num_classes: num_classes as usize,
                img_size: imgsz,
                ..Default::default()
            })],
            ..Default::default()
        },
        data: DataConfig {
            pipeline: DataPipeline::Dir,
            sources: DataSources {
                train: train_src,
                val: val_src,
            },
            ..Default::default()
        },
        train: TrainConfig {
            epochs,
            batch_size: batch,
            optimizer: OptimizerCfg {
                lr: 1e-3,
                ..OptimizerCfg::default()
            },
            scheduler: SchedulerCfg::default(),
            warmup_epochs: 5.0,
            amp: true,
            grad_clip: 10.0,
            ..TrainConfig::default()
        },
        ..RunConfig::default()
    })
}

#[cfg(not(feature = "torch"))]
fn cmd_train(
    _config_path: Option<&Path>,
    _data: Option<&Path>,
    _task: &str,
    _imgsz: u32,
    _epochs: u32,
    _batch: u32,
    _resume: bool,
    _overrides: &[String],
    _dry_run: bool,
) -> Result<()> {
    bail!("本二进制未启用 torch feature，无法训练")
}

#[cfg(feature = "torch")]
fn run_train(cfg: &av_core::config::RunConfig, resume: bool) -> Result<()> {
    let report = if resume {
        av_runtime::engine::train_resumed(cfg)
    } else {
        av_runtime::engine::train(cfg)
    }
    .map_err(anyhow::Error::from)?;
    println!("训练完成 ✔");
    println!("  task        = {}", report.task);
    println!("  final_loss  = {:.4}", report.final_loss);
    println!("  {} = {:.3}", report.metric, report.metric_value);
    if let Some((name, v)) = &report.secondary {
        println!("  {name} = {v:.3}");
    }
    println!("  权重与报告  = {}", report.run_dir);
    Ok(())
}

#[cfg(not(feature = "torch"))]
fn run_train(_cfg: &av_core::config::RunConfig, _resume: bool) -> Result<()> {
    bail!("本二进制未启用 torch feature（--features torch），无法训练；可用 --dry-run 校验配置")
}

#[cfg(feature = "torch")]
#[allow(clippy::too_many_arguments)]
fn cmd_infer(
    weights: &Path,
    config: Option<&std::path::Path>,
    input: &Path,
    conf: f32,
    iou: f32,
    slice: bool,
    slice_window: u32,
    slice_overlap: f32,
    save_viz: Option<&Path>,
) -> Result<()> {
    let _ = (conf, iou); // v0.1 引擎使用内置默认值；参数生效按 M2 批推理接口
    let cfg = resolve_config_for_weights(weights, config)?;
    let result = if slice {
        av_runtime::engine::infer_sliced(&cfg, weights, input, slice_window, slice_overlap)
            .map_err(anyhow::Error::from)?
    } else {
        av_runtime::engine::infer(&cfg, weights, input).map_err(anyhow::Error::from)?
    };
    if let Some(dir) = save_viz {
        let out = save_viz_image(input, &result, dir)?;
        println!("可视化已写入 {}", out.display());
    }
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

/// 检测框按类着色调色板（8 色循环）。
#[cfg(feature = "torch")]
const VIZ_PALETTE: [[u8; 3]; 8] = [
    [80, 250, 123],
    [255, 121, 198],
    [189, 147, 249],
    [241, 250, 140],
    [139, 233, 253],
    [255, 184, 108],
    [255, 85, 85],
    [114, 213, 163],
];

#[cfg(feature = "torch")]
/// 3×5 微型字模（MSB=左列；覆盖推理标签所需的 A-Z/0-9/空格/点/冒号）。
const VIZ_FONT: [(&str, [u8; 5]); 38] = [
    ("0", [0b111, 0b101, 0b101, 0b101, 0b111]),
    ("1", [0b010, 0b110, 0b010, 0b010, 0b111]),
    ("2", [0b111, 0b001, 0b111, 0b100, 0b111]),
    ("3", [0b111, 0b001, 0b111, 0b001, 0b111]),
    ("4", [0b101, 0b101, 0b111, 0b001, 0b001]),
    ("5", [0b111, 0b100, 0b111, 0b001, 0b111]),
    ("6", [0b111, 0b100, 0b111, 0b101, 0b111]),
    ("7", [0b111, 0b001, 0b010, 0b010, 0b010]),
    ("8", [0b111, 0b101, 0b111, 0b101, 0b111]),
    ("9", [0b111, 0b101, 0b111, 0b001, 0b111]),
    ("A", [0b111, 0b101, 0b111, 0b101, 0b101]),
    ("B", [0b110, 0b101, 0b110, 0b101, 0b110]),
    ("C", [0b111, 0b100, 0b100, 0b100, 0b111]),
    ("D", [0b110, 0b101, 0b101, 0b101, 0b110]),
    ("E", [0b111, 0b100, 0b111, 0b100, 0b111]),
    ("F", [0b111, 0b100, 0b111, 0b100, 0b100]),
    ("G", [0b111, 0b100, 0b101, 0b101, 0b111]),
    ("H", [0b101, 0b101, 0b111, 0b101, 0b101]),
    ("I", [0b111, 0b010, 0b010, 0b010, 0b111]),
    ("J", [0b011, 0b001, 0b001, 0b101, 0b111]),
    ("K", [0b101, 0b110, 0b100, 0b110, 0b101]),
    ("L", [0b100, 0b100, 0b100, 0b100, 0b111]),
    ("M", [0b101, 0b111, 0b111, 0b101, 0b101]),
    ("N", [0b101, 0b111, 0b111, 0b111, 0b101]),
    ("O", [0b111, 0b101, 0b101, 0b101, 0b111]),
    ("P", [0b111, 0b101, 0b111, 0b100, 0b100]),
    ("R", [0b111, 0b101, 0b111, 0b110, 0b101]),
    ("S", [0b111, 0b100, 0b111, 0b001, 0b111]),
    ("T", [0b111, 0b010, 0b010, 0b010, 0b010]),
    ("U", [0b101, 0b101, 0b101, 0b101, 0b111]),
    ("V", [0b101, 0b101, 0b101, 0b101, 0b010]),
    ("W", [0b101, 0b101, 0b111, 0b111, 0b101]),
    ("X", [0b101, 0b101, 0b010, 0b101, 0b101]),
    ("Y", [0b101, 0b101, 0b010, 0b010, 0b010]),
    ("Z", [0b111, 0b001, 0b010, 0b100, 0b111]),
    (" ", [0b000, 0b000, 0b000, 0b000, 0b000]),
    (".", [0b000, 0b000, 0b000, 0b000, 0b010]),
    (":", [0b010, 0b010, 0b000, 0b010, 0b010]),
];

#[cfg(feature = "torch")]
fn viz_glyph(ch: char) -> &'static [u8; 5] {
    let upper = ch.to_uppercase().next().unwrap_or(ch);
    let s = upper.to_string();
    VIZ_FONT
        .iter()
        .find(|(name, _)| *name == s)
        .map(|(_, g)| g)
        .unwrap_or(&[0b111, 0b101, 0b101, 0b101, 0b111])
}

#[cfg(feature = "torch")]
fn viz_put_pixel(img: &mut image::RgbImage, x: i64, y: i64, c: [u8; 3]) {
    if x >= 0 && y >= 0 && (x as u32) < img.width() && (y as u32) < img.height() {
        img.put_pixel(x as u32, y as u32, image::Rgb(c));
    }
}

#[cfg(feature = "torch")]
fn viz_hline(img: &mut image::RgbImage, x1: i64, x2: i64, y: i64, c: [u8; 3], t: i64) {
    for dy in 0..t {
        for x in x1.min(x2)..=x2.max(x1) {
            viz_put_pixel(img, x, y + dy, c);
        }
    }
}

#[cfg(feature = "torch")]
fn viz_vline(img: &mut image::RgbImage, x: i64, y1: i64, y2: i64, c: [u8; 3], t: i64) {
    for dx in 0..t {
        for y in y1.min(y2)..=y2.max(y1) {
            viz_put_pixel(img, x + dx, y, c);
        }
    }
}

#[cfg(feature = "torch")]
fn viz_text(img: &mut image::RgbImage, x: i64, y: i64, text: &str, scale: i64, c: [u8; 3]) {
    for (i, ch) in text.chars().enumerate() {
        let glyph = viz_glyph(ch);
        let gx = x + i as i64 * 4 * scale;
        for (row, bits) in glyph.iter().enumerate() {
            for col in 0..3 {
                if bits & (0b100 >> col) != 0 {
                    for dy in 0..scale {
                        for dx in 0..scale {
                            viz_put_pixel(
                                img,
                                gx + col * scale + dx,
                                y + row as i64 * scale + dy,
                                c,
                            );
                        }
                    }
                }
            }
        }
    }
}

/// 把检测/分类结果画到原图并存盘（`--save-viz`）。检测：框（按类着色，2px）
/// + 黑底标签 `C<id> <score>`；分类：左上角列出前 3 个预测。
#[cfg(feature = "torch")]
fn save_viz_image(input: &Path, result: &serde_json::Value, dir: &Path) -> Result<PathBuf> {
    let img = image::open(input)
        .map_err(|e| anyhow::anyhow!("读图失败 {}: {e}", input.display()))?
        .to_rgb8();
    let mut img = img;
    match result.get("task").and_then(|v| v.as_str()) {
        Some("detect") | Some("obb") => {
            if let Some(dets) = result.get("detections").and_then(|v| v.as_array()) {
                for d in dets {
                    let (Some(bbox), Some(class_id)) =
                        (d.get("bbox"), d.get("class_id").and_then(|v| v.as_u64()))
                    else {
                        continue;
                    };
                    let (Some(x1), Some(y1), Some(x2), Some(y2)) = (
                        bbox.get("x1").and_then(|v| v.as_f64()),
                        bbox.get("y1").and_then(|v| v.as_f64()),
                        bbox.get("x2").and_then(|v| v.as_f64()),
                        bbox.get("y2").and_then(|v| v.as_f64()),
                    ) else {
                        continue;
                    };
                    let color = VIZ_PALETTE[(class_id as usize) % VIZ_PALETTE.len()];
                    let (x1, y1, x2, y2) = (x1 as i64, y1 as i64, x2 as i64, y2 as i64);
                    viz_hline(&mut img, x1, x2 + 2, y1, color, 2);
                    viz_hline(&mut img, x1, x2 + 2, y2, color, 2);
                    viz_vline(&mut img, x1, y1, y2 + 2, color, 2);
                    viz_vline(&mut img, x2, y1, y2 + 2, color, 2);
                    let score = d.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let tag = format!("C{class_id} {score:.2}");
                    let (tw, th) = (tag.len() as i64 * 8 + 2, 12);
                    for dy in 0..th {
                        for dx in 0..tw {
                            viz_put_pixel(&mut img, x1 + dx, y1 - th + dy, [0, 0, 0]);
                        }
                    }
                    viz_text(&mut img, x1 + 1, y1 - th + 2, &tag, 2, color);
                }
            }
        }
        Some("classify") => {
            if let Some(preds) = result.get("predictions").and_then(|v| v.as_array()) {
                for (i, p) in preds.iter().take(3).enumerate() {
                    let class_id = p.get("class_id").and_then(|v| v.as_u64()).unwrap_or(0);
                    let prob = p.get("prob").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let tag = format!("C{class_id} {prob:.2}");
                    viz_text(&mut img, 4, 4 + i as i64 * 12, &tag, 2, VIZ_PALETTE[i % 8]);
                }
            }
        }
        _ => {}
    }
    std::fs::create_dir_all(dir)?;
    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("viz")
        .to_string();
    let out = dir.join(format!("{stem}.jpg"));
    img.save_with_format(&out, image::ImageFormat::Jpeg)?;
    Ok(out)
}

#[cfg(not(feature = "torch"))]
fn cmd_infer(
    _weights: &Path,
    _config: Option<&std::path::Path>,
    _input: &Path,
    _conf: f32,
    _iou: f32,
    _slice: bool,
    _slice_window: u32,
    _slice_overlap: f32,
    _save_viz: Option<&Path>,
) -> Result<()> {
    bail!("本二进制未启用 torch feature（--features torch），无法推理")
}

#[cfg(feature = "torch")]
fn cmd_eval(weights: &Path, report: Option<&std::path::Path>) -> Result<()> {
    let cfg = resolve_config_for_weights(weights, None)?;
    let result = av_runtime::engine::eval(&cfg, weights).map_err(anyhow::Error::from)?;
    let text = serde_json::to_string_pretty(&result)?;
    println!("{text}");
    if let Some(path) = report {
        std::fs::write(path, text)?;
        println!("报告已写入 {}", path.display());
    }
    Ok(())
}

#[cfg(all(feature = "torch", feature = "panel"))]
fn cmd_export(weights: &Path, format: &str, out: Option<&std::path::Path>) -> Result<()> {
    let out_path = match out {
        Some(p) => p.to_path_buf(),
        None => {
            let stem = weights
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .context("权重路径无文件名")?;
            weights
                .parent()
                .context("权重路径无父目录")?
                .join(format!("{stem}.safetensors"))
        }
    };
    let cfg = resolve_config_for_weights(weights, None)?;
    // 顺序很关键：先按配置建模型（VarStore 才有变量），再载入 checkpoint 覆写
    let mut vs = tch::nn::VarStore::new(tch::Device::Cpu);
    let _model = av_tasks::models::build_model(&vs.root(), &cfg).map_err(anyhow::Error::from)?;
    av_runtime::engine::load_checkpoint_dir(&mut vs, weights).map_err(anyhow::Error::from)?;
    av_runtime::engine::export_checkpoint(&vs, &out_path, format).map_err(anyhow::Error::from)?;
    println!("导出完成 ✔ {} ({})", out_path.display(), format);
    Ok(())
}

#[cfg(not(all(feature = "torch", feature = "panel")))]
fn cmd_export(_weights: &Path, _format: &str, _out: Option<&std::path::Path>) -> Result<()> {
    bail!("本二进制未启用 torch feature，无法导出")
}

#[cfg(all(feature = "panel", feature = "torch"))]
fn cmd_panel(port: u16, runs_dir: &Path) -> Result<()> {
    let rt = tokio::runtime::Runtime::new().context("创建 tokio 运行时失败")?;
    rt.block_on(async move {
        av_runtime::panel::serve(runs_dir.to_path_buf(), port)
            .await
            .map_err(anyhow::Error::from)
    })
}

#[cfg(not(all(feature = "panel", feature = "torch")))]
fn cmd_panel(_port: u16, _runs_dir: &Path) -> Result<()> {
    bail!("面板需要构建时启用 panel + torch feature（默认已启用）")
}

#[cfg(not(feature = "torch"))]
fn cmd_eval(_weights: &Path, _report: Option<&std::path::Path>) -> Result<()> {
    bail!("本二进制未启用 torch feature（--features torch），无法评测")
}

fn load_config(config_path: &Path) -> Result<av_core::config::RunConfig> {
    if !config_path.exists() {
        bail!(
            "配置文件不存在: {}\n  原因：路径拼写错误，或当前工作目录不是工作区根目录\n  下一步：运行 av init --task detect --data <数据目录> 自动生成最小配置；或直接用开箱即训的 configs/quick_detect.toml（合成数据）",
            config_path.display()
        );
    }
    av_core::config::RunConfig::from_path(config_path)
        .with_context(|| format!("读取配置 {}", config_path.display()))
}

fn apply_overrides(cfg: &mut av_core::config::RunConfig, overrides: &[String]) -> Result<()> {
    for ov in overrides {
        let (k, v) = ov.split_once('=').with_context(|| {
            format!(
                "--override 需要 key=value 形式，得到 {ov:?}\n  下一步：例如 --override seed=42 --override device=cpu --override output_dir=runs"
            )
        })?;
        match k {
            "seed" => cfg.seed = v.parse().with_context(|| {
                format!("seed 需要整数，得到 {v:?}\n  下一步：例如 --override seed=42")
            })?,
            "device" => cfg.device = v.to_string(),
            "output_dir" => cfg.output_dir = PathBuf::from(v),
            other => bail!(
                "不支持的 --override key: {other}\n  原因：CLI 覆写目前只开放少量顶层字段\n  下一步：可用 key 为 seed / device / output_dir；其余字段请直接改配置文件"
            ),
        }
    }
    cfg.validate().context("配置校验失败")?;
    Ok(())
}

fn print_run_plan(cfg: &av_core::config::RunConfig) {
    let run_id = cfg.effective_run_id();
    let tasks = cfg
        .model
        .tasks
        .iter()
        .map(|t| t.kind_name())
        .collect::<Vec<_>>()
        .join(",");
    println!("配置有效 ✔");
    println!("  run_id     = {run_id}");
    println!("  device     = {}", cfg.device);
    println!("  tasks      = {tasks}");
    println!("  epochs     = {}", cfg.train.epochs);
    println!("  batch_size = {}", cfg.train.batch_size);
    println!("  amp        = {}", cfg.train.amp);
    println!("  输出目录    = {}", cfg.output_dir.display());
    println!(
        "  数据源     = {}",
        match cfg.data.pipeline {
            av_core::config::DataPipeline::Synthetic =>
                "synthetic（内置合成，开箱即训）".to_string(),
            av_core::config::DataPipeline::AvPack => "avpack".to_string(),
            av_core::config::DataPipeline::Dir => "dir".to_string(),
        }
    );
}

#[cfg(feature = "torch")]
fn resolve_config_for_weights(
    weights: &Path,
    explicit: Option<&std::path::Path>,
) -> Result<av_core::config::RunConfig> {
    let path = match explicit {
        Some(p) => p.to_path_buf(),
        None => weights
            .parent()
            .map(|d| d.join("config.snapshot.toml"))
            .context("权重路径无父目录")?,
    };
    av_core::config::RunConfig::from_path(&path)
        .with_context(|| format!("读取模型配置 {}", path.display()))
}

// ---------------------------------------------------------------------------
// av init：脚手架（生成最小可用 TOML；探测数据布局与类别数）
// ---------------------------------------------------------------------------

/// `av init --task` 支持的任务类型。
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum InitTask {
    Detect,
    Classify,
    Obb,
}

impl InitTask {
    fn as_str(self) -> &'static str {
        match self {
            InitTask::Detect => "detect",
            InitTask::Classify => "classify",
            InitTask::Obb => "obb",
        }
    }
}

/// 数据目录布局探测结果。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DataLayout {
    /// images/<split> + labels/<split>（YOLO / DOTA 通用外壳）
    YoloDir,
    /// <split>/<wnid>/ 两级子目录（ImageNet ImageFolder）
    ImageFolder,
    Unknown,
}

const IMAGE_EXTS: &[&str] = &["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"];

/// 扫描标签目录（*.txt，每行首 token = 类 id），返回最大类 id。
/// 上限 4096 个文件，避免超大库在 CLI 预检里卡顿（类数统计只求上界样本）。
fn scan_max_class_id(labels_dir: &Path) -> Option<u32> {
    let mut max: Option<u32> = None;
    let mut scanned = 0usize;
    let Ok(rd) = fs::read_dir(labels_dir) else {
        return None;
    };
    for entry in rd.flatten() {
        let p = entry.path();
        if p.extension().and_then(|e| e.to_str()) != Some("txt") {
            continue;
        }
        let Ok(text) = fs::read_to_string(&p) else {
            continue;
        };
        for line in text.lines() {
            if let Some(cls) = line
                .split_whitespace()
                .next()
                .and_then(|t| t.parse::<u32>().ok())
            {
                max = Some(max.map_or(cls, |m: u32| m.max(cls)));
            }
        }
        scanned += 1;
        if scanned >= 4096 {
            break;
        }
    }
    max
}

/// YOLO/DOTA 布局下选一个有标注的 split 目录：labels/train → labels/val → labels 本体。
fn label_split_dir(root: &Path) -> PathBuf {
    let labels = root.join("labels");
    for split in ["train", "val"] {
        let d = labels.join(split);
        if d.is_dir() {
            return d;
        }
    }
    labels
}

fn dir_has_image(dir: &Path) -> bool {
    let Ok(rd) = fs::read_dir(dir) else {
        return false;
    };
    rd.flatten().any(|e| {
        e.path()
            .extension()
            .and_then(|x| x.to_str())
            .map(|x| IMAGE_EXTS.contains(&x.to_ascii_lowercase().as_str()))
            .unwrap_or(false)
    })
}

/// 统计 split 目录下含图片的 wnid 子目录数（ImageFolder 类别数）。
fn count_wnid_dirs(split_dir: &Path) -> Option<usize> {
    let rd = fs::read_dir(split_dir).ok()?;
    let mut n = 0usize;
    for entry in rd.flatten() {
        let p = entry.path();
        if p.is_dir() && dir_has_image(&p) {
            n += 1;
        }
    }
    (n > 0).then_some(n)
}

fn detect_data_layout(dir: &Path) -> DataLayout {
    if dir.join("images").is_dir() || dir.join("labels").is_dir() {
        return DataLayout::YoloDir;
    }
    // ImageFolder：<split>/<wnid>/ 两级结构；叶子目录里应有图片
    for split in ["train", "val"] {
        let split_dir = dir.join(split);
        if split_dir.is_dir() && count_wnid_dirs(&split_dir).is_some() {
            return DataLayout::ImageFolder;
        }
    }
    DataLayout::Unknown
}

/// Windows 绝对路径（含 `\`）写入 TOML 字符串会撞转义，统一转成正斜杠。
fn to_toml_path(p: &Path) -> String {
    p.display().to_string().replace('\\', "/")
}

/// 生成最小可用 TOML：非注释有效行 ≤ 12，其余超参全部走 RunConfig 内置默认。
/// 纯函数（不碰文件系统），便于单测直接喂 RunConfig::from_toml_str。
fn build_init_toml(
    task: InitTask,
    data: Option<&Path>,
    _layout: Option<DataLayout>,
    num_classes: usize,
    out_display: &str,
) -> String {
    let run_id = format!("my-{}", task.as_str());
    let model_block = "\
[model]
backbone = { family = \"simple-cnn\", depth = 1.0, width = 1.0, pretrained = \"none\" }
neck = { type = \"identity\", channels = [] }
";
    match (task, data) {
        // ---- 真实数据：detect（YOLO 目录）----
        (InitTask::Detect, Some(data)) => format!(
            "# av init 生成 —— detect 最小配置（其余超参走内置默认；调优参考 configs/detect_coco8.toml）\n\
             # 数据：{data}（探测到 YOLO 目录格式：images/<split> + labels/<split>，标注行为 cls cx cy w h）\n\
             # 类别数：{n}（自动统计 = 标注最大类 id + 1；对齐官方类表可用 --classes 覆盖）\n\
             # 下一步：av train -c {out}（先加 --dry-run 可只校验配置）\n\
             run_id = \"{run_id}\"\n\
             \n\
             {model_block}\n\
             [[model.tasks]]\n\
             kind = \"detect\"\n\
             num_classes = {n}\n\
             img_size = 320\n\
             \n\
             [data]\n\
             pipeline = \"dir\"\n\
             \n\
             [data.sources.train]\n\
             dir = \"{data}\"\n",
            data = to_toml_path(data),
            n = num_classes,
            out = out_display,
        ),
        // ---- 真实数据：obb（DOTA 9 列，detect 任务 + obb_mode）----
        (InitTask::Obb, Some(data)) => format!(
            "# av init 生成 —— obb 旋转框检测最小配置（其余超参走内置默认；参考 configs/detect_obb_dota8.toml）\n\
             # 数据：{data}（探测到 images/<split> + labels/<split>，按 DOTA 格式解析：cls x1 y1 x2 y2 x3 y3 x4 y4）\n\
             # 类别数：{n}（自动统计 = 标注最大类 id + 1；可用 --classes 覆盖）\n\
             # 下一步：av train -c {out}\n\
             run_id = \"{run_id}\"\n\
             \n\
             {model_block}\n\
             [[model.tasks]]\n\
             kind = \"detect\"          # OBB = detect 任务 + obb_mode = true\n\
             obb_mode = true\n\
             num_classes = {n}\n\
             img_size = 320\n\
             \n\
             [data.sources.train]      # data.pipeline 缺省即 \"dir\"\n\
             dir = \"{data}\"\n",
            data = to_toml_path(data),
            n = num_classes,
            out = out_display,
        ),
        // ---- 真实数据：classify（ImageFolder，data_dir 直连最省行）----
        (InitTask::Classify, Some(data)) => format!(
            "# av init 生成 —— classify 最小配置（其余超参走内置默认；参考 configs/classify_imagenette.toml）\n\
             # 数据：{data}（探测到 ImageFolder 结构：<split>/<wnid>/，类 id 按 wnid 字典序映射 0..N）\n\
             # 类别数：{n}（统计 train split 的 wnid 子目录数；可用 --classes 覆盖）\n\
             # 下一步：av train -c {out}\n\
             run_id = \"{run_id}\"\n\
             \n\
             {model_block}\n\
             [[model.tasks]]\n\
             kind = \"classify\"\n\
             num_classes = {n}\n\
             img_size = 64\n\
             data_dir = \"{data}\"\n\
             \n\
             [data]\n\
             pipeline = \"dir\"\n",
            data = to_toml_path(data),
            n = num_classes,
            out = out_display,
        ),
        // ---- 无数据：synthetic（开箱即训；OBB 无合成源，cmd_init 已拦截）----
        (task, None) => format!(
            "# av init 生成 —— {task} 最小配置（内置合成数据：无需数据集，开箱即训）\n\
             # 下一步：av train -c {out}；接到真实数据后重新运行 av init --task {task} --data <目录>\n\
             run_id = \"{run_id}\"\n\
             \n\
             {model_block}\n\
             [[model.tasks]]\n\
             kind = \"{kind}\"\n\
             num_classes = {n}\n\
             img_size = 64\n\
             \n\
             [data]\n\
             pipeline = \"synthetic\"\n",
            task = task.as_str(),
            kind = if task == InitTask::Obb { "obb" } else { task.as_str() },
            n = num_classes,
            out = out_display,
        ),
    }
}

/// `av init` 入口：校验 → 探测 → 生成。
fn cmd_init(
    task: InitTask,
    data: Option<&Path>,
    out: Option<&Path>,
    classes: Option<usize>,
) -> Result<()> {
    // --data 存在性（三段式：发生什么 / 为什么 / 下一步）
    if let Some(p) = data {
        if !p.exists() {
            bail!(
                "数据目录不存在: {}\n  原因：路径拼写错误、数据未下载，或当前工作目录不是工作区根目录\n  下一步：① 确认路径后重试；② 用脚本准备数据（如 scripts/get-data.ps1）；③ 省略 --data 生成合成数据配置先跑通流程",
                p.display()
            );
        }
        if !p.is_dir() {
            bail!(
                "--data 应指向数据集根目录，得到文件: {}\n  下一步：传入含 images/labels（检测）或 train/val（分类）的目录；打包容器请用 av pack --src <目录> --out <文件>（M2）",
                p.display()
            );
        }
    }

    // OBB 无合成数据源（引擎要求 dir + DOTA 格式），无 --data 时直接给出出路
    if task == InitTask::Obb && data.is_none() {
        bail!(
            "--task obb 需要真实数据（引擎的 OBB 路径只支持 dir 数据源的 DOTA 格式，无合成源）\n  下一步：av init --task obb --data <dota 数据目录>（结构参考 configs/detect_obb_dota8.toml 与 data/dota8）"
        );
    }

    // 布局探测 + 任务/布局匹配检查
    let layout = data.map(detect_data_layout);
    match (task, layout) {
        (InitTask::Detect | InitTask::Obb, Some(DataLayout::ImageFolder)) => bail!(
            "数据目录 {p} 是 ImageFolder 结构（<split>/<wnid>/），没有检测标注，无法用于 --task {t}\n  下一步：① 分类任务请改用 av init --task classify --data {p}；② 检测数据需含 images/<split> + labels/<split>",
            p = data.unwrap().display(),
            t = task.as_str(),
        ),
        (InitTask::Classify, Some(DataLayout::YoloDir)) => bail!(
            "数据目录 {p} 是 YOLO 结构（images/ + labels/），不是分类的 ImageFolder 结构\n  下一步：检测任务请用 av init --task detect --data {p}；分类数据需 <root>/train/<wnid>/*.图片",
            p = data.unwrap().display(),
        ),
        (_, Some(DataLayout::Unknown)) => bail!(
            "未在 {p} 识别出支持的数据布局（需要 images/<split> + labels/<split>，或 <root>/<split>/<wnid>/）\n  下一步：① 核对目录结构后重试；② 检测/分类想先跑通可省略 --data 用合成数据；③ 结构化打包走 av pack（M2）",
            p = data.unwrap().display(),
        ),
        _ => {}
    }

    // 类别数：显式 --classes 优先，否则自动统计；合成源走友好默认
    let (num_classes, detected) = match data {
        None => (
            classes.unwrap_or(match task {
                InitTask::Classify => 4,
                _ => 2,
            }),
            None,
        ),
        Some(p) => {
            let detected = auto_class_count(task, p);
            match classes.or(detected) {
                Some(n) => (n, detected),
                None => bail!(
                    "无法自动统计 {p} 的类别数（labels 无有效标注 / split 下无 wnid 子目录）\n  下一步：加 --classes <N> 显式指定（detect/obb ≥ 1，classify ≥ 2）",
                    p = p.display(),
                ),
            }
        }
    };
    let min_classes = if task == InitTask::Classify { 2 } else { 1 };
    if num_classes < min_classes {
        bail!(
            "类别数 {num_classes} 非法（{task} 至少需要 {min_classes}）\n  下一步：用 --classes <N> 指定，或核对数据目录是否数错",
            task = task.as_str(),
        );
    }
    // 显式 --classes 与自动统计不一致 → 温和提醒，不阻断
    if let (Some(given), Some(auto)) = (classes, detected) {
        if given != auto {
            println!(
                "提示：--classes {given} 与数据自动统计 {auto} 不一致（已按 {given} 写入；训练前请再确认）"
            );
        }
    }

    // 输出路径：缺省 configs/av_<task>.toml（无 configs/ 目录时落当前目录）；不覆盖已有文件
    let out_path = match out {
        Some(p) => p.to_path_buf(),
        None => {
            let base: PathBuf = if Path::new("configs").is_dir() {
                PathBuf::from("configs")
            } else {
                PathBuf::from(".")
            };
            base.join(format!("av_{}.toml", task.as_str()))
        }
    };
    if out_path.exists() {
        bail!(
            "目标文件已存在: {}\n  原因：av init 不覆盖已有文件，避免冲掉你的手工修改\n  下一步：换一个 --out 路径，或先删除/改名旧文件",
            out_path.display()
        );
    }
    if let Some(parent) = out_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .with_context(|| format!("创建输出目录 {}", parent.display()))?;
        }
    }

    let out_display = out_path.display().to_string();
    let toml = build_init_toml(task, data, layout, num_classes, &out_display);
    fs::write(&out_path, &toml).with_context(|| format!("写出配置 {}", out_path.display()))?;

    println!("已生成最小配置 {}", out_path.display());
    match data {
        Some(p) => println!(
            "  任务 = {}，数据 = {}（{}），类别数 = {}",
            task.as_str(),
            p.display(),
            match layout.unwrap_or(DataLayout::Unknown) {
                DataLayout::YoloDir => "YOLO/DOTA images+labels 目录",
                DataLayout::ImageFolder => "ImageFolder 目录",
                DataLayout::Unknown => "未知布局",
            },
            num_classes
        ),
        None => println!(
            "  任务 = {}，数据 = 内置合成源，类别数 = {}",
            task.as_str(),
            num_classes
        ),
    }
    println!(
        "下一步：av train -c {} --dry-run   # 先校验；去掉 --dry-run 开始训练",
        out_display
    );
    Ok(())
}

/// 按布局自动统计类别数：YOLO/DOTA 取标注最大类 id+1；ImageFolder 数 wnid 子目录。
fn auto_class_count(task: InitTask, root: &Path) -> Option<usize> {
    match task {
        InitTask::Classify => {
            for split in ["train", "val"] {
                let d = root.join(split);
                if d.is_dir() {
                    return count_wnid_dirs(&d);
                }
            }
            None
        }
        InitTask::Detect | InitTask::Obb => {
            scan_max_class_id(&label_split_dir(root)).map(|m| m as usize + 1)
        }
    }
}

/// 训练前数据侧预检（CLI 层）：目录存在性 + 类数一致性，
/// 把原本藏在「整集预解码之后/模型构建期」的失败提前到毫秒级、并给出下一步。
fn preflight_data_check(cfg: &RunConfig) -> Result<()> {
    if cfg.data.pipeline != DataPipeline::Dir {
        return Ok(()); // synthetic / avpack（M2）不在此检查
    }
    let train_split = cfg.data.sources.train.split.as_deref().unwrap_or("train");
    match cfg.model.tasks.first() {
        Some(TaskCfg::Classify(c)) => {
            let Some(root) = c
                .data_dir
                .clone()
                .or_else(|| cfg.data.sources.train.dir.clone())
            else {
                return Ok(()); // validate 层已拦截缺 dir 的配置
            };
            if !root.is_dir() {
                bail!(
                    "数据目录不存在: {}\n  原因：classify 数据源指向的目录当前不可见（路径错误 / 数据未下载 / 工作目录不对）\n  下一步：① av init --task classify --data <正确目录> 重新生成；② 或先用 configs/quick_classify.toml（合成数据）跑通",
                    root.display()
                );
            }
            if let Some(n) = count_wnid_dirs(&root.join(train_split)) {
                if n != c.num_classes {
                    bail!(
                        "类数不匹配：classify.num_classes = {cfg_n}，但 {root}/{split} 下数出 {n} 个含图片的 wnid 子目录\n  原因：类 id 按 wnid 字典序映射 0..N，头结点维度必须与目录类别数一致\n  下一步：把配置中 num_classes 改为 {n}，或重新运行 av init --task classify --data {root} 自动统计",
                        cfg_n = c.num_classes,
                        root = to_toml_path(&root),
                        split = train_split,
                    );
                }
            }
        }
        Some(TaskCfg::Detect(d)) => {
            let Some(root) = cfg.data.sources.train.dir.clone() else {
                return Ok(());
            };
            if !root.is_dir() {
                bail!(
                    "数据目录不存在: {}\n  原因：data.sources.train.dir 指向的目录当前不可见（路径错误 / 数据未下载 / 工作目录不对）\n  下一步：① 检测数据用 scripts/get-data.ps1 或参考 configs/detect_coco8.toml 下载；② 重新 av init --task {} --data <目录>；③ 或先用 configs/quick_detect.toml（合成数据）跑通",
                    root.display(),
                    if d.obb_mode { "obb" } else { "detect" },
                );
            }
            let labels = root.join("labels").join(train_split);
            if let Some(max) = scan_max_class_id(&labels) {
                let need = max as usize + 1;
                if need > d.num_classes {
                    bail!(
                        "类数不匹配：num_classes = {cfg_n}，但 {labels} 标注中出现类 id {max}（至少需要 {need} 类）\n  原因：类 id 超出头结点输出范围会在训练/解码期越界\n  下一步：把 [[model.tasks]] num_classes 改为 >= {need}，或重新运行 av init --task {} --data {} 自动统计",
                        if d.obb_mode { "obb" } else { "detect" },
                        to_toml_path(&root),
                        cfg_n = d.num_classes,
                        labels = to_toml_path(&labels),
                    );
                }
            }
        }
        _ => {}
    }
    Ok(())
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod init_tests {
    use super::*;

    fn temp_dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("av-init-test-{}-{}", std::process::id(), tag));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    fn touch(path: &Path) {
        if let Some(p) = path.parent() {
            fs::create_dir_all(p).unwrap();
        }
        fs::write(path, b"x").unwrap();
    }

    fn effective_lines(toml: &str) -> usize {
        toml.lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .count()
    }

    #[test]
    fn init_toml_detect_dir_parses() {
        let t = build_init_toml(
            InitTask::Detect,
            Some(Path::new("data/coco8")),
            Some(DataLayout::YoloDir),
            80,
            "configs/my_first.toml",
        );
        let cfg = RunConfig::from_toml_str(&t).expect("detect 配置必须可解析");
        assert!(effective_lines(&t) <= 12, "有效行数超预算:\n{t}");
        assert_eq!(cfg.model.tasks[0].kind_name(), "detect");
    }

    #[test]
    fn init_toml_classify_dir_parses() {
        let t = build_init_toml(
            InitTask::Classify,
            Some(Path::new("data/imagenette2-160")),
            Some(DataLayout::ImageFolder),
            10,
            "configs/my_first.toml",
        );
        let cfg = RunConfig::from_toml_str(&t).expect("classify 配置必须可解析");
        assert!(effective_lines(&t) <= 12);
        assert_eq!(cfg.model.tasks[0].kind_name(), "classify");
    }

    #[test]
    fn init_toml_obb_dir_parses() {
        let t = build_init_toml(
            InitTask::Obb,
            Some(Path::new("data/dota8")),
            Some(DataLayout::YoloDir),
            15,
            "configs/my_obb.toml",
        );
        let cfg = RunConfig::from_toml_str(&t).expect("obb 配置必须可解析");
        assert!(effective_lines(&t) <= 12);
        assert_eq!(cfg.model.tasks[0].kind_name(), "detect");
        assert!(matches!(&cfg.model.tasks[0], TaskCfg::Detect(d) if d.obb_mode));
    }

    #[test]
    fn init_toml_synthetic_detect_and_classify_parse() {
        for task in [InitTask::Detect, InitTask::Classify] {
            let t = build_init_toml(task, None, None, 3, "av_x.toml");
            let cfg = RunConfig::from_toml_str(&t)
                .unwrap_or_else(|e| panic!("{:?} 配置解析失败: {e}\n{t}", task));
            assert_eq!(cfg.data.pipeline, DataPipeline::Synthetic);
        }
    }

    #[test]
    fn init_toml_windows_backslash_path_still_parses() {
        // Windows 绝对路径的 `\` 会被 TOML 当转义符，必须已归一为 `/`
        let t = build_init_toml(
            InitTask::Detect,
            Some(Path::new("E:\\dev\\aegis-vision\\data\\coco8")),
            Some(DataLayout::YoloDir),
            80,
            "x.toml",
        );
        assert!(
            RunConfig::from_toml_str(&t).is_ok(),
            "反斜杠路径破坏 TOML:\n{t}"
        );
    }

    #[test]
    fn quick_templates_parse() {
        let detect = include_str!("../../../configs/quick_detect.toml");
        let cfg = RunConfig::from_toml_str(detect).expect("quick_detect 模板必须可解析");
        assert_eq!(cfg.model.tasks[0].kind_name(), "detect");
        assert_eq!(cfg.data.pipeline, DataPipeline::Synthetic);
        assert!(cfg.train.epochs <= 20);

        let classify = include_str!("../../../configs/quick_classify.toml");
        let cfg = RunConfig::from_toml_str(classify).expect("quick_classify 模板必须可解析");
        assert_eq!(cfg.model.tasks[0].kind_name(), "classify");
    }

    #[test]
    fn layout_detection_and_class_count() {
        // YOLO 布局：images/ + labels/train（最大类 id = 7 → 8 类）
        let d = temp_dir("yolo");
        touch(&d.join("images/train/a.jpg"));
        touch(&d.join("labels/train/a.txt"));
        fs::write(
            d.join("labels/train/a.txt"),
            b"3 0.5 0.5 0.1 0.1\n7 0.1 0.1 0.1 0.1\n",
        )
        .unwrap();
        assert_eq!(detect_data_layout(&d), DataLayout::YoloDir);
        assert_eq!(auto_class_count(InitTask::Detect, &d), Some(8));

        // ImageFolder 布局：train/<wnid>/ 含图片 → 2 类
        let c = temp_dir("imagefolder");
        touch(&c.join("train/n01440764/a.jpg"));
        touch(&c.join("train/n02102040/b.png"));
        touch(&c.join("val/n01440764/c.jpg"));
        assert_eq!(detect_data_layout(&c), DataLayout::ImageFolder);
        assert_eq!(auto_class_count(InitTask::Classify, &c), Some(2));

        // 未知布局
        let u = temp_dir("unknown");
        touch(&u.join("misc/a.jpg"));
        assert_eq!(detect_data_layout(&u), DataLayout::Unknown);
        assert_eq!(auto_class_count(InitTask::Detect, &u), None);
    }

    #[test]
    fn preflight_flags_missing_dir_and_class_mismatch() {
        // 数据目录不存在 → 报错且信息含下一步
        let cfg = RunConfig::from_toml_str(
            "[[model.tasks]]\nkind=\"detect\"\nnum_classes=2\n[data]\npipeline=\"dir\"\n[data.sources.train]\ndir=\"Z:/no/such/dir\"\n",
        )
        .unwrap();
        let err = preflight_data_check(&cfg).unwrap_err().to_string();
        assert!(
            err.contains("数据目录不存在") && err.contains("下一步"),
            "{err}"
        );

        // 类数不匹配：标注最大 id 7（需 8 类）> num_classes 2 → 拦截
        let d = temp_dir("mismatch");
        touch(&d.join("images/train/a.jpg"));
        touch(&d.join("labels/train/a.txt"));
        fs::write(d.join("labels/train/a.txt"), b"7 0.5 0.5 0.1 0.1\n").unwrap();
        let toml = format!(
            "[[model.tasks]]\nkind=\"detect\"\nnum_classes=2\n[data]\npipeline=\"dir\"\n[data.sources.train]\ndir=\"{}\"\n",
            to_toml_path(&d)
        );
        let cfg = RunConfig::from_toml_str(&toml).unwrap();
        let err = preflight_data_check(&cfg).unwrap_err().to_string();
        assert!(
            err.contains("类数不匹配") && err.contains("num_classes 改为 >= 8"),
            "{err}"
        );

        // 类数一致 → 放行
        let toml = toml.replace("num_classes=2", "num_classes=8");
        let cfg = RunConfig::from_toml_str(&toml).unwrap();
        assert!(preflight_data_check(&cfg).is_ok());

        // synthetic 管线不参与预检
        let cfg = RunConfig::from_toml_str(
            "[[model.tasks]]\nkind=\"detect\"\nnum_classes=2\nimg_size=64\n[data]\npipeline=\"synthetic\"\n",
        )
        .unwrap();
        assert!(preflight_data_check(&cfg).is_ok());
    }
}
