//! v0.1 训练 / 推理引擎：合成数据源 + YOLO 目录数据源，分类 / 检测单任务端到端。
//!
//! 范围声明（诚实边界，详见 PLAN）：
//! - 检测数据源：`synthetic`（内置合成）与 `dir`（YOLO images/labels 目录格式）；
//! - 回归损失 v0.1 用单点 cell 分配 + 掩码 L1；TAL / CIoU / DFL 完整版 M2 替换；
//! - AMP / EMA / 多卡 / 真实评测协议按 M2/M7 落地。

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use rayon::prelude::*;
use serde::Serialize;
use tch::nn::OptimizerConfig;
use tch::nn::VarStore;
use tch::{Device, Kind, Tensor};

use av_core::config::{DataPipeline, RunConfig, TaskCfg, TaskKind};
use av_core::error::{AvError, AvResult};
use av_core::geometry::Aabb;
use av_pretrain::av_weight as av_weight_store;
use av_pretrain::weight_adapter::{self, LayerMap};
use av_tasks::augment::{self, AugmentPlan};
use av_tasks::mask::mask_iou;
use av_tasks::models::{
    build_model, KeypointModel, PredictOutput, SegModel, TaskModel, TrainBatch,
};
use av_tasks::rng::XorShift;

use crate::dataset::{self, ClassifySample, KeypointSample, SampleTensor, SegSample};
use crate::eval_map::{CocoEvaluator, GtBox};

const EVAL_BATCH: i64 = 256;
const STEPS_PER_EPOCH: usize = 16;
const CKPT_DIR: &str = "best.ckpt";

/// checkpoint：tch 0.17 的 VarStore::save/load 在 Windows + libtorch 2.4 组合下
/// 存在序列化不兼容（_load_parameters 报 Expected GenericDict but got Object），
/// 故自研目录式 checkpoint：每个变量一个 Tensor::save 文件，按名字对齐写回。
fn save_checkpoint(vs: &VarStore, dir: &Path) -> AvResult<()> {
    fs::create_dir_all(dir)?;
    for (name, t) in vs.variables() {
        let f = dir.join(ckpt_file_name(&name));
        t.save(&f)
            .map_err(|e| AvError::train(format!("保存张量 {name} 失败: {e}")))?;
    }
    Ok(())
}

/// 同 [`save_checkpoint`]，额外写 meta.json 记录 epoch——`--resume` 续训的
/// 恢复点（last.ckpt）依此知道该从哪个 epoch 继续。
fn save_checkpoint_epoch(vs: &VarStore, dir: &Path, epoch: u32) -> AvResult<()> {
    save_checkpoint(vs, dir)?;
    fs::write(
        dir.join("meta.json"),
        format!("{{\"epoch\":{epoch}}}"),
    )
    .map_err(|e| AvError::train(format!("写 checkpoint 元数据失败: {e}")))
}

/// 读 checkpoint 的 epoch 元数据；无 meta.json（旧格式/最终 best.ckpt）返回 None。
fn read_checkpoint_epoch(dir: &Path) -> Option<u32> {
    let raw = fs::read_to_string(dir.join("meta.json")).ok()?;
    let v: serde_json::Value = serde_json::from_str(&raw).ok()?;
    v.get("epoch")?.as_u64().map(|e| e as u32)
}

fn load_checkpoint(vs: &mut VarStore, dir: &Path) -> AvResult<()> {    // 变量是 requires_grad 的叶子，原地写入必须包在 no_grad 里
    tch::no_grad(|| {
        for (name, mut t) in vs.variables() {
            let f = dir.join(ckpt_file_name(&name));
            let loaded =
                Tensor::load(&f).map_err(|e| AvError::train(format!("读取张量 {name} 失败: {e}")))?;
            // 结构变更（如 DFL 头）后旧 checkpoint 与新模型形状不一致时给出可读错误，
            // 而不是让 libtorch 在 copy_ 里 panic
            if loaded.size() != t.size() {
                return Err(AvError::train(format!(
                    "checkpoint 与当前模型结构不匹配: {name} 期望 {:?} 得到 {:?}（模型结构变更后请重训）",
                    t.size(),
                    loaded.size()
                )));
            }
            t.copy_(&loaded);
        }
        Ok(())
    })
}

/// 导出 checkpoint 为单文件格式。safetensors 走 tch 原生序列化（与 legacy
/// 目录格式的 Windows 缺陷无关，已实测往返一致）；"torch" 格式沿用目录式。
/// 从 checkpoint 目录载入变量（CLI 导出/发布流程用）。
pub fn load_checkpoint_dir(vs: &mut VarStore, dir: &Path) -> AvResult<()> {
    load_checkpoint(vs, dir)
}

/// 导出 checkpoint 为单文件格式。safetensors 走 tch 原生序列化（与 legacy
/// 目录格式的 Windows 缺陷无关）；"torch" 格式沿用目录式。
pub fn export_checkpoint(vs: &VarStore, dir: &Path, fmt: &str) -> AvResult<()> {
    match fmt {
        "safetensors" | "st" => {
            if !dir.to_string_lossy().ends_with(".safetensors") {
                return Err(AvError::train(
                    "safetensors 导出的 out 必须以 .safetensors 结尾",
                ));
            }
            let named: Vec<(String, Tensor)> =
                vs.variables().into_iter().collect();
            let refs: Vec<(&str, &Tensor)> =
                named.iter().map(|(n, t)| (n.as_str(), t)).collect();
            tch::Tensor::write_safetensors(&refs, dir)
                .map_err(|e| AvError::train(format!("safetensors 导出失败: {e}")))?;
            Ok(())
        }
        "torch" | "ckpt" => save_checkpoint(vs, dir),
        other => Err(AvError::train(format!("不支持的导出格式: {other}（safetensors | torch）"))),
    }
}

fn ckpt_file_name(name: &str) -> String {
    name.replace(['/', '\\'], "__").replace('.', "_")
}

#[derive(Debug, Clone, Serialize)]
pub struct TrainReport {
    pub run_id: String,
    pub task: String,
    pub epochs: u32,
    pub final_loss: f32,
    pub metric: String,
    pub metric_value: f32,
    pub secondary: Option<(String, f32)>,
    pub run_dir: String,
}

/// 训练入口：加载配置 → 装配模型 → 合成数据训练 → 评测 → 落盘
/// （runs/<run_id>/{best.pt, config.snapshot.toml, report.json}）。
pub fn train(cfg: &RunConfig) -> AvResult<TrainReport> {
    train_impl(cfg, false)
}

/// 续训入口（`av train --resume`）：从 runs/<run_id>/last.ckpt 恢复权重与
/// epoch 继续训练。当前仅 seg 路径接入（last.ckpt 周期存盘所在）；其余任务
/// 返回明确错误。EMA 统计不持久化，恢复后重新累计（日志如实提示）。
pub fn train_resumed(cfg: &RunConfig) -> AvResult<TrainReport> {
    match cfg.model.tasks.first() {
        Some(TaskCfg::Seg(_)) => train_impl(cfg, true),
        _ => Err(AvError::config(
            "--resume 目前仅 seg 训练路径实现（其余任务的周期存盘尚未接入）",
        )),
    }
}

fn train_impl(cfg: &RunConfig, resume: bool) -> AvResult<TrainReport> {
    let run_id = cfg.effective_run_id();
    let run_dir = cfg.output_dir.join(&run_id);
    fs::create_dir_all(&run_dir)?;
    if resume {
        // 续训：保留现有产物（清掉会把待恢复的 best/last 一并带走）
        println!("[resume] 续训模式：保留 runs/{run_id}/ 既有产物");
    } else {
        // 清理历史产物，避免旧权重残留
        let _ = fs::remove_dir_all(run_dir.join(CKPT_DIR));
        // metrics.jsonl 同理：run id 复用时从空文件开始，避免新旧 epoch 混流
        let _ = fs::remove_file(run_dir.join(crate::metrics::METRICS_FILE));
    }
    fs::write(run_dir.join("config.snapshot.toml"), cfg.snapshot_toml()?)?;

    let report = match cfg.model.tasks.first() {
        Some(TaskCfg::Classify(_)) => train_classify(cfg, &run_id, &run_dir)?,
        Some(TaskCfg::Detect(d)) if !d.obb_mode => match cfg.data.pipeline {
            DataPipeline::Synthetic => train_detect_synthetic(cfg, &run_id, &run_dir)?,
            // dir（YOLO 目录格式）与 avpack（.avpack 容器）共用同一训练循环
            DataPipeline::Dir | DataPipeline::AvPack => train_detect_yolo(cfg, &run_id, &run_dir)?,
        },
        // OBB（obb_mode=true）：真实数据走 DOTA 格式（dota8 已验证）
        Some(TaskCfg::Detect(d)) if d.obb_mode => match cfg.data.pipeline {
            DataPipeline::Dir => train_detect_obb(cfg, &run_id, &run_dir)?,
            _ => {
                return Err(AvError::config(
                    "OBB 训练需要 data.pipeline = \"dir\"（DOTA 格式，见 configs/detect_obb_dota8.toml）",
                ))
            }
        },
        // 实例分割（PLAN §4.3）：COCO 分割格式（coco8-seg 已验证）
        Some(TaskCfg::Seg(_)) => match cfg.data.pipeline {
            DataPipeline::Dir => train_seg(cfg, &run_id, &run_dir, resume)?,
            _ => {
                return Err(AvError::config(
                    "Seg 训练需要 data.pipeline = \"dir\"（COCO 分割格式，见 configs/seg_coco8.toml）",
                ))
            }
        },
        // 关键点（PLAN §4.4）：COCO 姿态格式（coco8-pose 已验证）
        Some(TaskCfg::Keypoint(_)) => match cfg.data.pipeline {
            DataPipeline::Dir => train_keypoint(cfg, &run_id, &run_dir)?,
            _ => {
                return Err(AvError::config(
                    "Keypoint 训练需要 data.pipeline = \"dir\"（COCO 姿态格式，见 configs/keypoint_coco8.toml）",
                ))
            }
        },
        _ => {
            return Err(AvError::config(
                "该任务类型在 v0.1 引擎未支持（见 PLAN §8 里程碑）",
            ))
        }
    };
    fs::write(
        run_dir.join("report.json"),
        serde_json::to_string_pretty(&report)
            .map_err(|e| AvError::train(format!("报告序列化失败: {e}")))?,
    )?;
    Ok(report)
}

/// 推理入口：从 run 目录快照重建模型 → 读权重 → 图片 → JSON 结果。
/// 检测产物坐标经 letterbox 逆映射还原回原图（PLAN §5.1）。
pub fn infer(cfg: &RunConfig, weights: &Path, input: &Path) -> AvResult<serde_json::Value> {
    let (model, device) = load_model(cfg, weights)?;
    let (x, lb, orig_w, orig_h) = dataset::decode_image_with_meta(
        input,
        model.img_size(),
        Device::Cpu,
        dataset::ResizeMode::Letterbox,
        imagenet_norm(cfg),
    )?;
    let x = x.to_device(device).unsqueeze(0);
    match model.predict(&x, 0.25, 0.5)? {
        PredictOutput::Classify { labels, confs } => {
            let preds: Vec<serde_json::Value> = labels
                .iter()
                .zip(&confs)
                .map(|(c, p)| serde_json::json!({ "class_id": c, "prob": p }))
                .collect();
            Ok(serde_json::json!({ "task": "classify", "predictions": preds }))
        }
        PredictOutput::Detect { per_image } => {
            let mut dets = per_image.first().cloned().unwrap_or_default();
            if let Some(lb) = lb {
                for d in &mut dets {
                    d.bbox = lb.restore_box(d.bbox, orig_w, orig_h);
                    if let Some(kps) = &mut d.keypoints {
                        for kp in kps.iter_mut() {
                            kp[0] = (kp[0] - lb.pad_left) / lb.scale;
                            kp[1] = (kp[1] - lb.pad_top) / lb.scale;
                        }
                    }
                }
            }
            Ok(serde_json::json!({
                "task": "detect",
                "image": { "width": orig_w, "height": orig_h },
                "detections": dets,
            }))
        }
        PredictOutput::Seg { per_image } => {
            // 掩码本身不进 JSON（80×80/实例过大），给类别/分数 + 掩码外接框 + 覆盖率
            let insts = per_image.first().cloned().unwrap_or_default();
            let mask_size = model.img_size() / 4;
            let mut preds = Vec::new();
            for it in insts {
                let (mut x0, mut y0, mut x1, mut y1, mut area) =
                    (i64::MAX, i64::MAX, 0i64, 0i64, 0usize);
                for (pi, &v) in it.mask.iter().enumerate() {
                    if v == 1 {
                        let (py, px) = ((pi / mask_size as usize) as i64, (pi % mask_size as usize) as i64);
                        x0 = x0.min(px);
                        y0 = y0.min(py);
                        x1 = x1.max(px);
                        y1 = y1.max(py);
                        area += 1;
                    }
                }
                preds.push(serde_json::json!({
                    "class_id": it.label,
                    "score": it.score,
                    "mask_bbox": [x0, y0, x1, y1],
                    "mask_area": area,
                }));
            }
            Ok(serde_json::json!({ "task": "seg", "predictions": preds }))
        }
        PredictOutput::Keypoint { per_image } => {
            // 关键点实例（Detection 载体：bbox + keypoints），画布坐标经 letterbox
            // 逆映射还原回原图（框与关键点同一变换，Detection::restore 统一处理）
            let mut dets = per_image.first().cloned().unwrap_or_default();
            if let Some(lb) = lb {
                for d in &mut dets {
                    d.restore(&lb);
                }
            }
            Ok(serde_json::json!({
                "task": "keypoint",
                "image": { "width": orig_w, "height": orig_h },
                "detections": dets,
            }))
        }
    }
}

/// 切片推理（SAHI 式，PLAN §5.1/§7.4）：高分辨率大图（工业相机 5MP~25MP+）保持
/// 原始分辨率滑窗逐块检测，窗口坐标映射回全图，最后全局 NMS 合并——
/// 小缺陷不会因为整图缩放而消失。
pub fn infer_sliced(
    cfg: &RunConfig,
    weights: &Path,
    input: &Path,
    window: u32,
    overlap_frac: f32,
) -> AvResult<serde_json::Value> {
    let (model, device) = load_model(cfg, weights)?;
    let img =
        image::open(input).map_err(|e| AvError::data(format!("读图失败 {input:?}: {e}")))?;
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    let s_model = model.img_size();
    let (tw, th) = (window.min(w).max(1), window.min(h).max(1));
    let stride = (((1.0 - overlap_frac.clamp(0.0, 0.8)) * tw as f32).round() as u32).max(1);

    // 滑窗起点：步进扫过，末尾强制补一个贴边窗口保证全覆盖
    fn starts(total: u32, win: u32, stride: u32) -> Vec<u32> {
        if total <= win {
            return vec![0];
        }
        let mut v = Vec::new();
        let mut p = 0u32;
        while p + win < total {
            v.push(p);
            p += stride;
        }
        v.push(total - win);
        v
    }
    let y_starts = starts(h, th, stride);
    let x_starts = starts(w, tw, stride);

    // 解码全部窗口（CPU），按批推理，坐标映射回原图
    let mut tiles: Vec<(u32, u32, Tensor, Option<av_core::geometry::Letterbox>)> = Vec::new();
    for &y0 in &y_starts {
        for &x0 in &x_starts {
            let crop = image::imageops::crop_imm(&rgb, x0, y0, tw, th).to_image();
            let (x, lb) = dataset::decode_rgb_with_meta(
                &crop,
                s_model,
                Device::Cpu,
                dataset::ResizeMode::Letterbox,
                imagenet_norm(cfg),
            )?;
            tiles.push((x0, y0, x, lb));
        }
    }

    let mut all: Vec<av_core::types::Detection> = Vec::new();
    for batch in tiles.chunks(8) {
        let xs: Vec<Tensor> = batch.iter().map(|(_, _, x, _)| x.copy()).collect();
        let x = Tensor::stack(&xs, 0).to_device(device);
        let per = model.predict(&x, 0.25, 0.5)?;
        let per_image = match per {
            PredictOutput::Detect { per_image } => per_image,
            PredictOutput::Classify { .. } => continue,
            // 切片推理只服务检测（分割掩码/关键点骨架跨片拼接另行设计）
            PredictOutput::Seg { .. } => continue,
            PredictOutput::Keypoint { .. } => continue,
        };
        for (bi, dets) in per_image.into_iter().enumerate() {
            let (x0, y0, _, lb) = &batch[bi];
            for mut d in dets {
                match lb {
                    // letterbox 窗：先逆变换回窗口坐标，再加窗口偏移
                    Some(lb) => {
                        d.bbox.x1 = (d.bbox.x1 - lb.pad_left) / lb.scale + *x0 as f32;
                        d.bbox.y1 = (d.bbox.y1 - lb.pad_top) / lb.scale + *y0 as f32;
                        d.bbox.x2 = (d.bbox.x2 - lb.pad_left) / lb.scale + *x0 as f32;
                        d.bbox.y2 = (d.bbox.y2 - lb.pad_top) / lb.scale + *y0 as f32;
                        if let Some(kps) = &mut d.keypoints {
                            for kp in kps.iter_mut() {
                                kp[0] = (kp[0] - lb.pad_left) / lb.scale + *x0 as f32;
                                kp[1] = (kp[1] - lb.pad_top) / lb.scale + *y0 as f32;
                            }
                        }
                    }
                    // stretch 窗：模型坐标 × (窗宽/模型输入) + 偏移
                    None => {
                        let sx = *x0 as f32;
                        let sy = *y0 as f32;
                        d.bbox.x1 += sx;
                        d.bbox.y1 += sy;
                        d.bbox.x2 += sx;
                        d.bbox.y2 += sy;
                    }
                }
                // 裁回图内
                d.bbox.x1 = d.bbox.x1.clamp(0.0, w as f32);
                d.bbox.y1 = d.bbox.y1.clamp(0.0, h as f32);
                d.bbox.x2 = d.bbox.x2.clamp(0.0, w as f32);
                d.bbox.y2 = d.bbox.y2.clamp(0.0, h as f32);
                if d.bbox.x2 > d.bbox.x1 && d.bbox.y2 > d.bbox.y1 {
                    all.push(d);
                }
            }
        }
    }
    // 碎片合并（跨窗同目标伪影抑制：中心距阈值取窗口短边 1/4）→ 全局 NMS
    let max_center_dist = 0.25 * tw.min(th) as f32;
    let all = merge_tile_fragments(all, 0.3, max_center_dist);
    let all = av_core::types::nms(all, 0.5);
    let tiles_cnt = (y_starts.len() * x_starts.len()) as u32;
    Ok(serde_json::json!({
        "task": "detect",
        "mode": "sliced",
        "image": { "width": w, "height": h },
        "tiles": tiles_cnt,
        "detections": all,
    }))
}

/// 训练循环 epoch 末的指标落盘钩子（面板实时化）：每个 epoch 向
/// `run_dir/metrics.jsonl` 追加一行 JSON，面板经 SSE 增量读取画实时曲线。
/// 落盘失败只告警不中断——观测通道永远不能杀死训练。
fn log_epoch_metrics(
    run_dir: &Path,
    epoch: u32,
    loss: f32,
    metric: &str,
    value: f32,
    secondary: Option<(&str, f32)>,
) {
    log_epoch_metrics_with_extra(run_dir, epoch, loss, metric, value, secondary, None);
}

/// 同 [`log_epoch_metrics`]，附加第三个可选指标（seg 用它落 P@0.5）。
/// `extra` 写为 `"extra": [名字, 数值]`——面板只读 metric/secondary，
/// 未知字段自然忽略，历史行不受影响。
fn log_epoch_metrics_with_extra(
    run_dir: &Path,
    epoch: u32,
    loss: f32,
    metric: &str,
    value: f32,
    secondary: Option<(&str, f32)>,
    extra: Option<(&str, f32)>,
) {
    let row = serde_json::json!({
        "epoch": epoch,
        "loss": loss,
        "metric": metric,
        "metric_value": value,
        "secondary": secondary.map(|(n, v)| serde_json::json!([n, v])),
        "extra": extra.map(|(n, v)| serde_json::json!([n, v])),
        "ts": crate::metrics::now_unix(),
    });
    if let Err(e) = crate::metrics::append(run_dir, &row) {
        tracing::warn!("metrics.jsonl 追加失败（忽略）: {e}");
    }
}

/// 切片碎片合并（SAHI 伪影抑制）：高分辨率滑窗推理时同一目标被相邻窗口
/// 各检一次，产生高重叠、但 IoU 低于 NMS 阈值的「碎片框」。这里在全局 NMS
/// 之前把满足「同类别 + IoU ≥ iou_thr + 中心距 < max_center_dist」的碎片
/// 合并进分数最高的代表框（框取并集外扩矩形，分数/类别/角度/关键点沿用代表）。
///
/// 贪心单趟：按分数降序扫描，每个检测要么并入已有簇（与簇代表框比较），
/// 要么自成一簇——复杂度 O(n²)，与既有 nms 同量级。
pub fn merge_tile_fragments(
    mut dets: Vec<av_core::types::Detection>,
    iou_thr: f32,
    max_center_dist: f32,
) -> Vec<av_core::types::Detection> {
    dets.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut out: Vec<av_core::types::Detection> = Vec::new();
    for d in dets {
        let (dcx, dcy) = (
            (d.bbox.x1 + d.bbox.x2) * 0.5,
            (d.bbox.y1 + d.bbox.y2) * 0.5,
        );
        let mut merged = false;
        for k in out.iter_mut() {
            if k.class_id != d.class_id {
                continue;
            }
            let (kcx, kcy) = ((k.bbox.x1 + k.bbox.x2) * 0.5, (k.bbox.y1 + k.bbox.y2) * 0.5);
            let dist = ((kcx - dcx).powi(2) + (kcy - dcy).powi(2)).sqrt();
            if k.bbox.iou(&d.bbox) >= iou_thr && dist < max_center_dist {
                // 并集框：碎片各自只看到目标一部分，并集才是完整目标
                k.bbox.x1 = k.bbox.x1.min(d.bbox.x1);
                k.bbox.y1 = k.bbox.y1.min(d.bbox.y1);
                k.bbox.x2 = k.bbox.x2.max(d.bbox.x2);
                k.bbox.y2 = k.bbox.y2.max(d.bbox.y2);
                merged = true;
                break;
            }
        }
        if !merged {
            out.push(d);
        }
    }
    out
}

/// 实例分割（PLAN §4.3）真实数据训练：COCO 分割格式（coco8-seg）+ 掩码 BCE/Dice。
///
/// 验收指标取**训练集**掩码 mIoU（任务规范：真实分割从零小模型 200 epochs 的
/// 诚实起点）；val split 每 interval 评测仅作泛化观察。
///
/// 数据管线 v2（`[data].cache`）：增强训练路径默认把 raw 全分辨率数据一次性
/// 缓存成 letterbox 内容贴片，逐 epoch 只在小图上增强——`ram`（rayon 并行 +
/// 双缓冲预取）或 `gpu`（显存驻留 + GPU 张量增强）；`auto` 按体积估算选层；
/// `off` 保留历史全分辨率逐 epoch 重编码路径（单测与验收口径均逐位/容差对齐）。

/// auto 模式下显存驻留的体积预算：画布堆 ≤ 4GiB 才选 gpu（16GB 卡留足
/// 模型/激活/工作区余量；更小显存的卡自动落 ram，行为不劣化）。
const GPU_CACHE_BUDGET_BYTES: u64 = 4 * 1024 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SegCacheMode {
    Gpu,
    Ram,
    Off,
}

/// 解析 `[data].cache`：auto 按画布堆体积估算（仅 CUDA 设备可选 gpu），
/// 显式值非法时告警回落 auto（配置校验已拦，此处兜底）。
fn resolve_seg_cache_mode(cfg_value: &str, n: usize, img_size: u32, device: Device) -> SegCacheMode {
    let bytes = n as u64 * 3 * img_size as u64 * img_size as u64 * 4;
    let is_cuda = matches!(device, Device::Cuda(_));
    match cfg_value {
        "gpu" if is_cuda => SegCacheMode::Gpu,
        "gpu" => {
            tracing::warn!("data.cache = \"gpu\" 但设备无 CUDA，回落 ram");
            SegCacheMode::Ram
        }
        "ram" => SegCacheMode::Ram,
        "off" => SegCacheMode::Off,
        "auto" | _ => {
            if is_cuda && bytes <= GPU_CACHE_BUDGET_BYTES {
                SegCacheMode::Gpu
            } else {
                SegCacheMode::Ram
            }
        }
    }
}

/// tch 的 Tensor 含 `*mut C_tensor`、未实现 `Sync`（保守声明）；ATen 张量本身
/// 原子引用计数、并发只读安全。显存驻留画布堆在主线程写、多线程并发读
/// （select/copy 派生视图），用此封装声明该安全性契约。
struct SyncTensor(tch::Tensor);
// SAFETY: 见结构体注释；Tensor: Send 已由 tch 保证，此处仅补 Sync（并发只读）。
unsafe impl Send for SyncTensor {}
unsafe impl Sync for SyncTensor {}

/// seg 增强路径编码器：封装 off/ram/gpu 三种数据管线，统一 `encode(idx, plans)`。
/// 全部实现共享同一份「预抽取 plan 序列」，执行顺序不影响结果可复现性。
enum SegEncoder {
    /// 历史路径：raw 全分辨率逐 epoch 重编码（单线程语义不变）
    Off {
        raw: std::sync::Arc<Vec<dataset::RawSegSample>>,
        img_size: u32,
        inorm: bool,
    },
    /// letterbox 内容贴片缓存 + rayon 并行编码（CPU 增强最高 40 倍提速）
    Ram {
        cache: std::sync::Arc<Vec<dataset::CachedSegSample>>,
        img_size: u32,
        inorm: bool,
    },
    /// 显存驻留画布堆 + GPU 张量增强（CPU 每 epoch 零参与，掩码除外）
    Gpu {
        stack: Arc<SyncTensor>,
        meta: Arc<Vec<dataset::CachedSegSample>>,
        img_size: u32,
        inorm: bool,
    },
}

impl SegEncoder {
    fn encode(&self, idx: &[usize], plans: &[AugmentPlan]) -> AvResult<Vec<SegSample>> {
        match self {
            SegEncoder::Off { raw, img_size, inorm } => idx
                .par_iter()
                .zip(plans)
                .map(|(&i, plan)| {
                    dataset::encode_seg_sample(&raw[i], *img_size, Device::Cpu, plan, *inorm)
                })
                .collect(),
            SegEncoder::Ram { cache, img_size, inorm } => {
                dataset::encode_seg_batch_cached(cache, idx, plans, *img_size, Device::Cpu, *inorm)
            }
            SegEncoder::Gpu { stack, meta, img_size, inorm } => idx
                .par_iter()
                .zip(plans)
                .map(|(&i, plan)| {
                    dataset::encode_seg_sample_gpu(&stack.0, i as u32, &meta[i], *img_size, plan, *inorm)
                })
                .collect(),
        }
    }

    fn describe(&self, n: usize) -> String {
        match self {
            SegEncoder::Off { .. } => "off（历史全分辨率路径）".into(),
            SegEncoder::Ram { .. } => {
                format!("ram（{n} 张内容贴片缓存 + rayon 并行 + 双缓冲预取）")
            }
            SegEncoder::Gpu { .. } => {
                format!("gpu（{n} 张画布显存驻留 + GPU 张量增强）")
            }
        }
    }
}

fn train_seg(cfg: &RunConfig, run_id: &str, run_dir: &Path, resume: bool) -> AvResult<TrainReport> {
    let (num_classes, img_size) = match cfg.model.tasks.first() {
        Some(TaskCfg::Seg(s)) => (s.num_classes as u32, s.img_size),
        _ => unreachable!("train_seg 只处理分割任务"),
    };
    let device = resolve_device(cfg);
    let root = cfg
        .data
        .sources
        .train
        .dir
        .clone()
        .ok_or_else(|| AvError::config("dir 数据源缺 data.sources.train.dir"))?;
    let split = cfg.data.sources.train.split.as_deref().unwrap_or("train");
    let aug_cfg = train_augment_cfg(cfg, TaskKind::Seg);
    let mut train_raw = if let Some(a) = &aug_cfg {
        println!(
            "[augment] seg 训练增强生效: flip={} hsv={:?} scale_jitter={:?} close_last_epochs={}",
            a.flip, a.hsv, a.scale_jitter, a.close_last_epochs
        );
        Some(dataset::load_cocoseg_dir_raw(&root, split)?)
    } else {
        None
    };
    let train = if train_raw.is_some() {
        Vec::new()
    } else {
        dataset::load_cocoseg_dir(&root, split, img_size, Device::Cpu, imagenet_norm(cfg))?
    };
    let val = match cfg.data.sources.val.dir.as_ref() {
        Some(d) => dataset::load_cocoseg_dir(
            d,
            cfg.data.sources.val.split.as_deref().unwrap_or("val"),
            img_size,
            Device::Cpu,
            imagenet_norm(cfg),
        )?,
        None => dataset::load_cocoseg_dir(&root, "val", img_size, Device::Cpu, imagenet_norm(cfg))?,
    };
    let train_n = train_raw.as_ref().map_or(train.len(), |r| r.len());
    let train_insts: usize = match &train_raw {
        Some(raw) => raw.iter().map(|s| s.polys.len()).sum(),
        None => train.iter().map(|s| s.masks.len()).sum(),
    };
    println!(
        "[seg] root={} train={}图/{}实例 val={}图/{}实例 classes={} img_size={} mask画布={}×{}",
        root.display(),
        train_n,
        train_insts,
        val.len(),
        val.iter().map(|s| s.masks.len()).sum::<usize>(),
        num_classes,
        img_size,
        img_size / 4,
        img_size / 4
    );

    // 数据管线 v2：增强路径的缓存层构建（一次性；构建后释放 raw 全分辨率
    // 像素，内存占用从 ~15MB/图 降到 ~1MB/图）。off 保留历史路径零变化。
    let seg_encoder: Option<Arc<SegEncoder>> = match (&train_raw, &aug_cfg) {
        (Some(raw), Some(_)) => {
            let mode = resolve_seg_cache_mode(&cfg.data.cache, raw.len(), img_size, device);
            match mode {
                SegCacheMode::Off => Some(Arc::new(SegEncoder::Off {
                    raw: Arc::new(std::mem::take(&mut train_raw).unwrap()),
                    img_size,
                    inorm: imagenet_norm(cfg),
                })),
                SegCacheMode::Ram => {
                    let (cache, bytes) = dataset::build_seg_cache(raw, img_size)?;
                    println!(
                        "[seg] 数据缓存: ram（{} 张内容贴片，{:.0}MB；rayon 并行 + 双缓冲预取）",
                        cache.len(),
                        bytes as f64 / 1e6
                    );
                    drop(std::mem::take(&mut train_raw));
                    Some(Arc::new(SegEncoder::Ram {
                        cache: Arc::new(cache),
                        img_size,
                        inorm: imagenet_norm(cfg),
                    }))
                }
                SegCacheMode::Gpu => {
                    let (cache, bytes) = dataset::build_seg_cache(raw, img_size)?;
                    let stack = {
                        // 显存不足（OOM）自动降级 ram，训练不中断
                        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            dataset::build_seg_canvas_stack(&cache, img_size, device)
                        }));
                        match res {
                            Ok(Ok(t)) => Some(t),
                            Ok(Err(e)) => {
                                tracing::warn!("显存驻留上传失败（{e}），降级 ram 缓存");
                                None
                            }
                            Err(_) => {
                                tracing::warn!("显存驻留上传 OOM，降级 ram 缓存");
                                None
                            }
                        }
                    };
                    match stack {
                        Some(t) => {
                            println!(
                                "[seg] 数据缓存: gpu（{} 张画布显存驻留 {:.0}MB；GPU 张量增强）",
                                cache.len(),
                                bytes as f64 * 4.0 / 1e6
                            );
                            drop(std::mem::take(&mut train_raw));
                            Some(Arc::new(SegEncoder::Gpu {
                                stack: Arc::new(SyncTensor(t)),
                                meta: Arc::new(cache),
                                img_size,
                                inorm: imagenet_norm(cfg),
                            }))
                        }
                        None => {
                            println!(
                                "[seg] 数据缓存: ram（降级；{} 张内容贴片，{:.0}MB）",
                                cache.len(),
                                bytes as f64 / 1e6
                            );
                            drop(std::mem::take(&mut train_raw));
                            Some(Arc::new(SegEncoder::Ram {
                                cache: Arc::new(cache),
                                img_size,
                                inorm: imagenet_norm(cfg),
                            }))
                        }
                    }
                }
            }
        }
        _ => None,
    };
    let encoder_desc = seg_encoder
        .as_ref()
        .map(|e| e.describe(train_n))
        .unwrap_or_else(|| "off（无增强，整集预解码）".into());
    if seg_encoder.is_some() {
        println!("[seg] 数据管线: {encoder_desc}");
    }

    let mut vs = VarStore::new(device);
    let model = build_model(&vs.root(), cfg)?;
    apply_pretrain(&vs, cfg)?;
    // --resume：last.ckpt（权重 + epoch 元数据）恢复到 VarStore，从下一 epoch 续训。
    // 放在 apply_pretrain 之后（恢复覆盖预训练初始化）、make_opt 之前（优化器
    // 绑定已恢复的张量）。EMA 统计不持久化，恢复后从当前权重重新累计。
    let mut start_epoch = 1u32;
    if resume {
        let last = run_dir.join("last.ckpt");
        match read_checkpoint_epoch(&last) {
            Some(done) => {
                load_checkpoint(&mut vs, &last)?;
                start_epoch = done + 1;
                println!(
                    "[resume] 已从 last.ckpt 恢复（完成 {done} epoch），从 epoch {start_epoch} 续训；EMA 重新累计"
                );
            }
            None => println!("[resume] 未找到 last.ckpt（runs/{run_id}/），从头训练"),
        }
    }
    let mut opt = make_opt(&vs, cfg)?;
    let mut rng = XorShift::new(cfg.seed);
    let bs = (cfg.train.batch_size as usize).max(1);
    let mut final_loss = 0f32;
    #[allow(unused_assignments)] // 初值仅占位：epochs>=1 时 eval_due 必在首次读取前赋值
    let (mut val_miou, mut val_r50, mut val_p50) = (0f32, 0f32, 0f32);
    // BN train/eval 装配（resnet18 骨干生效；冻结骨干时恒 eval）
    let bn_train = bn_train_mode(cfg);

    for epoch in start_epoch..=cfg.train.epochs {
        model.set_train(bn_train);
        opt.set_lr(schedule_lr(cfg, epoch));
        let mut order: Vec<usize> = (0..train_n).collect();
        for i in (1..order.len()).rev() {
            let j = rng.next_usize(i + 1);
            order.swap(i, j);
        }
        let strong_on = aug_cfg
            .as_ref()
            .map(|a| strong_aug_on(a, cfg, epoch))
            .unwrap_or(false);
        let mut aug_rng = epoch_aug_rng(cfg.seed, epoch);
        let mut epoch_loss = 0f32;
        let mut steps = 0usize;
        let acc_steps = cfg.train.accumulate_steps.max(1) as usize;
        opt.zero_grad();
        // 本 epoch 的增强方案按 shuffle 序在主线程一次抽完（与旧实现的逐样本
        // 抽签 RNG 消耗序逐位一致）——并行/预取只改执行顺序，不改随机流。
        let strong_plans: Vec<AugmentPlan> = match (&seg_encoder, &aug_cfg) {
            (Some(_), Some(a)) if strong_on => (0..order.len())
                .map(|_| augment::draw_plan(a, &mut aug_rng))
                .collect(),
            _ => Vec::new(),
        };
        let chunk_starts: Vec<usize> = (0..order.len()).step_by(bs).collect();
        let mut in_flight: Option<std::thread::JoinHandle<AvResult<Vec<SegSample>>>> = None;
        for (ci, &start) in chunk_starts.iter().enumerate() {
            let end = (start + bs).min(order.len());
            // 当前块：优先收上一轮预热线程的货，否则同步编码（仅首个块）
            let batch_samples: Vec<SegSample> = match in_flight.take() {
                Some(h) => h.join().map_err(|_| AvError::train("数据编码线程崩溃"))??,
                None => {
                    let idx = &order[start..end];
                    let plans: Vec<AugmentPlan> = if strong_on {
                        strong_plans[start..end].to_vec()
                    } else {
                        vec![AugmentPlan::none(); end - start]
                    };
                    match &seg_encoder {
                        Some(enc) => enc.encode(idx, &plans)?,
                        None => order[start..end]
                            .iter()
                            .map(|&i| Ok(train[i].clone()))
                            .collect::<AvResult<Vec<_>>>()?,
                    }
                }
            };
            // 预热下一块：与本次 GPU 计算重叠（CPU 编码不再是 GPU 的等待时间）
            if let (Some(enc), Some(&next_start)) = (&seg_encoder, chunk_starts.get(ci + 1)) {
                let next_end = (next_start + bs).min(order.len());
                let idx = order[next_start..next_end].to_vec();
                let plans: Vec<AugmentPlan> = if strong_on {
                    strong_plans[next_start..next_end].to_vec()
                } else {
                    vec![AugmentPlan::none(); next_end - next_start]
                };
                let enc = Arc::clone(enc);
                in_flight = Some(std::thread::spawn(move || enc.encode(&idx, &plans)));
            }
            let x = dataset::stack_seg_samples(&batch_samples)?.to_device(device);
            let batch = TrainBatch::Seg {
                masks: batch_samples.iter().map(|s| s.masks.clone()).collect(),
                labels: batch_samples.iter().map(|s| s.labels.clone()).collect(),
            };
            let loss = model.loss(&x, &batch)?;
            loss.backward();
            epoch_loss += loss.double_value(&[]) as f32;
            steps += 1;
            if steps % acc_steps == 0 {
                if cfg.train.grad_clip > 0.0 {
                    opt.clip_grad_norm(cfg.train.grad_clip as f64);
                }
                opt.step();
                opt.zero_grad();
            }
        }
        if steps % acc_steps != 0 {
            if cfg.train.grad_clip > 0.0 {
                opt.clip_grad_norm(cfg.train.grad_clip as f64);
            }
            opt.step();
            opt.zero_grad();
        }
        final_loss = epoch_loss / steps.max(1) as f32;
        let eval_due = epoch == 1
            || epoch == cfg.train.epochs
            || cfg.eval.interval_epochs == 0
            || epoch % cfg.eval.interval_epochs == 0;
        if eval_due {
            model.set_train(false); // 评测 = BN 推理语义（running 统计量）
            let TaskModel::Seg(m) = &model else {
                unreachable!("分割任务模型类型")
            };
            (val_miou, val_r50, val_p50, _, _) = eval_seg_samples(m, &val, device)?;
            model.set_train(bn_train); // 恢复训练态
            println!(
                "[seg] run={run_id} epoch={epoch}/{} loss={final_loss:.4} val掩码mIoU={val_miou:.3} val R@0.5={val_r50:.3} val P@0.5={val_p50:.3}",
                cfg.train.epochs
            );
            // 周期性存盘（可靠性）：seg 训练此前只在结束时写盘，中途异常退出
            // 全部丢失。last.ckpt 每 interval 覆盖更新（含 epoch 元数据，供
            // --resume 续训），失败不中断训练。
            match save_checkpoint_epoch(&vs, &run_dir.join("last.ckpt"), epoch) {
                Ok(()) => println!("[ckpt] last.ckpt 已更新（epoch {epoch}）"),
                Err(e) => tracing::warn!("last.ckpt 保存失败（忽略，不影响训练）: {e}"),
            }
        }
        log_epoch_metrics_with_extra(
            run_dir,
            epoch,
            final_loss,
            "mask_miou",
            val_miou,
            Some(("recall@0.5", val_r50)),
            Some(("precision@0.5", val_p50)),
        );
    }

    // 验收口径：训练集掩码 mIoU / R@0.5（8 图小数据的诚实过拟合基线）
    // 增强路径下训练样本为 raw，用 none() plan 重编码出与不增强逐位一致的验收集
    model.set_train(false); // 验收评测 = BN 推理语义
    let TaskModel::Seg(m) = &model else {
        unreachable!("分割任务模型类型")
    };
    let (train_miou, train_r50, train_p50, n_inst, _) = match &seg_encoder {
        Some(enc) => {
            // 增强路径下训练样本走编码器重编码（none plan：s=1 无增益，
            // 缓存路径与全分辨率路径逐位一致），得到「未增强视角」验收集
            let idx: Vec<usize> = (0..train_n).collect();
            let plans = vec![AugmentPlan::none(); train_n];
            let clean = enc.encode(&idx, &plans)?;
            eval_seg_samples(m, &clean, device)?
        }
        None => eval_seg_samples(m, &train, device)?,
    };

    save_checkpoint(&vs, &run_dir.join(CKPT_DIR))?;
    Ok(TrainReport {
        run_id: run_id.to_string(),
        task: "seg".into(),
        epochs: cfg.train.epochs,
        final_loss,
        metric: "train_mask_miou".into(),
        metric_value: train_miou,
        secondary: Some(("val_mask_miou".into(), val_miou)),
        run_dir: run_dir.display().to_string(),
    })
    .map(|r| {
        println!(
            "[seg] 训练集验收：掩码 mIoU={train_miou:.3} R@0.5={train_r50:.3} P@0.5={train_p50:.3}（{n_inst} 个 gt 实例）\
             | val mIoU={val_miou:.3} val R@0.5={val_r50:.3} val P@0.5={val_p50:.3}",
        );
        r
    })
}

/// 分割样本集评测：每 gt 实例取类无关最优掩码 IoU 的均值（掩码 mIoU）+
/// 类别正确且 IoU ≥ 0.5 的实例占比（R@0.5）+ 预测侧查准率 P@0.5
/// （按分数降序贪心一对一匹配：预测配对最佳未占用同类 gt，IoU ≥ 0.5 计 TP；
/// P@0.5 = TP / 预测总数，无预测时为 0）。
/// 返回 (miou, r50, p50, gt 实例数, 分类别 (类 id, mIoU, gt 数)，类 id 升序)。
fn eval_seg_samples(
    m: &SegModel,
    val: &[SegSample],
    device: Device,
) -> AvResult<(f32, f32, f32, usize, Vec<(u32, f32, usize)>)> {
    if val.is_empty() {
        return Err(AvError::data("验证集为空"));
    }
    let mut per_image: Vec<Vec<av_tasks::models::SegInstance>> = Vec::new();
    for chunk in val.chunks(4) {
        let x = dataset::stack_seg_samples(chunk)?.to_device(device);
        per_image.extend(m.predict(&x, 0.1, 0.5)?);
    }
    let mut sum_best = 0f32;
    let mut n_gt = 0usize;
    let mut hits = 0usize;
    let mut tp = 0usize;
    let mut n_pred = 0usize;
    let mut per_class_acc: std::collections::BTreeMap<u32, (f64, usize)> =
        std::collections::BTreeMap::new();
    for (gi, s) in val.iter().enumerate() {
        let preds = &per_image[gi];
        n_gt += s.masks.len();
        n_pred += preds.len();
        // 单图 IoU 矩阵（pred × gt）：mIoU / R@0.5 / P@0.5 三指标共用一次计算
        let ious: Vec<Vec<f32>> = preds
            .iter()
            .map(|d| s.masks.iter().map(|gt| mask_iou(gt, &d.mask)).collect())
            .collect();
        for (g, _gt_mask) in s.masks.iter().enumerate() {
            let label = s.labels[g];
            let best = ious
                .iter()
                .map(|row| row[g])
                .fold(0f32, f32::max);
            sum_best += best;
            let acc = per_class_acc.entry(label).or_insert((0.0, 0));
            acc.0 += best as f64;
            acc.1 += 1;
            if preds
                .iter()
                .enumerate()
                .any(|(pi, d)| d.label == label && ious[pi][g] >= 0.5)
            {
                hits += 1;
            }
        }
        tp += greedy_mask_match(
            preds.iter().map(|d| (d.label, d.score)).collect(),
            &s.labels,
            &ious,
            0.5,
        );
    }
    if n_gt == 0 {
        return Err(AvError::data("验证集无 gt 实例"));
    }
    let p50 = if n_pred == 0 { 0.0 } else { tp as f32 / n_pred as f32 };
    let per_class = per_class_acc
        .into_iter()
        .map(|(cls, (sum, n))| (cls, (sum / n as f64) as f32, n))
        .collect();
    Ok((
        sum_best / n_gt as f32,
        hits as f32 / n_gt as f32,
        p50,
        n_gt,
        per_class,
    ))
}

/// 单图 P@0.5 的贪心一对一匹配（PASCAL/COCO 惯例）：预测按分数降序，依次
/// 配对「同类且 IoU ≥ thr 的最佳未占用 gt」，每个 gt 至多被一个预测占用。
/// `ious[pi][g]` = 预测 pi 与 gt g 的掩码 IoU。返回 TP 数。
/// 纯函数（不依赖 libtorch），单测覆盖阈值/同类/一对一/分数优先四个要点。
fn greedy_mask_match(
    preds: Vec<(u32, f32)>,
    gt_labels: &[u32],
    ious: &[Vec<f32>],
    thr: f32,
) -> usize {
    let mut order: Vec<usize> = (0..preds.len()).collect();
    order.sort_by(|&a, &b| preds[b].1.total_cmp(&preds[a].1));
    let mut used = vec![false; gt_labels.len()];
    let mut tp = 0usize;
    for pi in order {
        let mut best_g = None;
        let mut best_v = thr;
        for (g, &used_g) in used.iter().enumerate() {
            if used_g || preds[pi].0 != gt_labels[g] {
                continue;
            }
            let v = ious[pi][g];
            if v >= best_v {
                best_v = v;
                best_g = Some(g);
            }
        }
        if let Some(g) = best_g {
            used[g] = true;
            tp += 1;
        }
    }
    tp
}

/// 关键点（PLAN §4.4）真实数据训练：COCO 姿态格式（coco8-pose）+ 直接回归头
/// （decode = "direct"，选型取舍见 av-tasks::keypoint 模块文档）。
///
/// 损失 = BCE(cls) + L1(box) + 可见性加权 L1(偏移) + BCE(可见性) + (1 − mean OKS)。
/// 验收指标取**训练集** PCK@0.5（预测点与 gt 可见点距离 < 0.1×img_size 的占比，
/// 按 gt 实例逐点诚实计数；任务规范：真实姿态从零小模型 200 epochs 的诚实起点）；
/// val split 每 interval 评测仅作泛化观察。
fn train_keypoint(cfg: &RunConfig, run_id: &str, run_dir: &Path) -> AvResult<TrainReport> {
    let (num_keypoints, img_size) = match cfg.model.tasks.first() {
        Some(TaskCfg::Keypoint(k)) => (k.num_keypoints, k.img_size),
        _ => unreachable!("train_keypoint 只处理关键点任务"),
    };
    let device = resolve_device(cfg);
    let root = cfg
        .data
        .sources
        .train
        .dir
        .clone()
        .ok_or_else(|| AvError::config("dir 数据源缺 data.sources.train.dir"))?;
    let split = cfg.data.sources.train.split.as_deref().unwrap_or("train");
    // 增强就绪 → 训练侧走「raw 样本 + 逐 epoch 随机编码」；否则保持既有整集
    // 预解码快速路径（val 永远走 plain 加载器，评测分布干净）。
    let aug_cfg = train_augment_cfg(cfg, TaskKind::Keypoint);
    let train_raw = if let Some(a) = &aug_cfg {
        println!(
            "[augment] keypoint 训练增强生效: flip={} hsv={:?} scale_jitter={:?} close_last_epochs={}",
            a.flip, a.hsv, a.scale_jitter, a.close_last_epochs
        );
        Some(dataset::load_cocopose_dir_raw(&root, split)?)
    } else {
        None
    };
    let train = if train_raw.is_some() {
        Vec::new()
    } else {
        dataset::load_cocopose_dir(&root, split, img_size, Device::Cpu, imagenet_norm(cfg))?
    };
    let val = match cfg.data.sources.val.dir.as_ref() {
        Some(d) => dataset::load_cocopose_dir(
            d,
            cfg.data.sources.val.split.as_deref().unwrap_or("val"),
            img_size,
            Device::Cpu,
            imagenet_norm(cfg),
        )?,
        None => dataset::load_cocopose_dir(&root, "val", img_size, Device::Cpu, imagenet_norm(cfg))?,
    };
    // 关键点模板一致性：标注点数必须与配置 num_keypoints 一致（头通道数在
    // 建模期固定；COCO 17 点模板混入其他点数的数据会静默错位，宁可报错）
    let train_kpts: Vec<&Vec<[f32; 3]>> = match &train_raw {
        Some(raw) => raw.iter().flat_map(|s| s.kpts.iter()).collect(),
        None => train.iter().flat_map(|s| s.kpts.iter()).collect(),
    };
    for gk in train_kpts
        .iter()
        .copied()
        .chain(val.iter().flat_map(|s| s.kpts.iter()))
    {
        if gk.len() != num_keypoints {
            return Err(AvError::data(format!(
                "标注关键点数 {} 与 keypoint.num_keypoints = {num_keypoints} 不一致\
                 （数据集与配置必须同一关键点模板）",
                gk.len()
            )));
        }
    }
    let train_n = train_raw.as_ref().map_or(train.len(), |r| r.len());
    let train_insts: usize = match &train_raw {
        Some(raw) => raw.iter().map(|s| s.kpts.len()).sum(),
        None => train.iter().map(|s| s.kpts.len()).sum(),
    };
    println!(
        "[keypoint] root={} train={}图/{}实例 val={}图/{}实例 K={} img_size={} stride=8",
        root.display(),
        train_n,
        train_insts,
        val.len(),
        val.iter().map(|s| s.kpts.len()).sum::<usize>(),
        num_keypoints,
        img_size
    );

    let vs = VarStore::new(device);
    let model = build_model(&vs.root(), cfg)?;
    apply_pretrain(&vs, cfg)?;
    let mut opt = make_opt(&vs, cfg)?;
    let mut rng = XorShift::new(cfg.seed);
    let bs = (cfg.train.batch_size as usize).max(1);
    let mut final_loss = 0f32;
    #[allow(unused_assignments)] // 初值仅占位：epochs>=1 时变量必在首次读取前赋值
    let (mut val_pck, mut val_oks) = (0f32, 0f32);
    // BN train/eval 装配（resnet18 骨干生效；冻结骨干时恒 eval）
    let bn_train = bn_train_mode(cfg);

    for epoch in 1..=cfg.train.epochs {
        model.set_train(bn_train);
        opt.set_lr(schedule_lr(cfg, epoch));
        let mut order: Vec<usize> = (0..train_n).collect();
        for i in (1..order.len()).rev() {
            let j = rng.next_usize(i + 1);
            order.swap(i, j);
        }
        let strong_on = aug_cfg
            .as_ref()
            .map(|a| strong_aug_on(a, cfg, epoch))
            .unwrap_or(false);
        let mut aug_rng = epoch_aug_rng(cfg.seed, epoch);
        let mut epoch_loss = 0f32;
        let mut steps = 0usize;
        let acc_steps = cfg.train.accumulate_steps.max(1) as usize;
        opt.zero_grad();
        for chunk in order.chunks(bs) {
            let batch_samples: Vec<KeypointSample> = chunk
                .iter()
                .map(|&i| -> AvResult<KeypointSample> {
                    match (&train_raw, &aug_cfg) {
                        // 增强路径：每样本抽一份 plan（像素与坐标同 plan ⇒ 坐标同步）
                        (Some(raw), Some(a)) => {
                            let plan = if strong_on {
                                augment::draw_plan(a, &mut aug_rng)
                            } else {
                                AugmentPlan::none()
                            };
                            dataset::encode_keypoint_sample(&raw[i], img_size, Device::Cpu, &plan, imagenet_norm(cfg))
                        }
                        // 快速路径：整集预解码直接克隆（历史行为，逐位一致）
                        _ => Ok(train[i].clone()),
                    }
                })
                .collect::<AvResult<Vec<_>>>()?;
            let x = dataset::stack_kp_samples(&batch_samples)?.to_device(device);
            let batch = TrainBatch::Keypoint {
                boxes: batch_samples.iter().map(|s| s.boxes.clone()).collect(),
                kpts: batch_samples.iter().map(|s| s.kpts.clone()).collect(),
                labels: batch_samples.iter().map(|s| s.labels.clone()).collect(),
            };
            let loss = model.loss(&x, &batch)?;
            loss.backward();
            epoch_loss += loss.double_value(&[]) as f32;
            steps += 1;
            if steps % acc_steps == 0 {
                if cfg.train.grad_clip > 0.0 {
                    opt.clip_grad_norm(cfg.train.grad_clip as f64);
                }
                opt.step();
                opt.zero_grad();
            }
        }
        if steps % acc_steps != 0 {
            if cfg.train.grad_clip > 0.0 {
                opt.clip_grad_norm(cfg.train.grad_clip as f64);
            }
            opt.step();
            opt.zero_grad();
        }
        final_loss = epoch_loss / steps.max(1) as f32;
        let eval_due = epoch == 1
            || epoch == cfg.train.epochs
            || cfg.eval.interval_epochs == 0
            || epoch % cfg.eval.interval_epochs == 0;
        if eval_due {
            model.set_train(false); // 评测 = BN 推理语义（running 统计量）
            let TaskModel::Keypoint(m) = &model else {
                unreachable!("关键点任务模型类型")
            };
            (val_pck, val_oks, _, _) = eval_kp_samples(m, &val, device)?;
            model.set_train(bn_train); // 恢复训练态
            println!(
                "[keypoint] run={run_id} epoch={epoch}/{} loss={final_loss:.4} val PCK@0.5={val_pck:.3} val meanOKS={val_oks:.3}",
                cfg.train.epochs
            );
        }
        log_epoch_metrics(
            run_dir,
            epoch,
            final_loss,
            "pck@0.5",
            val_pck,
            Some(("mean_oks", val_oks)),
        );
    }

    // 验收口径：训练集 PCK@0.5（8 图小数据的诚实过拟合基线）
    // 增强路径下训练样本为 raw，用 none() plan 重编码出与不增强逐位一致的验收集
    model.set_train(false); // 验收评测 = BN 推理语义
    let TaskModel::Keypoint(m) = &model else {
        unreachable!("关键点任务模型类型")
    };
    let (train_pck, train_oks, n_vis, n_inst) = match &train_raw {
        Some(raw) => {
            let clean: Vec<KeypointSample> = raw
                .iter()
                .map(|s| {
                    dataset::encode_keypoint_sample(
                        s,
                        img_size,
                        Device::Cpu,
                        &AugmentPlan::none(),
                        imagenet_norm(cfg),
                    )
                })
                .collect::<AvResult<Vec<_>>>()?;
            eval_kp_samples(m, &clean, device)?
        }
        None => eval_kp_samples(m, &train, device)?,
    };

    save_checkpoint(&vs, &run_dir.join(CKPT_DIR))?;
    Ok(TrainReport {
        run_id: run_id.to_string(),
        task: "keypoint".into(),
        epochs: cfg.train.epochs,
        final_loss,
        metric: "train_pck@0.5".into(),
        metric_value: train_pck,
        secondary: Some(("val_pck@0.5".into(), val_pck)),
        run_dir: run_dir.display().to_string(),
    })
    .map(|r| {
        println!(
            "[keypoint] 训练集验收：PCK@0.5={train_pck:.3} meanOKS={train_oks:.3}\
             （{n_inst} 个 gt 实例 / {n_vis} 个可见 gt 点）| val PCK@0.5={val_pck:.3} val meanOKS={val_oks:.3}",
        );
        r
    })
}

/// 关键点样本集评测：PCK@0.5 + 平均 OKS。
///
/// - **PCK@0.5**：对每个 gt 实例取框 IoU 最大（≥ 0.1，类无关）的预测实例，
///   其各通道关键点与该实例可见 gt 点逐一比对，距离 < 0.1×img_size 计命中；
///   命中数按**全部 gt 实例**的可见点诚实计数（无匹配预测 = 该实例全错过）。
/// - **mean OKS**：匹配上的实例对（pred 关键点, gt, s=sqrt(框面积)）的
///   [`av_tasks::oks::oks_scalar`] 均值（COCO 17 σ 常数表）。
///
/// 返回 (pck, mean_oks, 可见 gt 点数, gt 实例数)。
fn eval_kp_samples(
    m: &KeypointModel,
    samples: &[KeypointSample],
    device: Device,
) -> AvResult<(f32, f32, usize, usize)> {
    if samples.is_empty() {
        return Err(AvError::data("验证集为空"));
    }
    let thr = 0.1 * m.img_size() as f32;
    let mut per_image: Vec<Vec<av_core::types::Detection>> = Vec::new();
    for chunk in samples.chunks(4) {
        let x = dataset::stack_kp_samples(chunk)?.to_device(device);
        per_image.extend(m.predict(&x, 0.1, 0.5)?);
    }
    let (mut hits, mut visible, mut n_inst) = (0usize, 0usize, 0usize);
    let (mut ok_sum, mut ok_cnt) = (0f64, 0usize);
    for (gi, s) in samples.iter().enumerate() {
        for (g, gk) in s.kpts.iter().enumerate() {
            n_inst += 1;
            let n_vis_here = gk.iter().filter(|p| p[2] > 0.0).count();
            visible += n_vis_here;
            // gt 框（cxcywh 画布像素）→ xyxy，与预测同空间可比
            let (cx, cy, bw, bh) = {
                let b = s.boxes[g];
                (b[0], b[1], b[2], b[3])
            };
            let g_aabb = Aabb::new(cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0);
            // 匹配：类无关最大框 IoU 的预测实例（< 0.1 视为漏检，该实例全错过）
            let mut best: Option<(f32, &av_core::types::Detection)> = None;
            for d in &per_image[gi] {
                let v = g_aabb.iou(&d.bbox);
                if best.map(|(bv, _)| v > bv).unwrap_or(true) {
                    best = Some((v, d));
                }
            }
            let Some((best_iou, d)) = best else { continue };
            if best_iou < 0.1 {
                continue;
            }
            let Some(kps) = d.keypoints.as_ref() else { continue };
            // OKS（有可见点的实例才计均值）
            if n_vis_here > 0 {
                let scale = (bw * bh).sqrt().max(1e-3);
                ok_sum += av_tasks::oks::oks_scalar(kps, gk, scale) as f64;
                ok_cnt += 1;
            }
            for (j, gp) in gk.iter().enumerate() {
                if gp[2] <= 0.0 {
                    continue;
                }
                if let Some(pk) = kps.get(j) {
                    let dx = pk[0] - gp[0];
                    let dy = pk[1] - gp[1];
                    if dx * dx + dy * dy < thr * thr {
                        hits += 1;
                    }
                }
            }
        }
    }
    if visible == 0 {
        return Err(AvError::data("评测集无可见 gt 关键点"));
    }
    Ok((
        hits as f32 / visible as f32,
        (ok_sum / ok_cnt.max(1) as f64) as f32,
        visible,
        n_inst,
    ))
}

/// OBB（obb_mode=true）真实数据训练：DOTA 格式 + KFIoU 回归 + 旋转 NMS（PLAN §4.2）。
fn train_detect_obb(cfg: &RunConfig, run_id: &str, run_dir: &Path) -> AvResult<TrainReport> {
    let (num_classes, img_size) = match cfg.model.tasks.first() {
        Some(TaskCfg::Detect(d)) => (d.num_classes as u32, d.img_size),
        _ => unreachable!("train_detect_obb 只处理 OBB 任务"),
    };
    let device = resolve_device(cfg);
    let root = cfg
        .data
        .sources
        .train
        .dir
        .clone()
        .ok_or_else(|| AvError::config("dir 数据源缺 data.sources.train.dir"))?;
    let split = cfg.data.sources.train.split.as_deref().unwrap_or("train");
    let aug_cfg = train_augment_cfg(cfg, TaskKind::Detect);
    let train_raw = if let Some(a) = &aug_cfg {
        println!(
            "[augment] obb 训练增强生效: flip={} hsv={:?} scale_jitter={:?} close_last_epochs={}",
            a.flip, a.hsv, a.scale_jitter, a.close_last_epochs
        );
        Some(dataset::load_dota_dir_raw(&root, split)?)
    } else {
        None
    };
    let train = if train_raw.is_some() {
        Vec::new()
    } else {
        dataset::load_dota_dir(&root, split, img_size, Device::Cpu, imagenet_norm(cfg))?
    };
    let val = match cfg.data.sources.val.dir.as_ref() {
        Some(d) => dataset::load_dota_dir(
            d,
            cfg.data.sources.val.split.as_deref().unwrap_or("val"),
            img_size,
            Device::Cpu,
            imagenet_norm(cfg),
        )?,
        None => dataset::load_dota_dir(&root, "val", img_size, Device::Cpu, imagenet_norm(cfg))?,
    };
    let train_n = train_raw.as_ref().map_or(train.len(), |r| r.len());
    println!(
        "[obb] root={} train={} val={} classes={} img_size={}",
        root.display(),
        train_n,
        val.len(),
        num_classes,
        img_size
    );

    let vs = VarStore::new(device);
    let model = build_model(&vs.root(), cfg)?;
    let mut opt = make_opt(&vs, cfg)?;
    let mut rng = XorShift::new(cfg.seed);
    let bs = (cfg.train.batch_size as usize).max(1);
    let mut final_loss = 0f32;
    let mut miou = 0f32;
    let mut r50 = 0f32;
    // BN train/eval 装配（resnet18 骨干生效；冻结骨干时恒 eval）
    let bn_train = bn_train_mode(cfg);

    for epoch in 1..=cfg.train.epochs {
        model.set_train(bn_train);
        opt.set_lr(schedule_lr(cfg, epoch));
        let mut order: Vec<usize> = (0..train_n).collect();
        for i in (1..order.len()).rev() {
            let j = rng.next_usize(i + 1);
            order.swap(i, j);
        }
        let strong_on = aug_cfg
            .as_ref()
            .map(|a| strong_aug_on(a, cfg, epoch))
            .unwrap_or(false);
        let mut aug_rng = epoch_aug_rng(cfg.seed, epoch);
        let mut epoch_loss = 0f32;
        let mut steps = 0usize;
        let acc_steps = cfg.train.accumulate_steps.max(1) as usize;
        opt.zero_grad();
        for chunk in order.chunks(bs) {
            let batch_samples: Vec<dataset::ObbSample> = chunk
                .iter()
                .map(|&i| -> AvResult<dataset::ObbSample> {
                    match (&train_raw, &aug_cfg) {
                        (Some(raw), Some(a)) => {
                            let plan = if strong_on {
                                augment::draw_plan(a, &mut aug_rng)
                            } else {
                                AugmentPlan::none()
                            };
                            dataset::encode_obb_sample(&raw[i], img_size, Device::Cpu, &plan, imagenet_norm(cfg))
                        }
                        _ => Ok(train[i].clone()),
                    }
                })
                .collect::<AvResult<Vec<_>>>()?;
            let x = dataset::stack_obb_samples(&batch_samples)?.to_device(device);
            let batch = TrainBatch::Obb {
                boxes: batch_samples.iter().map(|s| s.boxes.clone()).collect(),
                labels: batch_samples.iter().map(|s| s.labels.clone()).collect(),
            };
            let loss = model.loss(&x, &batch)?;
            loss.backward();
            epoch_loss += loss.double_value(&[]) as f32;
            steps += 1;
            if steps % acc_steps == 0 {
                if cfg.train.grad_clip > 0.0 {
                    opt.clip_grad_norm(cfg.train.grad_clip as f64);
                }
                opt.step();
                opt.zero_grad();
            }
        }
        if steps % acc_steps != 0 {
            if cfg.train.grad_clip > 0.0 {
                opt.clip_grad_norm(cfg.train.grad_clip as f64);
            }
            opt.step();
            opt.zero_grad();
        }
        final_loss = epoch_loss / steps.max(1) as f32;
        // epoch 末评测：先切 BN 推理语义，评完恢复训练态（下一 epoch 继续训练）
        model.set_train(false);
        let TaskModel::Detect(m) = &model else {
            unreachable!("OBB 模型类型")
        };
        (miou, r50) = eval_obb_samples(m, &val, device)?;
        model.set_train(bn_train);
        if epoch == 1 || epoch % 10 == 0 || epoch == cfg.train.epochs {
            println!(
                "[obb] run={run_id} epoch={epoch}/{} loss={final_loss:.4} 旋转mIoU={miou:.3} R@0.5={r50:.3}",
                cfg.train.epochs
            );
        }
        log_epoch_metrics(
            run_dir,
            epoch,
            final_loss,
            "obb_miou",
            miou,
            Some(("recall@0.5", r50)),
        );
    }

    save_checkpoint(&vs, &run_dir.join(CKPT_DIR))?;
    Ok(TrainReport {
        run_id: run_id.to_string(),
        task: "obb".into(),
        epochs: cfg.train.epochs,
        final_loss,
        metric: "obb_miou".into(),
        metric_value: miou,
        secondary: Some(("recall@0.5".into(), r50)),
        run_dir: run_dir.display().to_string(),
    })
}

/// OBB 验证集评测：旋转 IoU（RotBox，角度感知）定位均值 + R@0.5（类别联合）。
fn eval_obb_samples(
    m: &av_tasks::models::DetectModel,
    val: &[dataset::ObbSample],
    device: Device,
) -> AvResult<(f32, f32)> {
    use av_core::geometry::RotBox;
    let n = val.len();
    if n == 0 {
        return Err(AvError::data("验证集为空"));
    }
    let mut per_image: Vec<Vec<av_core::types::Detection>> = Vec::new();
    for chunk in val.chunks(8) {
        let xs: Vec<Tensor> = chunk.iter().map(|s| s.x.copy()).collect();
        let x = Tensor::stack(&xs, 0).to_device(device);
        per_image.extend(m.predict(&x, 0.1, 0.5)?);
    }
    let mut best = vec![0f32; n];
    let mut ok = vec![false; n];
    for (gi, s) in val.iter().enumerate() {
        for d in &per_image[gi] {
            let dr = RotBox {
                cx: (d.bbox.x1 + d.bbox.x2) / 2.0,
                cy: (d.bbox.y1 + d.bbox.y2) / 2.0,
                w: d.bbox.x2 - d.bbox.x1,
                h: d.bbox.y2 - d.bbox.y1,
                theta: d.angle.unwrap_or(0.0),
            };
            let mut hit = false;
            for (b, &label) in s.boxes.iter().zip(&s.labels) {
                let gr = RotBox {
                    cx: b[0],
                    cy: b[1],
                    w: b[2],
                    h: b[3],
                    theta: b[4],
                };
                let v = gr.iou(&dr);
                if v > best[gi] {
                    best[gi] = v;
                }
                if d.class_id == label {
                    hit = true;
                }
            }
            if hit {
                ok[gi] = true;
            }
        }
    }
    let matched = best.iter().filter(|&&v| v >= 0.5).count();
    Ok((
        best.iter().sum::<f32>() / n as f32,
        matched as f32 / n as f32,
    ))
}

/// 评测入口：合成验证集或 YOLO 目录 val split 的冒烟指标（真实基准协议按 M8 落地）。
pub fn eval(cfg: &RunConfig, weights: &Path) -> AvResult<serde_json::Value> {
    let (model, device) = load_model(cfg, weights)?;
    match cfg.model.tasks.first() {
        Some(TaskCfg::Classify(c)) => {
            // dir 源：ImageFolder val split 真实 top1；synthetic：合成评测集冒烟
            let acc = if cfg.data.pipeline == DataPipeline::Dir {
                let (root, split) = classify_val_source(cfg);
                let (val, labels, class_map) = dataset::load_imagefolder(
                    &root,
                    &split,
                    c.img_size,
                    true,
                    Device::Cpu,
                    imagenet_norm(cfg),
                )?;
                if class_map.len() != c.num_classes {
                    return Err(AvError::config(format!(
                        "classify.num_classes = {} 与 ImageFolder 类别数 {} 不一致",
                        c.num_classes,
                        class_map.len()
                    )));
                }
                eval_classify_samples(&model, &val, &labels, device)?
            } else {
                eval_classify(&model, c.num_classes as u32, c.img_size, device)?
            };
            Ok(serde_json::json!({ "task": "classify", "top1": acc }))
        }
        Some(TaskCfg::Detect(d)) => {
            let TaskModel::Detect(m) = &model else {
                return Err(AvError::train("模型与任务不匹配"));
            };
            match cfg.data.pipeline {
                DataPipeline::Dir => {
                    let dir = cfg
                        .data
                        .sources
                        .val
                        .dir
                        .as_ref()
                        .or(cfg.data.sources.train.dir.as_ref())
                        .ok_or_else(|| AvError::config("dir 数据源缺失"))?;
                    let split = cfg.data.sources.val.split.as_deref().unwrap_or("val");
                    // 预解码固定 CPU，评测批内再搬到模型设备
                    let val =
                        dataset::load_yolo_dir(dir, split, d.img_size, Device::Cpu, imagenet_norm(cfg))?;
                    let evm = eval_detect_samples(m, &val, device)?;
                    Ok(serde_json::json!({
                        "task": "detect",
                        "mean_iou": evm.miou,
                        "recall@0.5": evm.r50,
                        "map50": evm.map50,
                        "map50_95": evm.map50_95,
                    }))
                }
                _ => {
                    let (miou, r50, _, _) =
                        eval_detect(&model, d.num_classes as u32, d.img_size, device)?;
                    Ok(serde_json::json!({ "task": "detect", "mean_iou": miou, "recall@0.5": r50 }))
                }
            }
        }
        Some(TaskCfg::Seg(_)) => {
            let TaskModel::Seg(m) = &model else {
                return Err(AvError::train("模型与任务不匹配"));
            };
            if cfg.data.pipeline != DataPipeline::Dir {
                return Err(AvError::config("seg 评测需要 data.pipeline = \"dir\"（COCO 分割格式）"));
            }
            let dir = cfg
                .data
                .sources
                .val
                .dir
                .as_ref()
                .or(cfg.data.sources.train.dir.as_ref())
                .ok_or_else(|| AvError::config("dir 数据源缺失"))?;
            let split = cfg.data.sources.val.split.as_deref().unwrap_or("val");
            let val = dataset::load_cocoseg_dir(dir, split, m.img_size(), Device::Cpu, imagenet_norm(cfg))?;
            let (miou, r50, p50, n_gt, per_class) = eval_seg_samples(m, &val, device)?;
            let per_class_json: serde_json::Map<String, serde_json::Value> = per_class
                .into_iter()
                .map(|(cls, v, n)| (cls.to_string(), serde_json::json!({ "mask_miou": v, "gt": n })))
                .collect();
            Ok(serde_json::json!({
                "task": "seg",
                "gt_instances": n_gt,
                "mask_miou": miou,
                "recall@0.5": r50,
                "precision@0.5": p50,
                "per_class_mask_miou": per_class_json,
            }))
        }
        Some(TaskCfg::Keypoint(_)) => {
            let TaskModel::Keypoint(m) = &model else {
                return Err(AvError::train("模型与任务不匹配"));
            };
            if cfg.data.pipeline != DataPipeline::Dir {
                return Err(AvError::config(
                    "keypoint 评测需要 data.pipeline = \"dir\"（COCO 姿态格式）",
                ));
            }
            let dir = cfg
                .data
                .sources
                .val
                .dir
                .as_ref()
                .or(cfg.data.sources.train.dir.as_ref())
                .ok_or_else(|| AvError::config("dir 数据源缺失"))?;
            let split = cfg.data.sources.val.split.as_deref().unwrap_or("val");
            let val = dataset::load_cocopose_dir(dir, split, m.img_size(), Device::Cpu, imagenet_norm(cfg))?;
            let (pck, mean_oks, n_vis, n_inst) = eval_kp_samples(m, &val, device)?;
            Ok(serde_json::json!({
                "task": "keypoint",
                "gt_instances": n_inst,
                "visible_kpts": n_vis,
                "pck@0.5": pck,
                "mean_oks": mean_oks,
            }))
        }
        _ => Err(AvError::config("该任务类型在 v0.1 引擎未支持")),
    }
}

// ---------------------------------------------------------------------------
// 内部实现
// ---------------------------------------------------------------------------

/// cfg.device（"cpu" / "cuda" / "cuda:N"）→ [`tch::Device`]。
///
/// 设备解析是唯一入口（训练/推理/评测共用）：请求 CUDA 但运行环境不可用时
/// 回退 CPU 并 `tracing::warn`。GPU 路线状态与复现步骤见 docs/gpu.md：
/// - 默认构建链接 CPU 版 libtorch 2.11 → `Cuda::is_available()=false`，直接回退；
/// - 旧实验（实验 A）：cu121/libtorch 2.4 版 libtorch（RTX 5060 Ti = sm_120）
///   首个 CUDA 内核启动即报 "no kernel image is available for execution on the
///   device"（cu121 无 sm_120 SASS、发行版剥离了可 JIT 的 PTX）；
/// - 现行 cu128/libtorch 2.11（实验二）已实测内核可执行，前提是 Windows/MSVC
///   下先走 [`crate::cuda_link::ensure_torch_cuda_loaded`] 强制加载 CUDA 后端；
///   [`cuda_kernels_usable`] 的微型内核探测保留为最后一道防误用闸门。
pub fn resolve_device(cfg: &RunConfig) -> Device {
    let spec = cfg.device.trim().to_ascii_lowercase();
    let requested = match spec.as_str() {
        "cpu" => Device::Cpu,
        "cuda" => Device::Cuda(0),
        _ if spec.starts_with("cuda:") => match spec["cuda:".len()..].parse::<usize>() {
            Ok(idx) => Device::Cuda(idx),
            Err(_) => {
                tracing::warn!("无法解析 device = {:?}（支持 cpu / cuda[:N]），回退 CPU", cfg.device);
                return Device::Cpu;
            }
        },
        _ => {
            tracing::warn!("未知 device = {:?}（支持 cpu / cuda[:N]），回退 CPU", cfg.device);
            return Device::Cpu;
        }
    };
    if requested.is_cuda() {
        // Windows/MSVC 专项：链接器会丢弃无符号引用的 torch_cuda 导入库，
        // 必须显式加载 torch_cuda.dll 才能注册 CUDA 后端（docs/gpu.md 实验二 §5）。
        crate::cuda_link::ensure_torch_cuda_loaded();
        let available = Device::cuda_if_available();
        if available == Device::Cpu {
            tracing::warn!(
                "device = {:?} 请求 CUDA，但当前环境不可用（CPU 版 libtorch 或无驱动/被禁用），回退 CPU",
                cfg.device
            );
            return Device::Cpu;
        }
        if let Device::Cuda(idx) = requested {
            if idx >= tch::Cuda::device_count().max(0) as usize {
                tracing::warn!(
                    "device = {:?} 超出可用范围（device_count = {}），回退 {}",
                    cfg.device,
                    tch::Cuda::device_count(),
                    match available { Device::Cuda(i) => format!("cuda:{i}"), _ => "cpu".into() }
                );
                return available;
            }
        }
        if !cuda_kernels_usable(requested) {
            tracing::warn!(
                "device = {:?} 的 CUDA 内核无法执行（本机 GPU 架构缺内核且无 PTX 可 JIT，\
                 报错详见 docs/gpu.md 实验 A），回退 CPU",
                cfg.device
            );
            return Device::Cpu;
        }
    }
    requested
}

/// 微型内核探测：设备探测（`Cuda::is_available` / cudnn）通过不代表库内有
/// 本机 GPU 架构的内核。启动一次极小的 sum 内核，失败（panic）即视为不可用。
/// tch 的 CUDA 错误以 panic 形式抛出，这里临时静默 panic hook 再恢复。
/// 仅在 `is_available()` 为真时才会被调用，CPU 版 libtorch 构建零开销。
fn cuda_kernels_usable(device: Device) -> bool {
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let ok = std::panic::catch_unwind(|| {
        let t = Tensor::from_slice(&[1.0f32, 2.0f32]).to_device(device);
        // sum 需要真正的内核启动；double_value 触发同步取回
        t.sum(Kind::Float).double_value(&[]) == 3.0
    })
    .unwrap_or(false);
    std::panic::set_hook(prev_hook);
    ok
}

/// 按配置解析设备并构建模型；同时返回模型设备（供样本张量 `.to_device` 对齐）。
fn load_model(cfg: &RunConfig, weights: &Path) -> AvResult<(TaskModel, Device)> {
    // 顺序很关键：先建模型（VarStore 里才有变量），再用 checkpoint 覆写变量
    let device = resolve_device(cfg);
    let mut vs = VarStore::new(device);
    let model = build_model(&vs.root(), cfg)?;
    load_checkpoint(&mut vs, weights)?;
    Ok((model, device))
}

fn make_opt(vs: &VarStore, cfg: &RunConfig) -> AvResult<tch::nn::Optimizer> {
    let lr = cfg.train.optimizer.lr as f64;
    tch::nn::Adam::default()
        .build(vs, lr)
        .map_err(|e| AvError::train(format!("优化器构建失败: {e}")))
}

/// 输入归一化域判定（训练/评测/推理共用）：`backbone.imagenet_norm = true`
/// 时数据管线输出 ImageNet mean/std 域（ImageNet 预训练骨干的 BN running
/// 统计量在 ImageNet 域，输入必须同域）；默认 false = [0,1] RGB 历史语义。
fn imagenet_norm(cfg: &RunConfig) -> bool {
    cfg.model.backbone.imagenet_norm
}

/// BN train 模式判定（训练循环用）：冻结骨干（`[pretrain].freeze_backbone =
/// true`）时 BN 恒 eval（Detectron/mmdet FrozenBN 微调配方，running 统计量
/// 不随目标域重估）；否则训练态（批统计 + running 更新，评测前切回 false）。
fn bn_train_mode(cfg: &RunConfig) -> bool {
    !cfg.pretrain.freeze_backbone
}

/// [pretrain] 预训练权重接入（预训练权重方案第一层）：在 build_model 之后、
/// 优化器 build 之前调用。
///
/// - `weight_path` 为 `.safetensors` 文件 → [`weight_adapter`] 读取适配；
///   为目录 → 按 avpretrain 原生格式读入（先校验 manifest 全量哈希再读）；
/// - `load_only_backbone = true` 时只匹配名字含 "backbone" 的目标变量；
/// - `freeze_backbone = true` 时把 backbone 变量 `set_requires_grad(false)`——
///   必须发生在 `make_opt` 之前：优化器按 `trainable_variables` 建参，
///   冻结张量不再累计梯度、Adam 自然跳过；
/// - 部分加载合法（外部权重极少同构），loaded/skipped/missing/unexpected
///   完整统计打印到 stdout（与训练日志同一通道）。
///
/// 默认 `enable = false` 时直接返回：既有链路零变化。
fn apply_pretrain(vs: &VarStore, cfg: &RunConfig) -> AvResult<()> {
    let pre = &cfg.pretrain;
    if !pre.enable {
        return Ok(());
    }
    let weight_path = pre.weight_path.as_deref().ok_or_else(|| {
        AvError::config("pretrain.enable = true 但未指定 pretrain.weight_path")
    })?;

    let sources: Vec<(String, Tensor)> = if weight_path.is_dir() {
        let manifest = av_weight_store::read_manifest(weight_path)?;
        av_weight_store::verify_hashes(weight_path, &manifest)
            .map_err(|e| AvError::train(format!("预训练权重校验失败: {e}")))?;
        println!(
            "[pretrain] avpretrain 目录 {}：{} 个张量，哈希校验通过（backbone={} task={} epoch={:?} created_at={}）",
            weight_path.display(),
            manifest.tensors.len(),
            manifest.backbone,
            manifest.task,
            manifest.epoch,
            manifest.created_at,
        );
        av_weight_store::read_all_named(weight_path, &manifest)?
    } else {
        weight_adapter::read_safetensors_all(weight_path)?
    };

    let vars = vs.variables();
    let targets: Vec<(String, Vec<i64>)> = vars
        .iter()
        .filter(|(n, _)| !pre.load_only_backbone || n.contains("backbone"))
        .map(|(n, t)| (n.clone(), t.size()))
        .collect();
    let map = match &pre.layer_map {
        Some(p) => LayerMap::from_toml_path(p)?,
        None => LayerMap::default(),
    };
    let report = weight_adapter::adapt(sources, &map, &targets);

    // 写回（no_grad 原地 copy_；copy_ 支持跨设备拷贝，CPU 权重可直接写入模型设备）
    let matched = &report.tensors;
    tch::no_grad(|| {
        for (name, mut t) in vs.variables() {
            if let Some((_, src)) = matched.iter().find(|(n, _)| *n == name) {
                t.copy_(src);
            }
        }
    });

    // 冻结 backbone：必须在 make_opt 之前（优化器按 trainable_variables 建参）
    let mut frozen = 0usize;
    if pre.freeze_backbone {
        for (name, t) in vs.variables() {
            if name.contains("backbone") {
                // set_requires_grad 经共享 TensorImpl 就地生效（tch VarStore::freeze 同款写法）
                let _ = t.set_requires_grad(false);
                frozen += 1;
            }
        }
    }

    println!(
        "[pretrain] 来源={} {}（frozen_backbone={frozen} / {} 个目标变量）",
        weight_path.display(),
        report.summary(),
        targets.len()
    );
    for sm in &report.skipped_shape_mismatch {
        tracing::debug!(
            "[pretrain] 形状不匹配跳过: {} ← {} 期望 {:?} 得到 {:?}",
            sm.target,
            sm.source,
            sm.expected,
            sm.got
        );
    }
    Ok(())
}
// ---------------------------------------------------------------------------
// 训练期增强（数据增强官任务 §2）：只在训练路径生效，val / 推理 / 评测永不增强
// ---------------------------------------------------------------------------

/// 训练源任务配置里的增强配置：取 `data.sources.train.tasks` 中同 kind 条目的
/// `augment`；无实际强度（全零）时返回 None，引擎保持既有「整集预解码」快速
/// 路径，行为与历史版本逐位一致。
fn train_augment_cfg(cfg: &RunConfig, kind: TaskKind) -> Option<av_core::config::AugmentCfg> {
    cfg.data
        .sources
        .train
        .tasks
        .iter()
        .find(|t| t.kind == kind)
        .map(|t| t.augment.clone())
        .filter(augment::has_strength)
}

/// 本 epoch 强增强是否生效：`close_last_epochs = N` 时最后 N 个 epoch 自动关
/// （模型在干净分布上收尾）。
fn strong_aug_on(a: &av_core::config::AugmentCfg, cfg: &RunConfig, epoch: u32) -> bool {
    augment::strong_aug_active(a, epoch, cfg.train.epochs)
}

/// 每 epoch 独立播种的增强 RNG（与洗牌 RNG 分流：增强抽样不扰动洗牌序列，
/// 同 seed 同 epoch ⇒ 同一串 plan，实验可复现）。
fn epoch_aug_rng(seed: u64, epoch: u32) -> XorShift {
    XorShift::new(
        seed ^ (epoch as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xA065_5EED,
    )
}

/// warmup + 余弦退火（PLAN §5.2）。
fn schedule_lr(cfg: &RunConfig, epoch: u32) -> f64 {    let lr0 = cfg.train.optimizer.lr as f64;
    let warm = cfg.train.warmup_epochs;
    if warm > 0.0 && (epoch as f32) <= warm {
        return lr0 * (epoch as f32 / warm).min(1.0) as f64;
    }
    let span = (cfg.train.epochs as f32 - warm).max(1.0);
    let t = (((epoch as f32) - warm) / span).clamp(0.0, 1.0) as f64;
    let min = lr0 * cfg.train.scheduler.lr_min_factor as f64;
    min + 0.5 * (lr0 - min) * (1.0 + t.cos())
}

fn train_classify(cfg: &RunConfig, run_id: &str, run_dir: &Path) -> AvResult<TrainReport> {
    match cfg.data.pipeline {
        DataPipeline::Synthetic => train_classify_synthetic(cfg, run_id, run_dir),
        DataPipeline::Dir => train_classify_imagenette(cfg, run_id, run_dir),
        DataPipeline::AvPack => Err(AvError::config(
            "avpack 数据源按 M2 落地（PLAN 附录 B）",
        )),
    }
}

/// 分类 dir 源数据根：classify.data_dir 优先，回落 data.sources.train.dir
/// （与检测侧共用 sources 结构）。
fn classify_data_root(cfg: &RunConfig) -> AvResult<PathBuf> {
    if let Some(TaskCfg::Classify(c)) = cfg.model.tasks.first() {
        if let Some(d) = &c.data_dir {
            return Ok(d.clone());
        }
    }
    cfg.data
        .sources
        .train
        .dir
        .clone()
        .ok_or_else(|| {
            AvError::config(
                "分类 dir 数据源缺失：需指定 classify.data_dir 或 data.sources.train.dir",
            )
        })
}

/// 分类 dir 源 val 侧解析：val 源缺省回落 train 根 + "val" split。
fn classify_val_source(cfg: &RunConfig) -> (PathBuf, String) {
    let split = cfg
        .data
        .sources
        .val
        .split
        .as_deref()
        .unwrap_or("val")
        .to_string();
    match cfg.data.sources.val.dir.as_ref() {
        Some(d) => (d.clone(), split),
        None => (classify_data_root(cfg).unwrap_or_default(), split),
    }
}

/// ImageNet ImageFolder（真实数据）分类训练：整集预解码到 CPU（img_size=64 时
/// ImageNette 全集约 0.6GB 内存，可接受）→ 每 epoch XorShift 洗牌 → 分批搬模型
/// 设备训练（CE 损失、梯度累加/裁剪沿用既有骨架）→ val split top1 评测。
fn train_classify_imagenette(
    cfg: &RunConfig,
    run_id: &str,
    run_dir: &Path,
) -> AvResult<TrainReport> {
    let (num_classes, img_size) = match cfg.model.tasks.first() {
        Some(TaskCfg::Classify(c)) => (c.num_classes as u32, c.img_size),
        _ => unreachable!("train_classify 只处理分类任务"),
    };
    let device = resolve_device(cfg);
    let root = classify_data_root(cfg)?;
    let train_split = cfg.data.sources.train.split.as_deref().unwrap_or("train");
    // 预解码固定 CPU（与检测侧同策略），训练批 stack 后再搬到模型设备
    let (train, train_labels, class_map) =
        dataset::load_imagefolder(&root, train_split, img_size, true, Device::Cpu, imagenet_norm(cfg))?;
    if class_map.len() != num_classes as usize {
        return Err(AvError::config(format!(
            "classify.num_classes = {num_classes} 与 ImageFolder 目录类别数 {} 不一致",
            class_map.len()
        )));
    }
    // val 必须复用 train 的 wnid→类id 映射（val 缺类时按目录排序推导会错位）
    let (val_root, val_split) = classify_val_source(cfg);
    let (val, val_labels, _) = dataset::load_imagefolder_with_classes(
        &val_root,
        &val_split,
        img_size,
        Some(&class_map),
        Device::Cpu,
        imagenet_norm(cfg),
    )?;
    println!(
        "[classify-imagenette] root={} train={} val={} classes={} img_size={}",
        root.display(),
        train.len(),
        val.len(),
        class_map.len(),
        img_size
    );

    let vs = VarStore::new(device);
    let model = build_model(&vs.root(), cfg)?;
    apply_pretrain(&vs, cfg)?;
    let mut opt = make_opt(&vs, cfg)?;
    let mut rng = XorShift::new(cfg.seed);
    let bs = (cfg.train.batch_size as usize).max(1);
    let mut final_loss = 0f32;
    let mut acc = 0f32;
    // BN train/eval 装配（resnet18 骨干生效；冻结骨干时恒 eval）
    let bn_train = bn_train_mode(cfg);

    for epoch in 1..=cfg.train.epochs {
        model.set_train(bn_train);
        opt.set_lr(schedule_lr(cfg, epoch));
        // 每 epoch 全集洗牌（Fisher-Yates，XorShift 确定性）
        let mut order: Vec<usize> = (0..train.len()).collect();
        for i in (1..order.len()).rev() {
            let j = rng.next_usize(i + 1);
            order.swap(i, j);
        }
        let mut epoch_loss = 0f32;
        let mut steps = 0usize;
        let acc_steps = cfg.train.accumulate_steps.max(1) as usize;
        opt.zero_grad();
        for chunk in order.chunks(bs) {
            let batch: Vec<ClassifySample> = chunk.iter().map(|&i| train[i].clone()).collect();
            let x = dataset::stack_classify(&batch)?.to_device(device);
            let y: Vec<i64> = chunk.iter().map(|&i| train_labels[i] as i64).collect();
            let labels = Tensor::from_slice(&y).to_device(device);
            let loss = model.loss(&x, &TrainBatch::Classify { labels })?;
            loss.backward();
            epoch_loss += loss.double_value(&[]) as f32;
            // 梯度累加（PLAN §5.2）：每 acc_steps 个 micro-step 才步进一次
            steps += 1;
            if steps % acc_steps == 0 {
                if cfg.train.grad_clip > 0.0 {
                    opt.clip_grad_norm(cfg.train.grad_clip as f64);
                }
                opt.step();
                opt.zero_grad();
            }
        }
        if steps % acc_steps != 0 {
            // 尾部不足 acc_steps 的梯度也要落一次步进
            if cfg.train.grad_clip > 0.0 {
                opt.clip_grad_norm(cfg.train.grad_clip as f64);
            }
            opt.step();
            opt.zero_grad();
        }
        final_loss = epoch_loss / steps.max(1) as f32;
        // 评测按间隔执行（val 全量推理在大验证集时不可忽略）
        let eval_due = epoch == 1
            || epoch == cfg.train.epochs
            || cfg.eval.interval_epochs == 0
            || epoch % cfg.eval.interval_epochs == 0;
        if eval_due {
            model.set_train(false); // 评测 = BN 推理语义（running 统计量）
            acc = eval_classify_samples(&model, &val, &val_labels, device)?;
            model.set_train(bn_train); // 恢复训练态
            println!(
                "[classify-imagenette] run={run_id} epoch={epoch}/{} loss={final_loss:.4} top1={acc:.3}",
                cfg.train.epochs
            );
        }
        log_epoch_metrics(run_dir, epoch, final_loss, "top1", acc, None);
    }

    save_checkpoint(&vs, &run_dir.join(CKPT_DIR))?;
    Ok(TrainReport {
        run_id: run_id.to_string(),
        task: "classify".into(),
        epochs: cfg.train.epochs,
        final_loss,
        metric: "top1".into(),
        metric_value: acc,
        secondary: None,
        run_dir: run_dir.display().to_string(),
    })
}

/// 合成数据分类冒烟训练（开箱即训，管线连通性验证；真实精度看 dir 源）。
fn train_classify_synthetic(
    cfg: &RunConfig,
    run_id: &str,
    run_dir: &Path,
) -> AvResult<TrainReport> {
    let task_cfg = match cfg.model.tasks.first() {
        Some(TaskCfg::Classify(c)) => (c.num_classes as u32, c.img_size),
        _ => unreachable!("train_classify 只处理分类任务"),
    };
    let (num_classes, img_size) = task_cfg;
    let device = resolve_device(cfg);
    let vs = VarStore::new(device);
    let model = build_model(&vs.root(), cfg)?;
    apply_pretrain(&vs, cfg)?;
    let mut opt = make_opt(&vs, cfg)?;
    let mut rng = XorShift::new(cfg.seed);
    let bs = cfg.train.batch_size as i64;
    let mut final_loss = 0f32;
    let mut acc = 0f32;
    // BN train/eval 装配（resnet18 骨干生效；冻结骨干时恒 eval）
    let bn_train = bn_train_mode(cfg);

    for epoch in 1..=cfg.train.epochs {
        model.set_train(bn_train);
        opt.set_lr(schedule_lr(cfg, epoch));
        let mut epoch_loss = 0f32;
        let acc_steps = cfg.train.accumulate_steps.max(1) as usize;
        opt.zero_grad();
        for si in 0..STEPS_PER_EPOCH {
            let (x, y, _) = synthetic_classify(&mut rng, bs, num_classes, img_size, device);
            let loss = model.loss(&x, &TrainBatch::Classify { labels: y })?;
            loss.backward();
            epoch_loss += loss.double_value(&[]) as f32;
            // 梯度累加（PLAN §5.2）：每 acc_steps 个 micro-step 才步进一次
            if (si + 1) % acc_steps == 0 {
                if cfg.train.grad_clip > 0.0 {
                    opt.clip_grad_norm(cfg.train.grad_clip as f64);
                }
                opt.step();
                opt.zero_grad();
            }
        }
        if STEPS_PER_EPOCH % acc_steps != 0 {
            // 尾部不足 acc_steps 的梯度也要落一次步进
            if cfg.train.grad_clip > 0.0 {
                opt.clip_grad_norm(cfg.train.grad_clip as f64);
            }
            opt.step();
            opt.zero_grad();
        }
        final_loss = epoch_loss / STEPS_PER_EPOCH as f32;
        model.set_train(false); // 评测 = BN 推理语义
        acc = eval_classify(&model, num_classes, img_size, device)?;
        model.set_train(bn_train); // 恢复训练态
        println!(
            "[classify-smoke] run={run_id} epoch={epoch}/{} loss={final_loss:.4} top1={acc:.3}",
            cfg.train.epochs
        );
        log_epoch_metrics(run_dir, epoch, final_loss, "top1", acc, None);
    }

    save_checkpoint(&vs, &run_dir.join(CKPT_DIR))?;
    Ok(TrainReport {
        run_id: run_id.to_string(),
        task: "classify".into(),
        epochs: cfg.train.epochs,
        final_loss,
        metric: "top1".into(),
        metric_value: acc,
        secondary: None,
        run_dir: run_dir.display().to_string(),
    })
}

fn train_detect_synthetic(
    cfg: &RunConfig,
    run_id: &str,
    run_dir: &Path,
) -> AvResult<TrainReport> {
    let (num_classes, img_size) = match cfg.model.tasks.first() {
        Some(TaskCfg::Detect(d)) => (d.num_classes as u32, d.img_size),
        _ => unreachable!("train_detect 只处理检测任务"),
    };
    let device = resolve_device(cfg);
    let vs = VarStore::new(device);
    let model = build_model(&vs.root(), cfg)?;
    apply_pretrain(&vs, cfg)?;
    let mut opt = make_opt(&vs, cfg)?;
    let mut rng = XorShift::new(cfg.seed);
    let bs = cfg.train.batch_size as i64;
    let mut final_loss = 0f32;
    let mut miou = 0f32;
    let mut r50 = 0f32;
    #[allow(unused_assignments)]
    let mut det_stats = 0f32;
    #[allow(unused_assignments)]
    let mut dbg;
    // BN train/eval 装配（resnet18 骨干生效；冻结骨干时恒 eval）
    let bn_train = bn_train_mode(cfg);

    for epoch in 1..=cfg.train.epochs {
        model.set_train(bn_train);
        opt.set_lr(schedule_lr(cfg, epoch));
        let mut epoch_loss = 0f32;
        let acc_steps = cfg.train.accumulate_steps.max(1) as usize;
        opt.zero_grad();
        for si in 0..STEPS_PER_EPOCH {
            let (x, boxes, labels) =
                synthetic_detect(&mut rng, bs, num_classes, img_size, device);
            let batch = TrainBatch::Detect {
                boxes: boxes.iter().map(|b| vec![*b]).collect(),
                labels: labels.iter().map(|&l| vec![l]).collect(),
            };
            if epoch % 10 == 0 {
                av_tasks::models::LOSS_DEBUG.with(|d| *d.borrow_mut() = Some(String::new()));
            }
            let loss = model.loss(&x, &batch)?;
            loss.backward();
            epoch_loss += loss.double_value(&[]) as f32;
            if (si + 1) % acc_steps == 0 {
                if cfg.train.grad_clip > 0.0 {
                    opt.clip_grad_norm(cfg.train.grad_clip as f64);
                }
                opt.step();
                opt.zero_grad();
            }
        }
        final_loss = epoch_loss / STEPS_PER_EPOCH as f32;
        model.set_train(false); // 评测 = BN 推理语义
        (miou, r50, det_stats, dbg) = eval_detect(&model, num_classes, img_size, device)?;
        model.set_train(bn_train); // 恢复训练态
        if epoch % 10 == 0 || epoch == cfg.train.epochs {
            println!(
                "[detect-smoke] run={run_id} epoch={epoch}/{} loss={final_loss:.4} mIoU={miou:.3} R@0.5={r50:.3} dets/图={det_stats:.1} {dbg}",
                cfg.train.epochs
            );
        }
        log_epoch_metrics(
            run_dir,
            epoch,
            final_loss,
            "mean_iou",
            miou,
            Some(("recall@0.5", r50)),
        );
    }

    save_checkpoint(&vs, &run_dir.join(CKPT_DIR))?;
    Ok(TrainReport {
        run_id: run_id.to_string(),
        task: "detect".into(),
        epochs: cfg.train.epochs,
        final_loss,
        metric: "mean_iou".into(),
        metric_value: miou,
        secondary: Some(("recall@0.5".into(), r50)),
        run_dir: run_dir.display().to_string(),
    })
}

/// YOLO 目录数据源训练：整集预解码（CPU）→ 每 epoch 洗牌 → 分批搬到模型设备训练
/// → val 冒烟指标。
fn train_detect_yolo(
    cfg: &RunConfig,
    run_id: &str,
    run_dir: &Path,
) -> AvResult<TrainReport> {
    let (num_classes, img_size) = match cfg.model.tasks.first() {
        Some(TaskCfg::Detect(d)) => (d.num_classes as u32, d.img_size),
        _ => unreachable!("train_detect_yolo 只处理检测任务"),
    };
    let device = resolve_device(cfg);
    // 数据源：dir 管线用 YOLO 目录根；avpack 管线用 `.avpack` 容器路径
    // （容器名约定 = 打包时的相对路径，加载器按 images/<split> + labels/<split> 过滤）
    let (root, pack) = match cfg.data.pipeline {
        DataPipeline::AvPack => {
            let p = cfg
                .data
                .sources
                .train
                .avpack
                .clone()
                .ok_or_else(|| AvError::config("avpack 数据源缺 data.sources.train.avpack"))?;
            (p.clone(), Some(p))
        }
        _ => {
            let d = cfg
                .data
                .sources
                .train
                .dir
                .clone()
                .ok_or_else(|| AvError::config("dir 数据源缺 data.sources.train.dir"))?;
            (d, None)
        }
    };
    let train_split = cfg.data.sources.train.split.as_deref().unwrap_or("train");
    // 数据集预解码固定在 CPU（预解码成本在图像管线，与模型设备解耦），
    // 训练批 stack 后再 .to_device(model_device)，见下方训练循环。
    // 增强就绪 → 训练侧走「raw 样本 + 逐 epoch 随机编码」，val 保持 plain。
    let aug_cfg = train_augment_cfg(cfg, TaskKind::Detect);
    let train_raw = if let Some(a) = &aug_cfg {
        println!(
            "[augment] detect 训练增强生效: mosaic={} mixup={} flip={} hsv={:?} scale_jitter={:?} close_last_epochs={}",
            a.mosaic, a.mixup, a.flip, a.hsv, a.scale_jitter, a.close_last_epochs
        );
        println!(
            "[augment] detect 串联顺序: mosaic → mixup → flip → hsv → scale（mixup 仅检测/分类语义，关键点不适用）"
        );
        Some(match &pack {
            Some(p) => dataset::load_yolo_avpack_raw(p, train_split)?,
            None => dataset::load_yolo_dir_raw(&root, train_split)?,
        })
    } else {
        None
    };
    let train = if train_raw.is_some() {
        Vec::new()
    } else {
        match &pack {
            Some(p) => dataset::load_yolo_avpack(p, train_split, img_size, Device::Cpu, imagenet_norm(cfg))?,
            None => dataset::load_yolo_dir(&root, train_split, img_size, Device::Cpu, imagenet_norm(cfg))?,
        }
    };
    let val = if cfg.data.pipeline == DataPipeline::AvPack {
        let val_split = cfg.data.sources.val.split.as_deref().unwrap_or("val");
        match cfg.data.sources.val.avpack.as_ref() {
            Some(p) => dataset::load_yolo_avpack(p, val_split, img_size, Device::Cpu, imagenet_norm(cfg))?,
            // val 源缺省：同一容器的 val split
            None => dataset::load_yolo_avpack(&root, val_split, img_size, Device::Cpu, imagenet_norm(cfg))?,
        }
    } else {
        match cfg.data.sources.val.dir.as_ref() {
            Some(d) => dataset::load_yolo_dir(
                d,
                cfg.data.sources.val.split.as_deref().unwrap_or("val"),
                img_size,
                Device::Cpu,
                imagenet_norm(cfg),
            )?,
            // val 源缺省：同一数据集根下的 val split
            None => dataset::load_yolo_dir(&root, "val", img_size, Device::Cpu, imagenet_norm(cfg))?,
        }
    };
    let train_n = train_raw.as_ref().map_or(train.len(), |r| r.len());
    println!(
        "[yolo] {}={} train={} val={} classes={} img_size={}",
        if pack.is_some() { "avpack" } else { "root" },
        root.display(),
        train_n,
        val.len(),
        num_classes,
        img_size
    );

    let vs = VarStore::new(device);
    let model = build_model(&vs.root(), cfg)?;
    apply_pretrain(&vs, cfg)?;
    let mut opt = make_opt(&vs, cfg)?;
    let mut rng = XorShift::new(cfg.seed);
    let bs = (cfg.train.batch_size as usize).max(1);
    let mut final_loss = 0f32;
    let mut miou = 0f32;
    let mut r50 = 0f32;
    // BN train/eval 装配（resnet18 骨干生效；冻结骨干时恒 eval）
    let bn_train = bn_train_mode(cfg);

    for epoch in 1..=cfg.train.epochs {
        model.set_train(bn_train);
        opt.set_lr(schedule_lr(cfg, epoch));
        let mut order: Vec<usize> = (0..train_n).collect();
        for i in (1..order.len()).rev() {
            let j = rng.next_usize(i + 1);
            order.swap(i, j);
        }
        let strong_on = aug_cfg
            .as_ref()
            .map(|a| strong_aug_on(a, cfg, epoch))
            .unwrap_or(false);
        let mut aug_rng = epoch_aug_rng(cfg.seed, epoch);
        let mut epoch_loss = 0f32;
        let mut steps = 0usize;
        let acc_steps = cfg.train.accumulate_steps.max(1) as usize;
        opt.zero_grad();
        for chunk in order.chunks(bs) {
            let batch_samples: Vec<SampleTensor> = chunk
                .iter()
                .map(|&i| -> AvResult<SampleTensor> {
                    match (&train_raw, &aug_cfg) {
                        (Some(raw), Some(a)) => {
                            // 组合增强（数据增强官二波）：抽签顺序固定
                            // mosaic 硬币 → mixup 硬币+λ → draw_plan(flip→scale→gains)，
                            // 与像素/坐标串联顺序 mosaic → mixup → flip → hsv → scale 对应。
                            // mosaic/mixup 概率全 0 时零消耗 RNG，行为与历史版本逐位一致。
                            let comp = if strong_on {
                                augment::draw_composite(a, &mut aug_rng)
                            } else {
                                augment::CompositeDraw::none()
                            };
                            let mosaic_img = if comp.mosaic {
                                // 4 图组 batch：锚点 + 3 个随机重复采样填充的伙伴
                                let j1 = aug_rng.next_usize(raw.len());
                                let j2 = aug_rng.next_usize(raw.len());
                                let j3 = aug_rng.next_usize(raw.len());
                                Some(dataset::mosaic4_raw([
                                    &raw[i], &raw[j1], &raw[j2], &raw[j3],
                                ])?)
                            } else {
                                None
                            };
                            let mix_img = if comp.mixup {
                                // 双样本融合（标签双份并集）：伙伴也随机重复采样
                                let k = aug_rng.next_usize(raw.len());
                                let base = mosaic_img.as_ref().unwrap_or(&raw[i]);
                                Some(dataset::mixup_raw(base, &raw[k], comp.mixup_lam)?)
                            } else {
                                None
                            };
                            let plan = if strong_on {
                                augment::draw_plan(a, &mut aug_rng)
                            } else {
                                AugmentPlan::none()
                            };
                            let composed: &dataset::RawDetectSample = mix_img
                                .as_ref()
                                .or(mosaic_img.as_ref())
                                .unwrap_or(&raw[i]);
                            dataset::encode_detect_sample(
                                composed,
                                img_size,
                                Device::Cpu,
                                dataset::ResizeMode::Letterbox,
                                &plan,
                                imagenet_norm(cfg),
                            )
                        }
                        _ => Ok(train[i].clone()),
                    }
                })
                .collect::<AvResult<Vec<_>>>()?;
            // 样本张量在 CPU 预解码，进模型前整批搬到模型设备
            let x = dataset::stack_samples(&batch_samples)?.to_device(device);
            let batch = TrainBatch::Detect {
                boxes: batch_samples.iter().map(|s| s.boxes.clone()).collect(),
                labels: batch_samples.iter().map(|s| s.labels.clone()).collect(),
            };
            let loss = model.loss(&x, &batch)?;
            loss.backward();
            epoch_loss += loss.double_value(&[]) as f32;
            steps += 1;
            if steps % acc_steps == 0 {
                if cfg.train.grad_clip > 0.0 {
                    opt.clip_grad_norm(cfg.train.grad_clip as f64);
                }
                opt.step();
                opt.zero_grad();
            }
        }
        if steps % acc_steps != 0 {
            if cfg.train.grad_clip > 0.0 {
                opt.clip_grad_norm(cfg.train.grad_clip as f64);
            }
            opt.step();
            opt.zero_grad();
        }
        final_loss = epoch_loss / steps.max(1) as f32;
        let TaskModel::Detect(m) = &model else {
            unreachable!("检测任务模型类型")
        };
        // 评测按间隔执行（大验证集时每 epoch 全量评测会主导耗时）
        let eval_due = epoch == 1
            || epoch == cfg.train.epochs
            || cfg.eval.interval_epochs == 0
            || epoch % cfg.eval.interval_epochs == 0;
        if eval_due {
            model.set_train(false); // 评测 = BN 推理语义（running 统计量）
            let evm = eval_detect_samples(m, &val, device)?;
            model.set_train(bn_train); // 恢复训练态
            (miou, r50) = (evm.miou, evm.r50);
            println!(
                "[detect-yolo] run={run_id} epoch={epoch}/{} loss={final_loss:.4} mIoU={miou:.3} R@0.5={r50:.3} mAP50={:.3} mAP50:95={:.3}",
                cfg.train.epochs, evm.map50, evm.map50_95
            );
        }
        log_epoch_metrics(
            run_dir,
            epoch,
            final_loss,
            "mean_iou",
            miou,
            Some(("recall@0.5", r50)),
        );
    }

    save_checkpoint(&vs, &run_dir.join(CKPT_DIR))?;
    Ok(TrainReport {
        run_id: run_id.to_string(),
        task: "detect".into(),
        epochs: cfg.train.epochs,
        final_loss,
        metric: "mean_iou".into(),
        metric_value: miou,
        secondary: Some(("recall@0.5".into(), r50)),
        run_dir: run_dir.display().to_string(),
    })
}

/// 检测评测指标束：冒烟指标（best-IoU 均值 / R@0.5）+ COCO 风格 mAP。
#[derive(Debug, Clone)]
struct DetectEvalMetrics {
    miou: f32,
    r50: f32,
    map50: f32,
    map50_95: f32,
    // 诊断字段：当前调用方暂未消费，保留给后续日志/报告扩展
    #[allow(dead_code)]
    dets_per_img: f32,
    #[allow(dead_code)]
    dbg: String,
}

/// 在（预解码）样本集上计算检测指标：冒烟 best-IoU / R@0.5 + COCO 风格 mAP
/// （预测与 gt 同处输入画布像素空间，直接可比）。
fn eval_detect_samples(
    m: &av_tasks::models::DetectModel,
    val: &[SampleTensor],
    device: Device,
) -> AvResult<DetectEvalMetrics> {
    let n = val.len();
    if n == 0 {
        return Err(AvError::data("验证集为空"));
    }
    let mut per_image: Vec<Vec<av_core::types::Detection>> = Vec::new();
    for chunk in val.chunks(8) {
        // 预解码样本在 CPU，推理批内搬到模型设备
        let x = dataset::stack_samples(chunk)?.to_device(device);
        per_image.extend(m.predict(&x, 0.1, 0.5)?);
    }
    // 指标定义（v0.1 冒烟）：best_ious = 无类过滤定位 IoU（纯定位能力）；
    // class_ok/R@0.5 = 类别也正确才计（定位+分类联合）。真实 mAP 协议由 eval_map 接管。
    let mut best_ious = vec![0f32; n];
    let mut class_ok = vec![false; n];
    for (gi, s) in val.iter().enumerate() {
        for d in &per_image[gi] {
            let mut matched = false;
            for (b, &label) in s.boxes.iter().zip(&s.labels) {
                let g = Aabb::new(b[0], b[1], b[2], b[3]);
                best_ious[gi] = best_ious[gi].max(g.iou(&d.bbox));
                if d.class_id == label {
                    matched = true;
                }
            }
            if matched {
                class_ok[gi] = true;
            }
        }
    }
    let matched_cnt = best_ious.iter().filter(|&&v| v >= 0.5).count();
    let total_dets: usize = per_image.iter().map(|d| d.len()).sum();
    let mut parts = vec![format!(
        "类对率={:.2}",
        class_ok.iter().filter(|&&v| v).count() as f32 / n as f32
    )];
    for gi in 0..n.min(3) {
        parts.push(format!(
            "图{gi} best_iou={:.2} dets={}",
            best_ious[gi],
            per_image[gi].len()
        ));
    }
    // COCO 风格 mAP（eval_map 协议）：逐图喂入预测与 gt（同一画布像素空间）
    let mut ev = CocoEvaluator::new();
    for (gi, s) in val.iter().enumerate() {
        let gts: Vec<GtBox> = s
            .boxes
            .iter()
            .zip(&s.labels)
            .map(|(b, &l)| GtBox::new(Aabb::new(b[0], b[1], b[2], b[3]), l))
            .collect();
        ev.update(gi as u32, &per_image[gi], &gts);
    }
    let map = ev.finalize();
    Ok(DetectEvalMetrics {
        miou: best_ious.iter().sum::<f32>() / n as f32,
        r50: matched_cnt as f32 / n as f32,
        map50: map.map50,
        map50_95: map.map50_95,
        dets_per_img: total_dets as f32 / n as f32,
        dbg: parts.join(" | "),
    })
}

fn eval_classify(model: &TaskModel, num_classes: u32, img_size: u32, device: Device) -> AvResult<f32> {
    let mut rng = XorShift::new(0x5EED_0001);
    // 合成评测集直接生成在模型设备上，与模型权重同侧
    let (x, _, labels) =
        synthetic_classify(&mut rng, EVAL_BATCH, num_classes, img_size, device);
    let TaskModel::Classify(m) = model else {
        return Err(AvError::train("模型与任务不匹配"));
    };
    let (pred, _) = m.predict(&x)?;
    Ok(pred
        .iter()
        .zip(&labels)
        .filter(|(a, b)| a == b)
        .count() as f32
        / labels.len() as f32)
}

/// 在（预解码）分类样本集上算 top1：按 EVAL_BATCH 分批推理，
/// 预解码样本在 CPU，批内搬到模型设备（与检测评测同策略）。
fn eval_classify_samples(
    model: &TaskModel,
    val: &[ClassifySample],
    labels: &[u32],
    device: Device,
) -> AvResult<f32> {
    let n = val.len();
    if n == 0 || labels.len() != n {
        return Err(AvError::data("分类验证集为空或标签数不匹配"));
    }
    let TaskModel::Classify(m) = model else {
        return Err(AvError::train("模型与任务不匹配"));
    };
    let mut correct = 0usize;
    for (ci, chunk) in val.chunks(EVAL_BATCH as usize).enumerate() {
        let start = ci * EVAL_BATCH as usize;
        let x = dataset::stack_classify(chunk)?.to_device(device);
        let (pred, _) = m.predict(&x)?;
        correct += pred
            .iter()
            .zip(&labels[start..start + chunk.len()])
            .filter(|(a, b)| a == b)
            .count();
    }
    Ok(correct as f32 / n as f32)
}

fn eval_detect(
    model: &TaskModel,
    num_classes: u32,
    img_size: u32,
    device: Device,
) -> AvResult<(f32, f32, f32, String)> {
    let mut rng = XorShift::new(0x5EED_0002);
    let n = 64usize;
    // 合成评测集直接生成在模型设备上，与模型权重同侧
    let (x, boxes, labels) =
        synthetic_detect(&mut rng, n as i64, num_classes, img_size, device);
    let TaskModel::Detect(m) = model else {
        return Err(AvError::train("模型与任务不匹配"));
    };
    let per_image = m.predict(&x, 0.25, 0.5)?;
    let total_dets: usize = per_image.iter().map(|d| d.len()).sum();
    let mut best_ious = vec![0f32; n];
    let mut class_ok = vec![false; n];
    // 注意：boxes 存的是 xyxy，必须用 Aabb::new 而不是 from_xywh
    let gt_boxes: Vec<Aabb> = boxes
        .iter()
        .map(|b| Aabb::new(b[0], b[1], b[2], b[3]))
        .collect();
    for (gi, g) in gt_boxes.iter().enumerate() {
        for d in &per_image[gi] {
            if d.class_id == labels[gi] {
                class_ok[gi] = true;
            }
            if d.class_id != labels[gi] {
                continue;
            }
            best_ious[gi] = best_ious[gi].max(g.iou(&d.bbox));
        }
    }
    let matched = best_ious.iter().filter(|&&v| v >= 0.5).count();
    let dbg = {
        let mut parts = vec![format!(
            "类对率={:.2} 无类过滤mIoU={:.2}",
            class_ok.iter().filter(|&&v| v).count() as f32 / n as f32,
            {
                let mut best_all = vec![0f32; n];
                for (gi, g) in gt_boxes.iter().enumerate() {
                    for d in &per_image[gi] {
                        best_all[gi] = best_all[gi].max(g.iou(&d.bbox));
                    }
                }
                best_all.iter().sum::<f32>() / n as f32
            }
        )];
        for gi in 0..4 {
            let top = per_image[gi].iter().max_by(|a, b| {
                a.score
                    .partial_cmp(&b.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            match top {
                Some(d) => parts.push(format!(
                    "图{gi} pred=({:.0},{:.0},{:.0},{:.0}) c{} | gt=({:.0},{:.0},{:.0},{:.0}) c{} iou={:.2}",
                    d.bbox.x1,
                    d.bbox.y1,
                    d.bbox.x2,
                    d.bbox.y2,
                    d.class_id,
                    boxes[gi][0],
                    boxes[gi][1],
                    boxes[gi][2],
                    boxes[gi][3],
                    labels[gi],
                    gt_boxes[gi].iou(&d.bbox)
                )),
                None => parts.push(format!("图{gi} 无检出")),
            }
        }
        parts.join(" | ")
    };
    Ok((
        best_ious.iter().sum::<f32>() / n as f32,
        matched as f32 / n as f32,
        total_dets as f32 / n as f32,
        dbg,
    ))
}

// ---------------------------------------------------------------------------
// 合成数据源（开箱即训；avpack/dir 解析按 M2/M7 落地）
// ---------------------------------------------------------------------------

/// 分类合成：噪声底 + 类相关亮块位置（类别 → 确定性位置/通道），可学习性有保证。
fn synthetic_classify(
    rng: &mut XorShift,
    n: i64,
    num_classes: u32,
    img: u32,
    device: Device,
) -> (Tensor, Tensor, Vec<u32>) {
    let s = img as usize;
    let n = n as usize;
    let mut buf = vec![0f32; n * 3 * s * s];
    let mut labels = vec![0u32; n];
    for ni in 0..n {
        let c = rng.next_usize(num_classes as usize);
        labels[ni] = c as u32;
        let base = ni * 3 * s * s;
        for v in buf[base..base + 3 * s * s].iter_mut() {
            *v = rng.next_f32() * 0.3;
        }
        // 类 c 的确定性地标：通道 c%3 上 (by, bx) 处 16x16 亮块
        let ch = ni * 3 * s * s + (c % 3) * s * s;
        let bx = (c * 13 + 7) % (s - 16);
        let by = (c * 29 + 3) % (s - 16);
        for yy in by..by + 16 {
            for xx in bx..bx + 16 {
                buf[ch + yy * s + xx] += 1.5;
            }
        }
    }
    let y: Vec<i64> = labels.iter().map(|&v| v as i64).collect();
    let x = Tensor::from_slice(&buf)
        .to_kind(Kind::Float)
        .clamp(0.0, 1.0) // 与真实图片 [0,1] 域对齐
        .to_device(device)
        .reshape([n as i64, 3, s as i64, s as i64]);
    let y = Tensor::from_slice(&y).to_device(device);
    (x, y, labels)
}

/// 检测合成：噪声底 + 通道 0 上的随机亮方块，标签为方块 xyxy。
fn synthetic_detect(
    rng: &mut XorShift,
    n: i64,
    num_classes: u32,
    img: u32,
    device: Device,
) -> (Tensor, Vec<[f32; 4]>, Vec<u32>) {
    let s = img as usize;
    let n = n as usize;
    let mut buf = vec![0f32; n * 3 * s * s];
    let mut boxes = Vec::with_capacity(n);
    let mut labels = Vec::with_capacity(n);
    for ni in 0..n {
        for v in buf[ni * 3 * s * s..(ni + 1) * 3 * s * s].iter_mut() {
            *v = rng.next_f32() * 0.3;
        }
        let sw = rng.next_range(0.18, 0.4) * s as f32;
        let sh = rng.next_range(0.18, 0.4) * s as f32;
        let cx = rng.next_range(sw / 2.0 + 1.0, s as f32 - sw / 2.0 - 1.0);
        let cy = rng.next_range(sh / 2.0 + 1.0, s as f32 - sh / 2.0 - 1.0);
        let (x1, y1, x2, y2) = (
            (cx - sw / 2.0).round(),
            (cy - sh / 2.0).round(),
            (cx + sw / 2.0).round(),
            (cy + sh / 2.0).round(),
        );
        labels.push(rng.next_usize(num_classes as usize) as u32);
        boxes.push([x1, y1, x2, y2]);
        // 类别决定方块所在通道（label%3）：给模型可学的类别信号
        let ch_off = ni * 3 * s * s + (labels[ni] as usize % 3) * s * s;
        for yy in (y1 as usize)..(y2 as usize).min(s) {
            for xx in (x1 as usize)..(x2 as usize).min(s) {
                buf[ch_off + yy * s + xx] += 1.5;
            }
        }
    }
    let x = Tensor::from_slice(&buf)
        .to_kind(Kind::Float)
        .clamp(0.0, 1.0) // 与真实图片 [0,1] 域对齐
        .to_device(device)
        .reshape([n as i64, 3, s as i64, s as i64]);
    (x, boxes, labels)
}

// ---- 集成测试桥接（av-runtime::testing） ----

pub(crate) fn testing_load_model(cfg: &RunConfig, weights: &Path) -> AvResult<TaskModel> {
    load_model(cfg, weights).map(|(model, _)| model)
}

pub(crate) fn testing_synthetic_detect(
    rng: &mut XorShift,
    n: i64,
    num_classes: u32,
    img_size: u32,
) -> (Tensor, Vec<[f32; 4]>, Vec<u32>) {
    synthetic_detect(rng, n, num_classes, img_size, Device::Cpu)
}

pub(crate) fn testing_eval_kp_samples(
    m: &KeypointModel,
    samples: &[KeypointSample],
) -> AvResult<(f32, f32, usize, usize)> {
    eval_kp_samples(m, samples, Device::Cpu)
}

// ---------------------------------------------------------------------------
// 切片碎片合并单测（手算对照见各断言注释）
// ---------------------------------------------------------------------------

#[cfg(test)]
mod fragment_merge_tests {
    use super::merge_tile_fragments;
    use av_core::geometry::Aabb;

    fn det(x1: f32, y1: f32, x2: f32, y2: f32, score: f32, class_id: u32) -> av_core::types::Detection {
        av_core::types::Detection {
            bbox: Aabb::new(x1, y1, x2, y2),
            score,
            class_id,
            angle: None,
            keypoints: None,
        }
    }

    /// 手算：a=(0,0,40,40) s0.9 与 b=(20,0,60,40) s0.6 同类——
    /// inter=(20..40)×(0..40)=800，union=1600+1600-800=2400，IoU=1/3≈0.333≥0.3；
    /// 中心 (20,20) vs (40,20) 距 20 < 25（窗口 100 的 1/4）⇒ 合并，
    /// 代表 a，并集框 (0,0,60,40)，分数 0.9。
    #[test]
    fn two_overlapping_fragments_merge_into_one() {
        let out = merge_tile_fragments(
            vec![det(0.0, 0.0, 40.0, 40.0, 0.9, 0), det(20.0, 0.0, 60.0, 40.0, 0.6, 0)],
            0.3,
            25.0,
        );
        assert_eq!(out.len(), 1, "两个重叠碎片必须合并为一个");
        assert!((out[0].score - 0.9).abs() < 1e-6, "代表取分数最高者");
        let b = &out[0].bbox;
        assert!((b.x1 - 0.0).abs() < 1e-6 && (b.y1 - 0.0).abs() < 1e-6);
        assert!((b.x2 - 60.0).abs() < 1e-6 && (b.y2 - 40.0).abs() < 1e-6, "并集框 {:?}", b);
    }

    /// 手算对照：IoU 与中心距须同时满足，且只合并同类。
    /// - c=(0,0,300,300) s0.8 与 d=(100,100,300,300) s0.7：inter=200²=40000，
    ///   union=90000+40000-40000=90000，IoU=0.444≥0.3，但中心 (150,150) vs
    ///   (200,200) 距 √2·50≈70.7 ≥ 25 ⇒ 大目标与其重叠检出**不得**被误并；
    /// - f/g 框几何与 a/b 完全同构（IoU 0.333、中心距 20）但类别不同 ⇒ 不并；
    /// - e 远处框孤立保留。
    #[test]
    fn merge_requires_iou_and_distance_and_same_class() {
        let dets = vec![
            det(20.0, 0.0, 60.0, 40.0, 0.6, 0),     // b（应并入 a）
            det(100.0, 100.0, 300.0, 300.0, 0.7, 0), // d
            det(0.0, 0.0, 40.0, 40.0, 0.9, 0),       // a（代表）
            det(0.0, 0.0, 300.0, 300.0, 0.8, 0),     // c
            det(1000.0, 1000.0, 1020.0, 1020.0, 0.5, 0), // e
            det(2000.0, 2000.0, 2040.0, 2040.0, 0.9, 1), // f（类 1）
            det(2020.0, 2000.0, 2060.0, 2040.0, 0.6, 0), // g（类 0，与 f 不同类）
        ];
        let out = merge_tile_fragments(dets, 0.3, 25.0);
        // 7 入，仅 a+b 合并一次 → 6 出（a', f, c, d, g, e）
        assert_eq!(out.len(), 6, "只有 a+b 合并，得 {:?}", out);
        let a = out
            .iter()
            .find(|k| (k.score - 0.9).abs() < 1e-6 && k.class_id == 0)
            .expect("代表框必须保留");
        assert!((a.bbox.x2 - 60.0).abs() < 1e-6 && (a.bbox.y2 - 40.0).abs() < 1e-6);
        // c 与 d（IoU 0.444 但中心距超限）都活着
        assert!(out.iter().any(|k| (k.score - 0.8).abs() < 1e-6));
        assert!(out.iter().any(|k| (k.score - 0.7).abs() < 1e-6));
        // f/g（同几何不同类）都活着
        assert_eq!(out.iter().filter(|k| (k.score - 0.9).abs() < 1e-6).count(), 2);
        assert!(out.iter().any(|k| (k.score - 0.6).abs() < 1e-6 && k.class_id == 0));
        assert!(out.iter().any(|k| (k.score - 0.5).abs() < 1e-6));
    }

    /// 并集后的代表框作为簇首参与后续比较：c 与原 a 的 IoU 为 0，
    /// 与长大后 a'=(0,0,60,40) 的 IoU = 800/3200 = 0.25 < 0.3（且中心距
    /// (60,20)→(30,20)=30 ≥ 25 双重不满足）⇒ 三个碎片收敛为两框。
    #[test]
    fn chain_merge_absorbs_via_grown_representative() {
        let a = det(0.0, 0.0, 40.0, 40.0, 0.9, 0);
        let b = det(20.0, 0.0, 60.0, 40.0, 0.6, 0); // 与 a：IoU 1/3，距 20 → 并
        let c = det(40.0, 0.0, 80.0, 40.0, 0.5, 0); // 与 a'：IoU 0.25 < 0.3，距 30 ≥ 25 → 不并
        let out = merge_tile_fragments(vec![a, b, c], 0.3, 25.0);
        assert_eq!(out.len(), 2, "c 中心距 30 超限，独立保留: {:?}", out);
        assert!((out[0].bbox.x2 - 60.0).abs() < 1e-6);
    }
}

/// P@0.5 贪心一对一匹配（[`crate::engine::greedy_mask_match`]）手算对照：
/// 阈值裁剪、同类约束、gt 一次性占用、分数降序优先四个要点各一例。
#[cfg(test)]
mod seg_precision_tests {
    use super::greedy_mask_match;

    fn ious(rows: &[&[f32]]) -> Vec<Vec<f32>> {
        rows.iter().map(|r| r.to_vec()).collect()
    }

    /// 基础：同类且达阈值的配对 + 阈值/异类排除。
    /// pred0(cls0, s0.9) iou=[0.8, 0.0]；pred1(cls1, s0.8) iou=[0.0, 0.3]；
    /// pred2(cls0, s0.7) iou=[0.2, 0.0]。gt=[cls0, cls1]
    /// → pred0↔g0（0.8）；pred1 与 g1 异类不配、0.3<0.5 不配；pred2 0.2<0.5。TP=1。
    #[test]
    fn threshold_and_class_constraints() {
        let preds = vec![(0u32, 0.9f32), (1, 0.8), (0, 0.7)];
        let m = ious(&[&[0.8, 0.0], &[0.0, 0.3], &[0.2, 0.0]]);
        assert_eq!(greedy_mask_match(preds, &[0, 1], &m, 0.5), 1);
    }

    /// 一对一：两个预测都高 IoU 命中同一个 gt 时只计一个 TP。
    /// pred0/1 均 cls0、与 g0 IoU 0.9；gt=[cls0] → TP=1（第二强预测为 FP）。
    #[test]
    fn one_to_one_no_double_count() {
        let preds = vec![(0u32, 0.9f32), (0, 0.8)];
        let m = ious(&[&[0.9], &[0.9]]);
        assert_eq!(greedy_mask_match(preds, &[0], &m, 0.5), 1);
    }

    /// 分数降序优先：高分预测先挑走自己最佳的 gt，低分预测才能命中剩下的。
    /// gt=[cls0, cls0]；predA(s0.9) iou=[0.55, 0.95]；predB(s0.8) iou=[0.95, 0.0]
    /// → A 先选 g1（其最佳 0.95），B 落 g0（0.95）→ TP=2。
    /// 若 A 抢占 g0，B 无 gt 可配 → TP=1：该断言同时锁定「分数优先 + 择优」。
    #[test]
    fn score_order_claims_best_match_first() {
        let preds = vec![(0u32, 0.9f32), (0, 0.8)];
        let m = ious(&[&[0.55, 0.95], &[0.95, 0.0]]);
        assert_eq!(greedy_mask_match(preds, &[0, 0], &m, 0.5), 2);
    }

    /// 空预测：TP=0（调用方据 n_pred 归零 P@0.5，不产生 NaN）。
    #[test]
    fn no_predictions_zero_tp() {
        assert_eq!(greedy_mask_match(vec![], &[0], &[], 0.5), 0);
    }
}

/// `[data].cache` 选层策略：auto 按画布堆体积与设备能力决定，显式值可覆盖。
#[cfg(test)]
mod seg_cache_mode_tests {
    use super::{resolve_seg_cache_mode, SegCacheMode};
    use tch::Device;

    #[test]
    fn auto_selects_gpu_only_within_budget_and_cuda() {
        let cuda = Device::Cuda(0);
        let cpu = Device::Cpu;
        // 526×640²×3×4B ≈ 2.6GB ≤ 4GiB 预算 → gpu
        assert_eq!(resolve_seg_cache_mode("auto", 526, 640, cuda), SegCacheMode::Gpu);
        // 20 万张远超预算 → ram
        assert_eq!(
            resolve_seg_cache_mode("auto", 200_000, 640, cuda),
            SegCacheMode::Ram
        );
        // 无 CUDA 永不选 gpu
        assert_eq!(resolve_seg_cache_mode("auto", 526, 640, cpu), SegCacheMode::Ram);
    }

    #[test]
    fn explicit_values_override_auto() {
        let cuda = Device::Cuda(0);
        let cpu = Device::Cpu;
        assert_eq!(resolve_seg_cache_mode("off", 526, 640, cuda), SegCacheMode::Off);
        assert_eq!(resolve_seg_cache_mode("ram", 526, 640, cuda), SegCacheMode::Ram);
        assert_eq!(resolve_seg_cache_mode("gpu", 526, 640, cuda), SegCacheMode::Gpu);
        // 显式 gpu 但无 CUDA → 回落 ram（引擎侧 warn）
        assert_eq!(resolve_seg_cache_mode("gpu", 526, 640, cpu), SegCacheMode::Ram);
    }
}

/// checkpoint epoch 元数据往返：save_checkpoint_epoch 写 meta.json，
/// read_checkpoint_epoch 读回；目录被清（旧格式/不存在）时返回 None。
#[cfg(test)]
mod ckpt_epoch_tests {
    use super::{read_checkpoint_epoch, save_checkpoint_epoch};
    use tch::nn::{self, VarStore};
    use tch::Device;

    #[test]
    fn checkpoint_epoch_roundtrip() {
        let dir = std::env::temp_dir().join(format!("av-ckpt-ep-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let vs = VarStore::new(Device::Cpu);
        let _fc = nn::linear(&(vs.root() / "head") / "fc", 3, 2, Default::default());
        save_checkpoint_epoch(&vs, &dir, 57).unwrap();
        assert_eq!(read_checkpoint_epoch(&dir), Some(57));
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(read_checkpoint_epoch(&dir), None, "目录不存在应返回 None");
    }
}
