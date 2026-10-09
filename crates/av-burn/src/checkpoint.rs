//! checkpoint 保存/加载：burn 二进制权重（`model.bp`）+ 配置快照
//! （`config.snapshot.toml`，约定与 av-runtime 的权重旁快照一致）。
//!
//! 目标是 **train → save → predict 零参数闭环**：预测侧只需权重目录，
//! 模型超参（imgsz/width/depth/类数/原型数）从快照自动重建，用户无需记超参。
//!
//! 权重用 burn 0.21 的 [`BinFileRecorder`]（FullPrecisionSettings，文件扩展名
//! `.bp` 自动附加）。**保存前请先 `model.valid()`**（Autodiff 模型剥去反传
//! 包装，得到内层后端的推理模型）；加载端用同款 recorder + 同版 burn。

use std::path::{Path, PathBuf};

use av_core::error::{AvError, AvResult};
use burn_core::module::Module;
use burn_core::record::{BinFileRecorder, FullPrecisionSettings};
use burn_core::tensor::backend::Backend;
use serde::{Deserialize, Serialize};

use crate::seg::{SegNet, SegNetCfg};

/// 快照文件名（权重同目录，与 av-runtime 的 config.snapshot.toml 约定一致）。
pub const SNAPSHOT_FILE: &str = "config.snapshot.toml";

/// 快照格式版本（当前 1）。
const SNAPSHOT_FORMAT: u32 = 1;

/// SegNet 装配快照（TOML 序列化，`config.snapshot.toml`）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegSnapshot {
    /// 快照格式版本。
    pub format: u32,
    /// 输入画布边长 S（正方）。
    pub imgsz: u32,
    /// 骨干宽度乘子。
    pub width: f32,
    /// 骨干深度乘子。
    pub depth: f32,
    /// 类别数。
    pub num_classes: usize,
    /// 原型数 K。
    pub num_protos: usize,
    /// 掩码 BCE 权重（loss 重建需要）。
    pub loss_w_bce: f64,
    /// Dice 权重。
    pub loss_w_dice: f64,
    /// 类别名（行号 = 类别下标；目录模式自动探测时为 `class_N` 占位）。
    #[serde(default)]
    pub classes: Vec<String>,
}

impl SegSnapshot {
    /// 快照 → 装配参数。
    pub fn to_cfg(&self) -> SegNetCfg {
        SegNetCfg {
            width: self.width,
            depth: self.depth,
            num_classes: self.num_classes,
            num_protos: self.num_protos,
            loss_w_bce: self.loss_w_bce,
            loss_w_dice: self.loss_w_dice,
        }
    }

    /// 类别名（快照未带名字时生成 `class_N` 占位，保证下标可读）。
    pub fn class_name(&self, idx: usize) -> String {
        self.classes
            .get(idx)
            .cloned()
            .unwrap_or_else(|| format!("class_{idx}"))
    }
}

/// 保存模型 + 快照到 `dir`（自动建目录）。返回权重文件路径（`model.bp`）。
///
/// 传 `model.valid()` 的推理模型（非 Autodiff）。
pub fn save_model<B: Backend>(
    model: &SegNet<B>,
    snapshot: &SegSnapshot,
    dir: &Path,
) -> AvResult<PathBuf> {
    if snapshot.format != SNAPSHOT_FORMAT {
        return Err(AvError::config(format!(
            "快照 format={} 与本实现（{SNAPSHOT_FORMAT}）不符",
            snapshot.format
        )));
    }
    std::fs::create_dir_all(dir)?;
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    model
        .clone()
        .save_file(dir.join("model"), &recorder)
        .map_err(|e| AvError::config(format!("权重写入失败: {e}")))?;
    let toml_text = toml::to_string_pretty(snapshot)
        .map_err(|e| AvError::config(format!("快照序列化失败: {e}")))?;
    std::fs::write(dir.join(SNAPSHOT_FILE), toml_text)?;
    Ok(dir.join("model.bp"))
}

/// 从权重目录（含 `model.bp` + `config.snapshot.toml`）加载模型。
/// 传 `model.bp` 文件路径亦可（自动取其父目录）。
pub fn load_model<B: Backend>(
    weights: &Path,
    device: &B::Device,
) -> AvResult<(SegNet<B>, SegSnapshot)> {
    let dir = if weights.is_dir() {
        weights.to_path_buf()
    } else {
        weights
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."))
    };
    let snap_text = std::fs::read_to_string(dir.join(SNAPSHOT_FILE))?;
    let snapshot: SegSnapshot = toml::from_str(&snap_text)?;
    if snapshot.format != SNAPSHOT_FORMAT {
        return Err(AvError::config(format!(
            "快照 format={} 与本实现（{SNAPSHOT_FORMAT}）不符",
            snapshot.format
        )));
    }
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    let model = SegNet::<B>::new(&snapshot.to_cfg(), device)?;
    let model = model
        .load_file(dir.join("model"), &recorder, device)
        .map_err(|e| AvError::config(format!("权重加载失败: {e}")))?;
    Ok((model, snapshot))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NdArrayB;
    use burn_core::tensor::Tensor;
    use burn_ndarray::NdArrayDevice;

    fn micro_snapshot() -> SegSnapshot {
        SegSnapshot {
            format: SNAPSHOT_FORMAT,
            imgsz: 128,
            width: 0.0625,
            depth: 0.33,
            num_classes: 3,
            num_protos: 8,
            loss_w_bce: 1.0,
            loss_w_dice: 1.0,
            classes: vec!["a".into(), "b".into(), "c".into()],
        }
    }

    /// 保存 → 加载闭环：同输入前向输出逐位一致（确定性后端）。
    /// 注意先取参照输出再保存——前向会推进 BN 统计，保存/加载两侧必须
    /// 从同一状态出发。
    #[test]
    fn save_load_roundtrip_identical_forward() {
        let device = NdArrayDevice::default();
        let snap = micro_snapshot();
        let model = SegNet::<NdArrayB>::new(&snap.to_cfg(), &device).unwrap();
        let x = Tensor::<NdArrayB, 4>::ones([1, 3, 128, 128], &device);
        let (p1, c1) = model.forward_seg(x.clone());
        let d1 = p1.into_data().convert::<f32>().to_vec::<f32>().unwrap();
        let e1 = c1.into_data().convert::<f32>().to_vec::<f32>().unwrap();

        let dir = std::env::temp_dir().join(format!(
            "avb-ckpt-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let bp = save_model(&model, &snap, &dir).expect("保存");
        assert!(bp.ends_with("model.bp"));
        assert!(dir.join(SNAPSHOT_FILE).exists());

        let (loaded, snap2) = load_model::<NdArrayB>(&dir, &device).expect("加载");
        assert_eq!(snap2.num_classes, 3);
        assert_eq!(snap2.class_name(2), "c");

        let (p2, c2) = loaded.forward_seg(x);
        let d2 = p2.into_data().convert::<f32>().to_vec::<f32>().unwrap();
        let e2 = c2.into_data().convert::<f32>().to_vec::<f32>().unwrap();
        assert_eq!(d1, d2, "原型输出应逐位一致");
        assert_eq!(e1, e2, "系数输出应逐位一致");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// 直接传 model.bp 文件路径也能加载（自动取父目录）。
    #[test]
    fn load_accepts_bp_file_path() {
        let device = NdArrayDevice::default();
        let snap = micro_snapshot();
        let model = SegNet::<NdArrayB>::new(&snap.to_cfg(), &device).unwrap();
        let dir = std::env::temp_dir().join(format!(
            "avb-ckpt-test2-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        save_model(&model, &snap, &dir).expect("保存");
        let (_, s2) = load_model::<NdArrayB>(&dir.join("model.bp"), &device).expect("加载");
        assert_eq!(s2.imgsz, 128);
        std::fs::remove_dir_all(&dir).ok();
    }
}
