//! AV 原生权重格式（目录名约定 `avpretrain`）。
//!
//! 布局：一个目录 = 一份权重快照 ——
//!
//! ```text
//! <dir>/
//!   manifest.json          # 本 crate 的元信息 + 每张量 blake3 哈希
//!   backbone_c1_weight     # 每变量一个 Tensor::save 文件（libtorch 格式）
//!   backbone_c1_bias
//!   ...
//! ```
//!
//! 这是 `av-runtime::engine` 目录式 checkpoint（tch 0.17 的 `VarStore::save/load`
//! 在 Windows + libtorch 2.4 下序列化不兼容，自研的「每变量一文件 + 形状校验 +
//! no_grad copy_ 写回」方案）的通用化复用：变量文件名规则与其完全一致
//! （[`tensor_file_name`]），因此训练产出的 `best.ckpt/` 目录本身就是合法的
//! avpretrain 权重（补一份 manifest 即成可分发快照）。
//!
//! 哈希算法用 blake3（项目统一依赖，比 SHA256 快一个量级）；manifest 字段名
//! 沿用 `hash`。校验 [`verify_hashes`] 一次报出**所有**不匹配/缺失文件并定位，
//! 而不是在第一个坏文件上中止。

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use av_core::error::{AvError, AvResult};

/// 格式名（写入 manifest.format）。
pub const FORMAT_NAME: &str = "avpretrain";
/// manifest schema 版本；不兼容变更时递增。
pub const FORMAT_VERSION: u32 = 1;
/// manifest 文件名。
pub const MANIFEST_FILE: &str = "manifest.json";

// ---------------------------------------------------------------------------
// manifest 数据结构
// ---------------------------------------------------------------------------

/// 导出时的业务元信息（框架版本/时间戳自动生成）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightMeta {
    /// 骨干类型（如 "simple-cnn"）
    pub backbone: String,
    /// 来源数据集（如 "synthetic" / "coco128"）
    pub source_dataset: String,
    /// 训练 epoch（推理快照可填 None）
    pub epoch: Option<u32>,
    /// 任务（"detect" / "classify" / ...）
    pub task: String,
}

impl Default for WeightMeta {
    fn default() -> Self {
        Self {
            backbone: "unknown".into(),
            source_dataset: "unknown".into(),
            epoch: None,
            task: "unknown".into(),
        }
    }
}

/// 单张量条目：变量名、文件名、blake3 hex（字段名 `hash`）、形状、dtype。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TensorInfo {
    pub name: String,
    pub file: String,
    pub hash: String,
    pub shape: Vec<i64>,
    pub dtype: String,
}

/// manifest.json 的 schema。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WeightManifest {
    /// 固定 "avpretrain"
    pub format: String,
    pub format_version: u32,
    /// 生成方框架版本（av-pretrain crate 版本）
    pub framework_version: String,
    /// 骨干类型
    pub backbone: String,
    /// 来源数据集
    pub source_dataset: String,
    pub epoch: Option<u32>,
    /// 任务
    pub task: String,
    /// RFC3339 风格 UTC 时间戳（无 chrono 依赖，手写 civil-from-days 换算）
    pub created_at: String,
    pub tensors: Vec<TensorInfo>,
}

// ---------------------------------------------------------------------------
// 命名与哈希
// ---------------------------------------------------------------------------

/// 变量名 → 文件名：与 `av-runtime::engine` checkpoint 完全一致
/// （`/`、`\` → `__`，`.` → `_`）。已知限制：替换后可能碰撞（如 "a/b" 与
/// "a__b"），与既有 checkpoint 同一取舍，变量命名由模型装配保证唯一。
pub fn tensor_file_name(name: &str) -> String {
    name.replace(['/', '\\'], "__").replace('.', "_")
}

/// 文件 blake3 hex（流式读取，不整载内存）。
pub fn hash_file(path: &Path) -> AvResult<String> {
    let mut f = fs::File::open(path)?;
    let mut hasher = blake3::Hasher::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher.finalize().to_hex().to_string())
}

/// 当前时刻的 RFC3339 风格 UTC 时间戳（毫秒精度；Howard Hinnant civil-from-days
/// 算法，避免引入 chrono）。
pub fn iso8601_now() -> String {
    let d = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = d.as_secs() as i64;
    let ms = d.subsec_millis();
    let days = secs.div_euclid(86_400);
    let sod = secs.rem_euclid(86_400);

    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { y + 1 } else { y };

    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{ms:03}Z",
        sod / 3600,
        (sod % 3600) / 60,
        sod % 60
    )
}

// ---------------------------------------------------------------------------
// manifest 读写 + 哈希校验（纯逻辑，无 libtorch 依赖）
// ---------------------------------------------------------------------------

/// 写 manifest.json（pretty JSON）。
pub fn write_manifest(dir: &Path, manifest: &WeightManifest) -> AvResult<()> {
    fs::create_dir_all(dir)?;
    let s = serde_json::to_string_pretty(manifest)
        .map_err(|e| AvError::train(format!("manifest 序列化失败: {e}")))?;
    fs::write(dir.join(MANIFEST_FILE), s)?;
    Ok(())
}

/// 读 manifest.json。
pub fn read_manifest(dir: &Path) -> AvResult<WeightManifest> {
    let p = dir.join(MANIFEST_FILE);
    let s = fs::read_to_string(&p)?;
    let m: WeightManifest = serde_json::from_str(&s)
        .map_err(|e| AvError::train(format!("解析 {} 失败: {e}", p.display())))?;
    if m.format != FORMAT_NAME {
        return Err(AvError::train(format!(
            "{} 不是 {} 格式（format = {:?}）",
            p.display(),
            FORMAT_NAME,
            m.format
        )));
    }
    Ok(m)
}

/// 单个哈希不匹配项（expected/actual 均为 blake3 hex；actual 为占位串表示文件缺失/不可读）。
#[derive(Debug, Clone, PartialEq)]
pub struct HashMismatch {
    pub name: String,
    pub file: PathBuf,
    pub expected: String,
    pub actual: String,
}

/// 校验失败：携带**全部**不匹配项（一次修完，而不是逐个试错）。
#[derive(Debug, Clone, PartialEq)]
pub struct HashVerifyError {
    pub mismatches: Vec<HashMismatch>,
}

impl std::fmt::Display for HashVerifyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "权重哈希校验失败，{} 个文件不匹配/缺失:",
            self.mismatches.len()
        )?;
        for m in &self.mismatches {
            write!(f, "\n  - {} ({})", m.name, m.file.display())?;
            write!(
                f,
                "\n      期望 {} 实际 {}",
                truncate(&m.expected, 16),
                truncate(&m.actual, 24)
            )?;
        }
        Ok(())
    }
}

/// UTF-8 安全截断（省略号示意；哈希 hex / 占位串都足够表达）。
fn truncate(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        s.to_string()
    } else {
        let head: String = s.chars().take(max_chars).collect();
        format!("{head}…")
    }
}

impl std::error::Error for HashVerifyError {}

/// 逐张量重算文件 blake3 并与 manifest 比对；**不短路**，报告全部不匹配项。
pub fn verify_hashes(dir: &Path, manifest: &WeightManifest) -> Result<(), HashVerifyError> {
    let mut mismatches = Vec::new();
    for info in &manifest.tensors {
        let file = dir.join(&info.file);
        let actual = hash_file(&file).unwrap_or_else(|_| "<文件缺失或不可读>".into());
        if actual != info.hash {
            mismatches.push(HashMismatch {
                name: info.name.clone(),
                file,
                expected: info.hash.clone(),
                actual,
            });
        }
    }
    if mismatches.is_empty() {
        Ok(())
    } else {
        Err(HashVerifyError { mismatches })
    }
}

// ---------------------------------------------------------------------------
// 张量读写（torch feature）
// ---------------------------------------------------------------------------

/// 每变量一个 `Tensor::save` 文件写入 `dir`（与 engine checkpoint 同方案）。
#[cfg(feature = "torch")]
pub fn save_named(dir: &Path, vars: &[(String, tch::Tensor)]) -> AvResult<()> {
    fs::create_dir_all(dir)?;
    for (name, t) in vars {
        let f = dir.join(tensor_file_name(name));
        t.save(&f)
            .map_err(|e| AvError::train(format!("保存张量 {name} 失败: {e}")))?;
    }
    Ok(())
}

/// 基于已落盘的变量文件构建 manifest（逐文件 blake3 + 张量形状/dtype）。
#[cfg(feature = "torch")]
pub fn build_manifest(
    dir: &Path,
    vars: &[(String, tch::Tensor)],
    meta: WeightMeta,
) -> AvResult<WeightManifest> {
    let mut tensors = Vec::with_capacity(vars.len());
    for (name, t) in vars {
        let file = tensor_file_name(name);
        let hash = hash_file(&dir.join(&file))?;
        tensors.push(TensorInfo {
            name: name.clone(),
            file,
            hash,
            shape: t.size(),
            dtype: format!("{:?}", t.kind()),
        });
    }
    Ok(WeightManifest {
        format: FORMAT_NAME.into(),
        format_version: FORMAT_VERSION,
        framework_version: env!("CARGO_PKG_VERSION").into(),
        backbone: meta.backbone,
        source_dataset: meta.source_dataset,
        epoch: meta.epoch,
        task: meta.task,
        created_at: iso8601_now(),
        tensors,
    })
}

/// 一站式导出：save_named → build_manifest → write_manifest。
#[cfg(feature = "torch")]
pub fn export_pretrain_dir(
    dir: &Path,
    vars: &[(String, tch::Tensor)],
    meta: WeightMeta,
) -> AvResult<WeightManifest> {
    save_named(dir, vars)?;
    let manifest = build_manifest(dir, vars, meta)?;
    write_manifest(dir, &manifest)?;
    Ok(manifest)
}

/// 加载权重写回既有变量（`vars` 为 VarStore 变量的 shallow clone，原地 copy_
/// 即写穿到 VarStore）：形状校验给可读错误（结构变更防护），no_grad 内写回。
#[cfg(feature = "torch")]
pub fn load_named(dir: &Path, vars: &mut [(String, tch::Tensor)]) -> AvResult<()> {
    tch::no_grad(|| {
        for (name, t) in vars.iter_mut() {
            let f = dir.join(tensor_file_name(name));
            let loaded = tch::Tensor::load(&f).map_err(|e| {
                AvError::train(format!(
                    "读取张量 {name} 失败: {e}（{} 不存在或损坏？）",
                    f.display()
                ))
            })?;
            // 结构变更（模型/权重不同构）时给可读错误，而不是让 libtorch 在 copy_ panic
            if loaded.size() != t.size() {
                return Err(AvError::train(format!(
                    "权重与当前模型结构不匹配: {name} 期望 {:?} 得到 {:?}",
                    t.size(),
                    loaded.size()
                )));
            }
            t.copy_(&loaded);
        }
        Ok(())
    })
}

/// 按 manifest 读入目录中**全部**张量（供适配器/分析用；写回模型走 load_named）。
#[cfg(feature = "torch")]
pub fn read_all_named(
    dir: &Path,
    manifest: &WeightManifest,
) -> AvResult<Vec<(String, tch::Tensor)>> {
    manifest
        .tensors
        .iter()
        .map(|info| {
            let f = dir.join(&info.file);
            let t = tch::Tensor::load(&f)
                .map_err(|e| AvError::train(format!("读取张量 {} 失败: {e}", info.name)))?;
            Ok((info.name.clone(), t))
        })
        .collect()
}

#[cfg(all(test, feature = "torch"))]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tch::Tensor;

    fn temp_dir(tag: &str) -> PathBuf {
        static N: AtomicUsize = AtomicUsize::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("av-pretrain-test-{tag}-{}-{n}", std::process::id()))
    }

    fn sample_vars() -> Vec<(String, tch::Tensor)> {
        use tch::Kind;
        vec![
            (
                "backbone.c1.weight".into(),
                Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0]).reshape([2i64, 2]),
            ),
            (
                "backbone.c1.bias".into(),
                Tensor::from_slice(&[0.5f32, -0.5]),
            ),
            (
                "head.fc.weight".into(),
                Tensor::from_slice(&[1i64, 2, 3]).to_kind(Kind::Int),
            ),
        ]
    }

    #[test]
    fn save_load_roundtrip_preserves_values() {
        use tch::Kind;
        let dir = temp_dir("roundtrip");
        let vars = sample_vars();
        export_pretrain_dir(&dir, &vars, WeightMeta::default()).expect("导出应成功");

        // manifest 元信息
        let m = read_manifest(&dir).expect("manifest 应可读");
        assert_eq!(m.format, FORMAT_NAME);
        assert_eq!(m.format_version, FORMAT_VERSION);
        assert_eq!(m.tensors.len(), 3);
        assert_eq!(m.tensors[0].name, "backbone.c1.weight");
        assert_eq!(m.tensors[0].file, "backbone_c1_weight");
        assert_eq!(m.tensors[0].shape, vec![2, 2]);
        assert_eq!(m.tensors[2].dtype, format!("{:?}", Kind::Int));
        assert!(m.created_at.ends_with('Z'));

        // 文件名规则与 engine checkpoint 一致（'.' → '_'）
        assert!(dir.join("head_fc_weight").is_file());

        // 读回值一致（含非默认 dtype）
        let mut back = sample_vars();
        for (_, t) in back.iter_mut() {
            *t = Tensor::zeros(t.size(), (t.kind(), t.device()));
        }
        load_named(&dir, &mut back).expect("读回应成功");
        for ((_, a), (_, b)) in vars.iter().zip(&back) {
            assert_eq!(a.size(), b.size());
            let diff = (a - b).abs().max().double_value(&[]);
            assert!(diff == 0.0, "读回值应一致，max diff = {diff}");
        }
        verify_hashes(&dir, &m).expect("校验应通过");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_named_reports_shape_mismatch() {
        let dir = temp_dir("shape");
        let vars = sample_vars();
        export_pretrain_dir(&dir, &vars, WeightMeta::default()).expect("导出应成功");

        // 目标变量形状不同 → 可读错误并指名变量与两侧形状
        let mut wrong = vec![(
            "backbone.c1.weight".to_string(),
            Tensor::zeros([3i64, 2], (tch::Kind::Float, tch::Device::Cpu)),
        )];
        let err = load_named(&dir, &mut wrong).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("backbone.c1.weight"), "got: {msg}");
        assert!(
            msg.contains("[3, 2]") && msg.contains("[2, 2]"),
            "got: {msg}"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn verify_hashes_locates_tampered_file_and_reports_all() {
        let dir = temp_dir("tamper");
        let vars = sample_vars();
        let m = export_pretrain_dir(&dir, &vars, WeightMeta::default()).expect("导出应成功");

        // 篡改一个变量文件（backbone.c1.bias），另删一个（head.fc.weight）
        let victim = dir.join("backbone_c1_bias");
        let bytes = fs::read(&victim).unwrap();
        let mut tampered = bytes.clone();
        tampered[0] ^= 0xFF;
        fs::write(&victim, &tampered).unwrap();
        fs::remove_file(dir.join("head_fc_weight")).unwrap();

        let err = verify_hashes(&dir, &m).unwrap_err();
        assert_eq!(err.mismatches.len(), 2, "应一次报出全部问题: {err:?}");
        let names: Vec<&str> = err.mismatches.iter().map(|x| x.name.as_str()).collect();
        assert!(
            names.contains(&"backbone.c1.bias"),
            "应定位到被篡改变量: {err}"
        );
        assert!(names.contains(&"head.fc.weight"), "应定位到缺失变量: {err}");
        assert!(
            err.to_string().contains("backbone_c1_bias"),
            "错误信息应含文件路径"
        );
        // 未篡改变量不在报告中
        assert!(!names.contains(&"backbone.c1.weight"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn tensor_file_name_matches_engine_checkpoint_scheme() {
        assert_eq!(tensor_file_name("backbone.c1.weight"), "backbone_c1_weight");
        assert_eq!(tensor_file_name("a/b.c"), "a__b_c");
        assert_eq!(tensor_file_name(r"a\b.c"), "a__b_c");
    }

    #[test]
    fn manifest_roundtrip_rejects_foreign_format() {
        let dir = temp_dir("manifest");
        fs::create_dir_all(&dir).unwrap();
        let m = WeightManifest {
            format: "other".into(),
            format_version: 0,
            framework_version: "0".into(),
            backbone: "b".into(),
            source_dataset: "d".into(),
            epoch: None,
            task: "t".into(),
            created_at: iso8601_now(),
            tensors: vec![],
        };
        write_manifest(&dir, &m).unwrap();
        let err = read_manifest(&dir).unwrap_err();
        assert!(err.to_string().contains(FORMAT_NAME), "got: {err}");
        let _ = fs::remove_dir_all(&dir);
    }
}
