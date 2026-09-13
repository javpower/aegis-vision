//! 训练指标增量落盘（面板实时化第一层）：训练循环每个 epoch 结束时向
//! `runs/<id>/metrics.jsonl` 追加一行 JSON（epoch/loss/metric/时间戳）。
//!
//! 设计原则与面板一致——**数据主权在文件**：面板挂了、进程崩了，已写行都在；
//! 读取侧支持字节偏移增量读（SSE 每次轮询只推新行），残行（写一半）自动跳过、
//! 下次从完整行边界继续。

use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use serde_json::Value;

/// 指标文件名（约定固定，面板 SSE 按它增量读）。
pub const METRICS_FILE: &str = "metrics.jsonl";

/// 当前 UNIX 时间戳（秒）。落盘失败不影响训练——调用方只 log 不中断。
pub fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// 追加一行指标 JSON 到 `run_dir/metrics.jsonl`（不存在则创建，append 模式）。
///
/// 单行写入（line + '\n' 一次 write），SSE 侧按行边界增量读，不会读到半行。
pub fn append(run_dir: &Path, row: &Value) -> std::io::Result<()> {
    let mut line = row.to_string();
    line.push('\n');
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(run_dir.join(METRICS_FILE))?;
    f.write_all(line.as_bytes())?;
    f.flush()
}

/// 读取全部完整行（解析失败的行跳过；文件不存在返回空）。
pub fn read_all(run_dir: &Path) -> Vec<Value> {
    read_from(run_dir, 0)
        .map(|(_, rows)| rows)
        .unwrap_or_default()
}

/// 从字节偏移 `offset` 起增量读取：返回 (新偏移, 新增完整行)。
///
/// - 只消费以 '\n' 结尾的完整行（并发写时最后一行可能残缺，跳过留给下次）；
/// - 文件不存在 / 无新数据 → `(offset, [])`。
pub fn read_from(run_dir: &Path, offset: u64) -> std::io::Result<(u64, Vec<Value>)> {
    let path = run_dir.join(METRICS_FILE);
    let Ok(mut f) = std::fs::File::open(&path) else {
        return Ok((offset, Vec::new()));
    };
    let len = f.metadata()?.len();
    if len <= offset {
        return Ok((offset, Vec::new()));
    }
    f.seek(SeekFrom::Start(offset))?;
    let mut buf = Vec::with_capacity((len - offset) as usize);
    f.read_to_end(&mut buf)?;

    let mut rows = Vec::new();
    let mut consumed = 0usize;
    // 只取完整行（最后一个 '\n' 之前的部分）
    while let Some(nl) = buf[consumed..].iter().position(|&b| b == b'\n') {
        let line = &buf[consumed..consumed + nl];
        consumed += nl + 1;
        if line.is_empty() {
            continue;
        }
        if let Ok(v) = serde_json::from_slice::<Value>(line) {
            rows.push(v);
        }
    }
    Ok((offset + consumed as u64, rows))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::path::PathBuf;

    fn temp_dir(tag: &str) -> PathBuf {
        static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        std::env::temp_dir().join(format!("av-metrics-{tag}-{}-{n}", std::process::id()))
    }

    #[test]
    fn append_then_read_roundtrip() {
        let dir = temp_dir("roundtrip");
        std::fs::create_dir_all(&dir).unwrap();
        for e in 1..=3u32 {
            append(
                &dir,
                &json!({"epoch": e, "loss": 1.0 / e as f32, "metric": "mean_iou", "metric_value": 0.1 * e as f32, "ts": now_unix()}),
            )
            .unwrap();
        }
        let rows = read_all(&dir);
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0]["epoch"], 1);
        assert_eq!(rows[2]["epoch"], 3);
        assert!((rows[1]["metric_value"].as_f64().unwrap() - 0.2).abs() < 1e-6);
        assert_eq!(rows[0]["metric"], "mean_iou");
        assert!(rows[0]["ts"].as_u64().unwrap() > 0);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn read_from_incremental_offset() {
        let dir = temp_dir("incremental");
        std::fs::create_dir_all(&dir).unwrap();
        append(&dir, &json!({"epoch": 1})).unwrap();
        append(&dir, &json!({"epoch": 2})).unwrap();
        // 全量
        let (off1, rows1) = read_from(&dir, 0).unwrap();
        assert_eq!(rows1.len(), 2);
        assert!(off1 > 0);
        // 无新数据
        let (off2, rows2) = read_from(&dir, off1).unwrap();
        assert_eq!(rows2.len(), 0);
        assert_eq!(off2, off1);
        // 追加两行后只推增量
        append(&dir, &json!({"epoch": 3})).unwrap();
        append(&dir, &json!({"epoch": 4})).unwrap();
        let (off3, rows3) = read_from(&dir, off1).unwrap();
        assert_eq!(rows3.len(), 2);
        assert_eq!(rows3[0]["epoch"], 3);
        assert_eq!(rows3[1]["epoch"], 4);
        assert!(off3 > off1);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn partial_last_line_is_skipped() {
        let dir = temp_dir("partial");
        std::fs::create_dir_all(&dir).unwrap();
        append(&dir, &json!({"epoch": 1})).unwrap();
        // 模拟并发写了一半（无换行结尾）
        let p = dir.join(METRICS_FILE);
        let mut f = std::fs::OpenOptions::new().append(true).open(&p).unwrap();
        use std::io::Write;
        f.write_all(br#"{"epoch": 2"#).unwrap();
        f.flush().unwrap();
        let rows = read_all(&dir);
        assert_eq!(rows.len(), 1, "残行必须被跳过");
        assert_eq!(rows[0]["epoch"], 1);
        // 补全换行后即可读出
        let mut f = std::fs::OpenOptions::new().append(true).open(&p).unwrap();
        f.write_all(b"}\n").unwrap();
        let rows = read_all(&dir);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[1]["epoch"], 2);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn missing_file_returns_empty() {
        let dir = temp_dir("missing");
        std::fs::create_dir_all(&dir).unwrap();
        assert!(read_all(&dir).is_empty());
        let (off, rows) = read_from(&dir, 0).unwrap();
        assert_eq!(off, 0);
        assert!(rows.is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }
}
