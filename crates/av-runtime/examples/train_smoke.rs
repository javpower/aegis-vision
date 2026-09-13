//! 端到端冒烟示例：加载工作区的 `configs/classify_smoke.toml`，训练一个分类模型并打印报告。
//!
//! 运行（首次编译会自动下载 CPU 版 libtorch，约 180MB）：
//!
//! ```text
//! cargo run -p av-runtime --example train_smoke
//! ```
//!
//! 说明：
//! - examples 的运行目录（cwd）是 crate 根，工作区级配置路径用 `CARGO_MANIFEST_DIR` 拼接；
//! - 训练产物写入工作区根 `runs/`，与 CLI 行为一致；
//! - 需要 `torch` feature（默认启用）；`--no-default-features` 构建时本示例打印提示退出。

#[cfg(feature = "torch")]
use std::path::PathBuf;

#[cfg(feature = "torch")]
use av_core::config::RunConfig;

#[cfg(feature = "torch")]
fn main() -> anyhow::Result<()> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let config_path = manifest_dir.join("../../configs/classify_smoke.toml");

    println!("加载配置：{}", config_path.display());
    let mut cfg = RunConfig::from_path(&config_path)?; // 解析 + 语义校验一步完成
    cfg.output_dir = manifest_dir.join("../../runs");

    println!("开始训练（内置合成数据，无需外部数据集）……");
    let report = av_runtime::engine::train(&cfg)?;

    println!("训练完成 ✔");
    println!("  run_id      = {}", report.run_id);
    println!("  task        = {}", report.task);
    println!("  epochs      = {}", report.epochs);
    println!("  final_loss  = {:.4}", report.final_loss);
    println!("  {} = {:.3}", report.metric, report.metric_value);
    if let Some((name, value)) = &report.secondary {
        println!("  {name} = {value:.3}");
    }
    println!("  权重与报告  = {}", report.run_dir);
    Ok(())
}

#[cfg(not(feature = "torch"))]
fn main() {
    eprintln!(
        "本示例需要 torch feature（默认启用）。请用：\
         cargo run -p av-runtime --example train_smoke"
    );
}
