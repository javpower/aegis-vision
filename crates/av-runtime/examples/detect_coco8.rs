//! 真实数据检测训练示例：Ultralytics coco8（YOLO `images/` + `labels/` 目录格式）。
//!
//! ```text
//! # 数据准备（二选一）：scripts/get-data.ps1 或手动解压 data/coco8.zip
//! cargo run -p av-runtime --example detect_coco8
//! ```
//!
//! 数据集不存在时打印提示后正常退出（非报错）。完整训练为 400 epoch；
//! 只想快速冒烟请用 `example train_smoke` 或 `configs/detect_smoke.toml`。
//! 需要 `torch` feature（默认启用）；`--no-default-features` 构建时打印提示退出。

#[cfg(feature = "torch")]
use std::path::PathBuf;

#[cfg(feature = "torch")]
use av_core::config::RunConfig;

#[cfg(feature = "torch")]
fn main() -> anyhow::Result<()> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let data_root = manifest_dir.join("../../data/coco8");

    // 前置校验：coco8 必须已就位（images/<split> 是 YOLO 目录格式的最小判据）
    if !data_root.join("images").join("train").is_dir() {
        eprintln!("未找到 coco8 数据集：{}", data_root.display());
        eprintln!("请先执行 scripts/get-data.ps1，或将 data/coco8.zip 解压到 data/coco8 后重试。");
        eprintln!("合成数据快速冒烟：cargo run -p av-runtime --example train_smoke");
        return Ok(());
    }

    let config_path = manifest_dir.join("../../configs/detect_coco8.toml");
    println!("加载配置：{}", config_path.display());
    let mut cfg = RunConfig::from_path(&config_path)?;

    // 配置中的 data.sources.*.dir 是相对工作区根的路径；示例的运行目录不固定，
    // 统一改写为绝对路径，保证任意 cwd 下都能定位数据集（val 源与 train 同根）。
    cfg.data.sources.train.dir = Some(data_root.clone());
    cfg.data.sources.val.dir = Some(data_root);
    cfg.output_dir = manifest_dir.join("../../runs");

    println!("开始检测训练（coco8，{} epochs）……", cfg.train.epochs);
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
         cargo run -p av-runtime --example detect_coco8"
    );
}
