//! 打印 `av train --dry-run` 的等价信息：配置解析 + 语义校验 + 生效 run_id + 运行计划。
//!
//! ```text
//! cargo run -p av-runtime --example print_plan
//! cargo run -p av-runtime --example print_plan -- ../../configs/detect_obb.toml
//! ```
//!
//! 该示例不依赖 `torch` 之外的任何运行时行为，适合在改配置后快速自检。

use std::path::PathBuf;

use av_core::config::{DataPipeline, RunConfig};

fn main() -> anyhow::Result<()> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    // 可选参数：配置文件路径；缺省用分类冒烟配置。
    // 相对路径按 cargo 运行目录（crate 根）解析。
    let config_path = match std::env::args().nth(1) {
        Some(p) => PathBuf::from(p),
        None => manifest_dir.join("../../configs/classify_smoke.toml"),
    };

    // `from_path` = TOML 反序列化 + `validate()` 语义校验，非法配置在此直接报错退出
    let cfg = RunConfig::from_path(&config_path)?;

    let tasks = cfg
        .model
        .tasks
        .iter()
        .map(|t| t.kind_name())
        .collect::<Vec<_>>()
        .join(",");
    let pipeline = match cfg.data.pipeline {
        DataPipeline::Synthetic => "synthetic（内置合成，开箱即训）".to_string(),
        DataPipeline::Dir => "dir（YOLO images/labels 目录）".to_string(),
        DataPipeline::AvPack => "avpack（M2 落地）".to_string(),
    };

    println!("配置有效 ✔");
    println!("  配置文件    = {}", config_path.display());
    // 生效 run_id：显式指定则原样使用；否则按配置快照 + 时间戳 blake3 生成 12 位
    println!("  run_id     = {}", cfg.effective_run_id());
    println!("  device     = {}", cfg.device);
    println!("  tasks      = {tasks}");
    println!("  epochs     = {}", cfg.train.epochs);
    println!("  batch_size = {}", cfg.train.batch_size);
    println!("  amp        = {}", cfg.train.amp);
    println!("  输出目录    = {}", cfg.output_dir.display());
    println!("  数据源     = {pipeline}");
    Ok(())
}
