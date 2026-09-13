//! M2 基准：csp-elan nano @640 真实数据，burn-wgpu 后端逐 epoch 计时。
//!
//! 对照臂 = tch 版 `runs/seg-harness-csp`（同数据同协议，实测 ~7.5s/epoch，
//! val mIoU 0.846）。本基准只产出**吞吐与 loss 收敛**对比（av-burn spike
//! 无 val 评测协议，mIoU 对比留待正式移植），口径诚实标注。
//!
//! 运行（显卡空闲时）：
//! ```text
//! CARGO_TARGET_DIR=target-burn cargo run --release -p av-burn --features wgpu \
//!     --example bench_csp640 -- <数据根目录> [epochs]
//! ```

use std::path::Path;
use std::time::Instant;

use av_burn::data::load_cocoseg_dir;
use av_burn::seg::{SegNet, SegNetCfg};
use av_burn::train::{TrainCfg, cosine_lr, make_optimizer};
use av_burn::wgpu_check::WgpuTrainB;
use burn_core::tensor::{Tensor, TensorData};
use burn_wgpu::WgpuDevice;

/// M2 基准后端 = WgpuTrainB（0.21 的 Wgpu 类型默认已含 Fusion + autotune，
/// 实测 27s/epoch 即融合优化后数字；BURN_AUTOTUNE 级别未暴露调节口）。

/// 极简 xorshift 洗牌（避免引入 rand 依赖；与仓库 rng 风格一致）。
fn shuffle(idx: &mut [usize], seed: u64) {
    let mut s = seed | 1;
    for i in (1..idx.len()).rev() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        idx.swap(i, (s as usize) % (i + 1));
    }
}

fn main() -> av_core::AvResult<()> {
    let args: Vec<String> = std::env::args().collect();
    let root = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "E:/xunlian/yolo_dataset".into());
    let epochs: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(120);
    let img_size: u32 = 640;
    let batch: usize = 16;

    let device = WgpuDevice::default();
    println!("[bench] 数据 {root} epochs={epochs} batch={batch} img={img_size}");
    let t_load = Instant::now();
    let data = load_cocoseg_dir(Path::new(&root), "train", img_size)?;
    println!(
        "[bench] 载入 {} 张（{:.1}s，含 letterbox 预解码）",
        data.len(),
        t_load.elapsed().as_secs_f32()
    );

    let mut model = SegNet::<WgpuTrainB>::new(
        &SegNetCfg {
            width: 0.25,
            depth: 0.33,
            num_classes: 4,
            num_protos: 32,
            loss_w_bce: 1.0,
            loss_w_dice: 1.0,
        },
        &device,
    )?;
    let steps_per_epoch = (data.len() / batch).max(1);
    let tcfg = TrainCfg {
        lr: 1e-3,
        lr_min: 1e-5,
        weight_decay: 5e-4,
        max_grad_norm: 10.0,
        total_steps: steps_per_epoch * epochs,
    };
    let mut optim = make_optimizer(&tcfg);

    let s = img_size as usize;
    let mut order: Vec<usize> = (0..data.len()).collect();
    let mut epoch_times = Vec::new();
    for epoch in 1..=epochs {
        shuffle(&mut order, 0x9E37_79B9_7F4A_7C15 ^ (epoch as u64));
        let t0 = Instant::now();
        let mut loss_sum = 0f32;
        let mut steps = 0usize;
        for chunk in order.chunks(batch) {
            // 组批：像素平铺 [B,3,S,S]（samples 已预 letterbox 到 [0,1] CHW）
            let n = chunk.len();
            let mut buf = Vec::with_capacity(n * 3 * s * s);
            for &i in chunk {
                buf.extend_from_slice(&data[i].pixels);
            }
            let x = Tensor::<WgpuTrainB, 4>::from_data(
                TensorData::new(buf, [n, 3, s, s]),
                &device,
            );
            let masks: Vec<Vec<Vec<u8>>> = chunk.iter().map(|&i| data[i].masks.clone()).collect();
            let labels: Vec<Vec<u32>> = chunk.iter().map(|&i| data[i].labels.clone()).collect();
            let lr = cosine_lr(&tcfg, (epoch - 1) * steps_per_epoch + steps);
            let (m, v) = av_burn::train::train_step(model, &mut optim, lr, |m| {
                m.loss(x.clone(), &masks, &labels)
            });
            model = m;
            loss_sum += v;
            steps += 1;
        }
        let dt = t0.elapsed().as_secs_f32();
        epoch_times.push(dt);
        println!(
            "[bench] epoch {epoch}/{} loss={:.4} time={dt:.1}s",
            epochs,
            loss_sum / steps.max(1) as f32
        );
    }
    epoch_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let steady = epoch_times[epoch_times.len() / 2];
    println!(
        "[bench] 完成：中位 epoch={steady:.1}s 首 epoch（含 wgpu JIT）={:.1}s —— tch 对照臂 ~7.5s/epoch",
        epoch_times[epoch_times.len() - 1]
    );
    Ok(())
}
