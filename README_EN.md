<div align="center">

# AegisVision

**Multi-task vision training & inference, natively in Rust**

Detection · Oriented BBoxes · Instance Segmentation · Keypoints · Classification
Single binary · Zero Python · Embeddable (crates.io) · ONNX export

[![crates.io](https://img.shields.io/crates/v/aegisvision-runtime.svg)](https://crates.io/crates/aegisvision-runtime)
[![docs.rs](https://img.shields.io/docsrs/aegisvision-runtime)](https://docs.rs/aegisvision-runtime)
[![license](https://img.shields.io/crates/l/aegisvision-runtime.svg)](https://github.com/javpower/aegis-vision)

[简体中文](README.md) | English

</div>

---

## ⚡ One-command training

```bash
# Dataset = Ultralytics directory layout or data.yaml; classes auto-detected
av-runtime train --data datasets/glass_logo/data.yaml --imgsz 1280 --epochs 100
```

Auto-assembled: csp-elan backbone (YOLOv8-isomorphic) + official pretrain import +
mosaic/flip/HSV augmentation + **AMP + content-tile caching + double-buffered
prefetch** + EMA + fitness-based best.ckpt. For OBB/segmentation/keypoints and
custom backbones, use a TOML config (`av init` generates the template).

## 📊 Benchmarked against Ultralytics YOLO26n

Same machine, same dataset, same 1280 input, 100 epochs — industrial glass-logo
detection (3 classes, 1296/144 images, training directly on 5472×3648 originals):

| | mAP50 | mAP50:95 | R@0.5 | per epoch |
|---|---|---|---|---|
| **AegisVision (this repo)** | **0.9996** | 0.734 | **1.000** | 34 s |
| Ultralytics YOLO26n | 0.995 | **0.751** | 1.000 | **31.3 s** |

> No Python runtime, fully auditable weights and training; accuracy and speed are
> in the same league.

More real-data baselines (coco128 / ImageNette / coco8-seg / coco8-pose / dota8 /
backbone×pretrain ablation) in [runs/COMPARISON.md](runs/COMPARISON.md).

## 📦 Embed as a library (in-app online training)

```toml
[dependencies]
aegisvision-runtime = "0.3"
```

```rust
use av_core::config::RunConfig;
use std::path::Path;

let cfg: RunConfig = toml::from_str(&toml_text)?;

let report = av_runtime::api::train(&cfg)?;
let detections = av_runtime::api::infer(&cfg, Path::new("runs/run/best.ckpt"),
                                        Path::new("sample.jpg"))?;
av_runtime::api::export_safetensors(&cfg, Path::new("runs/run/best.ckpt"),
                                    Path::new("best.safetensors"))?;
```

Five layered crates: `aegisvision-core` (config/geometry) → `pretrain` (weight
import) → `tasks` (backbones/heads/losses) → `plugins` (registry) →
`runtime` (engine/CLI/panel).

## 🌐 Cross-platform inference (ONNX)

```bash
av-runtime export -w runs/run/best.ckpt --format safetensors
python scripts/export_onnx.py --backbone csp-elan --ckpt best.safetensors \
    --out best.onnx --imgsz 640 --classes 3 --verify
```

Consumed directly by ONNX Runtime on C#/Java/C++/JS/Android; preprocessing
(/255, normalization) is baked into the graph. All four backbones supported,
export verified against Rust inference to ≤ 1e-5.

## 🧩 Five tasks × four backbones

| Backbone | classify | detect / OBB | seg | keypoint |
|---|---|---|---|---|
| csp-elan (YOLOv8-isomorphic) | ✅ | ✅ | ✅ | ✅ |
| resnet18 (full ImageNet import) | ✅ | ✅ | ✅ | ✅ |
| dinov2 (ViT-S/14 official + QKV fusion) | ✅ | ✅ (imgsz in multiples of 448) | ✅ | ✅ |
| simple-cnn (teaching/smoke) | ✅ | ✅ | ✅ | ✅ |

Custom backbones: implement the trait → register → select in config
(three steps via `av-plugins`).

## ✨ Engineering

- **All five losses hand-written with hand-computed unit tests**: TAL + CIoU +
  DFL / KFIoU + rotated NMS / YOLACT prototypes + Dice / OKS / cross-entropy
- **Data pipeline v2**: content-tile cache + streaming build (train directly on
  60MP originals, no downsampling) + double-buffered prefetch + SIMD resize —
  measured 195 → 34 s/epoch on identical data/resolution
- **Modern training loop**: AMP fp16 (dynamic grad scaling), EMA, cosine
  schedule, grad accumulation/clipping, deterministic seeding
- **.avpack single-file dataset container** (blake3 + mmap zero-copy), SAHI-style
  sliced inference, SSE live panel, `--save-viz` visualization
- **Bidirectional pretrain channel**: safetensors import (regex layer mapping +
  partial-load report) / export (verified readable from Python)
- GPU: RTX 50-series (Blackwell/sm_120) tested — only the driver required

## ⚙️ Getting started

Requirements: Rust stable + MSVC (Windows). CPU path needs no setup (CPU
libtorch auto-downloads on first build); GPU configured by one script.

```powershell
.\scripts\setup-env.ps1                 # Windows (Linux: ./scripts/setup-env.sh)
cargo build --release -p av-runtime     # produces target\release\av-runtime.exe

av-runtime train --data <dataset or data.yaml> --imgsz 640
av-runtime infer -w runs/<id>/best.ckpt --input sample.jpg --save-viz viz
av-runtime eval   -w runs/<id>/best.ckpt --report report.json
```

## 📚 Docs

- [docs/USAGE.md](docs/USAGE.md) — full CLI, five-task data formats, GPU setup,
  benchmark tables, backbone×task matrix, known-issues archive
- [runs/COMPARISON.md](runs/COMPARISON.md) — backbone×pretrain ablation study
- [runs/M2-BURN-BENCHMARK.md](runs/M2-BURN-BENCHMARK.md) — burn-wgpu dual-track

## 🎯 Honest boundaries

Every claim maps to reproducible measurements and unit tests; unfinished items
are stated as such: mosaic/mixup currently detect-only; AMP/EMA wired on the
detect path; multi-GPU scheduled; ONNX export uses a Python side-car (deployment
side needs zero Python, export side needs torch); `--resume` currently seg-only;
the burn-wgpu backend is a cross-vendor validation track (~1/3.6 of tch
throughput). Full archive in USAGE §8.

## License

Dual-licensed MIT OR Apache-2.0.
