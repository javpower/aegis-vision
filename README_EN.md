<div align="center">

# AegisVision

**Multi-task visual training and inference, natively in Rust**

Object Detection · Instance Segmentation · Oriented Bounding Boxes · Keypoints · Classification —
five tasks, one training system

[简体中文](README.md) | English

</div>

---

## Why AegisVision

Deep learning engineering has long been anchored to the Python ecosystem: training scripts,
export chains, and system boundaries held together by glue code. AegisVision takes a different
path — **a YOLO-grade training system implemented from scratch in Rust**, consolidated into a
single memory-safe, auditable binary with no interpreter dependency.

| | AegisVision | Conventional Python stack |
|---|---|---|
| Runtime | Single binary, zero Python | Python + hundreds of packages |
| Memory safety | Pure Rust (unsafe surface = tch FFI only) | C/C++ extensions |
| Deployment | Copy and run | Environment rebuilds |
| Auditability | Every loss/algorithm hand-written with hand-computed unit tests | Opaque high-level APIs |

## Verified Results (reproducible; configs ship with the repository)

**Five-task baselines on real data**

| Task | Dataset | Result |
|---|---|---|
| Object detection | coco128 (real COCO subset, 300ep, CPU, from scratch) | mAP50 **0.846** / mAP50:95 0.521 |
| Instance segmentation | coco8-seg (real COCO subset) | mask mIoU **0.978** / R@0.5 1.000 |
| Classification | ImageNette (9,469 real images) | top1 **0.623** (44-second training) |
| Keypoints | coco8-pose (real COCO subset) | PCK@0.5 **0.815** |
| Oriented bounding boxes | dota8 (real aerial DOTA) | Full pipeline, GPU-converged |

**Backbone and pre-training ablation** (automotive part instance segmentation, 526/93 images,
2,219 instances, 120ep)

| Backbone | Pre-training | val mIoU | R@0.5 | P@0.5 |
|---|---|---|---|---|
| ResNet18 | Full ImageNet import (11.2M params) | **0.855** | 0.985 | 0.980 |
| **CSP-ELAN** (YOLOv8-native) | Full YOLOv8n backbone (3.2M params) | **0.846** | 0.979 | 0.974 |
| ResNet18 | None | 0.829 | 0.974 | 0.974 |
| simple-cnn | YOLOv8n stem | 0.799 | 0.950 | 0.883 |
| simple-cnn | None | 0.786 | 0.930 | 0.893 |
| DINOv2 ViT-S/14 | DINOv2 (448px / 672px) | 0.746 / 0.786 | — | — |

Full per-class metrics and measurement definitions: [runs/COMPARISON.md](runs/COMPARISON.md).

**Engineering performance**

- Instance-segmentation data pipeline rewrite: **288 s → 3 s per epoch (96×)** —
  content-patch caching + rayon-parallel encoding + double-buffered prefetch +
  GPU-resident datasets with on-GPU augmentation (`[data].cache`)
- Throughput A/B on the same GPU/dataset/batch: **3.0 s/epoch vs Ultralytics
  YOLO11n-seg 4.0 s/epoch**
- Training throughput benchmark (same protocol, csp-elan@640): tch 7.5 s/epoch,
  burn-wgpu 27 s/epoch (cross-vendor GPU track, [M2 report](runs/M2-BURN-BENCHMARK.md))

## Core Features

- **Five tasks, one training system** — TAL + CIoU + DFL (detection), YOLACT prototype
  masks + Dice (segmentation), KFIoU + rotated NMS (OBB), OKS (keypoints), classification;
  every loss ships with hand-computed unit tests
- **CSP-ELAN production backbone** — layer-for-layer compatible with the Ultralytics YOLOv8
  backbone; the full official `yolov8n.pt` backbone (162 tensors) imports through a single
  mapping rule. Also included: ResNet18 (ImageNet, verified 100/100 tensor import) and
  DINOv2 ViT-S/14 (positional-embedding grid interpolation + QKV fusion import)
- **Bidirectional pretrained-weight channel** — import: safetensors + regex layer mapping
  (partial loading with full reporting); export: safetensors (verified readable from Python)
- **GPU training** — verified on RTX 50 series (Blackwell/sm_120); requires only the graphics
  driver, no CUDA Toolkit; built-in runtime fixes for two upstream Windows defects
- **Data pipeline v2** — `[data].cache = auto/gpu/ram/off` tiered caching; `--resume`
  checkpoint continuation; periodic `last.ckpt` snapshots; evaluation protocols covering
  mIoU / recall / precision / per-class metrics
- **Dual-backend track** — tch (performance, NVIDIA) + burn-wgpu (zero-install track,
  NVIDIA/AMD/Intel; spike verified trainable, 17/17 tests)
- **Production engineering** — `.avpack` single-file dataset container (blake3-verified),
  SAHI-style tiled inference for large images, live SSE training panel, one-command
  environment setup (Windows/Linux)
- **Pluggable** — custom backbones/heads in three steps: implement a trait, register,
  select in config

## Quick Start

Requirements: Rust stable + MSVC build tools (Windows). The CPU path requires no extra setup
(CPU libtorch downloads automatically on first build); the GPU path is one command.

```powershell
# Windows: auto-detects GPU and configures (Linux: ./scripts/setup-env.sh)
.\scripts\setup-env.ps1

cargo build --release -p av-runtime          # produces target\release\av-runtime.exe

# Generate a config from your data directory → train → infer
set AV=target\release\av-runtime.exe

%AV% init --task seg --data E:\data\my_dataset --out configs\my.toml
%AV% train -c configs\my.toml
%AV% infer -w runs\my\best.ckpt --input sample.png
%AV% eval  -w runs\my\best.ckpt --report report.json
```

Dataset layouts follow the Ultralytics directory convention (`images/<split>` +
`labels/<split>`); YOLO txt, COCO-seg polygons, COCO-pose, ImageFolder, and DOTA
quadrilaterals read out of the box.

## Workspace Layout

| Crate | Responsibility | libtorch |
|---|---|---|
| `av-core` | Config / geometry / types / format utilities | No |
| `av-pretrain` | Pretrained-weight import adapter (safetensors + layer mapping) | Optional |
| `av-tasks` | Backbones / heads / losses / augmentation / assigners | Optional (`torch`) |
| `av-plugins` | Backbone plugin registry | Optional |
| `av-runtime` | Training/inference engine, CLI, panel, avpack | Yes |
| `av-burn` | burn-wgpu backend verification spike | No (burn) |

## Documentation

- [docs/USAGE.md](docs/USAGE.md) — full CLI reference, per-task data formats, GPU setup,
  benchmark details, known-issues ledger
- [runs/COMPARISON.md](runs/COMPARISON.md) — backbone and pre-training ablation study
- [runs/M2-BURN-BENCHMARK.md](runs/M2-BURN-BENCHMARK.md) — dual-backend throughput benchmark
- [scripts/setup-env.ps1](scripts/setup-env.ps1) / [setup-env.sh](scripts/setup-env.sh) —
  one-command environment setup

## Honest Boundaries

Every capability claim maps to a reproducible measurement and unit tests; unfinished work is
labeled just as precisely: mosaic/mixup currently applies to detection only; AMP and
multi-GPU are milestone-scheduled; burn-wgpu training throughput is ~1/3.6 of tch
(positioned as the deployment/cross-vendor track); DINOv2 resolution is constrained by its
patch-14 grid. The full ledger lives in USAGE §8 (Known Issues).

## License

Dual-licensed under MIT OR Apache-2.0.
