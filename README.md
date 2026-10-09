<div align="center">

# AegisVision

**Rust 原生多任务视觉训练与推理框架**

目标检测 · OBB 旋转框 · 实例分割 · 关键点 · 图像分类 —— 五任务一套训练体系
单二进制 · 零 Python · 可嵌入（crates.io） · ONNX 跨平台出口

[![crates.io](https://img.shields.io/crates/v/aegisvision-runtime.svg)](https://crates.io/crates/aegisvision-runtime)
[![docs.rs](https://img.shields.io/docsrs/aegisvision-runtime)](https://docs.rs/aegisvision-runtime)
[![license](https://img.shields.io/crates/l/aegisvision-runtime.svg)](https://github.com/javpower/aegis-vision)

[English](README_EN.md) | 简体中文

</div>

---

## ⚡ 一条命令训练

```bash
# 开箱即跑：合成数据冒烟（无需数据集）
av-runtime train -c configs/quick_detect.toml
# 真实数据 = Ultralytics 目录约定或 data.yaml，类数自动探测
av-runtime train --data <数据集目录|data.yaml> --imgsz 1280 --epochs 100
```

自动装配：csp-elan 骨干（YOLOv8 同构）+ 官方预训练导入 + mosaic/flip/HSV 增强 +
**AMP 混合精度 + 内容贴片缓存 + 双缓冲预取** + EMA + 按 fitness 保存 best.ckpt。
复杂场景（OBB/分割/关键点/自定义骨干）用 TOML 配置文件，字段见 `av-runtime init` 生成模板。

## 📊 实测：对标 Ultralytics YOLO26n

同机（RTX 5060 Ti）同数据同 1280 输入同 100 epoch，工业玻璃 logo 检测
（3 类，1296/144 图，5472×3648 原图直训）：

| | mAP50 | mAP50:95 | R@0.5 | 每 epoch |
|---|---|---|---|---|
| **AegisVision（本仓库）** | **0.9996** | 0.734 | **1.000** | 34 秒 |
| Ultralytics YOLO26n | 0.995 | **0.751** | 1.000 | **31.3 秒** |

> 无 Python 运行时、模型权重与训练过程全程可审计；精度与速度已进入同一量级。

更多真实数据基线（coco128 检测 / ImageNette 分类 / coco8-seg / coco8-pose /
dota8 OBB / 骨干×预训练七臂对照）见 [docs/USAGE.md §5](docs/USAGE.md)。

## 📦 作为库嵌入（应用内在线训练）

```toml
[dependencies]
aegisvision-runtime = "0.3"
```

```rust
use av_core::config::RunConfig;
use std::path::Path;

// 配置 = TOML 文本（或程序化构造 RunConfig）
let cfg: RunConfig = toml::from_str(&toml_text)?;

// 应用内训练 → 推理 → 跨生态导出，全在宿主进程内完成
let report = av_runtime::api::train(&cfg)?;
let detections = av_runtime::api::infer(&cfg, Path::new("runs/run/best.ckpt"),
                                        Path::new("sample.jpg"))?;
av_runtime::api::export_safetensors(&cfg, Path::new("runs/run/best.ckpt"),
                                    Path::new("best.safetensors"))?;
```

五个 crate 按层发布：`aegisvision-core`（配置/几何）→ `pretrain`（权重导入）→
`tasks`（骨干/头/损失）→ `plugins`（插件注册）→ `runtime`（引擎/CLI/面板）。

## 🦀 纯 Rust 后端（aegisvision-burn，无 libtorch）

不想装 libtorch/C++ 工具链？burn 框架后端提供同款分割链路（CSP-ELAN +
YOLACT 式原型×系数），数据格式、conf/iou 语义、`SegInstance` 输出与主后端
完全一致，训练/推理一条命令：

```bash
cargo install aegisvision-burn --features wgpu   # GPU 版（CPU 版去掉 --features）
avb train   --data <目录|data.yaml> --epochs 100 --device gpu
avb predict --weights runs-avb/<name> --input img.jpg --save-viz out/ --json res.json
```

后端选型建议：生产训练/推理用主后端 `av-runtime`（tch/libtorch，最快）；受限环境
（无 libtorch、纯 Rust 构建）用 `avb`（wgpu/ndarray，wgpu 实测约为 tch 的
1/3 吞吐）。详见 [crates.io/crates/aegisvision-burn](https://crates.io/crates/aegisvision-burn)。

## 🌐 跨平台推理（ONNX）

```bash
av-runtime export -w runs/run/best.ckpt --format safetensors
python scripts/export_onnx.py --backbone csp-elan --ckpt best.safetensors \
    --out best.onnx --imgsz 640 --classes 3 --verify
```

C#/Java/C++/JS/Android 的 ONNX Runtime 直接消费；预处理（/255、归一化）
已并入图内。四个骨干全部支持，导出与 Rust 推理数值对拍 ≤ 1e-5。

## 🧩 五任务 × 四骨干

| 骨干 | classify | detect / OBB | seg | keypoint |
|---|---|---|---|---|
| csp-elan（YOLOv8 同构） | ✅ | ✅ | ✅ | ✅ |
| resnet18（ImageNet 全量导入） | ✅ | ✅ | ✅ | ✅ |
| dinov2（ViT-S/14 官方权重 + QKV 融合） | ✅ | ✅（448 倍数尺寸） | ✅ | ✅ |
| simple-cnn（教学/冒烟） | ✅ | ✅ | ✅ | ✅ |

自定义骨干：实现 trait → 注册 → 配置选用（`av-plugins` 三步接入）。

## ✨ 工程能力

- **五任务损失全手写 + 手算对照单测**：TAL + CIoU + DFL / KFIoU + 旋转 NMS /
  YOLACT 原型掩码 + Dice / OKS / 交叉熵
- **数据管线 v2**：内容贴片缓存 + 流式构建（60MP 原图直训免降采样）+
  双缓冲预取 + SIMD 缩放 —— 实测 195 → 34 秒/epoch（同数据同分辨率）
- **训练现代化**：AMP fp16（动态梯度缩放）、EMA、余弦退火、梯度累加/裁剪、
  `--resume`（seg）、确定性播种
- **.avpack 单文件数据容器**（blake3 校验 + mmap 零拷贝）、SAHI 式切片推理、
  SSE 实时观测面板、`--save-viz` 可视化落盘
- **预训练双向通道**：safetensors 导入（正则层映射 + 部分加载报告）/
  导出（Python 生态实测可读）
- GPU：RTX 50 系（Blackwell/sm_120）实测，仅需显卡驱动

## ⚙️ 快速开始

环境要求：Rust stable + MSVC（Windows）。CPU 路径零配置（首次构建自动下载
CPU libtorch）；GPU 一条命令自动配置。

```powershell
.\scripts\setup-env.ps1                       # Windows（Linux: ./scripts/setup-env.sh）
cargo build --release -p aegisvision-runtime  # 产出 target\release\av-runtime.exe

av-runtime train --data <数据集或 data.yaml> --imgsz 640   # 一条命令训练
av-runtime infer -w runs/<id>/best.ckpt --input sample.jpg --save-viz viz
av-runtime eval   -w runs/<id>/best.ckpt --report report.json
```

## 📚 文档

- [docs/USAGE.md](docs/USAGE.md) —— CLI 全命令、五任务数据格式、GPU 配置、
  基准详表（coco128/ImageNette/coco8-seg/coco8-pose/dota8 + 增强对照）、
  骨干×任务矩阵、已知问题档案

## 🎯 诚实边界

每个能力声明对应可复现的实测与单测，同时如实标注未完成项：mosaic/mixup 增强
目前仅检测任务；AMP/EMA 仅 detect 路径；多卡按里程碑排期；ONNX 导出走 Python
侧车（部署端零 Python，导出端需 torch）；`--resume` 目前仅 seg；burn-wgpu
后端为跨厂商技术验证轨（吞吐 ≈ tch 的 1/3.6）。完整档案见 USAGE §8。

## License

MIT OR Apache-2.0 双许可。
