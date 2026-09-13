<div align="center">

# AegisVision

**Rust 原生多任务视觉训练与推理框架**

目标检测 · 实例分割 · OBB 旋转框 · 关键点 · 图像分类 —— 五任务共享一套训练体系

[English](README_EN.md) | 简体中文

</div>

---

## 为什么是 AegisVision

深度学习工程长期被 Python 生态垄断：训练靠 Python 脚本，部署靠导出链，系统边界靠胶水。
AegisVision 选择另一条路——**用 Rust 从零实现 YOLO 级训练体系**，把整条链路收敛为一个
内存安全、可审计、无解释器依赖的二进制。

| | AegisVision | 传统 Python 方案 |
|---|---|---|
| 运行时 | 单二进制，零 Python | Python + 数百依赖包 |
| 内存安全 | 全 Rust（unsafe 面 = tch FFI） | 依赖 C/C++ 扩展 |
| 部署 | 拷贝即用 | 环境重建 |
| 可审计性 | 每个损失/算法手写 + 手算对照单测 | 黑盒高层 API |

## 实测结果（全部可复现，配置随仓库交付）

**五任务真实数据基线**

| 任务 | 数据集 | 结果 |
|---|---|---|
| 目标检测 | coco128（COCO 真实子集，300ep，CPU，从零） | mAP50 **0.846** / mAP50:95 0.521 |
| 实例分割 | coco8-seg（COCO 真实子集） | 掩码 mIoU **0.978** / R@0.5 1.000 |
| 图像分类 | ImageNette（9469 图真实数据） | top1 **0.623**（44 秒训练） |
| 关键点 | coco8-pose（COCO 真实子集） | PCK@0.5 **0.815** |
| OBB 旋转框 | dota8（真实航拍） | 管线全通，GPU 收敛 |

**骨干与预训练七臂对照研究**（汽车零部件实例分割，526/93 图，2219 实例，120ep）

| 骨干 | 预训练 | val mIoU | R@0.5 | P@0.5 |
|---|---|---|---|---|
| ResNet18 | ImageNet 全量导入（1120 万参数） | **0.855** | 0.985 | 0.980 |
| **CSP-ELAN**（YOLOv8 同构） | YOLOv8n 完整骨干（320 万参数） | **0.846** | 0.979 | 0.974 |
| ResNet18 | 无 | 0.829 | 0.974 | 0.974 |
| simple-cnn | YOLOv8n stem | 0.799 | 0.950 | 0.883 |
| simple-cnn | 无 | 0.786 | 0.930 | 0.893 |
| DINOv2 ViT-S/14 | DINOv2（448px / 672px） | 0.746 / 0.786 | — | — |

完整分类别指标与口径说明见 [runs/COMPARISON.md](runs/COMPARISON.md)。

**工程性能**

- 实例分割数据管线重写后 **288 秒 → 3 秒/epoch（96 倍）**：内容贴片缓存 +
  rayon 并行编码 + 双缓冲预取 + 显存驻留与 GPU 张量增强（`[data].cache`）
- 同卡同数据同 batch 吞吐对照：**3.0 秒/epoch vs Ultralytics YOLO11n-seg 4.0 秒/epoch**
- 训练吞吐基准（同协议 csp-elan@640）：tch 7.5 秒/epoch，burn-wgpu 27 秒/epoch
  （跨厂商 GPU 轨道，[M2 报告](runs/M2-BURN-BENCHMARK.md)）

## 核心特性

- **五任务一套训练体系**——TAL + CIoU + DFL（检测）、YOLACT 原型掩码 + Dice（分割）、
  KFIoU + 旋转 NMS（OBB）、OKS（关键点）、分类；每个损失均带手算对照单测
- **CSP-ELAN 正式骨干**——与 ultralytics YOLOv8 backbone 逐层同构，官方 `yolov8n.pt`
  完整骨干权重（162 张量）经单条映射全量导入；另有 ResNet18（ImageNet 全量导入实测
  100/100）、DINOv2 ViT-S/14（pos_embed 网格插值 + QKV 融合导入）
- **预训练权重双向通道**——导入：safetensors + 正则层映射（部分加载 + 完整报告）；
  导出：safetensors（Python 生态实测可读）
- **GPU 训练**——RTX 50 系（Blackwell/sm_120）实测，仅需显卡驱动，无需 CUDA Toolkit；
  内建 Windows 特有两个上游缺陷的运行时修复
- **数据管线 v2**——`[data].cache = auto/gpu/ram/off` 四档缓存；`--resume` 断点续训；
  周期性 last.ckpt 存盘；评测协议含 mIoU / recall / precision / 分类别指标
- **双后端路线**——tch（性能轨，NVIDIA）+ burn-wgpu（零安装轨，NVIDIA/AMD/Intel，
  spike 已验证可训练，17/17 测试）
- **工程完备**——`.avpack` 单文件数据容器（blake3 校验）、SAHI 式大图切片推理、
  SSE 实时观测面板、ONNX 导出（规划）、一键环境脚本（Windows/Linux）
- **插件化**——自定义骨干/头三步接入：实现 trait → 注册 → 配置选用

## 快速开始

环境要求：Rust stable + MSVC 构建工具（Windows）。CPU 路径零额外配置
（首次构建自动下载 CPU libtorch）；GPU 路径一条命令自动配置。

```powershell
# Windows：自动探测 GPU 并配置（Linux: ./scripts/setup-env.sh）
.\scripts\setup-env.ps1

cargo build --release -p av-runtime          # 产出 target\release\av-runtime.exe

# 从数据目录生成配置 → 训练 → 推理
set AV=target\release\av-runtime.exe

%AV% init --task seg --data E:\data\my_dataset --out configs\my.toml
%AV% train -c configs\my.toml
%AV% infer -w runs\my\best.ckpt --input sample.png
%AV% eval  -w runs\my\best.ckpt --report report.json
```

数据格式与 Ultralytics 官方目录约定兼容（`images/<split>` + `labels/<split>`），
YOLO txt、COCO-seg 多边形、COCO-pose、ImageFolder、DOTA 四角点开箱即读。

## 工作区结构

| crate | 职责 | 依赖 libtorch |
|---|---|---|
| `av-core` | 配置/几何/类型/格式工具 | 否 |
| `av-pretrain` | 预训练权重导入适配（safetensors + 层映射） | 可选 |
| `av-tasks` | 骨干/头/损失/增强/分配器 | 可选（feature `torch`） |
| `av-plugins` | 骨干插件注册 | 可选 |
| `av-runtime` | 训练/推理引擎、CLI、面板、avpack | 是 |
| `av-burn` | burn-wgpu 后端技术验证 spike | 否（burn） |

## 文档

- [docs/USAGE.md](docs/USAGE.md) —— CLI 全命令、五任务数据格式、GPU 配置、
  基准详表、已知问题档案
- [runs/COMPARISON.md](runs/COMPARISON.md) —— 骨干与预训练七臂对照研究
- [runs/M2-BURN-BENCHMARK.md](runs/M2-BURN-BENCHMARK.md) —— 双后端吞吐基准
- [scripts/setup-env.ps1](scripts/setup-env.ps1) / [setup-env.sh](scripts/setup-env.sh) —— 一键环境

## 诚实边界

本项目的每个能力声明都对应可复现的实测与单测，同时如实标注未完成项：
mosaic/mixup 目前仅检测任务；AMP/多卡按里程碑排期；burn-wgpu 训练吞吐为 tch 的
约 1/3.6（部署/跨厂商轨定位）；DINOv2 受 patch-14 网格约束限定分辨率。
完整档案见 USAGE §8 已知问题。

## License

MIT OR Apache-2.0 双许可。
