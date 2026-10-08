# AegisVision 详细使用文档

> 入门与功能状态见 [README](../README.md)。本文是完整参考：CLI 全命令、库引入、
> 数据格式、GPU 配置、基准详表、发布清单、已知问题档案。

---

## 1. 环境要求与安装

| 项 | CPU 路径（默认） | GPU 路径 |
|---|---|---|
| Rust | stable MSVC（rust-toolchain.toml 已锁） | 同左 |
| libtorch 2.11 | 首次构建自动下载 CPU 版（~200MB） | 自动下载 cu128 版（~2.8GB，含 RTX 50 系 sm_120 内核） |
| 显卡 | 无要求 | NVIDIA（只装驱动，无需 CUDA Toolkit）；AMD 见下 |

**免编译路径（推荐给纯使用者）**：GitHub Releases 提供预编译 `av-runtime`
（Windows/Linux/macOS CPU 版 + Windows CUDA 版，推 `v*` 标签自动构建，
见 `.github/workflows/release.yml`）——下载解压即用，CUDA 版运行环境仍需
`setup-env.ps1` 下载 libtorch（见 §6）。以下内容面向需要编译（开发者/嵌入方）。

```powershell
# Windows 一键配置（自动探测 GPU：有 N 卡装 cu128，无卡走 CPU；AMD 如实回退并说明）
.\scripts\setup-env.ps1                  # 加 -SmokeTest 顺带跑合成数据冒烟验证
# Linux: ./scripts/setup-env.sh          # 同款策略；ROCm 待上游 tch/libtorch 支持
# 之后编译/训练：
cargo build --release -p av-runtime
```

libtorch 下载源为 PyTorch 官方（download.pytorch.org，已验证 2.11.0 cu128/cpu 的
win/linux 文件名），安装到 `<仓库>\.libtorch\`（已 gitignore），检测到既有安装
（含 `E:\libs\libtorch-cu128-2.11` 旧约定路径）会直接复用；`-Persist` 写用户级
环境变量。环境配置细节（libtorch 路径/长路径）：`scripts/setup-env.ps1`。

---

## 2. CLI 命令参考

| 命令 | 用途 | 状态 |
|---|---|---|
| `av init --task <t> --data <dir> --out <f>` | 生成最小配置；自动探测数据格式（YOLO/ImageFolder/DOTA）+ 统计类数 | ✅ |
| `av train -c <toml> [--dry-run] [--override k=v]` | 训练；`--resume` 续训 | ✅ |
| `av infer -w <ckpt> [--config <toml>] --input <图>` | 推理（JSON 输出，坐标已还原原图） | ✅ |
| `av infer --slice --slice-window N` | 高分辨率大图切片推理（SAHI 式） | ✅ |
| `av eval -w <ckpt> [--report f]` | 评测（检测 mIoU/mAP50/mAP50:95；分类 top1；OBB 旋转 mIoU；关键点 PCK/OKS） | ✅ |
| `av pack --src <dir> --out <f>` | 打包 .avpack 单文件容器（blake3 逐文件校验） | ✅ |
| `av export -w <ckpt> --format safetensors` | 权重导出（跨生态互认，Python safetensors 实测可读） | ✅ |
| `av panel --port 8080` | 观测面板（runs 浏览 + 指标视图，5 秒刷新） | ✅ |
| `av distill` / `av nas` | 蒸馏 / 骨干选型 | ⏳ M9 |

通用约定：错误信息三段式（发生了什么/为什么/下一步）；训练产物统一落
`runs/<run_id>/{best.ckpt/, config.snapshot.toml, report.json}`。

---

## 2A. Rust 库引入（其他项目）

分层依赖按需取用：

```toml
[dependencies]
av-core = "0.1"      # 几何/配置/格式工具：无 libtorch，二进制极小
av-runtime = "0.1"   # 完整训练/推理引擎
av-pretrain = "0.1"  # 预训练权重导入
```

**训练**：

```rust
use av_core::config::RunConfig;
use av_runtime::engine;

let cfg = RunConfig::from_path("configs/detect_coco8.toml")?;
let report = engine::train(&cfg)?;
println!("{} = {}", report.metric, report.metric_value);
```

**推理**（普通 + 大图切片）：

```rust
let result = engine::infer(&cfg, weights, input)?;
let result = engine::infer_sliced(&cfg, weights, input, 256, 0.2)?;
```

**纯工具（无 libtorch 项目也能用）**：

```rust
use av_core::geometry::{Aabb, RotBox};
use av_core::types::nms;

let iou = Aabb::new(0.,0.,10.,10.).iou(&Aabb::new(5.,0.,15.,10.));  // 0.33
let kept = nms(detections, 0.5);
let riou = rot_a.iou(&rot_b);   // 旋转框 IoU（Sutherland–Hodgman）
```

**.avpack 容器读写**：

```rust
av_runtime::avpack::pack_dir(Path::new("data/coco8"), Path::new("data/c.avpack"))?;
let r = av_runtime::avpack::AvPackReader::open(Path::new("data/c.avpack"))?;
let jpeg = r.read("images/train/000000000009.jpg")?;
```

**自定义骨干三步接入**：实现 `BaseBackbone`（参考 backbone.rs / backbone_resnet.rs /
backbone_dino.rs）→ `av_plugin!(Backbone, "名字")` → 配置 `family = "名字"`。

**非 Rust 系统集成**：CLI 子进程（JSON 输出）与 .avpack 公开格式现在即可用；
ONNX 导出按 M8 交付后任意 ONNX Runtime 语言消费。

---

## 3. 五任务与数据格式

| 任务 | kind | 数据格式 | 真实数据实测（本机 CPU） |
|---|---|---|---|
| 目标检测 | `detect`（obb_mode=false） | YOLO txt / .avpack / DOTA+OBB | coco128：mAP50=0.846, mAP50:95=0.521, mIoU=0.879 |
| 分类 | `classify` | ImageFolder | ImageNette：val top1=0.623（44 秒训练） |
| OBB 旋转框 | `detect` + `obb_mode=true` | DOTA 4 角点 | GPU 640px 训练中（ep1 即 mIoU=0.253） |
| 实例分割 | `seg` | COCO 多边形 txt | 训练集掩码 mIoU=0.978, R@0.5=1.0 |
| 关键点 | `keypoint` | COCO pose txt | 训练集 PCK@0.5=0.815, meanOKS=0.580 |

数据集（全部真实、Ultralytics 官方格式兼容）：coco8（检测/分割/姿态冒烟）、
coco128、dota8、ImageNette。打包为单文件容器：`av pack --src data/coco8 --out data/coco8.avpack`。

快速上手模板（合成数据，无需数据集）：`configs/quick_detect.toml`、`quick_classify.toml`。

---

## 4. 训练体系与已知边界

**已实现**：TAL 标签分配 + CIoU + DFL（检测）；KFIoU + 角度分支 + 旋转 NMS（OBB）；
YOLACT 式 MaskBranch + Dice（分割）；直接回归 + OKS（关键点）；rayon 多核并行解码；
梯度累加 + 裁剪 + 余弦退火 + EMA；letterbox；增强流水线（翻转/HSV/尺度抖动 +
close_last_epochs）；预训练权重导入（safetensors）；GPU 训练（resolve_device +
CUDA shim）；观测面板；.avpack 容器。

**数据管线 v2（`[data].cache`，seg 增强路径）**：旧路径每个 epoch 从原始全分辨率
重编码（单线程串行，GPU 利用率 <5%）。v2 一次性缓存 letterbox 内容贴片，逐 epoch
只在小图上增强：`auto`（默认，画布堆 ≤4GiB 且有 CUDA → 显存驻留，否则内存缓存）/
`gpu`（显存驻留 + GPU 张量增强，CPU 零参与）/ `ram`（rayon 并行 + 双缓冲预取，
GPU 与编码重叠）/ `off`（历史路径）。语义对齐：掩码坐标逐位一致；s=1 无增益时
像素逐位一致；增益/缩放路径为定点次序差（单测容差锁定），none/flip 逐位一致。

**数据管线 v2（`[data].cache`，detect 增强路径）**：同款内容贴片缓存——mosaic/mixup/
flip/hsv/scale 全部在贴片上进行，全分辨率重采样一次性完成。dir 源为**流式构建**
（逐文件解码→缩放→丢弃，可直接吃 5472×3648 原始数据集，无需预降采样）；avpack
源就地转换；`off` 保留 raw 逐 epoch 路径（gpu 档未实现，按 ram）。none-plan 编码
与 raw 路径逐位一致；mosaic/mixup 因重采样链不同像素有差（增强语义不变）。
配套优化：批间双缓冲预取（编码线程与 GPU 计算重叠）+ SIMD 缩放
（fast_image_resize，Bilinear 卷积核统一全部数据准备路径）。实测 glass-logo
（5472×3648 原图直训，1280 输入）：195 → 43 秒/epoch（4.5×）。
1280 输入建议 `batch_size ≤ 4` + `RAYON_NUM_THREADS ≤ 16`（commit 内存稳定 ~38GB；
b8/24 线程在长训练后段有 commit 耗尽风险）。已知边界：detect/OBB/关键点尚无
周期存盘，`--resume` 目前仅 seg 支持。

**推理可视化**：`av infer --save-viz <DIR>` 把检测框（按类 8 色循环着色）与
`C<id> <score>` 标签画在原图上存为 `<DIR>/<文件名>.jpg`（无第三方绘图依赖，
内置 3×5 微型字模）。

**骨干 × 任务支持矩阵**（`backbone.family` 配置选择；keypoint 此前硬编码
simple-cnn、detect 未接 dino，现已全量接线并有用例锁定）：

| 骨干 | classify | detect/OBB | seg | keypoint | ONNX 导出 |
|---|---|---|---|---|---|
| csp-elan | ✅ | ✅（P3/P4/P5，width/depth 可缩放） | ✅ | ✅ | ✅ detect 图 |
| resnet18 | ✅ | ✅（layer1/2/3 = stride 4/8/16） | ✅ | ✅ | ✅ classify 图 |
| dinov2 | ✅ | ✅（ViTDet 式金字塔 conv 分支随任务训练；**img_size 须为 448 倍数**） | ✅ | ✅ | ✅ classify 图 |
| simple-cnn | ✅ | ✅ | ✅ | ✅ | ✅ classify 图 |

ONNX 导出：`python scripts/export_onnx.py --backbone <名称> --ckpt <safetensors>
--imgsz <尺寸> --classes <N> --verify`（dinov2 的官方 HF checkpoint 自动套用
命名适配 + QKV 融合；预处理已并入图内）。

**AMP 混合精度（`[train].amp`，detect 路径）**：CUDA 上 fp16 autocast +
动态梯度缩放（torch GradScaler 同款：放大 loss 反向、反缩放前检测 inf/NaN
跳步回退、稳定后按 2000 步间隔放大）。tch 的 autocast 固定 fp16 且无原生
GradScaler，缩放器为手写实现；CPU 自动 no-op。评测推理同走 autocast。
实测同数据同 batch：步进 ~130ms → ~87ms（fp32 → fp16，b4）。

**EMA + best 选择（detect 路径）**：EMA 影子权重每个 optimizer step 更新
（前期衰减调度 `d = min(decay, (1+t)/(10+t))`，与 Ultralytics ModelEMA 同款），
eval 与 checkpoint 消费影子权重；`best.ckpt` 按
`fitness = 0.9·mAP50:95 + 0.1·mAP50` 保存最优 EMA 快照（不再是"最后一轮"）。

**按里程碑排期（PLAN §8）**：mosaic/mixup 增强、heatmap/SimDR 关键点档、
RoIAlign 精度分割档、RF-DETR 头（M6）、多卡/memmap 容器（M7）、
ONNX/TensorRT 导出（M8，Python 侧车）、蒸馏/NAS（M9）。

---

## 5. 基准详表（全部本机 CPU 实测，无合成数据造假）

### 5.1 检测 coco128（128 图真实 COCO，80 类，300ep，与 YOLOv8n 同口径）

| 项 | YOLOv8n 8.4.148（COCO 预训练起步，CPU 36.3 分钟） | AV tiny（从零，CPU ~2 小时） |
|---|---|---|
| mAP50 | 0.820 | **0.846** |
| mAP50:95 | **0.681** | 0.521 |

差距归因：预训练权重（YOLOv8n 2 epochs 即 mAP50=0.539 的实测证明）+ 模型容量
（3.2M vs <0.1M 演示骨干）。收敛路径：预训练导入（ResNet18 已通）→ csp-elan 正式
骨干 → P2 层。工程维度 AV 结构性优势：单二进制、零 Python 运行时、全内存安全。

### 5.2 分类 ImageNette（9469+3925 真实图，10 类）

| 项 | AV v0.2 |
|---|---|
| val top1（20ep，44 秒全程） | 0.623 |
| ResNet18 预训练导入 | 100/100 张量（torchvision 官方权重） |

### 5.3 实例分割 coco8-seg（8 训练图过拟合验收）

| 项 | AV v0.2 |
|---|---|
| 训练集掩码 mIoU / R@0.5 | 0.978 / 1.000 |
| val（4 图） | 泛化基线（需真实规模数据） |

### 5.4 关键点 coco8-pose（7 实例 92 可见点）

| 项 | AV v0.2 |
|---|---|
| 训练集 PCK@0.5 / meanOKS | 0.815 / 0.580 |
| val PCK@0.5 | 0.259（4 图泛化基线） |

### 5.5 OBB dota8（真实航拍，15 类）

**第一期根因定位与修复（OBB 调试官）**：所有配置（CPU/GPU、320/640px、150-800ep）
旋转 mIoU 恒 ≈0.005 的真凶 = `loss_obb` 正样本权重掩码 `pos_w[flat]` **漏写批索引 gi**
——bs=4 时图 1-3 的回归分支完全得不到梯度（bs=1 探针不可见，故单测全过）。一行修复 +
双向回归测试（buggy FAIL / 修复 PASS）。

| 配置 | 旋转 mIoU | R@0.5 | 说明 |
|---|---|---|---|
| 320px 150ep（修复前） | 0.008 | 0.000 | 320 后目标 2-6px 物理不可见 + pos_w bug |
| 640px 400ep GPU（修复后） | **0.403@ep50，训练中** | — | 修复 + 高分辨率叠加生效，10 倍以上改善 |

### 5.6 数据增强 A/B（coco8-pose 关键点，200ep）

| 臂 | train PCK | val PCK |
|---|---|---|
| 无增强 | 0.826 | 0.224 |
| flip+HSV+scale | 0.663 | 0.224（持平） |

结论：4 图量级下训练方差（±0.1）大于增强效应；泛化的真实杠杆是预训练与数据量。
增强机制已就绪，等真实规模数据生效。

### 5.7 数据增强 A/B（coco8 检测 mosaic/mixup，400ep，数据增强官二波）

三臂同 seed=3 / CPU / simple-cnn / img320 / 400ep / close_last_epochs=40，
基线臂 = flip 0.5 + hsv ±0.1 + scale [0.9,1.1]（§5.6 同款）；mosaic 臂追加
`mosaic = 1.0`（4 图拼 1，2×2 网格 + 拼接线裁剪）；mosaic+mixup 臂再追加
`mixup = 0.2`（λ~Beta(0.2,0.2) 像素加权，框/类别取并集）。串联顺序
mosaic → mixup → flip → hsv → scale（YOLO 惯例）。配置：
`configs/detect_coco8_ab_aug.toml` / `detect_coco8_ab_mosaic.toml` /
`detect_coco8_ab_mixup.toml`，产物在 `runs-aug2/`。

| 臂 | val mIoU（末 epoch） | val R@0.5 | val mAP50 | val mIoU（41 次评测最优） | val mIoU（末 10 次均值） | train mIoU | train mAP50 |
|---|---|---|---|---|---|---|---|
| 基线 flip+hsv+scale | 0.318 | 0.250 | 0.000 | 0.452 | 0.208 | 0.981 | 1.000 |
| +mosaic 1.0 | 0.332 | 0.500 | 0.000 | 0.470 | 0.345 | 0.936 | 0.857 |
| +mosaic 1.0 +mixup 0.2 | **0.359** | 0.500 | 0.000 | **0.489** | **0.374** | 0.843 | 0.688 |

读数说明（如实）：

- **val mAP50 三臂全程 0.000**（41 次评测无一非零）——tiny 骨干 + 4 图 val 上
  预测框始终进不了 mAP 判定区间，该指标在本量级无区分度，比较以 mIoU/R@0.5 为准。
- **val：mosaic/mixup 单调向好但幅度在训练方差内**。末 epoch 0.318 → 0.332 →
  0.359，全程最优 0.452 → 0.470 → 0.489，末 10 次均值 0.208 → 0.345 → 0.374
  （收尾段稳定性改善最明显：基线末期 mIoU 崩到 0.13~0.26，mosaic 臂稳在
  0.33~0.46）。与 §5.6 关键点 A/B 结论一致：4 图量级下方差大于效应。
- **train 拟合度按预期随增强强度递减**（0.981 → 0.936 → 0.843 mIoU；mAP50
  1.000 → 0.857 → 0.688）：mosaic 的 4 图拼画布改变了目标尺度/位置分布、
  mixup 像素混合稀释单图证据——强正则生效的直接证据；train 评测视角是
  「未增强原始图」，对重增强臂天然偏严。
- **代价**：全程耗时 基线 8.8 min → mosaic 32 min → mosaic+mixup 34 min
  （每样本 4 图组合作画 + partner resize 是纯 CPU 图像管线成本，GPU 化或
  缓存可摊薄）。

结论：mosaic/mixup 在 4 图量级已表现出方向一致的正向信号（val 最优/收尾段
均值 +0.04~+0.17 mIoU）与明确的正则效应（train 拟合递减），精度杠杆仍需真实
规模数据兑现；机制、坐标同步与复现性（固定 RNG 消耗序）已就绪并被单测锁定。

---

## 6. GPU 训练（RTX 5060 Ti 实测打通）

**结论：Windows 原生 CUDA 可用**——tch 0.24（libtorch 2.11）+ cu128 发行版原生含
sm_120 内核；两个必须处理的坑已有内建方案：

1. tch 0.26（libtorch 2.13）MSVC 胶水编译失败（上游 bug）→ 锁 tch 0.24；
2. torch-sys ≥0.20 不自动加载 torch_cuda.dll → `cuda_link.rs` 进程启动时
   LoadLibrary 修复（否则 CUDA 静默回退 CPU）。

```powershell
# 构建（cu128 libtorch 2.85GB，解压至 E:\libs\libtorch-cu128-2.11）
$env:LIBTORCH = "E:\libs\libtorch-cu128-2.11"
CARGO_TARGET_DIR=target-gpu cargo build --release -p av-runtime
# 运行（无"回退 CPU"告警 = 真在 GPU）
$env:PATH = "E:\libs\libtorch-cu128-2.11\lib;$env:PATH"
.\target-gpu\release\av-runtime.exe train -c configs/detect_coco8.toml
```

实验矩阵与证据：初版 gpu 实验归档（cu121 内核对 sm_120 不可用的过程记录保留于
项目历史）。`resolve_device` 在 CUDA 不可用时自动回退 CPU 并告警，`Cuda` 探测/
cudnn/内核启动三级检查内建。

---

## 7. 发布清单（crates.io 就绪状态）

- [x] 四 crate + av-pretrain 元数据（description/license/keywords/categories/readme）
- [x] LICENSE-MIT + LICENSE-APACHE 双许可
- [x] 薄 README（crates.io 页面渲染）× 5
- [x] examples/ 可编译示例
- [x] `cargo package --list` 零警告
- [ ] 仓库公开后补 repository URL（Cargo.toml 注释占位已留）
- [ ] BENCHMARK §3 的 YOLOv8n 300ep 对比表定稿（基线已实测完成）
- [ ] 版本号策略：deny_unknown_fields 配置字段增删 = 破坏性变更（SemVer）
- [ ] 按依赖序 publish：av-core → av-pretrain → av-tasks → av-runtime

## 8. 已知问题档案（全部有据）

| # | 问题 | 处置 |
|---|---|---|
| 1 | tch 0.26（libtorch 2.13）MSVC 14.44 胶水编译失败（C7555/C2059 级联，上游 bug） | 锁 tch 0.24（2.11 可编译）；勿擅自升级 |
| 2 | tch 0.17 VarStore::save/load legacy 格式 Windows 损坏 | 自研目录式 checkpoint（blake3 校验） |
| 3 | torch-sys ≥0.20 不自动加载 torch_cuda.dll | cuda_link.rs LoadLibrary shim |
| 4 | avpack 读取为整集载入内存 | 已闭环：`AvPackReader` 持有 memmap 只读映射，容器不进堆、条目字节零拷贝切片、按名 O(1) 索引（新增 `entry`/`bytes`；`read` 保留条目级 blake3 校验语义） |
| 5 | 训练不可逐位复现（模型初始权重走 torch 全局 RNG） | `deterministic = true` 已接线：训练入口播种 `tch::manual_seed(seed)`，实测两次运行 epoch1 loss 逐位一致；数据侧 shuffle/增强本就由 seed 驱动的 XorShift 决定 |
| 6 | 直接运行 exe 需 libtorch\lib 在 PATH | cargo run 自动注入；setup-env.ps1 已设 |

---

## 9. 路线图（从当前 v0.2 到发布）

| 波次 | 内容 | 前置 |
|---|---|---|
| 3 | 预训练 A/B 矩阵（五任务 × 从零/预训练）、coco128 全增强复测、碎片框合并 | ResNet18 ✓ |
| 4 | DINOv2/ViT 骨干族（进行中）、csp-elan 正式骨干、P2 层 | GPU ✓ |
| 5 | COCO val2017 5k 评测校准（对齐 pycocotools）、AMP/多卡 | 波次 3 |
| 6 | ONNX/TensorRT 导出侧车、观测面板 SSE 实时流 | 波次 5 |
| 7 | BENCHMARK 定稿（四维对比表）→ crates.io 发布 | 全部 |
