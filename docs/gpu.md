# GPU 路线实验记录（Windows / MSVC / CUDA）

> 本文归档 AegisVision 在 RTX 5060 Ti（Blackwell，sm_120）上打通 Windows 原生
> CUDA 训练的两轮实验（实验 A / 实验二）。源码中 `docs/gpu.md 实验 A`、
> `实验二 §5` 等引用均指本文相应章节。

## 1. 结论

**Windows 原生 CUDA 可用**：tch 0.24（libtorch 2.11）+ cu128 发行版在 RTX 50 系
上内核实测可执行，仅需 NVIDIA 驱动（无需安装 CUDA Toolkit）。两个必须处理的坑
（版本配对、torch_cuda.dll 加载）已有内建方案，见 §2 与实验二 §5。

## 2. 版本配对（每个 tch 版本硬绑定一个 libtorch）

| tch | libtorch | 状态 |
|---|---|---|
| 0.24（**当前锁定**，`Cargo.toml` 用 `"=0.24.0"` 锁死） | 2.11.0 | ✅ MSVC 14.44 可编译；cu128 发行版原生含 sm_120 内核 |
| 0.26 | 2.13.0 | ❌ MSVC 胶水编译失败（C7555/C2059 级联，上游 bug，勿升级） |

tch 与 libtorch 的 ABI 一一对应，升级 tch 必须同步更换 libtorch 并重跑本实验链。

## 3. 实验 A：cu121 / libtorch 2.4 在 sm_120 上不可用

- **现象**：`Cuda::is_available()` / cudnn 探测全部通过，但首个 CUDA 内核启动即报
  `no kernel image is available for execution on the device`。
- **根因**：libtorch 2.4 的 CUDA 发行版是 cu121，官方预编译内核只覆盖到 sm_90；
  RTX 5060 Ti 是 sm_120——既无本架构 SASS，发行版又剥离了可驱动 JIT 的 PTX。
- **诊断工具**：`cargo run -p aegisvision-runtime --example cuda_smoke --features torch`
  （elementwise → matmul → conv2d 前向+反向 → 检测模型 loss+backward 逐级探测）。
- **结论**：该组合在 RTX 50 系上不可用，CUDA 路线需要 ≥ cu128 的发行版（实验二）。

## 4. 实验二：cu128 / libtorch 2.11 打通

1. **版本筛选**：逐版本编译试错，0.26/2.13 在 MSVC 14.44 下编译失败（§2 表），
   最高可用配对为 tch 0.24 + libtorch 2.11.0。
2. **内核可用性**：libtorch 2.11 的 cu128 发行版原生含 RTX 5060 Ti
   （sm_120/Blackwell）内核，`cuda_smoke` 全链路（含 conv 反向）实测通过。
3. **环境搭建**：下载 PyTorch 官方 cu128 libtorch 包（~2.8GB）解压后：

   ```powershell
   $env:LIBTORCH = "E:\libs\libtorch-cu128-2.11"        # 构建期
   $env:PATH = "E:\libs\libtorch-cu128-2.11\lib;$env:PATH"  # 运行期
   ```

   或直接用仓库脚本（自动探测 GPU 并下载/复用）：
   `.\scripts\setup-env.ps1`（Windows）/ `./scripts/setup-env.sh`（Linux）。
4. **引擎侧检查**：`resolve_device` 对 CUDA 做三级检查——DLL 加载（§5）→
   `Cuda::is_available()` / `device_count` → 微型内核探测（`sum` 真实启动一次
   内核并同步取回）。任一级失败自动回退 CPU 并 `tracing::warn`，绝不会静默
   假装在 GPU 上训练。运行时看到「回退 CPU」告警即环境未打通。
5. **Windows/MSVC 专项：必须显式加载 torch_cuda.dll**。torch-sys ≥ 0.20 移除了
   「引用 torch_cuda 符号」的 dummy 依赖（该结论只对 ELF 成立）：MSVC 链接器
   不链接无符号引用的导入库，torch_cuda.dll 不被加载、其静态注册器不运行，
   ATen dispatcher 里没有 CUDA 后端——任何 CUDA 算子报
   `Could not run 'aten::*' with arguments from the 'CUDA' backend`，
   `Cuda::is_available()` 恒为 false，引擎静默回退 CPU。修复：
   `crates/av-runtime/src/cuda_link.rs` 在请求 CUDA 时
   `LoadLibraryA("torch_cuda.dll")` 强制加载（幂等，失败按 CUDA 不可用兜底）。
   非 Windows（ELF/Mach-O）动态库随链接自动加载，无需补丁。
6. **验证命令**：

   ```powershell
   CARGO_TARGET_DIR=target-gpu cargo build --release -p aegisvision-runtime
   .\target-gpu\release\av-runtime.exe train -c configs/detect_coco8.toml
   # 无「回退 CPU」告警 = 真在 GPU
   ```

## 5. 相关源码索引

| 位置 | 内容 |
|---|---|
| `Cargo.toml`（workspace） | tch `"=0.24.0"` 锁版说明 |
| `crates/av-runtime/src/cuda_link.rs` | torch_cuda.dll LoadLibrary shim |
| `crates/av-runtime/src/engine.rs` `resolve_device` | 三级检查与 CPU 回退 |
| `crates/av-runtime/examples/cuda_smoke.rs` | 内核可用性逐级冒烟 |
| `scripts/setup-env.ps1` / `.sh` | 一键环境配置 |
| `docs/USAGE.md` §6 | 用户视角的 GPU 训练指南 |
