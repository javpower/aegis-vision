# AegisVision 一键环境配置（Windows）
#
# 用法：
#   .\scripts\setup-env.ps1                    # 自动探测：有 N 卡装 CUDA 版，否则 CPU 版
#   .\scripts\setup-env.ps1 -Device cuda       # 强制 CUDA（无 N 卡报错）
#   .\scripts\setup-env.ps1 -Device cpu        # 强制 CPU（零配置，libtorch CPU 版首次构建自动下载）
#   .\scripts\setup-env.ps1 -Persist           # 额外写入用户级环境变量（setx）
#   .\scripts\setup-env.ps1 -SmokeTest         # 配置完跑一次合成数据冒烟训练验证后端
#
# 行为：
#   - CUDA：下载官方 cu128 libtorch 2.11（与 Cargo.toml 锁定的 tch 0.24 配对，
#     原生含 RTX 50 系 sm_120 内核）到 <仓库>\.libtorch\ 并配置环境变量；
#     已有合法安装（含 E:\libs\libtorch-cu128-2.11 旧约定路径）直接复用。
#   - CPU：清掉 LIBTORCH 即可——torch-sys 会在首次构建时自动下载 CPU 版（~200MB）。
#   - AMD/Intel GPU：libtorch 2.11 官方没有 ROCm/DirectML 构建（ROCm 止于 2.5），
#     如实回退 CPU 并说明原因（依赖上游 tch/libtorch 升级，受 MSVC 兼容性阻塞）。
#   - 只需显卡驱动，不需要安装 CUDA Toolkit（cu128 发行版自带 CUDA 运行时）。

param(
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "auto",
    [string]$InstallDir = "",
    [switch]$Persist,
    [switch]$SmokeTest
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
$TorchVersion = "2.11.0"
$CudaSuffix = "cu128"
$CudaMinDriver = "570.00"   # CUDA 12.8 官方最低 Windows 驱动；低版本走 CUDA 次版本兼容，仅告警

function Write-Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "    $msg" -ForegroundColor Green }
function Write-Warn2($msg){ Write-Host "    $msg" -ForegroundColor Yellow }

# ---- GPU 探测 --------------------------------------------------------------
function Get-GpuInfo {
    $info = @{ Nvidia = $null; Amd = $null }
    try {
        $smi = & nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>$null
        if ($LASTEXITCODE -eq 0 -and $smi) {
            $parts = ($smi | Select-Object -First 1) -split ","
            $info.Nvidia = @{ Name = $parts[0].Trim(); Driver = $parts[1].Trim() }
        }
    } catch { }
    try {
        $amd = Get-CimInstance Win32_VideoController -ErrorAction Stop |
            Where-Object { $_.Name -match "AMD|Radeon" } | Select-Object -First 1
        if ($amd) { $info.Amd = $amd.Name }
    } catch { }
    return $info
}

# ---- 后端决策 --------------------------------------------------------------
$gpu = Get-GpuInfo
Write-Step "GPU 探测"
if ($gpu.Nvidia) { Write-Ok "NVIDIA: $($gpu.Nvidia.Name)（驱动 $($gpu.Nvidia.Driver)）" } else { Write-Warn2 "NVIDIA: 未检测到" }
if ($gpu.Amd)    { Write-Warn2 "AMD: $gpu.Amd（libtorch $TorchVersion 无 ROCm 构建，见下）" }

$Backend = $Device
if ($Backend -eq "auto") {
    if ($gpu.Nvidia) { $Backend = "cuda" } else { $Backend = "cpu" }
    Write-Ok "auto → $Backend"
}
if ($Backend -eq "cuda") {
    if (-not $gpu.Nvidia) {
        Write-Host "错误：-Device cuda 但未检测到 NVIDIA GPU（nvidia-smi 不可用）。" -ForegroundColor Red
        Write-Host "若无独显请改用 -Device cpu；若驱动未装请先安装 NVIDIA 驱动。" -ForegroundColor Red
        exit 1
    }
    $tooOld = $false
    try { $tooOld = [version]$gpu.Nvidia.Driver -lt [version]$CudaMinDriver } catch { $tooOld = $false }
    if ($tooOld) {
        Write-Warn2 "驱动 $($gpu.Nvidia.Driver) 低于 CUDA 12.8 推荐 $CudaMinDriver：依赖 CUDA 次版本兼容，通常可用，异常再升级驱动"
    }
    if ($gpu.Amd) { Write-Warn2 "同时存在 AMD GPU：libtorch 后端只用 NVIDIA，AMD 卡不参与计算" }
}

# ---- libtorch 就位 ---------------------------------------------------------
$envLocal = Join-Path $Repo "scripts\env.local.ps1"
if ($InstallDir -eq "") { $InstallDir = Join-Path $Repo ".libtorch" }

if ($Backend -eq "cpu") {
    Write-Step "CPU 后端：清空 LIBTORCH（torch-sys 首次构建自动下载 CPU 版 $TorchVersion）"
    Remove-Item Env:\LIBTORCH -ErrorAction SilentlyContinue
    $env:LIBTORCH = $null
    $libtorchDir = ""
} else {
    # 复用优先级：现有 LIBTORCH → 旧约定路径 E:\libs\... → 本仓库 .libtorch
    $candidates = @()
    if ($env:LIBTORCH) { $candidates += $env:LIBTORCH }
    $candidates += (Join-Path $InstallDir "libtorch-$CudaSuffix-$TorchVersion")
    foreach ($root in @($InstallDir, "E:\libs")) {
        if (Test-Path $root) {
            $found = Get-ChildItem -Directory -Path $root -Filter "libtorch-$CudaSuffix-2.11*" -ErrorAction SilentlyContinue |
                Select-Object -First 1
            if ($found) { $candidates += $found.FullName }
        }
    }
    $libtorchDir = $null
    foreach ($c in ($candidates | Select-Object -Unique)) {
        if ($c -and (Test-Path (Join-Path $c "lib\torch_cuda.dll"))) { $libtorchDir = $c; break }
    }
    if ($libtorchDir) {
        Write-Step "CUDA libtorch 已存在，直接复用"
        Write-Ok $libtorchDir
    } else {
        $url = "https://download.pytorch.org/libtorch/$CudaSuffix/libtorch-win-shared-with-deps-$TorchVersion%2B$CudaSuffix.zip"
        $zip = Join-Path $InstallDir "libtorch-win-$CudaSuffix-$TorchVersion.zip"
        $libtorchDir = Join-Path $InstallDir "libtorch-$CudaSuffix-$TorchVersion"
        Write-Step "下载 CUDA libtorch $TorchVersion（约 2.8GB，支持断点续传）"
        Write-Ok $url
        New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
        & curl.exe -L -C - --retry 5 --retry-delay 3 -o $zip $url
        if ($LASTEXITCODE -ne 0) { Write-Host "下载失败（exit $LASTEXITCODE），重跑本脚本可续传" -ForegroundColor Red; exit 1 }
        Write-Step "解压到 $libtorchDir"
        & tar -xf $zip -C $InstallDir
        if ($LASTEXITCODE -ne 0) {
            Write-Warn2 "tar 解压失败，回退 Expand-Archive（较慢）"
            Expand-Archive -Path $zip -DestinationPath $InstallDir -Force
        }
        if (-not (Test-Path (Join-Path $libtorchDir "lib\torch_cuda.dll"))) {
            Write-Host "错误：解压后未找到 torch_cuda.dll，安装目录结构异常" -ForegroundColor Red
            exit 1
        }
        Remove-Item $zip -ErrorAction SilentlyContinue
        Write-Ok "解压完成"
    }
    $env:LIBTORCH = $libtorchDir
}
if ($libtorchDir -ne "") { $env:PATH = "$libtorchDir\lib;$env:PATH" }

# ---- 持久化 ----------------------------------------------------------------
Write-Step "写环境脚本 $envLocal"
@"
# 由 setup-env.ps1 生成（Device=$Backend）。每个新会话先 dot-source 本文件：
#   . \$Repo\scripts\env.local.ps1
if ("$Backend" -eq "cuda") {
    `$env:LIBTORCH = "$libtorchDir"
    `$env:PATH = "`$env:LIBTORCH\lib;`$env:PATH"
} else {
    Remove-Item Env:\LIBTORCH -ErrorAction SilentlyContinue
}
"@ | Out-File -Encoding utf8 $envLocal
Write-Ok "每个新终端先执行: . $envLocal"

if ($Persist) {
    if ($Backend -eq "cuda") {
        setx LIBTORCH $libtorchDir | Out-Null
        Write-Ok "已写入用户级环境变量 LIBTORCH（setx，新终端生效）"
    } else {
        Write-Warn2 "CPU 后端无需持久化 LIBTORCH"
    }
}

# ---- 冒烟验证（可选）--------------------------------------------------------
if ($SmokeTest) {
    Write-Step "冒烟训练：合成数据 2 epochs（首次构建需数分钟下载/编译）"
    $dev = if ($Backend -eq "cuda") { "cuda:0" } else { "cpu" }
    cargo run -p av-runtime -- train -c configs/quick_detect.toml --override device=$dev --override train.epochs=2
    if ($LASTEXITCODE -ne 0) { Write-Host "冒烟失败，请把上方日志发到 issue" -ForegroundColor Red; exit 1 }
    Write-Ok "冒烟通过：$Backend 后端真实可用"
}

Write-Step "完成"
Write-Host @"
后续命令（本会话已生效）:
  cargo build --release -p av-runtime          # 编译
  cargo run -p av-runtime -- train -c configs/quick_detect.toml   # 冒烟训练
GPU 训练（cu128）详见 docs/USAGE.md §6
"@ -ForegroundColor Gray
