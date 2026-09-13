#!/usr/bin/env bash
# AegisVision 一键环境配置（Linux）
#
# 用法：
#   ./scripts/setup-env.sh                  # 自动探测：有 N 卡装 CUDA 版，否则 CPU 版
#   ./scripts/setup-env.sh --device cuda    # 强制 CUDA
#   ./scripts/setup-env.sh --device cpu     # 强制 CPU（首次构建自动下载 CPU libtorch）
#   ./scripts/setup-env.sh --persist        # 写入 ~/.bashrc（LIBTORCH/PATH）
#   ./scripts/setup-env.sh --smoke-test     # 配置完跑一次合成数据冒烟训练
#
# 说明：
#   - 下载官方 libtorch 2.11（与 Cargo.toml 锁定的 tch 0.24 配对）到 <仓库>/.libtorch/
#   - 只需 NVIDIA 驱动，不需要安装 CUDA Toolkit（cu128 发行版自带 CUDA 运行时）
#   - AMD GPU：libtorch 2.11 官方无 ROCm 构建（ROCm 止于 2.5.x，与 tch 0.24 锁定的
#     2.11 不配对）——如实回退 CPU 并提示；待上游 tch 升级后此处接入 ROCm 下载
#   - 需要依赖：curl、unzip、rustup（Rust stable；建议 clang 或 gcc ≥ 9）

set -euo pipefail

TORCH_VERSION="2.11.0"
CUDA_SUFFIX="cu128"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL_DIR="${LIBTORCH_INSTALL_DIR:-$REPO/.libtorch}"
ENV_LOCAL="$REPO/scripts/env.local.sh"

DEVICE="auto"; PERSIST=0; SMOKE=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --device) DEVICE="$2"; shift 2 ;;
        --device=*) DEVICE="${1#*=}"; shift ;;
        --persist) PERSIST=1; shift ;;
        --smoke-test) SMOKE=1; shift ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done
[[ "$DEVICE" =~ ^(auto|cpu|cuda)$ ]] || { echo "--device 须为 auto|cpu|cuda"; exit 1; }

step()  { printf '\n==> %s\n' "$1"; }
ok()    { printf '    %s\n' "$1"; }
warn()  { printf '    %s\n' "$1" >&2; }

# ---- GPU 探测 ---------------------------------------------------------------
step "GPU 探测"
NV_NAME=""; NV_DRIVER=""
if command -v nvidia-smi >/dev/null 2>&1; then
    NV_LINE="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1 || true)"
    if [[ -n "$NV_LINE" ]]; then
        NV_NAME="${NV_LINE%%,*}"; NV_DRIVER="${NV_LINE#*,}"
        ok "NVIDIA: $NV_NAME（驱动 $NV_DRIVER）"
    fi
fi
[[ -z "$NV_NAME" ]] && warn "NVIDIA: 未检测到"
if lspci -nn 2>/dev/null | grep -qiE 'amd|radeon'; then
    warn "AMD GPU: libtorch $TORCH_VERSION 无 ROCm 构建（上游止于 2.5.x），本脚本回退 CPU"
fi

# ---- 后端决策 ---------------------------------------------------------------
BACKEND="$DEVICE"
if [[ "$BACKEND" == "auto" ]]; then
    if [[ -n "$NV_NAME" ]]; then BACKEND="cuda"; else BACKEND="cpu"; fi
    ok "auto → $BACKEND"
fi
if [[ "$BACKEND" == "cuda" && -z "$NV_NAME" ]]; then
    echo "错误：--device cuda 但未检测到 NVIDIA GPU（无 nvidia-smi）。无独显请用 --device cpu" >&2
    exit 1
fi

# ---- libtorch 就位 ----------------------------------------------------------
unset LIBTORCH || true
LIBTORCH_DIR=""
if [[ "$BACKEND" == "cuda" ]]; then
    for c in "${LIBTORCH:-}" "$INSTALL_DIR/libtorch-$CUDA_SUFFIX-$TORCH_VERSION"; do
        if [[ -n "$c" && -f "$c/lib/libtorch_cuda.so" ]]; then LIBTORCH_DIR="$c"; break; fi
    done
    if [[ -n "$LIBTORCH_DIR" ]]; then
        step "CUDA libtorch 已存在，直接复用"; ok "$LIBTORCH_DIR"
    else
        URL="https://download.pytorch.org/libtorch/$CUDA_SUFFIX/libtorch-shared-with-deps-$TORCH_VERSION%2B$CUDA_SUFFIX.zip"
        ZIP="$INSTALL_DIR/libtorch-linux-$CUDA_SUFFIX-$TORCH_VERSION.zip"
        LIBTORCH_DIR="$INSTALL_DIR/libtorch-$CUDA_SUFFIX-$TORCH_VERSION"
        step "下载 CUDA libtorch $TORCH_VERSION（约 2.5GB，支持断点续传）"
        ok "$URL"
        mkdir -p "$INSTALL_DIR"
        curl -L -C - --retry 5 --retry-delay 3 -o "$ZIP" "$URL"
        step "解压到 $LIBTORCH_DIR"
        unzip -q -o "$ZIP" -d "$INSTALL_DIR"
        [[ -f "$LIBTORCH_DIR/lib/libtorch_cuda.so" ]] || { echo "解压异常：缺 libtorch_cuda.so"; exit 1; }
        rm -f "$ZIP"
    fi
    export LIBTORCH="$LIBTORCH_DIR"
    export PATH="$LIBTORCH_DIR/lib:$PATH"
else
    step "CPU 后端：不设 LIBTORCH（torch-sys 首次构建自动下载 CPU 版 $TORCH_VERSION）"
fi

# ---- 持久化 -----------------------------------------------------------------
step "写环境脚本 $ENV_LOCAL"
{
    echo "# 由 setup-env.sh 生成（Device=$BACKEND）。source 本文件以恢复环境："
    echo "#   source $ENV_LOCAL"
    if [[ "$BACKEND" == "cuda" ]]; then
        echo "export LIBTORCH=\"$LIBTORCH_DIR\""
        echo "export PATH=\"\$LIBTORCH/lib:\$PATH\""
    else
        echo "unset LIBTORCH"
    fi
} > "$ENV_LOCAL"
ok "每个新终端先执行: source $ENV_LOCAL"

if [[ "$PERSIST" -eq 1 ]]; then
    if [[ "$BACKEND" == "cuda" ]]; then
        {
            echo ""
            echo "# AegisVision（setup-env.sh 写入）"
            echo "export LIBTORCH=\"$LIBTORCH_DIR\""
            echo "export PATH=\"\$LIBTORCH/lib:\$PATH\""
        } >> "$HOME/.bashrc"
        ok "已追加到 ~/.bashrc"
    else
        warn "CPU 后端无需持久化 LIBTORCH"
    fi
fi

# ---- 冒烟验证（可选）---------------------------------------------------------
if [[ "$SMOKE" -eq 1 ]]; then
    step "冒烟训练：合成数据 2 epochs（首次构建需数分钟）"
    DEV="cpu"; [[ "$BACKEND" == "cuda" ]] && DEV="cuda:0"
    cargo run -p av-runtime -- train -c configs/quick_detect.toml \
        --override "device=$DEV" --override "train.epochs=2"
    ok "冒烟通过：$BACKEND 后端真实可用"
fi

step "完成"
cat <<'EOF'
后续命令（本会话已生效）:
  cargo build --release -p av-runtime          # 编译
  cargo run -p av-runtime -- train -c configs/quick_detect.toml   # 冒烟训练
EOF
