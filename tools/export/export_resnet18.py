#!/usr/bin/env python3
"""export_resnet18.py —— torchvision ResNet18 ImageNet 预训练权重 → safetensors。

把 torchvision 官方 ResNet18（IMAGENET1K_V1）state_dict 原样导出为 AV
[pretrain] 可读的 .safetensors：层名即 torchvision 命名（conv1/bn1/layer1.0.conv1/
.../fc.weight），AV 侧 ResNet18 骨干按同名结构装配（变量名只多一个
"backbone." 前缀），配 configs/resnet18_map.toml 一条前缀映射即可全量导入。

包含 BN running_mean/running_var（AV 侧 tch nn::batch_norm 的 no_train 变量，
可被 weight_adapter 直接加载）与 num_batches_tracked（AV 侧无对应变量，导入
报告中以 unexpected 呈现——行业惯例：部分加载 + 完整报告）。

用法（E:/libs/yolo-bench venv，已有 torch + torchvision + safetensors）::

    python tools/export/export_resnet18.py \
        --dst data/pretrain/resnet18_imagenet.safetensors

首次运行会从 download.pytorch.org 下载权重（约 45 MB，缓存到
~/.cache/torch/hub/checkpoints/）。
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description="torchvision ResNet18 ImageNet → safetensors")
    ap.add_argument(
        "--dst",
        default="data/pretrain/resnet18_imagenet.safetensors",
        help="输出 .safetensors 路径（默认 data/pretrain/resnet18_imagenet.safetensors）",
    )
    args = ap.parse_args()

    import torch  # 延迟导入：--help 不需要 torch
    from safetensors.torch import save_file

    os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)

    # IMAGENET1K_V1：与论文一致的原始 76.13% top-1 权重（FP32）
    from torchvision.models import ResNet18_Weights, resnet18

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    sd = model.state_dict()

    out: dict[str, torch.Tensor] = {}
    for key, value in sd.items():
        if not torch.is_tensor(value):
            continue
        # 独立存储（safetensors 拒绝共享内存的张量）+ 连续 + CPU；
        # 浮点统一 fp32（tch 读取侧约定），num_batches_tracked 保持 int64
        t = value.detach().to("cpu").contiguous().clone()
        if t.dtype in (torch.float64, torch.float16, torch.bfloat16):
            t = t.to(torch.float32)
        out[key] = t

    if not out:
        print("[export] state_dict 为空", file=sys.stderr)
        return 2

    save_file(out, args.dst)
    total = sum(t.numel() for t in out.values())
    n_params = sum(1 for k, t in out.items() if t.dtype == torch.float32)
    print(f"[export] {len(out)} 张量 / {total:,} 元素（float32 张量 {n_params} 个）-> {args.dst}")
    for key in ("conv1.weight", "bn1.weight", "bn1.running_mean", "layer1.0.conv1.weight",
                "layer4.1.bn2.running_var", "fc.weight"):
        t = out.get(key)
        print(f"  {key}: {tuple(t.shape)} {t.dtype}" if t is not None else f"  {key}: 缺失!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
