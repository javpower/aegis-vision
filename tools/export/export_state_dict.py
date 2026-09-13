#!/usr/bin/env python3
"""export_state_dict.py —— 外部 PyTorch 权重 → safetensors 导出侧车。

把任意 .pt / .pth（torch.save 产物）转换为 AV [pretrain] 可读的 .safetensors：

- ``ultralytics`` 整模型对象（YOLO('yolov8n.pt') 之类）→ 取 ``obj.model.state_dict()``；
- 带 ``state_dict()`` 方法的 nn.Module → 直接调用；
- 其余按 ``dict`` 处理，只保留值为 torch.Tensor 的键。

safetensors 只存张量数据，层名/结构与 PyTorch state_dict 一一对应；
AV 侧的改名/转置交给 [pretrain].layer_map（av-pretrain::weight_adapter）。

用法（在 E:/libs/yolo-bench venv，已有 torch + ultralytics；safetensors 需
``pip install safetensors``）::

    python tools/export/export_state_dict.py \
        --src yolov8n.pt --dst data/pretrain/yolov8n_backbone.safetensors \
        --backbone-prefix "model.0.,model.1.,model.2.,model.3."   # 逗号分隔多前缀
"""

from __future__ import annotations

import argparse
import sys


def extract_state_dict(obj):
    """ultralytics 整模型/检查点 dict / nn.Module / 裸 state_dict → state_dict。"""
    # ① ultralytics 整模型对象：.model 即 nn.Module
    inner = getattr(obj, "model", None)
    if inner is not None and hasattr(inner, "state_dict"):
        try:
            return inner.state_dict()
        except Exception as exc:  # noqa: BLE001 —— 失败则继续按其他形态尝试
            print(f"[export] obj.model.state_dict() 失败（{exc}）", file=sys.stderr)
    # ② ultralytics 检查点 dict：{'model': DetectionModel, 'epoch': ..., ...}
    if isinstance(obj, dict):
        ckpt_model = obj.get("model")
        if ckpt_model is not None and hasattr(ckpt_model, "state_dict"):
            try:
                return ckpt_model.state_dict()
            except Exception as exc:  # noqa: BLE001
                print(f"[export] obj['model'].state_dict() 失败（{exc}）", file=sys.stderr)
    # ③ nn.Module
    if hasattr(obj, "state_dict"):
        try:
            return obj.state_dict()
        except Exception as exc:  # noqa: BLE001
            print(f"[export] obj.state_dict() 失败（{exc}）", file=sys.stderr)
    # ④ 裸 state_dict dict（只保留 tensor 值）
    if isinstance(obj, dict):
        return obj
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="PyTorch .pt/.pth → safetensors 导出")
    ap.add_argument("--src", required=True, help="输入 .pt/.pth 路径")
    ap.add_argument("--dst", required=True, help="输出 .safetensors 路径")
    ap.add_argument(
        "--backbone-prefix",
        default=None,
        help="层名前缀过滤（逗号分隔多前缀）；缺省保留全部张量",
    )
    args = ap.parse_args()

    import torch  # 延迟导入：--help 不需要 torch
    from safetensors.torch import save_file

    import os

    parent = os.path.dirname(os.path.abspath(args.dst))
    os.makedirs(parent, exist_ok=True)  # safetensors 不自动创建父目录

    obj = torch.load(args.src, map_location="cpu", weights_only=False)
    sd = extract_state_dict(obj)
    if sd is None:
        print("[export] 无法从源文件提取 state_dict（不是 Module / ultralytics 模型 / dict）",
              file=sys.stderr)
        return 2

    prefixes = None
    if args.backbone_prefix:
        prefixes = tuple(p.strip() for p in args.backbone_prefix.split(",") if p.strip())

    out: dict[str, torch.Tensor] = {}
    for key, value in sd.items():
        if not torch.is_tensor(value):
            continue
        if prefixes is not None and not key.startswith(prefixes):
            continue
        # 独立存储（safetensors 拒绝共享内存的张量）+ 连续 + CPU
        t = value.detach().to("cpu").contiguous().clone()
        # tch 0.17 safetensors 不支持 F8/复数等 dtype，常见浮点统一到 fp32
        if t.dtype in (torch.float64, torch.float16, torch.bfloat16):
            t = t.to(torch.float32)
        out[key] = t

    if not out:
        print("[export] 过滤后没有可导出张量（检查 --backbone-prefix）", file=sys.stderr)
        return 2

    save_file(out, args.dst)
    total = sum(t.numel() for t in out.values())
    print(f"[export] {len(out)} 张量 / {total:,} 参数 -> {args.dst}")
    for key, t in list(out.items())[:10]:
        print(f"  {key}: {tuple(t.shape)} {t.dtype}")
    if len(out) > 10:
        print(f"  ... 共 {len(out)} 个")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
