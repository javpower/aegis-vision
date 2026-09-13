# tools/export/export_dinov2.py —— DINOv2 官方预训练权重导出（safetensors）
#
# 用法（工作区根目录）：
#   E:/libs/yolo-bench/Scripts/python.exe tools/export/export_dinov2.py
#
# 做什么：
#   1) 优先 HuggingFace 直连下载 facebook/dinov2-small 的 model.safetensors
#      （HF 官方即 safetensors 格式，键名为 HF transformers 命名）到
#      data/pretrain/dinov2_small.safetensors；
#   2) 网络不通时自动回退 hf-mirror.com 同路径；
#   3) 用 safetensors 读回校验，打印键名样例 10 个（供 Rust 侧映射表编写）
#      与完整键名统计；
#   4) 可选 --hub：改走 torch.hub（facebookresearch/dinov2 的 dinov2_vits14），
#      导出 torch.hub 原生命名（blocks.{i}.attn.qkv.* 融合版）的 safetensors，
#      配合 av-pretrain::weight_adapter::dinov2_layer_map 的预设映射可近乎直配。
#
# 依赖：safetensors（yolo-bench 环境已装）；torch 仅 --hub 模式需要。

import argparse
import os
import sys
import urllib.request

DEFAULT_URL = "https://huggingface.co/facebook/dinov2-small/resolve/main/model.safetensors"
MIRROR_URL = "https://hf-mirror.com/facebook/dinov2-small/resolve/main/model.safetensors"
DEFAULT_DST = os.path.join("data", "pretrain", "dinov2_small.safetensors")


def download(url: str, dst: str, timeout: int = 60) -> bool:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r, open(dst, "wb") as f:
            total = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                total += len(chunk)
        print(f"[export-dinov2] 下载完成 {url} -> {dst}（{total} 字节）")
        return True
    except Exception as e:  # noqa: BLE001 —— 网络错误类型繁多，统一回退
        if os.path.exists(dst):
            os.remove(dst)  # 半截文件不留
        print(f"[export-dinov2] 下载失败 {url}: {type(e).__name__}: {e}")
        return False


def inspect(path: str) -> None:
    from safetensors import safe_open

    with safe_open(path, framework="pt") as f:
        keys = sorted(f.keys())
        print(f"[export-dinov2] {path}: {len(keys)} 个张量")
        print("[export-dinov2] 键名样例 10 个（映射表编写用）：")
        for k in keys[:10]:
            print(f"  {k} {tuple(f.get_slice(k).get_shape())}")
        print("[export-dinov2] 非层级键（embedding / 最终 norm 等）：")
        for k in keys:
            if "encoder.layer" not in k and "blocks." not in k:
                print(f"  {k} {tuple(f.get_slice(k).get_shape())}")
        if keys and any(k.startswith("encoder.layer.0.") for k in keys):
            print("[export-dinov2] layer.0 全部键：")
            for k in keys:
                if k.startswith("encoder.layer.0."):
                    print(f"  {k} {tuple(f.get_slice(k).get_shape())}")


def export_from_hub(dst: str) -> None:
    """备选：torch.hub 官方仓库（facebookresearch/dinov2）→ hub 原生命名 safetensors。"""
    import torch  # noqa: PLC0415 —— 仅 hub 模式需要

    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    from safetensors.torch import save_file

    save_file(state, dst)
    print(f"[export-dinov2] torch.hub 导出完成 -> {dst}（{len(state)} 个张量，hub 命名）")


def main() -> int:
    ap = argparse.ArgumentParser(description="导出 DINOv2 官方预训练权重为 safetensors")
    ap.add_argument("--dst", default=DEFAULT_DST, help="输出路径（默认 %(default)s）")
    ap.add_argument(
        "--hub", action="store_true",
        help="改走 torch.hub（facebookresearch/dinov2），导出 hub 原生命名",
    )
    args = ap.parse_args()

    if os.path.exists(args.dst) and not args.hub:
        print(f"[export-dinov2] 已存在 {args.dst}，跳过下载（如需重下请先删除）")
    elif args.hub:
        export_from_hub(args.dst)
    else:
        # 优先 HF 直连，网络不通回退 hf-mirror.com 同路径
        if not download(DEFAULT_URL, args.dst):
            if not download(MIRROR_URL, args.dst):
                print("[export-dinov2] 两个源均失败；可手动下载后放到目标路径，"
                      "或改用 --hub 走 torch.hub", file=sys.stderr)
                return 1
    inspect(args.dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
