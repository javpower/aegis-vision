# -*- coding: utf-8 -*-
"""AegisVision detect 模型 ONNX 导出（Python 侧车，M8）。

用法：
    python scripts/export_onnx.py --ckpt runs/glass-logo-v4/best.ckpt.safetensors \
        --out runs/glass-logo-v4/best.onnx --imgsz 640 --classes 3 \
        --levels 8,16 [--verify sample.jpg]

流程：safetensors 权重 → PyTorch 镜像模型（与 Rust 逐算子同构）→
torch.onnx.export → onnxruntime 与镜像输出对拍。

依赖：pip install torch onnx onnxruntime safetensors
输出：[1, 4+nc, N] 解码后 xywh（+类别分数 [1, nc, N]），anchors 行主序
（大 stride 层在前，与 Rust decode_level 的层级顺序一致）。
"""
import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

REG_MAX = 16
DFL_SHIFT = (REG_MAX - 1) / 2.0
BASE_CHANNELS = [64, 128, 256, 512, 1024]
BASE_REPEATS = [3, 6, 6, 3]


def scale_channels(ch, width):
    return max(int(math.ceil(ch * width / 8.0)) * 8, 8)


def scale_repeats(n, depth):
    return max(round(n * depth), 1)


class ConvBnSilu(nn.Module):
    """conv(bias=False) + BN(eps 1e-3) + SiLU —— 与 Rust ConvBnSilu 同构。"""

    def __init__(self, in_ch, out_ch, k, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride, k // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_ch, eps=1e-3)

    def forward(self, x):
        return F.silu(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.cv1 = ConvBnSilu(c, c, 3, 1)
        self.cv2 = ConvBnSilu(c, c, 3, 1)

    def forward(self, x):
        return x + self.cv2(self.cv1(x))  # shortcut（C2f 内恒开）


class C2f(nn.Module):
    def __init__(self, c1, c2, n):
        super().__init__()
        hidden = c2 // 2
        self.cv1 = ConvBnSilu(c1, 2 * hidden, 1, 1)
        self.cv2 = ConvBnSilu((2 + n) * hidden, c2, 1, 1)
        self.m = nn.ModuleList([Bottleneck(hidden) for _ in range(n)])

    def forward(self, x):
        a, b = self.cv1(x).chunk(2, 1)
        parts = [a, b]
        for blk in self.m:
            parts.append(blk(parts[-1]))
        return self.cv2(torch.cat(parts, 1))


class SPPF(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        half = c1 // 2
        self.cv1 = ConvBnSilu(c1, half, 1, 1)
        self.cv2 = ConvBnSilu(4 * half, c2, 1, 1)

    def forward(self, x):
        y = self.cv1(x)
        y1 = F.max_pool2d(y, 5, 1, 2)
        y2 = F.max_pool2d(y1, 5, 1, 2)
        y3 = F.max_pool2d(y2, 5, 1, 2)
        return self.cv2(torch.cat([y, y1, y2, y3], 1))


class LevelHead(nn.Module):
    """解耦卷积头：cls1/cls2/cls3 + box1/box2/box3（3x3 padding1 + 1x1）。"""

    def __init__(self, in_c, mid, nc):
        super().__init__()
        self.cls1 = nn.Conv2d(in_c, mid, 3, 1, 1)
        self.cls2 = nn.Conv2d(mid, mid, 3, 1, 1)
        self.cls3 = nn.Conv2d(mid, nc, 1)
        self.box1 = nn.Conv2d(in_c, mid, 3, 1, 1)
        self.box2 = nn.Conv2d(mid, mid, 3, 1, 1)
        self.box3 = nn.Conv2d(mid, 4 * REG_MAX, 1)

    def forward(self, feat):
        cls = self.cls3(F.relu(self.cls2(F.relu(self.cls1(feat)))))
        hid = F.relu(self.box2(F.relu(self.box1(feat))))
        return cls, self.box3(hid)


class DetectModel(nn.Module):
    """backbone(0-9) + 两级头 + DFL/解码（与 Rust DetectModel 推理路径同构）。"""

    def __init__(self, width, depth, nc, levels):
        super().__init__()
        ch = [scale_channels(c, width) for c in BASE_CHANNELS]
        rep = [scale_repeats(n, depth) for n in BASE_REPEATS]
        self.backbone = nn.Sequential()
        lay = self.backbone
        idx = 0

        def conv(i, o, k, s):
            lay.add_module(str(idx_global[0]), ConvBnSilu(i, o, k, s))
            idx_global[0] += 1

        idx_global = [0]
        conv(3, ch[0], 3, 2)
        conv(ch[0], ch[1], 3, 2)
        lay.add_module(str(idx_global[0]), C2f(ch[1], ch[1], rep[0]))
        idx_global[0] += 1
        conv(ch[1], ch[2], 3, 2)
        p3_idx = idx_global[0]
        lay.add_module(str(idx_global[0]), C2f(ch[2], ch[2], rep[1]))
        idx_global[0] += 1
        conv(ch[2], ch[3], 3, 2)
        p4_idx = idx_global[0]
        lay.add_module(str(idx_global[0]), C2f(ch[3], ch[3], rep[2]))
        idx_global[0] += 1
        conv(ch[3], ch[4], 3, 2)
        lay.add_module(str(idx_global[0]), C2f(ch[4], ch[4], rep[2] if False else rep[3]))
        idx_global[0] += 1
        lay.add_module(str(idx_global[0]), SPPF(ch[4], ch[4]))
        idx_global[0] += 1
        self.p3_idx, self.p4_idx = p3_idx, p4_idx
        self.head = nn.ModuleDict()
        self.levels = list(levels)
        for s in levels:
            self.head.add_module(f"s{s}", LevelHead(ch[2] if s == 8 else ch[3], 64, nc))
        self.nc = nc

    def load_av_weights(self, path):
        from safetensors.torch import load_file

        sd = load_file(path)
        # AV 变量名 = 镜像 state_dict 键（backbone.N.* / head.sN.*）；
        # num_batches_tracked 为 PyTorch 侧独有，缺失容忍。
        missing, unexpected = self.load_state_dict(sd, strict=False)
        unexpected = [k for k in unexpected]
        print(f"loaded={len(sd)} missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print("  missing 样例:", missing[:5])
        assert not unexpected, f"镜像存在多余参数: {unexpected[:5]}"

    def forward(self, x):
        p3 = p4 = None
        for i, m in enumerate(self.backbone):
            x = m(x)
            if i == self.p3_idx:
                p3 = x
            elif i == self.p4_idx:
                p4 = x
        outs = []
        for s, (cls, dist) in zip(self.levels, [(8, (p3,)), (16, (p4,))]):
            head = self.head[f"s{s}"]
            feat = (p3, p4)[self.levels.index(s)]
            c, d = head(feat)
            n, _, h, w = d.shape
            prob = d.reshape(n, 4, REG_MAX, h, w).softmax(2)
            bins = torch.arange(REG_MAX, dtype=torch.float32).reshape(1, 1, REG_MAX, 1, 1)
            t = (prob * bins).sum(2) - DFL_SHIFT
            s_ = float(s)
            hh = torch.arange(h, dtype=torch.float32).reshape(1, 1, h, 1) + 0.5
            ww = torch.arange(w, dtype=torch.float32).reshape(1, 1, 1, w) + 0.5
            cy = (hh + t[:, 1:2].tanh()) * s_
            cx = (ww + t[:, 0:1].tanh()) * s_
            bw = t[:, 2:3].exp() * s_
            bh = t[:, 3:4].exp() * s_
            box = torch.cat([cx, cy, bw, bh], 1).reshape(n, 4, h * w)
            outs.append((c.reshape(n, self.nc, h * w).sigmoid(), box))
        # 大 stride 层在前（Rust decode_level 层级顺序一致）
        outs.sort(key=lambda o: -o[1].shape[2])
        scores = torch.cat([o[0] for o in outs], 2)
        boxes = torch.cat([o[1] for o in outs], 2)
        return torch.cat([boxes, scores], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="safetensors 权重（av export 产物）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--classes", type=int, required=True)
    ap.add_argument("--width", type=float, default=0.25)
    ap.add_argument("--depth", type=float, default=0.33)
    ap.add_argument("--levels", default="8,16")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--verify", nargs="?", const="1", help="用 onnxruntime 对拍镜像输出（随机输入图）")
    args = ap.parse_args()
    levels = [int(x) for x in args.levels.split(",")]

    model = DetectModel(args.width, args.depth, args.classes, levels)
    model.load_av_weights(args.ckpt)
    model.eval()
    dummy = torch.zeros(1, 3, args.imgsz, args.imgsz)
    torch.onnx.export(
        model, dummy, args.out, opset_version=args.opset, dynamo=False, input_names=["images"],
        output_names=["output"], dynamic_axes={"images": {0: "batch"}, "output": {0: "batch"}},
        do_constant_folding=True,
    )
    print(f"ONNX 导出完成: {args.out}")

    if args.verify:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
        x = torch.randn(1, 3, args.imgsz, args.imgsz)
        with torch.no_grad():
            ref = model(x).numpy()
        got = sess.run(None, {"images": x.numpy()})[0]
        box_d = float(np.abs(ref[:, :4] - got[:, :4]).max())
        box_rel = box_d / max(float(np.abs(ref[:, :4]).max()), 1e-9)
        sc_d = float(np.abs(ref[:, 4:] - got[:, 4:]).max())
        print(f"对拍: shape={got.shape} box 相对误差={box_rel:.2e} score 绝对误差={sc_d:.2e}")
        assert sc_d < 1e-4 and box_rel < 1e-4, "对拍超差"
        print("对拍通过 ✔")


if __name__ == "__main__":
    main()
