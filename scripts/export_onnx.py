# -*- coding: utf-8 -*-
"""AegisVision ONNX 导出（M8，Python 侧车 + onnxruntime 对拍验证）。

支持的骨干 × 任务：
    csp-elan   detect   （检测：骨干 0-9 + 解耦头 + DFL/解码，输出 [1,4+nc,N] xywh+分数）
    resnet18   classify （torchvision 同构 + GAP + fc，输出 [1,nc] logits）
    dino-v2    classify （ViT-S/14 + cls token + fc）
    simple-cnn classify （4 层 CNN + GAP + fc）

权重：`av export -w <ckpt> --format safetensors` 的产物（键名 = AV 变量名，
与镜像 state_dict 逐字对齐；missing/unexpected 必须为 0 才算加载成功）。
预处理（/255 + 可选 ImageNet mean/std）已并入图内，部署端输入原始 0-255 RGB。

依赖：pip install torch onnx onnxscript safetensors onnxruntime
"""
import argparse
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
REG_MAX = 16
DFL_SHIFT = (REG_MAX - 1) / 2.0
BASE_CHANNELS = [64, 128, 256, 512, 1024]
BASE_REPEATS = [3, 6, 6, 3]


def scale_channels(ch, width):
    return max(int(math.ceil(ch * width / 8.0)) * 8, 8)


def scale_repeats(n, depth):
    return max(round(n * depth), 1)


# ---------------------------------------------------------------------------
# csp-elan（detect，见 backbone_cspelan.rs）
# ---------------------------------------------------------------------------

class ConvBnSilu(nn.Module):
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
        return x + self.cv2(self.cv1(x))


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


class CspElanDetect(nn.Module):
    """backbone(0-9) + 两级解耦头 + DFL/解码，输出 [1,4+nc,N]（xywh + 分数）。"""

    def __init__(self, width, depth, nc, levels):
        super().__init__()
        ch = [scale_channels(c, width) for c in BASE_CHANNELS]
        rep = [scale_repeats(n, depth) for n in BASE_REPEATS]
        self.levels = list(levels)
        self.nc = nc
        bb = self.backbone = nn.Sequential()
        self.p3_idx, self.p4_idx = 4, 6
        bb.add_module("0", ConvBnSilu(3, ch[0], 3, 2))
        bb.add_module("1", ConvBnSilu(ch[0], ch[1], 3, 2))
        bb.add_module("2", C2f(ch[1], ch[1], rep[0]))
        bb.add_module("3", ConvBnSilu(ch[1], ch[2], 3, 2))
        bb.add_module("4", C2f(ch[2], ch[2], rep[1]))
        bb.add_module("5", ConvBnSilu(ch[2], ch[3], 3, 2))
        bb.add_module("6", C2f(ch[3], ch[3], rep[2]))
        bb.add_module("7", ConvBnSilu(ch[3], ch[4], 3, 2))
        bb.add_module("8", C2f(ch[4], ch[4], rep[3]))
        bb.add_module("9", SPPF(ch[4], ch[4]))
        self.head = nn.ModuleDict({f"s{s}": LevelHead(ch[2] if s == 8 else ch[3], 64, nc)
                                   for s in levels})

    def forward(self, x):
        p3 = p4 = None
        for i, m in enumerate(self.backbone):
            x = m(x)
            if i == self.p3_idx:
                p3 = x
            elif i == self.p4_idx:
                p4 = x
        outs = []
        for s in self.levels:
            feat = p3 if s == 8 else p4
            c, d = self.head[f"s{s}"](feat)
            n, _, h, w = d.shape
            prob = d.reshape(n, 4, REG_MAX, h, w).softmax(2)
            bins = torch.arange(REG_MAX, dtype=torch.float32).reshape(1, 1, REG_MAX, 1, 1)
            t = (prob * bins).sum(2) - DFL_SHIFT
            s_ = float(s)
            cx = (torch.arange(w, dtype=torch.float32).reshape(1, 1, 1, w) + 0.5
                  + t[:, 0:1].tanh()) * s_
            cy = (torch.arange(h, dtype=torch.float32).reshape(1, 1, h, 1) + 0.5
                  + t[:, 1:2].tanh()) * s_
            bw = t[:, 2:3].exp() * s_
            bh = t[:, 3:4].exp() * s_
            outs.append((c.reshape(n, self.nc, h * w).sigmoid(),
                         torch.cat([cx, cy, bw, bh], 1).reshape(n, 4, h * w)))
        outs.sort(key=lambda o: -o[1].shape[2])
        return torch.cat([torch.cat([o[1] for o in outs], 2),
                          torch.cat([o[0] for o in outs], 2)], 1)


# ---------------------------------------------------------------------------
# resnet18（classify，torchvision 同构，见 backbone_resnet.rs）
# ---------------------------------------------------------------------------

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False), nn.BatchNorm2d(out_ch))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        identity = self.downsample(x) if self.downsample is not None else x
        return F.relu(out + identity)


class ResNet18Classify(nn.Module):
    """conv1/bn1/maxpool/layer1-4 + GAP + fc；输入 0-255 RGB，ImageNet 归一化在图内。"""

    def __init__(self, nc):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = nn.Sequential(BasicBlock(64, 64, 1), BasicBlock(64, 64, 1))
        self.layer2 = nn.Sequential(BasicBlock(64, 128, 2), BasicBlock(128, 128, 1))
        self.layer3 = nn.Sequential(BasicBlock(128, 256, 2), BasicBlock(256, 256, 1))
        self.layer4 = nn.Sequential(BasicBlock(256, 512, 2), BasicBlock(512, 512, 1))
        self.fc = nn.Linear(512, nc)

    def forward(self, raw):
        x = raw / 255.0
        x = (x - torch.tensor(IMAGENET_MEAN).reshape(1, 3, 1, 1)) / \
            torch.tensor(IMAGENET_STD).reshape(1, 3, 1, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 3, 2, 1)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(x.mean((2, 3)))


# ---------------------------------------------------------------------------
# dino-v2（classify，ViT-S/14，见 backbone_dino.rs）
# ---------------------------------------------------------------------------

class Gamma(nn.Module):
    """LayerScale（Rust 侧键名 lsN.gamma）。"""

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))


class DinoAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.heads = heads
        self.scale = 1.0 / math.sqrt(dim // heads)

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)).softmax(-1)
        return (attn @ v).transpose(1, 2).reshape(b, n, c)


class DinoBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = DinoAttention(dim, heads)
        self.ls1 = Gamma(dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(dim, dim * 4)
        self.mlp.fc2 = nn.Linear(dim * 4, dim)
        self.ls2 = Gamma(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x)) * self.ls1.gamma
        x = x + self.mlp.fc2(F.gelu(self.mlp.fc1(self.norm2(x)))) * self.ls2.gamma
        return x


class DinoV2Classify(nn.Module):
    """patch embed + cls/pos + 12 blocks + norm + cls token + fc。

    pos_embed 在导出尺寸下为常量（网格不匹配时双三次插值，DINOv2 官方同款）。
    """

    def __init__(self, nc, grid_side):
        super().__init__()
        dim, depth, heads = 384, 12, 6
        self.grid_side = grid_side
        self.backbone = nn.Module()
        bb = self.backbone
        bb.patch_embed = nn.Module()
        bb.patch_embed.proj = nn.Conv2d(3, dim, 14, 14)
        bb.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        bb.pos_embed = nn.Parameter(torch.zeros(1, 1 + grid_side * grid_side, dim))
        bb.blocks = nn.Sequential(*[DinoBlock(dim, heads) for _ in range(depth)])
        bb.norm = nn.LayerNorm(dim, eps=1e-6)
        self.fc = nn.Linear(dim, nc)

    def forward(self, raw):
        x = raw / 255.0
        p = self.backbone.patch_embed.proj(x)
        b, c, gh, gw = p.shape
        patches = p.reshape(b, c, gh * gw).transpose(1, 2)
        cls = self.backbone.cls_token.expand(b, -1, -1)
        t = torch.cat([cls, patches], 1) + self.interpolate_pos(gh, gw)
        for blk in self.backbone.blocks:
            t = blk(t)
        cls_out = self.backbone.norm(t)[:, 0]
        return self.fc(cls_out)

    def interpolate_pos(self, gh, gw):
        pe = self.backbone.pos_embed
        n = pe.shape[1] - 1
        sqrt_n = int(math.isqrt(n))
        if gh * gw == n and gh == gw:
            return pe
        patch = pe[:, 1:].reshape(1, sqrt_n, sqrt_n, -1).permute(0, 3, 1, 2)
        patch = F.interpolate(patch, size=(gw, gh), mode="bicubic",
                              align_corners=True, antialias=True)
        patch = patch.permute(0, 2, 3, 1).reshape(1, -1, pe.shape[-1])
        return torch.cat([pe[:, :1], patch], 1)


# ---------------------------------------------------------------------------
# simple-cnn（classify，见 backbone.rs）
# ---------------------------------------------------------------------------

class SimpleCnnClassify(nn.Module):
    def __init__(self, nc, width):
        super().__init__()
        self.c1 = nn.Conv2d(3, width, 3, 2, 1)
        self.c2 = nn.Conv2d(width, width * 2, 3, 2, 1)
        self.c3 = nn.Conv2d(width * 2, width * 4, 3, 2, 1)
        self.c4 = nn.Conv2d(width * 4, width * 8, 3, 2, 1)
        self.fc = nn.Linear(width * 8, nc)

    def forward(self, raw):
        x = raw / 255.0
        x = F.relu(self.c4(F.relu(self.c3(F.relu(self.c2(F.relu(self.c1(x))))))))
        return self.fc(x.mean((2, 3)))


def build(args):
    if args.backbone == "csp-elan":
        return CspElanDetect(args.width, args.depth, args.classes,
                             [int(x) for x in args.levels.split(",")])
    if args.backbone == "resnet18":
        return ResNet18Classify(args.classes)
    if args.backbone == "dino-v2":
        return DinoV2Classify(args.classes, args.imgsz // 14)
    if args.backbone == "simple-cnn":
        return SimpleCnnClassify(args.classes, max(int(round(16 * args.width)), 4))
    raise SystemExit(f"不支持的骨干: {args.backbone}")


def adapt_dinov2_official(sd):
    """HF transformers 官方 dinov2 命名 → AV 镜像键名；Q/K/V 三条 Linear 按
    cat([q,k,v], dim=0) 融合为单条 qkv（与 Rust backbone_dino 加载器同规则）。"""
    out = {}
    qkv_w, qkv_b = {}, {}
    for k, v in sd.items():
        if "mask_token" in k or "register_tokens" in k:
            continue
        if k.startswith("embeddings.cls_token"):
            out["backbone.cls_token"] = v.reshape(1, 1, -1)
        elif k.startswith("embeddings.position_embeddings"):
            out["backbone.pos_embed"] = v
        elif k.startswith("embeddings.patch_embeddings.projection."):
            out["backbone.patch_embed.proj." + k.rsplit(".", 1)[1]] = v
        elif k.startswith("layernorm."):
            out["backbone.norm." + k.rsplit(".", 1)[1]] = v
        elif k.startswith("encoder.layer."):
            n, rest = k.split(".")[2], k.split(".", 3)[3]
            if rest.startswith(("layer_scale_1", "layer_scale1", "ls1")):
                out[f"backbone.blocks.{n}.ls1.gamma"] = v.reshape(-1)
            elif rest.startswith(("layer_scale_2", "layer_scale2", "ls2")):
                out[f"backbone.blocks.{n}.ls2.gamma"] = v.reshape(-1)
            elif rest.startswith("norm1."):
                out[f"backbone.blocks.{n}.norm1." + rest.rsplit(".", 1)[1]] = v
            elif rest.startswith("norm2."):
                out[f"backbone.blocks.{n}.norm2." + rest.rsplit(".", 1)[1]] = v
            elif rest.startswith("attention.attention."):
                # attention.attention.<query|key|value>.<weight|bias>（精确四段解析，
                # 避免 bias 被 .query. 前缀分支抢先吞进 weight 桶）
                parts = rest.split(".")
                if len(parts) != 4:
                    raise SystemExit(f"未识别的 dinov2 注意键: {k}")
                part, kind = parts[2], parts[3]
                idx = {"query": 0, "key": 1, "value": 2}[part]
                bucket = qkv_w if kind == "weight" else qkv_b
                bucket.setdefault(n, {})[idx] = v
            elif rest.startswith("attention.output.dense."):
                out[f"backbone.blocks.{n}.attn.proj." + rest.rsplit(".", 1)[1]] = v
            elif rest.startswith("mlp."):
                out[f"backbone.blocks.{n}.mlp." + rest[4:]] = v
            else:
                raise SystemExit(f"未识别的 dinov2 键: {k}")
        else:
            raise SystemExit(f"未识别的 dinov2 键: {k}")
    for n, parts in qkv_w.items():
        out[f"backbone.blocks.{n}.attn.qkv.weight"] = torch.cat(
            [parts[0], parts[1], parts[2]], dim=0)
    for n, parts in qkv_b.items():
        out[f"backbone.blocks.{n}.attn.qkv.bias"] = torch.cat(
            [parts[0], parts[1], parts[2]], dim=0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="csp-elan",
                    choices=["csp-elan", "resnet18", "dino-v2", "simple-cnn"])
    ap.add_argument("--ckpt", help="safetensors 权重（缺省 = 随机初始化，仅验证图结构）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--classes", type=int, required=True)
    ap.add_argument("--width", type=float, default=0.25)
    ap.add_argument("--depth", type=float, default=0.33)
    ap.add_argument("--levels", default="8,16")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--verify", nargs="?", const="1", help="onnxruntime 对拍（随机输入）")
    args = ap.parse_args()

    model = build(args)
    if args.ckpt:
        from safetensors.torch import load_file

        sd = load_file(args.ckpt)
        if any(k.startswith(("encoder.", "embeddings.")) for k in sd):
            sd = adapt_dinov2_official(sd)
            print("已套用 HF 官方命名适配（含 QKV 融合）")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"loaded={len(sd)} missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print("  missing:", missing[:6])
        assert not unexpected, f"镜像存在多余参数: {unexpected[:6]}"
    else:
        print("未提供 --ckpt：随机初始化（仅验证图结构/ONNX 合法性）")
    model.eval()
    dummy = torch.zeros(1, 3, args.imgsz, args.imgsz)
    torch.onnx.export(model, dummy, args.out, opset_version=args.opset, dynamo=False,
                      input_names=["images"], output_names=["output"],
                      dynamic_axes={"images": {0: "batch"}, "output": {0: "batch"}},
                      do_constant_folding=True)
    print(f"ONNX 导出完成: {args.out}")

    if args.verify:
        import onnxruntime as ort

        sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
        x = torch.randn(1, 3, args.imgsz, args.imgsz)
        with torch.no_grad():
            ref = model(x).numpy()
        got = sess.run(None, {"images": x.numpy()})[0]
        dmax = float(np.abs(ref - got).max())
        print(f"onnxruntime 对拍: shape={got.shape} max|Δ|={dmax:.2e}")
        assert dmax < 1e-3, "对拍超差"
        print("对拍通过 ✔")


if __name__ == "__main__":
    main()
