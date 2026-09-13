#!/usr/bin/env python3
"""build_comparison.py —— 汇总六臂验收评测 → runs/COMPARISON.md

读 runs/<arm>/eval-final.json（mask_miou/recall/precision/per_class）+
runs/<arm>/metrics.jsonl（全程最优 val mIoU），生成对比表。
"""
import io
import json
import os

RUNS = "runs"
ARMS = [
    ("seg-harness-none",      "simple-cnn",      "无（从零）"),
    ("seg-harness-pretrain",  "simple-cnn",      "yolov8n stem（2.3万参数切片）"),
    ("seg-harness-r18-none",  "ResNet18",        "无（从零）"),
    ("seg-harness-r18",       "ResNet18",        "ImageNet（1120万参数全量）"),
    ("seg-harness-dino",      "DINOv2 ViT-S/14", "DINOv2（448px）"),
    ("seg-harness-dino672",   "DINOv2 ViT-S/14", "DINOv2（672px）"),
    ("seg-harness-csp",       "csp-elan",        "yolov8n 完整骨干（320万参数）"),
]
NAMES = {"0": "线束", "1": "锁体", "2": "限位器", "3": "防撞块"}


def load(arm):
    out = {}
    ej = os.path.join(RUNS, arm, "eval-final.json")
    if os.path.isfile(ej):
        out["eval"] = json.load(io.open(ej, encoding="utf-8"))
    mj = os.path.join(RUNS, arm, "metrics.jsonl")
    if os.path.isfile(mj):
        best = 0.0
        for line in io.open(mj, encoding="utf-8"):
            try:
                r = json.loads(line)
                best = max(best, float(r.get("metric_value") or 0.0))
            except Exception:
                pass
        out["best_miou"] = best
    rp = os.path.join(RUNS, arm, "report.json")
    if os.path.isfile(rp):
        out["report"] = json.load(io.open(rp, encoding="utf-8"))
    return out


def fmt(v):
    return f"{v:.3f}" if isinstance(v, (int, float)) else "—"


def main():
    lines = ["# 六臂（含对照）实例分割对比报告", "", f"数据：E:/xunlian/yolo_dataset（4 类，526/93，2219 实例）",
             "协议：csp/resnet18/simple-cnn 各 120ep@640（DINOv2 受 patch-14 约束 448/672），seed=7，",
             "增强 flip0.5+hsv+scale±0.1、收尾 15ep 关增强；验收 = val split。", ""]
    lines.append("| 臂 | 骨干 | 预训练 | val mIoU | 全程最优 mIoU | R@0.5 | P@0.5 |")
    lines.append("|---|---|---|---|---|---|---|")
    per_class = {}
    for arm, family, pre in ARMS:
        d = load(arm)
        ev = d.get("eval")
        if not ev:
            lines.append(f"| {arm} | {family} | {pre} | （未完成/未评测） | — | — | — |")
            continue
        best = fmt(d.get("best_miou")) if d.get("best_miou") else "—"
        lines.append(
            f"| {arm} | {family} | {pre} | {fmt(ev.get('mask_miou'))} | {best} | "
            f"{fmt(ev.get('recall@0.5'))} | {fmt(ev.get('precision@0.5'))} |"
        )
        for cid, st in (ev.get("per_class_mask_miou") or {}).items():
            per_class.setdefault(NAMES.get(cid, cid), []).append(
                f"{arm}: {st['mask_miou']:.3f}（{st['gt']}gt）"
            )
    lines.append("")
    if per_class:
        lines.append("## 分类别 mIoU")
        for cls, vals in per_class.items():
            lines.append(f"- **{cls}**：{'；'.join(vals)}")
        lines.append("")
    lines.append("## 口径说明")
    lines.append("- R@0.5 = 同类预测与 gt IoU≥0.5 的 gt 占比；P@0.5 = 分数降序贪心一对一匹配的 TP/预测数")
    lines.append("- mIoU = 每 gt 实例取类无关最优掩码 IoU 的均值（漏检按低值计入）")
    lines.append("- DINOv2 臂分辨率 448/672 由 patch-14 网格约束决定，与其余 640 臂存在分辨率差异（对比时需计入）")
    io.open(os.path.join(RUNS, "COMPARISON.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("COMPARISON.md 已生成")


if __name__ == "__main__":
    main()
