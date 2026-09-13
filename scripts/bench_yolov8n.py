# -*- coding: utf-8 -*-
"""YOLOv8n 基线脚本 —— BENCHMARK.md §3 的对比数据来源。

用法（需 Python 3.10+，建议独立 venv）：
    python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    python -m pip install ultralytics
    python scripts/bench_yolov8n.py [--epochs 300] [--imgsz 320]

口径与 AV 侧对齐：coco128 训练、训练 split 自评、imgsz 一致。
结果落 data/yolov8n_coco128_results.json 供 BENCHMARK.md 引用。
"""
import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "data" / "coco128.yaml"


def ensure_data_yaml():
    """coco128 数据集 yaml（Ultralytics 官方定义的等价内联版）。"""
    data_root = (ROOT / "data" / "coco128").resolve()
    if not (data_root / "images").is_dir():
        raise SystemExit(
            f"未找到 {data_root}——请先下载 coco128 并解压到 data/coco128"
            "（https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip）"
        )
    yaml = {
        "path": str(data_root),
        "train": "images/train2017",
        "val": "images/train2017",  # coco128 无独立 val，训练 split 自评（与 AV 口径一致）
        "names": {
            i: n
            for i, n in enumerate(
                [
                    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
                    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
                    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
                    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
                    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
                    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
                    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
                    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
                    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
                    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
                    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
                    "scissors", "teddy bear", "hair drier", "toothbrush",
                ]
            )
        },
    }
    import yaml as pyyaml  # ultralytics 依赖链自带

    DATA_YAML.write_text(pyyaml.safe_dump(yaml, allow_unicode=True), encoding="utf-8")
    return str(DATA_YAML)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--model", default="yolov8n.pt")
    args = ap.parse_args()

    from ultralytics import YOLO  # 延迟导入：参数错误时不触发重依赖

    data = ensure_data_yaml()
    model = YOLO(args.model)
    t0 = time.time()
    model.train(
        data=data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=16,
        device="cpu",
        project=str(ROOT / "runs" / "yolov8n-bench"),
        name="coco128",
        exist_ok=True,
        verbose=True,
    )
    elapsed = time.time() - t0
    metrics = model.val(data=data, imgsz=args.imgsz, device="cpu")
    out = {
        "framework": f"ultralytics {__import__('ultralytics').__version__} (CPU)",
        "model": args.model,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "train_seconds": round(elapsed, 1),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
    }
    out_path = ROOT / "data" / "yolov8n_coco128_results.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"已写入 {out_path}")


if __name__ == "__main__":
    main()
