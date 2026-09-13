# scripts/bench_ultralytics_seg.py —— ultralytics 同卡同数据基准（管线对比用）
# 用法: D:/Dev/Tools/Miniconda3/python.exe scripts/bench_ultralytics_seg.py
# 输出: 每 epoch 耗时（ultralytics 自带计时打印），供与 aegis-vision 管线 v2 对比
from ultralytics import YOLO

if __name__ == "__main__":  # Windows DataLoader 多进程必需
    model = YOLO("yolo11n-seg.pt")  # 官方预训练权重（~5.7MB）
    model.train(
        data=r"E:/xunlian/yolo_dataset/data.yaml",
        epochs=2,            # 只测吞吐，不比精度（精度对比见交付报告口径说明）
        imgsz=640,
        batch=16,
        device=0,
        workers=8,
        cache=False,
        plots=False,
        verbose=True,
        name="av_bench_yolo11n_seg",
    )
