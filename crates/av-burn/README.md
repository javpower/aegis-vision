# aegisvision-burn（lib 名 `av_burn`）

AegisVision 的 **纯 Rust 分割后端**（burn 0.21 框架）：CSP-ELAN 骨干 +
YOLACT 式「原型 × 系数」实例分割，训练 → 保存 → 加载 → 推理完整闭环，
**零 libtorch / C++ 工具链依赖**。语义与主后端（tch 版 `aegisvision-tasks` /
`aegisvision-runtime`）逐链路对齐：同款数据格式（YOLO-seg 目录、data.yaml）、
同款 `SegInstance` 输出、同款 conf/iou 语义。

安装（二选一）：

```bash
cargo add aegisvision-burn                      # 库依赖
cargo install aegisvision-burn                  # CLI（CPU，ndarray 后端）
cargo install aegisvision-burn --features wgpu  # CLI（GPU，wgpu 后端）
```

一条命令训练（与主 CLI `av` 体验一致）：

```bash
avb train --data <数据目录|data.yaml> --epochs 100        # CPU；GPU 加 --device gpu
avb predict --weights runs-avb/<name> --input img.jpg --save-viz out/ --json res.json
```

库 API 最小示例：

```rust
use av_burn::{SegNet, NdArrayB, TrainB};
use av_burn::checkpoint::{save_model, SegSnapshot};

let device = Default::default();
let cfg = av_burn::seg::SegNetCfg { width: 0.25, depth: 0.33,
    num_classes: 2, num_protos: 32, loss_w_bce: 1.0, loss_w_dice: 1.0 };
let model = SegNet::<TrainB>::new(&cfg, &device).unwrap();
// ... train_step 循环见 crate 文档/示例 ...
let snap = SegSnapshot { format: 1, imgsz: 640, width: 0.25, depth: 0.33,
    num_classes: 2, num_protos: 32, loss_w_bce: 1.0, loss_w_dice: 1.0,
    classes: vec!["cat".into(), "dog".into()] };
save_model(&model.valid(), &snap, "runs-avb/demo".as_ref()).unwrap();

// 推理：超参从快照自动重建
use av_burn::checkpoint::load_model;
use av_burn::infer::predict_image;
let (model, snap) = load_model::<NdArrayB>("runs-avb/demo".as_ref(), &device).unwrap();
let rgb = image::open("img.jpg").unwrap().to_rgb8();
let (instances, _lb) =
    predict_image(&model, &device, &rgb, snap.imgsz, 0.25, 0.7).unwrap();
```

诚实边界（与 tch 版的差异，详见 crate 文档）：无 ultralytics 权重命名对齐
（checkpoint 为 burn 原生格式）；头内上采样为 nearest（ndarray 无双线性反传，
wgpu 可切回）；ndarray 后端无 BLAS，适合冒烟/小模型，正式训练用 wgpu 或
tch 主后端。性能基线：wgpu 实测 nano@640 ≈ 27s/epoch（tch CUDA ≈ 7.5s）。

本 crate 是 [AegisVision](https://crates.io/crates/aegisvision-runtime) 工作区成员。
框架完整介绍见[工作区根 README](https://github.com/javpower/aegis-vision#readme)。

双许可：[MIT](https://github.com/javpower/aegis-vision/blob/main/LICENSE-MIT) 或
[Apache-2.0](https://github.com/javpower/aegis-vision/blob/main/LICENSE-APACHE)。
