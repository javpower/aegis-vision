# av-runtime

AegisVision 训练 / 推理引擎（库）与 `av` CLI（薄封装）。配置驱动的端到端工作流：
配置加载 → 语义校验 → 训练 → 评测 → checkpoint 落盘 → 推理 JSON。单二进制部署，
无 Python 运行时依赖。

本 crate 是 [AegisVision](https://crates.io/crates/aegisvision-runtime) 工作区成员（主入口）。
框架完整介绍、能力矩阵与快速开始见
[工作区根 README](https://github.com/javpower/aegis-vision#readme)。

可编译示例（`cargo run -p aegisvision-runtime --example <名字>`）：

- `train_smoke`：合成数据分类端到端冒烟；
- `detect_coco8`：真实数据（coco8，YOLO 目录格式）检测训练；
- `print_plan`：等价于 `av train --dry-run` 的配置解析 + 校验 + 运行计划打印。

双许可：[MIT](https://github.com/javpower/aegis-vision/blob/main/LICENSE-MIT) 或
[Apache-2.0](https://github.com/javpower/aegis-vision/blob/main/LICENSE-APACHE)。
