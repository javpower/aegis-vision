# av-core

AegisVision 公共抽象层：六大核心 trait（Backbone / Neck / TaskHead / Loss / Decoder / Dataset）、
强类型配置 schema（`deny_unknown_fields` + 语义校验 + 快照）、几何工具（AABB / 旋转框 IoU /
letterbox 映射）与插件注册表（`av_plugin!` 宏）。

无 `torch` feature 时为纯 Rust 实现（可在无 libtorch 环境编译，用于 CI 快速回归与 docs.rs 构建）。

本 crate 是 [AegisVision](https://crates.io/crates/av-runtime) 工作区成员。框架完整介绍、
能力矩阵与快速开始见[工作区根 README](https://github.com/aegisvision/aegis-vision#readme)。

双许可：[MIT](https://github.com/aegisvision/aegis-vision/blob/main/LICENSE-MIT) 或
[Apache-2.0](https://github.com/aegisvision/aegis-vision/blob/main/LICENSE-APACHE)。
