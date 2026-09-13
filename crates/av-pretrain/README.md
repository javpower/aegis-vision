# av-pretrain

AegisVision 预训练权重层（预训练权重方案第一层）：

- **原生权重格式（`av_weight`）**：目录式「每变量一文件」（`Tensor::save`，与
  `av-runtime` 引擎 checkpoint 同一套已验证方案）+ `manifest.json` 元信息
  （框架版本、骨干、来源数据集、epoch、任务、created_at）+ 每张量 blake3 哈希
  （字段名 `hash`）与全量校验 `verify_hashes`（一次报出所有不匹配项并定位文件）。
- **外部权重导入适配（`weight_adapter`）**：读取 PyTorch `.safetensors`
  （包装 `tch::Tensor::read_safetensors`），按 `LayerMap`（正则/前缀 `from` →
  `to` 改名、可选转置）映射到目标模型变量，并与目标形状逐一比对，
  输出 loaded / skipped_shape_mismatch / missing / unexpected 四类完整报告
  （部分加载是行业常态，但必须可见）。

feature 门控与 av-core 一致：`torch`（默认）启用张量层；无 libtorch 环境下
`--no-default-features` 仍可用 manifest + 哈希校验纯逻辑部分。

```toml
[pretrain]
enable = true
weight_path = "data/pretrain/yolov8n_backbone.safetensors"
load_only_backbone = true
freeze_backbone = true
# layer_map = "layer_map.toml"  # [[entries]] from/to/transpose
```
