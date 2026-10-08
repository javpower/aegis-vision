//! 外部预训练权重导入适配器：safetensors → AV 变量名空间。
//!
//! 行业常态是**部分加载**：外部权重与目标模型极少同构。适配器职责不是
//! 强行全部对上，而是把「改了什么名、转了什么置、差了多少形」算清楚，
//! 输出 loaded / skipped_shape_mismatch / missing / unexpected 四类完整清单
//! （[`AdaptReport`]），加载成功与否由调用方按报告决定。
//!
//! 映射语义（[`LayerMap`]，可由 TOML 配置）：
//! - `from`：正则（推荐写 `^...$` 锚定；替换串支持 `$1` 捕获引用）；
//!   编译失败时回退为字面前缀匹配（`to + 剩余部分`）；
//! - `to`：目标变量名（作为 `from` 匹配部分的替换文本）；
//! - `transpose`：2 维源张量转置后比对（nn.Linear [out,in] 与外部 [in,out]
//!   约定差异的兜底）；
//! - 逐条取**第一个**匹配的映射；没有任何映射命中时回退为**同名直配**
//!   （原生 avpretrain 目录 / 名字已对齐的 safetensors 无需映射文件）。

use std::path::Path;

use serde::{Deserialize, Serialize};

use av_core::error::{AvError, AvResult};

// ---------------------------------------------------------------------------
// 读取
// ---------------------------------------------------------------------------

/// 读取 safetensors 文件为命名张量列表（包装 tch 0.17 原生 API，不自解析格式）。
pub fn read_safetensors_all(path: &Path) -> AvResult<Vec<(String, tch::Tensor)>> {
    tch::Tensor::read_safetensors(path)
        .map_err(|e| AvError::train(format!("读取 safetensors {} 失败: {e}", path.display())))
}

// ---------------------------------------------------------------------------
// 层映射
// ---------------------------------------------------------------------------

/// 单条层映射。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LayerMapping {
    /// 源层名匹配：正则（支持 `$1` 捕获引用）或字面前缀
    pub from: String,
    /// 目标变量名（替换文本）
    pub to: String,
    /// 2 维源张量先转置再比对（默认 false）
    #[serde(default)]
    pub transpose: bool,
}

/// 层映射集合（TOML 文件用 `[[entries]]` 数组表表达）。
///
/// ```toml
/// [[entries]]
/// from = '^model\.0\.conv\.'
/// to = 'backbone.c1.'
/// transpose = false
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct LayerMap {
    #[serde(default)]
    pub entries: Vec<LayerMapping>,
}

impl LayerMap {
    /// 从 TOML 文件加载层映射。
    pub fn from_toml_path(p: &Path) -> AvResult<Self> {
        let s = std::fs::read_to_string(p)
            .map_err(|e| AvError::config(format!("读取层映射 {} 失败: {e}", p.display())))?;
        toml::from_str(&s)
            .map_err(|e| AvError::config(format!("层映射 {} 解析失败: {e}", p.display())))
    }
}

// ---------------------------------------------------------------------------
// 适配报告
// ---------------------------------------------------------------------------

/// 成功载入的目标变量记录。
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct LoadedTensor {
    pub target: String,
    pub source: String,
    pub shape: Vec<i64>,
}

/// 形状不匹配被跳过的记录（两侧形状都给出，便于排查转置/宽度差异）。
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ShapeMismatch {
    pub target: String,
    pub source: String,
    pub expected: Vec<i64>,
    pub got: Vec<i64>,
}

/// 适配完整报告：四类清单（行业惯例：部分加载合法，但必须完整可见）。
///
/// `tensors` 为匹配成功的目标名张量（供调用方写回 VarStore，不参与序列化；
/// tch 的 `Tensor` 无 `Clone`，故本结构不可整克隆）。
#[derive(Debug, Default, Serialize)]
pub struct AdaptReport {
    pub loaded: Vec<LoadedTensor>,
    pub skipped_shape_mismatch: Vec<ShapeMismatch>,
    /// 目标模型中没有任何源对应的变量
    pub missing: Vec<String>,
    /// 没有落到目标模型变量上的源张量（无映射命中且同名直配也不存在等）
    pub unexpected: Vec<String>,
    #[serde(skip)]
    pub tensors: Vec<(String, tch::Tensor)>,
}

impl AdaptReport {
    /// 单行统计摘要（日志用）。
    pub fn summary(&self) -> String {
        format!(
            "loaded={} skipped_shape_mismatch={} missing={} unexpected={}",
            self.loaded.len(),
            self.skipped_shape_mismatch.len(),
            self.missing.len(),
            self.unexpected.len()
        )
    }
}

// ---------------------------------------------------------------------------
// 适配
// ---------------------------------------------------------------------------

/// 把源张量按映射改写/转置并与目标形状比对，产出报告与可写回张量。
///
/// 纯函数：不写任何模型变量，只做命名映射、按需转置、形状校验与分类。
pub fn adapt(
    sources: Vec<(String, tch::Tensor)>,
    map: &LayerMap,
    target_shapes: &[(String, Vec<i64>)],
) -> AdaptReport {
    // 逐条预编译正则；编译失败的条目回退字面前缀匹配（见模块注释）
    let compiled: Vec<(Option<regex::Regex>, &LayerMapping)> = map
        .entries
        .iter()
        .map(|e| (regex::Regex::new(&e.from).ok(), e))
        .collect();

    let mut report = AdaptReport::default();
    for (src_name, src) in sources {
        let (dst_name, transposed) = match map_name(&compiled, &src_name) {
            Some((dst, tr)) => (dst, tr),
            None => (src_name.clone(), false), // 同名直配回退
        };
        // 按需转置（仅 2 维张量；其余维度无转置语义，保持原样由形状比对裁决）
        let src = if transposed && src.dim() == 2 {
            src.transpose(0, 1).contiguous()
        } else {
            src
        };
        match target_shapes.iter().find(|(n, _)| *n == dst_name) {
            Some((_, expected)) => {
                let got = src.size();
                if &got == expected {
                    report.loaded.push(LoadedTensor {
                        target: dst_name.clone(),
                        source: src_name.clone(),
                        shape: got,
                    });
                    report.tensors.push((dst_name, src));
                } else {
                    report.skipped_shape_mismatch.push(ShapeMismatch {
                        target: dst_name.clone(),
                        source: src_name.clone(),
                        expected: expected.clone(),
                        got,
                    });
                }
            }
            None => report.unexpected.push(format!(
                "源 {src_name} → 目标 {dst_name} 在模型变量中不存在"
            )),
        }
    }
    // missing：目标变量集合 − loaded 集合
    for (n, _) in target_shapes {
        if !report.loaded.iter().any(|l| l.target == *n) {
            report.missing.push(n.clone());
        }
    }
    report
}

/// 返回 (目标名, 是否转置)；无命中返回 None。
fn map_name(
    compiled: &[(Option<regex::Regex>, &LayerMapping)],
    name: &str,
) -> Option<(String, bool)> {
    for (re, e) in compiled {
        match re {
            Some(re) => {
                if re.is_match(name) {
                    // 替换文本支持 $1/$name 捕获引用（regex crate 约定）
                    let replaced = re.replace(name, e.to.as_str());
                    return Some((replaced.into_owned(), e.transpose));
                }
            }
            None => {
                if let Some(rest) = name.strip_prefix(&e.from) {
                    return Some((format!("{}{}", e.to, rest), e.transpose));
                }
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// DINOv2 预设映射（官方 DINOv2 checkpoint → AV 变量名）
// ---------------------------------------------------------------------------

/// DINOv2 官方 checkpoint → AV 变量名的预设映射（正则改写，供引擎通用
/// `[pretrain]` + `layer_map` 通道使用）。
///
/// **适用范围与局限**（重要）：
/// - 目标名前缀 `backbone.` 对应 av-tasks::backbone_dino 的 DinoV2Backbone
///   （建模时挂在 `p / "backbone"` 下，变量名形如 `backbone.blocks.0.attn.qkv.weight`）；
/// - **推荐路径仍是 DINOv2Backbone::load_dinov2_weights**：它能做 pos_embed
///   网格双三次插值（518→目标分辨率）与 HF 拆分式 QKV（query/key/value 三条
///   Linear）的融合——这两件事是正则改名 + 形状比对的通用适配器做不到的，
///   本预设映射下 pos_embed 会以形状不符跳过、QKV 三分量无对应目标；
/// - 对 torch.hub 原生命名（`blocks.{i}.attn.qkv.*` 融合版，可用
///   tools/export/export_dinov2.py --hub 导出）本预设可全量改名直配；
/// - `dinov2.` 前缀的 HF transformers 键名按当前版本实测键名编写
///   （`embeddings.*` / `encoder.layer.{i}.*` / `layernorm.*`）。
pub fn dinov2_layer_map() -> LayerMap {
    let e = |from: &str, to: &str| LayerMapping {
        from: from.into(),
        to: to.into(),
        transpose: false,
    };
    LayerMap {
        entries: vec![
            // embeddings / 全局
            e(r"^dinov2\.embeddings\.cls_token$", "backbone.cls_token"),
            e(
                r"^dinov2\.embeddings\.position_embeddings$",
                "backbone.pos_embed",
            ),
            e(
                r"^dinov2\.embeddings\.patch_embeddings\.projection\.(.+)$",
                "backbone.patch_embed.proj.$1",
            ),
            e(r"^dinov2\.layernorm\.(.+)$", "backbone.norm.$1"),
            // encoder 层（norm1/norm2 为当前 HF 版本命名；layernorm_before/after 为旧版）
            e(
                r"^dinov2\.encoder\.layer\.(\d+)\.attention\.output\.dense\.(.+)$",
                "backbone.blocks.$1.attn.proj.$2",
            ),
            e(
                r"^dinov2\.encoder\.layer\.(\d+)\.norm1\.(.+)$",
                "backbone.blocks.$1.norm1.$2",
            ),
            e(
                r"^dinov2\.encoder\.layer\.(\d+)\.norm2\.(.+)$",
                "backbone.blocks.$1.norm2.$2",
            ),
            e(
                r"^dinov2\.encoder\.layer\.(\d+)\.layernorm_before\.(.+)$",
                "backbone.blocks.$1.norm1.$2",
            ),
            e(
                r"^dinov2\.encoder\.layer\.(\d+)\.layernorm_after\.(.+)$",
                "backbone.blocks.$1.norm2.$2",
            ),
            e(
                r"^dinov2\.encoder\.layer\.(\d+)\.mlp\.fc1\.(.+)$",
                "backbone.blocks.$1.mlp.fc1.$2",
            ),
            e(
                r"^dinov2\.encoder\.layer\.(\d+)\.mlp\.fc2\.(.+)$",
                "backbone.blocks.$1.mlp.fc2.$2",
            ),
            e(
                r"^dinov2\.encoder\.layer\.(\d+)\.layer_scale1\.lambda1$",
                "backbone.blocks.$1.ls1.gamma",
            ),
            e(
                r"^dinov2\.encoder\.layer\.(\d+)\.layer_scale2\.lambda1$",
                "backbone.blocks.$1.ls2.gamma",
            ),
            // torch.hub 原生命名（融合 QKV）直配
            e(r"^blocks\.(\d+)\.(.+)$", "backbone.blocks.$1.$2"),
            e(r"^patch_embed\.proj\.(.+)$", "backbone.patch_embed.proj.$1"),
            e(r"^pos_embed$", "backbone.pos_embed"),
            e(r"^cls_token$", "backbone.cls_token"),
            e(r"^norm\.(weight|bias)$", "backbone.norm.$1"),
        ],
    }
}

// ---------------------------------------------------------------------------
// ResNet18 预设映射（torchvision ResNet18 → AV 变量名）
// ---------------------------------------------------------------------------

/// torchvision ResNet18 → AV 变量名的预设映射（供引擎通用 `[pretrain]` +
/// `layer_map` 通道；等价 TOML 见 configs/resnet18_map.toml）。
///
/// **BN 统计量无需特殊处理**：tch 0.24 `nn::batch_norm2d` 的
/// running_mean/running_var 是 VarStore 命名变量（no_train，见
/// av-tasks::backbone_resnet 模块注释），与普通权重同一条导入链路；
/// conv/bn 参数与统计量全部与 torchvision **同名**，故一条前缀映射即可
/// 覆盖全部 100 个骨干张量（20 conv + 40 BN 参数 + 40 BN 统计量）。
///
/// `fc.*`（1000 类分类头）与 `*.num_batches_tracked`（tch 无对应变量）
/// 不匹配任何映射、同名直配也无目标 → 报告中以 unexpected 完整呈现
/// （行业惯例：部分加载 + 完整报告）。
pub fn resnet18_preset() -> LayerMap {
    LayerMap {
        entries: vec![LayerMapping {
            from: r"^(conv1|bn1|layer[1-4])\.(.+)$".into(),
            to: "backbone.$1.$2".into(),
            // torchvision conv/linear 权重 [out,in,...] 与 tch 变量同约定，无需转置
            transpose: false,
        }],
    }
}

#[cfg(all(test, feature = "torch"))]
mod tests {
    use super::*;
    use tch::{Kind, Tensor};

    fn f32_tensor(v: &[f32], shape: &[i64]) -> Tensor {
        Tensor::from_slice(v).to_kind(Kind::Float).reshape(shape)
    }

    /// 手算对照：映射改名 + 转置 + 形状不匹配跳过 + missing/unexpected。
    #[test]
    fn adapt_mapping_transpose_and_shape_mismatch_hand_computed() {
        // 源：s1 [2,3] = 1..6；s2 [3] = 7..9；s3 无映射价值；s4 与目标形状不符
        let sources = vec![
            (
                "model.0.conv.weight".into(),
                f32_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]),
            ),
            (
                "model.0.conv.bias".into(),
                f32_tensor(&[7.0, 8.0, 9.0], &[3]),
            ),
            ("extra.unknown".into(), f32_tensor(&[42.0], &[1])),
            (
                "model.1.conv.weight".into(),
                f32_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]),
            ),
        ];
        let map = LayerMap {
            entries: vec![
                LayerMapping {
                    from: r"^model\.0\.conv\.(.+)$".into(),
                    to: "backbone.c1.$1".into(),
                    transpose: true, // [2,3] → [3,2]
                },
                LayerMapping {
                    from: r"^model\.1\.conv\.weight$".into(),
                    to: "backbone.c1.weight".into(), // 目标期望 [3,2]，源 [2,3] → 跳过
                    transpose: false,
                },
            ],
        };
        let targets = vec![
            ("backbone.c1.weight".to_string(), vec![3i64, 2]),
            ("backbone.c1.bias".to_string(), vec![3]),
            ("head.fc.weight".to_string(), vec![4, 3]),
        ];

        let report = adapt(sources, &map, &targets);

        // loaded：weight（转置后）+ bias
        assert_eq!(report.loaded.len(), 2, "{report:?}");
        assert_eq!(report.loaded[0].target, "backbone.c1.weight");
        assert_eq!(report.loaded[0].source, "model.0.conv.weight");
        assert_eq!(report.loaded[0].shape, vec![3, 2]);

        // 转置数值手算：t[i][j] = s[j][i]；s = [[1,2,3],[4,5,6]] → t = [[1,4],[2,5],[3,6]]
        let w = &report
            .tensors
            .iter()
            .find(|(n, _)| n == "backbone.c1.weight")
            .unwrap()
            .1;
        assert_eq!(w.size(), vec![3, 2]);
        let expect = [[1.0f32, 4.0], [2.0, 5.0], [3.0, 6.0]];
        for (i, row) in expect.iter().enumerate() {
            for (j, v) in row.iter().enumerate() {
                assert!(
                    (w.double_value(&[i as i64, j as i64]) - *v as f64).abs() < 1e-6,
                    "t[{i}][{j}] 应为 {v}"
                );
            }
        }

        // skipped_shape_mismatch：s4 [2,3] vs 期望 [3,2]
        assert_eq!(report.skipped_shape_mismatch.len(), 1, "{report:?}");
        let sm = &report.skipped_shape_mismatch[0];
        assert_eq!(sm.source, "model.1.conv.weight");
        assert_eq!(sm.expected, vec![3, 2]);
        assert_eq!(sm.got, vec![2, 3]);

        // missing：head.fc.weight 无任何源对应
        assert_eq!(report.missing, vec!["head.fc.weight".to_string()]);

        // unexpected：extra.unknown 同名直配落空
        assert_eq!(report.unexpected.len(), 1, "{report:?}");
        assert!(report.unexpected[0].contains("extra.unknown"));
    }

    /// 无映射文件时的同名直配回退（原生 avpretrain 目录即此路径）。
    #[test]
    fn identity_fallback_loads_matching_names() {
        let sources = vec![("backbone.c1.weight".into(), f32_tensor(&[1.0, 2.0], &[2]))];
        let targets = vec![
            ("backbone.c1.weight".to_string(), vec![2]),
            ("backbone.c1.bias".to_string(), vec![2]),
        ];
        let report = adapt(sources, &LayerMap::default(), &targets);
        assert_eq!(report.loaded.len(), 1);
        assert_eq!(report.loaded[0].target, "backbone.c1.weight");
        assert_eq!(report.missing, vec!["backbone.c1.bias".to_string()]);
        assert!(report.unexpected.is_empty());
    }

    /// 非法正则回退为字面前缀匹配。
    #[test]
    fn invalid_regex_falls_back_to_literal_prefix() {
        let targets = vec![("x.blk0.w".to_string(), vec![1i64])];
        let src = || vec![("encoder.blk0.w".into(), f32_tensor(&[1.0], &[1]))];

        // 非法正则 "([" → 编译失败 → 回退字面前缀 "(["，不匹配 "encoder.blk0.w"
        let bad = LayerMap {
            entries: vec![LayerMapping {
                from: "([".into(),
                to: "x.".into(),
                transpose: false,
            }],
        };
        let r1 = adapt(src(), &bad, &targets);
        assert_eq!(r1.loaded.len(), 0, "{r1:?}");
        assert_eq!(r1.unexpected.len(), 1, "{r1:?}");

        // 合法字面前缀语义（无正则元字符）：strip_prefix + 拼接
        let prefix = LayerMap {
            entries: vec![LayerMapping {
                from: "encoder.".into(),
                to: "x.".into(),
                transpose: false,
            }],
        };
        let r2 = adapt(src(), &prefix, &targets);
        assert_eq!(r2.loaded.len(), 1, "{r2:?}");
        assert_eq!(r2.loaded[0].target, "x.blk0.w");
    }

    /// LayerMap TOML 反序列化（[[entries]] 数组表）。
    #[test]
    fn layer_map_from_toml_str() {
        #[derive(Deserialize)]
        struct Wrap {
            #[serde(default)]
            entries: Vec<LayerMapping>,
        }
        let s = r#"
            [[entries]]
            from = '^model\.0\.conv\.'
            to = 'backbone.c1.'
            transpose = true

            [[entries]]
            from = 'plain.prefix.'
            to = 'p.'
        "#;
        let w: Wrap = toml::from_str(s).expect("应可解析");
        assert_eq!(w.entries.len(), 2);
        assert!(w.entries[0].transpose);
        assert!(!w.entries[1].transpose);
    }

    /// adapt 报告可序列化（供日志/产物落盘）。
    #[test]
    fn report_is_serializable() {
        let sources = vec![("a.w".into(), f32_tensor(&[1.0], &[1]))];
        let report = adapt(
            sources,
            &LayerMap::default(),
            &[("a.w".to_string(), vec![1])],
        );
        let s = serde_json::to_string(&report).expect("应可序列化");
        assert!(s.contains("\"loaded\"") && s.contains("\"missing\""));
        assert!(!s.contains("tensors"), "张量数据不应进入序列化输出");
    }

    /// DINOv2 预设映射：全部条目正则合法 + HF/hub 键名样例改写正确 +
    /// 与目标变量名精确命中（loaded）。
    #[test]
    fn dinov2_preset_map_rewrites_hf_and_hub_names() {
        let map = dinov2_layer_map();
        assert!(!map.entries.is_empty());
        // 全部条目必须编译为合法正则（否则运行期静默回退字面前缀）
        for e in &map.entries {
            assert!(regex::Regex::new(&e.from).is_ok(), "非法正则: {}", e.from);
        }
        let targets = vec![
            (
                "backbone.blocks.0.attn.proj.weight".to_string(),
                vec![384, 384],
            ),
            ("backbone.blocks.11.ls2.gamma".to_string(), vec![384]),
            (
                "backbone.patch_embed.proj.weight".to_string(),
                vec![384, 3, 14, 14],
            ),
            ("backbone.norm.bias".to_string(), vec![384]),
            ("backbone.blocks.5.mlp.fc1.bias".to_string(), vec![1536]),
        ];
        let ones = |shape: &[i64]| Tensor::ones(shape, (tch::Kind::Float, tch::Device::Cpu));
        let sources = vec![
            (
                "dinov2.encoder.layer.0.attention.output.dense.weight".into(),
                ones(&[384, 384]),
            ),
            (
                "dinov2.encoder.layer.11.layer_scale2.lambda1".into(),
                ones(&[384]),
            ),
            (
                "dinov2.embeddings.patch_embeddings.projection.weight".into(),
                ones(&[384, 3, 14, 14]),
            ),
            ("dinov2.layernorm.bias".into(), ones(&[384])),
            ("blocks.5.mlp.fc1.bias".into(), ones(&[1536])),
        ];
        let report = adapt(sources, &map, &targets);
        assert_eq!(report.loaded.len(), 5, "{report:?}");
        assert!(report.skipped_shape_mismatch.is_empty());
        // mask_token 不在映射表中 → unexpected（pos_embed 有映射条目，见 r3）
        let r2 = adapt(
            vec![("dinov2.embeddings.mask_token".into(), ones(&[1, 384]))],
            &map,
            &[("backbone.pos_embed".to_string(), vec![1, 257, 384])],
        );
        assert!(r2.loaded.is_empty() && r2.unexpected.len() == 1, "{r2:?}");
        // pos_embed 有映射条目，但 518 训练网格（37×37）与目标网格形状不符 →
        // skipped_shape_mismatch：网格插值必须走 DinoV2Backbone::load_dinov2_weights
        // （通用适配器不做插值，见 dinov2_layer_map 文档）
        let r3 = adapt(
            vec![(
                "dinov2.embeddings.position_embeddings".into(),
                ones(&[1, 1370, 384]),
            )],
            &map,
            &[("backbone.pos_embed".to_string(), vec![1, 257, 384])],
        );
        assert!(
            r3.loaded.is_empty() && r3.skipped_shape_mismatch.len() == 1,
            "{r3:?}"
        );
    }

    /// ResNet18 预设映射：torchvision 名 → `backbone.` 前缀，conv/BN 参数与
    /// running 统计量全覆盖；fc / num_batches_tracked 落 unexpected。
    #[test]
    fn resnet18_preset_maps_torchvision_names() {
        let map = resnet18_preset();
        let ones = |shape: &[i64]| Tensor::ones(shape, (tch::Kind::Float, tch::Device::Cpu));
        let sources = vec![
            ("conv1.weight".into(), ones(&[64, 3, 7, 7])),
            ("bn1.weight".into(), ones(&[64])),
            ("bn1.running_mean".into(), ones(&[64])),
            ("layer1.0.conv1.weight".into(), ones(&[64, 64, 3, 3])),
            ("layer2.0.downsample.1.running_var".into(), ones(&[128])),
            ("layer4.1.bn2.bias".into(), ones(&[512])),
            // 无目标：1000 类分类头 / tch 无 num_batches_tracked 变量
            ("fc.weight".into(), ones(&[1000, 512])),
            ("layer1.0.bn1.num_batches_tracked".into(), ones(&[1])),
        ];
        let targets = vec![
            ("backbone.conv1.weight".to_string(), vec![64, 3, 7, 7]),
            ("backbone.bn1.weight".to_string(), vec![64]),
            ("backbone.bn1.running_mean".to_string(), vec![64]),
            (
                "backbone.layer1.0.conv1.weight".to_string(),
                vec![64, 64, 3, 3],
            ),
            (
                "backbone.layer2.0.downsample.1.running_var".to_string(),
                vec![128],
            ),
            ("backbone.layer4.1.bn2.bias".to_string(), vec![512]),
        ];
        let report = adapt(sources, &map, &targets);
        assert_eq!(report.loaded.len(), 6, "{report:?}");
        assert!(report.missing.is_empty(), "100 个骨干目标全部可映射");
        assert!(report.skipped_shape_mismatch.is_empty());
        assert_eq!(report.unexpected.len(), 2, "{:?}", report.unexpected);
        assert!(report.unexpected.iter().any(|u| u.contains("fc.weight")));
        assert!(report
            .unexpected
            .iter()
            .any(|u| u.contains("num_batches_tracked")));
        assert_eq!(
            report.loaded[0].target, "backbone.conv1.weight",
            "映射后目标名必须与 AV 变量名逐字一致"
        );
    }
}
