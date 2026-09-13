//! DINOv2 ViT 骨干族（ViT-S/14 起步）——RF-DETR 同款配方（其骨干即 DINOv2 预训练）。
//!
//! 结构 = 官方 dinov2_vits14：PatchEmbed（14×14 conv stride 14）+ cls token +
//! 12 × TransformerBlock（LN → 融合 QKV 的 6 头 MHSA → 残差 → LN → 4× GELU MLP → 残差，
//! 带 LayerScale）+ 最终 LayerNorm。变量命名沿用 torch.hub 原生命名
//! （`blocks.{i}.norm1` / `attn.qkv` / `ls1.gamma` ...），使官方 checkpoint 近乎同名直配。
//!
//! 与 SimpleCnnBackbone 的关系：同 [`BaseBackbone`] 契约。本期 DINOv2 只接分类
//! （`forward_pooled` = cls token）；`forward_features` 输出 ViTDet 式简单特征金字塔
//! （stride 8/16/32 三级 ×256 通道），供下一波检测头金字塔接入。
//!
//! 关键难点：**位置编码插值**。官方权重按 518px（37×37 patch 网格）训练，
//! `pos_embed [1, 1+N, 384]` 含 cls 行 + 网格行。输入其他分辨率时把网格部分
//! reshape 回 [1, 37, 37, D] 双三次插值到目标网格再摊平（[`Self::interpolate_pos_encoding`]，
//! 参考实现即 tch 自带 vision::dinov2）。导入权重时（[`Self::load_dinov2_weights`]）
//! 额外做一次同款插值把 checkpoint 的 37×37 烘焙到目标网格。
//!
//! **register tokens**：DINOv2-with-registers 系 checkpoint 含 `register_tokens [1,4,D]`，
//! 且其 pos_embed 行布局为 [cls, R×register, patch...]。本实现的前向不含 register
//! （plain ViT-S/14），导入时自动识别：`register_tokens` 本身跳过（记入
//! skipped_unrecognized，注释见 [`DinoLoadStats`]），pos_embed 剥离前 R 个 register
//! 行后只对 patch 网格行插值——两个家族的 checkpoint 都能正确导入。
//!
//! 模型规模：`depth` / `width` 配置字段是 csp-elan 的缩放旋钮，本实现固定
//! ViT-S/14（384 宽 / 12 层 / 6 头），保证与官方预训练严格同构（不同宽度会
//! 静默破坏导入，宁可不做）。

use std::path::Path;

use tch::nn;
use tch::nn::Module;
use tch::{IndexOp, Kind, Tensor};

use av_core::config::BackboneCfg;
use av_core::error::{AvError, AvResult};
use av_core::traits::{BaseBackbone, BackboneSpec, FeatureMap, FeaturePyramid, LevelSpec};

pub const FAMILY_NAME: &str = "dinov2";

/// patch 边长（/14 家族）。
const PATCH_SIZE: i64 = 14;
/// ViT-S/14 官方规格。
const EMBED_DIM: i64 = 384;
const DEPTH: usize = 12;
const NUM_HEADS: i64 = 6;
/// ViTDet 式简单金字塔的统一输出通道。
const PYRAMID_CHANNELS: i64 = 256;
/// DINOv2 官方 LayerNorm eps（HF Dinov2Config.layer_norm_eps 同值）。
const LN_EPS: f64 = 1e-6;

fn ln_config() -> nn::LayerNormConfig {
    nn::LayerNormConfig {
        eps: LN_EPS,
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// 子模块（命名与 torch.hub 官方一致，便于权重直配）
// ---------------------------------------------------------------------------

/// 14×14 stride-14 卷积 patch embedding → [B, N, D]。
struct PatchEmbed {
    proj: nn::Conv2D,
}

impl PatchEmbed {
    fn new(p: nn::Path) -> Self {
        let config = nn::ConvConfig {
            stride: PATCH_SIZE,
            ..Default::default()
        };
        Self {
            proj: nn::conv2d(p / "proj", 3, EMBED_DIM, PATCH_SIZE, config),
        }
    }

    fn forward(&self, xs: &Tensor) -> Tensor {
        let xs = xs.apply(&self.proj); // [B, D, H/14, W/14]
        let (b, c, h, w) = xs.size4().unwrap();
        xs.reshape([b, c, h * w]).transpose(1, 2) // [B, N, D]
    }
}

/// 融合 QKV 的多头自注意力（官方 checkpoint 即单条 qkv Linear [3D, D]）。
struct Attention {
    qkv: nn::Linear,
    proj: nn::Linear,
    num_heads: i64,
    scale: f64,
}

impl Attention {
    fn new(p: nn::Path, dim: i64, num_heads: i64) -> Self {
        Self {
            qkv: nn::linear(&p / "qkv", dim, dim * 3, Default::default()),
            proj: nn::linear(&p / "proj", dim, dim, Default::default()),
            num_heads,
            scale: 1.0 / (((dim / num_heads) as f64).sqrt()),
        }
    }

    fn forward(&self, xs: &Tensor) -> Tensor {
        let (b, n, c) = xs.size3().unwrap();
        let qkv = self
            .qkv
            .forward(xs)
            .reshape([b, n, 3, self.num_heads, c / self.num_heads])
            .permute([2, 0, 3, 1, 4]);
        let q = qkv.get(0) * self.scale;
        let k = qkv.get(1);
        let v = qkv.get(2);
        let attn = q.matmul(&k.transpose(-2, -1)).softmax(-1, Kind::Float);
        attn.matmul(&v)
            .transpose(1, 2)
            .reshape([b, n, c])
            .apply(&self.proj)
    }
}

/// 4× GELU MLP（ViT-S 用普通 GELU，非 SwiGLU——swiglu 仅 giant 档）。
struct Mlp {
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl Mlp {
    fn new(p: nn::Path, dim: i64) -> Self {
        Self {
            fc1: nn::linear(&p / "fc1", dim, dim * 4, Default::default()),
            fc2: nn::linear(&p / "fc2", dim * 4, dim, Default::default()),
        }
    }

    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.apply(&self.fc1).gelu("none").apply(&self.fc2)
    }
}

/// TransformerBlock：LN → MHSA → LayerScale → 残差 → LN → MLP → LayerScale → 残差。
/// DINOv2 的 layer_scale 初始为 1.0（官方 layer_scale_init=1.0，checkpoint 有实值可导入）。
struct Block {
    norm1: nn::LayerNorm,
    attn: Attention,
    ls1: Tensor,
    norm2: nn::LayerNorm,
    mlp: Mlp,
    ls2: Tensor,
}

impl Block {
    fn new(p: nn::Path, dim: i64, num_heads: i64) -> Self {
        Self {
            norm1: nn::layer_norm(&p / "norm1", vec![dim], ln_config()),
            attn: Attention::new(&p / "attn", dim, num_heads),
            ls1: (&p / "ls1").var("gamma", &[dim], nn::Init::Const(1.0)),
            norm2: nn::layer_norm(&p / "norm2", vec![dim], ln_config()),
            mlp: Mlp::new(&p / "mlp", dim),
            ls2: (&p / "ls2").var("gamma", &[dim], nn::Init::Const(1.0)),
        }
    }

    fn forward(&self, xs: &Tensor) -> Tensor {
        let x = xs + &(self.attn.forward(&xs.apply(&self.norm1)) * &self.ls1);
        &x + &(self.mlp.forward(&x.apply(&self.norm2)) * &self.ls2)
    }
}

// ---------------------------------------------------------------------------
// 骨干
// ---------------------------------------------------------------------------

pub struct DinoV2Backbone {
    patch_embed: PatchEmbed,
    cls_token: Tensor,
    /// [1, 1+N, D]；行 0 = cls，行 1.. = patch 网格（行优先摊平）。
    pos_embed: Tensor,
    blocks: Vec<Block>,
    norm: nn::LayerNorm,
    // ViTDet 式简单金字塔分支（3×3 conv 对齐到 256 通道；随机初始化，检测接入时训练）
    conv_s8: nn::Conv2D,
    conv_s16: nn::Conv2D,
    conv_s32: nn::Conv2D,
    img_size: i64,
    grid_side: i64,
}

impl DinoV2Backbone {
    /// `img_size` 须同时为 14 的倍数（patch 网格整数）与 32 的倍数
    /// （金字塔 stride 8/16/32 整除 + 项目 32 对齐约定）→ 实际即 224 的倍数。
    pub fn new(p: &nn::Path, _cfg: &BackboneCfg, img_size: u32) -> AvResult<Self> {
        let img = img_size as i64;
        if img % PATCH_SIZE != 0 {
            return Err(AvError::config(format!(
                "{FAMILY_NAME}: img_size 必须为 {PATCH_SIZE} 的倍数（patch 网格整数），得到 {img}"
            )));
        }
        if img % 32 != 0 {
            return Err(AvError::config(format!(
                "{FAMILY_NAME}: img_size 必须为 32 的倍数（金字塔 stride 8/16/32 整除 + \
                 项目 32 对齐约定），得到 {img}"
            )));
        }
        let grid_side = img / PATCH_SIZE;
        let n = grid_side * grid_side;
        let cc = nn::ConvConfig {
            padding: 1,
            ..Default::default()
        };
        Ok(Self {
            patch_embed: PatchEmbed::new(p / "patch_embed"),
            cls_token: p.var("cls_token", &[1, 1, EMBED_DIM], nn::Init::Const(0.)),
            pos_embed: p.var("pos_embed", &[1, 1 + n, EMBED_DIM], nn::Init::Const(0.)),
            blocks: (0..DEPTH)
                .map(|i| Block::new(p / "blocks" / i, EMBED_DIM, NUM_HEADS))
                .collect(),
            norm: nn::layer_norm(p / "norm", vec![EMBED_DIM], ln_config()),
            conv_s8: nn::conv2d(p / "pyramid" / "stride8", EMBED_DIM, PYRAMID_CHANNELS, 3, cc),
            conv_s16: nn::conv2d(p / "pyramid" / "stride16", EMBED_DIM, PYRAMID_CHANNELS, 3, cc),
            conv_s32: nn::conv2d(p / "pyramid" / "stride32", EMBED_DIM, PYRAMID_CHANNELS, 3, cc),
            img_size: img,
            grid_side,
        })
    }

    /// 分类池化通道 = cls token 维度。
    pub fn pooled_channels(&self) -> i64 {
        EMBED_DIM
    }

    /// 检测金字塔通道（stride 8/16/32 统一 256）。
    pub fn stride_channels(&self, stride: u32) -> AvResult<i64> {
        match stride {
            8 | 16 | 32 => Ok(PYRAMID_CHANNELS),
            other => Err(AvError::shape(format!(
                "{FAMILY_NAME} 不存在 stride {other} 的特征层"
            ))),
        }
    }

    /// token 化：patch embed + cls 拼接 + 位置编码（必要时插值）。
    fn prepare_tokens(&self, xs: &Tensor) -> Tensor {
        let (b, _c, h, w) = xs.size4().unwrap();
        let patch = self.patch_embed.forward(xs);
        let cls = self.cls_token.expand([b, -1, -1], false);
        let x = Tensor::cat(&[&cls, &patch], 1);
        let pos = self.interpolate_pos_encoding(patch.size()[1], w, h);
        &x + &pos
    }

    /// 位置编码插值（DINOv2 官方同款，参考 tch vision::dinov2 实现）：
    /// 网格部分 [1, N, D] → [1, s, s, D] → 双三次上/下采样到 [1, h/14, w/14, D] → 摊平，
    /// cls 行原样拼接。目标网格与训练网格一致时零拷贝返回。
    fn interpolate_pos_encoding(&self, npatch: i64, w: i64, h: i64) -> Tensor {
        let n = self.pos_embed.size()[1] - 1;
        if npatch == n && w == h {
            return self.pos_embed.copy();
        }
        let class_pos = self.pos_embed.i((.., ..1));
        let patch_pos = self.pos_embed.i((.., 1..));
        let dim = EMBED_DIM;
        let sqrt_n = (n as f64).sqrt();
        // 官方 +0.1 技巧：源网格 37×37@518 这类非整比值时四舍五入到最近整数网格
        let (w0, h0) = ((w / PATCH_SIZE) as f64 + 0.1, (h / PATCH_SIZE) as f64 + 0.1);
        let patch_pos = patch_pos
            .reshape([1, sqrt_n as i64, sqrt_n as i64, dim])
            .permute([0, 3, 1, 2])
            .upsample_bicubic2d(
                [w0 as i64, h0 as i64],
                false,
                w0 / sqrt_n,
                h0 / sqrt_n,
            )
            .permute([0, 2, 3, 1])
            .reshape([1, -1, dim]);
        Tensor::cat(&[&class_pos, &patch_pos], 1)
    }

    /// 编码到最终 LayerNorm 后的 token 序列 [B, 1+N, D]。
    fn encode_tokens(&self, xs: &Tensor) -> Tensor {
        let mut t = self.prepare_tokens(xs);
        for blk in &self.blocks {
            t = blk.forward(&t);
        }
        t.apply(&self.norm)
    }

    /// 最终 patch tokens [B, N, D] → 空间网格 [B, D, H', W']。
    /// reshape 在 permute 后的非连续视图上会隐式拷贝（torch reshape 语义）。
    fn patch_grid(&self, tokens: &Tensor) -> Tensor {
        let b = tokens.size()[0];
        tokens
            .i((.., 1..))
            .permute([0, 2, 1])
            .reshape([b, EMBED_DIM, self.grid_side, self.grid_side])
    }

    // -----------------------------------------------------------------------
    // 权重导入
    // -----------------------------------------------------------------------

    /// 官方 DINOv2 权重导入（safetensors）：
    /// - 命名映射：HF transformers（`embeddings.*` / `encoder.layer.{i}.*`，见
    ///   [`map_source_name`]）与 torch.hub 原生（`blocks.{i}.attn.qkv.*`）两套命名都支持；
    ///   HF 新版把 QKV 拆成 query/key/value 三条 Linear，导入时按官方顺序
    ///   `cat([q, k, v], dim=0)` 融合进本实现的单条 qkv Linear（与
    ///   transformers `Dinov2SelfAttention.qkv` / hub `attn.qkv` 同布局）；
    /// - pos_embed：剥离 register 行（registers 家族）后按目标网格双三次插值（见模块注释）；
    /// - 形状不符的映射命中跳过并记录（部分加载合法，统计完整可见）。
    ///
    /// 注：safetensors 读取直接用 tch 原生 `Tensor::read_safetensors`
    /// （av-pretrain::weight_adapter 的读取即该 API 的薄包装；av-tasks 不依赖
    /// av-pretrain，避免为读一个文件新增 crate 依赖。HF → 本实现的完整映射
    /// 预设在 av-pretrain::weight_adapter::dinov2_layer_map，供引擎通用 [pretrain]
    /// 通道做形状兼容张量的同名加载）。
    pub fn load_dinov2_weights(&mut self, path: &Path) -> AvResult<DinoLoadStats> {
        let sources = Tensor::read_safetensors(path).map_err(|e| {
            AvError::train(format!("读取 safetensors {} 失败: {e}", path.display()))
        })?;
        let mut stats = DinoLoadStats {
            sources_total: sources.len(),
            ..Default::default()
        };

        // register tokens 识别：registers 家族 checkpoint 含 [1, R, D] 的 register_tokens
        let register_rows = sources
            .iter()
            .filter(|(n, _)| n == "register_tokens" || n == "dinov2.register_tokens")
            .map(|(_, t)| t.size()[1])
            .next()
            .unwrap_or(0);
        stats.register_tokens_stripped = register_rows > 0;

        let mut applied: Vec<(String, Tensor)> = Vec::new();
        // HF 拆分式 QKV 暂存：0=q 1=k 2=v；8=bias 标志位用高 4 位编码
        let mut qkv_w: std::collections::HashMap<(usize, u8), Tensor> = Default::default();
        let mut qkv_b: std::collections::HashMap<(usize, u8), Tensor> = Default::default();

        for (src_name, src) in sources {
            match map_source_name(&src_name) {
                Mapped::Skipped(reason) => {
                    stats.skipped_unrecognized.push(format!("{src_name}（{reason}）"));
                }
                Mapped::Qkv { layer, part, bias } => {
                    let store = if bias { &mut qkv_b } else { &mut qkv_w };
                    store.insert((layer, part), src);
                }
                Mapped::Direct(name) => {
                    if name == "pos_embed" {
                        let adapted = self.adapt_pos_embed(&src, register_rows);
                        let n_src = src.size()[1] - 1 - register_rows;
                        stats.pos_embed_interpolated =
                            n_src != self.pos_embed.size()[1] - 1 || register_rows > 0;
                        applied.push((name, adapted));
                    } else {
                        applied.push((name, src));
                    }
                }
            }
        }

        // 融合拆分式 QKV：官方顺序 [q; k; v]（transformers / hub 布局一致）
        let mut layer_ids: Vec<usize> = qkv_w.keys().map(|(l, _)| *l).collect();
        layer_ids.extend(qkv_b.keys().map(|(l, _)| *l));
        layer_ids.sort_unstable();
        layer_ids.dedup();
        for layer in layer_ids {
            for (store, suffix) in [(&qkv_w, "weight"), (&qkv_b, "bias")] {
                let (Some(q), Some(k), Some(v)) = (
                    store.get(&(layer, 0)),
                    store.get(&(layer, 1)),
                    store.get(&(layer, 2)),
                ) else {
                    continue;
                };
                let fused = Tensor::cat(&[q, k, v], 0);
                applied.push((format!("blocks.{layer}.attn.qkv.{suffix}"), fused));
            }
        }

        stats.expected = applied.len();
        let mut mismatch: Vec<String> = Vec::new();
        tch::no_grad(|| {
            for (name, src) in &applied {
                if self.paste(name, src) {
                    stats.loaded += 1;
                } else {
                    mismatch.push(name.clone());
                }
            }
        });
        stats.skipped_shape_mismatch = mismatch;
        Ok(stats)
    }

    /// checkpoint pos_embed → 目标网格：剥离 cls 后的前 `register_rows` 个 register 行，
    /// 只对 patch 网格部分插值（或零拷贝）。
    fn adapt_pos_embed(&self, src: &Tensor, register_rows: i64) -> Tensor {
        let class_pos = src.i((.., ..1));
        let grid = src.i((.., 1 + register_rows..));
        let target_n = self.pos_embed.size()[1] - 1;
        let n_src = grid.size()[1];
        if n_src == target_n && register_rows == 0 {
            return grid.copy();
        }
        let dim = EMBED_DIM;
        let s = (n_src as f64).sqrt().round() as i64;
        let g = self.grid_side;
        let grid = grid
            .reshape([1, s, s, dim])
            .permute([0, 3, 1, 2])
            .upsample_bicubic2d([g, g], false, g as f64 / s as f64, g as f64 / s as f64)
            .permute([0, 2, 3, 1])
            .reshape([1, -1, dim]);
        Tensor::cat(&[&class_pos, &grid], 1)
    }

    /// 按本实现变量名把源张量写进对应字段（形状一致才拷贝，返回是否成功）。
    /// tch 0.24 字段约定：Conv/Linear 的 `ws: Tensor` + `bs: Option<Tensor>`，
    /// LayerNorm 的 `ws/bs` 均为 Option。
    fn paste(&mut self, name: &str, src: &Tensor) -> bool {
        fn copy_same(dst: &mut Tensor, src: &Tensor) -> bool {
            if dst.size() == src.size() {
                dst.copy_(src);
                true
            } else {
                false
            }
        }
        fn copy_opt(dst: Option<&mut Tensor>, src: &Tensor) -> bool {
            match dst {
                Some(t) => copy_same(t, src),
                None => false,
            }
        }

        match name {
            "patch_embed.proj.weight" => copy_same(&mut self.patch_embed.proj.ws, src),
            "patch_embed.proj.bias" => copy_opt(self.patch_embed.proj.bs.as_mut(), src),
            "cls_token" => copy_same(&mut self.cls_token, src),
            "pos_embed" => copy_same(&mut self.pos_embed, src),
            "norm.weight" => copy_opt(self.norm.ws.as_mut(), src),
            "norm.bias" => copy_opt(self.norm.bs.as_mut(), src),
            other => {
                let Some(rest) = other.strip_prefix("blocks.") else {
                    return false;
                };
                let Some((idx, sub)) = rest.split_once('.') else {
                    return false;
                };
                let Ok(i) = idx.parse::<usize>() else {
                    return false;
                };
                let Some(blk) = self.blocks.get_mut(i) else {
                    return false;
                };
                match sub {
                    "norm1.weight" => copy_opt(blk.norm1.ws.as_mut(), src),
                    "norm1.bias" => copy_opt(blk.norm1.bs.as_mut(), src),
                    "attn.qkv.weight" => copy_same(&mut blk.attn.qkv.ws, src),
                    "attn.qkv.bias" => copy_opt(blk.attn.qkv.bs.as_mut(), src),
                    "attn.proj.weight" => copy_same(&mut blk.attn.proj.ws, src),
                    "attn.proj.bias" => copy_opt(blk.attn.proj.bs.as_mut(), src),
                    "norm2.weight" => copy_opt(blk.norm2.ws.as_mut(), src),
                    "norm2.bias" => copy_opt(blk.norm2.bs.as_mut(), src),
                    "mlp.fc1.weight" => copy_same(&mut blk.mlp.fc1.ws, src),
                    "mlp.fc1.bias" => copy_opt(blk.mlp.fc1.bs.as_mut(), src),
                    "mlp.fc2.weight" => copy_same(&mut blk.mlp.fc2.ws, src),
                    "mlp.fc2.bias" => copy_opt(blk.mlp.fc2.bs.as_mut(), src),
                    "ls1.gamma" => copy_same(&mut blk.ls1, src),
                    "ls2.gamma" => copy_same(&mut blk.ls2, src),
                    _ => false,
                }
            }
        }
    }
}

/// checkpoint 源名 → 本实现规范名（相对名，无 "backbone." 前缀）的映射结果。
#[derive(Debug, PartialEq)]
enum Mapped {
    /// 一一对应的直接映射
    Direct(String),
    /// HF 拆分式 QKV 分量（导入时融合，见 [`DinoV2Backbone::load_dinov2_weights`]）
    Qkv { layer: usize, part: u8, bias: bool },
    /// 非骨干张量（附原因，进报告）
    Skipped(&'static str),
}

/// 支持的源命名（均可带可选的 `dinov2.` HF 前缀）：
/// - **HF transformers（facebook/dinov2-small 实测键名）**：
///   `embeddings.cls_token` / `embeddings.mask_token` /
///   `embeddings.patch_embeddings.projection.{weight,bias}` /
///   `embeddings.position_embeddings` /
///   `encoder.layer.{i}.attention.attention.{query,key,value}.{weight,bias}`（拆分 QKV）/
///   `encoder.layer.{i}.attention.output.dense.{weight,bias}` /
///   `encoder.layer.{i}.norm1|norm2.{weight,bias}` /
///   `encoder.layer.{i}.mlp.fc1|fc2.{weight,bias}` /
///   `encoder.layer.{i}.layer_scale1|2.lambda1` / `layernorm.{weight,bias}`
/// - **HF 旧版变体**：`position_embedding`、`attention.attention.qkv.*`（融合）、
///   `layernorm_before|after.*`、`layer_scale2.lambda2`
/// - **torch.hub 原生**：`blocks.{i}.norm1|attn.qkv|attn.proj|norm2|mlp.fc1|mlp.fc2|ls1|ls2.*`、
///   `patch_embed.proj.*`、`pos_embed`、`cls_token`、`norm.*`
fn map_source_name(src: &str) -> Mapped {
    let n = src.strip_prefix("dinov2.").unwrap_or(src);

    // 全局 token / embedding
    if n == "cls_token" || n == "embeddings.cls_token" {
        return Mapped::Direct("cls_token".into());
    }
    if matches!(n, "pos_embed" | "position_embedding" | "embeddings.position_embeddings") {
        return Mapped::Direct("pos_embed".into());
    }
    if n == "mask_token" || n == "embeddings.mask_token" {
        return Mapped::Skipped("mask_token（本实现无 masked forward）");
    }
    if n == "register_tokens" {
        return Mapped::Skipped("register_tokens（已识别并剥离 pos_embed 对应行）");
    }
    if n.starts_with("head.") {
        return Mapped::Skipped("任务头（分类头按任务重训）");
    }
    // patch embed
    if let Some(rest) = n.strip_prefix("embeddings.patch_embeddings.projection.") {
        return Mapped::Direct(format!("patch_embed.proj.{rest}"));
    }
    if n.starts_with("patch_embed.") {
        return Mapped::Direct(n.into());
    }
    // 最终 LayerNorm
    if n == "norm.weight" || n == "norm.bias" {
        return Mapped::Direct(n.into());
    }
    if n == "layernorm.weight" || n == "layernorm.bias" {
        return Mapped::Direct(format!("norm.{}", &n["layernorm.".len()..]));
    }
    // HF encoder.layer.{i}.{sub}
    if let Some(rest) = n.strip_prefix("encoder.layer.") {
        let Some((idx, sub)) = rest.split_once('.') else {
            return Mapped::Skipped("无法解析的 encoder.layer 层名");
        };
        let Ok(layer) = idx.parse::<usize>() else {
            return Mapped::Skipped("无法解析的层号");
        };
        let b = format!("blocks.{layer}.");
        // 拆分式 QKV（实测 HF 新版命名）
        for (part, name) in [(0u8, "query"), (1u8, "key"), (2u8, "value")] {
            if let Some(t) = sub.strip_prefix(&format!("attention.attention.{name}.")) {
                match t {
                    "weight" => return Mapped::Qkv { layer, part, bias: false },
                    "bias" => return Mapped::Qkv { layer, part, bias: true },
                    _ => return Mapped::Skipped("无法识别的 QKV 子张量"),
                }
            }
        }
        let mapped = if let Some(t) = sub.strip_prefix("attention.attention.qkv.") {
            format!("{b}attn.qkv.{t}")
        } else if let Some(t) = sub.strip_prefix("attention.attention.proj.") {
            // 防御：个别版本把 proj 放 attention.attention 下
            format!("{b}attn.proj.{t}")
        } else if let Some(t) = sub.strip_prefix("attention.output.dense.") {
            format!("{b}attn.proj.{t}")
        } else if let Some(t) = sub.strip_prefix("layernorm_before.") {
            format!("{b}norm1.{t}")
        } else if let Some(t) = sub.strip_prefix("norm1.") {
            format!("{b}norm1.{t}")
        } else if let Some(t) = sub.strip_prefix("layernorm_after.") {
            format!("{b}norm2.{t}")
        } else if let Some(t) = sub.strip_prefix("norm2.") {
            format!("{b}norm2.{t}")
        } else if let Some(t) = sub.strip_prefix("mlp.fc1.") {
            format!("{b}mlp.fc1.{t}")
        } else if let Some(t) = sub.strip_prefix("mlp.fc2.") {
            format!("{b}mlp.fc2.{t}")
        } else if sub == "layer_scale1.lambda1" || sub == "layer_scale1.lambda2" {
            format!("{b}ls1.gamma")
        } else if sub == "layer_scale2.lambda1" || sub == "layer_scale2.lambda2" {
            format!("{b}ls2.gamma")
        } else {
            return Mapped::Skipped("无法识别的 encoder.layer 子层");
        };
        return Mapped::Direct(mapped);
    }
    // torch.hub 原生命名直通
    if n.starts_with("blocks.") {
        return Mapped::Direct(n.into());
    }
    Mapped::Skipped("无法识别的名字")
}

// ---------------------------------------------------------------------------
// 导入统计
// ---------------------------------------------------------------------------

/// 权重导入完整统计（部分加载合法，但必须完整可见——与 AdaptReport 同哲学）。
#[derive(Debug, Default, Clone)]
pub struct DinoLoadStats {
    /// checkpoint 总张量数
    pub sources_total: usize,
    /// 成功写入的目标骨干张量数
    pub loaded: usize,
    /// checkpoint 提供的目标骨干张量数（QKV 融合后计；分母）
    pub expected: usize,
    /// pos_embed 是否经过网格插值/register 剥离
    pub pos_embed_interpolated: bool,
    /// checkpoint 是否含 register tokens（已剥离）
    pub register_tokens_stripped: bool,
    /// 映射命中但形状不符而跳过的名字
    pub skipped_shape_mismatch: Vec<String>,
    /// 非骨干/无法识别的张量（mask_token / register_tokens / head 等，附原因）
    pub skipped_unrecognized: Vec<String>,
}

impl DinoLoadStats {
    /// 导入比例 = loaded / expected（骨干目标张量的覆盖率；QKV 融合后按目标计）。
    pub fn load_ratio(&self) -> f32 {
        if self.expected == 0 {
            0.0
        } else {
            self.loaded as f32 / self.expected as f32
        }
    }

    pub fn summary(&self) -> String {
        format!(
            "loaded={}/{}（ratio {:.3}，源 {} 个张量）pos_embed 插值={} register 剥离={} \
             形状不符={} 非骨干跳过={}",
            self.loaded,
            self.expected,
            self.load_ratio(),
            self.sources_total,
            self.pos_embed_interpolated,
            self.register_tokens_stripped,
            self.skipped_shape_mismatch.len(),
            self.skipped_unrecognized.len(),
        )
    }
}

// ---------------------------------------------------------------------------
// BaseBackbone：分类走 cls token，检测金字塔走 ViTDet 式三分支
// ---------------------------------------------------------------------------

impl BaseBackbone for DinoV2Backbone {
    fn forward_features(&self, x: &Tensor) -> AvResult<FeaturePyramid> {
        let (_b, _c, h, w) = x.size4().unwrap();
        let tokens = self.encode_tokens(x);
        let grid = self.patch_grid(&tokens); // [B, D, H', W']
        let (h8, w8) = (h / 8, w / 8);
        let (h16, w16) = (h / 16, w / 16);
        let (h32, w32) = (h / 32, w / 32);
        // ViTDet 式简单金字塔：上/下采样到目标 stride 尺寸 + 3×3 conv 对齐通道
        let s8 = grid
            .upsample_bicubic2d([h8, w8], false, None, None)
            .apply(&self.conv_s8);
        let s16 = grid
            .upsample_bicubic2d([h16, w16], false, None, None)
            .apply(&self.conv_s16);
        let s32 = grid.adaptive_avg_pool2d([h32, w32]).apply(&self.conv_s32);
        let mut pyramid = FeaturePyramid::default();
        pyramid.levels.push(FeatureMap::new(s8, 8)?);
        pyramid.levels.push(FeatureMap::new(s16, 16)?);
        pyramid.levels.push(FeatureMap::new(s32, 32)?);
        pyramid.validate_ascending()?;
        Ok(pyramid)
    }

    /// 分类池化特征：最终 LayerNorm 后的 cls token [B, 384]。
    fn forward_pooled(&self, x: &Tensor) -> AvResult<Tensor> {
        let tokens = self.encode_tokens(x);
        Ok(tokens.i((.., 0)))
    }

    fn spec(&self) -> BackboneSpec {
        BackboneSpec {
            levels: vec![
                LevelSpec {
                    stride: 8,
                    channels: PYRAMID_CHANNELS as usize,
                },
                LevelSpec {
                    stride: 16,
                    channels: PYRAMID_CHANNELS as usize,
                },
                LevelSpec {
                    stride: 32,
                    channels: PYRAMID_CHANNELS as usize,
                },
            ],
        }
    }
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "torch"))]
mod tests {
    use super::*;

    fn backbone_at(img: u32) -> DinoV2Backbone {
        let vs = nn::VarStore::new(tch::Device::Cpu);
        let cfg = BackboneCfg::default();
        DinoV2Backbone::new(&vs.root(), &cfg, img).unwrap()
    }

    fn no_nan(t: &Tensor) -> bool {
        t.isnan().sum(Kind::Float).double_value(&[]) == 0.0
    }

    #[test]
    fn constructor_rejects_non_multiple_sizes() {
        let vs = nn::VarStore::new(tch::Device::Cpu);
        let cfg = BackboneCfg::default();
        // 非 14 倍数
        assert!(DinoV2Backbone::new(&vs.root(), &cfg, 224 + 14 + 3).is_err());
        // 14 倍数但非 32 倍数（金字塔不整除）
        assert!(DinoV2Backbone::new(&vs.root(), &cfg, 154).is_err());
    }

    #[test]
    fn pyramid_and_pooled_contract_224() {
        let backbone = backbone_at(224);
        let x = Tensor::randn([2, 3, 224, 224], (tch::Kind::Float, tch::Device::Cpu));
        let pyramid = backbone.forward_features(&x).unwrap();
        let strides: Vec<u32> = pyramid.levels.iter().map(|l| l.stride).collect();
        assert_eq!(strides, vec![8, 16, 32]);
        let sizes: Vec<Vec<i64>> = pyramid.levels.iter().map(|l| l.tensor.size()).collect();
        assert_eq!(sizes[0], vec![2, 256, 28, 28]);
        assert_eq!(sizes[1], vec![2, 256, 14, 14]);
        assert_eq!(sizes[2], vec![2, 256, 7, 7]);
        let pooled = backbone.forward_pooled(&x).unwrap();
        assert_eq!(pooled.size(), vec![2, 384]);
        assert!(no_nan(&pooled));
        assert_eq!(backbone.spec().levels.len(), 3);
    }

    #[test]
    fn pos_embed_interpolation_changes_grid_and_keeps_cls() {
        let vs = nn::VarStore::new(tch::Device::Cpu);
        let cfg = BackboneCfg::default();
        let mut backbone = DinoV2Backbone::new(&vs.root(), &cfg, 224).unwrap(); // 16×16 网格
        // 构造默认 pos_embed 零初始化（零插值仍是零，测不出变化），先随机化
        let rand = Tensor::randn(
            backbone.pos_embed.size(),
            (tch::Kind::Float, tch::Device::Cpu),
        );
        tch::no_grad(|| backbone.pos_embed.copy_(&rand));

        // 1) 目标网格与训练网格一致 → 零拷贝返回，逐位一致
        let npatch_224 = 16 * 16;
        let same = backbone.interpolate_pos_encoding(npatch_224, 224, 224);
        assert_eq!(
            tensor_max_diff(&same, &backbone.pos_embed),
            0.0,
            "同网格应零拷贝逐位一致"
        );

        // 2) 448 输入（32×32 网格）→ 插值：cls 行不变，网格行变为插值结果
        let npatch_448 = 32 * 32;
        let interp = backbone.interpolate_pos_encoding(npatch_448, 448, 448);
        assert_eq!(interp.size(), vec![1, 1 + npatch_448, EMBED_DIM]);
        let cls_diff = tensor_max_diff(&backbone.pos_embed.i((.., ..1)), &interp.i((.., ..1)));
        assert!(cls_diff < 1e-6, "cls 行不应被插值改动，diff={cls_diff}");
        // 双三次插值是源网格的函数：输出应有限且非全零（源随机非零）
        let grid_after = interp.i((.., 1..));
        assert!(no_nan(&grid_after));
        assert!(
            grid_after.abs().max().double_value(&[]) > 1e-3,
            "随机源的插值网格不应全零"
        );
    }

    fn tensor_max_diff(a: &Tensor, b: &Tensor) -> f64 {
        (a - b).abs().max().double_value(&[])
    }

    #[test]
    fn source_name_maps_hf_and_hub() {
        // HF transformers 命名（facebook/dinov2-small 实测键名）
        match map_source_name("encoder.layer.3.attention.attention.query.weight") {
            Mapped::Qkv { layer, part, bias } => {
                assert_eq!((layer, part, bias), (3, 0, false));
            }
            other => panic!("应为 Qkv 分量，得到 {other:?}"),
        }
        match map_source_name("encoder.layer.3.attention.attention.value.bias") {
            Mapped::Qkv { layer, part, bias } => {
                assert_eq!((layer, part, bias), (3, 2, true));
            }
            other => panic!("应为 Qkv 分量，得到 {other:?}"),
        }
        match map_source_name("dinov2.encoder.layer.0.attention.attention.key.bias") {
            Mapped::Qkv { layer, part, bias } => {
                assert_eq!((layer, part, bias), (0, 1, true));
            }
            other => panic!("应为 Qkv 分量，得到 {other:?}"),
        }
        assert_eq!(
            map_source_name("encoder.layer.0.attention.output.dense.bias"),
            Mapped::Direct("blocks.0.attn.proj.bias".into())
        );
        assert_eq!(
            map_source_name("encoder.layer.11.norm1.weight"),
            Mapped::Direct("blocks.11.norm1.weight".into())
        );
        assert_eq!(
            map_source_name("encoder.layer.11.layer_scale2.lambda1"),
            Mapped::Direct("blocks.11.ls2.gamma".into())
        );
        assert_eq!(
            map_source_name("embeddings.patch_embeddings.projection.weight"),
            Mapped::Direct("patch_embed.proj.weight".into())
        );
        assert_eq!(
            map_source_name("embeddings.position_embeddings"),
            Mapped::Direct("pos_embed".into())
        );
        assert_eq!(
            map_source_name("embeddings.cls_token"),
            Mapped::Direct("cls_token".into())
        );
        assert_eq!(
            map_source_name("layernorm.bias"),
            Mapped::Direct("norm.bias".into())
        );
        // HF 旧版变体
        assert_eq!(
            map_source_name("dinov2.encoder.layer.0.attention.attention.qkv.weight"),
            Mapped::Direct("blocks.0.attn.qkv.weight".into())
        );
        assert_eq!(
            map_source_name("dinov2.encoder.layer.0.layernorm_before.weight"),
            Mapped::Direct("blocks.0.norm1.weight".into())
        );
        assert_eq!(
            map_source_name("dinov2.position_embedding"),
            Mapped::Direct("pos_embed".into())
        );
        // torch.hub 原生命名直通
        assert_eq!(
            map_source_name("blocks.5.attn.qkv.weight"),
            Mapped::Direct("blocks.5.attn.qkv.weight".into())
        );
        assert_eq!(
            map_source_name("blocks.5.ls1.gamma"),
            Mapped::Direct("blocks.5.ls1.gamma".into())
        );
        assert_eq!(
            map_source_name("patch_embed.proj.bias"),
            Mapped::Direct("patch_embed.proj.bias".into())
        );
        // 非骨干张量跳过（附原因）
        assert!(matches!(
            map_source_name("embeddings.mask_token"),
            Mapped::Skipped(_)
        ));
        assert!(matches!(map_source_name("register_tokens"), Mapped::Skipped(_)));
        assert!(matches!(map_source_name("head.fc.weight"), Mapped::Skipped(_)));
    }

    /// 用一个极小合成 safetensors 走一遍融合 QKV 与形状不符跳过路径
    /// （不依赖官方权重文件；官方权重的全量导入验证见 av-runtime 集成测试）。
    #[test]
    fn load_fuses_split_qkv_and_reports_stats() {
        let dim = EMBED_DIM;
        let vs = nn::VarStore::new(tch::Device::Cpu);
        let cfg = BackboneCfg::default();
        let mut backbone = DinoV2Backbone::new(&vs.root(), &cfg, 224).unwrap();

        // 构造 layer 0 的源张量（q/k/v 拆分 + 一个形状不符项）
        let mk = |v: f64, shape: &[i64]| {
            Tensor::ones(shape, (tch::Kind::Float, tch::Device::Cpu)) * v
        };
        let dir =
            std::env::temp_dir().join(format!("av-dino-test-{}.safetensors", std::process::id()));
        let entries: Vec<(&str, Tensor)> = vec![
            (
                "encoder.layer.0.attention.attention.query.weight",
                mk(1.0, &[dim, dim]),
            ),
            ("encoder.layer.0.attention.attention.key.weight", mk(2.0, &[dim, dim])),
            ("encoder.layer.0.attention.attention.value.weight", mk(3.0, &[dim, dim])),
            ("encoder.layer.0.attention.attention.query.bias", mk(4.0, &[dim])),
            ("encoder.layer.0.attention.attention.key.bias", mk(5.0, &[dim])),
            ("encoder.layer.0.attention.attention.value.bias", mk(6.0, &[dim])),
            // 形状不符（正确为 [4*dim, dim]）→ 应被跳过并记录
            ("encoder.layer.1.mlp.fc1.weight", mk(7.0, &[dim, dim])),
        ];
        Tensor::write_safetensors(&entries, &dir).unwrap();

        let stats = backbone.load_dinov2_weights(&dir).unwrap();
        // 融合后的 qkv.weight [3*dim, dim]：第 0 段应全 1、第 1 段全 2、第 2 段全 3
        let fused = backbone.blocks[0].attn.qkv.ws.copy();
        for (seg, v) in [(0i64, 1.0), (1, 2.0), (2, 3.0)] {
            let part = fused.narrow(0, seg * dim, dim);
            assert!(
                (part.max().double_value(&[]) - v).abs() < 1e-6
                    && (part.min().double_value(&[]) - v).abs() < 1e-6,
                "融合段 {seg} 应全为 {v}"
            );
        }
        let fused_b = backbone.blocks[0].attn.qkv.bs.as_ref().unwrap().copy();
        assert!((fused_b.double_value(&[0]) - 4.0).abs() < 1e-6);
        assert!((fused_b.double_value(&[dim]) - 5.0).abs() < 1e-6);
        assert_eq!(stats.loaded, 2); // qkv.weight + qkv.bias（fc1 形状不符跳过）
        assert_eq!(stats.expected, 3);
        assert_eq!(stats.skipped_shape_mismatch, vec!["blocks.1.mlp.fc1.weight"]);
        assert_eq!(stats.sources_total, entries.len() as usize);
        assert!(stats.load_ratio() < 0.9);
        let _ = std::fs::remove_file(&dir);
    }
}
