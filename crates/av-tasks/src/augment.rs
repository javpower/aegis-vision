//! 数据增强原语（数据增强官任务 §1）：纯 Rust、无 tch / image 依赖，
//! 「样本级变换函数 + 增强配置」的纯函数风格，全部可手算单测。
//!
//! 坐标同步约定（与加载器同一映射链）：所有几何变换都作用在**原始图像像素空间**
//! （letterbox 之前），像素与坐标按同一份 [`AugmentPlan`] 走
//! `flip → scale → letterbox/拉伸` 的线性映射链——翻转用 `x' = w − x`、缩放用
//! `× s`（图像同倍率 resize 由调用方完成），letterbox 编码与坐标映射复用
//! av-runtime::dataset 既有函数，因此坐标约定与不增强时逐位一致。
//!
//! HSV 简化说明（任务书授权）：不做完整 RGB→HSV→RGB 空间转换，
//! [`AugmentCfg::hsv`] 三分量直接作为 R/G/B 通道的乘性抖动幅度
//! （增益 ∈ [1−g, 1+g]）——通道比例抖动近似覆盖色调/饱和度、全通道幅度近似
//! 亮度，视觉效果上是颜色/明暗抖动，对增强目的（颜色不变性）足够，
//! 计算量为每像素 3 次乘法。

use av_core::config::AugmentCfg;

use crate::rng::XorShift;

/// COCO 17 关键点水平翻转交换表：翻转后 `new[i] = old[COCO17_FLIP_SWAP[i]]`
/// （0 鼻不动；1↔2 左右眼、3↔4 左右耳、5↔6 肩、7↔8 肘、9↔10 腕、11↔12 髋、
/// 13↔14 膝、15↔16 踝）。**仅当关键点模板恰为 17 点**时使用——其它点数模板
/// 无内置语义，翻转让坐标镜像但索引不交换（否则会静默错位）。
pub const COCO17_FLIP_SWAP: [usize; 17] =
    [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15];

/// 单样本增强方案：由 [`draw_plan`] 按配置一次性抽好（抛硬币翻转、缩放系数、
/// 每通道增益），同一份 plan 同时作用于像素与坐标——坐标同步的前提是
/// 「一次采样、处处使用」。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AugmentPlan {
    /// 是否水平翻转。
    pub flip: bool,
    /// 缩放系数（1.0 = 不缩放）。
    pub scale: f32,
    /// R/G/B 通道乘性增益（[1.0; 3] = 不抖动）。
    pub rgb_gains: [f32; 3],
}

impl AugmentPlan {
    /// 恒等方案（等价于不增强；`encode_*` 走同一条编码路径，输出与
    /// 既有预解码加载器逐位一致，见 dataset 单测）。
    pub fn none() -> Self {
        Self {
            flip: false,
            scale: 1.0,
            rgb_gains: [1.0; 3],
        }
    }

    pub fn is_none(&self) -> bool {
        !self.flip && self.scale == 1.0 && self.rgb_gains == [1.0; 3]
    }
}

/// 配置是否含任何实际增强强度（引擎据此决定走「逐 epoch 随机编码」还是
/// 既有「整集预解码」快速路径；全零配置 = 行为与历史版本完全一致）。
pub fn has_strength(cfg: &AugmentCfg) -> bool {
    cfg.mosaic > 0.0
        || cfg.mixup > 0.0
        || cfg.flip > 0.0
        || cfg.hsv.iter().any(|&g| g > 0.0)
        || cfg.scale_jitter.is_some()
}

/// `close_last_epochs` 语义：训练最后 `close_last_epochs` 个 epoch 关强增强
/// （让模型在干净分布上收尾），即 `epoch <= total − N` 时返回 true；
/// N = 0 恒开；N ≥ total 从第 1 个 epoch 就关（saturating 下溢安全）。
pub fn strong_aug_active(cfg: &AugmentCfg, epoch: u32, total_epochs: u32) -> bool {
    cfg.close_last_epochs == 0 || epoch <= total_epochs.saturating_sub(cfg.close_last_epochs)
}

/// 按配置抽一份样本级 plan。随机数消耗顺序固定：flip 抛硬币 → 缩放系数 →
/// R/G/B 增益（同配置同 RNG 状态 ⇒ 同 plan，保证实验可复现）。
pub fn draw_plan(cfg: &AugmentCfg, rng: &mut XorShift) -> AugmentPlan {
    AugmentPlan {
        flip: cfg.flip > 0.0 && rng.next_f32() < cfg.flip,
        scale: match cfg.scale_jitter {
            Some([lo, hi]) => rng.next_range(lo, hi),
            None => 1.0,
        },
        rgb_gains: [
            1.0 + rng.next_range(-cfg.hsv[0], cfg.hsv[0]),
            1.0 + rng.next_range(-cfg.hsv[1], cfg.hsv[1]),
            1.0 + rng.next_range(-cfg.hsv[2], cfg.hsv[2]),
        ],
    }
}

// ---------------------------------------------------------------------------
// 像素域（RGB8 交错缓冲：w * h * 3 字节）
// ---------------------------------------------------------------------------

/// RGB8 缓冲水平翻转（原地）：逐行镜像 3 字节像素组。
pub fn hflip_rgb(w: usize, h: usize, rgb: &mut [u8]) {
    for y in 0..h {
        let row = &mut rgb[y * w * 3..(y + 1) * w * 3];
        for x in 0..w / 2 {
            let (a, b) = (x * 3, (w - 1 - x) * 3);
            row.swap(a, b);
            row.swap(a + 1, b + 1);
            row.swap(a + 2, b + 2);
        }
    }
}

/// RGB8 缓冲通道乘性增益（HSV 简化版，原地）：`v = clamp(round(v · gain), 0, 255)`。
/// gain 全 1 时逐位恒等（v·1.0 精确、round 回同值）。
///
/// 每调用先建 3×256 查找表（768 次乘加），逐像素只查表不再做 f32 运算——
/// 表项与逐字节计算同公式逐位一致，增强热路径上每像素省 3 次浮点乘加。
pub fn mul_rgb(rgb: &mut [u8], gains: [f32; 3]) {
    if gains == [1.0, 1.0, 1.0] {
        return; // 恒等增益直接返回（文档承诺的逐位恒等）
    }
    let mut lut = [[0u8; 256]; 3];
    for (c, gc) in gains.iter().enumerate() {
        for (v, slot) in lut[c].iter_mut().enumerate() {
            *slot = (v as f32 * gc).round().clamp(0.0, 255.0) as u8;
        }
    }
    for px in rgb.chunks_exact_mut(3) {
        px[0] = lut[0][px[0] as usize];
        px[1] = lut[1][px[1] as usize];
        px[2] = lut[2][px[2] as usize];
    }
}

// ---------------------------------------------------------------------------
// 坐标域（原始图像像素空间，f32 连续坐标）
// ---------------------------------------------------------------------------

/// 单点水平镜像：`x' = w − x`（框 / 关键点 / 多边形角点通用）。
pub fn flip_x(x: f32, w: f32) -> f32 {
    w - x
}

/// xyxy 框水平镜像（x1<x2 自动保持）。
pub fn flip_box_xyxy(b: [f32; 4], w: f32) -> [f32; 4] {
    [w - b[2], b[1], w - b[0], b[3]]
}

/// cxcywh 框水平镜像（宽高不变，仅中心 x 镜像）。
pub fn flip_box_cxcywh(b: [f32; 4], w: f32) -> [f32; 4] {
    [w - b[0], b[1], b[2], b[3]]
}

/// 关键点实例集水平镜像 + COCO 17 交换：每个 [x, y, v] 三元组的 x 镜像
/// （v 标志随点走）；恰 17 点的实例整体按 [`COCO17_FLIP_SWAP`] 交换索引
/// （v 与坐标一同换位，左右对称点语义保持正确）。
pub fn flip_keypoints(kpts: &mut [Vec<[f32; 3]>], w: f32) {
    for g in kpts.iter_mut() {
        for p in g.iter_mut() {
            p[0] = w - p[0];
        }
    }
    swap_coco17_keypoints(kpts);
}

/// 仅交换 COCO 17 关键点索引（不碰坐标）——供调用方已自行完成坐标镜像、
/// 只需左右语义换位的场景（如 encode 期先镜像后换位的两步式写法）。
pub fn swap_coco17_keypoints(kpts: &mut [Vec<[f32; 3]>]) {
    for g in kpts.iter_mut() {
        if g.len() == 17 {
            let old = g.clone();
            for (i, &src) in COCO17_FLIP_SWAP.iter().enumerate() {
                g[i] = old[src];
            }
        }
    }
}

/// 多边形点集水平镜像（点序不变——简单多边形镜像后仍简单，偶奇栅格化无歧义）。
pub fn flip_polygon(poly: &mut [[f32; 2]], w: f32) {
    for p in poly.iter_mut() {
        p[0] = w - p[0];
    }
}

/// 点坐标缩放（图像由调用方同倍率 resize；坐标 × s，原点为图左上角）。
pub fn scale_point(p: &mut [f32; 2], s: f32) {
    p[0] *= s;
    p[1] *= s;
}

/// 缩放后的图像尺寸（与坐标缩放同一取整规则：`(w·s).round()`，调用方 resize
/// 与坐标使用同一 (aw, ah)，避免尺寸/坐标脱节）。
pub fn scaled_dims(w: u32, h: u32, s: f32) -> (u32, u32) {
    (
        ((w as f32 * s).round() as u32).max(1),
        ((h as f32 * s).round() as u32).max(1),
    )
}

// ---------------------------------------------------------------------------
// Mosaic（数据增强官二波 §1）：4 图拼 1（2×2 网格，raw 像素域）
// ---------------------------------------------------------------------------

/// mosaic 画布尺寸：2×2 网格，象限 = 基准图（锚点样本）尺寸 `(w, h)`。
pub fn mosaic_canvas_dims(w: u32, h: u32) -> (u32, u32) {
    (w * 2, h * 2)
}

/// 把恰为 `qw × qh` 的 RGB8 源整块拷进画布以 `(x0, y0)` 为左上角的象限
/// （行式 memcpy，无插值——像素缩放由调用方在拷贝前完成）。
pub fn mosaic_paste_quadrant(
    canvas: &mut [u8],
    canvas_w: usize,
    src: &[u8],
    qw: usize,
    qh: usize,
    x0: usize,
    y0: usize,
) {
    debug_assert_eq!(src.len(), qw * qh * 3, "源缓冲必须是象限尺寸");
    for ry in 0..qh {
        let dst = ((y0 + ry) * canvas_w + x0) * 3;
        let src_off = ry * qw * 3;
        canvas[dst..dst + qw * 3].copy_from_slice(&src[src_off..src_off + qw * 3]);
    }
}

/// mosaic 单框换算：源图 `(sw, sh)` 的 xyxy 框随像素 stretch（`sw,sh → qw,qh`）
/// 映射到以 `(x0, y0)` 为左上角的象限（与像素 resize 同一线性映射），再按象限
/// 四边（= 拼接线 / 画布边）**裁剪**——越界部分直接截断、不做越界还原（YOLO
/// 惯例）；裁剪后宽或高 ≤ 0 视为退化目标，返回 `None`（整框丢弃）。
pub fn mosaic_map_box(
    b: [f32; 4],
    sw: f32,
    sh: f32,
    qw: f32,
    qh: f32,
    x0: f32,
    y0: f32,
) -> Option<[f32; 4]> {
    let (sx, sy) = (qw / sw, qh / sh);
    let x1 = (b[0] * sx + x0).clamp(x0, x0 + qw);
    let y1 = (b[1] * sy + y0).clamp(y0, y0 + qh);
    let x2 = (b[2] * sx + x0).clamp(x0, x0 + qw);
    let y2 = (b[3] * sy + y0).clamp(y0, y0 + qh);
    if x2 - x1 <= 0.0 || y2 - y1 <= 0.0 {
        None
    } else {
        Some([x1, y1, x2, y2])
    }
}

/// mosaic 单来源项：像素已 stretch 到象限尺寸 `(qw, qh)` 的 RGB8 + **原始尺寸**
/// （框所在空间，与像素 resize 前一致）+ 原图像素 xyxy 框与类别。
///
/// 恒等约定：源尺寸恰为 `(qw, qh)` 时像素必须原样传入（不做 resize），保证
/// 「像素与框走同一映射」逐位成立。
pub struct MosaicItem<'a> {
    pub rgb: &'a [u8],
    pub src_w: u32,
    pub src_h: u32,
    pub boxes: &'a [[f32; 4]],
    pub labels: &'a [u32],
}

/// 4 图拼 1（2×2 网格，纯函数）：象限顺序 0=左上、1=右上、2=左下、3=右下；
/// 输出画布 `(2qw × 2qh)` RGB8 + 合并 xyxy 框（画布像素域、已按拼接线裁剪、
/// 退化目标丢弃）+ 合并类别（与框一一对应）。本实现取**固定等分网格**
/// （对象完全落在所属象限内，不跨拼接线）；随机中心点变体属后续工作。
pub fn mosaic_compose(
    qw: u32,
    qh: u32,
    items: &[MosaicItem<'_>; 4],
) -> (Vec<u8>, Vec<[f32; 4]>, Vec<u32>) {
    let (qw, qh) = (qw as usize, qh as usize);
    let (cw, ch) = (qw * 2, qh * 2);
    let mut canvas = vec![0u8; cw * ch * 3];
    let origins = [(0usize, 0usize), (qw, 0), (0, qh), (qw, qh)];
    let (mut boxes, mut labels) = (Vec::new(), Vec::new());
    for (k, item) in items.iter().enumerate() {
        let (x0, y0) = origins[k];
        mosaic_paste_quadrant(&mut canvas, cw, item.rgb, qw, qh, x0, y0);
        for (b, &l) in item.boxes.iter().zip(item.labels) {
            if let Some(mb) = mosaic_map_box(
                *b,
                item.src_w as f32,
                item.src_h as f32,
                qw as f32,
                qh as f32,
                x0 as f32,
                y0 as f32,
            ) {
                boxes.push(mb);
                labels.push(l);
            }
        }
    }
    (canvas, boxes, labels)
}

// ---------------------------------------------------------------------------
// Mixup（数据增强官二波 §2）：检测/分类双样本加权融合（关键点任务不适用：
// 两套人体拓扑叠加后关键点语义无解，引擎只在检测路径接入——见 dataset::mixup_raw）
// ---------------------------------------------------------------------------

/// mixup 融合系数 λ 的对称 Beta 先验强度 α（默认 0.2，U 形分布 → 多数样本
/// 接近「以一张图为主」）。AugmentCfg 未预留 α 字段（av-core 冻结期），
/// 以常量暴露，后续配置化只需改此处。
pub const MIXUP_BETA_ALPHA: f32 = 0.2;

/// 对称 Beta(α, α) 随机数（mixup 融合系数 λ）：两个独立 Gamma(α, 1) 的归一化
/// 比例，Gamma 形状采样用 Marsaglia-Tsang（α < 1 时 boost 技巧
/// `G(α) = G(α+1)·U^(1/α)`，正态分量走 Box-Muller）。α ≤ 0 恒返 0.5
/// （退化为 50/50 融合）。同 RNG 状态同序列（确定性）。
pub fn beta_symmetric(rng: &mut XorShift, alpha: f32) -> f32 {
    if alpha <= 0.0 {
        return 0.5;
    }
    let g1 = gamma_one(rng, alpha);
    let g2 = gamma_one(rng, alpha);
    let s = g1 + g2;
    if s <= 0.0 {
        0.5
    } else {
        // f32 极小 Gamma 可能下溢出 0/1 端点，夹回开区间（对增强语义无影响）
        (g1 / s).clamp(f32::EPSILON, 1.0 - f32::EPSILON)
    }
}

/// 单个 Gamma(α, 1) 样本（Marsaglia & Tsang 2000 squeeze 检验）。
fn gamma_one(rng: &mut XorShift, shape: f32) -> f32 {
    let (boost, a) = if shape < 1.0 {
        let u = rng.next_f32().max(1e-12);
        (u.powf(1.0 / shape), shape + 1.0)
    } else {
        (1.0, shape)
    };
    let d = a - 1.0 / 3.0;
    let c = (1.0 / (9.0 * d)).sqrt();
    loop {
        let u1 = rng.next_f32().max(1e-12);
        let u2 = rng.next_f32();
        // Box-Muller 标准正态
        let z = (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos();
        let v = (1.0 + c * z).powi(3);
        if v <= 0.0 {
            continue;
        }
        let u = rng.next_f32();
        let z2 = z * z;
        if u < 1.0 - 0.0331 * z2 * z2 || u.ln() < 0.5 * z2 + d * (1.0 - v + v.ln()) {
            return boost * d * v;
        }
    }
}

/// mixup 像素融合：`out = round(a·λ + b·(1−λ))`（逐字节）。两缓冲必须等长
/// （不同尺寸的图由调用方先 stretch 到一致）。λ=1 逐位等于 a，λ=0 逐位等于 b。
pub fn mixup_rgb(a: &[u8], b: &[u8], lam: f32) -> Vec<u8> {
    debug_assert_eq!(a.len(), b.len(), "mixup 两缓冲必须等长");
    let w = 1.0 - lam;
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as f32 * lam + y as f32 * w).round().clamp(0.0, 255.0) as u8)
        .collect()
}

// ---------------------------------------------------------------------------
// 组合增强抽样：mosaic → mixup →（draw_plan 的）flip → hsv → scale 串联
// ---------------------------------------------------------------------------

/// 组合增强（mosaic / mixup）一次抽样结果：两枚硬币 + mixup 融合系数。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompositeDraw {
    pub mosaic: bool,
    pub mixup: bool,
    pub mixup_lam: f32,
}

impl CompositeDraw {
    /// 全关（等价于不做组合增强）。
    pub fn none() -> Self {
        Self {
            mosaic: false,
            mixup: false,
            mixup_lam: 0.5,
        }
    }
}

/// 按配置抽组合增强。随机消耗顺序固定：mosaic 硬币 → mixup 硬币 → λ
/// （同配置同 RNG 状态 ⇒ 同结果，实验可复现）。关键不变量：**概率全 0 时
/// 零消耗 RNG**——后续 [`draw_plan`] 的随机流与未接入 mosaic/mixup 的历史
/// 版本逐位一致（「概率=0 行为不变」由单测锁定）。
pub fn draw_composite(cfg: &AugmentCfg, rng: &mut XorShift) -> CompositeDraw {
    let mosaic = cfg.mosaic > 0.0 && rng.next_f32() < cfg.mosaic;
    let mixup = cfg.mixup > 0.0 && rng.next_f32() < cfg.mixup;
    let mixup_lam = if mixup {
        beta_symmetric(rng, MIXUP_BETA_ALPHA)
    } else {
        0.5
    };
    CompositeDraw {
        mosaic,
        mixup,
        mixup_lam,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use av_core::config::AugmentCfg;

    // ----- 坐标翻转（手算） -----

    #[test]
    fn flip_box_xyxy_hand_computed() {
        // w=100：box (10,20)-(30,40) → (100-30, 20)-(100-10, 40) = (70,20)-(90,40)
        let b = flip_box_xyxy([10.0, 20.0, 30.0, 40.0], 100.0);
        assert_eq!(b, [70.0, 20.0, 90.0, 40.0]);
        // 触边框不越界；两次翻转回到原位
        let b2 = flip_box_xyxy([0.0, 0.0, 50.5, 8.0], 50.5);
        assert_eq!(b2, [0.0, 0.0, 50.5, 8.0]);
        let round = flip_box_xyxy(flip_box_xyxy([3.25, 4.0, 17.75, 9.5], 32.0), 32.0);
        assert_eq!(round, [3.25, 4.0, 17.75, 9.5]);
    }

    #[test]
    fn flip_box_cxcywh_hand_computed() {
        // w=64：中心 24 → 40，宽高不变
        let b = flip_box_cxcywh([24.0, 16.0, 8.0, 4.0], 64.0);
        assert_eq!(b, [40.0, 16.0, 8.0, 4.0]);
    }

    /// COCO 17 交换表语义：镜像 + 换位后，新的左眼 = 旧右眼镜像。
    #[test]
    fn flip_keypoints_swaps_coco17_pairs() {
        // 3 个代表点验证：0 鼻不动、1(左眼)↔2(右眼)、16(左踝)↔15(右踝)
        let mut g: Vec<[f32; 3]> = (0..17)
            .map(|i| {
                [
                    10.0 + i as f32,
                    20.0 + i as f32,
                    if i == 3 { 0.0 } else { 2.0 },
                ]
            })
            .collect();
        g[1] = [10.0, 21.0, 2.0]; // 左眼 (10,21) v=2
        g[2] = [30.0, 22.0, 2.0]; // 右眼 (30,22) v=2
        g[15] = [50.0, 35.0, 1.0]; // 右踝 (50,35) v=1
        g[16] = [60.0, 36.0, 0.0]; // 左踝 (60,36) v=0

        let mut insts = vec![g];
        flip_keypoints(&mut insts, 100.0);
        let g = &insts[0];
        // 鼻子（索引 0）：x 镜像、索引不动
        assert_eq!(g[0][0], 100.0 - 10.0, "鼻子 x 应镜像");
        assert_eq!(g[0][1], 20.0);
        // 新左眼（索引 1）= 旧右眼（索引 2）镜像：100-30=70
        assert_eq!(g[1][0], 70.0, "新[1] 应为旧[2] 镜像");
        assert_eq!(g[1][1], 22.0);
        assert_eq!(g[1][2], 2.0, "v 随点换位");
        // 新右眼（索引 2）= 旧左眼（索引 1）镜像：100-10=90
        assert_eq!(g[2][0], 90.0);
        assert_eq!(g[2][1], 21.0);
        // 新右踝（索引 15）= 旧左踝（索引 16）镜像
        assert_eq!(g[15][0], 40.0, "100-60");
        assert_eq!(g[15][2], 0.0);
        // 新左踝（索引 16）= 旧右踝（索引 15）镜像
        assert_eq!(g[16][0], 50.0);
        assert_eq!(g[16][2], 1.0);
        // 成对点都换位：新左肘（索引 7）= 旧右肘（索引 8）镜像
        assert_eq!(g[7][0], 82.0, "100-18");
        assert_eq!(g[7][1], 28.0);
    }

    /// 非 17 点模板：只镜像坐标、不交换索引（防静默错位）。
    #[test]
    fn flip_keypoints_non_17_no_swap() {
        let mut insts = vec![vec![[10.0, 0.0, 2.0], [20.0, 1.0, 1.0], [30.0, 2.0, 2.0]]];
        flip_keypoints(&mut insts, 40.0);
        let g = &insts[0];
        assert_eq!(g[0], [30.0, 0.0, 2.0]);
        assert_eq!(g[1], [20.0, 1.0, 1.0]);
        assert_eq!(g[2], [10.0, 2.0, 2.0]);
    }

    #[test]
    fn flip_polygon_hand_computed() {
        let mut poly = [[2.0, 2.0], [10.0, 2.0], [10.0, 8.0], [2.0, 8.0]];
        flip_polygon(&mut poly, 12.0);
        assert_eq!(poly, [[10.0, 2.0], [2.0, 2.0], [2.0, 8.0], [10.0, 8.0]]);
    }

    // ----- 缩放 -----

    #[test]
    fn scaled_dims_and_points() {
        assert_eq!(scaled_dims(640, 480, 1.1), (704, 528));
        assert_eq!(scaled_dims(640, 480, 0.9), (576, 432));
        assert_eq!(scaled_dims(10, 10, 1.0), (10, 10));
        assert_eq!(scaled_dims(3, 3, 0.05), (1, 1), "不缩到 0");
        let mut p = [100.0, 50.0];
        scale_point(&mut p, 0.5);
        assert_eq!(p, [50.0, 25.0]);
    }

    // ----- 像素域 -----

    /// 2×1 图 [红, 蓝] 翻转 → [蓝, 红]。
    #[test]
    fn hflip_rgb_two_pixels() {
        let mut rgb = vec![255, 0, 0, 0, 0, 255];
        hflip_rgb(2, 1, &mut rgb);
        assert_eq!(rgb, vec![0, 0, 255, 255, 0, 0]);
    }

    /// 3×2 图翻转：行内镜像、行序不变。
    #[test]
    fn hflip_rgb_rows_independent() {
        // 像素 (x,y) 编码为 R=x+1, G=y+1, B=0
        let mut rgb = Vec::new();
        for y in 0..2 {
            for x in 0..3 {
                rgb.extend_from_slice(&[x + 1, y + 1, 0]);
            }
        }
        hflip_rgb(3, 2, &mut rgb);
        for (i, px) in rgb.chunks(3).enumerate() {
            let (x, y) = ((2 - i % 3) as u8, (i / 3) as u8); // 翻转后位置 i 的来源 x
            assert_eq!(px, &[x + 1, y + 1, 0], "像素 {i}");
        }
    }

    /// 通道乘性增益手算 + 上下限截断；gain=1 逐位恒等。
    #[test]
    fn mul_rgb_gain_and_clamp() {
        let mut rgb = vec![200, 100, 10, 0, 255, 128];
        mul_rgb(&mut rgb, [2.0, 0.5, 1.0]);
        // [200*2→255 截断, 100*0.5=50, 10*1=10]，[0, 127.5→round 128? 255*0.5=127.5 → 128, 128*1=128]
        assert_eq!(rgb[0], 255, "上截断");
        assert_eq!(rgb[1], 50);
        assert_eq!(rgb[2], 10);
        assert_eq!(rgb[3], 0, "下截断");
        assert_eq!(rgb[4], 128, "127.5 四舍五入到 128");
        assert_eq!(rgb[5], 128);

        let mut same = vec![7, 128, 255];
        mul_rgb(&mut same, [1.0, 1.0, 1.0]);
        assert_eq!(same, vec![7, 128, 255], "gain=1 应逐位恒等");
    }

    // ----- plan 抽取 / close_last 语义 -----

    #[test]
    fn draw_plan_deterministic_and_in_range() {
        let cfg = AugmentCfg {
            flip: 1.0,
            hsv: [0.1, 0.2, 0.3],
            scale_jitter: Some([0.9, 1.1]),
            ..AugmentCfg::default()
        };
        let mut a = XorShift::new(42);
        let mut b = XorShift::new(42);
        for _ in 0..50 {
            let pa = draw_plan(&cfg, &mut a);
            let pb = draw_plan(&cfg, &mut b);
            assert_eq!(pa, pb, "同种子同序列应可复现");
            assert!(pa.flip, "p=1 必翻");
            assert!(
                (0.9..1.1).contains(&pa.scale),
                "scale 应在 [0.9, 1.1): {pa:?}"
            );
            for (c, (&gain, &g)) in pa.rgb_gains.iter().zip(&cfg.hsv).enumerate() {
                assert!(
                    (1.0 - g..1.0 + g).contains(&gain),
                    "通道 {c} 增益应在 [1−{g}, 1+{g}): {gain}"
                );
            }
        }
        // p=0 永不翻；无 scale_jitter → 1.0
        let mut r = XorShift::new(7);
        let off = AugmentCfg::default();
        for _ in 0..50 {
            let p = draw_plan(&off, &mut r);
            assert!(!p.flip);
            assert_eq!(p.scale, 1.0);
            assert_eq!(p.rgb_gains, [1.0; 3]);
            assert!(p.is_none());
        }
    }

    #[test]
    fn strong_aug_active_close_last_semantics() {
        let mut cfg = AugmentCfg {
            close_last_epochs: 20,
            ..AugmentCfg::default()
        };
        // 200 epochs：1..=180 开，181..=200 关
        assert!(strong_aug_active(&cfg, 1, 200));
        assert!(strong_aug_active(&cfg, 180, 200));
        assert!(!strong_aug_active(&cfg, 181, 200));
        assert!(!strong_aug_active(&cfg, 200, 200));
        // N=0：恒开（含边界）
        cfg.close_last_epochs = 0;
        assert!(strong_aug_active(&cfg, 200, 200));
        // N ≥ total：从第 1 个 epoch 就关
        cfg.close_last_epochs = 300;
        assert!(!strong_aug_active(&cfg, 1, 200));
    }

    #[test]
    fn has_strength_matches_config_axes() {
        assert!(!has_strength(&AugmentCfg::default()));
        assert!(has_strength(&AugmentCfg {
            flip: 0.5,
            ..AugmentCfg::default()
        }));
        assert!(has_strength(&AugmentCfg {
            hsv: [0.0, 0.2, 0.0],
            ..AugmentCfg::default()
        }));
        assert!(has_strength(&AugmentCfg {
            scale_jitter: Some([0.9, 1.1]),
            ..AugmentCfg::default()
        }));
        // 二波新增轴：mosaic / mixup 任一 > 0 即有强度
        assert!(has_strength(&AugmentCfg {
            mosaic: 1.0,
            ..AugmentCfg::default()
        }));
        assert!(has_strength(&AugmentCfg {
            mixup: 0.2,
            ..AugmentCfg::default()
        }));
    }

    // ----- mosaic（手算） -----

    #[test]
    fn mosaic_canvas_dims_hand_computed() {
        assert_eq!(mosaic_canvas_dims(2, 3), (4, 6));
        assert_eq!(mosaic_canvas_dims(1, 1), (2, 2));
    }

    /// 单框换算：缩放 + 平移 + 拼接线裁剪 + 退化丢弃，全部手算对照。
    #[test]
    fn mosaic_map_box_hand_computed() {
        // 恒等缩放（源=象限尺寸），象限原点 (0,0)：原样通过
        assert_eq!(
            mosaic_map_box([0.5, 0.5, 1.5, 1.5], 2.0, 2.0, 2.0, 2.0, 0.0, 0.0),
            Some([0.5, 0.5, 1.5, 1.5])
        );
        // 平移：右上象限 (x0=2)：[0,0,1,1] → [2,0,3,1]
        assert_eq!(
            mosaic_map_box([0.0, 0.0, 1.0, 1.0], 2.0, 2.0, 2.0, 2.0, 2.0, 0.0),
            Some([2.0, 0.0, 3.0, 1.0])
        );
        // 缩放：源 4×4 → 象限 2×2（比例 0.5），象限原点 (0,2)：
        // [2,2,4,4] → (1,1,2,2) + (0,2) = [1,3,2,4]
        assert_eq!(
            mosaic_map_box([2.0, 2.0, 4.0, 4.0], 4.0, 4.0, 2.0, 2.0, 0.0, 2.0),
            Some([1.0, 3.0, 2.0, 4.0])
        );
        // 拼接线裁剪：越界部分截断（不做越界还原）
        // [-1, 0, 3, 1] 在 2×2 象限 (0,0) → 裁剪到 [0,0,2,1]
        assert_eq!(
            mosaic_map_box([-1.0, 0.0, 3.0, 1.0], 2.0, 2.0, 2.0, 2.0, 0.0, 0.0),
            Some([0.0, 0.0, 2.0, 1.0])
        );
        // 完全在象限外 → 裁剪后面积 0 → None
        assert_eq!(
            mosaic_map_box([4.0, 4.0, 6.0, 6.0], 2.0, 2.0, 2.0, 2.0, 0.0, 0.0),
            None
        );
        // 裁剪后零面积（线/点）→ None
        assert_eq!(
            mosaic_map_box([1.0, 1.0, 1.0, 2.0], 2.0, 2.0, 2.0, 2.0, 0.0, 0.0),
            None
        );
    }

    /// 4 图拼 1：画布尺寸、象限像素归属、框合并换算，全部手算对照。
    #[test]
    fn mosaic_compose_hand_computed() {
        // 象限 2×2 → 画布 4×4；每象限纯色便于核对像素归属
        let px = |r, g, b| {
            vec![[r, g, b]; 4]
                .into_iter()
                .flatten()
                .collect::<Vec<u8>>()
        };
        let red = px(255, 0, 0);
        let green = px(0, 255, 0);
        let blue = px(0, 0, 255);
        let white = px(255, 255, 255);
        // 每图一个框（各自坐标），换算后应逐一落位
        let items = [
            MosaicItem {
                rgb: &red,
                src_w: 2,
                src_h: 2,
                boxes: &[[0.0, 0.0, 2.0, 2.0]],
                labels: &[7],
            },
            MosaicItem {
                rgb: &green,
                src_w: 2,
                src_h: 2,
                boxes: &[[0.0, 0.0, 1.0, 1.0]],
                labels: &[8],
            },
            MosaicItem {
                rgb: &blue,
                src_w: 2,
                src_h: 2,
                boxes: &[[1.0, 1.0, 2.0, 2.0]],
                labels: &[9],
            },
            MosaicItem {
                rgb: &white,
                src_w: 2,
                src_h: 2,
                boxes: &[[0.0, 1.0, 1.0, 2.0]],
                labels: &[10],
            },
        ];
        let (canvas, boxes, labels) = mosaic_compose(2, 2, &items);
        // 画布 4×4 RGB8
        assert_eq!(canvas.len(), 4 * 4 * 3);
        // 象限像素归属：TL=红、TR=绿、BL=蓝、BR=白
        let pixel = |x: usize, y: usize| &canvas[(y * 4 + x) * 3..(y * 4 + x) * 3 + 3];
        assert_eq!(pixel(0, 0), &[255, 0, 0], "左上=红");
        assert_eq!(pixel(3, 0), &[0, 255, 0], "右上=绿");
        assert_eq!(pixel(0, 3), &[0, 0, 255], "左下=蓝");
        assert_eq!(pixel(3, 3), &[255, 255, 255], "右下=白");
        // 框合并：顺序 = 象限顺序，坐标各自换算
        assert_eq!(
            boxes,
            vec![
                [0.0, 0.0, 2.0, 2.0], // TL 恒等
                [2.0, 0.0, 3.0, 1.0], // TR 平移 (2,0)
                [1.0, 3.0, 2.0, 4.0], // BL [1,1,2,2] + (0,2)
                [2.0, 3.0, 3.0, 4.0], // BR [0,1,1,2] + (2,2)
            ]
        );
        assert_eq!(labels, vec![7, 8, 9, 10]);
    }

    /// 缩放来源：源 4×4 → 象限 2×2（像素已按调用方约定缩放到象限尺寸，
    /// src_w/src_h 保留原始 4×4 供框换算），框随像素同一 0.5 比例映射。
    #[test]
    fn mosaic_compose_scaled_source_maps_boxes() {
        // 像素已缩放到象限 2×2（调用方约定）；src 4×4 仅用于框换算
        let rgb2 = vec![9u8; 2 * 2 * 3];
        let items = [
            MosaicItem {
                rgb: &rgb2,
                src_w: 4,
                src_h: 4,
                boxes: &[[2.0, 2.0, 4.0, 4.0]],
                labels: &[1],
            },
            MosaicItem {
                rgb: &rgb2,
                src_w: 4,
                src_h: 4,
                boxes: &[],
                labels: &[],
            },
            MosaicItem {
                rgb: &rgb2,
                src_w: 4,
                src_h: 4,
                boxes: &[],
                labels: &[],
            },
            MosaicItem {
                rgb: &rgb2,
                src_w: 4,
                src_h: 4,
                boxes: &[],
                labels: &[],
            },
        ];
        let (_, boxes, labels) = mosaic_compose(2, 2, &items);
        assert_eq!(boxes, vec![[1.0, 1.0, 2.0, 2.0]]);
        assert_eq!(labels, vec![1]);
    }

    // ----- mixup（手算） -----

    #[test]
    fn mixup_rgb_lambda_hand_computed() {
        // λ=0.25：round(100·0.25 + 200·0.75) = 175；round(0·0.25+100·0.75) = 75；
        // round(255·0.25 + 0·0.75) = round(63.75) = 64
        let out = mixup_rgb(&[100, 0, 255], &[200, 100, 0], 0.25);
        assert_eq!(out, vec![175, 75, 64]);
        // λ=1 → 逐位 a；λ=0 → 逐位 b
        let a = [1u8, 127, 253];
        let b = [200u8, 60, 9];
        assert_eq!(mixup_rgb(&a, &b, 1.0), a.to_vec(), "λ=1 应逐位等于 a");
        assert_eq!(mixup_rgb(&a, &b, 0.0), b.to_vec(), "λ=0 应逐位等于 b");
        // 上下限截断不越界
        let out = mixup_rgb(&[255, 0], &[255, 0], 0.5);
        assert_eq!(out, vec![255, 0]);
    }

    /// Beta(α, α) 采样：确定性、值域、均值 0.5（对称 Beta 的精确均值）。
    #[test]
    fn beta_symmetric_deterministic_and_centered() {
        assert_eq!(beta_symmetric(&mut XorShift::new(1), 0.0), 0.5, "α≤0 → 0.5");
        assert_eq!(beta_symmetric(&mut XorShift::new(1), -1.0), 0.5);
        let mut a = XorShift::new(42);
        let mut b = XorShift::new(42);
        let n = 4000;
        let mut sum = 0f32;
        for _ in 0..n {
            let x = beta_symmetric(&mut a, 0.2);
            let y = beta_symmetric(&mut b, 0.2);
            assert_eq!(x, y, "同种子应可复现");
            assert!(x > 0.0 && x < 1.0, "λ 应在开区间 (0,1): {x}");
            sum += x;
        }
        // 对称 Beta 均值恒 0.5；4000 样本均值应在 ±0.05 内
        let mean = sum / n as f32;
        assert!(
            (mean - 0.5).abs() < 0.05,
            "Beta(0.2,0.2) 均值应≈0.5，得到 {mean}"
        );
    }

    // ----- 组合增强抽样：概率=0 零消耗（逐位不变的关键） -----

    #[test]
    fn draw_composite_zero_prob_consumes_nothing() {
        // mosaic=0 且 mixup=0：不消耗任何随机数 → 后续 draw_plan 流与
        // 历史版本逐位一致
        let cfg = AugmentCfg::default();
        let mut r1 = XorShift::new(123);
        let mut r2 = XorShift::new(123);
        for _ in 0..50 {
            let d = draw_composite(&cfg, &mut r1);
            assert_eq!(d, CompositeDraw::none());
            assert_eq!(r1.next_f32(), r2.next_f32(), "RNG 状态不应被推动");
        }
    }

    #[test]
    fn draw_composite_coins_and_lambda() {
        // p=1 必命中；只开 mosaic 时 λ 恒 0.5 且不消耗 Beta 采样
        let mut cfg = AugmentCfg {
            mosaic: 1.0,
            ..AugmentCfg::default()
        };
        let mut r = XorShift::new(9);
        for _ in 0..50 {
            let d = draw_composite(&cfg, &mut r);
            assert!(d.mosaic && !d.mixup && d.mixup_lam == 0.5);
        }
        // mosaic + mixup 双开：两枚硬币都命中，λ ∈ (0,1)
        cfg.mixup = 1.0;
        let mut r = XorShift::new(5);
        let mut first = None;
        for i in 0..50 {
            let d = draw_composite(&cfg, &mut r);
            assert!(d.mosaic && d.mixup);
            assert!(d.mixup_lam > 0.0 && d.mixup_lam < 1.0);
            if i == 0 {
                first = Some(d);
            }
        }
        // 同种子可复现
        let mut r2 = XorShift::new(5);
        assert_eq!(draw_composite(&cfg, &mut r2), first.unwrap());
    }
}
