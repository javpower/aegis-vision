//! OKS（Object Keypoint Similarity，PLAN §4.4 关键点质量度量/损失）。
//!
//! 定义（COCO keypoints 协议，pycocotools 同式）：
//! `OKS = Σ[exp(−d²/(2·s²·k²))·δ(v>0)] / Σδ(v>0)`，其中
//! - `d²` = 预测点与 gt 点的欧氏距离平方（画布像素域）；
//! - `s` = 目标尺度 = sqrt(实例面积)（本实现用 gt 框面积 w·h 代替分割面积，
//!   与任务规范一致；coco 级标注里 mask 面积更准，框面积是其常见近似）；
//! - `k` = COCO 17 关键点 per-kpt 落差常数（[`COCO_SIGMAS`]，pycocotools 的
//!   falloff/10；超集/自定义模板按 [`sigma_for`] 循环末位）；
//! - `v` = gt 可见性标志（COCO：0=未标注 / 1=标注但遮挡 / 2=可见），v>0 计入。
//!
//! **两种形态**：
//! - [`oks_scalar`]：纯标量版（评测用，engine 报 mean OKS）；
//! - [`oks_loss`]：张量版 `1 − mean(OKS)`，对 pred 坐标全可微（exp 的二次距离
//!   项可微），供 [`crate::models::KeypointModel`] 的损失直接消费。
//!
//! 取值域 (0, 1]：同点 → 1；偏移越大越接近 0。损失 `1 − OKS` 同点 → 0。

/// COCO 17 关键点 per-kpt 落差常数（pycocotools falloff/10）。
pub const COCO_SIGMAS: [f32; 17] = [
    0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107, 0.107,
    0.087, 0.087, 0.089, 0.089,
];

/// 第 j 个关键点的 sigma 常数（j 超出 COCO 17 时循环末位——自定义模板的
/// 退化近似，PLAN 允许模板注入，正式自定义表按配置落地）。
pub fn sigma_for(j: usize) -> f32 {
    COCO_SIGMAS[j.min(COCO_SIGMAS.len() - 1)]
}

/// 前 K 个关键点的 sigma 表。
pub fn sigma_table(k: usize) -> Vec<f32> {
    (0..k).map(sigma_for).collect()
}

/// 标量 OKS（评测用）：pred/gt 均为 K×[x, y, v]（v 用 **gt** 的可见性标志，
/// 预测侧 v 位忽略）；scale = sqrt(实例面积)（像素）。无可见点返回 0。
pub fn oks_scalar(pred: &[[f32; 3]], gt: &[[f32; 3]], scale: f32) -> f32 {
    let s = scale.max(1e-3) as f64;
    let mut num = 0f64;
    let mut den = 0usize;
    for (j, (p, g)) in pred.iter().zip(gt.iter()).enumerate() {
        if g[2] <= 0.0 {
            continue; // δ(v>0)：不可见点不计入分子分母
        }
        let dx = (p[0] - g[0]) as f64;
        let dy = (p[1] - g[1]) as f64;
        let d2 = dx * dx + dy * dy;
        let k = sigma_for(j) as f64;
        num += (-d2 / (2.0 * s * s * k * k)).exp();
        den += 1;
    }
    if den == 0 {
        return 0.0;
    }
    (num / den as f64) as f32
}

// ---------------------------------------------------------------------------
// 可微张量版（PLAN §4.4 OKS 损失）
// ---------------------------------------------------------------------------

/// 张量 OKS 损失 `1 − mean_i(OKS_i)`，对 pred 全可微。
///
/// - `pred`/`gt`：[G, K, 2]（画布像素坐标；gt 只取坐标，可见性走 `vis`）；
/// - `vis`：[G, K]，0/1 浮点（δ(v>0) 已由调用方二值化）；
/// - `scale`：[G]，实例尺度 sqrt(面积)（像素）。
///
/// 全部逐元素广播实现（无 gather），负距离指数天然有界（d² 大 → exp → 0），
/// scale/sigma 均 > 0 无除零；G=0 或全不可见由调用方保证不调用
/// （内部 clamp_min 兜底为损失 1 的常数，反传零梯度不断图）。
#[cfg(feature = "torch")]
pub fn oks_loss(pred: &tch::Tensor, gt: &tch::Tensor, vis: &tch::Tensor, scale: &tch::Tensor) -> tch::Tensor {
    use tch::Kind;
    let size = pred.size();
    let (g, k) = (size[0], size[1]);
    let device = pred.device();
    let sig: Vec<f32> = sigma_table(k as usize);
    let sigma = tch::Tensor::from_slice(&sig)
        .to_device(device)
        .to_kind(Kind::Float)
        .reshape([1i64, k, 1]);
    let s2 = (scale * scale).reshape([g, 1i64, 1]); // [G,1,1]
    let denom = &(&s2 * 2.0) * &sigma * &sigma; // [G,K,1] = 2·s²·σ²（σ 需平方！漏平方会高估 OKS）
    let d = pred - gt; // [G,K,2]
    let d2 = (d.select(2, 0) * d.select(2, 0) + d.select(2, 1) * d.select(2, 1)).reshape([g, k, 1]);
    let e = (&d2 / &denom).neg().exp().reshape([g, k]); // [G,K]（显式回压，防 [G,K,1]×[G,K] 静默广播）
    let num = (e * vis).sum_dim_intlist(&[1i64][..], false, Kind::Float); // [G]
    let den = vis.sum_dim_intlist(&[1i64][..], false, Kind::Float).clamp_min(1.0);
    let oks_mean = (&num / &den).mean(Kind::Float);
    // 损失 = 1 − mean OKS（标量在左的 `1.0 - &t` 会踩 E0282，改写为右乘加）
    oks_mean * -1.0 + 1.0
}

#[cfg(all(test, feature = "torch"))]
mod tests {
    use super::*;
    use tch::{Device, Kind, Tensor};

    /// 同点 → OKS = 1（损失 0）；已知偏移 → 手算值（容差 1e-3）。
    #[test]
    fn oks_same_point_is_one_and_known_offset_matches_hand_computed() {
        // 单点（K=1，nose sigma=0.026），实例尺度 s=32（32×32 框）
        let gt = [[10.0f32, 10.0, 2.0]];
        let same = [[10.0f32, 10.0, 2.0]];
        let oks = oks_scalar(&same, &gt, 32.0);
        assert!((oks - 1.0).abs() < 1e-6, "同点 OKS 应为 1，got {oks}");

        // pred 偏移 (2, 0)：OKS = exp(−4 / (2·32²·0.026²))
        let off = [[12.0f32, 10.0, 2.0]];
        let oks = oks_scalar(&off, &gt, 32.0);
        let expected =
            (-(4.0f64) / (2.0 * 32.0 * 32.0 * 0.026 * 0.026)).exp() as f32;
        assert!(
            (oks - expected).abs() < 1e-3,
            "已知偏移 OKS 应为 {expected}，got {oks}"
        );
        assert!((expected - 0.0556).abs() < 1e-3, "手算校验: {expected}");
    }

    /// 可见性 δ(v>0)：不可见 gt 点不计入分子分母；全不可见 → 0。
    #[test]
    fn oks_visibility_gating() {
        // K=2：点 0 可见且偏移 (2,0)（OKS≈0.0556），点 1 不可见（应被剔除）
        let gt = [[10.0f32, 10.0, 2.0], [50.0, 50.0, 0.0]];
        let pred = [[12.0f32, 10.0, 2.0], [91.0, 91.0, 2.0]];
        let oks = oks_scalar(&pred, &gt, 32.0);
        let expected = (-(4.0f64) / (2.0 * 32.0 * 32.0 * 0.026 * 0.026)).exp() as f32;
        assert!(
            (oks - expected).abs() < 1e-3,
            "不可见点应被剔除：期望 {expected}，got {oks}"
        );

        // 全不可见 → 0（无可用监督）
        let gt0 = [[10.0f32, 10.0, 0.0]];
        assert_eq!(oks_scalar(&pred, &gt0, 32.0), 0.0);
    }

    /// 多点加权：K=2 全可见，点 0 偏移 (2,0)、点 1 同点
    /// → OKS = (exp(−4/(2·32²·0.026²)) + 1) / 2。
    #[test]
    fn oks_multi_point_mean() {
        let gt = [[10.0f32, 10.0, 2.0], [50.0, 50.0, 2.0]];
        let pred = [[12.0f32, 10.0, 2.0], [50.0, 50.0, 2.0]];
        let oks = oks_scalar(&pred, &gt, 32.0);
        let e0 = (-(4.0f64) / (2.0 * 32.0 * 32.0 * 0.026 * 0.026)).exp();
        let expected = ((e0 + 1.0) / 2.0) as f32;
        assert!((oks - expected).abs() < 1e-3, "期望 {expected}，got {oks}");
    }

    /// 张量损失：同点 → ≈0；已知偏移 → 1 − 手算 OKS；对 pred 反传有梯度。
    #[test]
    fn oks_loss_tensor_matches_scalar_and_backprops() {
        // G=2 实例 × K=2 点：实例 0 全同点（OKS=1），实例 1 点 0 偏移 (2,0)
        let pred = Tensor::from_slice(&[
            10.0f32, 10.0, 50.0, 50.0, // 实例 0
            12.0, 10.0, 50.0, 50.0, // 实例 1
        ])
        .to_device(Device::Cpu)
        .to_kind(Kind::Float)
        .reshape([2i64, 2, 2])
        .set_requires_grad(true);
        let gt = Tensor::from_slice(&[
            10.0f32, 10.0, 50.0, 50.0, 10.0, 10.0, 50.0, 50.0,
        ])
        .to_device(Device::Cpu)
        .to_kind(Kind::Float)
        .reshape([2i64, 2, 2]);
        let vis = Tensor::from_slice(&[1.0f32, 1.0, 1.0, 1.0])
            .to_device(Device::Cpu)
            .reshape([2i64, 2]);
        let scale = Tensor::from_slice(&[32.0f32, 32.0])
            .to_device(Device::Cpu)
            .reshape([2i64]);

        let loss = oks_loss(&pred, &gt, &vis, &scale);
        // 实例 0 全同点 OKS=1；实例 1 = (exp(−4/(2·32²·0.026²)) + 1) / 2；
        // 损失 = 1 − 两者均值
        let e0 = (-(4.0f64) / (2.0 * 32.0 * 32.0 * 0.026 * 0.026)).exp();
        let expected = 1.0 - (1.0 + (e0 + 1.0) / 2.0) / 2.0;
        let got = loss.double_value(&[]);
        assert!(
            (got - expected).abs() < 1e-3,
            "张量 OKS 损失应 = {expected}，got {got}"
        );
        loss.backward(); // 反传不应 panic
        let g = pred.grad();
        let gnorm = g.abs().sum(Kind::Float).double_value(&[]);
        assert!(gnorm > 0.0, "pred 坐标应收到非零梯度（norm={gnorm}）");
    }
}
