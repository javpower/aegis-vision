//! KFIoU / ProbIoU 旋转框回归损失（PLAN §4.2 OBB，全部可微张量运算）。
//!
//! 数学基础（The KFIoU Loss for Rotated Object Detection, arXiv:2301.02696）：
//! 把旋转框 (cx, cy, w, h, θ) 视为其等效二维高斯（均匀矩形的二阶中心矩）
//!
//! ```text
//! Σ = R(θ) · diag(w², h²)/12 · R(θ)ᵀ
//!   σxx = (w²cos²θ + h²sin²θ)/12
//!   σyy = (w²sin²θ + h²cos²θ)/12
//!   σxy = (w² − h²)·sinθcosθ/12
//! ```
//!
//! 两个分布的不匹配度用 KL 散度度量（协方差差/中心差全部二次型，无逐点 IoU
//! 的不可导边界问题，且对 θ 的周期性天然闭合——θ 与 θ+π 给出同一 Σ）：
//!
//! ```text
//! D_KL(N1‖N2) = ½[ tr(Σ2⁻¹Σ1) + (μ2−μ1)ᵀΣ2⁻¹(μ2−μ1) + ln(|Σ2|/|Σ1|) − 2 ]
//! ```
//!
//! KFIoU 把 KL 经一阶 Taylor（exp(−KL) ≈ 1 − KL）转成 IoU 形状的量后，
//! 论文再套单调变换（log1p / sqrt1p）匹配 IoU 损失曲线：相同框 → 0。
//! [`probiou_element`] 是对照路径：高斯 Bhattacharyya 系数的 −ln。
//!
//! 约定：入参 box5 张量的**通道维（dim 1）必须为 5**（cx, cy, w, h, θ 弧度、
//! 像素单位），与全库 [N, C, H, W] 布局一致——逐 cell 用 [N,5,H,W]，逐对用 [N,5]；
//! 返回逐 element 损失（其余维度保持不变），掩码 / 归一 / 加权由调用方完成
//! （models::loss_obb）。

use tch::Tensor;

/// 均匀矩形二阶中心矩系数：Var = diag(w², h²)/12。
const COV: f64 = 1.0 / 12.0;

/// KL → 损失的单调变换（论文 fun 参数）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KfTransform {
    /// L = ln(1 + KL)（论文默认：梯度对小 KL 更平滑）
    Log1p,
    /// L = √(1 + KL) − 1（对大 KL 梯度衰减更强）
    Sqrt1p,
    /// L = KL（一阶 Taylor 直接形式：pseudo-IoU = 1 − KL）
    Identity,
}

/// box5 通道维（dim 1 = 5）的高斯协方差分量 (σxx, σxy, σyy)。
/// 返回三个去掉通道维的张量（[N,5,H,W] → [N,H,W]；[N,5] → [N]）。
fn covariance(box5: &Tensor) -> (Tensor, Tensor, Tensor) {
    debug_assert_eq!(box5.size().get(1).copied(), Some(5), "box5 通道维必须为 5");
    let w = box5.select(1, 2);
    let h = box5.select(1, 3);
    let th = box5.select(1, 4);
    let sin = th.sin();
    let cos = th.cos();
    let m2 = &w * &w * COV; // w²/12
    let n2 = &h * &h * COV; // h²/12
    let sxx = &m2 * &cos * &cos + &n2 * &sin * &sin;
    let syy = &m2 * &sin * &sin + &n2 * &cos * &cos;
    let sxy = (&m2 - &n2) * &sin * &cos;
    (sxx, sxy, syy)
}

/// KL(Np‖Nt)：输入两盒的协方差分量（引用）与中心差张量。返回逐 element 的
/// KL ≥ 0（数学上恒非负；浮点噪声可能产生 ~1e-7 级负值，由 kfiou_element 截断）。
#[allow(clippy::too_many_arguments)]
fn kl_gaussians(
    pxx: &Tensor,
    pxy: &Tensor,
    pyy: &Tensor,
    txx: &Tensor,
    txy: &Tensor,
    tyy: &Tensor,
    dx: &Tensor,
    dy: &Tensor,
) -> Tensor {
    let det_t = (txx * tyy - txy * txy).clamp_min(1e-12);
    let det_p = (pxx * pyy - pxy * pxy).clamp_min(1e-12);
    // tr(Σt⁻¹Σp) = (tyy·pxx + txx·pyy − 2·txy·pxy)/det_t
    let tr = (&(tyy * pxx + txx * pyy) - txy * pxy * 2.0) / &det_t;
    // 马氏距离项 (μt−μp)ᵀ Σt⁻¹ (μt−μp) = (tyy·dx² − 2txy·dx·dy + txx·dy²)/det_t
    let quad = ((tyy * dx) * dx + (txx * dy) * dy - (txy * dx) * dy * 2.0) / &det_t;
    let log_term = (&det_t / &det_p).log();
    (tr + quad + log_term - 2.0) * 0.5
}

/// KFIoU 逐 element 损失：pred5/target5 通道维 = 5；相同框 → 0，值域 [0, +∞)。
pub fn kfiou_element(pred5: &Tensor, target5: &Tensor, transform: KfTransform) -> Tensor {
    let (pxx, pxy, pyy) = covariance(pred5);
    let (txx, txy, tyy) = covariance(target5);
    let dx = target5.select(1, 0) - pred5.select(1, 0);
    let dy = target5.select(1, 1) - pred5.select(1, 1);
    let kl = kl_gaussians(&pxx, &pxy, &pyy, &txx, &txy, &tyy, &dx, &dy);
    // KL 数学上恒 ≥ 0；截断浮点噪声负值，保证 log1p/sqrt1p 定义域安全
    let kl = kl.clamp_min(0.0);
    match transform {
        KfTransform::Log1p => (&kl + 1.0).log(),
        KfTransform::Sqrt1p => (&kl + 1.0).sqrt() - 1.0,
        KfTransform::Identity => kl,
    }
}

/// ProbIoU 逐 element 损失（对照路径）：高斯 Bhattacharyya 系数
///
/// ```text
/// BC = √( √(|Σp||Σt|) / |Σ*| ) · exp(−(1/8)·ΔμᵀΣ*⁻¹Δμ)，  Σ* = (Σp+Σt)/2
/// loss = −ln(BC)
/// ```
///
/// BC ∈ (0, 1]（相同框 → 1 → loss 0）。与 KFIoU 同族（高斯拟合），论文中作为
/// KFIoU 的对照基线（mmrotate ProbiouLoss 同款几何）。
pub fn probiou_element(pred5: &Tensor, target5: &Tensor) -> Tensor {
    let (pxx, pxy, pyy) = covariance(pred5);
    let (txx, txy, tyy) = covariance(target5);
    let dx = target5.select(1, 0) - pred5.select(1, 0);
    let dy = target5.select(1, 1) - pred5.select(1, 1);
    // Σ* = (Σp+Σt)/2
    let sxx = (&pxx + &txx) * 0.5;
    let sxy = (&pxy + &txy) * 0.5;
    let syy = (&pyy + &tyy) * 0.5;
    let det_s = (&sxx * &syy - &sxy * &sxy).clamp_min(1e-12);
    let det_p = (&pxx * &pyy - &pxy * &pxy).clamp_min(1e-12);
    let det_t = (&txx * &tyy - &txy * &txy).clamp_min(1e-12);
    // q = Δμᵀ Σ*⁻¹ Δμ
    let q = ((&syy * &dx) * &dx + (&sxx * &dy) * &dy - (&sxy * &dx) * &dy * 2.0) / &det_s;
    let bc = (((&det_p * &det_t).sqrt() / &det_s).sqrt()) * (&q * -0.125).exp();
    bc.clamp_min(1e-7).log().neg()
}

#[cfg(all(test, feature = "torch"))]
mod tests {
    use super::*;
    use tch::{Device, Kind};

    const TOL: f64 = 1e-3;

    /// [1,5,1,1] 的单框张量。
    fn b5(cx: f64, cy: f64, w: f64, h: f64, theta: f64) -> Tensor {
        let v = [cx as f32, cy as f32, w as f32, h as f32, theta as f32];
        Tensor::from_slice(&v).reshape([1i64, 5, 1, 1])
    }

    fn scalar(t: &Tensor) -> f64 {
        t.double_value(&[])
    }

    #[test]
    fn kfiou_identical_boxes_is_zero() {
        let a = b5(3.0, 7.0, 10.0, 6.0, 0.3);
        for tr in [
            KfTransform::Log1p,
            KfTransform::Sqrt1p,
            KfTransform::Identity,
        ] {
            let l = scalar(&kfiou_element(&a, &a, tr));
            assert!(l.abs() < 1e-6, "{tr:?}: got {l}");
        }
        let p = scalar(&probiou_element(&a, &a));
        assert!(p.abs() < 1e-6, "probiou: got {p}");
    }

    #[test]
    fn kfiou_quarter_turn_with_wh_swap_is_near_zero() {
        // (w,h,θ) 与 (h,w,θ+π/2) 是同一几何框：Σ 恒等 → KL ≈ 0（f32 三角误差级）
        let a = b5(3.0, 7.0, 10.0, 6.0, 0.3);
        let b = b5(3.0, 7.0, 6.0, 10.0, 0.3 + std::f64::consts::FRAC_PI_2);
        let l = scalar(&kfiou_element(&a, &b, KfTransform::Identity));
        assert!(l.abs() < TOL, "KFIoU 90°+互换应 ≈0，got {l}");
        let p = scalar(&probiou_element(&a, &b));
        assert!(p.abs() < TOL, "ProbIoU 90°+互换应 ≈0，got {p}");
    }

    #[test]
    fn kfiou_known_angle_hand_computed() {
        // 盒 1 = (0,0,10,6,0)；盒 2 = (0,0,10,6,30°)。同中心、同 det = 25：
        //   Σ1 = diag(25/3, 3)
        //   Σ2xx = 84/12 = 7，Σ2yy = 52/12 = 13/3，Σ2xy = 32·sin30°·cos30°/12 = 2.309401
        //   tr(Σ2⁻¹Σ1) = (Σ2yy·Σ1xx + Σ2xx·Σ1yy)/25 = (325/9 + 21)/25 = 514/225
        //   KL = ½(514/225 − 2) = 32/225 ≈ 0.142222
        let a = b5(0.0, 0.0, 10.0, 6.0, 0.0);
        let b = b5(0.0, 0.0, 10.0, 6.0, std::f64::consts::PI / 6.0);
        let kl = scalar(&kfiou_element(&a, &b, KfTransform::Identity));
        assert!((kl - 32.0 / 225.0).abs() < TOL, "KL={kl}");
        let log1p = scalar(&kfiou_element(&a, &b, KfTransform::Log1p));
        // ln(1 + 32/225) = ln(257/225) ≈ 0.132976
        assert!(
            (log1p - (257.0f64 / 225.0).ln()).abs() < TOL,
            "log1p={log1p}"
        );
        let sqrt1p = scalar(&kfiou_element(&a, &b, KfTransform::Sqrt1p));
        // √(1 + KL) − 1 ≈ 0.068748
        assert!(
            (sqrt1p - ((1.0f64 + 32.0f64 / 225.0).sqrt() - 1.0)).abs() < TOL,
            "sqrt1p={sqrt1p}"
        );
    }

    #[test]
    fn kfiou_center_offset_hand_computed() {
        // 盒 1 = (0,0,10,6,0)；盒 2 = (2,0,10,6,0)：同 Σ = diag(25/3,3)（det=25）
        //   tr = 2，马氏项 = dx²·(3/25) = 4·0.12 = 0.48，ln 项 = 0
        //   KL = ½(2 + 0.48 − 2) = 0.24
        let a = b5(0.0, 0.0, 10.0, 6.0, 0.0);
        let b = b5(2.0, 0.0, 10.0, 6.0, 0.0);
        let kl = scalar(&kfiou_element(&a, &b, KfTransform::Identity));
        assert!((kl - 0.24).abs() < TOL, "KL={kl}");
        let log1p = scalar(&kfiou_element(&a, &b, KfTransform::Log1p));
        assert!((log1p - 1.24f64.ln()).abs() < TOL, "log1p={log1p}");
    }

    #[test]
    fn probiou_known_angle_hand_computed() {
        // 同上 30° 情形：Σ* = ((25/3+7)/2, (3+13/3)/2, 2.309401/2)
        //   det* = 7.666667·3.666667 − 1.154701² = 26.777778；det1 = det2 = 25；q = 0
        //   BC = √(25/26.777778) ≈ 0.966281 → loss = −ln(BC) ≈ 0.034319
        let a = b5(0.0, 0.0, 10.0, 6.0, 0.0);
        let b = b5(0.0, 0.0, 10.0, 6.0, std::f64::consts::PI / 6.0);
        let l = scalar(&probiou_element(&a, &b));
        assert!((l - 0.0343191).abs() < TOL, "probiou loss={l}");
        // 角度差越大损失越大（同 90°：等价于 w/h 互换 → 回到 ≈0，见上测）
    }

    #[test]
    fn kfiou_backward_smoke() {
        // 反传通路：梯度存在且有限（先 reshape 再挂 requires_grad，保证 pred 是叶子）
        let pred = Tensor::from_slice(&[1.0f32, 2.0, 10.0, 6.0, 0.2])
            .reshape([1i64, 5, 1, 1])
            .set_requires_grad(true);
        let target = b5(2.0, 3.0, 10.0, 6.0, 0.5);
        let loss = kfiou_element(&pred, &target, KfTransform::Log1p).sum(Kind::Float);
        loss.backward();
        let g = pred.grad();
        assert!(g.numel() == 5);
        let gv = g.to_device(Device::Cpu).to_kind(Kind::Float).reshape([-1]);
        for i in 0..5i64 {
            let v = gv.double_value(&[i]);
            assert!(v.is_finite(), "grad[{i}] = {v}");
        }
    }
}
