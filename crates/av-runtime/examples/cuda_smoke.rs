//! 实验 A（GPU 冒烟，结论记录在 docs/gpu.md）：检测当前链接的 libtorch 能否在
//! NVIDIA CUDA 设备上执行真实训练路径的算子（elementwise → matmul → conv2d
//! 前向+反向 → 项目检测模型 loss+backward）。
//!
//! 背景：RTX 5060 Ti 是 Blackwell 架构（sm_120）。libtorch 2.4.0 的 CUDA 发行版是
//! cu121，官方预编译内核只覆盖到 sm_90；能否运行取决于包内 PTX 是否被驱动 JIT。
//! 若报 "no kernel image is available for execution on the device"，即该算子既无
//! 本架构 SASS 也无 PTX 可 JIT → CUDA 路线在当前组合下不可用。
//!
//! 用默认（CPU 版）libtorch 2.4 编译本例会得到 cpu-fallback 结果，同样有验证价值
//! （证明设备探测与回退逻辑正确）：
//! ```text
//! cargo run -p av-runtime --example cuda_smoke --no-default-features --features torch
//! ```
//! 用 CUDA 版 libtorch（下载并解压 cu121 包后）：
//! ```text
//! LIBTORCH=E:\libs\libtorch-cu121 \
//! cargo run -p av-runtime --example cuda_smoke --no-default-features --features torch
//! ```

use tch::nn::VarStore;
use tch::{Cuda, Device, Kind, Tensor};

use av_core::config::RunConfig;
use av_tasks::models::{build_model, TrainBatch};

fn main() {
    println!("== CUDA smoke (libtorch via torch-sys 0.17 / tch 0.17) ==");
    println!("[probe] Cuda::device_count = {}", Cuda::device_count());
    println!("[probe] Cuda::is_available = {}", Cuda::is_available());
    println!(
        "[probe] Cuda::cudnn_is_available = {}",
        Cuda::cudnn_is_available()
    );
    let device = Device::cuda_if_available();
    println!("[probe] Device::cuda_if_available = {device:?}");

    if device == Device::Cpu {
        println!("RESULT: cpu-fallback（未探测到 CUDA 运行时：链接的是 CPU 版 libtorch，或无 NVIDIA 驱动）");
        return;
    }

    // 步骤 1：elementwise（ATen 通用内核；若包内有 PTX，此处由驱动 JIT）
    let a = Tensor::randn([1024, 1024], (Kind::Float, device));
    let b = &a + 1.0;
    println!(
        "[step1 elementwise add] ok, sum={:.3e}",
        b.sum(Kind::Float).double_value(&[])
    );

    // 步骤 2：matmul（cuBLAS 内核库）
    let c = a.matmul(&b);
    println!(
        "[step2 matmul cuBLAS] ok, norm={:.3e}",
        c.norm().double_value(&[])
    );

    // 步骤 3：conv2d 前向 + 反向（cuDNN / ATen conv 路径，训练核心算子）
    let ws = Tensor::randn([8, 3, 3, 3], (Kind::Float, device)).set_requires_grad(true);
    let bs = Tensor::randn([8], (Kind::Float, device)).set_requires_grad(true);
    let x = Tensor::randn([2, 3, 64, 64], (Kind::Float, device));
    let y = x.conv2d(&ws, Some(&bs), [1, 1], [1, 1], [1, 1], 1);
    let loss = y.square().mean(Kind::Float);
    loss.backward();
    println!(
        "[step3 conv2d fwd+bwd] ok, loss={:.6} w-grad-norm={:.6}",
        loss.double_value(&[]),
        ws.grad().norm().double_value(&[])
    );

    // 步骤 4：项目真实模型路径（build_model + loss + backward，与引擎训练一致）
    let cfg = RunConfig::from_toml_str(include_str!("../../../configs/detect_smoke.toml"))
        .expect("加载 configs/detect_smoke.toml");
    let vs = VarStore::new(device);
    let model = build_model(&vs.root(), &cfg).expect("build_model");
    let n = 2i64;
    let x2 = Tensor::randn([n, 3, 64, 64], (Kind::Float, device));
    let batch = TrainBatch::Detect {
        boxes: vec![vec![[10.0, 10.0, 30.0, 30.0]]; n as usize],
        labels: vec![vec![1u32]; n as usize],
    };
    let loss2 = model.loss(&x2, &batch).expect("model.loss");
    loss2.backward();
    println!(
        "[step4 detect model loss+bwd] ok, loss={:.6}",
        loss2.double_value(&[])
    );

    println!("RESULT: cuda-ok（全部算子在 CUDA 上前向+反向通过）");
}
