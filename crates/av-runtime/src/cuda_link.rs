//! Windows/MSVC CUDA 链接补丁（tch 0.24 + libtorch cu128 实测所需，docs/gpu.md 实验二）。
//!
//! 背景：torch-sys ≥ 0.20 移除了「引用 torch_cuda 符号」的 dummy 依赖
//! （build.rs 注释称其 "not necessary anymore"——该结论只对 ELF 成立）。
//! MSVC 链接器不会链接无符号引用的导入库，于是 torch_cuda.dll 不被加载、
//! 其静态注册器不运行，ATen dispatcher 里没有 CUDA 后端——任何 CUDA 算子报
//! `Could not run 'aten::*' with arguments from the 'CUDA' backend`，
//! `Cuda::is_available()` 恒为 false，引擎会静默回退 CPU。
//!
//! 处理：请求 CUDA 时先用 `LoadLibraryA("torch_cuda.dll")` 强制加载，
//! 其静态注册器随即向 dispatcher 注册全部 CUDA 内核。幂等且开销可忽略
//! （重复调用只递增模块引用计数）；加载失败（CPU 版 libtorch / 无 GPU /
//! DLL 缺失）返回 false，由 [`crate::engine::resolve_device`] 的既有
//! 探测 + 回退逻辑继续兜底，行为与旧版一致。

#[cfg(windows)]
pub fn ensure_torch_cuda_loaded() -> bool {
    #[link(name = "kernel32")]
    extern "system" {
        fn LoadLibraryA(name: *const u8) -> *mut std::ffi::c_void;
    }
    let ok = unsafe { !LoadLibraryA(b"torch_cuda.dll\0".as_ptr()).is_null() };
    if !ok {
        tracing::debug!(
            "torch_cuda.dll 加载失败（CPU 版 libtorch 或非 CUDA 环境），按 CUDA 不可用处理"
        );
    }
    ok
}

#[cfg(not(windows))]
pub fn ensure_torch_cuda_loaded() -> bool {
    // 非 Windows（ELF/Mach-O）：链接进导入表的动态库不依赖符号引用，
    // torch_cuda 随链接自动加载，无需补丁。
    true
}
