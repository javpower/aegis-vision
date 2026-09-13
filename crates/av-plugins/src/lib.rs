//! # av-plugins
//!
//! 插件机制与内置扩展（PLAN §6）。
//!
//! v1 静态注册：实现放本 crate（或用户自己的插件 crate），在 [`register_all`]
//! 里用 `av_core::av_plugin!` 宏登记，随二进制编译——零链接器黑魔法、可断点调试。
//! 动态库加载（libloading + C ABI）仅面向"不开源质检规则包分发"场景，列为 v2。

/// 登记本 crate 提供的全部插件（runtime 启动时调用）。
pub fn register_all() -> av_core::AvResult<()> {
    // M9 落地能力在此逐个登记：
    //   跨模态适配（Modality） / 骨干选型器（Backbone） / 拓扑规则引擎（Rule）…
    Ok(())
}

#[cfg(test)]
mod tests {
    use av_core::registry::{self, Category};

    use super::*;

    #[test]
    fn register_all_succeeds() {
        register_all().unwrap();
    }

    #[test]
    fn av_plugin_macro_registers_globally() {
        av_core::av_plugin!(Rule, "test-macro-rule");
        assert!(registry::lookup(Category::Rule, "test-macro-rule"));
    }
}
