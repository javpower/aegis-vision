//! 插件注册表（v1 静态注册，PLAN §6.0）。
//!
//! 不依赖 inventory / 链接器段技巧；插件 crate 在装配函数里通过
//! [`crate::av_plugin!`] 宏或 [`register`] 登记，运行时按注册名查找。

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use crate::error::{AvError, AvResult};

/// 插件类别（与配置里的注册名一一对应）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Category {
    /// 跨模态适配（RGB/深度/红外/X 光/点云投影）
    Modality,
    /// 骨干
    Backbone,
    /// 特征 Neck
    Neck,
    /// 任务头
    Head,
    /// 损失
    Loss,
    /// 后处理
    PostProc,
    /// 拓扑规则
    Rule,
}

impl Category {
    pub fn as_str(self) -> &'static str {
        match self {
            Category::Modality => "modality",
            Category::Backbone => "backbone",
            Category::Neck => "neck",
            Category::Head => "head",
            Category::Loss => "loss",
            Category::PostProc => "postproc",
            Category::Rule => "rule",
        }
    }
}

/// 注册表本体：按类别分桶的注册名列表。
#[derive(Debug, Default)]
pub struct Registry {
    entries: HashMap<Category, Vec<String>>,
}

impl Registry {
    /// 注册；重名返回错误（PLAN §6.0：注册冲突尽早暴露）。
    pub fn register(&mut self, cat: Category, name: &str) -> AvResult<()> {
        let bucket = self.entries.entry(cat).or_default();
        if bucket.iter().any(|n| n == name) {
            return Err(AvError::config(format!(
                "插件重复注册: {}/{name}",
                cat.as_str()
            )));
        }
        bucket.push(name.to_string());
        Ok(())
    }

    pub fn lookup(&self, cat: Category, name: &str) -> bool {
        self.entries
            .get(&cat)
            .is_some_and(|b| b.iter().any(|n| n == name))
    }

    pub fn list(&self, cat: Category) -> Vec<String> {
        self.entries.get(&cat).cloned().unwrap_or_default()
    }
}

static GLOBAL: OnceLock<Mutex<Registry>> = OnceLock::new();

fn global() -> &'static Mutex<Registry> {
    GLOBAL.get_or_init(|| Mutex::new(Registry::default()))
}

/// 往全局注册表登记一个插件名。
pub fn register(cat: Category, name: &str) -> AvResult<()> {
    global()
        .lock()
        .expect("registry 互斥锁中毒")
        .register(cat, name)
}

/// 查询某个注册名是否存在。
pub fn lookup(cat: Category, name: &str) -> bool {
    global()
        .lock()
        .expect("registry 互斥锁中毒")
        .lookup(cat, name)
}

/// 列出某类别全部注册名（按注册顺序）。
pub fn list(cat: Category) -> Vec<String> {
    global().lock().expect("registry 互斥锁中毒").list(cat)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_lookup_list() {
        register(Category::Backbone, "csp-elan").unwrap();
        assert!(lookup(Category::Backbone, "csp-elan"));
        assert!(!lookup(Category::Backbone, "vit-hybrid"));
        assert_eq!(list(Category::Backbone), vec!["csp-elan".to_string()]);
    }

    #[test]
    fn duplicate_registration_errors() {
        register(Category::Loss, "ciou").unwrap();
        let err = register(Category::Loss, "ciou").unwrap_err();
        assert!(err.to_string().contains("重复注册"));
    }
}
