//! 全框架统一错误类型（PLAN §3：AvError 枚举 Config/Shape/Io/Device/Train/Data）。

use thiserror::Error;

/// 库层统一结果别名。
pub type AvResult<T> = Result<T, AvError>;

#[derive(Debug, Error)]
pub enum AvError {
    #[error("配置错误: {0}")]
    Config(String),
    #[error("形状契约被违反: {0}")]
    Shape(String),
    #[error("设备错误: {0}")]
    Device(String),
    #[error("训练错误: {0}")]
    Train(String),
    #[error("数据错误: {0}")]
    Data(String),
    #[error("IO 错误: {0}")]
    Io(#[from] std::io::Error),
    #[error("TOML 解析错误: {0}")]
    Toml(#[from] toml::de::Error),
}

impl AvError {
    pub fn config(msg: impl Into<String>) -> Self {
        Self::Config(msg.into())
    }

    pub fn shape(msg: impl Into<String>) -> Self {
        Self::Shape(msg.into())
    }

    pub fn data(msg: impl Into<String>) -> Self {
        Self::Data(msg.into())
    }

    pub fn train(msg: impl Into<String>) -> Self {
        Self::Train(msg.into())
    }

    pub fn device(msg: impl Into<String>) -> Self {
        Self::Device(msg.into())
    }
}
