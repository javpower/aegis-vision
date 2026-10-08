//! 强类型配置 schema（PLAN §2.4 / 附录 A）。
//!
//! - 全部容器 `deny_unknown_fields`：字段拼错在加载期报错（serde 已知限制：
//!   内部 tag 枚举 [`TaskCfg`] 无法做 unknown 拒绝，由 [`TaskCfg::validate`] 补语义校验）。
//! - 三层合并中的「模板层」即 `configs/*.toml`；CLI `--override` 优先级最高。

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::conventions::AngleDomain;
use crate::error::{AvError, AvResult};

// ---------------------------------------------------------------------------
// 顶层
// ---------------------------------------------------------------------------

/// 一次训练/推理运行的总配置（configs/*.toml 的直接映射）。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct RunConfig {
    pub run_id: String,
    pub seed: u64,
    pub device: String,
    pub output_dir: PathBuf,
    pub model: ModelConfig,
    pub data: DataConfig,
    pub train: TrainConfig,
    pub eval: EvalConfig,
    pub panel: PanelConfig,
    pub pretrain: PretrainCfg,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            run_id: String::new(),
            seed: 42,
            device: "cuda:0".into(),
            output_dir: PathBuf::from("runs"),
            model: ModelConfig::default(),
            data: DataConfig::default(),
            train: TrainConfig::default(),
            eval: EvalConfig::default(),
            panel: PanelConfig::default(),
            pretrain: PretrainCfg::default(),
        }
    }
}

impl RunConfig {
    /// 解析 + 校验（加载期拒绝非法配置，PLAN §2.4）。
    pub fn from_toml_str(s: &str) -> AvResult<Self> {
        let cfg: Self = toml::from_str(s)?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn from_path(p: &Path) -> AvResult<Self> {
        let s = std::fs::read_to_string(p)?;
        Self::from_toml_str(&s)
    }

    /// 语义校验：serde 管不到的取值范围/组合约束都在这里。
    pub fn validate(&self) -> AvResult<()> {
        if self.model.tasks.is_empty() {
            return Err(AvError::config("model.tasks 不能为空"));
        }
        for t in &self.model.tasks {
            t.validate()?;
        }

        if self.train.epochs == 0 {
            return Err(AvError::config("train.epochs 必须 >= 1"));
        }
        if self.train.batch_size == 0 {
            return Err(AvError::config("train.batch_size 必须 >= 1"));
        }
        if self.train.accumulate_steps == 0 {
            return Err(AvError::config("train.accumulate_steps 必须 >= 1"));
        }
        if self.train.optimizer.lr <= 0.0 {
            return Err(AvError::config("train.optimizer.lr 必须 > 0"));
        }
        if !(0.0 < self.train.scheduler.lr_min_factor && self.train.scheduler.lr_min_factor <= 1.0)
        {
            return Err(AvError::config(
                "train.scheduler.lr_min_factor 需在 (0, 1] 区间",
            ));
        }
        if !(0.0 < self.train.ema_decay && self.train.ema_decay < 1.0) {
            return Err(AvError::config("train.ema_decay 需在 (0, 1) 区间"));
        }
        if self.train.warmup_epochs > self.train.epochs as f32 {
            return Err(AvError::config("train.warmup_epochs 不能超过 epochs"));
        }

        if self.data.workers == 0 || self.data.prefetch == 0 {
            return Err(AvError::config("data.workers / data.prefetch 必须 >= 1"));
        }
        if self.data.pipeline != DataPipeline::Synthetic {
            let src = &self.data.sources.train;
            match self.data.pipeline {
                DataPipeline::Dir => {
                    // 分类任务可用 classify.data_dir 指定 ImageFolder 根目录，
                    // 与 data.sources.train.dir 二选一
                    let classify_has_dir = matches!(
                        self.model.tasks.first(),
                        Some(TaskCfg::Classify(c)) if c.data_dir.is_some()
                    );
                    if src.dir.is_none() && !classify_has_dir {
                        return Err(AvError::config(
                            "data.pipeline = \"dir\" 需要指定 data.sources.train.dir",
                        ));
                    }
                }
                DataPipeline::AvPack => {
                    if src.avpack.is_none() {
                        return Err(AvError::config(
                            "data.pipeline = \"avpack\" 需要指定 data.sources.train.avpack",
                        ));
                    }
                }
                DataPipeline::Synthetic => {}
            }
        }
        for st in self
            .data
            .sources
            .train
            .tasks
            .iter()
            .chain(&self.data.sources.val.tasks)
        {
            if let Some([lo, hi]) = st.augment.scale_jitter {
                if lo <= 0.0 || lo > hi {
                    return Err(AvError::config(
                        "augment.scale_jitter 需满足 0 < min <= max",
                    ));
                }
            }
        }

        if self.panel.port == 0 {
            return Err(AvError::config("panel.port 必须 >= 1"));
        }

        if self.pretrain.enable && self.pretrain.weight_path.is_none() {
            return Err(AvError::config(
                "pretrain.enable = true 需要指定 pretrain.weight_path",
            ));
        }
        Ok(())
    }

    /// 生效 run_id：显式指定则用之；否则按配置快照 + 时间戳 blake3 生成（PLAN §7.3）。
    pub fn effective_run_id(&self) -> String {
        if !self.run_id.is_empty() {
            return self.run_id.clone();
        }
        let snapshot = self.snapshot_toml().unwrap_or_default();
        let mut h = blake3::Hasher::new();
        h.update(snapshot.as_bytes());
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        h.update(&ts.to_le_bytes());
        let hex = h.finalize().to_hex().to_string();
        hex[..12].to_string()
    }

    /// 最终生效配置的 TOML 快照（训练启动时写入 runs/<run_id>/config.snapshot.toml）。
    pub fn snapshot_toml(&self) -> AvResult<String> {
        toml::to_string_pretty(self).map_err(|e| AvError::config(format!("配置序列化失败: {e}")))
    }
}

// ---------------------------------------------------------------------------
// model
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct ModelConfig {
    pub backbone: BackboneCfg,
    pub neck: NeckCfg,
    pub tasks: Vec<TaskCfg>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            backbone: BackboneCfg::default(),
            neck: NeckCfg::default(),
            tasks: vec![TaskCfg::Detect(DetectCfg::default())],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct BackboneCfg {
    pub family: String,
    pub depth: f32,
    pub width: f32,
    pub pretrained: String,
    /// 输入归一化域：false（默认）= [0,1] RGB（合成数据与 non-pretrain 路径的
    /// 历史行为）；true = ImageNet mean/std 归一化（(x/255 − mean)/std，
    /// mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225]）。ImageNet 预训练骨干
    /// （BN running 统计量在 ImageNet 域）必须开启，否则域失配削弱预训练效果
    /// （预训练 A/B 矩阵第一期根因 #3，见 BENCHMARK.md）。训练/评测/推理共用，
    /// 由引擎读本字段透传给数据管线。
    pub imagenet_norm: bool,
}

impl Default for BackboneCfg {
    fn default() -> Self {
        Self {
            family: "csp-elan".into(),
            depth: 1.0,
            width: 1.0,
            pretrained: "none".into(),
            imagenet_norm: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct NeckCfg {
    #[serde(rename = "type")]
    pub kind: String,
    pub channels: Vec<usize>,
}

impl Default for NeckCfg {
    fn default() -> Self {
        Self {
            kind: "pan".into(),
            channels: vec![128, 256, 512],
        }
    }
}

/// 任务配置（内部 tag：`kind = "detect"` / `"obb"` / `"seg"` / `"keypoint"` / `"classify"`）。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TaskCfg {
    Detect(DetectCfg),
    Obb(ObbCfg),
    Seg(SegCfg),
    Keypoint(KeypointCfg),
    Classify(ClassifyCfg),
}

impl TaskCfg {
    pub fn kind_name(&self) -> &'static str {
        match self {
            TaskCfg::Detect(_) => "detect",
            TaskCfg::Obb(_) => "obb",
            TaskCfg::Seg(_) => "seg",
            TaskCfg::Keypoint(_) => "keypoint",
            TaskCfg::Classify(_) => "classify",
        }
    }

    pub fn loss_weight(&self) -> f32 {
        match self {
            TaskCfg::Detect(c) => c.loss_weight,
            TaskCfg::Obb(c) => c.loss_weight,
            TaskCfg::Seg(c) => c.loss_weight,
            TaskCfg::Keypoint(c) => c.loss_weight,
            TaskCfg::Classify(c) => c.loss_weight,
        }
    }

    pub fn validate(&self) -> AvResult<()> {
        match self {
            TaskCfg::Detect(c) => {
                check_one_of("detect.head", &c.head, &["yolo", "rfdetr"])?;
                check_img_size(c.img_size)?;
                if c.num_classes == 0 {
                    return Err(AvError::config("detect.num_classes 必须 >= 1"));
                }
                for (name, w) in [
                    ("loss_cls_weight", c.loss_cls_weight),
                    ("loss_ciou_weight", c.loss_ciou_weight),
                    ("loss_dfl_weight", c.loss_dfl_weight),
                ] {
                    if !(w >= 0.0 && w.is_finite()) {
                        return Err(AvError::config(format!("detect.{name} 必须 >= 0 且有限")));
                    }
                }
                check_weight(c.loss_weight)
            }
            TaskCfg::Obb(c) => {
                check_one_of("obb.head", &c.head, &["yolo", "rfdetr"])?;
                check_one_of("obb.rot_nms", &c.rot_nms, &["probiou", "polygon"])?;
                check_weight(c.loss_weight)
            }
            TaskCfg::Seg(c) => {
                check_one_of("seg.head", &c.head, &["yolact", "direct"])?;
                check_img_size(c.img_size)?;
                if c.num_classes == 0 {
                    return Err(AvError::config("seg.num_classes 必须 >= 1"));
                }
                if c.num_protos == 0 {
                    return Err(AvError::config("seg.num_protos 必须 >= 1"));
                }
                for (name, w) in [
                    ("loss_bce_weight", c.loss_bce_weight),
                    ("loss_dice_weight", c.loss_dice_weight),
                ] {
                    if !(w >= 0.0 && w.is_finite()) {
                        return Err(AvError::config(format!("seg.{name} 必须 >= 0 且有限")));
                    }
                }
                check_weight(c.loss_weight)
            }
            TaskCfg::Keypoint(c) => {
                check_one_of(
                    "keypoint.decode",
                    &c.decode,
                    &["heatmap", "simdr", "direct"],
                )?;
                if c.num_keypoints == 0 {
                    return Err(AvError::config("keypoint.num_keypoints 必须 >= 1"));
                }
                check_img_size(c.img_size)?;
                if !(c.loss_oks_weight >= 0.0 && c.loss_oks_weight.is_finite()) {
                    return Err(AvError::config("keypoint.loss_oks_weight 必须 >= 0 且有限"));
                }
                check_weight(c.loss_weight)
            }
            TaskCfg::Classify(c) => {
                if c.num_classes < 2 {
                    return Err(AvError::config("classify.num_classes 必须 >= 2"));
                }
                check_img_size(c.img_size)?;
                check_weight(c.loss_weight)
            }
        }
    }
}

fn check_img_size(s: u32) -> AvResult<()> {
    if s < 32 || !s.is_multiple_of(32) {
        return Err(AvError::config("img_size 必须 >= 32 且为 32 的倍数"));
    }
    Ok(())
}

fn check_one_of(field: &str, got: &str, allowed: &[&str]) -> AvResult<()> {
    if allowed.contains(&got) {
        Ok(())
    } else {
        Err(AvError::config(format!(
            "{field} = \"{got}\" 非法，允许值: {allowed:?}"
        )))
    }
}

fn check_weight(w: f32) -> AvResult<()> {
    if !(w >= 0.0 && w.is_finite()) {
        return Err(AvError::config("loss_weight 必须 >= 0 且有限"));
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct DetectCfg {
    /// yolo | rfdetr
    pub head: String,
    pub loss_weight: f32,
    pub assigner: String,
    pub obb_mode: bool,
    /// 类别数（COCO=80，自定义场景按需）
    pub num_classes: usize,
    /// 合成源/推理输入边长（32 的倍数）
    pub img_size: u32,
    /// TAL 路径分类分支损失权重
    pub loss_cls_weight: f32,
    /// TAL 路径 CIoU 回归损失权重
    pub loss_ciou_weight: f32,
    /// TAL 路径 DFL 分布损失权重
    pub loss_dfl_weight: f32,
    /// 检测头特征层级（stride，升序）。默认 [8, 16]；工业小缺陷场景配置
    /// [4, 8, 16] 启用 P2（stride 4）高分辨率层。取值须为骨干支持的
    /// stride（simple-cnn：4/8/16），模型装配期校验。
    pub head_levels: Vec<u32>,
}

impl Default for DetectCfg {
    fn default() -> Self {
        Self {
            head: "yolo".into(),
            loss_weight: 1.0,
            assigner: "tal".into(),
            obb_mode: false,
            num_classes: 80,
            img_size: 64,
            loss_cls_weight: 1.0,
            loss_ciou_weight: 5.0,
            loss_dfl_weight: 1.5,
            head_levels: vec![8, 16],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ObbCfg {
    pub head: String,
    pub loss_weight: f32,
    pub assigner: String,
    pub angle: AngleDomain,
    /// probiou | polygon
    pub rot_nms: String,
}

impl Default for ObbCfg {
    fn default() -> Self {
        Self {
            head: "yolo".into(),
            loss_weight: 1.0,
            assigner: "tal".into(),
            angle: AngleDomain::default(),
            rot_nms: "probiou".into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct SegCfg {
    /// yolact（实时档）| direct（精度档）
    pub head: String,
    pub loss_weight: f32,
    /// 类别数（COCO=80，coco8-seg 标注沿用原类 id）
    pub num_classes: usize,
    /// 合成源/推理输入边长（32 的倍数）
    pub img_size: u32,
    /// 原型掩码数（YOLACT K）
    pub num_protos: usize,
    /// 掩码 BCE 损失权重（低分辨率掩码监督）
    pub loss_bce_weight: f32,
    /// 掩码 Dice 损失权重
    pub loss_dice_weight: f32,
}

impl Default for SegCfg {
    fn default() -> Self {
        Self {
            head: "yolact".into(),
            loss_weight: 1.0,
            num_classes: 80,
            img_size: 64,
            num_protos: 32,
            loss_bce_weight: 1.0,
            loss_dice_weight: 1.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct KeypointCfg {
    pub num_keypoints: usize,
    /// heatmap（精度档）| simdr（实时档）| direct（直接回归档，当前唯一落地实现）
    pub decode: String,
    pub loss_weight: f32,
    /// 合成源/推理输入边长（32 的倍数）
    pub img_size: u32,
    /// OKS 损失项权重（1 − mean OKS，对回归坐标可微；0 = 纯 L1 偏移回归）
    pub loss_oks_weight: f32,
}

impl Default for KeypointCfg {
    fn default() -> Self {
        Self {
            num_keypoints: 17,
            decode: "heatmap".into(),
            loss_weight: 1.0,
            img_size: 64,
            loss_oks_weight: 1.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ClassifyCfg {
    pub num_classes: usize,
    pub loss_weight: f32,
    /// 合成源/推理输入边长（32 的倍数）
    pub img_size: u32,
    /// ImageFolder 数据集根目录（`data.pipeline = "dir"` 分类任务的可选覆写；
    /// 缺省回落 `data.sources.train.dir`，与检测侧同用 sources 结构，
    /// split 约定：train 源缺省 "train"，val 源缺省 "val"）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_dir: Option<PathBuf>,
}

impl Default for ClassifyCfg {
    fn default() -> Self {
        Self {
            num_classes: 1000,
            loss_weight: 1.0,
            img_size: 64,
            data_dir: None,
        }
    }
}

// ---------------------------------------------------------------------------
// data
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct DataConfig {
    pub pipeline: DataPipeline,
    pub workers: u32,
    pub prefetch: u32,
    /// 训练数据缓存层（seg/检测增强路径）：`auto`（默认，估算能装进显存则
    /// 显存驻留，否则内存缓存）/ `gpu`（显存驻留，增强在 GPU 张量域）/
    /// `ram`（内存缓存 letterbox 内容贴片，rayon 并行增强）/ `off`
    /// （历史路径：raw 全分辨率每 epoch 重编码）。
    pub cache: String,
    pub sources: DataSources,
}

impl Default for DataConfig {
    fn default() -> Self {
        Self {
            pipeline: DataPipeline::Dir,
            workers: 8,
            prefetch: 4,
            cache: "auto".into(),
            sources: DataSources::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataPipeline {
    #[serde(rename = "avpack")]
    AvPack,
    #[serde(rename = "dir")]
    Dir,
    /// v0.1 内置合成数据源：随机噪声 + 类相关图案（分类）/ 随机方块（检测），
    /// 开箱即训、不需要外部数据集，用于端到端冒烟与集成测试。
    #[serde(rename = "synthetic")]
    Synthetic,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct DataSources {
    pub train: DataSourceCfg,
    pub val: DataSourceCfg,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default, deny_unknown_fields)]
pub struct DataSourceCfg {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub avpack: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dir: Option<PathBuf>,
    pub tasks: Vec<SourceTaskCfg>,
    /// 数据集子目录名（YOLO 约定 images/<split>）；train 源缺省 "train"，val 源缺省 "val"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub split: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SourceTaskCfg {
    pub kind: TaskKind,
    /// dota | coco | yolo_txt | imagenet_folder …
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
    #[serde(default)]
    pub augment: AugmentCfg,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskKind {
    Detect,
    Obb,
    Seg,
    Keypoint,
    Classify,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct AugmentCfg {
    pub mosaic: f32,
    pub mixup: f32,
    pub hsv: [f32; 3],
    pub scale_jitter: Option<[f32; 2]>,
    pub close_last_epochs: u32,
    /// 水平翻转概率 p ∈ [0,1]（0 = 关闭）。翻转时框 / 关键点 / 掩码坐标同步镜像，
    /// COCO 17 点模板还会交换左右对称关键点索引（见 av-tasks::augment）。
    pub flip: f32,
}

impl Default for AugmentCfg {
    fn default() -> Self {
        Self {
            mosaic: 0.0,
            mixup: 0.0,
            hsv: [0.0; 3],
            scale_jitter: None,
            close_last_epochs: 0,
            flip: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// train / eval / panel
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct TrainConfig {
    pub epochs: u32,
    pub batch_size: u32,
    pub accumulate_steps: u32,
    pub optimizer: OptimizerCfg,
    pub warmup_epochs: f32,
    pub scheduler: SchedulerCfg,
    pub amp: bool,
    pub deterministic: bool,
    pub ema_decay: f32,
    pub grad_clip: f32,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            epochs: 300,
            batch_size: 16,
            accumulate_steps: 1,
            optimizer: OptimizerCfg::default(),
            warmup_epochs: 3.0,
            scheduler: SchedulerCfg::default(),
            amp: true,
            deterministic: false,
            ema_decay: 0.9999,
            grad_clip: 10.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct OptimizerCfg {
    #[serde(rename = "type")]
    pub kind: OptimizerKind,
    pub lr: f32,
    pub weight_decay: f32,
}

impl Default for OptimizerCfg {
    fn default() -> Self {
        Self {
            kind: OptimizerKind::AdamW,
            lr: 1e-3,
            weight_decay: 5e-4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OptimizerKind {
    AdamW,
    Sgd,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct SchedulerCfg {
    #[serde(rename = "type")]
    pub kind: SchedulerKind,
    pub lr_min_factor: f32,
}

impl Default for SchedulerCfg {
    fn default() -> Self {
        Self {
            kind: SchedulerKind::Cosine,
            lr_min_factor: 0.01,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SchedulerKind {
    Cosine,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct EvalConfig {
    pub interval_epochs: u32,
    pub protocols: Vec<String>,
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            interval_epochs: 10,
            protocols: vec!["coco-map".into()],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct PanelConfig {
    pub enabled: bool,
    pub port: u16,
}

impl Default for PanelConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            port: 8080,
        }
    }
}

// ---------------------------------------------------------------------------
// pretrain（预训练权重导入：预训练权重方案第一层）
// ---------------------------------------------------------------------------

/// [pretrain] 段：外部 PyTorch 预训练权重 / 原生 avpretrain 目录的导入与冻结。
/// 引擎侧在 build_model 之后、优化器 build 之前消费（av-runtime::engine）。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct PretrainCfg {
    /// 是否启用预训练权重导入（默认 false：不配置时既有训练/推理链路零变化）
    pub enable: bool,
    /// 权重来源：.safetensors 文件（外部 PyTorch 导出）或 avpretrain 权重目录
    /// （含 manifest.json；训练产出的 best.ckpt/ 目录补 manifest 即是）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weight_path: Option<PathBuf>,
    /// 只加载名字含 "backbone" 的目标变量（默认 true：头通常需要按任务重训）
    pub load_only_backbone: bool,
    /// 加载后冻结 backbone（requires_grad = false，优化器按 trainable_variables
    /// 自然跳过；默认 false）
    pub freeze_backbone: bool,
    /// 层映射 TOML（[[entries]] from/to/transpose）：把外部层名改写到 AV 变量名
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layer_map: Option<PathBuf>,
}

impl Default for PretrainCfg {
    fn default() -> Self {
        Self {
            enable: false,
            weight_path: None,
            load_only_backbone: true,
            freeze_backbone: false,
            layer_map: None,
        }
    }
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL: &str = "[data.sources.train]\ndir = \"data/train\"\n";
    const APPENDIX_A: &str = include_str!("../../../configs/detect_obb.toml");

    #[test]
    fn minimal_toml_fills_defaults() {
        let cfg = RunConfig::from_toml_str(MINIMAL).unwrap();
        assert_eq!(cfg.train.batch_size, 16);
        assert_eq!(cfg.train.epochs, 300);
        assert!(cfg.train.amp);
        assert_eq!(cfg.model.tasks.len(), 1);
        assert_eq!(cfg.model.tasks[0].kind_name(), "detect");
    }

    #[test]
    fn parse_appendix_a_config() {
        let cfg = RunConfig::from_toml_str(APPENDIX_A).unwrap();
        assert_eq!(cfg.train.epochs, 300);
        assert_eq!(cfg.model.tasks.len(), 2);
        assert_eq!(cfg.model.tasks[0].kind_name(), "detect");
        assert_eq!(cfg.model.tasks[1].kind_name(), "obb");
        assert_eq!(cfg.data.pipeline, DataPipeline::AvPack);
        assert_eq!(cfg.data.sources.train.tasks[0].augment.mosaic, 1.0);
        assert_eq!(cfg.model.tasks[1].loss_weight(), 1.0);
    }

    #[test]
    fn deny_unknown_top_level_key() {
        let err = RunConfig::from_toml_str("run_idd = \"x\"\n").unwrap_err();
        assert!(err.to_string().contains("unknown field"), "got: {err}");
    }

    #[test]
    fn validate_rejects_zero_epochs() {
        let t = format!("{MINIMAL}[train]\nepochs = 0\n");
        let err = RunConfig::from_toml_str(&t).unwrap_err();
        assert!(err.to_string().contains("epochs"), "got: {err}");
    }

    #[test]
    fn validate_rejects_missing_data_source() {
        let err = RunConfig::from_toml_str("").unwrap_err();
        assert!(err.to_string().contains("data.sources.train"), "got: {err}");
    }

    #[test]
    fn validate_rejects_bad_scale_jitter() {
        let t = concat!(
            "[data.sources.train]\n",
            "dir = \"d\"\n",
            "[[data.sources.train.tasks]]\n",
            "kind = \"detect\"\n",
            "[data.sources.train.tasks.augment]\n",
            "scale_jitter = [2.0, 0.5]\n"
        );
        let err = RunConfig::from_toml_str(t).unwrap_err();
        assert!(err.to_string().contains("scale_jitter"), "got: {err}");
    }

    #[test]
    fn snapshot_roundtrip() {
        let cfg = RunConfig::from_toml_str(APPENDIX_A).unwrap();
        let snap = cfg.snapshot_toml().unwrap();
        let cfg2 = RunConfig::from_toml_str(&snap).unwrap();
        assert_eq!(cfg, cfg2);
    }

    #[test]
    fn effective_run_id_stable_when_explicit() {
        let mut cfg = RunConfig::from_toml_str(MINIMAL).unwrap();
        cfg.run_id = "fixed-run".into();
        assert_eq!(cfg.effective_run_id(), "fixed-run");
    }

    #[test]
    fn effective_run_id_generated_when_empty() {
        let cfg = RunConfig::from_toml_str(MINIMAL).unwrap();
        assert_eq!(cfg.effective_run_id().len(), 12);
    }

    #[test]
    fn classify_data_dir_satisfies_dir_pipeline() {
        // 分类任务用 classify.data_dir 时可不配 data.sources.train.dir
        let t = concat!(
            "[[model.tasks]]\n",
            "kind = \"classify\"\n",
            "num_classes = 10\n",
            "img_size = 64\n",
            "data_dir = \"data/imagenette2-160\"\n",
            "[data]\n",
            "pipeline = \"dir\"\n",
        );
        let cfg = RunConfig::from_toml_str(t).unwrap();
        assert_eq!(
            cfg.model.tasks[0],
            TaskCfg::Classify(ClassifyCfg {
                num_classes: 10,
                img_size: 64,
                data_dir: Some(PathBuf::from("data/imagenette2-160")),
                ..ClassifyCfg::default()
            })
        );
        // 快照回读稳定（Option 字段 skip_serializing_if 不破坏往返）
        let snap = cfg.snapshot_toml().unwrap();
        assert_eq!(RunConfig::from_toml_str(&snap).unwrap(), cfg);

        // 两者都没有 → 仍拒绝
        let t2 = concat!(
            "[[model.tasks]]\n",
            "kind = \"classify\"\n",
            "num_classes = 10\n",
            "[data]\n",
            "pipeline = \"dir\"\n",
        );
        let err = RunConfig::from_toml_str(t2).unwrap_err();
        assert!(err.to_string().contains("data.sources.train"), "got: {err}");
    }

    #[test]
    fn pretrain_defaults_when_absent() {
        let cfg = RunConfig::from_toml_str(MINIMAL).unwrap();
        assert!(!cfg.pretrain.enable);
        assert!(cfg.pretrain.weight_path.is_none());
        assert!(cfg.pretrain.load_only_backbone);
        assert!(!cfg.pretrain.freeze_backbone);
        assert!(cfg.pretrain.layer_map.is_none());
        // 快照回读稳定（新段参与快照序列化）
        let snap = cfg.snapshot_toml().unwrap();
        assert_eq!(RunConfig::from_toml_str(&snap).unwrap(), cfg);
    }

    #[test]
    fn pretrain_enable_requires_weight_path() {
        let t = format!("{MINIMAL}[pretrain]\nenable = true\n");
        let err = RunConfig::from_toml_str(&t).unwrap_err();
        assert!(err.to_string().contains("weight_path"), "got: {err}");

        let t = format!("{MINIMAL}[pretrain]\nenable = true\nweight_path = \"w.safetensors\"\n");
        let cfg = RunConfig::from_toml_str(&t).unwrap();
        assert_eq!(
            cfg.pretrain.weight_path.as_deref(),
            Some(Path::new("w.safetensors"))
        );
    }

    #[test]
    fn pretrain_rejects_unknown_field() {
        let t = format!("{MINIMAL}[pretrain]\nenabled = true\n");
        let err = RunConfig::from_toml_str(&t).unwrap_err();
        assert!(err.to_string().contains("unknown field"), "got: {err}");
    }

    #[test]
    fn pretrain_demo_config_is_valid() {
        let cfg = RunConfig::from_toml_str(include_str!("../../../configs/pretrain_demo.toml"))
            .expect("pretrain_demo 示例配置必须合法");
        assert!(cfg.pretrain.enable);
        assert!(cfg.pretrain.weight_path.is_some());
        assert!(cfg.pretrain.load_only_backbone);
        assert!(!cfg.pretrain.freeze_backbone);
    }
}
