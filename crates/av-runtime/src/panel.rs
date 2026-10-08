//! 观测面板（PLAN §7.1，axum 本地服务）：浏览 runs 目录、渲染各 run 的
//! 配置快照与指标报告，并经 SSE（`GET /sse/:id`）实时推送训练曲线。
//!
//! 设计原则（PLAN 同款）：**数据主权在文件**（runs/<id>/report.json +
//! config.snapshot.toml + metrics.jsonl），面板只是视图——面板挂了训练不丢数据。
//! 实时化（M8）：训练循环每 epoch 向 metrics.jsonl 追加一行，SSE 端点每 2 秒
//! 按字节偏移增量读取并推送给浏览器 EventSource，前端内联 SVG 画曲线。

use std::convert::Infallible;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use axum::extract::{Path as AxPath, State};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::Html;
use axum::routing::get;
use axum::{Json, Router};
use serde_json::Value;

use av_core::error::{AvError, AvResult};

type Ctx = Arc<PathBuf>;

/// SSE 轮询 metrics.jsonl 的间隔（epoch 粒度的曲线，2 秒足够实时）。
const SSE_POLL_INTERVAL: Duration = Duration::from_secs(2);

/// 启动面板（阻塞直至 Ctrl-C）。`runs_dir` 即配置的 output_dir。
pub async fn serve(runs_dir: PathBuf, port: u16) -> AvResult<()> {
    let app = router(runs_dir);
    let addr = format!("127.0.0.1:{port}");
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(|e| AvError::train(format!("面板端口 {port} 绑定失败: {e}")))?;
    println!("观测面板: http://{addr}（Ctrl-C 退出）");
    axum::serve(listener, app)
        .await
        .map_err(|e| AvError::train(format!("面板服务异常: {e}")))?;
    Ok(())
}

/// 面板路由（serve 与集成测试共用）。
pub fn router(runs_dir: PathBuf) -> Router {
    let ctx: Ctx = Arc::new(runs_dir);
    Router::new()
        .route("/", get(index))
        .route("/run/:id", get(run_detail))
        .route("/api/runs", get(api_runs))
        // SSE 实时指标流（EventSource 消费端在 run 详情页）
        .route("/sse/:id", get(sse_run))
        .with_state(ctx)
}

fn valid_id(id: &str) -> bool {
    // 防路径穿越：id 不允许包含分隔符或 ..
    !id.is_empty() && !id.contains("..") && !id.contains('/') && !id.contains('\\')
}

fn list_runs(runs_dir: &Path) -> Vec<(String, bool)> {
    // report.json（已完成）或 metrics.jsonl / config.snapshot.toml（训练中）任一存在即算 run
    let mut ids: Vec<(String, bool)> = std::fs::read_dir(runs_dir)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.join("report.json").is_file()
                || p.join(crate::metrics::METRICS_FILE).is_file()
                || p.join("config.snapshot.toml").is_file()
        })
        .filter_map(|p| p.file_name().and_then(|n| n.to_str().map(String::from)))
        .map(|id| {
            let done = runs_dir.join(&id).join("report.json").is_file();
            (id, done)
        })
        .collect();
    ids.sort();
    ids
}

/// 读 run 详情页所需的静态报告；训练中（report.json 尚未生成）返回 None。
fn read_report(runs_dir: &Path, id: &str) -> AvResult<Option<Value>> {
    let path = runs_dir.join(id).join("report.json");
    if !path.is_file() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(&path)
        .map_err(|e| AvError::data(format!("读取 {}: {e}", path.display())))?;
    serde_json::from_str(&text)
        .map(Some)
        .map_err(|e| AvError::data(format!("report.json 解析失败: {e}")))
}

async fn index(State(ctx): State<Ctx>) -> Html<String> {
    let runs = list_runs(&ctx);
    let rows = if runs.is_empty() {
        "<tr><td colspan=3>暂无训练记录——先跑一次 av train</td></tr>".into()
    } else {
        runs.iter()
            .map(|(id, done)| {
                let status = if *done { "已完成" } else { "<b style='color:#f5a623'>训练中</b>" };
                format!("<tr><td><a href=\"/run/{id}\">{id}</a></td><td>{id}</td><td>{status}</td></tr>")
            })
            .collect::<Vec<_>>()
            .join("")
    };
    Html(format!(
        "<!doctype html><meta charset=utf-8><meta http-equiv=refresh content=5><title>AV 面板</title>\
         <body style='font-family:system-ui;background:#16161a;color:#e8e6e3;padding:24px'>\
         <h1>AegisVision 训练面板</h1><p>每 5 秒自动刷新 · 实时曲线见各 run 详情页（SSE）· 数据源：{}</p>\
         <table border=1 cellpadding=6 style='border-collapse:collapse'>\
         <tr><th>run</th><th>id</th><th>状态</th></tr>{}</table></body>",
        ctx.display(),
        rows
    ))
}

async fn run_detail(State(ctx): State<Ctx>, AxPath(id): AxPath<String>) -> Html<String> {
    if !valid_id(&id) {
        return Html("非法 run id".into());
    }
    let mut rows = String::new();
    match read_report(&ctx, &id) {
        Ok(Some(report)) => {
            for key in ["task", "epochs", "final_loss", "metric", "metric_value"] {
                if let Some(v) = report.get(key) {
                    rows.push_str(&format!("<tr><td>{key}</td><td><b>{v}</b></td></tr>"));
                }
            }
            if let Some(sec) = report.get("secondary") {
                rows.push_str(&format!("<tr><td>secondary</td><td>{sec}</td></tr>"));
            }
        }
        Ok(None) => {
            rows.push_str(
                "<tr><td>report.json</td><td><b style='color:#f5a623'>训练中——尚未生成，\
                 曲线随下方 SSE 实时出现</b></td></tr>",
            );
        }
        Err(e) => return Html(format!("<meta charset=utf-8>读取失败: {e}")),
    }
    let snapshot = std::fs::read_to_string(ctx.join(&id).join("config.snapshot.toml"))
        .unwrap_or_else(|_| "(无快照)".into());
    Html(format!(
        "<!doctype html><meta charset=utf-8><title>AV run {id}</title>\
         <body style='font-family:system-ui;background:#16161a;color:#e8e6e3;padding:24px'>\
         <h1>run: {id}</h1>\
         <p><a href='/' style='color:#7ab'>&larr; 返回列表</a>｜本页由 SSE 实时更新，无需刷新</p>\
         <table border=1 cellpadding=6 style='border-collapse:collapse'>{rows}</table>\
         <h3>实时曲线（<code>/sse/{id}</code>，每 2 秒）</h3>\
         <p id=live-status style='color:#8a8f98'>连接中…</p>\
         <svg id=chart width=720 height=220 style='background:#0d0d10;border:1px solid #2a2a30'>\
         <text id=chart-empty x=360 y=110 fill=#5a5f68 text-anchor=middle>\
         等待 metrics.jsonl 行…（旧 run 无此文件则始终为空）</text></svg>\
         <p style='color:#8a8f98'>\
         <span style='color:#e5484d'>■</span> loss　\
         <span style='color:#46a758'>■</span> <span id=metric-name>metric</span></p>\
         <h3>metrics.jsonl 尾部</h3>\
         <pre id=log style='background:#0d0d10;padding:12px;max-height:240px;overflow:auto'></pre>\
         <h3>config.snapshot.toml</h3>\
         <pre style='background:#0d0d10;padding:12px'>{snapshot}</pre>\
         <script>
const eps=[], loss=[], met=[];
const statusEl=document.getElementById('live-status');
const logEl=document.getElementById('log');
const chart=document.getElementById('chart');
const fmt=v=>(v===null||v===undefined||isNaN(v))?'-':Number(v).toFixed(4);
function status(t){{statusEl.textContent=t;}}
function draw(){{
  const W=720,H=220;
  let s='<text x=8 y=14 fill=#5a5f68 font-size=11>epoch '+ (eps.length?eps[0]+' … '+eps[eps.length-1]:'-') +'（'+eps.length+' 行）</text>';
  const line=(a,color)=>{{
    const v=a.filter(x=>x!==null&&x!==undefined&&!isNaN(x));
    if(v.length<1)return '';
    const mn=Math.min(...v),mx=Math.max(...v),rg=(mx-mn)||1;
    return `<polyline fill=none stroke=${{color}} stroke-width=2 points=`+
      a.map((y,i)=>{{ if(y===null||y===undefined||isNaN(y))return '';
        const px=40+(i/Math.max(eps.length-1,1))*(W-60);
        const py=H-24-((y-mn)/rg)*(H-54);
        return `${{px.toFixed(1)}},${{py.toFixed(1)}}`;}}).filter(Boolean).join(' ')+
      ` stroke-linejoin=round/>`+
      `<text x=${{W-8}} y=18 fill=${{color}} font-size=11 text-anchor=end>${{mx.toFixed(3)}}</text>`+
      `<text x=${{W-8}} y=${{H-8}} fill=${{color}} font-size=11 text-anchor=end>${{mn.toFixed(3)}}</text>`;
  }};
  s+=line(loss,'#e5484d')+line(met,'#46a758');
  chart.innerHTML=s;
}}
const es=new EventSource('/sse/{id}');
es.onmessage=ev=>{{
  let d; try{{d=JSON.parse(ev.data);}}catch{{return;}}
  if(d.status==='waiting'){{status('等待 metrics.jsonl…（训练未开始，或为历史 run）');return;}}
  if(d.epoch===undefined)return;
  eps.push(d.epoch);loss.push(d.loss);met.push(d.metric_value??null);
  if(d.metric)document.getElementById('metric-name').textContent=d.metric;
  logEl.textContent+=JSON.stringify(d)+'\\n';logEl.scrollTop=logEl.scrollHeight;
  draw();
  status(`epoch ${{d.epoch}} · loss ${{fmt(d.loss)}} · ${{d.metric??''}} ${{fmt(d.metric_value)}}`+
    (d.secondary?` · ${{d.secondary[0]}} ${{fmt(d.secondary[1])}}`:''));
}};
es.onerror=()=>status('SSE 断开——浏览器将自动重连…');
         </script></body>"
    ))
}

async fn api_runs(State(ctx): State<Ctx>) -> Json<Value> {
    Json(
        serde_json::json!({ "runs": list_runs(&ctx).into_iter().map(|(id, _)| id).collect::<Vec<_>>() }),
    )
}

// ---------------------------------------------------------------------------
// SSE 实时指标流
// ---------------------------------------------------------------------------

/// mpsc → Stream 适配（避免引入 tokio-stream / futures 运行时依赖；只显式
/// 声明已在依赖树内的 futures-core trait 包）。
struct EventStream(tokio::sync::mpsc::Receiver<Result<Event, Infallible>>);

impl futures_core::Stream for EventStream {
    type Item = Result<Event, Infallible>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.get_mut().0.poll_recv(cx)
    }
}

/// `GET /sse/:id`：每 [`SSE_POLL_INTERVAL`] 按字节偏移增量读
/// `runs/<id>/metrics.jsonl`，每个新行作为一条 `data:` 事件推送。
///
/// - 连接建立即推已有全部行（前端刷新页面不丢历史）；
/// - 文件尚未出现 → 先推 `{"status":"waiting"}`，出现后自动开始推；
/// - 客户端断开 → 发送端 `send` 失败 → 任务退出，无泄漏。
async fn sse_run(
    State(ctx): State<Ctx>,
    AxPath(id): AxPath<String>,
) -> Sse<impl futures_core::Stream<Item = Result<Event, Infallible>>> {
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(64);
    if !valid_id(&id) {
        // 非法 id：立即关通道，返回空流（客户端收到空 event-stream 后断开）
        drop(tx);
        return Sse::new(EventStream(rx)).keep_alive(KeepAlive::default());
    }
    let run_dir = ctx.join(&id);
    tokio::spawn(async move {
        let mut offset = 0u64;
        let mut announced_waiting = false;
        let mut ticker = tokio::time::interval(SSE_POLL_INTERVAL);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            ticker.tick().await;
            match crate::metrics::read_from(&run_dir, offset) {
                Ok((new_offset, rows)) => {
                    offset = new_offset;
                    for row in rows {
                        let ev = Event::default().data(row.to_string());
                        if tx.send(Ok(ev)).await.is_err() {
                            return; // 客户端已断开
                        }
                    }
                    if offset == 0 && !announced_waiting {
                        announced_waiting = true;
                        let ev = Event::default().data("{\"status\":\"waiting\"}");
                        if tx.send(Ok(ev)).await.is_err() {
                            return;
                        }
                    }
                }
                Err(e) => {
                    tracing::debug!("SSE 读 metrics 失败（下轮重试）: {e}");
                }
            }
        }
    });
    Sse::new(EventStream(rx)).keep_alive(KeepAlive::default())
}
