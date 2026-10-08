//! 面板 SSE 实时流集成测试：真起 axum 服务（127.0.0.1 随机端口），裸 TCP 发
//! HTTP 请求，断言 `text/event-stream` 响应头与增量推送的 metrics 行。

#![cfg(all(feature = "panel", feature = "torch"))]

use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};

use av_runtime::metrics;
use av_runtime::panel::router;

fn temp_runs(tag: &str) -> std::path::PathBuf {
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("av-panel-sse-{tag}-{}-{n}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// 起真实服务（端口 0 随机分配），返回地址。
async fn spawn_server(runs_dir: std::path::PathBuf) -> std::net::SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, router(runs_dir)).await.unwrap();
    });
    addr
}

/// 发送请求并读取响应直至「条件满足或超时」。SSE 流无限长，不能读到 EOF，
/// 以谓词提前收手。
async fn http_roundtrip(
    addr: std::net::SocketAddr,
    path: &str,
    until: impl Fn(&str) -> bool,
    limit: Duration,
) -> String {
    let mut s = tokio::net::TcpStream::connect(addr).await.unwrap();
    let req = format!("GET {path} HTTP/1.1\r\nHost: {addr}\r\nAccept: text/event-stream\r\n\r\n");
    s.write_all(req.as_bytes()).await.unwrap();
    let mut buf: Vec<u8> = Vec::new();
    let read_all = async {
        loop {
            let mut chunk = [0u8; 2048];
            let n = s.read(&mut chunk).await.expect("读取失败");
            if n == 0 {
                break; // 服务端关闭（普通 JSON 响应带 Connection: close）
            }
            buf.extend_from_slice(&chunk[..n]);
            if until(&String::from_utf8_lossy(&buf)) {
                break;
            }
        }
    };
    tokio::time::timeout(limit, read_all)
        .await
        .expect("等待响应超时");
    String::from_utf8_lossy(&buf).into_owned()
}

/// 已有 2 行 metrics 的 run：SSE 连上即推历史快照，响应头为 text/event-stream。
#[tokio::test]
async fn sse_delivers_existing_metrics_rows() {
    let runs = temp_runs("snapshot");
    let run_dir = runs.join("run-live");
    std::fs::create_dir_all(&run_dir).unwrap();
    metrics::append(
        &run_dir,
        &serde_json::json!({"epoch": 1, "loss": 2.5, "metric": "mean_iou", "metric_value": 0.1}),
    )
    .unwrap();
    metrics::append(
        &run_dir,
        &serde_json::json!({"epoch": 2, "loss": 1.5, "metric": "mean_iou", "metric_value": 0.4}),
    )
    .unwrap();

    let addr = spawn_server(runs.clone()).await;
    let text = http_roundtrip(
        addr,
        "/sse/run-live",
        |t| t.contains("\"epoch\":2"),
        Duration::from_secs(15),
    )
    .await;

    let head = text.to_ascii_lowercase();
    assert!(head.contains("http/1.1 200 ok"), "响应行: {text:.200}");
    assert!(
        head.contains("content-type: text/event-stream"),
        "必须是 text/event-stream: {text:.300}"
    );
    // 两条历史行都应作为 data: 事件推送（连接即推快照）
    assert!(text.contains("data:"), "应有 SSE data 事件");
    assert!(text.contains("\"epoch\":1"));
    assert!(text.contains("\"epoch\":2"));
}

/// 训练中的 run（metrics.jsonl 尚未生成）：先推 waiting 事件；随后文件一出现，
/// 2 秒轮询内推到新行——这正是「面板实时看训练」的链路。
#[tokio::test]
async fn sse_waits_then_pushes_incremental_rows() {
    let runs = temp_runs("incremental");
    let run_dir = runs.join("run-running");
    std::fs::create_dir_all(&run_dir).unwrap();

    let addr = spawn_server(runs.clone()).await;
    let mut s = tokio::net::TcpStream::connect(addr).await.unwrap();
    s.write_all(b"GET /sse/run-running HTTP/1.1\r\nHost: x\r\n\r\n")
        .await
        .unwrap();

    // 阶段一：waiting 事件（第一个 tick 立即触发）
    let mut buf: Vec<u8> = Vec::new();
    let got_waiting = async {
        loop {
            let mut chunk = [0u8; 2048];
            let n = s.read(&mut chunk).await.unwrap();
            if n == 0 {
                panic!("SSE 连接被提前关闭");
            }
            buf.extend_from_slice(&chunk[..n]);
            if String::from_utf8_lossy(&buf).contains("\"status\":\"waiting\"") {
                break;
            }
        }
    };
    tokio::time::timeout(Duration::from_secs(10), got_waiting)
        .await
        .expect("应收到 waiting 事件");

    // 阶段二：模拟训练循环写第 3 个 epoch，2 秒轮询内必须推到浏览器
    let got_increment = async {
        loop {
            let mut chunk = [0u8; 2048];
            let n = s.read(&mut chunk).await.unwrap();
            if n == 0 {
                panic!("SSE 连接被提前关闭");
            }
            buf.extend_from_slice(&chunk[..n]);
            if String::from_utf8_lossy(&buf).contains("\"epoch\":3") {
                break;
            }
        }
    };
    let writer = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(500)).await;
        metrics::append(
            &run_dir,
            &serde_json::json!({"epoch": 3, "loss": 0.9, "metric": "mean_iou", "metric_value": 0.6}),
        )
        .unwrap();
    });
    tokio::time::timeout(Duration::from_secs(10), got_increment)
        .await
        .expect("新写入的 epoch 行应在 2 秒轮询内推送");
    writer.await.unwrap();
}

/// runs 列表 API 与页面：训练中（只有 config.snapshot.toml）的 run 也要可见。
#[tokio::test]
async fn api_runs_lists_in_progress_runs() {
    let runs = temp_runs("list");
    let run_dir = runs.join("run-wip");
    std::fs::create_dir_all(&run_dir).unwrap();
    std::fs::write(run_dir.join("config.snapshot.toml"), "# snapshot").unwrap();

    let addr = spawn_server(runs).await;
    let text = http_roundtrip(
        addr,
        "/api/runs",
        |t| t.contains('}'),
        Duration::from_secs(10),
    )
    .await;
    assert!(
        text.contains("run-wip"),
        "训练中的 run 必须出现在列表: {text}"
    );
}
