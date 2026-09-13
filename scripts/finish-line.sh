#!/usr/bin/env bash
# finish-line.sh —— 夜间收尾自动化：等实验完成 → 逐臂验收评测 → 汇总对比报告
# 由监督 agent 启动/看护；幂等，可重复执行。
set -u
cd "E:/dev/aegis-vision" || exit 1
export PATH="/e/libs/libtorch-cu128-2.11/lib:$PATH"
AV=./target-cuda/release/av-runtime.exe
L=runs/finish-line.log
PY=D:/Dev/Tools/Miniconda3/python.exe
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$L"; }

log "finish-line 启动（pid $$）"

# 1) csp 臂：等训练自然结束（report.json = 训练终点标记）
while [ ! -f runs/seg-harness-csp/report.json ]; do
  sleep 60
done
log "csp 臂完成"

# 2) dino672：无 report 且无进程 → resume 重启（上限 6 次；进程活着就等）
for i in 1 2 3 4 5 6; do
  [ -f runs/seg-harness-dino672/report.json ] && break
  if ! tasklist //FI "IMAGENAME eq av-runtime.exe" 2>/dev/null | grep -q av-runtime; then
    log "dino672 无进程且未完成 → --resume 重启（第 $i 次）"
    $AV train -c configs/seg_harness_dino672.toml --resume >> runs/seg_harness_dino672.log 2>&1
  else
    sleep 180
  fi
done
log "dino672 阶段结束"

# 3) 全臂验收评测（幂等：有 eval-final.json 就跳过）
for arm in seg-harness-pretrain seg-harness-none seg-harness-r18 seg-harness-r18-none seg-harness-csp seg-harness-dino seg-harness-dino672; do
  if [ -f "runs/$arm/report.json" ] && [ ! -f "runs/$arm/eval-final.json" ]; then
    log "验收评测 $arm"
    $AV eval -w "runs/$arm/best.ckpt" --report "runs/$arm/eval-final.json" >> "$L" 2>&1 \
      || log "$arm 评测失败（见上）"
  fi
done

# 4) 汇总对比报告
"$PY" scripts/build_comparison.py >> "$L" 2>&1 && log "COMPARISON.md 已生成" || log "对比报告生成失败"
touch runs/ALL-DONE
log "finish-line 全部完成"
