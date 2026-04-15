#!/usr/bin/env python3
"""Live training dashboard — reads train_log.jsonl, serves interactive charts.

Usage:
    python scripts/training_dashboard.py --log-dir checkpoints/argus_prime_cc_ddp_v2/
    python scripts/training_dashboard.py --log-dir checkpoints/run_a/ --log-dir checkpoints/run_b/

Opens browser at http://localhost:8050
"""

import argparse
import json
import os
import sys
from pathlib import Path

from flask import Flask, jsonify, send_from_directory

app = Flask(__name__)
LOG_DIRS = []


def read_jsonl(path):
    entries = []
    if not os.path.exists(path):
        return entries
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


@app.route("/")
def index():
    return DASHBOARD_HTML


@app.route("/api/data")
def api_data():
    runs = {}
    for log_dir in LOG_DIRS:
        name = Path(log_dir).name
        train_log = os.path.join(log_dir, "train_log.jsonl")
        runs[name] = {
            "train": read_jsonl(train_log),
        }
        # Auto-detect extra logs
        val_log = os.path.join(log_dir, "val_log.jsonl")
        if os.path.exists(val_log):
            runs[name]["val"] = read_jsonl(val_log)
        grad_log = os.path.join(log_dir, "grad_log.jsonl")
        if os.path.exists(grad_log):
            runs[name]["grads"] = read_jsonl(grad_log)
        weight_log = os.path.join(log_dir, "weight_log.jsonl")
        if os.path.exists(weight_log):
            runs[name]["weights"] = read_jsonl(weight_log)
    return jsonify(runs)


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Training Monitor</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Literata:opsz,wght@7..72,400;7..72,500;7..72,600&family=Geist+Mono:wght@400;500&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg: oklch(0.14 0.008 260);
    --surface: oklch(0.18 0.008 260);
    --surface-raised: oklch(0.21 0.008 260);
    --border: oklch(0.28 0.006 260);
    --text-primary: oklch(0.92 0.006 260);
    --text-secondary: oklch(0.62 0.006 260);
    --text-tertiary: oklch(0.45 0.006 260);
    --accent: oklch(0.72 0.12 180);
    --accent-dim: oklch(0.45 0.08 180);
    --warn: oklch(0.72 0.14 60);
    --danger: oklch(0.68 0.18 25);
    --good: oklch(0.68 0.12 155);
  }

  html { font-size: 15px; }
  body {
    font-family: 'Geist Mono', monospace;
    background: var(--bg);
    color: var(--text-primary);
    min-height: 100vh;
    -webkit-font-smoothing: antialiased;
  }

  header {
    padding: 24px 32px 16px;
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 24px;
    border-bottom: 1px solid var(--border);
  }

  header h1 {
    font-family: 'Literata', serif;
    font-size: 1.35rem;
    font-weight: 500;
    letter-spacing: -0.01em;
    color: var(--text-primary);
  }

  .header-status {
    font-size: 0.75rem;
    color: var(--text-tertiary);
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .header-status .dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--accent);
    animation: pulse 2.5s ease-in-out infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 0.4; }
    50% { opacity: 1; }
  }

  .controls {
    padding: 12px 32px;
    display: flex;
    gap: 20px;
    align-items: center;
    flex-wrap: wrap;
    border-bottom: 1px solid oklch(0.22 0.005 260);
  }

  .control-group {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .control-group label {
    font-size: 0.7rem;
    color: var(--text-tertiary);
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }

  select, input[type="range"] {
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text-primary);
    padding: 4px 8px;
    font-family: inherit;
    font-size: 0.75rem;
    border-radius: 3px;
    outline: none;
  }

  select:focus, input:focus { border-color: var(--accent-dim); }
  input[type="range"] { width: 100px; padding: 0; }

  .range-val {
    font-size: 0.7rem;
    color: var(--text-secondary);
    min-width: 28px;
    text-align: right;
  }

  .grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1px;
    background: oklch(0.22 0.005 260);
    padding: 1px;
  }

  .panel {
    background: var(--surface);
    padding: 20px 24px 8px;
    min-height: 260px;
  }

  .panel-title {
    font-size: 0.7rem;
    color: var(--text-tertiary);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 4px;
  }

  .panel-value {
    font-family: 'Literata', serif;
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 12px;
    letter-spacing: -0.02em;
  }

  .panel-value .unit {
    font-size: 0.75rem;
    font-weight: 400;
    color: var(--text-secondary);
    font-family: 'Geist Mono', monospace;
    margin-left: 4px;
  }

  .panel-value .delta {
    font-size: 0.7rem;
    font-family: 'Geist Mono', monospace;
    margin-left: 8px;
  }

  .delta.up { color: var(--danger); }
  .delta.down { color: var(--good); }
  .delta.neutral { color: var(--text-tertiary); }

  .js-plotly-plot .plotly .modebar { display: none !important; }

  footer {
    padding: 16px 32px;
    font-size: 0.7rem;
    color: var(--text-tertiary);
    text-align: right;
    border-top: 1px solid oklch(0.22 0.005 260);
  }

  .empty-state {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 60vh;
    flex-direction: column;
    gap: 12px;
  }

  .empty-state h2 {
    font-family: 'Literata', serif;
    font-weight: 500;
    font-size: 1.1rem;
    color: var(--text-secondary);
  }

  .empty-state p {
    font-size: 0.8rem;
    color: var(--text-tertiary);
  }
</style>
</head>
<body>

<header>
  <h1>Training Monitor</h1>
  <div class="header-status">
    <div class="dot"></div>
    <span id="status-text">Connecting...</span>
  </div>
</header>

<div class="controls">
  <div class="control-group">
    <label>Smoothing</label>
    <input type="range" id="smoothing" min="0" max="0.99" step="0.01" value="0.6">
    <span class="range-val" id="smooth-val">0.6</span>
  </div>
  <div class="control-group">
    <label>Range</label>
    <select id="timerange">
      <option value="all" selected>All</option>
      <option value="1000">Last 1K steps</option>
      <option value="5000">Last 5K steps</option>
      <option value="10000">Last 10K steps</option>
    </select>
  </div>
  <div class="control-group">
    <label>Run</label>
    <select id="run-select"></select>
  </div>
</div>

<div class="grid">
  <div class="panel">
    <div class="panel-title">Loss</div>
    <div class="panel-value" id="val-loss">--</div>
    <div id="chart-loss"></div>
  </div>
  <div class="panel">
    <div class="panel-title">Gradient Norm</div>
    <div class="panel-value" id="val-grad">--</div>
    <div id="chart-grad"></div>
  </div>
  <div class="panel">
    <div class="panel-title">Learning Rate</div>
    <div class="panel-value" id="val-lr">--</div>
    <div id="chart-lr"></div>
  </div>
  <div class="panel">
    <div class="panel-title">Throughput</div>
    <div class="panel-value" id="val-toks">--</div>
    <div id="chart-toks"></div>
  </div>
  <div class="panel">
    <div class="panel-title">GPU Memory</div>
    <div class="panel-value" id="val-mem">--</div>
    <div id="chart-mem"></div>
  </div>
  <div class="panel">
    <div class="panel-title">Step Rate</div>
    <div class="panel-value" id="val-steprate">--</div>
    <div id="chart-steprate"></div>
  </div>
</div>

<footer id="footer">Waiting for data...</footer>

<script>
const PLOTLY_LAYOUT_BASE = {
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(0,0,0,0)',
  font: { family: 'Geist Mono, monospace', size: 10, color: '#8a8fa0' },
  margin: { l: 48, r: 16, t: 4, b: 32 },
  xaxis: {
    gridcolor: 'rgba(255,255,255,0.04)',
    zerolinecolor: 'rgba(255,255,255,0.06)',
    tickfont: { size: 9 },
  },
  yaxis: {
    gridcolor: 'rgba(255,255,255,0.04)',
    zerolinecolor: 'rgba(255,255,255,0.06)',
    tickfont: { size: 9 },
  },
  showlegend: false,
  height: 180,
};

const ACCENT = 'oklch(0.72 0.12 180)';
const ACCENT_DIM = 'oklch(0.45 0.08 180)';
const WARN_COLOR = 'oklch(0.72 0.14 60)';
const RAW_COLOR = 'rgba(255,255,255,0.08)';

function ema(data, alpha) {
  if (!data.length || alpha <= 0) return data;
  const result = [data[0]];
  for (let i = 1; i < data.length; i++) {
    result.push(alpha * result[i-1] + (1 - alpha) * data[i]);
  }
  return result;
}

function computeStepRate(steps) {
  // Approximate steps/sec from step intervals (assumes uniform logging interval)
  const rates = [];
  for (let i = 1; i < steps.length; i++) {
    // We don't have timestamps — estimate from step delta
    // This will be refined if we add timestamps
    rates.push(null);
  }
  return rates;
}

function formatNum(n, decimals=2) {
  if (n === null || n === undefined || isNaN(n)) return '--';
  if (Math.abs(n) >= 1000) return n.toLocaleString('en', {maximumFractionDigits: 0});
  return n.toFixed(decimals);
}

function formatDelta(current, initial) {
  if (!initial || !current) return '';
  const pct = ((current - initial) / Math.abs(initial) * 100);
  const sign = pct >= 0 ? '+' : '';
  const cls = pct < -1 ? 'down' : pct > 1 ? 'up' : 'neutral';
  return `<span class="delta ${cls}">${sign}${pct.toFixed(1)}%</span>`;
}

let allData = {};
let chartInited = false;

async function fetchData() {
  try {
    const resp = await fetch('/api/data');
    allData = await resp.json();
    updateUI();
    document.getElementById('status-text').textContent =
      `Live \u2014 ${Object.keys(allData).length} run${Object.keys(allData).length !== 1 ? 's' : ''}`;
  } catch (e) {
    document.getElementById('status-text').textContent = 'Connection lost';
  }
}

function updateUI() {
  const runSelect = document.getElementById('run-select');
  const currentRun = runSelect.value;
  const runNames = Object.keys(allData);

  // Update run selector
  if (runSelect.options.length !== runNames.length) {
    runSelect.innerHTML = '';
    runNames.forEach(name => {
      const opt = document.createElement('option');
      opt.value = name;
      opt.textContent = name;
      runSelect.appendChild(opt);
    });
    if (currentRun && runNames.includes(currentRun)) {
      runSelect.value = currentRun;
    }
  }

  const selectedRun = runSelect.value;
  if (!selectedRun || !allData[selectedRun]) return;

  let entries = allData[selectedRun].train || [];
  if (!entries.length) return;

  // Apply time range filter
  const range = document.getElementById('timerange').value;
  if (range !== 'all') {
    const maxStep = entries[entries.length - 1].step;
    const cutoff = maxStep - parseInt(range);
    entries = entries.filter(e => e.step >= cutoff);
  }

  const alpha = parseFloat(document.getElementById('smoothing').value);
  const steps = entries.map(e => e.step);
  const loss = entries.map(e => e.loss);
  const bpb = entries.map(e => e.bpb);
  const grad = entries.map(e => e.grad_norm);
  const lr = entries.map(e => e.lr);
  const toks = entries.map(e => e.tok_s);
  const mfu = entries.map(e => e.mfu * 100);
  const mem = entries.map(e => e.mem_gb);

  const lossSmooth = ema(loss, alpha);
  const gradSmooth = ema(grad, alpha);
  const toksSmooth = ema(toks, alpha);

  // Compute step rate (steps per second)
  const stepIntervals = [];
  const stepRateSteps = [];
  // Use tok/s and tokens_per_step to derive step rate
  // tokens_per_step is constant, so step_rate ~ tok_s / tokens_per_step
  // We don't know tokens_per_step exactly, but we can derive from the data
  const stepDelta = steps.length > 1 ? steps[1] - steps[0] : 100;
  const tokPerLogInterval = toks.map((t, i) => t); // tok/s is already the rate
  // step_rate = tok_s / (batch_size * block_size * accum_steps * world_size / step_delta_time)
  // Simpler: derive from step progression assuming uniform logging
  // steps[i] / elapsed ~ step_rate. We'll just use tok_s normalized.

  // Latest values
  const latest = entries[entries.length - 1];
  const first = entries[0];

  document.getElementById('val-loss').innerHTML =
    `${formatNum(latest.loss, 3)}<span class="unit">CE</span>${formatDelta(latest.loss, first.loss)}`;
  document.getElementById('val-grad').innerHTML =
    `${formatNum(latest.grad_norm, 3)}${latest.grad_norm > 1.0 ? '<span class="unit" style="color:var(--danger)">CLIPPED</span>' : ''}`;
  document.getElementById('val-lr').innerHTML =
    `${latest.lr.toExponential(2)}`;
  document.getElementById('val-toks').innerHTML =
    `${formatNum(latest.tok_s, 0)}<span class="unit">tok/s</span><span class="unit" style="margin-left:12px">${(latest.mfu*100).toFixed(1)}% MFU</span>`;
  document.getElementById('val-mem').innerHTML =
    `${formatNum(latest.mem_gb, 1)}<span class="unit">GB</span>`;
  document.getElementById('val-steprate').innerHTML =
    `${formatNum(latest.tok_s / 65536, 3)}<span class="unit">steps/s</span>`;

  document.getElementById('footer').textContent =
    `Step ${latest.step.toLocaleString()} \u2014 ${entries.length} log entries \u2014 refreshes every 30s`;

  const config = { displayModeBar: false, responsive: true };

  // Loss chart
  Plotly.react('chart-loss', [
    { x: steps, y: loss, type: 'scattergl', mode: 'lines', line: { color: RAW_COLOR, width: 1 }, hoverinfo: 'skip' },
    { x: steps, y: lossSmooth, type: 'scattergl', mode: 'lines', line: { color: ACCENT, width: 1.5 } },
  ], { ...PLOTLY_LAYOUT_BASE, yaxis: { ...PLOTLY_LAYOUT_BASE.yaxis, title: { text: 'loss', font: { size: 9 } } } }, config);

  // Grad chart
  const clipLine = grad.map(() => 1.0);
  Plotly.react('chart-grad', [
    { x: steps, y: grad, type: 'scattergl', mode: 'lines', line: { color: RAW_COLOR, width: 1 }, hoverinfo: 'skip' },
    { x: steps, y: gradSmooth, type: 'scattergl', mode: 'lines', line: { color: WARN_COLOR, width: 1.5 } },
    { x: steps, y: clipLine, type: 'scattergl', mode: 'lines', line: { color: 'rgba(255,255,255,0.12)', width: 1, dash: 'dot' }, hoverinfo: 'skip' },
  ], { ...PLOTLY_LAYOUT_BASE }, config);

  // LR chart
  Plotly.react('chart-lr', [
    { x: steps, y: lr, type: 'scattergl', mode: 'lines', line: { color: ACCENT, width: 1.5 } },
  ], { ...PLOTLY_LAYOUT_BASE }, config);

  // Throughput chart
  const toksMean = toks.reduce((a,b) => a+b, 0) / toks.length;
  const toksStd = Math.sqrt(toks.reduce((a,b) => a + (b - toksMean)**2, 0) / toks.length);
  const bandHi = toks.map(() => toksMean + toksStd);
  const bandLo = toks.map(() => toksMean - toksStd);
  Plotly.react('chart-toks', [
    { x: steps, y: bandHi, type: 'scattergl', mode: 'lines', line: { color: 'transparent' }, showlegend: false, hoverinfo: 'skip' },
    { x: steps, y: bandLo, type: 'scattergl', mode: 'lines', line: { color: 'transparent' }, fill: 'tonexty', fillcolor: 'rgba(255,255,255,0.03)', showlegend: false, hoverinfo: 'skip' },
    { x: steps, y: toks, type: 'scattergl', mode: 'lines', line: { color: RAW_COLOR, width: 1 }, hoverinfo: 'skip' },
    { x: steps, y: toksSmooth, type: 'scattergl', mode: 'lines', line: { color: ACCENT, width: 1.5 } },
  ], { ...PLOTLY_LAYOUT_BASE, yaxis: { ...PLOTLY_LAYOUT_BASE.yaxis, title: { text: 'tok/s', font: { size: 9 } } } }, config);

  // Memory chart
  Plotly.react('chart-mem', [
    { x: steps, y: mem, type: 'scattergl', mode: 'lines', line: { color: ACCENT, width: 1.5 }, fill: 'tozeroy', fillcolor: 'rgba(100,220,200,0.05)' },
  ], { ...PLOTLY_LAYOUT_BASE, yaxis: { ...PLOTLY_LAYOUT_BASE.yaxis, title: { text: 'GB', font: { size: 9 } }, rangemode: 'tozero' } }, config);

  // Step rate chart (derived from tok/s)
  const stepRate = toks.map(t => t / 65536);
  const stepRateSmooth = ema(stepRate, alpha);
  Plotly.react('chart-steprate', [
    { x: steps, y: stepRate, type: 'scattergl', mode: 'lines', line: { color: RAW_COLOR, width: 1 }, hoverinfo: 'skip' },
    { x: steps, y: stepRateSmooth, type: 'scattergl', mode: 'lines', line: { color: ACCENT, width: 1.5 } },
  ], { ...PLOTLY_LAYOUT_BASE, yaxis: { ...PLOTLY_LAYOUT_BASE.yaxis, title: { text: 'steps/s', font: { size: 9 } } } }, config);
}

// Controls
document.getElementById('smoothing').addEventListener('input', e => {
  document.getElementById('smooth-val').textContent = e.target.value;
  updateUI();
});
document.getElementById('timerange').addEventListener('change', updateUI);
document.getElementById('run-select').addEventListener('change', updateUI);

// Initial fetch + polling
fetchData();
setInterval(fetchData, 30000);
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Live Training Dashboard")
    parser.add_argument("--log-dir", action="append", required=True,
                        help="Checkpoint directory containing train_log.jsonl (can specify multiple)")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    global LOG_DIRS
    LOG_DIRS = args.log_dir

    for d in LOG_DIRS:
        log_path = os.path.join(d, "train_log.jsonl")
        if os.path.exists(log_path):
            with open(log_path) as f:
                n = sum(1 for _ in f)
            print(f"  {d}: {n} entries")
        else:
            print(f"  {d}: no train_log.jsonl yet (will appear when training starts)")

    print(f"\nDashboard: http://localhost:{args.port}")
    import webbrowser
    webbrowser.open(f"http://localhost:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
