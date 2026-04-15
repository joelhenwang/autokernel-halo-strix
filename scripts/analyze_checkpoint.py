#!/usr/bin/env python3
"""Checkpoint analyzer — weight/grad analysis + hierarchical heatmap explorer.

Loads a checkpoint, computes per-layer weight norms, gradient norms,
parameter breakdown, and text generation. Outputs a self-contained
report.html with interactive Canvas-based heatmap explorer.

Usage:
    python scripts/analyze_checkpoint.py \
        --checkpoint checkpoints/.../step_9000.pt \
        --model models/argus_prime.py --class-name ArgusPrime \
        --dataset datasets/common_crawl_sample.bin
"""

import argparse
import importlib.util
import json
import math
import os
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_from_file(model_path: str, class_name: str):
    spec = importlib.util.spec_from_file_location("user_model", model_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["user_model"] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, class_name)()


# ---------------------------------------------------------------------------
# Weight analysis (from state_dict, no model needed)
# ---------------------------------------------------------------------------

def analyze_weights(state_dict):
    """Compute per-layer and per-block weight statistics."""
    layers = OrderedDict()
    for name, tensor in state_dict.items():
        if tensor.ndim == 0:
            continue
        t = tensor.float()
        layers[name] = {
            "shape": list(tensor.shape),
            "numel": tensor.numel(),
            "norm": t.norm().item(),
            "mean": t.mean().item(),
            "std": t.std().item(),
            "min": t.min().item(),
            "max": t.max().item(),
            "abs_mean": t.abs().mean().item(),
        }
    return layers


def build_block_hierarchy(state_dict):
    """Group parameters into blocks for hierarchical heatmap."""
    blocks = defaultdict(lambda: defaultdict(dict))

    for name, tensor in state_dict.items():
        if tensor.ndim < 2:
            continue
        parts = name.split(".")
        # Find block identifier (e.g., layers.0, layers.1, ...)
        block_name = "root"
        param_name = name
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts) and parts[i+1].isdigit():
                block_name = f"layers.{parts[i+1]}"
                param_name = ".".join(parts[i+2:])
                break
            elif part in ("tok_embeddings", "output", "norm", "film"):
                block_name = part
                param_name = ".".join(parts[1:]) if len(parts) > 1 else "weight"
                break

        t = tensor.float()

        # Downsample for Level 2 heatmap
        rows, cols = tensor.shape[0], tensor.shape[-1]
        max_dim = 128
        if rows > max_dim or cols > max_dim:
            # Reshape to 2D if needed
            mat = t.reshape(rows, -1)
            cols = mat.shape[1]
            # Block-average downsample
            row_step = max(1, rows // max_dim)
            col_step = max(1, cols // max_dim)
            out_rows = min(rows, max_dim)
            out_cols = min(cols, max_dim)
            downsampled = torch.zeros(out_rows, out_cols)
            for r in range(out_rows):
                for c in range(out_cols):
                    r_start, r_end = r * row_step, min((r + 1) * row_step, rows)
                    c_start, c_end = c * col_step, min((c + 1) * col_step, cols)
                    downsampled[r, c] = mat[r_start:r_end, c_start:c_end].abs().mean()
            cells = downsampled.tolist()
            block_size_info = f"{row_step}x{col_step} blocks"
        else:
            cells = t.reshape(rows, -1).abs().tolist()
            block_size_info = "raw"

        blocks[block_name][param_name] = {
            "norm": t.norm().item(),
            "abs_mean": t.abs().mean().item(),
            "shape": list(tensor.shape),
            "cells": cells,
            "block_size": block_size_info,
        }

    return dict(blocks)


# ---------------------------------------------------------------------------
# Gradient analysis (requires model + one forward/backward)
# ---------------------------------------------------------------------------

def analyze_gradients(model, dataset_path, block_size=256, batch_size=4):
    """Run one forward+backward pass, collect per-layer gradient norms."""
    device = next(model.parameters()).device

    # Load a small batch from .bin
    raw = np.fromfile(dataset_path, dtype=np.uint16, count=(block_size + 1) * batch_size)
    tokens = torch.from_numpy(raw.astype(np.int64)).reshape(batch_size, block_size + 1)
    input_ids = tokens[:, :-1].to(device)
    targets = tokens[:, 1:].to(device)

    model.train()
    model.zero_grad()

    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        output = model(input_ids)
        loss = nn.functional.cross_entropy(output.view(-1, output.size(-1)), targets.view(-1))
    loss.backward()

    grads = OrderedDict()
    for name, param in model.named_parameters():
        if param.grad is not None:
            g = param.grad.float()
            w = param.data.float()
            w_norm = w.norm().item()
            g_norm = g.norm().item()
            grads[name] = {
                "grad_norm": g_norm,
                "weight_norm": w_norm,
                "ratio": g_norm / max(w_norm, 1e-10),
                "shape": list(param.shape),
            }

    model.zero_grad()
    return grads, loss.item()


# ---------------------------------------------------------------------------
# Parameter breakdown
# ---------------------------------------------------------------------------

def param_breakdown(model):
    """Count parameters by module type."""
    counts = defaultdict(int)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        n = param.numel()
        name_lower = name.lower()
        if "attn" in name_lower or "wq" in name_lower or "wk" in name_lower or "wv" in name_lower or "wo" in name_lower:
            counts["attention"] += n
        elif "ffn" in name_lower or "gate_up" in name_lower or "w_down" in name_lower:
            counts["ffn"] += n
        elif "conv" in name_lower:
            counts["conv"] += n
        elif "ttt" in name_lower:
            counts["ttt"] += n
        elif "embed" in name_lower or "tok_embed" in name_lower:
            counts["embedding"] += n
        elif "film" in name_lower:
            counts["film"] += n
        elif "norm" in name_lower:
            counts["norm"] += n
        else:
            counts["other"] += n
    return dict(counts)


# ---------------------------------------------------------------------------
# Text generation
# ---------------------------------------------------------------------------

def generate_samples(model, prompts, max_tokens=100, temperatures=(0.0, 0.7)):
    """Generate text from fixed prompts at different temperatures."""
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    device = next(model.parameters()).device
    model.eval()

    results = {}
    for prompt in prompts:
        results[prompt] = {}
        tokens = enc.encode(prompt)
        for temp in temperatures:
            input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
            generated = list(tokens)
            with torch.no_grad():
                for _ in range(max_tokens):
                    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                        logits = model(input_ids)
                    next_logits = logits[0, -1, :]
                    if temp <= 0:
                        next_token = next_logits.argmax().item()
                    else:
                        probs = torch.softmax(next_logits / temp, dim=-1)
                        next_token = torch.multinomial(probs, 1).item()
                    generated.append(next_token)
                    if next_token == 50256:
                        break
                    input_ids = torch.tensor([generated], dtype=torch.long, device=device)
            results[prompt][f"temp_{temp}"] = enc.decode(generated)

    model.train()
    return results


# ---------------------------------------------------------------------------
# Report HTML template
# ---------------------------------------------------------------------------

def build_report_html(weight_data, grad_data, hierarchy, breakdown, generations, summary):
    """Build self-contained interactive HTML report."""
    return REPORT_TEMPLATE.replace(
        "/*__WEIGHT_DATA__*/", json.dumps(weight_data)
    ).replace(
        "/*__GRAD_DATA__*/", json.dumps(grad_data)
    ).replace(
        "/*__HIERARCHY__*/", json.dumps(hierarchy)
    ).replace(
        "/*__BREAKDOWN__*/", json.dumps(breakdown)
    ).replace(
        "/*__GENERATIONS__*/", json.dumps(generations)
    ).replace(
        "/*__SUMMARY__*/", json.dumps(summary)
    )


REPORT_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Checkpoint Analysis</title>
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
    border-bottom: 1px solid var(--border);
  }

  header h1 {
    font-family: 'Literata', serif;
    font-size: 1.35rem;
    font-weight: 500;
    letter-spacing: -0.01em;
  }

  header .meta {
    font-size: 0.75rem;
    color: var(--text-tertiary);
    margin-top: 4px;
  }

  .tabs {
    display: flex;
    gap: 0;
    border-bottom: 1px solid var(--border);
    padding: 0 32px;
  }

  .tab {
    padding: 10px 20px;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-tertiary);
    cursor: pointer;
    border-bottom: 2px solid transparent;
    transition: color 0.15s, border-color 0.15s;
  }

  .tab:hover { color: var(--text-secondary); }
  .tab.active { color: var(--text-primary); border-bottom-color: var(--accent); }

  .tab-content { display: none; padding: 24px 32px; }
  .tab-content.active { display: block; }

  .chart-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
    margin-bottom: 24px;
  }

  .chart-card {
    background: var(--surface);
    padding: 20px 24px 12px;
    border-radius: 4px;
  }

  .chart-card h3 {
    font-size: 0.7rem;
    color: var(--text-tertiary);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 12px;
  }

  /* Heatmap explorer */
  #heatmap-container {
    position: relative;
    background: var(--surface);
    border-radius: 4px;
    overflow: hidden;
  }

  #heatmap-canvas {
    display: block;
    cursor: pointer;
  }

  .heatmap-controls {
    display: flex;
    gap: 16px;
    align-items: center;
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
  }

  .heatmap-controls label {
    font-size: 0.7rem;
    color: var(--text-tertiary);
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }

  .heatmap-controls select {
    background: var(--bg);
    border: 1px solid var(--border);
    color: var(--text-primary);
    padding: 4px 8px;
    font-family: inherit;
    font-size: 0.75rem;
    border-radius: 3px;
  }

  .breadcrumb {
    font-size: 0.72rem;
    color: var(--text-secondary);
    padding: 8px 16px;
    background: oklch(0.16 0.006 260);
  }

  .breadcrumb span { cursor: pointer; }
  .breadcrumb span:hover { color: var(--accent); }
  .breadcrumb .sep { color: var(--text-tertiary); margin: 0 6px; }

  #tooltip {
    position: fixed;
    display: none;
    background: oklch(0.12 0.01 260);
    border: 1px solid var(--border);
    padding: 8px 12px;
    font-size: 0.72rem;
    line-height: 1.5;
    border-radius: 3px;
    pointer-events: none;
    z-index: 100;
    max-width: 320px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.4);
  }

  .gen-section { margin-bottom: 24px; }
  .gen-prompt {
    font-size: 0.8rem;
    color: var(--accent);
    margin-bottom: 8px;
    font-weight: 500;
  }
  .gen-output {
    background: var(--surface);
    padding: 12px 16px;
    border-radius: 4px;
    font-size: 0.78rem;
    line-height: 1.6;
    white-space: pre-wrap;
    margin-bottom: 8px;
    color: var(--text-secondary);
  }
  .gen-label {
    font-size: 0.65rem;
    color: var(--text-tertiary);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 4px;
  }

  .js-plotly-plot .plotly .modebar { display: none !important; }
</style>
</head>
<body>

<header>
  <h1>Checkpoint Analysis</h1>
  <div class="meta" id="header-meta"></div>
</header>

<div class="tabs">
  <div class="tab active" data-tab="overview">Overview</div>
  <div class="tab" data-tab="heatmap">Heatmap Explorer</div>
  <div class="tab" data-tab="generations">Generations</div>
</div>

<div class="tab-content active" id="tab-overview">
  <div class="chart-row">
    <div class="chart-card">
      <h3>Weight Norms by Layer</h3>
      <div id="chart-wnorm"></div>
    </div>
    <div class="chart-card">
      <h3>Gradient Norms by Layer</h3>
      <div id="chart-gnorm"></div>
    </div>
  </div>
  <div class="chart-row">
    <div class="chart-card">
      <h3>Grad / Weight Ratio</h3>
      <div id="chart-ratio"></div>
    </div>
    <div class="chart-card">
      <h3>Parameter Breakdown</h3>
      <div id="chart-breakdown"></div>
    </div>
  </div>
</div>

<div class="tab-content" id="tab-heatmap">
  <div class="heatmap-controls">
    <div>
      <label>Mode</label>
      <select id="hm-mode">
        <option value="weights">Weights</option>
        <option value="grads">Gradients</option>
      </select>
    </div>
    <div>
      <label>Colormap</label>
      <select id="hm-cmap">
        <option value="viridis" selected>Viridis</option>
        <option value="magma">Magma</option>
        <option value="inferno">Inferno</option>
      </select>
    </div>
  </div>
  <div class="breadcrumb" id="breadcrumb">
    <span data-level="0">Model</span>
  </div>
  <div id="heatmap-container">
    <canvas id="heatmap-canvas"></canvas>
  </div>
  <div id="tooltip"></div>
</div>

<div class="tab-content" id="tab-generations">
  <div id="gen-content"></div>
</div>

<script>
// Embedded data
const WEIGHT_DATA = /*__WEIGHT_DATA__*/{};
const GRAD_DATA = /*__GRAD_DATA__*/{};
const HIERARCHY = /*__HIERARCHY__*/{};
const BREAKDOWN = /*__BREAKDOWN__*/{};
const GENERATIONS = /*__GENERATIONS__*/{};
const SUMMARY = /*__SUMMARY__*/{};

const ACCENT = 'oklch(0.72 0.12 180)';
const ACCENT_DIM = 'oklch(0.45 0.08 180)';
const SURFACE = '#2a2d38';
const TEXT_SEC = '#8a8fa0';

// --- Colormaps ---
const COLORMAPS = {
  viridis: [[68,1,84],[72,35,116],[64,67,135],[52,94,141],[41,120,142],[32,144,140],[34,167,132],[68,190,112],[121,209,81],[189,222,38],[253,231,37]],
  magma: [[0,0,4],[18,14,54],[51,16,101],[90,17,126],[130,26,129],[167,42,118],[203,63,98],[233,92,72],[252,130,48],[254,177,42],[252,229,65]],
  inferno: [[0,0,4],[14,11,53],[46,9,99],[85,15,109],[120,28,109],[153,44,96],[183,63,73],[209,89,47],[228,121,24],[240,160,26],[245,200,48]],
};

function sampleColormap(name, t) {
  t = Math.max(0, Math.min(1, t));
  const cm = COLORMAPS[name] || COLORMAPS.viridis;
  const idx = t * (cm.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.min(lo + 1, cm.length - 1);
  const f = idx - lo;
  const r = Math.round(cm[lo][0] * (1-f) + cm[hi][0] * f);
  const g = Math.round(cm[lo][1] * (1-f) + cm[hi][1] * f);
  const b = Math.round(cm[lo][2] * (1-f) + cm[hi][2] * f);
  return `rgb(${r},${g},${b})`;
}

// --- Tabs ---
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById('tab-' + tab.dataset.tab).classList.add('active');
    if (tab.dataset.tab === 'heatmap') drawHeatmap();
  });
});

// --- Header ---
document.getElementById('header-meta').textContent =
  `Step ${SUMMARY.step || '?'} \u2014 ${SUMMARY.total_params ? (SUMMARY.total_params / 1e6).toFixed(1) + 'M params' : ''} \u2014 Loss: ${SUMMARY.loss ? SUMMARY.loss.toFixed(4) : '?'}`;

// --- Overview charts ---
const plotConfig = { displayModeBar: false, responsive: true };
const layoutBase = {
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(0,0,0,0)',
  font: { family: 'Geist Mono, monospace', size: 9, color: TEXT_SEC },
  margin: { l: 48, r: 16, t: 8, b: 80 },
  xaxis: { gridcolor: 'rgba(255,255,255,0.04)', tickangle: -45, tickfont: { size: 8 } },
  yaxis: { gridcolor: 'rgba(255,255,255,0.04)' },
  height: 260,
  showlegend: false,
};

// Weight norms bar chart
const wNames = Object.keys(WEIGHT_DATA).filter(k => WEIGHT_DATA[k].norm > 0);
const wNorms = wNames.map(k => WEIGHT_DATA[k].norm);
const wLabels = wNames.map(k => k.replace(/^layers\.\d+\./, '').substring(0, 20));
Plotly.newPlot('chart-wnorm', [{
  x: wLabels, y: wNorms, type: 'bar',
  marker: { color: ACCENT, opacity: 0.8 },
}], { ...layoutBase }, plotConfig);

// Gradient norms bar chart
const gNames = Object.keys(GRAD_DATA);
const gNorms = gNames.map(k => GRAD_DATA[k].grad_norm);
const gLabels = gNames.map(k => k.replace(/^layers\.\d+\./, '').substring(0, 20));
Plotly.newPlot('chart-gnorm', [{
  x: gLabels, y: gNorms, type: 'bar',
  marker: { color: 'oklch(0.72 0.14 60)', opacity: 0.8 },
}], { ...layoutBase }, plotConfig);

// Grad/weight ratio
const rNames = Object.keys(GRAD_DATA);
const rVals = rNames.map(k => GRAD_DATA[k].ratio);
const rLabels = rNames.map(k => k.replace(/^layers\.\d+\./, '').substring(0, 20));
Plotly.newPlot('chart-ratio', [{
  x: rLabels, y: rVals, type: 'bar',
  marker: { color: 'oklch(0.68 0.12 155)', opacity: 0.8 },
}], { ...layoutBase, yaxis: { ...layoutBase.yaxis, type: 'log' } }, plotConfig);

// Parameter breakdown pie
const bKeys = Object.keys(BREAKDOWN);
const bVals = bKeys.map(k => BREAKDOWN[k]);
Plotly.newPlot('chart-breakdown', [{
  labels: bKeys, values: bVals, type: 'pie', hole: 0.45,
  textinfo: 'label+percent', textfont: { size: 10 },
  marker: { colors: bKeys.map((_, i) => sampleColormap('viridis', i / Math.max(1, bKeys.length - 1))) },
}], { ...layoutBase, margin: { l: 16, r: 16, t: 16, b: 16 }, height: 260 }, plotConfig);

// --- Heatmap Explorer ---
const canvas = document.getElementById('heatmap-canvas');
const ctx = canvas.getContext('2d');
const tooltip = document.getElementById('tooltip');
const breadcrumb = document.getElementById('breadcrumb');

let hmLevel = 0; // 0=blocks, 1=matrices, 2=cells
let hmBlock = null;
let hmMatrix = null;
let hmRects = []; // current clickable regions

function getHeatmapMode() { return document.getElementById('hm-mode').value; }
function getCmap() { return document.getElementById('hm-cmap').value; }

function drawHeatmap() {
  const container = document.getElementById('heatmap-container');
  const width = container.clientWidth;
  const dpr = window.devicePixelRatio || 1;

  if (hmLevel === 0) drawLevel0(width, dpr);
  else if (hmLevel === 1) drawLevel1(width, dpr);
  else if (hmLevel === 2) drawLevel2(width, dpr);

  updateBreadcrumb();
}

function drawLevel0(width, dpr) {
  const blocks = Object.keys(HIERARCHY).sort((a, b) => {
    const aNum = parseInt(a.split('.')[1]) || 0;
    const bNum = parseInt(b.split('.')[1]) || 0;
    if (a.startsWith('layers') && b.startsWith('layers')) return aNum - bNum;
    if (a.startsWith('layers')) return 1;
    if (b.startsWith('layers')) return -1;
    return a.localeCompare(b);
  });

  const cols = Math.ceil(Math.sqrt(blocks.length * 1.5));
  const rows = Math.ceil(blocks.length / cols);
  const pad = 4;
  const cellW = Math.floor((width - pad * (cols + 1)) / cols);
  const cellH = Math.max(60, cellW * 0.6);
  const height = rows * (cellH + pad) + pad;

  canvas.width = width * dpr;
  canvas.height = height * dpr;
  canvas.style.width = width + 'px';
  canvas.style.height = height + 'px';
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, width, height);

  // Compute block-level norms
  const mode = getHeatmapMode();
  const values = blocks.map(bk => {
    const params = HIERARCHY[bk];
    let total = 0, count = 0;
    Object.values(params).forEach(p => {
      total += mode === 'weights' ? p.abs_mean : (GRAD_DATA[bk + '.' + Object.keys(params)[count]]?.grad_norm || p.abs_mean);
      count++;
    });
    return count > 0 ? total / count : 0;
  });

  const maxVal = Math.max(...values, 1e-10);
  const cmap = getCmap();
  hmRects = [];

  blocks.forEach((bk, i) => {
    const col = i % cols;
    const row = Math.floor(i / cols);
    const x = pad + col * (cellW + pad);
    const y = pad + row * (cellH + pad);
    const t = values[i] / maxVal;

    ctx.fillStyle = sampleColormap(cmap, t);
    ctx.fillRect(x, y, cellW, cellH);

    // Label
    ctx.fillStyle = t > 0.5 ? 'rgba(0,0,0,0.7)' : 'rgba(255,255,255,0.7)';
    ctx.font = '10px Geist Mono, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    const label = bk.replace('layers.', 'L');
    ctx.fillText(label, x + cellW/2, y + cellH/2 - 6);
    ctx.font = '9px Geist Mono, monospace';
    ctx.fillStyle = t > 0.5 ? 'rgba(0,0,0,0.5)' : 'rgba(255,255,255,0.4)';
    ctx.fillText(values[i].toFixed(4), x + cellW/2, y + cellH/2 + 8);

    hmRects.push({ x, y, w: cellW, h: cellH, name: bk, value: values[i], level: 0 });
  });
}

function drawLevel1(width, dpr) {
  if (!hmBlock || !HIERARCHY[hmBlock]) return;
  const params = HIERARCHY[hmBlock];
  const names = Object.keys(params);

  const cols = Math.ceil(Math.sqrt(names.length * 1.5));
  const rows = Math.ceil(names.length / cols);
  const pad = 4;
  const cellW = Math.floor((width - pad * (cols + 1)) / cols);
  const cellH = Math.max(60, cellW * 0.5);
  const height = rows * (cellH + pad) + pad;

  canvas.width = width * dpr;
  canvas.height = height * dpr;
  canvas.style.width = width + 'px';
  canvas.style.height = height + 'px';
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, width, height);

  const mode = getHeatmapMode();
  const values = names.map(n => {
    const p = params[n];
    if (mode === 'weights') return p.abs_mean;
    const fullName = hmBlock + '.' + n;
    return GRAD_DATA[fullName]?.grad_norm || p.abs_mean;
  });
  const maxVal = Math.max(...values, 1e-10);
  const cmap = getCmap();
  hmRects = [];

  names.forEach((n, i) => {
    const col = i % cols;
    const row = Math.floor(i / cols);
    const x = pad + col * (cellW + pad);
    const y = pad + row * (cellH + pad);
    const t = values[i] / maxVal;

    ctx.fillStyle = sampleColormap(cmap, t);
    ctx.fillRect(x, y, cellW, cellH);

    ctx.fillStyle = t > 0.5 ? 'rgba(0,0,0,0.7)' : 'rgba(255,255,255,0.7)';
    ctx.font = '9px Geist Mono, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    const label = n.length > 18 ? n.substring(0, 18) + '\u2026' : n;
    ctx.fillText(label, x + cellW/2, y + cellH/2 - 8);
    ctx.font = '8px Geist Mono, monospace';
    ctx.fillStyle = t > 0.5 ? 'rgba(0,0,0,0.5)' : 'rgba(255,255,255,0.4)';
    const shape = params[n].shape.join('\u00d7');
    ctx.fillText(shape, x + cellW/2, y + cellH/2 + 6);

    hmRects.push({ x, y, w: cellW, h: cellH, name: n, value: values[i], level: 1, data: params[n] });
  });
}

function drawLevel2(width, dpr) {
  if (!hmBlock || !hmMatrix || !HIERARCHY[hmBlock]?.[hmMatrix]) return;
  const data = HIERARCHY[hmBlock][hmMatrix];
  const cells = data.cells;
  if (!cells || !cells.length) return;

  const numRows = cells.length;
  const numCols = cells[0].length;
  const cellSize = Math.max(2, Math.min(8, Math.floor(width / numCols)));
  const drawW = numCols * cellSize;
  const drawH = numRows * cellSize;
  const height = drawH + 4;

  canvas.width = Math.max(width, drawW) * dpr;
  canvas.height = height * dpr;
  canvas.style.width = Math.max(width, drawW) + 'px';
  canvas.style.height = height + 'px';
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, width, height);

  // Find min/max for normalization
  let minVal = Infinity, maxVal = -Infinity;
  cells.forEach(row => row.forEach(v => { minVal = Math.min(minVal, v); maxVal = Math.max(maxVal, v); }));
  const range = maxVal - minVal || 1;
  const cmap = getCmap();
  hmRects = [];

  const offX = Math.max(0, (width - drawW) / 2);
  for (let r = 0; r < numRows; r++) {
    for (let c = 0; c < numCols; c++) {
      const t = (cells[r][c] - minVal) / range;
      ctx.fillStyle = sampleColormap(cmap, t);
      ctx.fillRect(offX + c * cellSize, r * cellSize, cellSize, cellSize);
    }
  }

  // Single rect for tooltip
  hmRects.push({ x: offX, y: 0, w: drawW, h: drawH, name: hmMatrix, level: 2, cells, numRows, numCols, cellSize, offX, minVal, range });
}

function updateBreadcrumb() {
  let html = '<span data-level="0">Model</span>';
  if (hmLevel >= 1) html += `<span class="sep">\u203a</span><span data-level="1">${hmBlock}</span>`;
  if (hmLevel >= 2) html += `<span class="sep">\u203a</span><span data-level="2">${hmMatrix}</span>`;
  breadcrumb.innerHTML = html;
  breadcrumb.querySelectorAll('span[data-level]').forEach(s => {
    s.addEventListener('click', () => {
      const level = parseInt(s.dataset.level);
      if (level < hmLevel) {
        hmLevel = level;
        if (level === 0) { hmBlock = null; hmMatrix = null; }
        if (level === 1) { hmMatrix = null; }
        drawHeatmap();
      }
    });
  });
}

canvas.addEventListener('click', e => {
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;

  for (const r of hmRects) {
    if (x >= r.x && x <= r.x + r.w && y >= r.y && y <= r.y + r.h) {
      if (r.level === 0) { hmBlock = r.name; hmLevel = 1; drawHeatmap(); }
      else if (r.level === 1) { hmMatrix = r.name; hmLevel = 2; drawHeatmap(); }
      break;
    }
  }
});

canvas.addEventListener('mousemove', e => {
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;

  for (const r of hmRects) {
    if (mx >= r.x && mx <= r.x + r.w && my >= r.y && my <= r.y + r.h) {
      tooltip.style.display = 'block';
      tooltip.style.left = (e.clientX + 12) + 'px';
      tooltip.style.top = (e.clientY - 8) + 'px';

      if (r.level === 2 && r.cells) {
        const col = Math.floor((mx - r.offX) / r.cellSize);
        const row = Math.floor(my / r.cellSize);
        if (row >= 0 && row < r.numRows && col >= 0 && col < r.numCols) {
          const val = r.cells[row][col];
          tooltip.innerHTML = `<b>${r.name}</b><br>Cell [${row}, ${col}]<br>Value: ${val.toFixed(6)}`;
        }
      } else {
        let html = `<b>${r.name}</b><br>`;
        if (r.data) html += `Shape: ${r.data.shape.join('\u00d7')}<br>Norm: ${r.data.norm.toFixed(4)}<br>Block size: ${r.data.block_size}`;
        else html += `Value: ${r.value.toFixed(6)}`;
        tooltip.innerHTML = html;
      }
      return;
    }
  }
  tooltip.style.display = 'none';
});

canvas.addEventListener('mouseleave', () => { tooltip.style.display = 'none'; });

document.addEventListener('keydown', e => {
  if (e.key === 'Escape' && hmLevel > 0) {
    hmLevel--;
    if (hmLevel === 0) { hmBlock = null; hmMatrix = null; }
    if (hmLevel === 1) { hmMatrix = null; }
    drawHeatmap();
  }
});

document.getElementById('hm-mode').addEventListener('change', drawHeatmap);
document.getElementById('hm-cmap').addEventListener('change', drawHeatmap);
window.addEventListener('resize', () => { if (document.querySelector('.tab[data-tab="heatmap"]').classList.contains('active')) drawHeatmap(); });

// --- Generations ---
const genContent = document.getElementById('gen-content');
if (Object.keys(GENERATIONS).length > 0) {
  Object.entries(GENERATIONS).forEach(([prompt, outputs]) => {
    let html = `<div class="gen-section"><div class="gen-prompt">\u276f ${prompt}</div>`;
    Object.entries(outputs).forEach(([key, text]) => {
      const label = key.replace('temp_', 'Temperature ');
      html += `<div class="gen-label">${label}</div><div class="gen-output">${text.replace(/</g, '&lt;').replace(/>/g, '&gt;')}</div>`;
    });
    html += '</div>';
    genContent.innerHTML += html;
  });
} else {
  genContent.innerHTML = '<p style="color:var(--text-tertiary)">No generations available (run with --prompts to enable)</p>';
}

// Initial draw
setTimeout(drawHeatmap, 100);
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Checkpoint Analyzer")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--model", required=True, help="Path to model .py file")
    parser.add_argument("--class-name", required=True, help="Model class name")
    parser.add_argument("--dataset", default=None, help="Pre-tokenized .bin for gradient analysis")
    parser.add_argument("--prompts", nargs="*", default=[], help="Prompts for text generation")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-tokens", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Step 1: Load checkpoint, extract state_dict, free checkpoint ---
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    step = ckpt.get("step", 0) if isinstance(ckpt, dict) else 0
    total_tokens = ckpt.get("total_tokens", 0) if isinstance(ckpt, dict) else 0
    del ckpt
    print(f"  Step: {step}, Tokens: {total_tokens:,}")

    # --- Step 2: Weight analysis (no model needed) ---
    print("Analyzing weights...")
    weight_data = analyze_weights(state_dict)
    hierarchy = build_block_hierarchy(state_dict)
    total_params = sum(v["numel"] for v in weight_data.values())
    print(f"  {len(weight_data)} parameters, {total_params / 1e6:.1f}M total")

    # --- Step 3: Build model, load weights ---
    print("Loading model...")
    sys.path.insert(0, ".")
    model = load_model_from_file(args.model, args.class_name)
    model.load_state_dict(state_dict, strict=False)
    del state_dict
    model = model.to(device)

    # --- Step 4: Gradient analysis ---
    grad_data = {}
    eval_loss = None
    if args.dataset:
        print("Analyzing gradients...")
        grad_data, eval_loss = analyze_gradients(model, args.dataset, batch_size=args.batch_size)
        print(f"  {len(grad_data)} grad norms, eval loss: {eval_loss:.4f}")

    # --- Step 5: Parameter breakdown ---
    breakdown = param_breakdown(model)

    # --- Step 6: Text generation ---
    generations = {}
    if args.prompts:
        print(f"Generating text ({len(args.prompts)} prompts)...")
        generations = generate_samples(model, args.prompts, max_tokens=args.num_tokens)

    del model

    # --- Step 7: Write outputs ---
    ckpt_path = Path(args.checkpoint)
    out_dir = ckpt_path.parent / f"{ckpt_path.stem}_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    summary = {
        "step": step,
        "total_tokens": total_tokens,
        "total_params": total_params,
        "loss": eval_loss,
        "checkpoint": str(ckpt_path),
    }

    # Save JSON files
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "weight_analysis.json", "w") as f:
        json.dump(weight_data, f)
    if grad_data:
        with open(out_dir / "grad_analysis.json", "w") as f:
            json.dump(grad_data, f)
    if generations:
        with open(out_dir / "generations.json", "w") as f:
            json.dump(generations, f, indent=2)

    # Generate PNG plots
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Weight norms
        fig, ax = plt.subplots(figsize=(12, 4))
        names = [k for k in weight_data if weight_data[k]["numel"] > 100]
        norms = [weight_data[k]["norm"] for k in names]
        ax.bar(range(len(names)), norms, color='#59c4b0', alpha=0.8)
        ax.set_ylabel("L2 Norm")
        ax.set_title("Weight Norms by Layer")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.split('.')[-1][:12] for n in names], rotation=45, ha='right', fontsize=6)
        plt.tight_layout()
        plt.savefig(plots_dir / "weight_norms.png", dpi=150, bbox_inches='tight')
        plt.close()

        if grad_data:
            fig, ax = plt.subplots(figsize=(12, 4))
            gnames = list(grad_data.keys())
            gnorms = [grad_data[k]["grad_norm"] for k in gnames]
            ax.bar(range(len(gnames)), gnorms, color='#d4a843', alpha=0.8)
            ax.set_ylabel("Grad Norm")
            ax.set_title("Gradient Norms by Layer")
            ax.set_xticks(range(len(gnames)))
            ax.set_xticklabels([n.split('.')[-1][:12] for n in gnames], rotation=45, ha='right', fontsize=6)
            plt.tight_layout()
            plt.savefig(plots_dir / "grad_norms.png", dpi=150, bbox_inches='tight')
            plt.close()

        # Param breakdown pie
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(breakdown.values(), labels=breakdown.keys(), autopct='%1.1f%%', colors=plt.cm.viridis(np.linspace(0.2, 0.9, len(breakdown))))
        ax.set_title("Parameter Breakdown")
        plt.tight_layout()
        plt.savefig(plots_dir / "param_breakdown.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  PNGs saved to {plots_dir}/")
    except ImportError:
        print("  matplotlib not available, skipping PNGs")

    # Build and save HTML report
    # Strip cells from hierarchy for grad data matching (grad_data uses full param names)
    report_html = build_report_html(weight_data, grad_data, hierarchy, breakdown, generations, summary)
    with open(out_dir / "report.html", "w") as f:
        f.write(report_html)

    print(f"\nAnalysis complete: {out_dir}/")
    print(f"  report.html  — interactive heatmap explorer")
    print(f"  summary.json — machine-readable metrics")
    if grad_data:
        print(f"  grad_analysis.json — per-layer gradient stats")
    if generations:
        print(f"  generations.json — text samples")


if __name__ == "__main__":
    main()
