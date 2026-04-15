---
title: "Training Monitor: Live Dashboard + Checkpoint Analyzer + Callbacks"
domain: training
type: spec
status: active
related:
  - scripts/train_ddp.py
  - halo_training/evaluate.py
  - halo_training/metrics.py
  - halo_training/callbacks.py
  - knowledge/training/training_antipatterns.md
tags: [%training, %monitoring, %dashboard, %checkpoint-analysis, %visualization, %heatmap]
---

# Training Monitor: Live Dashboard + Checkpoint Analyzer + Callbacks

## Problem

Training ARGUS-PRIME across multi-day DDP runs with only `train_log.jsonl` as output. No visual monitoring, no per-layer diagnostics, no checkpoint quality analysis. Need to detect issues (instability, dead layers, overfitting) before they waste hours of compute.

## Architecture

Three components, all reading the same data:

```
train_log.jsonl ──────────────► Live Dashboard (localhost:8050)
grad_log.jsonl  ──────────────►   (auto-detects extra logs)
weight_log.jsonl ─────────────►
val_log.jsonl ────────────────►

checkpoint.pt ────────────────► Checkpoint Analyzer
                                  └──► step_N_analysis/
                                        ├── report.html (heatmap explorer)
                                        ├── summary.json
                                        └── plots/*.png
```

---

## Component 1: Live Dashboard

**File:** `scripts/training_dashboard.py` (~300 lines)

**Tech:** Flask backend + Plotly.js frontend, single file with embedded HTML template.

**Run:** `python scripts/training_dashboard.py --log-dir checkpoints/argus_prime_cc_ddp_v2/`

Opens browser on Machine 0 at `localhost:8050`.

### Panels (6)

**Row 1 (primary):**
- **Loss & BPB curve** — dual y-axis, raw + EMA smoothing (configurable alpha), epoch boundaries marked
- **Gradient norm** — with horizontal lines at clip threshold and warning zones

**Row 2 (training health):**
- **Learning rate schedule** — shows position in cosine decay
- **tok/s & MFU** — dual y-axis, stability band (mean ± 1σ)

**Row 3 (system):**
- **Memory usage** — GPU memory over time
- **Step rate** — steps/sec derived from step intervals, highlights slowdowns

### Controls
- Smoothing slider (EMA window)
- Time range selector (last 1h, 6h, 24h, all)
- Multi-run dropdown (select different `--log-dir` paths to overlay runs)

### Auto-detection of extra logs
When callback logs exist in the same directory:
- `grad_log.jsonl` → adds per-layer gradient heatmap-over-time panel
- `weight_log.jsonl` → adds per-layer weight norm evolution panel
- `val_log.jsonl` → overlays validation loss on training loss curve
- `generations/` → adds text samples tab

### Data refresh
- Polls `/api/data` every 30 seconds
- Backend reads `train_log.jsonl` incrementally (seeks to last read position)
- Timestamps derived from step count + known steps/sec (current logs lack timestamps)

### Dependencies
- `flask` (pip install flask)

---

## Component 2: Checkpoint Analyzer

**File:** `scripts/analyze_checkpoint.py` (~400 lines)

**Run:**
```bash
python scripts/analyze_checkpoint.py \
    --checkpoint checkpoints/.../step_9000.pt \
    --model models/argus_prime.py --class-name ArgusPrime \
    --dataset datasets/common_crawl_sample.bin \
    --prompts "The meaning of life is" "In 1945, the" "def fibonacci(" \
    --batch-size 4 --num-tokens 100
```

### Memory-efficient pipeline

```python
# 1. Load checkpoint, extract state_dict, free checkpoint immediately
ckpt = torch.load(path, map_location="cpu")
state_dict = ckpt["model_state_dict"]
del ckpt  # free optimizer state, step counter, etc.

# 2. Compute weight analysis directly from state_dict (no model needed)
weight_norms = analyze_weights(state_dict)
heatmap_data_weights = build_hierarchical_heatmap(state_dict, mode="weights")

# 3. Build model, load weights, free state_dict duplicate
model = load_model(...)
model.load_state_dict(state_dict)
del state_dict

# 4. Gradient analysis (one forward+backward on small batch)
grad_data = analyze_gradients(model, batch)
heatmap_data_grads = build_hierarchical_heatmap(model, mode="grads")

# 5. Text generation
generations = generate_samples(model, prompts)

# 6. Free model
del model
```

Peak memory: ~0.8 GB for ARGUS-PRIME (model + one batch activations).

### What it computes

**Weight analysis** (from state_dict, before model load):
- Per-layer weight norms (L2)
- Per-layer min/max/mean/std
- Per-block aggregate norms
- Hierarchical heatmap data at all 3 zoom levels

**Gradient analysis** (model loaded, one forward+backward):
- Per-layer gradient norms on a small batch
- Grad-to-weight ratio per layer (which params are actively learning)
- Hierarchical heatmap data for gradients

**Parameter breakdown:**
- Count by module type (attention, FFN, conv, TTT, embeddings)
- Percentage data for pie chart

**Text generation:**
- Fixed prompts, temperature 0.0 (greedy) + 0.7 (sampling)
- Stored as text in JSON

### Output structure

```
step_9000_analysis/
├── report.html          # self-contained: charts + hierarchical heatmap explorer
├── summary.json         # top-level metrics for cross-checkpoint comparison
├── weight_analysis.json # full per-layer stats + downsampled matrices
├── grad_analysis.json   # full per-layer grad stats + downsampled matrices
├── generations.json     # prompt → output pairs
└── plots/
    ├── weight_norms.png
    ├── grad_norms.png
    └── param_breakdown.png
```

### Hierarchical Heatmap Explorer (in report.html)

Interactive Canvas-based visualization with semantic zoom:

```
Level 0: Model Overview
┌─────────┬─────────┬─────────┬─────────┐
│ Block 0  │ Block 1  │ Block 2  │  ...16  │  ← each block = 1 colored square
│  (avg)   │  (avg)   │  (avg)   │         │     color = normalized aggregate
└─────────┴─────────┴─────────┴─────────┘
                    click Block 1
                        ↓
Level 1: Block Internals
┌──────┬──────┬──────┬──────┬──────┐
│ w_qkv │ w_out │ w_gate│ w_up  │ w_down│  ← each matrix = 1 colored square
│768×2304│768×768│768×2816│      │       │     color = normalized per-matrix
└──────┴──────┴──────┴──────┴──────┘
                    click w_qkv
                        ↓
Level 2: Matrix Cells
┌─┬─┬─┬─┬─┬─┬─┬─┐
│ │ │ │ │ │ │ │ │  ← individual weight values
├─┼─┼─┼─┼─┼─┼─┼─┤     color = per-cell value
│ │ │ │ │ │ │ │ │     (downsampled for large matrices)
└─┴─┴─┴─┴─┴─┴─┴─┘
         Escape → back to Level 1
```

**Interaction:**
- Click → zoom into element
- Escape → zoom back out
- Hover → tooltip with exact values (name, shape, norm, min/max/mean)
- Toggle: weights / gradients / grad-to-weight ratio
- Colormap selector (viridis, magma, RdBu diverging)

**Downsampling strategy (Level 2):**
- Matrices up to 128×128: show raw values
- Larger matrices: downsample to 128×128 blocks, each cell = average of its block
- Tooltip shows original dims + block size

**Normalization (precomputed):**
- Level 0: normalized across all blocks
- Level 1: normalized across all matrices in a block
- Level 2: normalized across cells in a matrix
- All normalization values stored in analysis JSON

**Tech:** Vanilla Canvas API (no library) + embedded JSON data. Self-contained HTML file, no server needed to view.

---

## Component 3: Training Callbacks (Future Runs)

**Modified file:** `scripts/train_ddp.py`

Four optional callbacks, each enabled by CLI flag:

### `--log-per-layer-grads N`
Every N steps, log per-layer gradient norms to `grad_log.jsonl`:
```json
{"step": 100, "grads": {"blocks.0.w_qkv": 0.012, "blocks.0.w_out": 0.008, ...}}
```
Overhead: ~0 (reads `.grad.norm()` values already computed by backward pass)

### `--log-weight-norms N`
Every N steps, log per-layer weight norms to `weight_log.jsonl`:
```json
{"step": 100, "weights": {"blocks.0.w_qkv": 1.234, "blocks.0.w_out": 0.987, ...}}
```
Overhead: ~0 (reads `.data.norm()`)

### `--generate-at-checkpoint`
At each checkpoint save, generate text from 5 fixed prompts, save to `generations/step_9000.json`:
```json
{"prompts": [...], "outputs_greedy": [...], "outputs_sampled": [...]}
```
Overhead: ~2-3 sec per checkpoint (5 prompts × 100 tokens)

### `--val-every N --val-dataset PATH`
Every N steps, run eval on held-out slice, log to `val_log.jsonl`:
```json
{"step": 1000, "val_loss": 4.12, "val_bpb": 1.651}
```
Overhead: depends on slice size (~1 sec for 1000 tokens)

---

## Phased Delivery

| Phase | Component | Usable on |
|-------|-----------|-----------|
| 1 | Live Dashboard | Current DDP run (reads existing train_log.jsonl) |
| 2 | Checkpoint Analyzer | Existing checkpoints from current run |
| 3 | Training Callbacks | Next training run (modifications to train_ddp.py) |

Phase 1 and 2 are independent and can be built in parallel. Phase 3 depends on the current run completing first (don't modify running script).

## Files Created
- `scripts/training_dashboard.py` (~300 lines)
- `scripts/analyze_checkpoint.py` (~400 lines)

## Files Modified
- `scripts/train_ddp.py` (add 4 CLI flags + callback implementations, ~100 lines added)

## Dependencies
- `flask` (dashboard server)
- `matplotlib` (PNG chart generation in analyzer)
- Both likely already installed. No external accounts or services.
