---
title: "BPB + MFU Analysis for 15-Minute Bakeoffs"
domain: architectures
type: reference
status: active
related:
  - mad_llm_scientist/EVALUATION_GUIDE.md
tags: [%bpb, %mfu, %metrics, %analysis]
---

# BPB + MFU Analysis for 15-Minute Bakeoffs

## Goal

Use a fixed 15-minute training budget to rank candidate architectures by **actual learning progress on target hardware**, not by theoretical elegance.

The right question is:

> Which model learns best per wall-clock second on Strix Halo?

---

## Primary metric

Use validation bits-per-byte after the fixed budget:

$$
\text{val\_bpb} = \frac{\mathcal{L}_{CE}}{\ln 2} \cdot \frac{N_{\text{tokens}}}{N_{\text{bytes}}}
$$

This captures:

- **learning efficiency**: signal extracted per token
- **throughput**: how many tokens fit in 15 minutes
- **tokenizer fairness**: byte normalization makes vocabularies more comparable
- **parameter efficiency**: smaller/faster models can win if they learn well enough

---

## MFU meaning

MFU = Model FLOP Utilization:

$$
\text{MFU} = \frac{\text{Achieved FLOPS}}{\text{Theoretical Peak FLOPS}}
$$

Observed values:

| Mode | MFU | Wasted compute |
|---|---:|---:|
| Eager | ~16% | 84% |
| `torch.compile` | ~30% | 70% |

Interpretation:

- eager mode wastes a lot of time on Python dispatch and kernel launch overhead
- `torch.compile` helps, but most hardware capacity is still unused
- higher MFU means more useful work in the same time budget

---

## Main bottlenecks

### 1. Framework overhead
Eager PyTorch launches many small kernels with host-side overhead between them. `torch.compile` reduces this by fusing ops into larger graphs.

### 2. Memory bandwidth
Many transformer ops are memory-bound on Strix Halo, especially:

| Operation | Typical intensity | Bound by |
|---|---:|---|
| LayerNorm | ~5–10 FLOP/byte | memory |
| GELU / activations | ~1 FLOP/byte | memory |
| Softmax | ~5 FLOP/byte | memory |
| Residual add | ~0.5 FLOP/byte | memory |

The ridge point is roughly:

$$
\frac{15\ \text{TFLOPS}}{170\ \text{GB/s}} \approx 88\ \text{FLOP/byte}
$$

So only sufficiently dense compute tends to saturate the device.

### 3. No matrix cores
The 8060S has no MFMA/tensor-core-style hardware. Matmuls run on general-purpose vector units, so peak throughput is lower and software cannot fully recover that loss.

---

## What this means for architecture search

MFU is not just a runtime metric. It is part of the architecture ranking signal.

Architectures that reduce memory traffic, launch count, and scatter/gather overhead will usually win this bakeoff even if they look less elegant on paper.

### Architectures likely to help

| Property | MFU impact | Why |
|---|---|---|
| Fewer, larger matmuls | Higher | better arithmetic intensity |
| RMSNorm over LayerNorm | Higher | simpler and more fuseable |
| SwiGLU / GeGLU | Slightly higher | fuses gating and activation |
| GQA over MHA | Higher | less KV traffic |
| Linear attention / state-space | Much higher | avoids quadratic attention bottlenecks |
| Short convolutions | Higher | high reuse, easy to fuse |
| Deep-narrow stacks | Lower | more sequential launches |
| Mixture of Experts | Lower | scatter/gather hurts locality |

---

## Practical ways to improve MFU

### Low effort
- increase batch size as much as memory allows
- use `torch.compile`
- prefer low-overhead compile modes for short bakeoffs
- avoid accidental DDP or synchronization overhead
- use fused optimizer variants where available

### Medium effort
- use attention implementations that avoid materializing full `S × S` matrices
- fuse normalization + residual + dropout where possible
- use gradient accumulation to simulate larger batches
- verify that fusion actually happens by inspecting compiled graphs/kernels

---

## Recommended training pattern

```python
accumulation_steps = 4

for i, batch in enumerate(dataloader):
    with torch.amp.autocast("cuda", dtype=torch.float16):
        loss = model(**batch).loss / accumulation_steps

    scaler.scale(loss).backward()

    if (i + 1) % accumulation_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
```

This helps by:

- simulating larger batches
- reducing optimizer-step frequency
- improving utilization without increasing activation memory too much

---

## Mental model for the bakeoff

Fixed:

- dataset
- hardware
- time budget
- precision

Variable:

- architecture
- hyperparameters
- implementation details

Objective:

$$
\text{val\_bpb} = f(\text{tokens seen}, \text{learning per token}, \text{generalization})
$$

Improving MFU means:

- more tokens seen in 15 minutes
- more learning per wall-clock second
- effectively more data for free

---

## Bottom line

For this bakeoff, the right ranking criterion is:

> which model learns best per wall-clock second on Strix Halo?

Optimize for:

- `val_bpb`
- MFU
- fusion
- memory efficiency
- larger effective batches

Avoid:

- launch-heavy designs
- memory-traffic-heavy designs
- scatter/gather-heavy designs

In short: **every point of MFU you recover is more data, more learning, and a better architecture choice under the fixed 15-minute budget.**