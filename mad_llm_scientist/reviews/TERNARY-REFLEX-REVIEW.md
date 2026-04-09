# Ternary Reflex Review

## Summary

The current ternary-reflex architecture is a promising fit for Strix Halo constraints:

- `TernaryLinear` with Straight-Through Estimator (STE)
- `ElementWiseGRU` for matmul-free sequence mixing
- `ReflexBlock` as an 8-layer ternary stack
- `GeniusBlock` as a parallel hybrid with convolution + Griffin recurrence
- `EntropyRouter` for caveman routing
- `DualPathModel` as the full integration
- Total ternary weight footprint: **1.17 MB**, which fits in L2 cache

## Implementation Status

### Completed
- 30/30 unit tests passing
- Small model training is working
- Gradient explosion has been identified and partially mitigated

### Known Good Configuration
The small setup trains stably:

- Vocabulary: `<= 10K`
- Hidden size: `<= 128`
- Learning rate: `1e-3`
- Gradient clipping: `0.1`
- `nan_to_num()` applied before clipping

Observed result:
- Loss improved from `30 → 23` over 50 steps

### Current Pipeline
- `training/train_ternary_reflex.py`
  - uses `torch.compile`
  - uses `fp16`
  - uses phased training
- `training/quick_test.py`
  - validation and smoke testing

---

## Main Problem

### Gradient Explosion at Full Scale
The ternary STE combined with large-vocabulary cross-entropy (`50K` vocab) causes extreme gradient explosions, reaching around `10^36`, which crashes the GPU on ROCm.

This is likely not a ROCm-specific bug by itself. The more probable cause is the combination of:

1. `fp16` numeric range limits
2. STE instability
3. Large-vocabulary softmax early in training

---

## Diagnosis

### Likely Failure Drivers
- **FP16 overflow**
  - PyTorch warns that FP16 can overflow gradients
- **STE sensitivity**
  - ternary surrogate gradients can be unstable
- **Large output head**
  - full `50K` softmax makes the loss path heavy and amplifies instability

### Important Interpretation
The immediate issue is probably **not** “ROCm is bad at this.”

It is more likely:

> `FP16 + STE + full 50K softmax too early`

---

## Recommended Fix Order

### 1. Switch FP16 to BF16
Use BF16 if possible.

Why:
- larger numeric range
- lower overflow risk
- better fit for unstable training

### 2. Keep Master Weights in Full Precision
Use real-valued latent weights for optimization and quantize only in the forward pass.

Why:
- this matches standard quantization-aware training practice
- the quantized operator itself is not meant to be directly optimized

### 3. Make STE Backward Less Aggressive
Use a clipped or bounded STE instead of a pure identity surrogate.

Why:
- gradients should saturate outside a bounded interval
- this reduces runaway updates from large hidden activations

### 4. Lower the Learning Rate Further
For the full model, treat `1e-4` as an upper bound.

Suggested range:
- `3e-5` to `1e-5`

Use warmup.

### 5. Clip Gradients Before the Optimizer Step
Use strict non-finite checks during debugging.

Recommended:
- `clip_grad_norm_`
- `error_if_nonfinite=True`

Keep `nan_to_num()` only as a temporary guard.

### 6. Add Label Smoothing
Start with a small value such as:

- `0.01`
- `0.02`
- `0.05`

Why:
- reduces early overconfidence
- can soften output-layer gradients

### 7. Decouple Core Validation from Full Vocabulary
Validate the ternary backbone on a smaller or capped vocabulary first.

Why:
- large-vocab LM training is inherently harder
- full-softmax cost and instability both grow with vocab size

---

## Practical Stabilization Recipe

### Training Changes to Try Next
- BF16 autocast instead of FP16
- FP32 master params
- clipped STE
- smaller LR with warmup
- label smoothing
- strict grad checks

### Example Training Loop
```python
# Key changes:
# - bf16 autocast, not fp16
# - fp32 master params in optimizer
# - clipped STE
# - tiny LR + warmup
# - label smoothing
# - strict grad checks

scaler = None  # for bf16 you often do not need GradScaler

criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.02)

for step, batch in enumerate(loader):
    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(batch["input_ids"])
        loss = criterion(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            batch["input_ids"][:, 1:].reshape(-1),
        )

    loss.backward()

    total_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=0.05,
        error_if_nonfinite=True,
    )

    optimizer.step()
    scheduler.step()
```

### Example Ternary STE
```python
class TernarySTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, thresh=0.5, clip_val=1.0):
        ctx.save_for_backward(w)
        ctx.clip_val = clip_val

        out = torch.zeros_like(w)
        out = torch.where(w > thresh, torch.ones_like(out), out)
        out = torch.where(w < -thresh, -torch.ones_like(out), out)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (w,) = ctx.saved_tensors
        clip_val = ctx.clip_val
        mask = (w.abs() <= clip_val).to(grad_out.dtype)
        return grad_out * mask, None, None
```

---

## Suggested Test Sequence

### Phase 1: Numeric Stability Check
1. Run **BF16 + current model + same small vocab**
2. Compare against FP16
3. Confirm whether overflow is the main issue

### Phase 2: Full Model Stabilization
4. Run **BF16 + clipped STE + LR `3e-5` + warmup + clip `0.05`**
5. Add **label smoothing `0.02`**

### Phase 3: Progressive Unfreezing
6. If instability remains, freeze the ternary core
7. Train embeddings + head first
8. Unfreeze gradually

### Phase 4: Output Head Alternatives
9. If the vocab head is still problematic, test:
   - more efficient CE
   - approximate large-vocab strategies
   - reduced/debug vocabulary

---

## Debugging Checklist

Add a minimal debug harness that logs:

- per-layer gradient norms for the first 200 steps
- first non-finite gradient source
- embedding gradient norm
- LM head gradient norm
- first ternary block gradient norm
- max absolute logits before CE

This will help identify whether the explosion starts in:

- the **LM head / CE path**
- or the **ternary recurrent stack**

---

## Concise Action Items

1. Replace `fp16` with `bf16`
2. Keep full-precision master weights
3. Use clipped STE instead of identity STE
4. Lower LR to `3e-5` to `1e-5` with warmup
5. Clip gradients with non-finite checks
6. Add label smoothing around `0.02`
7. Validate with smaller vocab before full `50K`
8. Log per-layer gradient norms and max logits

---

## Bottom Line

The architecture itself is plausible and efficient enough to be worth pursuing.

The current blocker is training stability, not the core design.

The next best move is to make the training recipe numerically safer before changing the model further.

---

## Note

Generated code is optional, not a direct solution.