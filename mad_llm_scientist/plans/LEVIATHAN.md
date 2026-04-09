# LEVIATHAN

**Train a 1-3B Teacher on Strix Halo, Distill into 250M Student**

## Hypothesis

Knowledge distillation from a larger teacher dramatically improves small model quality (proven by LFM2, GPT-4 → smaller models, etc.). Our current plans assume GPT-2 Medium (355M) as teacher — but it's trained on WebText, not our data. What if we train our OWN teacher (1-3B) on Strix Halo's 108GB unified memory, then distill into our 250M student?

**Feasibility:** MegaTrain (2604.05091) shows an RTX 3090 (24GB) trains 14B at 30 TFLOPS. Strix Halo has 4.5x more memory (108GB unified). Training a 1B teacher is comfortable. The training machine has `flash_attn` built for ROCm.

**Source:** MegaTrain techniques (COLOSSEUM.md) + LFM2 distillation protocol (2511.23404)

---

## Why Our Own Teacher Beats GPT-2 Medium

| | GPT-2 Medium (pre-trained) | Our 1B Teacher |
|---|---|---|
| Params | 355M | **1B** (3x larger) |
| Data | WebText (general web) | **BabyLM** (same as student!) |
| Distribution match | Poor (different data) | **Perfect** (identical data) |
| Architecture | Transformer (attention) | **LLaMA-style** (flash-attn on ROCm) |
| Logit quality | Good but mismatched | **Best** (larger + matched) |

The teacher trained on the SAME data produces logits that perfectly match the student's learning task. No distribution mismatch.

---

## Teacher Architecture

The training machine has `flash_attn` for ROCm. Use a standard LLaMA-style transformer — the BEST architecture for knowledge distillation (proven by LFM2).

| Parameter | 1B Teacher | 3B Teacher (optional) |
|-----------|-----------|---------------------|
| d_model | 2048 | 2560 |
| n_layers | 24 | 32 |
| n_heads | 16 (GQA, 4 KV heads) | 20 (GQA, 4 KV heads) |
| ffn_inner | 5504 (SwiGLU) | 6912 (SwiGLU) |
| vocab_size | 50257 | 50257 |
| RoPE theta | 500,000 | 500,000 |
| Attention | flash-attn (ROCm build) | flash-attn |
| Params | ~1.0B | ~3.0B |

## Memory Feasibility (Strix Halo, 128 GB (~116 GB GPU-visible))

| Component | 1B Teacher | 3B Teacher |
|-----------|-----------|-----------|
| BF16 weights | 2 GB | 6 GB |
| FP32 optimizer (m, v) | 8 GB | 24 GB |
| Activations (batch=16, seq=512) | ~2 GB | ~4 GB |
| **Total** | **~12 GB** | **~34 GB** |
| Fits in 128 GB (~116 GB GPU-visible)? | **YES (11%)** | **YES (31%)** |

With COLOSSEUM's CPU-side optimizer: optimizer states in CPU-preferred memory. GPU only needs weights + activations.

---

## Two-Phase Protocol

### Phase 1: Train Teacher (30-60 min)

Train 1B LLaMA-style teacher on BabyLM using COLOSSEUM techniques.

| Parameter | Value |
|-----------|-------|
| Model | 1B LLaMA (24L, d=2048, GQA) |
| Dataset | `datasets/babylm-strict-small/` (~16M tokens) |
| Epochs | 4-8 (multi-epoch mandatory for 16M tokens) |
| Batch | 16 × 512 = 8K tokens/step |
| LR | 3e-4 cosine → 3e-5, 200-step warmup |
| Optimizer | DeepSpeed CPUAdam (ROCm) or manual CPU AdamW |
| Precision | BF16 mixed + flash-attn |
| Checkpointing | Every 4 layers (COLOSSEUM technique) |
| Est. throughput | ~5K tok/s (1B is 4x larger than 250M) |
| Est. time | 16M × 6 epochs / 5K = ~19K seconds ÷ 60 = ~32 min |

**Save checkpoint** after training. The teacher is frozen for Phase 2.

### Phase 2: Distill into Student (15 min)

Train any 250M student architecture with KD loss from the frozen 1B teacher.

```python
# Both models in unified memory simultaneously
teacher = load_checkpoint("teacher_1b.pt").eval()  # 2 GB
student = StudentArchitecture()                      # 500 MB
# Total: 2.5 GB — trivially fits

for batch in dataloader:
    input_ids = batch["input_ids"]
    
    # Teacher forward (frozen, no grad)
    with torch.no_grad():
        teacher_logits = teacher(input_ids).logits    # zero-copy on unified memory
    
    # Student forward
    student_logits = student(input_ids).logits
    
    # LFM2's tempered decoupled Top-K distillation
    T = 2.0
    K = 32
    
    # Top-K truncation on teacher logits
    topk_vals, topk_idx = teacher_logits.topk(K, dim=-1)
    teacher_trunc = torch.full_like(teacher_logits, float('-inf'))
    teacher_trunc.scatter_(-1, topk_idx, topk_vals)
    
    # Combined loss
    ce_loss = F.cross_entropy(student_logits.view(-1, V), input_ids[:, 1:].view(-1))
    kd_loss = T**2 * F.kl_div(
        F.log_softmax(student_logits / T, dim=-1),
        F.softmax(teacher_trunc / T, dim=-1),
        reduction='batchmean'
    )
    loss = 0.7 * ce_loss + 0.3 * kd_loss
```

**Unified memory advantage:** Teacher (2 GB) + Student (0.5 GB) + optimizer (2 GB) + activations (2 GB) = ~6.5 GB. On 128 GB (~116 GB GPU-visible), this is 6%. Both models live in the same address space. Zero-copy teacher logits.

---

## Combined Timeline

| Time | Activity | Model | Tokens |
|------|----------|-------|--------|
| 0-32 min | Train 1B teacher | Teacher | ~96M token-exposures (16M × 6 epochs) |
| 32-47 min | Distill into 250M student | Student + frozen teacher | ~15-22M tokens (COLOSSEUM batch) |
| **Total** | **~47 min** | | |

**vs standard 15-min training:** LEVIATHAN takes 3x longer but the student gets knowledge from a model 4x its size trained for 6x more effective tokens. The quality improvement should be substantial.

**vs GPT-2 Medium teacher:** Same 15-min distillation phase, but our teacher is 3x larger and perfectly data-matched.

---

## Compatibility

| Student Architecture | Compatible | Notes |
|---|---|---|
| ALL 17+ architectures | YES | KD is architecture-agnostic |
| AMADEUS / MAESTRO | YES | Best candidates (simplest base) |
| Caveman LFM | YES | Tier 1, first to try |
| With Self-Curriculum | YES | Curriculum sampling during Phase 2 |
| With Lottery Forge | YES | KD during Temper phase |

---

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| 1B teacher underfits on 16M tokens | MEDIUM | 6-8 epochs. Chinchilla-optimal for 1B is 20B tokens, but KD only needs the teacher's RELATIVE ranking of tokens, not absolute quality. |
| Teacher training takes too long | MEDIUM | 1B @ 5K tok/s: 32 min for 6 epochs. Acceptable. Reduce to 1 epoch (5 min) if needed — even partially trained teachers help. |
| KD loss interferes with CE loss | LOW | LFM2's tempered Top-K decoupling handles this. The 0.7/0.3 split is validated. |
| flash-attn ROCm issues | LOW | Already installed on training machine. If issues: use torch SDPA as fallback. |

## Success Criteria

1. 1B teacher achieves loss < 3.5 on BabyLM (teacher is learning)
2. Student with LEVIATHAN KD outperforms student with CE-only by > 5% on val_bpb
3. Student with LEVIATHAN KD outperforms student with GPT-2 Medium KD
4. Total wall-clock (teacher + distillation) < 60 min
5. Both models fit in unified memory simultaneously (verified)

## Implementation Roadmap

1. Implement 1B LLaMA-style teacher (reuse HuggingFace `LlamaForCausalLM` config)
2. Train teacher with COLOSSEUM techniques (CPU optimizer, activation checkpoint)
3. Save teacher checkpoint
4. Implement KD loss (tempered decoupled Top-K, from LFM2)
5. Train student (any architecture) with KD from frozen teacher
6. Ablation: CE-only vs GPT-2 Medium KD vs LEVIATHAN KD
7. Measure: val_bpb improvement from each KD source

---

## Hardware Optimization Notes (Strix Halo gfx1151)

> Added 2026-04-09 based on AMADEUS implementation findings.

### Attention Warning: 1B Teacher Needs flash_attn
The 1B teacher model uses standard attention. On gfx1151 without MFMA, naive attention is 0.05x. **Must use flash_attn** (ROCm build available on the training machine). Without flash_attn, teacher training is impractically slow.

### Teacher Training Budget
1B model at ~2-3K tok/s (Mode B, layer-streaming) → very slow. Consider:
- Use `halo_training` Mode B with activation checkpointing
- Budget 4-8 hours for teacher training (not 15 min)
- DeepSpeed CPUAdam may be needed for 1B optimizer state (~8 GB)

### Student Distillation
250M student inherits teacher's architecture patterns. Student training can use autokernel patterns. Apply via `autokernel.optimize(student, training=True)`.

### Memory: 1B model needs ~40-60 GB in training
Fits in 116 GB GPU-visible memory but tight with optimizer state. Use gradient checkpointing (Mode B).
