# SELF-CURRICULUM

**Self-Curriculum Learning with Knowledge Injection for 15-Minute Training**

## The Blind Spot

Every architecture plan says "train on OpenWebText ~100M tokens." But 65% of those tokens are glue words the model learns in the first minute. We WASTE 14 minutes of training on patterns already learned.

**Self-curriculum learning** makes the model its own difficulty estimator: sample hard sequences more often, easy sequences less. Combined with **knowledge injection** (factual content front-loaded for Engram), this turns 15 minutes of training into the equivalent of ~45 minutes on the patterns that matter.

**Works with ANY of our 12 architectures.** It's a DataLoader strategy, not a model change.

---

## Training Phases

### Phase 0: Knowledge Priming (0-2 min, ~4M tokens)
- **Data:** Wikipedia + textbook-style text ONLY
- **Sampling:** Uniform
- **Purpose:** Engram tables absorb factual N-grams before general training begins
- **Architecture:** All components active except MTP heads

### Phase 1: Warm-up (2-5 min, ~6M tokens)
- **Data:** Full mixed dataset
- **Sampling:** Uniform (alpha=0)
- **Purpose:** Model learns basic language patterns. Difficulty scoring BEGINS (recording, not yet sampling).
- **Architecture:** MTP heads activate

### Phase 2: Self-Curriculum (5-13 min, ~15M tokens)
- **Data:** Full mixed dataset
- **Sampling:** Self-curriculum, alpha annealing 0 → 1.5
- **Purpose:** Hard content sequences get 3× exposure. Easy glue sequences deprioritized.
- **KD:** Knowledge distillation from teacher starts (if available)

### Phase 3: Polish (13-15 min, ~4M tokens)
- **Data:** Top-25% hardest sequences only
- **Sampling:** Alpha fixed at 1.5
- **Purpose:** Final concentration on unlearned patterns

---

## Self-Curriculum Sampler

### Sampling Probability

```
P(seq_i) = difficulty_score[i]^alpha / Σ_j difficulty_score[j]^alpha
```

- alpha=0: uniform (standard training)
- alpha=1.5: hard sequences sampled ~3× more than easy ones

### Difficulty Score (per sequence)

```
difficulty_score[i] = β * old_score + (1-β) * content_word_loss[i]
```

- `β = 0.9` (exponential moving average)
- `content_word_loss` = CE loss computed ONLY on non-glue tokens
- Uses the shared `GLUE_TOKEN_IDS` dictionary

### Alpha Annealing

```
alpha = min(1.5, 1.5 * max(0, (step - warmup_steps) / (total_steps - warmup_steps)))
```

Ramps from 0 to 1.5 over the self-curriculum phase.

### Score Refresh (prevent staleness)

Every 500 steps: forward-only pass on 1000 random unvisited examples. Update their scores. Prevents the sampler from ignoring sequences it hasn't seen recently.

### Implementation

```python
class SelfCurriculumSampler:
    def __init__(self, dataset, glue_token_ids, total_steps):
        self.dataset = dataset
        self.scores = torch.ones(len(dataset))
        self.glue_set = set(glue_token_ids)
        self.total_steps = total_steps
        self.last_seen = torch.zeros(len(dataset), dtype=torch.long)
    
    def get_alpha(self, step):
        warmup = self.total_steps * 0.33  # Phase 0+1
        if step < warmup:
            return 0.0
        progress = (step - warmup) / (self.total_steps - warmup)
        return min(1.5, 1.5 * progress)
    
    def sample_batch(self, batch_size, step):
        alpha = self.get_alpha(step)
        if alpha == 0:
            return torch.randint(0, len(self.dataset), (batch_size,))
        probs = self.scores ** alpha
        probs = probs / probs.sum()
        return torch.multinomial(probs, batch_size, replacement=True)
    
    def update_scores(self, indices, logits, targets, step, beta=0.9):
        for i, idx in enumerate(indices):
            content_mask = torch.tensor([t.item() not in self.glue_set for t in targets[i]])
            if content_mask.any():
                cl = F.cross_entropy(logits[i][content_mask], targets[i][content_mask])
                self.scores[idx] = beta * self.scores[idx] + (1-beta) * cl.item()
            self.last_seen[idx] = step
    
    def refresh_stale(self, model, step, n_samples=1000):
        """Re-score examples not seen in 500+ steps"""
        stale = (step - self.last_seen > 500).nonzero().squeeze()
        if len(stale) == 0:
            return
        sample = stale[torch.randperm(len(stale))[:n_samples]]
        with torch.no_grad():
            for idx in sample:
                tokens = self.dataset[idx]
                logits = model(tokens.unsqueeze(0)).logits.squeeze(0)
                content_mask = torch.tensor([t.item() not in self.glue_set for t in tokens[1:]])
                if content_mask.any():
                    cl = F.cross_entropy(logits[:-1][content_mask], tokens[1:][content_mask])
                    self.scores[idx] = cl.item()
                self.last_seen[idx] = step
```

---

## Knowledge Injection

### Knowledge-Dense Sequence Selection

Pre-compute a `knowledge_density` score for all sequences before training:

```python
def knowledge_density(tokens, glue_set):
    content_ratio = sum(1 for t in tokens if t not in glue_set) / len(tokens)
    # Higher content ratio = more knowledge-dense
    return content_ratio
```

Sequences with `knowledge_density > 0.5` are tagged "knowledge" and used in Phase 0.

### Data Sources

| Source | Tokens | Purpose | Phase 0 eligible |
|--------|--------|---------|-----------------|
| OpenWebText | 60M | General language | No (too noisy) |
| Wikipedia | 25M | Factual knowledge | **Yes** |
| Code (The Stack) | 10M | Structured reasoning | No |
| Textbooks/cleaned | 5M | Clean factual | **Yes** |
| **Total** | **100M** | | |

Phase 0 uses the ~30M knowledge-eligible tokens (Wikipedia + textbooks). In 2 minutes at ~2M tokens/min throughput, the model sees ~4M tokens of factual content.

---

## Expected Impact

### Effective Training Budget Comparison

| Strategy | Easy tokens seen | Hard tokens seen | Effective hard budget |
|----------|-----------------|------------------|----------------------|
| Uniform (standard) | ~16M (65%) | ~9M (35%) | 9M |
| **Self-curriculum** | ~7M (28%) | **~18M (72%)** | **18M (2×)** |
| **Self-curriculum + KD** | ~7M | ~18M + teacher signal | **~27M equivalent** |

The model sees 2× more hard tokens AND each hard token has richer gradient signal from knowledge distillation. Combined: ~3× effective training on hard patterns.

### Per-Phase Token Distribution

| Phase | Duration | Tokens | Focus |
|-------|----------|--------|-------|
| 0: Knowledge Prime | 2 min | 4M | 100% factual (Wikipedia/textbook) |
| 1: Warm-up | 3 min | 6M | Uniform mix |
| 2: Self-Curriculum | 8 min | 15M | 3× weight on hard content |
| 3: Polish | 2 min | 4M | Top-25% hardest only |
| **Total** | **15 min** | **~29M** | |

---

## Integration with Eval Framework

The self-curriculum sampler and eval framework share:
- `GLUE_TOKEN_IDS` dictionary
- Content/glue loss split computation
- Per-token loss tracking

The eval framework can VISUALIZE the curriculum effect:
- Plot: content-word loss over time (should drop faster than uniform baseline)
- Plot: sampling distribution over time (should shift toward hard sequences)
- Plot: difficulty score distribution (should separate into easy/hard clusters)

---

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Overfitting to hard examples (repetitive sampling) | MEDIUM | EMA smoothing (β=0.9); alpha annealing starts at 0; refresh stale scores |
| Stale scores lead to bad sampling | LOW | Refresh 1000 examples every 500 steps; EMA prevents jumps |
| Knowledge priming too specialized | LOW | Phase 0 is only 2 min; general training follows immediately |
| Overhead of per-token content masking | LOW | Glue set is a Python set (O(1) lookup); content mask computed once per batch |

---

## Implementation Roadmap

1. Create shared `glue_tokens.py` with `GLUE_TOKEN_IDS`
2. Pre-tokenize data mix (OWT + Wikipedia + code + textbooks) into memory-mapped format
3. Compute `knowledge_density` scores, tag knowledge-dense sequences
4. Implement `SelfCurriculumSampler` class
5. Implement phase scheduler (0→1→2→3 with auto-transition)
6. Integrate with training loop (any architecture)
7. Add curriculum visualization to eval framework
8. Test: compare uniform vs self-curriculum on GPT-2 Small, measure content-word loss improvement

---

## Hardware Optimization Notes (Strix Halo gfx1151)

> Added 2026-04-09 based on AMADEUS implementation findings.

### DataLoader-Only — Compatible with Any Architecture
No architecture changes needed. Works as a training enhancement with any plan.

### Performance Fix: GPU-Vectorize Content Masks
The content mask computation (filtering glue tokens) should NOT use a Python loop. Pre-compute at dataset init:
```python
# At init:
self.glue_mask = torch.zeros(vocab_size, dtype=torch.bool, device='cuda')
self.glue_mask[list(GLUE_TOKEN_IDS)] = True

# At runtime (fully vectorized, no Python loop):
content_mask = ~self.glue_mask[targets]  # (batch, seq) bool tensor
content_loss = (per_token_loss * content_mask).sum() / content_mask.sum()
```
This eliminates CPU-GPU sync that would otherwise occur every batch.

### Recommended Architecture Pairing
- **Best with Caveman LFM / Parallel Caveman** — Engram tables benefit from Phase 0 knowledge priming
- **Good with AMADEUS** — SwiGLU FFN stores knowledge; curriculum helps convergence
- **Less useful with Resonant Loop** — no Engram tables, shared block has limited capacity for factual storage

### Throughput Impact
Negligible if content masks are GPU-vectorized. Score refresh (every 500 steps, 1000 examples) adds ~1% overhead — forward-only pass, no backward.
