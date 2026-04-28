---
title: "Alignment Technique Implementation Details: D2PO, ConfPO, AlphaPO, Magpie"
domain: training
type: implementation-reference
status: active
tags: [alignment, dpo, simpo, d2po, confpo, alphapo, magpie, preference-optimization, token-weighting]
related:
  - instruct_alignment_techniques_2025_2026.md
  - sft_pipeline.md
---

# Alignment Technique Implementation Details

Exact formulas, code, and hyperparameters for four alignment techniques.
Target: drop-in PyTorch implementation for 80-170M parameter models.

---

## 1. D2PO: Temporal Decay for Preference Optimization

**Paper:** "Earlier Tokens Contribute More: Learning Direct Preference Optimization From Temporal Decay Perspective"
**Authors:** Ruichen Shao, Bei Li, Gangao Liu, Yang Chen, Xiang Zhou, Jingang Wang, Xunliang Cai, Peng Li
**Venue:** ICLR 2025 | **arXiv:** 2502.14340
**Code:** https://github.com/LotuSrc/D2PO (built on LlamaFactory)

### 1.1 Core Mechanism

Standard DPO/SimPO treats all tokens equally when computing sequence log-probabilities. D2PO applies exponential decay to weight earlier tokens more heavily, based on the finding that earlier tokens are more critical for alignment quality.

The decay weight for token at position t is: **gamma^t** where gamma < 1.0.

### 1.2 Exact Implementation (from D2PO source code)

The key modification is in `get_batch_logps`:

```python
def get_batch_logps(
    logits: torch.Tensor,       # (B, T, V)
    labels: torch.Tensor,       # (B, T) with IGNORE_INDEX for non-scored positions
    label_pad_token_id: int = IGNORE_INDEX,
    gamma: float = 1.0          # temporal decay factor; < 1.0 enables D2PO
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute weighted log-probabilities with temporal decay."""
    
    # Standard shift for next-token prediction
    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    
    # Build mask: 1 for scored tokens, 0 for padding/prompt
    loss_mask = labels != label_pad_token_id
    labels[labels == label_pad_token_id] = 0
    
    # Per-token log probabilities
    per_token_logps = torch.gather(
        logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
    ).squeeze(2)
    
    # D2PO: apply gamma^(cumulative_position) decay
    # torch.cumsum(loss_mask, dim=1) gives position 1,2,3,... for scored tokens
    # gamma^position decays later tokens exponentially
    decay_weights = torch.pow(gamma, torch.cumsum(loss_mask, dim=1))
    
    weighted_logps = (per_token_logps * decay_weights * loss_mask).sum(-1)
    valid_length = loss_mask.sum(-1)
    
    return weighted_logps, valid_length
```

### 1.3 How It Modifies DPO and SimPO

For **DPO**, the standard log-prob sum `log pi(y|x) = sum_t log pi(y_t|y_<t, x)` becomes:

```
log_pi_d2po(y|x) = sum_t [gamma^t * log pi(y_t | y_<t, x)]
```

The DPO loss is then:

```
L_D2PO = -E[log sigma(beta * (log_pi_d2po(y_w|x) - log_ref_d2po(y_w|x)
                              - log_pi_d2po(y_l|x) + log_ref_d2po(y_l|x)))]
```

For **SimPO** (reference-free, length-normalized), the reward becomes:

```
r_D2PO_SimPO(y;x) = (beta / valid_len) * sum_t [gamma^t * log pi(y_t | y_<t, x)]
```

Then: `L = -E[log sigma(r_D2PO_SimPO(y_w) - r_D2PO_SimPO(y_l) - margin)]`

### 1.4 Drop-in Modification for Existing DPO Code

To add D2PO to `halo_training/dpo.py`, modify `compute_log_probs`:

```python
def compute_log_probs(model, input_ids, mask, gamma=1.0):
    """Compute per-token log probabilities with optional D2PO temporal decay."""
    with torch.amp.autocast("cuda", dtype=torch.float16):
        logits = model(input_ids)  # (B, T, V)

    shift_logits = logits[:, :-1, :]   # (B, T-1, V)
    shift_labels = input_ids[:, 1:]     # (B, T-1)
    shift_mask = mask[:, 1:]            # (B, T-1)

    log_probs = F.log_softmax(shift_logits.float(), dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

    # D2PO temporal decay
    if gamma < 1.0:
        positions = torch.cumsum(shift_mask, dim=1)  # 1, 2, 3, ... for scored tokens
        decay_weights = torch.pow(gamma, positions.float())
        masked_log_probs = token_log_probs * decay_weights * shift_mask.float()
    else:
        masked_log_probs = token_log_probs * shift_mask.float()

    return masked_log_probs.sum(dim=-1)  # (B,)
```

### 1.5 Recommended Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| gamma | **0.98** | Default in D2PO repo for Llama-3-8B |
| gamma range | 0.95 - 0.99 | gamma < 1.0 activates D2PO mode |
| gamma = 1.0 | Standard DPO | No decay (baseline) |
| beta (DPO) | 0.1 | Same as standard DPO |
| pref_loss | sigmoid | Standard Bradley-Terry |
| Other params | Unchanged | LR, epochs, etc. same as base method |

**Key insight:** gamma=0.98 means at position 50, weight = 0.98^50 = 0.364. At position 100, weight = 0.98^100 = 0.133. So the last third of a 150-token response contributes only ~5% as much as the first tokens.

### 1.6 Results

- +5.9 to 8.8 points on AlpacaEval 2 over vanilla DPO
- +3.3 to 9.7 points on Arena-Hard over vanilla DPO
- No degradation on MMLU, GSM8K, MATH (general capabilities preserved)
- Works across model sizes and architectures

---

## 2. ConfPO: Confidence-Based Token Selection

**Paper:** "ConfPO: Exploiting Policy Model Confidence for Critical Token Selection in Preference Optimization"
**Authors:** Hee Suk Yoon, Eunseop Yoon, Mark Hasegawa-Johnson, Sungwoong Kim, Chang D. Yoo
**Venue:** ICML 2025 | **arXiv:** 2506.08712

### 2.1 Core Mechanism

Standard DPO/SimPO uniformly adjusts all token probabilities regardless of their relevance to the preference. ConfPO identifies "preference-critical" tokens using the policy model's own confidence and only optimizes on those tokens.

**Key insight:** Tokens with lower model confidence (lower probability) have larger gradients. There is a strong negative correlation between token probability and gradient magnitude. By selecting only low-confidence tokens, ConfPO focuses the KL budget on the most impactful tokens.

### 2.2 Exact Formulas

**Token selection indicator:**

```
s(y_i) = 1[pi_theta(y_i | x, y_<i) <= tau]
```

where tau is the per-response average token probability:

```
tau_w = (1 / |y_w|) * sum_i pi_theta(y_i^w | x, y_<i^w)    # for chosen
tau_l = (1 / |y_l|) * sum_i pi_theta(y_i^l | x, y_<i^l)    # for rejected
```

Tokens with probability BELOW the mean are selected (they are the "preference-critical" ones).

**ConfPO loss (SimPO variant, reference-free):**

```
L_ConfPO = -E[log sigma(
    (beta / |y_s^w|) * sum_i [s(y_i^w) * log pi_theta(y_i^w | x, y_<i^w)]
  - (beta / |y_s^l|) * sum_i [s(y_i^l) * log pi_theta(y_i^l | x, y_<i^l)]
  - gamma
)]
```

where `|y_s|` = number of selected tokens (those passing the threshold).

**Gradient norm relationship (theoretical justification):**

```
||grad_theta log pi_theta(y_i | x, y_<i)|| = ||grad_theta pi_theta(y_i | x, y_<i)|| / pi_theta(y_i | x, y_<i)
```

Low-probability tokens have inversely proportional gradient magnitude, making them the most informative for preference optimization.

### 2.3 Drop-in Implementation

```python
def compute_log_probs_confpo(model, input_ids, mask):
    """Compute ConfPO-weighted log probabilities.
    
    Selects only tokens where model confidence is below the per-sequence mean.
    Zero additional compute: uses probabilities already computed in forward pass.
    """
    with torch.amp.autocast("cuda", dtype=torch.float16):
        logits = model(input_ids)  # (B, T, V)

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = mask[:, 1:].float()

    # Per-token probabilities (for confidence thresholding)
    probs = F.softmax(shift_logits.float(), dim=-1)
    token_probs = probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)  # (B, T-1)
    
    # Per-token log probabilities (for the actual loss)
    log_probs = F.log_softmax(shift_logits.float(), dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

    # Compute per-sequence mean probability (tau threshold)
    masked_probs = token_probs * shift_mask
    tau = masked_probs.sum(dim=-1, keepdim=True) / shift_mask.sum(dim=-1, keepdim=True).clamp(min=1)

    # Select tokens with confidence BELOW mean (preference-critical)
    selection_mask = (token_probs <= tau).float() * shift_mask  # s(y_i)
    
    # Count selected tokens for normalization
    n_selected = selection_mask.sum(dim=-1).clamp(min=1)  # |y_s|

    # Weighted log-prob sum over selected tokens only
    selected_log_probs = (token_log_probs * selection_mask).sum(dim=-1)

    return selected_log_probs, n_selected


def confpo_loss(
    policy_model, ref_model,
    chosen_ids, chosen_mask, rejected_ids, rejected_mask,
    beta: float = 0.1, gamma_margin: float = 0.0,
    use_reference: bool = True,
):
    """ConfPO loss: DPO/SimPO with confidence-based token selection."""
    
    # Policy log probs with ConfPO selection
    pi_chosen_logps, n_chosen = compute_log_probs_confpo(policy_model, chosen_ids, chosen_mask)
    pi_rejected_logps, n_rejected = compute_log_probs_confpo(policy_model, rejected_ids, rejected_mask)
    
    if use_reference:
        # DPO variant: use reference model (also with ConfPO selection)
        with torch.no_grad():
            ref_chosen_logps, _ = compute_log_probs_confpo(ref_model, chosen_ids, chosen_mask)
            ref_rejected_logps, _ = compute_log_probs_confpo(ref_model, rejected_ids, rejected_mask)
        
        chosen_reward = beta * (pi_chosen_logps / n_chosen - ref_chosen_logps / n_chosen)
        rejected_reward = beta * (pi_rejected_logps / n_rejected - ref_rejected_logps / n_rejected)
    else:
        # SimPO variant: reference-free, length-normalized
        chosen_reward = beta * pi_chosen_logps / n_chosen
        rejected_reward = beta * pi_rejected_logps / n_rejected
    
    loss = -F.logsigmoid(chosen_reward - rejected_reward - gamma_margin).mean()
    
    with torch.no_grad():
        accuracy = (chosen_reward > rejected_reward).float().mean()
    
    return loss, {
        "chosen_reward": chosen_reward.mean().item(),
        "rejected_reward": rejected_reward.mean().item(),
        "accuracy": accuracy.item(),
        "loss": loss.item(),
        "avg_selected_chosen": n_chosen.mean().item(),
        "avg_selected_rejected": n_rejected.mean().item(),
    }
```

### 2.4 Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| tau | Auto-computed | Per-sequence mean probability; no tuning needed |
| beta | Same as base | Use same beta as DPO (0.1) or SimPO (2.5) |
| gamma_margin | Same as base | Use same margin as SimPO if applicable |
| Additional params | **NONE** | Zero hyperparameters added |

**Key property:** ConfPO adds ZERO hyperparameters and ZERO computational overhead (uses already-computed forward pass probabilities).

### 2.5 Results

- Mistral-7B: AlpacaEval2 LC 27.1% (SimPO) -> 28.9% (ConfPO)
- Llama-3-8B: Arena-Hard 32.6% -> 32.8%
- Consistent gains across all 4 model configurations tested
- Mitigates overoptimization / reward hacking by using KL budget more efficiently

### 2.6 Combining D2PO + ConfPO

Both are free improvements and can stack. Apply D2PO's gamma decay AND ConfPO's token selection:

```python
def compute_log_probs_d2po_confpo(model, input_ids, mask, gamma=0.98):
    """Combined D2PO temporal decay + ConfPO confidence selection."""
    with torch.amp.autocast("cuda", dtype=torch.float16):
        logits = model(input_ids)

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = mask[:, 1:].float()

    probs = F.softmax(shift_logits.float(), dim=-1)
    token_probs = probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
    
    log_probs = F.log_softmax(shift_logits.float(), dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

    # ConfPO: select low-confidence tokens
    masked_probs = token_probs * shift_mask
    tau = masked_probs.sum(-1, keepdim=True) / shift_mask.sum(-1, keepdim=True).clamp(min=1)
    selection_mask = (token_probs <= tau).float() * shift_mask

    # D2PO: temporal decay on selected positions
    positions = torch.cumsum(selection_mask, dim=1)
    decay_weights = torch.pow(gamma, positions)

    # Combined: selected + decayed log-probs
    weighted_logps = (token_log_probs * selection_mask * decay_weights).sum(-1)
    n_selected = selection_mask.sum(-1).clamp(min=1)

    return weighted_logps, n_selected
```

---

## 3. AlphaPO: Reward Shape Matters

**Paper:** "AlphaPO: Reward Shape Matters for LLM Alignment"
**Authors:** Aman Gupta, Shao Tang, Qingquan Song, Sirou Zhu, Jiwoo Hong, Ankan Saha, Viral Gupta, Noah Lee, Eunki Kim, Siyu Zhu, Parag Agrawal, Natesh Pillai, S. Sathiya Keerthi
**Venue:** ICML 2025 (Proceedings of the 42nd ICML) | **arXiv:** 2501.03884

### 3.1 Core Mechanism

Standard SimPO uses log reward: `r(y;x) = (beta/|y|) * log pi(y|x)`. AlphaPO generalizes this by parameterizing the reward function shape with alpha, derived from alpha-divergence with length normalization.

When alpha=0, AlphaPO reduces exactly to SimPO (log reward).
Positive alpha values (0.1-0.25) control likelihood displacement more conservatively, leading to better alignment.

### 3.2 Exact Formulas

**AlphaPO reward function:**

```
r_alpha(y; x) = beta * (1 - pi_theta(y|x)^(-alpha/|y|)) / alpha
```

where `pi_theta(y|x)^(1/|y|)` is the geometric mean token probability (length-normalized).

In the limit alpha -> 0, using L'Hopital's rule:
```
lim_{alpha->0} r_alpha(y;x) = (beta/|y|) * log pi_theta(y|x)   [= SimPO reward]
```

Special cases:
- alpha = 0: SimPO (log reward)
- alpha = 1: inverse-linear reward: `r(y;x) = beta * (-1/pi_bar + 1)`
- alpha = -1: linear reward: `r(y;x) = beta * (pi_bar - 1)`
  where `pi_bar = pi_theta(y|x)^(1/|y|)` is the geometric mean token probability.

**AlphaPO loss (Bradley-Terry with margin):**

```
L_AlphaPO = -E[log sigma(
    (-beta/alpha) * pi_theta(y_w|x)^(-alpha/|y_w|)
  + (beta/alpha) * pi_theta(y_l|x)^(-alpha/|y_l|)
  - gamma
)]
```

Equivalently, using per-token log-probs:

```
L_AlphaPO = -E[log sigma(
    (-beta/alpha) * exp(-alpha * (1/|y_w|) * sum_t log pi(y_w_t | ...))
  + (beta/alpha) * exp(-alpha * (1/|y_l|) * sum_t log pi(y_l_t | ...))
  - gamma
)]
```

### 3.3 Drop-in Implementation

```python
def alphapo_loss(
    policy_model,
    chosen_ids, chosen_mask,
    rejected_ids, rejected_mask,
    beta: float = 2.5,
    gamma_margin: float = 0.25,  # gamma/beta from paper; actual gamma = gamma_margin * beta
    alpha: float = 0.25,
):
    """AlphaPO loss: SimPO with alpha-parameterized reward shape.
    
    alpha=0 reduces to SimPO exactly.
    Recommended: alpha in [0.1, 0.25] for best alignment.
    """
    # Compute average log-probs (length-normalized)
    pi_chosen_logps = compute_log_probs(policy_model, chosen_ids, chosen_mask)      # sum of log-probs
    pi_rejected_logps = compute_log_probs(policy_model, rejected_ids, rejected_mask)
    
    chosen_len = chosen_mask[:, 1:].float().sum(dim=-1).clamp(min=1)
    rejected_len = rejected_mask[:, 1:].float().sum(dim=-1).clamp(min=1)
    
    # Average log-prob per token
    avg_logp_chosen = pi_chosen_logps / chosen_len      # (1/|y_w|) * sum log pi
    avg_logp_rejected = pi_rejected_logps / rejected_len  # (1/|y_l|) * sum log pi
    
    actual_gamma = gamma_margin * beta  # paper reports gamma/beta
    
    if abs(alpha) < 1e-8:
        # SimPO limit: alpha -> 0
        chosen_reward = beta * avg_logp_chosen
        rejected_reward = beta * avg_logp_rejected
    else:
        # AlphaPO: r_alpha = beta * (1 - exp(-alpha * avg_logp)) / alpha
        # The loss uses: (-beta/alpha) * exp(-alpha * avg_logp_w) + (beta/alpha) * exp(-alpha * avg_logp_l)
        chosen_reward = (-beta / alpha) * torch.exp(-alpha * avg_logp_chosen)
        rejected_reward = (-beta / alpha) * torch.exp(-alpha * avg_logp_rejected)
        # Net: chosen_reward - rejected_reward
        # = (beta/alpha) * [exp(-alpha * avg_logp_l) - exp(-alpha * avg_logp_w)]
    
    loss = -F.logsigmoid(chosen_reward - rejected_reward - actual_gamma).mean()
    
    with torch.no_grad():
        accuracy = (chosen_reward > rejected_reward).float().mean()
    
    return loss, {
        "chosen_reward": chosen_reward.mean().item(),
        "rejected_reward": rejected_reward.mean().item(),
        "accuracy": accuracy.item(),
        "loss": loss.item(),
    }
```

### 3.4 Recommended Hyperparameters (from paper Table 3)

| Model | alpha | beta | gamma/beta | LR |
|-------|-------|------|------------|-----|
| Mistral-7B-Instruct | **0.25** | 2.5 | 0.1 | 7e-7 |
| Llama-3-8B-Instruct (PairRM) | **0.25** | 2.5 | 1.0 | 1e-6 |
| Llama-3-8B-Instruct (ArmoRM) | **0.25** | 10.0 | 0.3 | 1.1e-6 |
| Gemma-2-9B-Instruct | **0.1** | 10.0 | 0.5 | 8e-7 |

**Tuning guidance:**
- alpha = 0.25 is the best starting point for most models
- alpha = 0.1 for very capable models (Gemma-2-9B)
- Positive alpha values reduce likelihood displacement (conservative)
- Negative alpha values increase displacement (aggressive)
- Optimal alpha is slightly positive; drop-off less steep on positive side
- beta, gamma, LR from SimPO can be reused initially; coordinate-wise search from there
- Training: 1 epoch, cosine LR with warmup 0.1, global batch 128, max seq len 2048
- Optimizer: AdamW

### 3.5 Understanding Alpha's Effect

**Alpha controls likelihood displacement intensity:**
- alpha > 0: LESS aggressive displacement. Preferred response probability decreases less.
- alpha = 0: SimPO behavior (log reward).
- alpha < 0: MORE aggressive displacement. Both preferred and dispreferred likelihoods drop faster.

**Why it helps:** Standard SimPO/DPO can suffer from catastrophic likelihood displacement where preferred response probability drops too much. Slightly positive alpha (0.1-0.25) provides a regularization effect that maintains better preferred likelihoods while still separating the margin.

**Gradient magnitude** `|d_loss/d_v|` is non-monotonic in alpha. Large |alpha| values impose regularization by causing vanishing gradients for already-separated samples (positive margin). This prevents over-optimization.

### 3.6 Results

- Llama-3-8B: AE2 LC 42.05% (SimPO) -> **45.37%** (AlphaPO) = +7.9% relative
- Mistral-7B: AE2 LC 29.71% (SimPO) -> **33.03%** (AlphaPO) = +11.2% relative
- 15-50% relative improvement over DPO across models
- Also improves HellaSwag and TruthfulQA over SimPO
- Does not increase generation length significantly

### 3.7 Combining AlphaPO with D2PO and ConfPO

AlphaPO modifies the reward shape; D2PO modifies token weighting; ConfPO selects tokens. All three are orthogonal and can be combined:

```python
def full_enhanced_loss(
    policy_model, chosen_ids, chosen_mask, rejected_ids, rejected_mask,
    beta=2.5, gamma_margin=0.25, alpha=0.25, d2po_gamma=0.98,
):
    """AlphaPO + D2PO + ConfPO combined."""
    # Get D2PO+ConfPO weighted log-probs
    pi_chosen, n_chosen = compute_log_probs_d2po_confpo(
        policy_model, chosen_ids, chosen_mask, gamma=d2po_gamma
    )
    pi_rejected, n_rejected = compute_log_probs_d2po_confpo(
        policy_model, rejected_ids, rejected_mask, gamma=d2po_gamma
    )
    
    avg_logp_w = pi_chosen / n_chosen
    avg_logp_l = pi_rejected / n_rejected
    actual_gamma = gamma_margin * beta
    
    if abs(alpha) < 1e-8:
        chosen_reward = beta * avg_logp_w
        rejected_reward = beta * avg_logp_l
    else:
        chosen_reward = (-beta / alpha) * torch.exp(-alpha * avg_logp_w)
        rejected_reward = (-beta / alpha) * torch.exp(-alpha * avg_logp_l)
    
    loss = -F.logsigmoid(chosen_reward - rejected_reward - actual_gamma).mean()
    return loss
```

---

## 4. Magpie: Synthetic Instruction Data Generation

**Paper:** "Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing"
**Authors:** Zhangchen Xu, Fengqing Jiang, Luyao Niu, Yuntian Deng, Radha Poovendran, Yejin Choi, Bill Yuchen Lin
**Venue:** arXiv 2406.08464 (Jun 2024, revised Oct 2024)
**Code:** https://github.com/magpie-align/magpie

### 4.1 Core Technique

Feed ONLY the chat template prefix (before the user message position) to an aligned LLM. The model's autoregressive nature causes it to generate a realistic user instruction. Then generate the response normally.

**This works because:** Aligned models (instruction-tuned) have learned the distribution of user queries during their RLHF/SFT training. When given just the template header, they sample from this learned distribution.

### 4.2 Exact Template Format

For **Llama-3-Instruct**:
```
<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n
```

The model generates a user query, terminated by EOS. Then construct the full prompt:
```
<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{generated_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
```

And generate the response.

**General formula:** `input = T_pre_query` -> model generates query q -> `full_prompt = T_pre_query + q + T_post_query` -> model generates response r.

### 4.3 Step-by-Step Pipeline

```
Step 1: INSTRUCTION GENERATION
  for i in range(N):
      input_tokens = tokenize(pre_query_template)  # just the header
      query = model.generate(input_tokens, until=EOS, temperature=1.0)
      instructions.append(query)

Step 2: RESPONSE GENERATION  
  for query in instructions:
      prompt = pre_query_template + query + post_query_template
      response = model.generate(tokenize(prompt), until=EOS)
      dataset.append({"instruction": query, "response": response})

Step 3: QUALITY FILTERING (optional but recommended)
  # Use reward model or quality classifier
  # Filter by: quality rating, safety, difficulty, language
  # Keep top ~7.5% (300K from 4M)

Step 4: PREFERENCE PAIR GENERATION (for DPO)
  for query in filtered_instructions:
      responses = [model.generate(query, temperature=0.8) for _ in range(5)]  # k=5 samples
      scores = reward_model.score(query, responses)
      chosen = responses[argmax(scores)]
      rejected = responses[argmin(scores)]
      dpo_data.append({"prompt": query, "chosen": chosen, "rejected": rejected})
```

### 4.4 Quality Filtering Criteria

The Magpie pipeline uses automatic tagging with 6 dimensions:
1. **Quality:** very poor / poor / average / good / excellent (via LLM judge)
2. **Difficulty:** very easy / easy / medium / hard / very hard
3. **Task category:** information seeking, creative writing, advice, planning, math, etc.
4. **Safety:** via Llama-Guard-2 (<1% flagged as potentially harmful)
5. **Reward score:** via PairRM or ArmoRM
6. **Language:** filter for target language

Recommended: keep quality >= "average", diverse task distribution.

### 4.5 Adapting for API-Based Models (Claude/GPT-4)

The original technique requires access to model weights to feed just the template prefix. For API-based models, the adaptation is:

**Option A: Direct prompting (recommended for API)**
```python
INSTRUCTION_GEN_PROMPT = """Generate a realistic, diverse user instruction or question 
that someone might ask an AI assistant. The instruction should be:
- Natural and varied in complexity
- Cover diverse topics (coding, writing, analysis, math, planning, etc.)  
- Range from simple to complex
Just output the instruction directly, nothing else."""

def generate_magpie_api(client, n_samples=10000, batch_size=50):
    """Generate Magpie-style data using Claude/GPT-4 API."""
    dataset = []
    
    for batch_start in range(0, n_samples, batch_size):
        # Step 1: Generate diverse instructions
        instructions = []
        for _ in range(batch_size):
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=256,
                temperature=1.0,  # high temp for diversity
                messages=[{"role": "user", "content": INSTRUCTION_GEN_PROMPT}]
            )
            instructions.append(response.content[0].text.strip())
        
        # Step 2: Generate responses  
        for instr in instructions:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2048,
                temperature=0.7,
                messages=[{"role": "user", "content": instr}]
            )
            dataset.append({
                "instruction": instr,
                "response": response.content[0].text
            })
    
    return dataset
```

**Option B: Use open-weight model for instruction generation, API for responses**
```python
# Use Llama-3-70B-Instruct locally for instruction generation (Magpie technique)
# Use Claude/GPT-4 for high-quality responses (teacher distillation)
# This gives diverse queries + expert responses
```

**Option C: Prefill technique (Claude API supports this)**
Claude's API allows setting a partial assistant response. Use this for a closer analog:
```python
# Generate instruction by having Claude complete a "user asked:" pattern
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=256,
    temperature=1.0,
    system="You are roleplaying as a diverse set of users interacting with an AI. Generate the next user message.",
    messages=[
        {"role": "user", "content": "What's the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": ""}  # Claude completes this
    ]
)
```

### 4.6 Sample Size Requirements

| Scale | Samples | Quality | Use Case |
|-------|---------|---------|----------|
| Smoke test | 1K | Unfiltered | Verify pipeline works |
| Minimum viable | 10K | Curated | Small model alignment (matches 1M generic) |
| Standard | 50K | Filtered | Good baseline for 80-170M models |
| Full Magpie | 300K | Top 7.5% of 4M | Matches Llama-3-8B-Instruct quality |
| For DPO pairs | 10-60K | High quality | k=5 samples per instruction, reward-ranked |

**Critical finding from DataFlow paper:** 10K curated samples outperform 1M generic samples. For small models (80-170M), quality >>> quantity. 10-50K high-quality Magpie samples are sufficient.

### 4.7 Output Data Format

**SFT format (ShareGPT-compatible):**
```json
{
    "conversations": [
        {"from": "human", "value": "How do I implement a binary search tree in Python?"},
        {"from": "gpt", "value": "Here's a Python implementation of a BST..."}
    ]
}
```

**DPO/preference format:**
```json
{
    "prompt": "How do I implement a binary search tree in Python?",
    "chosen": [
        {"role": "user", "content": "How do I implement a binary search tree in Python?"},
        {"role": "assistant", "content": "Here's a clean implementation..."}
    ],
    "rejected": [
        {"role": "user", "content": "How do I implement a binary search tree in Python?"},
        {"role": "assistant", "content": "A binary search tree is a tree..."}
    ]
}
```

**Multi-turn format:**
```json
{
    "conversations": [
        {"from": "human", "value": "Explain quicksort"},
        {"from": "gpt", "value": "Quicksort is a divide-and-conquer..."},
        {"from": "human", "value": "What's its worst case complexity?"},
        {"from": "gpt", "value": "The worst case is O(n^2)..."}
    ]
}
```

### 4.8 Cost Estimates

| Method | Samples | Cost | Notes |
|--------|---------|------|-------|
| Magpie-Air (8B local) | 3M raw | 206 GPU-hours | ~$0.12 per 1K |
| Magpie-Pro (70B local) | 1M raw | 614 GPU-hours | ~$1.10 per 1K |
| Claude API (Sonnet) | 50K | ~$15-25 | $3/M input + $15/M output tokens |
| GPT-4o-mini API | 50K | ~$5-10 | Much cheaper, good quality |

---

## 5. Recommended Combined Pipeline for Small Models

```
1. DATA GENERATION: Magpie (50K instructions via API or local model)
   -> Quality filter to 10-25K high-quality SFT pairs
   -> Generate k=5 responses per query, rank with reward model
   -> Create 10K DPO preference pairs

2. SFT: 1 epoch on curated instruction data
   - LR: 2e-5, AdamW, cosine schedule

3. PREFERENCE OPTIMIZATION: AlphaPO + D2PO + ConfPO (1 epoch)
   - alpha = 0.25 (AlphaPO reward shape)
   - d2po_gamma = 0.98 (temporal decay)
   - ConfPO token selection (automatic, zero overhead)
   - beta = 2.5, gamma_margin = 0.5 (SimPO-based, tune per model)
   - LR: 5e-7 to 1e-6
   - No reference model needed (SimPO-style)
```

This stacks three orthogonal free/cheap improvements on top of the SimPO baseline, which itself eliminates the reference model overhead of DPO.
