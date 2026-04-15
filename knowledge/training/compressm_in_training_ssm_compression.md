---
title: "CompreSSM: In-Training Compression of State Space Models"
paper: "The Curious Case of In-Training Compression of State Space Models (ICLR 2026)"
authors: "Chahine, Nazari, Rus, Rusch (MIT CSAIL / Max Planck / Ellis Institute)"
arxiv: "https://arxiv.org/abs/2510.02823v4"
code: "https://github.com/camail-official/compressm"
tags: [ssm, compression, control-theory, balanced-truncation, gramian, hankel, lru, mamba]
---

# CompreSSM: In-Training Compression of State Space Models

## 1. What This Paper Does (Plain English)

SSMs (State Space Models) like Mamba, S4, and LRU use a hidden "state" vector to remember information from past inputs -- like a conveyor belt carrying notes through a sequence. The bigger this state, the more notes you can carry, but the slower it goes.

**The core question:** Do you actually *need* all those notes, or are some just dead weight?

**CompreSSM's answer:** Most of the state dimensions become useless early in training. You can identify them using 200-year-old control theory (Gramians, Hankel singular values), surgically remove them *during training*, and get a faster, smaller model that performs *better* than one trained small from scratch.

**Why "better than training small from scratch"?** Because starting big lets the model explore a richer optimization landscape and discover good weight configurations. Then you prune away the deadwood while keeping the learned structure -- structure that a small model would never find on its own.

---

## 2. Background: What is an SSM?

### The Recurrence (the beating heart)

An SSM processes a sequence one step at a time using a hidden state:

```
h[k+1] = A * h[k] + B * x[k]     <- state update
y[k]   = C * h[k] + D * x[k]     <- output
```

| Symbol | Shape | Meaning |
|--------|-------|---------|
| `h[k]` | (n,) | Hidden state at step k -- the model's "memory" |
| `x[k]` | (p,) | Input at step k |
| `y[k]` | (q,) | Output at step k |
| `A` | (n, n) | State transition -- how memory evolves |
| `B` | (n, p) | Input projection -- how input writes to memory |
| `C` | (q, n) | Output projection -- how memory is read |
| `D` | (q, p) | Skip connection (often ignored) |

### Analogy: The Orchestra

Think of the state `h` as an orchestra with `n` musicians:
- **A** is the sheet music telling each musician how to evolve their note based on what everyone else played
- **B** is the conductor telling each musician how to respond to the audience (input)
- **C** is the microphone picking up which musicians the audience actually hears

**The key insight:** Some musicians contribute nothing to what the audience hears. CompreSSM finds and fires them mid-concert.

### PyTorch -- Basic SSM

```python
import torch
import torch.nn as nn

class SimpleSSM(nn.Module):
    """Minimal diagonal SSM (like LRU/S4)."""
    def __init__(self, input_dim: int, state_dim: int, output_dim: int):
        super().__init__()
        self.state_dim = state_dim
        # Diagonal A stored as log for stability (eigenvalues < 1)
        self.A_log = nn.Parameter(torch.randn(state_dim) - 1.0)  # negative -> stable
        self.B = nn.Parameter(torch.randn(state_dim, input_dim) * 0.01)
        self.C = nn.Parameter(torch.randn(output_dim, state_dim) * 0.01)

    @property
    def A_diag(self):
        # Eigenvalues with magnitude < 1 (stability requirement)
        return torch.sigmoid(self.A_log)  # values in (0, 1)

    def forward(self, x):
        """x: (batch, seq_len, input_dim) -> (batch, seq_len, output_dim)"""
        B, T, _ = x.shape
        A = self.A_diag  # (state_dim,)
        h = torch.zeros(B, self.state_dim, device=x.device)
        outputs = []
        for t in range(T):
            h = A * h + x[:, t] @ self.B.T  # diagonal A -> element-wise multiply
            y = h @ self.C.T
            outputs.append(y)
        return torch.stack(outputs, dim=1)
```

---

## 3. The Math Behind CompreSSM

### Step 1: Gramians -- Measuring "Influence"

Two questions about each state dimension:
1. **Controllability:** Can the input *reach* this dimension? (Can we write to it?)
2. **Observability:** Does this dimension affect the *output*? (Can we read from it?)

These are answered by solving two **discrete Lyapunov equations**:

**Controllability Gramian P:**
```
A * P * A^T  -  P  +  B * B^T  =  0

Closed form:  P = sum_{i=0}^{inf} A^i * B * B^T * (A^T)^i
```

**Observability Gramian Q:**
```
A^T * Q * A  -  Q  +  C^T * C  =  0

Closed form:  Q = sum_{i=0}^{inf} (A^T)^i * C^T * C * A^i
```

### Analogy: The Gramian as a "Usefulness Score"

Imagine each state dimension is a wire connecting input to output:
- **P** measures how much signal flows *into* each wire (controllability)
- **Q** measures how much signal flows *out* of each wire to the output (observability)
- A wire that's hard to reach (low P) OR never read (low Q) is useless

### Step 2: Hankel Singular Values -- The Combined Score

The **Hankel singular values (HSVs)** combine both Gramians into a single importance ranking:

```
sigma = sort_descending( sqrt(eigenvalues(P * Q)) )
```

- `sigma_1 >= sigma_2 >= ... >= sigma_n > 0`
- Large sigma_i -> dimension i is both reachable AND heard -> **keep it**
- Small sigma_i -> dimension i is either unreachable, inaudible, or both -> **cut it**

### Analogy: The Karaoke Machine

Imagine a karaoke machine with `n` audio channels:
- P measures how well the singer's voice reaches each channel
- Q measures how loudly each channel plays through the speakers
- The HSV is like the *end-to-end volume* of each channel: voice -> channel -> speaker
- Channels with near-zero end-to-end volume can be unplugged without anyone noticing

### Step 3: Balanced Truncation -- The Surgery

**The idea:** Transform the state space so that the importance ranking is explicit (balanced realization), then simply chop off the bottom dimensions.

1. **Find balancing transformation T** such that in the new coordinates, `P = Q = diag(sigma_1, ..., sigma_n)`
2. **Apply it:** `A_b = T^{-1} * A * T`,  `B_b = T^{-1} * B`,  `C_b = C * T`
3. **Truncate:** Keep only the top-r rows/columns

**Error bound (from control theory):**
```
||G - G_truncated||_inf  <=  2 * sum_{i=r+1}^{n} sigma_i
```

This is a *guaranteed* bound -- you know exactly how much error you're introducing.

### Step 4: Energy Fraction Tolerance

Choose the truncation rank r by retaining a fraction of total energy:

```
r = min_k { sum_{i=1}^{k} sigma_i  >=  (1 - tau) * sum_{i=1}^{n} sigma_i }
```

where `tau` in [0, 1] is the tolerance (e.g., tau=0.05 discards 5% of total HSV energy).

### Step 5: Why "During Training" Works (Lemma 3.1)

The paper's key theoretical contribution: HSVs change *smoothly* between gradient steps.

**Lemma 3.1 (Weyl's Theorem applied):**
```
|sigma_i(after_step) - sigma_i(before_step)|  <=  max_eigenvalue(delta_H)
```

where `delta_H` is the perturbation to the Hankel matrix from one gradient step.

**Why this matters:** If the HSVs barely move between steps, then the importance ranking established early in training is *reliable*. You don't need to wait until the end to know which dimensions are useless.

### Analogy: Sorting Books on a Shelf

In the first few minutes of training, the model is like a librarian sorting books. Very quickly, the important books go to the front shelf and the irrelevant ones go to the back. Lemma 3.1 says: once this ordering stabilizes (which happens fast), it *stays* stable. So you can confidently remove the back-shelf books early.

### Step 6: The Diagonal Shortcut

Most modern SSMs (LRU, S4, S5, Mamba's core) use **diagonal A** matrices. This makes Gramian computation trivial -- no need to solve Lyapunov equations iteratively:

```
P_ij = (B * B^T)_ij / (1 - lambda_i * lambda_j)
Q_ij = (C^T * C)_ij / (1 - lambda_i * lambda_j)
```

where `lambda_i` are the diagonal entries of A. This is O(n^2) instead of O(n^3).

### Step 7: Balanced State Matrix Partitioning

After balancing, the system is partitioned:

```
A = [A_11  A_12]    B = [B_1]    C = [C_1  C_2]
    [A_21  A_22]        [B_2]
```

Where `A_11` is (r x r), `B_1` is (r x p), `C_1` is (q x r). The truncated system keeps only the "1" blocks.

### Step 8: Hankel Matrix Construction

The Hankel matrix H, whose eigenvalues are the HSVs:

```
H = ( P^{1/2} * Q * P^{1/2} )^{1/2}
```

Training perturbations:
```
A' = A + delta_A,   B' = B + delta_B,   C' = C + delta_C
P' = P + delta_P,   Q' = Q + delta_Q
H' = H + delta_H
```

### Key Assumptions

- **Assumption 2.1**: All eigenvalues of A have amplitude < 1 (stability)
- **Assumption 2.2**: (A, B) is controllable
- **Assumption 2.3**: (A, C) is observable

---

## 4. The CompreSSM Algorithm (8-Step Process)

1. Extract discrete system matrices (A, B, C) from model weights
2. Solve discrete Lyapunov equations for controllability/observability Gramians
3. Compute Hankel singular values via eigenvalue decomposition
4. Identify minimum rank r retaining (1 - tau) energy fraction
5. Compute balancing transformation if reduction sufficiently large
6. Transform system to diagonal balanced realization
7. Truncate to rank r
8. Replace model weights with reduced matrices

### Compression Schedule

- **Standard (LRA tasks):** 4 equidistant reduction attempts during warm-up (first 10% of training). Only reduce if new dim < 95% of current dim.
- **sMNIST:** Attempted truncation throughout all training (no learning rate decay).
- **IMDB:** Initial 1k step waiting period, then 2-3 reductions by step 3k.

---

## 5. PyTorch Implementation

### 5a. Gramian Computation (Diagonal SSM)

```python
def compute_gramians_diagonal(A_diag, B, C):
    """
    Compute controllability (P) and observability (Q) Gramians
    for a diagonal SSM in closed form.

    Args:
        A_diag: (n,) diagonal eigenvalues, |lambda_i| < 1
        B: (n, p) input matrix
        C: (q, n) output matrix

    Returns:
        P: (n, n) controllability Gramian
        Q: (n, n) observability Gramian
    """
    n = A_diag.shape[0]

    # Denominator: 1 - lambda_i * lambda_j  (n x n matrix)
    lam = A_diag.unsqueeze(1)          # (n, 1)
    denom = 1.0 - lam * lam.T         # (n, n)

    # Numerators
    BBT = B @ B.T                      # (n, n)
    CTC = C.T @ C                      # (n, n)

    P = BBT / denom                    # element-wise
    Q = CTC / denom

    return P, Q
```

### 5b. Hankel Singular Values

```python
def hankel_singular_values(P, Q):
    """
    Compute Hankel singular values from Gramians.

    sigma = sqrt(eigenvalues(P @ Q)), sorted descending.
    """
    PQ = P @ Q                                       # (n, n)
    eigenvalues = torch.linalg.eigvalsh(PQ)          # real since P, Q symmetric PD
    eigenvalues = torch.clamp(eigenvalues, min=0)    # numerical safety
    hsv = torch.sqrt(eigenvalues)
    hsv, _ = torch.sort(hsv, descending=True)
    return hsv
```

### 5c. Determine Truncation Rank

```python
def find_truncation_rank(hsv, tau=0.05):
    """
    Find minimum rank r retaining (1 - tau) fraction of total energy.

    Args:
        hsv: sorted Hankel singular values (descending)
        tau: tolerance -- fraction of energy allowed to discard

    Returns:
        r: number of dimensions to keep
    """
    total_energy = hsv.sum()
    threshold = (1.0 - tau) * total_energy
    cumulative = torch.cumsum(hsv, dim=0)
    r = int((cumulative >= threshold).nonzero(as_tuple=True)[0][0].item()) + 1
    return r
```

### 5d. Balanced Truncation (Core Algorithm)

```python
def balanced_truncation(A_diag, B, C, r):
    """
    Apply balanced truncation to a diagonal SSM, reducing state dim from n to r.

    Returns new (A_diag_r, B_r, C_r) with state_dim = r.
    """
    n = A_diag.shape[0]
    P, Q = compute_gramians_diagonal(A_diag, B, C)

    # Step 1: Cholesky of P  ->  P = L @ L^T
    L = torch.linalg.cholesky(P + 1e-6 * torch.eye(n, device=P.device))

    # Step 2: Form L^T @ Q @ L, take its eigendecomposition
    M = L.T @ Q @ L                          # (n, n) symmetric PD
    eigvals, U = torch.linalg.eigh(M)        # ascending order
    eigvals = torch.clamp(eigvals, min=0)

    # Hankel singular values = sqrt(eigvals), sorted descending
    sigma = torch.sqrt(eigvals).flip(0)
    U = U.flip(1)                             # match descending order

    # Step 3: Balancing transformation
    # T = L @ U @ Sigma^{-1/4}    (maps to balanced coordinates)
    sigma_quarter_inv = sigma.pow(-0.25)
    T = L @ U @ torch.diag(sigma_quarter_inv)
    T_inv = torch.diag(sigma.pow(0.25)) @ U.T @ torch.linalg.inv(L)

    # Step 4: Transform system to balanced realization
    A_full = torch.diag(A_diag)
    A_bal = T_inv @ A_full @ T               # (n, n)
    B_bal = T_inv @ B                         # (n, p)
    C_bal = C @ T                             # (q, n)

    # Step 5: Truncate to top-r dimensions
    A_r = A_bal[:r, :r]
    B_r = B_bal[:r, :]
    C_r = C_bal[:, :r]

    # For diagonal SSMs, extract new diagonal A
    A_diag_r = torch.diag(A_r)

    return A_diag_r, B_r, C_r
```

### 5e. Full CompreSSM Training Loop

```python
class CompreSSMTrainer:
    """
    Training loop with in-training balanced truncation.

    Schedule: 4 equidistant reduction attempts during LR warm-up
    (first 10% of training). Only reduce if new dim < 95% of current.
    """
    def __init__(self, model, optimizer, total_steps, tau=0.05,
                 num_reductions=4, warmup_frac=0.1):
        self.model = model
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.tau = tau

        # Schedule reductions during warm-up
        warmup_steps = int(total_steps * warmup_frac)
        self.reduction_steps = [
            int(warmup_steps * (i + 1) / num_reductions)
            for i in range(num_reductions)
        ]
        self.min_reduction_ratio = 0.95  # only reduce if < 95% of current

    def train_step(self, batch, step):
        # Normal training step
        loss = self.model(batch)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Check if this is a reduction step
        if step in self.reduction_steps:
            self.attempt_reduction()

        return loss.item()

    @torch.no_grad()
    def attempt_reduction(self):
        """Try to compress each SSM layer via balanced truncation."""
        for name, layer in self.model.named_modules():
            if not isinstance(layer, SimpleSSM):
                continue

            A_diag = layer.A_diag
            B = layer.B.data
            C = layer.C.data

            # Compute HSVs
            P, Q = compute_gramians_diagonal(A_diag, B, C)
            hsv = hankel_singular_values(P, Q)

            # Find target rank
            r = find_truncation_rank(hsv, self.tau)

            # Only reduce if meaningfully smaller
            if r >= self.min_reduction_ratio * layer.state_dim:
                print(f"  {name}: r={r} >= 95% of {layer.state_dim}, skipping")
                continue

            # Apply balanced truncation
            new_A, new_B, new_C = balanced_truncation(A_diag, B, C, r)

            # Surgery: replace layer parameters in-place
            layer.state_dim = r
            layer.A_log = nn.Parameter(
                torch.log(new_A / (1 - new_A + 1e-8))  # inverse sigmoid
            )
            layer.B = nn.Parameter(new_B)
            layer.C = nn.Parameter(new_C)

            print(f"  {name}: {A_diag.shape[0]} -> {r} dims "
                  f"(kept {100*(1-self.tau):.0f}% energy)")

        # IMPORTANT: rebuild optimizer with new params
        self.optimizer = type(self.optimizer)(
            self.model.parameters(),
            **self.optimizer.defaults
        )
```

### 5f. Pragmatic Variant (with Rollback)

```python
@torch.no_grad()
def attempt_reduction_pragmatic(self, val_loader, eval_fn):
    """
    Pragmatic variant: try 10% reduction, rollback if validation degrades.
    No tau hyperparameter needed.
    """
    # Save checkpoint
    checkpoint = {
        'model': {k: v.clone() for k, v in self.model.state_dict().items()},
        'optimizer': self.optimizer.state_dict(),
    }
    baseline_metric = eval_fn(self.model, val_loader)

    for name, layer in self.model.named_modules():
        if not isinstance(layer, SimpleSSM):
            continue

        n = layer.state_dim
        r = max(1, int(n * 0.9))  # reduce by 10%

        A_diag = layer.A_diag
        new_A, new_B, new_C = balanced_truncation(
            A_diag, layer.B.data, layer.C.data, r
        )

        # Apply tentatively
        layer.state_dim = r
        layer.A_log = nn.Parameter(torch.log(new_A / (1 - new_A + 1e-8)))
        layer.B = nn.Parameter(new_B)
        layer.C = nn.Parameter(new_C)

    # Brief fine-tune + evaluate
    new_metric = eval_fn(self.model, val_loader)

    if new_metric < baseline_metric:
        # Rollback
        self.model.load_state_dict(checkpoint['model'])
        print("Reduction reverted -- validation degraded")
    else:
        print(f"Reduction accepted -- {baseline_metric:.4f} -> {new_metric:.4f}")
```

---

## 6. Experimental Results

### Table 1: CompreSSM vs Baseline (Top-3 Mean Across 5 Seeds)

| Dataset | tau | Final State Dim | CompreSSM Acc | Baseline Acc | Delta |
|---------|-----|-----------------|---------------|--------------|-------|
| **CIFAR10** | 0.10 | 92.6 | **85.7%** | 81.8% | +3.9% |
| **CIFAR10** | 0.15 | 57.4 | **84.4%** | 78.2% | +6.2% |
| **sMNIST** | 0.10 | 27.6 | **96.9%** | 96.0% | +0.9% |
| **sMNIST** | 0.15 | 12.7 | **95.9%** | 92.6% | +3.3% |
| **ListOps** | 0.10 | 81.8 | **51.8%** | 46.3% | +5.5% |
| **Pathfinder** | 0.10 | 51.2 | 97.9% | 97.9% | 0.0% |
| **AAN** | 0.10 | 84.4 | 87.5% | 87.9% | -0.4% |
| **IMDB** | 0.10 | 119.6 | 82.8% | 83.5% | -0.7% |

**Key pattern:** CompreSSM wins big on tasks where state capacity correlates with difficulty (sMNIST, CIFAR10, ListOps). Marginal on tasks where all dims already matter (AAN, IMDB, Pathfinder).

### Table 2: Training Speedup Comparison

| Method | CIFAR10 dim=93 | CIFAR10 Acc | Relative Speed |
|--------|----------------|-------------|----------------|
| Full baseline (dim=384) | -- | 86.5% | 1.0x |
| Baseline (dim=93) | 93 | 81.8% | 1.6x |
| **CompreSSM** (tau=0.1) | 93 | **85.7%** | **1.5x** |
| Knowledge Distillation | 93 | 83.5% | 0.52x (slower!) |
| HNN Regularization | 93 | -- | 0.06x (100x slower) |

### Hyperparameters

| Task | Depth | Hidden | State | Steps | Batch | LR | Dropout |
|------|-------|--------|-------|-------|-------|----|---------|
| sMNIST | 1 | 88 | 256 | 200k | 50 | 4e-4 | 0.1 |
| CIFAR10 | 6 | 512 | 384 | 180k | 50 | warmup->1e-3 | 0.1 |
| ListOps | 6 | 128 | 256 | 80k | 32 | warmup->1e-3 | 0.0 |
| IMDB | 1 | 256 | 192 | 50k | 32 | warmup->1e-3 | 0.1 |
| AAN | 6 | 128 | 256 | 100k | 64 | warmup->1e-3 | 0.1 |
| Pathfinder | 6 | 192 | 256 | 500k | 64 | warmup->1e-3 | 0.0 |

LR schedule (LRA): warm from 1e-7 to 1e-3 over first 10% of steps, cosine decay to 1e-7.

---

## 7. Pros and Cons

### Pros

| Advantage | Details |
|-----------|---------|
| **Theoretically principled** | Backed by 200+ years of control theory. Error bounds are *guaranteed*, not empirical. |
| **Better than training small** | On CIFAR10: 85.7% at dim 93 vs 81.8% baseline at same dim -- huge gap. |
| **1.5x training speedup** | Smaller state = fewer ops per step. Speedup compounds over remaining training. |
| **Works during training** | No separate compression stage. No teacher model needed. |
| **Cheap for diagonal SSMs** | Gramians are O(n^2) closed-form, not O(n^3) Lyapunov solves. |
| **Smooth HSV dynamics** | Lemma 3.1 guarantees stable importance rankings -- safe to prune early. |
| **Pragmatic variant** | No hyperparameter tuning needed -- just try 10% reduction with rollback. |

### Cons

| Disadvantage | Details |
|--------------|---------|
| **Task-dependent** | Works great on sMNIST, CIFAR10 -- marginal on AAN, IMDB, Pathfinder. Requires correlation between state capacity and task difficulty. |
| **LTI assumption** | Assumes time-invariant dynamics. Mamba/selective SSMs are time-varying -- requires "mean LTI surrogate" approximation. |
| **JAX only (official)** | Reference implementation is JAX. No official PyTorch release. |
| **LRU-focused** | Only tested on LRU + brief Mamba extension. No S4/S5/RWKV/Griffin experiments. |
| **Small-scale experiments** | Largest model ~6 layers, dim 512. No billion-parameter experiments. |
| **Optimizer rebuild** | After truncation, optimizer state (momentum, Adam buffers) must be reset or projected. |
| **Non-uniform compression** | Each layer may compress to different sizes, complicating batched operations. |

---

## 8. Limitations

1. **No language modeling experiments.** All benchmarks are classification (sMNIST, CIFAR, LRA). Language generation quality after truncation is untested.

2. **Selective SSMs (Mamba) are a stretch.** Mamba's A/B/C change per input token (time-varying). The paper uses an averaged "mean LTI surrogate" -- this loses the entire point of selectivity. The theoretical guarantees weaken significantly.

3. **No interaction with torch.compile / kernel fusion.** The paper doesn't address how dynamic dimension changes interact with compiled graphs, fused kernels, or hardware-specific optimizations.

4. **Compression ceiling.** On tasks like Pathfinder where all dimensions matter, you can't compress without quality loss. The method honestly shows this, but it means you need to *try it* to know if it works for your task.

5. **Single-pass assumption.** The error bound `||G - G_hat||_inf <= 2 * sum(sigma_i)` assumes a single truncation. Multiple successive truncations accumulate error without a clean combined bound.

---

## 9. Hardware / Ecosystem Compatibility

| Platform | Status | Notes |
|----------|--------|-------|
| **NVIDIA (CUDA)** | Full support | JAX and PyTorch both work. Gramian computation is standard linear algebra. |
| **AMD ROCm (gfx1151)** | Compatible in principle | Gramian computation is pure matmul + eigendecomposition -- uses rocBLAS/rocSOLVER. No MFMA needed. The compression is metadata surgery, not a new kernel. |
| **Apple Silicon (MPS)** | Compatible | `torch.linalg.eigh` works on MPS. Overhead is negligible. |
| **CPU (AVX-512)** | Full support | Gramian computation is cheap. Actually ideal for the control-theory math. |
| **TPU (JAX)** | Native | The official implementation IS JAX. |
| **torch.compile** | Needs care | After truncation, compiled graphs must be re-traced (dimension changed). Add `torch._dynamo.reset()` after each reduction. |
| **Mamba (triton kernels)** | Needs adaptation | Mamba's custom CUDA/Triton kernels assume fixed state dim. Dimension change requires kernel recompilation or fallback to eager mode during reduction steps. |
| **DDP / FSDP** | Compatible | Reduction must happen synchronously across all ranks. Broadcast new weights after truncation. |

### Strix Halo (gfx1151) Relevance

This is pure math -- eigendecomposition and matrix multiplies on (n x n) where n is typically 64-384. These fit entirely in L2 cache (6 MB). The overhead per reduction step is microseconds. The *benefit* is smaller state dims -> fewer FLOPs per token for the entire rest of training.

---

## 10. Extension to Selective SSMs (Mamba)

For input-dependent Linear Time-Varying systems (like Mamba), the method uses **"mean LTI surrogates"** -- averaging the A, B, C dynamics over a batch of input data to obtain a single LTI system per channel. This enables per-channel balancing and rank selection.

**Caveats:**
- Loses the input-dependent selectivity that makes Mamba powerful
- Theoretical guarantees from the LTI setting no longer strictly hold
- The paper treats this as a proof-of-concept extension, not a core result

---

## 11. Learning Roadmap: From Zero to CompreSSM

### Level 1: Linear Algebra Foundations (1-2 weeks)

**What to learn:**
- Vectors, matrices, matrix multiplication
- Eigenvalues and eigenvectors ("which directions does a matrix stretch?")
- Symmetric matrices and their special properties
- Matrix decompositions (eigendecomposition, SVD, Cholesky)

**Resources:**
- 3Blue1Brown "Essence of Linear Algebra" (YouTube, free, visual)
- Gilbert Strang's MIT 18.06 lectures (YouTube, free)

**Checkpoint:** Can you explain what `eigenvalues = torch.linalg.eigvalsh(M)` computes and why?

### Level 2: Recurrence & Sequences (1 week)

**What to learn:**
- What is a recurrence: `h[t+1] = f(h[t], x[t])`
- RNNs as a special case
- Why matrix A controls "memory decay" (eigenvalues < 1 = stable, = forgetting)
- The convolution view of LTI systems (optional but helpful)

**Exercise:** Implement the `SimpleSSM` class above. Feed it a sine wave. Plot the hidden states. See which dimensions respond.

### Level 3: Control Theory Basics (1-2 weeks)

**What to learn:**
- Controllability: "can I steer the state anywhere with the right input?"
- Observability: "can I distinguish internal states by looking at the output?"
- Gramians as energy measures
- Lyapunov equations (just the concept -- "what matrix P satisfies A*P*A^T - P + B*B^T = 0?")

**Analogy to internalize:** A Gramian is like a "heat map" of where energy accumulates in the state space. Hot dimensions carry lots of signal. Cold dimensions are dead weight.

**Resources:**
- Steve Brunton's "Control Bootcamp" (YouTube, free, excellent)
- Karl Astrom's "Feedback Systems" Chapter 8 (free PDF)

**Exercise:** Implement `compute_gramians_diagonal` above. Create a 10-dim SSM. Compute Gramians. Set 3 dimensions of B to zero. Verify those dimensions get near-zero Gramian entries.

### Level 4: Hankel Singular Values & Balanced Truncation (1 week)

**What to learn:**
- HSVs combine controllability + observability into one score
- Balanced realization = coordinate system where P = Q = diagonal
- Truncation in balanced coordinates = optimal low-rank approximation
- The error bound: `||G - G_hat||_inf <= 2 * sum(sigma_i, truncated)`

**Exercise:** Implement the full `balanced_truncation` function. Create a 64-dim SSM, truncate to 16 dims. Compare outputs on random sequences -- verify the error is within the bound.

### Level 5: SSM Architectures (1-2 weeks)

**What to learn:**
- S4, S5: structured state spaces with HiPPO initialization
- LRU: simplified diagonal SSM (what this paper uses)
- Mamba: input-dependent (selective) SSM
- How these fit into a deep learning stack (embedding -> SSM layers -> head)

**Resources:**
- Albert Gu's "Annotated S4" blog post
- Mamba paper (Gu & Dao, 2023)
- Sasha Rush's "The Annotated S4" (Harvard)

### Level 6: CompreSSM Itself (1 week)

**Now read the paper.** With Levels 1-5, every equation should click:
1. Equations 1-2: "That's the SSM recurrence I implemented"
2. Equations 3-5: "That's the Gramian I computed"
3. Equation 7: "That's the HSV formula"
4. Equation 8: "That's the energy threshold"
5. Lemma 3.1: "Weyl says eigenvalues are Lipschitz under perturbation"
6. Algorithm 1: "That's my training loop with periodic truncation"

**Final exercise:** Clone the repo (`github.com/camail-official/compressm`), run sMNIST, watch the state dimensions shrink during training. Then port it to PyTorch using the code above.

### Level 7: Connect to Your Own Work

For AMADEUS/ARGUS-PRIME style models, the diagonal SSM layers (Griffin scan, Mamba scan) are exactly the kind of layers CompreSSM targets. Potential experiments:
1. Profile HSVs of a trained model to see how many state dims are actually used
2. Apply in-training truncation during BabyLM runs
3. Measure tok/s improvement from smaller state dimensions

---

## 12. Comparison to Alternative Compression Methods

| Method | Advantage | Limitation |
|--------|-----------|------------|
| **CompreSSM** | Superior at aggressive compression; in-training; principled | Requires state dim-performance correlation |
| **Hankel Nuclear Norm (HNN) Regularization** | Theoretically principled | 100x slower (Gramian evaluations every step) |
| **Knowledge Distillation** | Comparable for modest compression | Drops with aggressive compression; requires full teacher training |
| **Baseline (train small)** | Simplest approach | Worse quality -- misses structure learned in larger space |
| **Post-hoc pruning** | No training modification | Typically inferior to in-training methods |

---

## References

- Paper: https://arxiv.org/abs/2510.02823v4
- Code: https://github.com/camail-official/compressm
- Related: S4 (Gu et al., 2022), Mamba (Gu & Dao, 2023), LRU (Orvieto et al., 2023)
- Control theory: Antoulas "Approximation of Large-Scale Dynamical Systems" (2005)
