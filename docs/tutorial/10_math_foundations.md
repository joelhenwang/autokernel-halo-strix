# Part 10: Math Foundations for Reading Papers

## Goal
Build the mathematical vocabulary you need to read LLM research papers. Every concept is taught with three things: (1) an intuition you can hold in your head, (2) the formal definition, (3) a code example you can run. By the end, you will be able to open a paper like "XSA: Cross-State Attention" and understand the equations.

## Why This Matters
Parts 11-14 require you to read papers, extract ideas, and implement them. If you cannot read the math, you are stuck copying other people's code without understanding what it does or why. This part gives you independence.

**This is the longest part in the series.** Set aside a full day. Work through the code examples by hand first, then verify with the computer. The goal is not memorization — it is building intuition that you can apply to new papers you have never seen.

---

## 10.1 Linear Algebra for Transformers

### Vectors as Points in Space

A vector is a list of numbers. In ML, vectors represent things: a word, a hidden state, a gradient direction.

```python
import numpy as np
import torch

# A 4-dimensional vector
v = np.array([1.0, 2.0, 3.0, 4.0])

# In transformers, hidden states are vectors of dimension d_model
# GPT-2 small: d_model = 768
# That means every token at every layer is a point in 768-dimensional space
hidden_state = np.random.randn(768)
print(f"Hidden state shape: {hidden_state.shape}")
print(f"Magnitude (L2 norm): {np.linalg.norm(hidden_state):.4f}")
```

**Intuition:** Think of a vector as an arrow pointing from the origin to a point. The direction encodes "what kind of thing this is" (noun? verb? question?). The magnitude encodes "how strongly."

### Matrices as Transformations

A matrix is a function that takes a vector and returns a new vector. It stretches, rotates, and projects.

```python
# A 3x3 matrix
A = np.array([
    [2, 0, 0],
    [0, 1, 0],
    [0, 0, 0.5],
])

v = np.array([1.0, 1.0, 1.0])
result = A @ v
print(f"Input:  {v}")
print(f"Output: {result}")
# Output: [2. 1. 0.5]
# The matrix stretched dimension 0 by 2x, kept dimension 1, and shrank dimension 2 by half.
```

**In transformers, every learned weight matrix IS a transformation.** The attention projection matrices (W_Q, W_K, W_V) transform hidden states into query, key, and value spaces. The MLP weight matrices transform hidden states through nonlinear feature spaces.

### Matrix Multiplication: Why Attention is Q @ K^T @ V

The self-attention formula is:

```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
```

Let us break this apart.

```python
import torch

# Setup: batch=1, seq_len=4, d_model=8, n_heads=2, d_k=4
B, T, D = 1, 4, 8
d_k = D // 2  # 4

# Random input
X = torch.randn(B, T, D)

# Projection matrices (learned weights)
W_Q = torch.randn(D, d_k)  # Project to query space
W_K = torch.randn(D, d_k)  # Project to key space
W_V = torch.randn(D, d_k)  # Project to value space

# Step 1: Project X into Q, K, V
Q = X @ W_Q    # (1, 4, 4)  — "what am I looking for?"
K = X @ W_K    # (1, 4, 4)  — "what do I contain?"
V = X @ W_V    # (1, 4, 4)  — "what information do I provide?"

# Step 2: Q @ K^T — compute compatibility between every pair of tokens
# This is the core of attention: how much should token i attend to token j?
scores = Q @ K.transpose(-2, -1)  # (1, 4, 4) — attention scores
print(f"Attention scores shape: {scores.shape}")
print(f"scores[0] = \n{scores[0].detach().numpy().round(2)}")
# Each row i, column j says: "how compatible is query i with key j?"

# Step 3: Scale by sqrt(d_k) to prevent softmax saturation
scores = scores / (d_k ** 0.5)

# Step 4: Softmax — convert scores to probabilities (each row sums to 1)
attn_weights = torch.softmax(scores, dim=-1)
print(f"\nAttention weights (row sums to 1):")
print(attn_weights[0].detach().numpy().round(3))

# Step 5: Multiply by V — weighted sum of value vectors
output = attn_weights @ V  # (1, 4, 4)
print(f"\nOutput shape: {output.shape}")
# Each output token is a weighted combination of all value vectors,
# where the weights are the attention probabilities.
```

**Key insight:** `Q @ K^T` is a matrix of dot products. The dot product of two vectors measures their similarity. High dot product = the query and key are aligned = attend strongly. Low dot product = ignore.

### Eigenvalues and Eigenvectors

An eigenvector of matrix A is a vector that, when multiplied by A, only gets scaled (not rotated). The scale factor is the eigenvalue.

```
A @ v = lambda * v
```

where `v` is the eigenvector and `lambda` (the eigenvalue) is a scalar.

```python
import numpy as np

# A simple 2x2 matrix
A = np.array([
    [2.0, 1.0],
    [0.0, 3.0],
])

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors (columns):")
print(eigenvectors)

# Verify: A @ v should equal lambda * v
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lam = eigenvalues[i]
    Av = A @ v
    lam_v = lam * v
    print(f"\nEigenvector {i}: {v}")
    print(f"  A @ v    = {Av}")
    print(f"  lambda*v = {lam_v}")
    print(f"  Match: {np.allclose(Av, lam_v)}")
```

**Where this appears in papers:** Eigenvalues tell you about a matrix's behavior. If you repeatedly apply a matrix (like in a recurrent model), the eigenvalues determine whether the output grows, shrinks, or stays stable.

### Spectral Radius: Stability of Recurrent Systems

The spectral radius of a matrix is the largest absolute value of its eigenvalues:

```
rho(A) = max(|lambda_1|, |lambda_2|, ..., |lambda_n|)
```

**The rule:** If rho(A) < 1, repeatedly applying A causes values to decay to zero (stable). If rho(A) > 1, values explode to infinity (unstable). If rho(A) = 1, values are preserved (marginally stable).

```python
import numpy as np

def spectral_radius(A):
    eigenvalues = np.linalg.eigvals(A)
    return np.max(np.abs(eigenvalues))

# Stable matrix (all eigenvalues inside the unit circle)
A_stable = np.array([
    [0.5, 0.1],
    [0.2, 0.3],
])

# Unstable matrix (eigenvalue > 1)
A_unstable = np.array([
    [1.1, 0.1],
    [0.2, 0.9],
])

print(f"Stable matrix:   rho = {spectral_radius(A_stable):.4f}")
print(f"Unstable matrix: rho = {spectral_radius(A_unstable):.4f}")

# Demonstrate: apply the matrix 100 times and watch the trajectory
x = np.array([1.0, 1.0])
print(f"\nStable trajectory (applying A_stable 100 times):")
for i in range(100):
    x = A_stable @ x
    if i in [0, 1, 5, 10, 50, 99]:
        print(f"  Step {i:3d}: ||x|| = {np.linalg.norm(x):.6f}")

x = np.array([1.0, 1.0])
print(f"\nUnstable trajectory (applying A_unstable 100 times):")
for i in range(100):
    x = A_unstable @ x
    if i in [0, 1, 5, 10, 50, 99]:
        print(f"  Step {i:3d}: ||x|| = {np.linalg.norm(x):.6f}")
```

**Where this appears:** The Parcae paper uses spectral radius to ensure that looped models (which apply the same layers repeatedly) remain stable. They parameterize the transition matrix so that eigenvalues are guaranteed to be in (-1, 0), making rho(A) < 1 always.

### Vector Projection

The projection of vector u onto vector v answers: "how much of u points in the direction of v?"

```
proj_v(u) = (u . v / ||v||^2) * v
```

```python
import numpy as np

def project(u, v):
    """Project u onto v."""
    return (np.dot(u, v) / np.dot(v, v)) * v

def component_along(u, v):
    """Scalar component of u along v."""
    return np.dot(u, v) / np.linalg.norm(v)

u = np.array([3.0, 4.0])
v = np.array([1.0, 0.0])  # x-axis

proj = project(u, v)
print(f"u = {u}")
print(f"v = {v}")
print(f"proj_v(u) = {proj}")  # [3, 0] — the x-component of u
print(f"Scalar component: {component_along(u, v)}")  # 3.0

# The residual (u - proj) is perpendicular to v
residual = u - proj
print(f"Residual: {residual}")  # [0, 4] — the y-component
print(f"Residual dot v: {np.dot(residual, v)}")  # 0.0 — perpendicular!
```

**Where this appears:** The XSA (Cross-State Attention) paper. XSA projects the attention output to remove the component along the "self-value" direction. This forces the model to attend to OTHER tokens instead of reinforcing its own representation. The formula is literally a vector projection followed by subtraction.

### Orthogonality

Two vectors are orthogonal (perpendicular) if their dot product is zero:

```
u . v = 0  means  u is perpendicular to v
```

```python
import numpy as np

u = np.array([1.0, 0.0, 0.0])
v = np.array([0.0, 1.0, 0.0])
w = np.array([1.0, 1.0, 0.0])

print(f"u . v = {np.dot(u, v)}")  # 0 — orthogonal
print(f"u . w = {np.dot(u, w)}")  # 1 — not orthogonal

# Gram-Schmidt: make a set of orthogonal vectors from arbitrary vectors
def gram_schmidt(vectors):
    """Orthogonalize a set of vectors."""
    orthogonal = []
    for v in vectors:
        for u in orthogonal:
            v = v - project(u, v)  # Subtract projection onto each previous vector
        if np.linalg.norm(v) > 1e-10:
            orthogonal.append(v / np.linalg.norm(v))  # Normalize
    return orthogonal

# Example: orthogonalize 3 random vectors in 3D
vecs = [np.random.randn(3) for _ in range(3)]
ortho = gram_schmidt(vecs)

print("\nOrthogonalized vectors:")
for i, v in enumerate(ortho):
    print(f"  e{i} = {v.round(4)}")

print("\nDot products (should all be ~0):")
for i in range(len(ortho)):
    for j in range(i+1, len(ortho)):
        print(f"  e{i} . e{j} = {np.dot(ortho[i], ortho[j]):.10f}")
```

**Where this appears:** XSA forces the attention output to be orthogonal to the self-value vector. This is an explicit constraint built into the architecture, not learned.

### Torch Summary: Linear Algebra Operations

```python
import torch

# All the linear algebra operations used in transformers:
A = torch.randn(4, 4)
B = torch.randn(4, 4)
v = torch.randn(4)

# Matrix multiply
C = A @ B                           # (4, 4) @ (4, 4) -> (4, 4)
C = torch.matmul(A, B)              # Same thing

# Matrix-vector multiply
u = A @ v                           # (4, 4) @ (4,) -> (4,)

# Transpose
At = A.T                            # Transpose
At = A.transpose(0, 1)              # Same thing

# Eigenvalues
eigenvalues = torch.linalg.eigvals(A)

# Norms
norm_v = torch.norm(v)              # L2 norm
norm_A = torch.norm(A, p='fro')     # Frobenius norm

# Dot product
dot = torch.dot(v, v)               # Scalar

# Batch matrix multiply (for attention)
Q = torch.randn(2, 8, 4, 16)       # (batch, heads, seq, d_k)
K = torch.randn(2, 8, 4, 16)
scores = Q @ K.transpose(-2, -1)    # (2, 8, 4, 4) — attention scores
```

---

## 10.2 Probability for Language Models

### Probability Distributions Over Vocabulary

A language model outputs a probability distribution over the vocabulary for each position. After the softmax, the output is a vector of 50,257 numbers (for GPT-2) that sum to 1.

```python
import torch
import torch.nn.functional as F

# Raw logits from the model (before softmax)
vocab_size = 50257
logits = torch.randn(vocab_size)

# Softmax: convert logits to probabilities
# softmax(z_i) = exp(z_i) / sum(exp(z_j))
probs = F.softmax(logits, dim=-1)

print(f"Logits shape: {logits.shape}")
print(f"Probs shape:  {probs.shape}")
print(f"Probs sum:    {probs.sum().item():.6f}")  # 1.000000
print(f"Min prob:     {probs.min().item():.8f}")
print(f"Max prob:     {probs.max().item():.8f}")
print(f"Most likely token ID: {probs.argmax().item()}")

# Temperature scaling: control the "sharpness" of the distribution
# High temperature -> more uniform (creative)
# Low temperature -> more peaked (deterministic)
for temp in [0.1, 0.5, 1.0, 2.0, 5.0]:
    probs_t = F.softmax(logits / temp, dim=-1)
    entropy = -(probs_t * torch.log(probs_t + 1e-10)).sum()
    print(f"  temp={temp:.1f}  entropy={entropy:.2f}  "
          f"max_prob={probs_t.max():.4f}  effective_vocab={torch.exp(entropy):.0f}")
```

**Intuition:** `softmax(logits / temperature)` is like adjusting the contrast on a photo. Low temperature makes peaks sharper (model is more confident). High temperature flattens everything (model considers more options).

### Cross-Entropy Loss

The training loss for language models is cross-entropy: how surprised is the model by the correct next token?

```
CE = -log(p(correct_token))
```

If the model assigns probability 0.9 to the correct token, the loss is `-log(0.9) = 0.105`. If it assigns probability 0.01, the loss is `-log(0.01) = 4.605`. The model is "more surprised" and receives a larger penalty.

```python
import torch
import torch.nn.functional as F
import math

# Manual cross-entropy computation
vocab_size = 50257
logits = torch.randn(vocab_size)
correct_token = 42  # The actual next token

# Method 1: By hand
probs = F.softmax(logits, dim=-1)
loss_manual = -torch.log(probs[correct_token])
print(f"Manual CE loss: {loss_manual.item():.4f}")

# Method 2: PyTorch (numerically stable — uses log-sum-exp trick)
loss_pytorch = F.cross_entropy(logits.unsqueeze(0), torch.tensor([correct_token]))
print(f"PyTorch CE loss: {loss_pytorch.item():.4f}")

# They should match
print(f"Match: {torch.allclose(loss_manual, loss_pytorch, atol=1e-5)}")

# What does the loss value mean?
# At initialization, a random model assigns ~1/50257 to each token
# So the initial loss should be about:
initial_loss = -math.log(1 / vocab_size)
print(f"\nExpected initial loss (random): {initial_loss:.4f}")  # ~10.825
print("A well-trained small LLM reaches loss ~3.0-4.0")
print("This means the model assigns ~2-5% probability to the correct token on average")
print(f"  loss=4.0 -> p(correct) = {math.exp(-4.0):.4f} ({math.exp(-4.0)*100:.2f}%)")
print(f"  loss=3.0 -> p(correct) = {math.exp(-3.0):.4f} ({math.exp(-3.0)*100:.2f}%)")
print(f"  loss=2.0 -> p(correct) = {math.exp(-2.0):.4f} ({math.exp(-2.0)*100:.2f}%)")

# Batch cross-entropy (what the training loop actually uses)
batch_logits = torch.randn(8, 1024, vocab_size)  # (batch, seq_len, vocab)
batch_targets = torch.randint(0, vocab_size, (8, 1024))  # (batch, seq_len)
batch_loss = F.cross_entropy(
    batch_logits.reshape(-1, vocab_size),  # Flatten to (batch*seq, vocab)
    batch_targets.reshape(-1),             # Flatten to (batch*seq,)
)
print(f"\nBatch CE loss: {batch_loss.item():.4f} (should be ~{initial_loss:.1f} for random)")
```

### KL Divergence

KL divergence measures how different two probability distributions are. It answers: "how many extra bits do I need if I use distribution Q to encode data that actually follows distribution P?"

```
KL(P || Q) = sum(P(x) * log(P(x) / Q(x)))
```

KL divergence is NOT symmetric: KL(P || Q) != KL(Q || P).

```python
import torch
import torch.nn.functional as F

# Two distributions over 5 tokens
P = torch.tensor([0.5, 0.3, 0.1, 0.05, 0.05])  # "true" distribution
Q = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2])     # uniform approximation

# Manual KL divergence
kl_manual = (P * torch.log(P / Q)).sum()
print(f"KL(P || Q) manual: {kl_manual.item():.4f}")

# PyTorch KL divergence (note: expects LOG probabilities for Q)
kl_pytorch = F.kl_div(Q.log(), P, reduction='sum')
print(f"KL(P || Q) PyTorch: {kl_pytorch.item():.4f}")

# KL is not symmetric
kl_reverse = (Q * torch.log(Q / P)).sum()
print(f"KL(Q || P): {kl_reverse.item():.4f}")
print(f"KL(P||Q) != KL(Q||P): {not torch.allclose(kl_manual, kl_reverse)}")

# KL = 0 when P == Q
kl_same = (P * torch.log(P / P)).sum()
print(f"KL(P || P): {kl_same.item():.4f}")  # 0.0000
```

**Where KL divergence appears:**
- **Knowledge distillation:** Student model Q tries to match teacher model P. Loss includes KL(P || Q).
- **DPO (Direct Preference Optimization):** The loss prevents the policy from straying too far from the reference model. This is a KL constraint.
- **VAE (Variational Autoencoder):** KL between the learned posterior and a prior.

### Perplexity

Perplexity is the exponential of cross-entropy loss. It has a beautiful interpretation: **how many tokens is the model effectively choosing between?**

```
Perplexity = exp(cross_entropy_loss)
```

```python
import math

# If CE loss = 10.8 (random model, vocab = 50257)
ppl_random = math.exp(10.8)
print(f"Random model perplexity: {ppl_random:.0f}")  # ~49,021
# The model is choosing uniformly among ~49K tokens. Terrible.

# If CE loss = 4.0 (bad but trained model)
ppl_bad = math.exp(4.0)
print(f"Bad model perplexity: {ppl_bad:.1f}")  # ~54.6
# Choosing among ~55 tokens. Still bad.

# If CE loss = 3.0 (decent small model)
ppl_decent = math.exp(3.0)
print(f"Decent model perplexity: {ppl_decent:.1f}")  # ~20.1
# Choosing among ~20 tokens. Getting somewhere.

# If CE loss = 2.0 (good model)
ppl_good = math.exp(2.0)
print(f"Good model perplexity: {ppl_good:.1f}")  # ~7.4
# Choosing among ~7 tokens. Good!

# GPT-4 on typical web text: perplexity ~5-8
# Human prediction of English text: perplexity ~1.5-3
```

**Why perplexity instead of loss?** Perplexity is more interpretable. "The model has perplexity 20" means "it is choosing among 20 tokens at each position." Cross-entropy 3.0 is less intuitive.

---

## 10.3 Calculus for Training

### Gradient Descent

Training is optimization: find the parameters that minimize the loss. Gradient descent does this by repeatedly taking small steps in the direction that reduces the loss most.

```
theta_new = theta_old - learning_rate * gradient(loss, theta)
```

The gradient tells you: "which direction should I move each parameter to reduce the loss?"

```python
import torch

# Simple example: minimize f(x) = (x - 3)^2
# The minimum is at x = 3.
x = torch.tensor(5.0, requires_grad=True)
learning_rate = 0.1

print("Gradient descent on f(x) = (x - 3)^2:")
for step in range(20):
    # Forward: compute loss
    loss = (x - 3.0) ** 2
    
    # Backward: compute gradient
    loss.backward()
    
    # Update: step in the negative gradient direction
    with torch.no_grad():
        x -= learning_rate * x.grad
        x.grad.zero_()
    
    if step % 5 == 0:
        print(f"  step={step:2d}  x={x.item():.6f}  loss={loss.item():.6f}")

print(f"Final x: {x.item():.6f} (should be ~3.0)")
```

### The Chain Rule: Why Backpropagation Works

Neural networks are compositions of functions: `f(g(h(x)))`. The chain rule tells us how to differentiate through the composition:

```
d/dx f(g(h(x))) = f'(g(h(x))) * g'(h(x)) * h'(x)
```

Backpropagation is just the chain rule applied systematically from the output back to the input.

```python
import torch

# A tiny 2-layer network: y = W2 @ relu(W1 @ x)
x = torch.tensor([1.0, 2.0], requires_grad=False)
W1 = torch.randn(3, 2, requires_grad=True)  # Layer 1: 2 -> 3
W2 = torch.randn(1, 3, requires_grad=True)  # Layer 2: 3 -> 1
target = torch.tensor([5.0])

# Forward pass (chain of functions)
h = W1 @ x             # Linear transform
a = torch.relu(h)      # Nonlinearity
y = W2 @ a             # Linear transform
loss = (y - target) ** 2  # Loss

print(f"Forward: x={x} -> h={h.detach()} -> a={a.detach()} -> y={y.item():.4f}")
print(f"Loss: {loss.item():.4f}")

# Backward pass (chain rule, computed by autograd)
loss.backward()

print(f"\nGradients (computed by chain rule):")
print(f"  dL/dW2 = {W2.grad}")
print(f"  dL/dW1 = \n{W1.grad}")

# Verify manually for W2:
# loss = (W2 @ relu(W1 @ x) - target)^2
# d(loss)/d(W2) = 2 * (y - target) * relu(W1 @ x)^T
manual_dW2 = 2 * (y - target) * a.unsqueeze(0)
print(f"\n  Manual dL/dW2 = {manual_dW2.detach()}")
print(f"  Match: {torch.allclose(W2.grad, manual_dW2.detach())}")
```

### Learning Rate: The Most Important Hyperparameter

The learning rate controls step size. It is the single most impactful hyperparameter in training.

```python
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Minimize f(x, y) = x^2 + 10*y^2 (an elongated bowl)
# This is a classic test function because it shows how
# different learning rates behave.

def train_with_lr(lr, steps=100):
    x = torch.tensor([5.0, 5.0], requires_grad=True)
    trajectory = [x.detach().clone().numpy()]
    losses = []
    
    for _ in range(steps):
        loss = x[0]**2 + 10 * x[1]**2
        loss.backward()
        with torch.no_grad():
            x -= lr * x.grad
            x.grad.zero_()
        trajectory.append(x.detach().clone().numpy())
        losses.append(loss.item())
    
    return trajectory, losses

# Try three learning rates
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, lr, title in [
    (axes[0], 0.001, "Too small (lr=0.001)"),
    (axes[1], 0.08, "Good (lr=0.08)"),
    (axes[2], 0.11, "Too large (lr=0.11)"),
]:
    traj, losses = train_with_lr(lr)
    ax.plot([t[0] for t in traj], [t[1] for t in traj], 'b.-', alpha=0.5, markersize=2)
    ax.plot(traj[0][0], traj[0][1], 'ro', markersize=8, label='Start')
    ax.plot(0, 0, 'g*', markersize=12, label='Minimum')
    ax.set_title(f"{title}\nFinal loss: {losses[-1]:.4f}")
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("learning_rate_comparison.png", dpi=150)
print("Saved learning_rate_comparison.png")
```

**Rules of thumb:**
- Start with lr = 6e-4 for Adam/AdamW (this works for most LLMs under 1B)
- If loss spikes or NaN, reduce by 3x
- If loss decreases very slowly, increase by 2x
- Always use warmup (first 1-5% of steps with linearly increasing lr)

### Adam Optimizer

Adam combines two ideas:
1. **Momentum:** Keep a running average of past gradients (smooths out noise)
2. **Adaptive learning rate:** Give each parameter its own effective learning rate based on the magnitude of its recent gradients

```python
import torch

# Implement Adam from scratch to understand it
def adam_step(params, grads, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One step of Adam.
    
    params: list of parameter tensors
    grads: list of gradient tensors
    m: list of first moment estimates (momentum)
    v: list of second moment estimates (adaptive lr)
    t: current step number (for bias correction)
    """
    for i in range(len(params)):
        # Update biased first moment (momentum)
        m[i] = beta1 * m[i] + (1 - beta1) * grads[i]
        
        # Update biased second moment (squared gradient average)
        v[i] = beta2 * v[i] + (1 - beta2) * grads[i] ** 2
        
        # Bias correction (important in early steps when m and v are near 0)
        m_hat = m[i] / (1 - beta1 ** t)
        v_hat = v[i] / (1 - beta2 ** t)
        
        # Update parameter
        params[i] -= lr * m_hat / (torch.sqrt(v_hat) + eps)

# Test: minimize f(x, y) = x^2 + 10*y^2
x = torch.tensor([5.0, 5.0])
m = [torch.zeros_like(x)]
v = [torch.zeros_like(x)]

print("Adam optimization:")
for t in range(1, 201):
    grad = torch.tensor([2 * x[0], 20 * x[1]])  # Analytical gradient
    adam_step([x], [grad], m, v, t, lr=0.1)
    
    if t % 50 == 0:
        loss = x[0]**2 + 10 * x[1]**2
        print(f"  step={t:3d}  x=[{x[0]:.6f}, {x[1]:.6f}]  loss={loss:.6f}")
```

**Why Adam beats SGD for LLMs:**
- Different parameters need different learning rates (embeddings vs attention vs MLP)
- Adam automatically adjusts the learning rate per parameter
- Momentum smooths out the noisy gradients from mini-batch training

### Implement SGD from Scratch

```python
import torch

# Plain SGD — to appreciate why Adam is better
def sgd_from_scratch(lr=0.01, steps=1000):
    """Train a simple model with hand-written SGD."""
    
    # "Dataset": y = 2x + 3 with noise
    torch.manual_seed(42)
    X = torch.randn(100, 1)
    Y = 2 * X + 3 + 0.1 * torch.randn(100, 1)
    
    # Parameters
    w = torch.randn(1, requires_grad=True)
    b = torch.randn(1, requires_grad=True)
    
    losses = []
    for step in range(steps):
        # Forward
        y_pred = w * X + b
        loss = ((y_pred - Y) ** 2).mean()
        
        # Backward
        loss.backward()
        
        # SGD update
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
            w.grad.zero_()
            b.grad.zero_()
        
        losses.append(loss.item())
        
        if step % 200 == 0:
            print(f"  step={step:4d}  loss={loss.item():.6f}  "
                  f"w={w.item():.4f} (true: 2.0)  b={b.item():.4f} (true: 3.0)")
    
    return losses

print("SGD from scratch:")
losses = sgd_from_scratch(lr=0.01, steps=1000)
print(f"Final loss: {losses[-1]:.6f}")
```

---

## 10.4 State-Space Models and Control Theory

This section covers the math behind Mamba, Parcae, and other SSM-based architectures. If you are only building transformers, you can skim this section — but it becomes essential if you work with any recurrent or looped model.

### Continuous-Time SSMs

A state-space model is a system of differential equations:

```
dx/dt = Ax + Bu        (state equation)
y = Cx + Du             (output equation)
```

Where:
- `x` is the hidden state (what the model remembers)
- `u` is the input (current token embedding)
- `y` is the output (next hidden state or prediction)
- `A` is the state transition matrix (how memory evolves)
- `B` is the input matrix (how new information enters)
- `C` is the output matrix (how to read the state)
- `D` is the skip connection (direct input-to-output, often set to 0)

```python
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# A simple 2D continuous-time system
# dx/dt = Ax + Bu
A = np.array([
    [-0.5,  1.0],
    [-1.0, -0.5],
])  # Eigenvalues: -0.5 +/- 1.0j (stable oscillation with decay)

B = np.array([
    [1.0],
    [0.0],
])

C = np.array([[1.0, 0.0]])  # Observe first dimension

# Simulate with Euler method (crude but illustrative)
dt = 0.01
T = 20.0
steps = int(T / dt)

x = np.array([0.0, 0.0])
trajectory = [x.copy()]

for i in range(steps):
    # Input: a pulse at t=2
    t = i * dt
    u = np.array([1.0]) if 2.0 <= t <= 2.5 else np.array([0.0])
    
    # dx/dt = Ax + Bu
    dxdt = A @ x + B @ u.reshape(-1)
    x = x + dt * dxdt
    trajectory.append(x.copy())

trajectory = np.array(trajectory)
times = np.arange(len(trajectory)) * dt

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
ax1.plot(times, trajectory[:, 0], label='x[0]')
ax1.plot(times, trajectory[:, 1], label='x[1]')
ax1.axvspan(2.0, 2.5, alpha=0.2, color='red', label='Input pulse')
ax1.set_ylabel('State')
ax1.set_title('Continuous-Time SSM: Damped Oscillation')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Output
output = trajectory @ C.T
ax2.plot(times, output, label='y = Cx', color='green')
ax2.set_xlabel('Time')
ax2.set_ylabel('Output')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("ssm_continuous.png", dpi=150)
print("Saved ssm_continuous.png")
print(f"Eigenvalues of A: {np.linalg.eigvals(A)}")
```

**Intuition:** The SSM is a filter. It takes a stream of inputs and produces a stream of outputs. Matrix A controls the "memory" — it determines how quickly the system forgets old inputs. The eigenvalues of A determine the behavior: real negative eigenvalues give exponential decay, complex eigenvalues give oscillation.

### Discretization: Continuous to Discrete

We cannot use continuous-time equations with discrete tokens. We need to convert:

```
Continuous:  dx/dt = Ax + Bu
Discrete:    x[t] = A_bar * x[t-1] + B_bar * u[t]
```

The conversion uses **Zero-Order Hold (ZOH)** discretization with step size Delta:

```
A_bar = exp(A * Delta)
B_bar = (A_bar - I) * A^(-1) * B
```

```python
import numpy as np
from scipy.linalg import expm

# Continuous-time parameters
A_cont = np.array([
    [-0.5,  0.0],
    [ 0.0, -1.0],
])

B_cont = np.array([
    [1.0],
    [1.0],
])

# Discretize with ZOH (Zero-Order Hold)
def discretize_zoh(A, B, delta):
    """
    Zero-Order Hold discretization.
    This is what Mamba and Parcae use.
    
    A_bar = exp(A * delta)
    B_bar = A^{-1} (A_bar - I) B
    """
    d = A.shape[0]
    A_bar = expm(A * delta)
    B_bar = np.linalg.solve(A, (A_bar - np.eye(d)) @ np.eye(d)) @ B
    return A_bar, B_bar

delta = 0.1  # Step size (a hyperparameter in Mamba)
A_bar, B_bar = discretize_zoh(A_cont, B_cont, delta)

print("Continuous A:")
print(A_cont)
print(f"\nDiscrete A_bar (delta={delta}):")
print(A_bar)
print(f"\nEigenvalues of A_cont: {np.linalg.eigvals(A_cont)}")
print(f"Eigenvalues of A_bar:  {np.linalg.eigvals(A_bar)}")
print(f"Spectral radius of A_bar: {max(abs(np.linalg.eigvals(A_bar))):.6f}")
# Should be < 1 because continuous eigenvalues are negative

# Simulate the discrete system
seq_len = 100
u = np.zeros((seq_len, 1))
u[10:15] = 1.0  # Pulse

x = np.zeros(2)
outputs = []
for t in range(seq_len):
    x = A_bar @ x + (B_bar @ u[t]).flatten()
    outputs.append(x.copy())

outputs = np.array(outputs)
print(f"\nDiscrete simulation complete. Final state: {outputs[-1]}")
```

### Stability: Eigenvalues Inside the Unit Circle

For a discrete system `x[t] = A_bar * x[t-1] + B_bar * u[t]`, stability requires that all eigenvalues of A_bar lie inside the unit circle in the complex plane (magnitude < 1).

```python
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Demonstrate stable vs unstable discrete systems
def simulate_discrete(A_bar, B_bar, seq_len=200, input_type="pulse"):
    x = np.zeros(A_bar.shape[0])
    u = np.zeros((seq_len, B_bar.shape[1]))
    if input_type == "pulse":
        u[10:15] = 1.0
    elif input_type == "random":
        u = np.random.randn(seq_len, B_bar.shape[1]) * 0.1
    
    states = []
    for t in range(seq_len):
        x = A_bar @ x + (B_bar @ u[t]).flatten()
        states.append(np.linalg.norm(x))
    return states

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Stable: eigenvalues = [0.8, 0.5]
A1 = np.diag([0.8, 0.5])
B1 = np.array([[1.0], [1.0]])
states1 = simulate_discrete(A1, B1)
axes[0].plot(states1)
axes[0].set_title(f"Stable: eigs = {np.diag(A1)}\nrho = {max(abs(np.diag(A1))):.2f}")
axes[0].set_ylabel("||x||")

# Marginally stable: eigenvalues = [1.0, 0.5]
A2 = np.diag([1.0, 0.5])
B2 = np.array([[1.0], [1.0]])
states2 = simulate_discrete(A2, B2)
axes[1].plot(states2)
axes[1].set_title(f"Marginal: eigs = {np.diag(A2)}\nrho = {max(abs(np.diag(A2))):.2f}")

# Unstable: eigenvalues = [1.05, 0.5]
A3 = np.diag([1.05, 0.5])
B3 = np.array([[1.0], [1.0]])
states3 = simulate_discrete(A3, B3)
axes[2].plot(states3)
axes[2].set_title(f"Unstable: eigs = {np.diag(A3)}\nrho = {max(abs(np.diag(A3))):.2f}")

for ax in axes:
    ax.set_xlabel("Time step")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("stability_comparison.png", dpi=150)
print("Saved stability_comparison.png")
```

### Why Parcae Uses A = -exp(log_A)

The Parcae paper (looped transformers) needs to guarantee stability across ANY input sequence. Their solution is elegant: parameterize the transition matrix so that eigenvalues CANNOT escape the unit circle.

```python
import torch
import numpy as np

# The Parcae parameterization
# Instead of learning A directly, learn log_A and compute:
#   A = -exp(log_A)
# This guarantees A is in (-1, 0) for diagonal elements.
# After discretization, the eigenvalues of A_bar are in (0, 1).

def parcae_transition(log_A, delta):
    """
    Parcae-style parameterization.
    log_A: learnable parameter (any real number)
    delta: step size (learnable or fixed)
    
    Returns A_bar with eigenvalues guaranteed in (0, 1).
    """
    # A = -exp(log_A), guaranteed negative
    A = -torch.exp(log_A)
    
    # ZOH discretization for diagonal A:
    # A_bar = exp(A * delta) = exp(-exp(log_A) * delta)
    A_bar = torch.exp(A * delta)
    
    # A_bar is in (0, 1) because:
    # - A < 0 (always, since exp is positive and we negate)
    # - delta > 0 (step size is positive)
    # - exp(negative) is in (0, 1)
    
    return A_bar

# Demonstrate
log_A = torch.randn(8)  # 8-dimensional state
delta = torch.tensor(0.1)

A_bar = parcae_transition(log_A, delta)
print(f"log_A:  {log_A[:4].detach().numpy().round(3)}...")
print(f"A_bar:  {A_bar[:4].detach().numpy().round(3)}...")
print(f"All in (0, 1): {(A_bar > 0).all() and (A_bar < 1).all()}")
print(f"Spectral radius: {A_bar.abs().max().item():.6f}")

# No matter what log_A is, A_bar is always stable
for val in [-100, -1, 0, 1, 10, 100]:
    A_bar = parcae_transition(torch.tensor(float(val)), delta)
    print(f"  log_A={val:>6.1f}  ->  A_bar={A_bar.item():.10f}  (in (0,1): {0 < A_bar.item() < 1})")
```

### The Connection: Looped Models ARE Discrete Dynamical Systems

A looped transformer applies the same layers L times:

```
h[0] = embed(tokens)
h[1] = TransformerBlock(h[0])
h[2] = TransformerBlock(h[1])
...
h[L] = TransformerBlock(h[L-1])
```

This IS a discrete dynamical system. The TransformerBlock is the transition function. The "time steps" are loop iterations. The key question: **is this system stable?**

If the effective spectral radius of the Jacobian of TransformerBlock is > 1, the hidden states will explode as you increase L. If it is < 1, information decays. The Parcae injection term explicitly controls this stability.

```python
import torch
import torch.nn as nn

# Simplified illustration of the stability concern
class ToyLoopedModel(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.W = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x, loops=10):
        """Apply the same transform L times."""
        norms = []
        for _ in range(loops):
            x = self.norm(self.W(x) + x)  # Residual connection
            norms.append(x.norm().item())
        return x, norms

d = 64
model = ToyLoopedModel(d)
x = torch.randn(1, 10, d)

for loops in [5, 10, 20, 50]:
    out, norms = model(x, loops=loops)
    print(f"Loops={loops:3d}  ||h[0]||={norms[0]:.2f}  "
          f"||h[-1]||={norms[-1]:.2f}  ratio={norms[-1]/norms[0]:.4f}")
# If ratio >> 1: unstable. If ratio << 1: losing information.
# LayerNorm helps a lot but does not guarantee stability.
```

---

## 10.5 Information Theory Basics

### Entropy

Entropy measures the "surprise" or "uncertainty" of a probability distribution. High entropy = high uncertainty = many likely outcomes.

```
H(P) = -sum(P(x) * log2(P(x)))
```

```python
import numpy as np

def entropy(probs, base=2):
    """Compute entropy of a discrete distribution."""
    probs = np.array(probs)
    # Filter out zeros to avoid log(0)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log(probs) / np.log(base))

# Fair coin: maximum entropy for 2 outcomes
print(f"Fair coin:    H = {entropy([0.5, 0.5]):.4f} bits")

# Biased coin: less entropy
print(f"Biased coin:  H = {entropy([0.9, 0.1]):.4f} bits")

# Certain outcome: zero entropy (no surprise)
print(f"Certain:      H = {entropy([1.0, 0.0]):.4f} bits")

# Uniform over 50257 tokens (GPT-2 vocab)
uniform = np.ones(50257) / 50257
print(f"Uniform 50K:  H = {entropy(uniform):.4f} bits")  # = log2(50257) ≈ 15.6

# Peaked distribution (good model)
peaked = np.zeros(50257)
peaked[:10] = 0.1  # 10 tokens share all probability
print(f"Top-10 only:  H = {entropy(peaked):.4f} bits")  # = log2(10) ≈ 3.3

# Natural language: H ≈ 1.0-1.5 bits per character
# This is remarkably low — language is highly predictable
```

### Bits-Per-Byte (BPB)

Different tokenizers produce different numbers of tokens for the same text. BPB normalizes for this, making it possible to compare models trained with different tokenizers.

```
BPB = (CE_loss * num_tokens) / num_bytes / log(2)
```

Or equivalently: `BPB = CE_loss / (bytes_per_token * log(2))`

```python
import tiktoken
import math

enc = tiktoken.get_encoding("gpt2")

# Sample text
text = "The quick brown fox jumps over the lazy dog. This is a test of the tokenizer."
tokens = enc.encode(text)
num_bytes = len(text.encode("utf-8"))
num_tokens = len(tokens)
bytes_per_token = num_bytes / num_tokens

print(f"Text: {text[:50]}...")
print(f"Bytes: {num_bytes}")
print(f"Tokens: {num_tokens}")
print(f"Bytes per token: {bytes_per_token:.2f}")

# Convert CE loss to BPB
for ce_loss in [3.0, 3.5, 4.0, 5.0]:
    bpb = ce_loss / (bytes_per_token * math.log(2))
    print(f"  CE loss={ce_loss:.1f}  ->  BPB={bpb:.3f}")

# For reference:
# GPT-2 paper reports ~0.97 BPB on WikiText-103
# GPT-4 is estimated at ~0.6-0.7 BPB
# Shannon entropy of English ≈ 0.7-1.3 BPB
```

### Mutual Information

Mutual information measures how much knowing one variable tells you about another:

```
I(X; Y) = H(X) + H(Y) - H(X, Y)
         = KL(P(X,Y) || P(X)P(Y))
```

If X and Y are independent, I(X; Y) = 0. If X completely determines Y, I(X; Y) = H(Y).

```python
import numpy as np

def mutual_information(joint_prob):
    """
    Compute MI from a joint probability table.
    joint_prob[i,j] = P(X=i, Y=j)
    """
    # Marginals
    px = joint_prob.sum(axis=1)
    py = joint_prob.sum(axis=0)
    
    mi = 0.0
    for i in range(joint_prob.shape[0]):
        for j in range(joint_prob.shape[1]):
            if joint_prob[i, j] > 0:
                mi += joint_prob[i, j] * np.log2(
                    joint_prob[i, j] / (px[i] * py[j])
                )
    return mi

# Independent: X and Y tell you nothing about each other
independent = np.array([
    [0.25, 0.25],
    [0.25, 0.25],
])
print(f"Independent: MI = {mutual_information(independent):.4f} bits")

# Perfectly correlated: knowing X tells you Y
correlated = np.array([
    [0.5, 0.0],
    [0.0, 0.5],
])
print(f"Correlated:  MI = {mutual_information(correlated):.4f} bits")

# Partially correlated
partial = np.array([
    [0.3, 0.1],
    [0.1, 0.5],
])
print(f"Partial:     MI = {mutual_information(partial):.4f} bits")
```

**Where MI appears in LLM research:**
- Measuring how much information each layer adds
- Analyzing what attention heads learn to attend to
- Feature selection in data pipelines

---

## 10.6 Scaling Laws

### Chinchilla Scaling

Chinchilla (Hoffmann et al., 2022) established that for a fixed compute budget, there is an optimal ratio of model parameters to training tokens. The finding: **optimal tokens is approximately 20x the number of parameters.**

```python
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Chinchilla scaling law (simplified)
# L(N, D) = A / N^alpha + B / D^beta + E
# where N = params, D = tokens
# alpha ≈ 0.34, beta ≈ 0.28
# A, B, E are fitted constants

# The key result: for fixed compute C ∝ 6*N*D,
# optimal N* ∝ C^0.50, optimal D* ∝ C^0.50
# So D* / N* ≈ 20 (the "Chinchilla ratio")

def chinchilla_loss(N, D, A=406.4, B=410.7, alpha=0.34, beta=0.28, E=1.69):
    """Predicted loss given N params and D tokens."""
    return A / (N ** alpha) + B / (D ** beta) + E

# What happens at different param/token ratios?
param_counts = [10e6, 50e6, 124e6, 350e6, 1e9]

print("Chinchilla scaling predictions:")
print(f"{'Params':>10s}  {'Chinchilla tokens':>18s}  {'Predicted loss':>15s}")
print("-" * 50)

for N in param_counts:
    D_optimal = 20 * N  # Chinchilla ratio
    loss = chinchilla_loss(N, D_optimal)
    print(f"{N/1e6:>8.0f}M  {D_optimal/1e9:>15.1f}B  {loss:>15.4f}")

# Plot: loss vs tokens for different model sizes
fig, ax = plt.subplots(figsize=(10, 6))

for N, label in [(50e6, "50M"), (124e6, "124M"), (350e6, "350M")]:
    tokens = np.logspace(8, 11, 100)  # 100M to 100B tokens
    losses = [chinchilla_loss(N, D) for D in tokens]
    ax.semilogx(tokens, losses, label=f"{label} params")
    
    # Mark the Chinchilla-optimal point
    D_opt = 20 * N
    L_opt = chinchilla_loss(N, D_opt)
    ax.plot(D_opt, L_opt, 'o', markersize=8)

ax.set_xlabel("Training tokens")
ax.set_ylabel("Predicted loss")
ax.set_title("Chinchilla Scaling: Loss vs Training Tokens")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("chinchilla_scaling.png", dpi=150)
print("\nSaved chinchilla_scaling.png")
```

### Compute-Optimal Training

Given a compute budget (measured in FLOPs), how should you split between model size and training tokens?

```python
import numpy as np

# Approximate FLOPs for one forward pass: 2 * N * seq_len
# Total FLOPs for training: ~6 * N * D (forward + backward ≈ 3x forward)

def compute_flops(N, D):
    """Approximate total training FLOPs."""
    return 6 * N * D

def optimal_allocation(total_flops):
    """
    Given a compute budget, find optimal N and D.
    Chinchilla: N* ∝ C^0.50, D* = C / (6 * N*)
    """
    # Optimal N ≈ (C / 120)^0.5  (since D = 20N, C = 6*N*20N = 120N^2)
    N_opt = (total_flops / 120) ** 0.5
    D_opt = total_flops / (6 * N_opt)
    return N_opt, D_opt

print("Compute-optimal allocations:")
print(f"{'Budget (FLOPs)':>20s}  {'Params':>12s}  {'Tokens':>12s}  {'D/N ratio':>10s}")
print("-" * 60)

for budget_flops in [1e17, 1e18, 1e19, 1e20, 1e21]:
    N, D = optimal_allocation(budget_flops)
    print(f"{budget_flops:>20.0e}  {N/1e6:>10.1f}M  {D/1e9:>10.1f}B  {D/N:>10.1f}")

# Your RTX 4060 Ti budget:
# FP16 tensor core: ~176 TFLOPS
# Practical sustained: ~100 TFLOPS
# 24 hours = 86400 seconds
# Budget = 100e12 * 86400 = 8.64e18 FLOPs
print("\n--- Your Budget (RTX 4060 Ti, 24 hours) ---")
your_budget = 100e12 * 86400
N, D = optimal_allocation(your_budget)
print(f"Budget: {your_budget:.2e} FLOPs")
print(f"Optimal params: {N/1e6:.0f}M")
print(f"Optimal tokens: {D/1e9:.1f}B")
print(f"This suggests a ~85M model trained on ~1.7B tokens in 24 hours")
print(f"Or a 124M model trained on ~1.2B tokens (slightly compute-suboptimal)")
```

### Overtraining vs Undertraining

What happens when you deviate from the Chinchilla ratio?

```python
import numpy as np

def chinchilla_loss(N, D, A=406.4, B=410.7, alpha=0.34, beta=0.28, E=1.69):
    return A / (N ** alpha) + B / (D ** beta) + E

N = 124e6  # Fixed model size

ratios = [1, 5, 10, 20, 50, 100, 200]
print(f"\n124M model at different D/N ratios:")
print(f"{'D/N':>6s}  {'Tokens':>10s}  {'Loss':>8s}  {'vs Optimal':>12s}")
print("-" * 45)

optimal_loss = chinchilla_loss(N, 20 * N)

for ratio in ratios:
    D = ratio * N
    loss = chinchilla_loss(N, D)
    delta = loss - optimal_loss
    status = "<<< optimal" if ratio == 20 else ""
    print(f"{ratio:>6d}  {D/1e9:>8.1f}B  {loss:>8.4f}  {delta:>+10.4f}  {status}")
```

**Key takeaway:** Undertraining (too few tokens) wastes model parameters. Overtraining (too many tokens) gives diminishing returns. The Chinchilla ratio of ~20 tokens per parameter is the sweet spot for compute efficiency.

### Parcae Scaling Law

The Parcae paper (looped transformers) proposes a new scaling dimension: loop depth. Their finding:

```
Optimal loop depth L* ∝ compute^0.40
```

This means that as you increase compute, you should increase loop depth faster than model width.

```python
import numpy as np

# Parcae scaling: optimal loops as a function of compute
def optimal_loops(compute_flops, coefficient=0.5):
    """
    L* ∝ C^0.40
    The coefficient is approximate — the paper fits it empirically.
    """
    return coefficient * (compute_flops ** 0.40)

print("Parcae: Optimal loop depth vs compute:")
for budget in [1e17, 1e18, 1e19, 1e20]:
    loops = optimal_loops(budget)
    print(f"  Budget: {budget:.0e} FLOPs  ->  Optimal loops: ~{loops:.0f}")

# For your hardware (8.64e18 FLOPs in 24h):
your_loops = optimal_loops(8.64e18)
print(f"\n  Your budget (24h on 4060 Ti): ~{your_loops:.0f} loops")
print(f"  This aligns with JORMUNGANDR-HALO using 4-8 loops at 168M params")
```

---

## 10.7 Paper Reading Workflow

Reading ML papers is a skill. Most papers are 8-12 pages, but you do not need to read every word. Here is a systematic workflow.

### Step 1: Abstract + Conclusion (2 minutes)

Read the abstract and conclusion first. This tells you:
- What problem they solve
- What their approach is (one sentence)
- How well it works (numbers)

If the abstract is not relevant to your work, stop here. You have saved 30 minutes.

### Step 2: Figures and Tables (5 minutes)

Look at every figure and table. A good paper tells its story through visuals:
- **Figure 1:** Usually the architecture diagram
- **Main results table:** How they compare to baselines
- **Ablation table:** What components actually matter

```
Example from the XSA paper:
- Figure 1: Shows XSA removing the self-value component from attention output
- Table 1: XSA improves loss by 0.02-0.05 across model sizes
- Table 3 (ablation): Removing the projection step hurts badly
```

### Step 3: Method Section (15 minutes)

Now read the method in detail. Focus on:
- The key equation(s)
- What is new vs what is standard (e.g., "we modify the attention computation by...")
- Implementation details (hidden dim, number of layers, learning rate)

**For XSA, the key equation is:**

```
output = attn_output - proj_{self_value}(attn_output)
```

In code:

```python
import torch

def xsa_attention(Q, K, V, mask=None):
    """
    XSA: Cross-State Attention.
    Standard attention, but remove the self-value component from the output.
    """
    d_k = Q.shape[-1]
    
    # Standard attention
    scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn_weights = torch.softmax(scores, dim=-1)
    attn_output = attn_weights @ V  # (B, H, T, d_k)
    
    # XSA modification: remove the self-value component
    # self_value[i] = V[i] (the value vector of position i itself)
    # We want: output[i] = attn_output[i] - proj_{V[i]}(attn_output[i])
    
    # Projection: proj_v(u) = (u . v / ||v||^2) * v
    # Compute dot product of attn_output with self-value
    self_value = V  # (B, H, T, d_k) — V[i] for each position
    
    # Dot product along the last dimension
    dot = (attn_output * self_value).sum(dim=-1, keepdim=True)  # (B, H, T, 1)
    norm_sq = (self_value * self_value).sum(dim=-1, keepdim=True) + 1e-8  # (B, H, T, 1)
    
    # Projection
    projection = (dot / norm_sq) * self_value  # (B, H, T, d_k)
    
    # Remove projection
    xsa_output = attn_output - projection
    
    return xsa_output

# Test
B, H, T, D = 2, 4, 16, 32
Q = torch.randn(B, H, T, D)
K = torch.randn(B, H, T, D)
V = torch.randn(B, H, T, D)

standard_out = torch.softmax(Q @ K.transpose(-2, -1) / D**0.5, dim=-1) @ V
xsa_out = xsa_attention(Q, K, V)

print(f"Standard attention output shape: {standard_out.shape}")
print(f"XSA output shape: {xsa_out.shape}")
print(f"Output norm reduction: {xsa_out.norm() / standard_out.norm():.4f}")

# Verify: XSA output should be roughly orthogonal to self-value
# (dot product should be near zero)
dot_check = (xsa_out * V).sum(dim=-1)
print(f"Mean dot(XSA_output, self_value): {dot_check.mean().item():.6f}")
print(f"This should be near zero (orthogonal)")
```

### Step 4: Ablations (10 minutes)

The ablation table tells you what actually matters. Most papers have 5-10 components, but only 2-3 are essential.

```
Reading ablation tables:
1. Look at the "Full model" row (best result)
2. Look at which rows when removed cause the biggest drop
3. Those are the essential components
4. Everything else is nice-to-have

Example:
| Configuration     | Loss | Delta from Full |
|-------------------|------|-----------------|
| Full model        | 3.12 | --              |
| - XSA projection  | 3.19 | +0.07 (big!)    |
| - Layer norm      | 3.13 | +0.01 (minor)   |
| - Weight init     | 3.14 | +0.02 (minor)   |

Conclusion: XSA projection is the key contribution.
Layer norm variant and weight init barely matter.
```

### Step 5: Related Work (5 minutes)

Skim the related work section for:
- Papers they compare against (potential baselines for your experiments)
- Papers they build on (read these if you want deeper understanding)
- Papers in the same space (competitors / alternatives)

### Practice: Guided Reading of the XSA Paper

Here is a checklist for reading any paper:

```markdown
## Paper Reading Template

**Title:**
**Authors:**
**Date:**

### 1. Problem
What problem does this paper solve? (1 sentence)

### 2. Key Idea
What is the main technical contribution? (1 sentence)

### 3. Formula
The core equation (copy it here):

### 4. Results
Main result from the results table:
- Baseline: ___
- Their method: ___
- Improvement: ___

### 5. Ablation
Which component matters most?
Which components are optional?

### 6. Limitations
What does the paper NOT show?

### 7. Relevance
How does this apply to my model?
Should I implement this? (Y/N, why)
```

---

## Exercises

### Exercise 1: Spectral Radius and Stability

Compute the spectral radius of this matrix and determine whether the system is stable:

```python
import numpy as np

A = np.array([
    [0.6, 0.3],
    [-0.2, 0.8],
])

# YOUR CODE: compute eigenvalues, spectral radius, determine stability
eigenvalues = np.linalg.eigvals(A)
rho = np.max(np.abs(eigenvalues))

print(f"Eigenvalues: {eigenvalues}")
print(f"Spectral radius: {rho:.6f}")
print(f"Stable (rho < 1): {rho < 1}")

# Verify by simulating 200 steps
x = np.array([1.0, 1.0])
for i in range(200):
    x = A @ x
print(f"After 200 steps: ||x|| = {np.linalg.norm(x):.10f}")
print(f"State: {x}")
# If stable, ||x|| should be near 0 (decayed)
# If rho is close to 1, it will take many steps to fully decay
```

### Exercise 2: Implement DPO Loss

DPO (Direct Preference Optimization) loss from Rafailov et al. (2023):

```
L_DPO = -log(sigmoid(beta * (log(pi_chosen / pi_ref_chosen) - log(pi_rejected / pi_ref_rejected))))
```

Which simplifies to:

```
L_DPO = -log(sigmoid(beta * (log_ratio_chosen - log_ratio_rejected)))
```

where `log_ratio = log(pi(y|x)) - log(pi_ref(y|x))`

```python
import torch
import torch.nn.functional as F

def dpo_loss(
    log_probs_chosen,      # log pi(y_chosen | x) under the current policy
    log_probs_rejected,    # log pi(y_rejected | x) under the current policy
    ref_log_probs_chosen,  # log pi_ref(y_chosen | x) under the reference model
    ref_log_probs_rejected,# log pi_ref(y_rejected | x) under the reference model
    beta=0.1,              # Temperature parameter
):
    """
    Compute DPO loss.
    
    The model should assign higher probability to chosen responses
    (relative to the reference) than to rejected responses.
    """
    # Log ratios: how much has the policy changed from the reference?
    log_ratio_chosen = log_probs_chosen - ref_log_probs_chosen
    log_ratio_rejected = log_probs_rejected - ref_log_probs_rejected
    
    # The model is rewarded for increasing chosen probability
    # and penalized for increasing rejected probability
    logits = beta * (log_ratio_chosen - log_ratio_rejected)
    
    # Negative log-sigmoid loss
    loss = -F.logsigmoid(logits).mean()
    
    # Useful metrics
    rewards_chosen = beta * log_ratio_chosen.detach()
    rewards_rejected = beta * log_ratio_rejected.detach()
    reward_margin = (rewards_chosen - rewards_rejected).mean()
    
    return loss, {
        "loss": loss.item(),
        "reward_margin": reward_margin.item(),
        "rewards_chosen_mean": rewards_chosen.mean().item(),
        "rewards_rejected_mean": rewards_rejected.mean().item(),
        "accuracy": (logits > 0).float().mean().item(),  # How often chosen > rejected
    }

# Test with synthetic data
batch_size = 16
beta = 0.1

# Simulate log probs (these would come from your model in practice)
log_probs_chosen = torch.randn(batch_size) - 2.0     # Log probs are negative
log_probs_rejected = torch.randn(batch_size) - 2.0
ref_log_probs_chosen = torch.randn(batch_size) - 2.0
ref_log_probs_rejected = torch.randn(batch_size) - 2.0

loss, metrics = dpo_loss(
    log_probs_chosen, log_probs_rejected,
    ref_log_probs_chosen, ref_log_probs_rejected,
    beta=beta,
)

print(f"DPO Loss: {metrics['loss']:.4f}")
print(f"Reward margin (chosen - rejected): {metrics['reward_margin']:.4f}")
print(f"Accuracy (chosen > rejected): {metrics['accuracy']:.2%}")

# After training, accuracy should be high (>80%) and reward margin positive
# At initialization with random log probs, accuracy should be ~50%
```

### Exercise 3: Read the XSA Paper

Use the paper reading template above. For this exercise, you can find the XSA paper at: arxiv.org/abs/2504.00927 (or search for "Cross-State Attention").

Answer these questions:

1. What is the formula for the XSA projection?
2. In the ablation table, what happens when you remove the projection step?
3. Does XSA help more at smaller or larger model sizes?
4. Would you use XSA in your model? Why or why not?

If you cannot access the paper, use the XSA code example from Section 10.7 and answer:
- What does removing the self-value component force the model to do?
- Why might this help with training? (Hint: it prevents a token from reinforcing its own representation through attention.)

---

## Checkpoint

Before moving to Part 11, verify:
- [ ] You can compute eigenvalues of a 2x2 matrix and determine stability
- [ ] You can compute cross-entropy loss by hand and with `F.cross_entropy`
- [ ] You can explain what perplexity of 20 means in plain language
- [ ] You can compute KL divergence between two distributions
- [ ] You can explain why Parcae uses `-exp(log_A)` for parameterization
- [ ] You can read the XSA paper (or code) and explain the method to someone else
- [ ] You understand why Chinchilla says ~20 tokens per parameter is optimal
- [ ] You can implement DPO loss from the formula

---

## Quick Reference Card

Keep this handy when reading papers:

```
NOTATION           MEANING                              CODE
─────────────────────────────────────────────────────────────────────
x ∈ R^d            d-dimensional vector                 x = torch.randn(d)
W ∈ R^{m×n}        m-by-n matrix                        W = torch.randn(m, n)
||x||              L2 norm (Euclidean length)            torch.norm(x)
x · y              dot product                           torch.dot(x, y)
A @ B              matrix multiplication                 torch.matmul(A, B)
A^T                transpose                             A.T or A.transpose(-2, -1)
λ(A)               eigenvalues                           torch.linalg.eigvals(A)
ρ(A)               spectral radius (max |λ|)            max(abs(eigvals(A)))
σ(x)               softmax                              torch.softmax(x, dim=-1)
log p(x)           log probability                       F.log_softmax(logits, dim=-1)
H(P)               entropy                              -(P * log(P)).sum()
KL(P||Q)           KL divergence                         F.kl_div(Q.log(), P)
CE(P, Q)           cross-entropy                         F.cross_entropy(logits, targets)
PPL                perplexity = exp(CE)                  math.exp(ce_loss)
∇_θ L              gradient of L w.r.t. θ                loss.backward(); param.grad
dx/dt = Ax + Bu    continuous-time SSM                   (section 10.4)
x[t] = Āx[t-1]    discrete-time SSM                     (section 10.4)
```

---

**Previous: [Part 09 -- Train-Evaluate-Benchmark Framework](09_train_eval_benchmark.md)**
**Next: [Part 11 -- Architecture Design: From Papers to Code](11_architecture_design.md)**
