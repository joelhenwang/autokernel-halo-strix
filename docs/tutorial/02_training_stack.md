# Part 02: Training Stack -- GPT-2 on BabyLM with torch.compile

## Goal
Build a complete LLM training pipeline from scratch: model definition, tokenizer, dataset, training loop, mixed precision, and torch.compile. By the end, you will have a 124M-parameter model training at >10K tokens/sec on your 4060 Ti.

## Why This Matters
Every later part builds on this training stack. Profiling (Part 03) needs a running model. CUDA kernels (Part 04) need a model to plug into. Understanding the training loop end-to-end means you can debug anything -- you wrote every line.

---

## 2.1 Understanding GPT-2 Architecture

GPT-2 is an autoregressive language model. Given a sequence of tokens, it predicts the next token. That is the entire job. Everything below exists to make that prediction accurate.

### The Data Flow

```
Input token IDs:  [The, cat, sat, on]
       |
       v
+------------------+
| Token Embedding  |  vocab_size x d_model lookup table
+------------------+
       |
       v
+------------------+
| + Position Info  |  RoPE (rotary position embeddings)
+------------------+
       |
       v
+------------------+
| Transformer      |  x12 identical blocks:
| Block 1..12      |    Attention -> FFN -> repeat
+------------------+
       |
       v
+------------------+
| RMSNorm          |  Final normalization
+------------------+
       |
       v
+------------------+
| LM Head (Linear) |  d_model -> vocab_size (50257 logits)
+------------------+
       |
       v
Output logits:  probability over next token
```

### Component Breakdown

**Token Embeddings.** A lookup table of shape `(vocab_size, d_model)`. Each of the 50,257 tokens in GPT-2's vocabulary gets a learned vector of dimension 768. This is literally `nn.Embedding(50257, 768)` -- a matrix where row `i` is the vector for token `i`.

**Positional Information (RoPE).** Transformers have no built-in notion of word order -- "cat sat on mat" and "mat on sat cat" produce identical attention patterns without position encoding. Rotary Position Embeddings (RoPE) encode position by rotating the query and key vectors in attention. Unlike the original GPT-2's learned position embeddings, RoPE generalizes better to sequence lengths not seen during training. We use RoPE because it is the modern standard (LLaMA, Mistral, Qwen all use it).

**Transformer Block.** Each block has two sub-layers:
1. **Multi-Head Attention** -- lets each token look at all previous tokens and decide what to attend to. With Grouped Query Attention (GQA), we use fewer key/value heads than query heads, saving memory without hurting quality.
2. **Feed-Forward Network (FFN)** -- a two-layer MLP that processes each token independently. We use SwiGLU, which multiplies a gated pathway (sigmoid-weighted) with a linear pathway. SwiGLU consistently outperforms ReLU in practice.

Both sub-layers use a pre-norm pattern: normalize first, then apply the layer, then add the residual. This is more stable than post-norm (original Transformer) during training.

**RMSNorm.** Root Mean Square normalization. Simpler than LayerNorm -- it skips the mean subtraction and bias, using only the RMS of the input for scaling. Faster and just as effective. Formula: `output = x / sqrt(mean(x^2) + eps) * weight`.

**LM Head.** A linear projection from `d_model` (768) to `vocab_size` (50,257). The output is 50,257 raw scores (logits). During training, we apply cross-entropy loss against the true next token. During inference, we sample from the softmax of these logits.

### Why GPT-2 and Not Something Newer?

Three reasons:

1. **Simplicity.** GPT-2 is a clean stack of transformer blocks. No mixture-of-experts, no hybrid SSM layers, no sliding window attention. When something breaks, you know where to look.

2. **Well-understood.** Thousands of people have trained GPT-2 variants. Expected loss curves, throughput numbers, and failure modes are documented. You have a ground truth to compare against.

3. **Good baseline.** Our "modern GPT-2" (RoPE + GQA + SwiGLU + RMSNorm) is actually a small LLaMA. This is the architecture that powers production models. The concepts transfer directly to training 7B+ models later.

---

## 2.2 Building GPT-2 from Scratch in PyTorch

Our model configuration:

| Parameter | Value | Notes |
|-----------|-------|-------|
| d_model | 768 | Hidden dimension |
| n_layers | 12 | Transformer blocks |
| n_heads | 12 | Query heads (64 dim each) |
| n_kv_heads | 4 | Key/Value heads (GQA, 3:1 ratio) |
| vocab_size | 50257 | GPT-2 tokenizer vocabulary |
| max_seq_len | 1024 | Context window |
| ffn_dim | 2048 | SwiGLU intermediate size |

This gives roughly 124M parameters, fitting comfortably in 16GB VRAM with full AdamW state.

### Full Model Code

Create `models/gpt2_modern.py`:

```python
"""
models/gpt2_modern.py -- Modern GPT-2 (LLaMA-style) 124M parameter model.

Architecture: RoPE + GQA + SwiGLU + RMSNorm (pre-norm).
This is NOT the original GPT-2. It is a modern rewrite using techniques
from LLaMA/Mistral that are strictly better in every measurable way.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class GPT2Config:
    vocab_size: int = 50257
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12       # query heads
    n_kv_heads: int = 4     # key/value heads (GQA)
    max_seq_len: int = 1024
    ffn_dim: int = 2048     # SwiGLU intermediate
    norm_eps: float = 1e-6
    dropout: float = 0.0    # no dropout -- modern practice for <1B models


# ---------------------------------------------------------------------------
# RMSNorm -- simpler and faster than LayerNorm
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    Unlike LayerNorm, RMSNorm does NOT subtract the mean or add a bias.
    It only rescales by the root-mean-square, then applies a learned weight.
    
    Formula: output = x / sqrt(mean(x^2) + eps) * weight
    
    Why RMSNorm over LayerNorm?
    - ~10-15% faster (no mean computation, no bias)
    - Equal or better training quality in practice
    - Used by LLaMA, Mistral, Gemma, and most modern LLMs
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, dim)
        # Compute in fp32 for numerical stability, even if x is fp16
        x_float = x.float()
        rms = torch.sqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        output = x_float / rms
        # Cast back to input dtype and apply learned scale
        return (output * self.weight).to(x.dtype)


# ---------------------------------------------------------------------------
# Rotary Position Embeddings (RoPE)
# ---------------------------------------------------------------------------
def precompute_rope_frequencies(dim: int, max_seq_len: int, theta: float = 10000.0):
    """
    Precompute the complex exponentials for RoPE.
    
    RoPE encodes position by rotating pairs of dimensions in the query/key
    vectors. Position k rotates dimension pair (i, i+1) by angle k * freq_i,
    where freq_i decreases exponentially with dimension index.
    
    This means nearby positions have similar rotations (small angle difference),
    and distant positions have very different rotations. The model learns to
    use these rotation patterns to understand word order.
    """
    # Frequency for each dimension pair: theta^(-2i/dim) for i = 0, 1, ..., dim/2-1
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    # Position indices
    t = torch.arange(max_seq_len).float()
    # Outer product: (seq_len, dim/2) -- angle for each position and dimension pair
    angles = torch.outer(t, freqs)
    # Complex exponentials: e^(i * angle) = cos(angle) + i*sin(angle)
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    return freqs_cis  # shape: (max_seq_len, dim/2)


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary embeddings to query or key tensor.
    
    x shape: (batch, n_heads, seq_len, head_dim)
    freqs_cis shape: (seq_len, head_dim/2) -- complex numbers
    
    We reshape x into pairs of consecutive dimensions, treat each pair as
    a complex number, multiply by the rotation, then reshape back.
    """
    batch, n_heads, seq_len, head_dim = x.shape
    # Reshape to pairs: (batch, n_heads, seq_len, head_dim/2, 2)
    x_pairs = x.float().reshape(batch, n_heads, seq_len, -1, 2)
    # View as complex: (batch, n_heads, seq_len, head_dim/2)
    x_complex = torch.view_as_complex(x_pairs)
    # Broadcast freqs_cis to match: (1, 1, seq_len, head_dim/2)
    freqs = freqs_cis[:seq_len].unsqueeze(0).unsqueeze(0)
    # Multiply (rotate) and convert back to real pairs
    x_rotated = torch.view_as_real(x_complex * freqs)
    # Flatten pairs back: (batch, n_heads, seq_len, head_dim)
    return x_rotated.reshape(batch, n_heads, seq_len, head_dim).to(x.dtype)


# ---------------------------------------------------------------------------
# Grouped Query Attention (GQA)
# ---------------------------------------------------------------------------
class GroupedQueryAttention(nn.Module):
    """
    Multi-head attention with fewer key/value heads than query heads.
    
    Standard multi-head attention: 12 Q heads, 12 K heads, 12 V heads.
    GQA with 4 KV heads: 12 Q heads, 4 K heads, 4 V heads.
    
    Each KV head is shared by (n_heads / n_kv_heads) = 3 query heads.
    This saves 2/3 of KV memory with negligible quality loss.
    
    Why GQA?
    - KV cache size during inference drops proportionally
    - Training speed is slightly faster (fewer parameters to update)
    - Quality difference vs full MHA is within noise for <1B models
    """
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        self.n_rep = config.n_heads // config.n_kv_heads  # how many Q heads per KV head

        # Separate projections for Q, K, V
        self.w_q = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.w_k = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.w_v = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.w_o = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.w_q(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        # q: (batch, n_heads, seq_len, head_dim)
        # k, v: (batch, n_kv_heads, seq_len, head_dim)

        # Apply RoPE to queries and keys (NOT values -- values don't need position)
        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)

        # Expand KV heads to match Q heads by repeating
        # Each KV head is shared by n_rep query heads
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)  # (batch, n_heads, seq_len, head_dim)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # Scaled dot-product attention with causal mask
        # PyTorch 2.0+ has a fused implementation that handles the mask efficiently
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,    # causal mask applied automatically with is_causal=True
            dropout_p=0.0,
            is_causal=True,    # this enables the causal (autoregressive) mask
        )
        # attn_out: (batch, n_heads, seq_len, head_dim)

        # Concatenate heads and project
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.w_o(attn_out)


# ---------------------------------------------------------------------------
# SwiGLU Feed-Forward Network
# ---------------------------------------------------------------------------
class SwiGLUFFN(nn.Module):
    """
    SwiGLU: Swish-Gated Linear Unit.
    
    Standard FFN:   output = W2(ReLU(W1(x)))
    SwiGLU FFN:     output = W2(SiLU(W_gate(x)) * W_up(x))
    
    SiLU(x) = x * sigmoid(x), also called "swish".
    The gate path (SiLU) controls how much of the up-projection passes through.
    
    Why SwiGLU over ReLU?
    - Consistently lower loss at same parameter count (PaLM, LLaMA papers)
    - The gating mechanism lets the network learn more expressive transformations
    - Extra parameter cost (3 matrices vs 2) is worth it
    
    Note: ffn_dim is the intermediate size. Total params = 3 * d_model * ffn_dim
    (gate + up + down), vs 2 * d_model * ffn_dim for standard FFN. We compensate
    by using a smaller ffn_dim (2048 vs the usual 4*768=3072).
    """
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.w_gate = nn.Linear(config.d_model, config.ffn_dim, bias=False)
        self.w_up = nn.Linear(config.d_model, config.ffn_dim, bias=False)
        self.w_down = nn.Linear(config.ffn_dim, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gate path: apply SiLU activation
        gate = F.silu(self.w_gate(x))
        # Up path: linear projection (no activation)
        up = self.w_up(x)
        # Element-wise multiply (gating) then project back down
        return self.w_down(gate * up)


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    """
    One transformer block: Attention + FFN with pre-norm residual connections.
    
    Pre-norm pattern:
        x = x + Attention(RMSNorm(x))
        x = x + FFN(RMSNorm(x))
    
    The residual connection (x + ...) ensures gradients flow directly through
    the network without vanishing. RMSNorm before each sub-layer keeps
    activations stable. This ordering (norm-first) is more stable than the
    original Transformer's post-norm pattern.
    """
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attn = GroupedQueryAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.ffn = SwiGLUFFN(config)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        # Attention with residual
        x = x + self.attn(self.attn_norm(x), freqs_cis)
        # FFN with residual
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------
class GPT2Modern(nn.Module):
    """
    Modern GPT-2: LLaMA-style architecture at 124M parameter scale.
    
    Changes from original GPT-2:
    - RMSNorm instead of LayerNorm
    - RoPE instead of learned position embeddings
    - GQA instead of full multi-head attention
    - SwiGLU instead of GELU FFN
    - No bias terms anywhere (modern convention)
    
    These changes collectively improve training stability, throughput,
    and final model quality.
    """
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config

        # Token embedding: maps token IDs to vectors
        # No position embedding -- RoPE handles position in attention
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)

        # Stack of transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # Final normalization before LM head
        self.final_norm = RMSNorm(config.d_model, config.norm_eps)

        # LM head: project hidden states to vocabulary logits
        # Weight tying: share weights with token embedding (saves ~37M params)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

        # Precompute RoPE frequencies (these are constants, not parameters)
        head_dim = config.d_model // config.n_heads
        freqs_cis = precompute_rope_frequencies(head_dim, config.max_seq_len)
        self.register_buffer('freqs_cis', freqs_cis, persistent=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Weight initialization following GPT-2 / LLaMA conventions.
        
        Linear layers: Normal(0, 0.02)
        Embeddings: Normal(0, 0.02)
        
        The 0.02 std dev is small enough to start training stably but large
        enough that gradients are not vanishingly small.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor = None):
        """
        Forward pass.
        
        Args:
            input_ids: (batch, seq_len) token IDs
            targets: (batch, seq_len) target token IDs for loss computation
            
        Returns:
            logits: (batch, seq_len, vocab_size) if targets is None
            loss: scalar cross-entropy loss if targets is provided
        """
        batch, seq_len = input_ids.shape

        # Token embeddings
        x = self.tok_emb(input_ids)  # (batch, seq_len, d_model)

        # Pass through all transformer blocks
        for layer in self.layers:
            x = layer(x, self.freqs_cis)

        # Final norm
        x = self.final_norm(x)

        if targets is not None:
            # Training: compute loss
            # Project to vocabulary logits
            logits = self.lm_head(x)  # (batch, seq_len, vocab_size)
            # Cross-entropy loss: compare each position's prediction to the target
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),  # flatten to (batch*seq_len, vocab_size)
                targets.view(-1),                   # flatten to (batch*seq_len,)
                ignore_index=-1,                    # ignore padding tokens
            )
            return loss
        else:
            # Inference: only compute logits for the last position (next token)
            logits = self.lm_head(x[:, -1, :])  # (batch, vocab_size)
            return logits

    def count_parameters(self):
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # Subtract tied weights (counted twice)
        tied = self.tok_emb.weight.numel()
        return total - tied, trainable - tied


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    config = GPT2Config()
    model = GPT2Modern(config).cuda()
    total, trainable = model.count_parameters()
    print(f"Parameters: {total / 1e6:.1f}M total, {trainable / 1e6:.1f}M trainable")

    # Test forward pass
    x = torch.randint(0, config.vocab_size, (2, 128), device='cuda')
    targets = torch.randint(0, config.vocab_size, (2, 128), device='cuda')
    loss = model(x, targets)
    print(f"Test loss: {loss.item():.4f} (expected ~10.8 = ln(50257))")
    print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
```

Run this to verify:
```bash
python models/gpt2_modern.py
```

Expected output:
```
Parameters: ~124M total, ~124M trainable
Test loss: ~10.82 (random weights predict uniformly over 50257 tokens)
Memory used: ~0.5 GB
```

If the loss is close to `ln(50257) = 10.825`, the model is wired correctly. Random weights should produce approximately uniform predictions, and cross-entropy of a uniform distribution over N classes is `ln(N)`.

---

## 2.3 Tokenization with tiktoken

### What is Tokenization?

Language models do not read characters or words. They read **tokens** -- subword units learned from a large corpus. The GPT-2 tokenizer uses **Byte-Pair Encoding (BPE)**, which works like this:

1. Start with individual bytes (256 possible values).
2. Find the most frequent adjacent pair in the training corpus (e.g., "t" + "h").
3. Merge that pair into a new token ("th").
4. Repeat 50,000 times.

The result: common words become single tokens ("the" = 1 token), rare words get split ("Kubernetes" = "Kub" + "ern" + "etes" = 3 tokens), and any byte sequence can be encoded (no "unknown token" problem).

### Using tiktoken

```python
"""tokenizer_demo.py -- Understanding GPT-2 tokenization."""
import tiktoken

# Load GPT-2 tokenizer (50,257 tokens)
enc = tiktoken.get_encoding("gpt2")

# Basic encoding/decoding
text = "The cat sat on the mat."
tokens = enc.encode(text)
print(f"Text: {text}")
print(f"Tokens: {tokens}")
print(f"Token count: {len(tokens)}")
print(f"Decoded: {enc.decode(tokens)}")

# See individual tokens
for t in tokens:
    print(f"  {t:6d} -> '{enc.decode([t])}'")
```

Expected output:
```
Text: The cat sat on the mat.
Tokens: [464, 3797, 3332, 319, 262, 2603, 13]
Token count: 7
Decoded: The cat sat on the mat.
     464 -> 'The'
    3797 -> ' cat'
    3332 -> ' sat'
     319 -> ' on'
     262 -> ' the'
    2603 -> ' mat'
      13 -> '.'
```

Notice that spaces are attached to the beginning of words, not the end. This is a BPE convention.

### The EOS Token

When training on multiple documents, we need a separator so the model does not learn to continue one document into another. GPT-2 uses token ID **50256** (`<|endoftext|>`) as the end-of-sequence (EOS) token.

```python
eos_token = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})
print(f"EOS token: {eos_token}")  # [50256]
```

When preparing training data, we insert EOS between every document:

```
[Document 1 tokens] [50256] [Document 2 tokens] [50256] [Document 3 tokens] ...
```

This teaches the model that after EOS, a new unrelated context begins.

---

## 2.4 Dataset: BabyLM

### What is BabyLM?

BabyLM is a curated dataset of ~100M tokens designed to mimic the linguistic input a child receives in the first 13 years of life. It includes children's books, transcribed speech, Wikipedia (simplified), and other child-directed text.

Why BabyLM for this tutorial:
- **Small enough** to train on in hours, not days
- **High quality** -- curated, not scraped
- **Standard benchmark** -- you can compare your results to published baselines
- **Fits in RAM** -- the entire dataset is ~700MB of text

### Downloading BabyLM

```bash
# Download BabyLM strict-small (10M words, ~100M tokens)
mkdir -p datasets
cd datasets
# Option 1: From HuggingFace
pip install datasets
python -c "
from datasets import load_dataset
ds = load_dataset('babylm/babylm-data', 'strict_small', trust_remote_code=True)
# Save to text files
for split in ['train', 'dev', 'test']:
    if split in ds:
        with open(f'babylm_{split}.txt', 'w') as f:
            for ex in ds[split]:
                f.write(ex['text'] + '\n')
print('Done. Files: babylm_train.txt, babylm_dev.txt, babylm_test.txt')
"
```

If the HuggingFace route fails, download directly:
```bash
wget https://github.com/babylm/babylm.github.io/raw/main/babylm_data/babylm_10M.zip
unzip babylm_10M.zip
# Concatenate all training files
cat babylm_10M/*.train > babylm_train.txt
cat babylm_10M/*.dev > babylm_dev.txt
```

### Building the Dataset Class

The key operation: tokenize the entire corpus, then slice it into fixed-length windows for next-token prediction.

```python
"""training/dataset.py -- BabyLM dataset for next-token prediction."""
import os
import torch
from torch.utils.data import Dataset
import tiktoken


class BabyLMDataset(Dataset):
    """
    Tokenize the full corpus, concatenate with EOS tokens between documents,
    then serve fixed-length windows.
    
    Each sample is (input, target) where:
      input  = tokens[i : i + block_size]
      target = tokens[i+1 : i + block_size + 1]
    
    This is next-token prediction: given tokens 0..N-1, predict tokens 1..N.
    """
    def __init__(self, data_path: str, block_size: int = 1024):
        self.block_size = block_size
        enc = tiktoken.get_encoding("gpt2")
        eos = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

        # Read and tokenize
        print(f"Loading {data_path}...")
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Split into documents (double newline = document boundary)
        documents = text.split('\n\n')
        
        # Tokenize each document and join with EOS
        all_tokens = []
        for doc in documents:
            doc = doc.strip()
            if not doc:
                continue
            tokens = enc.encode(doc)
            all_tokens.extend(tokens)
            all_tokens.append(eos)  # EOS between documents

        self.tokens = torch.tensor(all_tokens, dtype=torch.long)
        self.n_tokens = len(self.tokens)
        # Number of complete windows we can extract
        self.n_samples = (self.n_tokens - 1) // block_size

        print(f"  Total tokens: {self.n_tokens:,}")
        print(f"  Block size: {block_size}")
        print(f"  Samples: {self.n_samples:,}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.block_size
        end = start + self.block_size + 1  # +1 because target is shifted by 1
        chunk = self.tokens[start:end]
        x = chunk[:-1]    # input:  tokens[0..block_size-1]
        y = chunk[1:]     # target: tokens[1..block_size]
        return x, y
```

### DataLoader Setup

```python
from torch.utils.data import DataLoader

train_dataset = BabyLMDataset("datasets/babylm_train.txt", block_size=1024)
val_dataset = BabyLMDataset("datasets/babylm_dev.txt", block_size=1024)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,          # 8 sequences of 1024 tokens = 8192 tokens per step
    shuffle=True,
    num_workers=2,         # parallel data loading
    pin_memory=True,       # faster CPU -> GPU transfer
    drop_last=True,        # drop incomplete final batch
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    drop_last=True,
)

print(f"Training batches per epoch: {len(train_loader)}")
print(f"Tokens per batch: {8 * 1024:,}")
```

**Memory check:** Each batch is 8 * 1024 * 2 bytes (int16 equivalent) = 16 KB. This is negligible. The GPU memory bottleneck is the model and optimizer, not the data.

---

## 2.5 Training Loop from Scratch

### The Complete Training Script

Create `training/train.py`:

```python
"""
training/train.py -- Complete training loop for Modern GPT-2 on BabyLM.

Usage:
    python training/train.py

Expected results on RTX 4060 Ti (16GB):
    - Throughput: ~10-15K tokens/sec (no compile), ~15-20K tok/s (with compile)
    - Starting loss: ~10.8
    - After 1 epoch: ~4.5-5.0
    - VRAM usage: ~3-4 GB
"""
import os
import sys
import time
import math
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.gpt2_modern import GPT2Modern, GPT2Config
from training.dataset import BabyLMDataset


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Model
config = GPT2Config()

# Training hyperparameters
BATCH_SIZE = 8
BLOCK_SIZE = 1024
LEARNING_RATE = 3e-4          # peak LR (after warmup)
WEIGHT_DECAY = 0.1            # AdamW weight decay
WARMUP_STEPS = 200            # linear warmup from 0 to LR
MAX_STEPS = 10_000            # total training steps (~1 epoch of BabyLM 100M)
GRAD_CLIP = 1.0               # max gradient norm
EVAL_INTERVAL = 250           # evaluate every N steps
CHECKPOINT_DIR = "checkpoints/gpt2_modern"
USE_COMPILE = True            # torch.compile (Section 2.6)

# Mixed precision
USE_AMP = True                # fp16 automatic mixed precision


# ---------------------------------------------------------------------------
# Learning rate schedule: linear warmup + cosine decay
# ---------------------------------------------------------------------------
def get_lr(step: int) -> float:
    """
    Learning rate schedule:
    1. Linear warmup from 0 to LEARNING_RATE over WARMUP_STEPS
    2. Cosine decay from LEARNING_RATE to LEARNING_RATE/10 over remaining steps
    
    Why this schedule?
    - Warmup prevents early instability (large gradients on random weights)
    - Cosine decay smoothly reduces LR, avoiding the sudden drops of step decay
    - Final LR = peak/10 (not 0) keeps the model learning until the end
    """
    if step < WARMUP_STEPS:
        # Linear warmup: 0 -> LEARNING_RATE
        return LEARNING_RATE * (step / WARMUP_STEPS)
    
    # Cosine decay: LEARNING_RATE -> LEARNING_RATE / 10
    decay_steps = MAX_STEPS - WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / decay_steps
    min_lr = LEARNING_RATE / 10
    return min_lr + 0.5 * (LEARNING_RATE - min_lr) * (1 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, val_loader, device, max_batches=50):
    """
    Compute average validation loss over max_batches.
    
    We limit batches because full evaluation is slow and we only need
    a rough estimate during training.
    """
    model.eval()
    total_loss = 0.0
    count = 0
    for i, (x, y) in enumerate(val_loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast('cuda', dtype=torch.float16, enabled=USE_AMP):
            loss = model(x, y)
        total_loss += loss.item()
        count += 1
    model.train()
    return total_loss / max(count, 1)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def train():
    device = torch.device('cuda')
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ----- Data -----
    print("=== Loading Data ===")
    train_dataset = BabyLMDataset("datasets/babylm_train.txt", block_size=BLOCK_SIZE)
    val_dataset = BabyLMDataset("datasets/babylm_dev.txt", block_size=BLOCK_SIZE)
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True, drop_last=True,
    )

    # ----- Model -----
    print("=== Building Model ===")
    model = GPT2Modern(config).to(device)
    total_params, _ = model.count_parameters()
    print(f"Parameters: {total_params / 1e6:.1f}M")

    # ----- torch.compile (Section 2.6) -----
    if USE_COMPILE:
        print("=== Compiling Model (this takes 30-60 seconds the first time) ===")
        model = torch.compile(model)

    # ----- Optimizer -----
    # Separate parameters: weight-decay for weights, no decay for norms/embeddings
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.dim() >= 2:
                decay_params.append(param)    # weight matrices
            else:
                no_decay_params.append(param) # norms, biases, embeddings
    
    optimizer = torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': WEIGHT_DECAY},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ], lr=LEARNING_RATE, betas=(0.9, 0.95), fused=True)

    # ----- Mixed Precision -----
    # GradScaler prevents fp16 underflow: scales loss up before backward,
    # then scales gradients down before optimizer step.
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

    # ----- Training -----
    print("=== Training ===")
    log_path = os.path.join(CHECKPOINT_DIR, "train_log.jsonl")
    log_file = open(log_path, 'w')

    step = 0
    epoch = 0
    best_val_loss = float('inf')
    tokens_processed = 0
    t_start = time.perf_counter()
    
    model.train()

    while step < MAX_STEPS:
        epoch += 1
        for batch_x, batch_y in train_loader:
            if step >= MAX_STEPS:
                break

            # Move data to GPU
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            tokens_in_batch = batch_x.numel()

            # --- Forward pass ---
            # autocast: automatically use fp16 for matmuls and fp32 for norms/losses
            with torch.amp.autocast('cuda', dtype=torch.float16, enabled=USE_AMP):
                loss = model(batch_x, batch_y)

            # --- Backward pass ---
            # Scale loss to prevent fp16 gradient underflow
            scaler.scale(loss).backward()

            # --- Gradient clipping ---
            # Unscale gradients before clipping (scaler needs to undo its scaling)
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            # --- Optimizer step ---
            # Update learning rate
            lr = get_lr(step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Step (scaler checks for inf/nan gradients and skips if found)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)  # free gradient memory

            # --- Logging ---
            step += 1
            tokens_processed += tokens_in_batch
            elapsed = time.perf_counter() - t_start
            tok_per_sec = tokens_processed / elapsed

            if step % 10 == 0:
                mem_gb = torch.cuda.max_memory_allocated() / 1e9
                log_entry = {
                    'step': step,
                    'loss': round(loss.item(), 4),
                    'lr': round(lr, 6),
                    'grad_norm': round(grad_norm.item(), 4),
                    'tok_per_sec': round(tok_per_sec),
                    'mem_gb': round(mem_gb, 2),
                    'epoch': epoch,
                }
                log_file.write(json.dumps(log_entry) + '\n')
                log_file.flush()
                print(
                    f"step {step:5d} | loss {loss.item():.4f} | "
                    f"lr {lr:.2e} | grad_norm {grad_norm.item():.2f} | "
                    f"{tok_per_sec:.0f} tok/s | {mem_gb:.1f} GB"
                )

            # --- Evaluation ---
            if step % EVAL_INTERVAL == 0:
                val_loss = evaluate(model, val_loader, device)
                print(f"  >>> val_loss: {val_loss:.4f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    # Save best checkpoint
                    ckpt_path = os.path.join(CHECKPOINT_DIR, "best.pt")
                    torch.save({
                        'step': step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'config': config,
                        'val_loss': val_loss,
                    }, ckpt_path)
                    print(f"  >>> Saved best checkpoint (val_loss={val_loss:.4f})")

    # Save final checkpoint
    ckpt_path = os.path.join(CHECKPOINT_DIR, "final.pt")
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
    }, ckpt_path)
    print(f"\nTraining complete. Final checkpoint saved to {ckpt_path}")
    print(f"Total tokens: {tokens_processed:,}")
    print(f"Total time: {time.perf_counter() - t_start:.0f}s")
    print(f"Average throughput: {tokens_processed / (time.perf_counter() - t_start):.0f} tok/s")

    log_file.close()


if __name__ == "__main__":
    train()
```

### Understanding Each Component

**Forward pass and loss.** The model takes `(input_ids, targets)` and returns a scalar loss. The loss is cross-entropy: for each position, how wrong was the model's probability distribution vs the actual next token? Lower is better. `ln(50257) = 10.825` is the theoretical maximum (random guessing). A well-trained 124M model on BabyLM should reach ~4.5.

**Backward pass.** `loss.backward()` computes gradients for every parameter. PyTorch's autograd walks the computation graph in reverse, applying the chain rule. This is where most computation happens -- backward is roughly 2x the cost of forward.

**Gradient clipping.** Neural networks sometimes produce enormous gradients (especially early in training or on unusual data). `clip_grad_norm_` scales all gradients down if their total L2 norm exceeds `GRAD_CLIP=1.0`. This prevents catastrophic parameter updates.

**AdamW optimizer.** Adam with decoupled weight decay. It maintains two running averages per parameter:
- **First moment (momentum):** exponential moving average of gradients. Smooths out noise.
- **Second moment (variance):** exponential moving average of squared gradients. Adapts learning rate per-parameter (large-gradient params get smaller updates).

Weight decay (0.1) gently pushes weights toward zero, preventing overfitting. We only apply it to weight matrices, not to normalization parameters or biases.

**Mixed precision (fp16 + GradScaler).** Most operations run in fp16 (half the memory, 2x faster on Tensor Cores). But some operations need fp32 precision:
- Loss computation (small numbers can vanish in fp16)
- Normalization (division by small RMS values)
- Gradient accumulation in the optimizer

`torch.amp.autocast` handles this automatically. `GradScaler` prevents fp16 gradients from underflowing to zero by temporarily scaling the loss up before backward, then scaling gradients back down before the optimizer step.

**Learning rate schedule.** We use linear warmup (200 steps) followed by cosine decay. Warmup prevents the optimizer from making huge updates on random-initialization gradients. Cosine decay is smooth and widely adopted.

---

## 2.6 torch.compile

### What torch.compile Does

`torch.compile(model)` is PyTorch's graph compiler (Inductor backend). When you call the compiled model for the first time, PyTorch:

1. **Traces** the forward pass to build a computation graph
2. **Fuses** consecutive element-wise operations into single CUDA kernels
3. **Generates** optimized Triton (or CUDA) kernel code
4. **Caches** the compiled kernels for subsequent calls

For example, RMSNorm requires: square -> mean -> add eps -> sqrt -> divide -> multiply. Without compile, that is 5 separate CUDA kernel launches. With compile, it becomes 1 fused kernel.

### How to Apply It

```python
model = GPT2Modern(config).to(device)
model = torch.compile(model)  # that is it
```

The first forward pass triggers compilation and takes 30-60 seconds. Every subsequent call uses the cached compiled code and is faster.

### Expected Speedup

On RTX 4060 Ti with our 124M model:

| Configuration | Throughput | Speedup |
|--------------|-----------|---------|
| No compile | ~10-12K tok/s | 1.0x |
| torch.compile | ~15-20K tok/s | 1.3-1.7x |

The speedup comes from:
- **Kernel fusion** -- fewer kernel launches, less overhead
- **Memory access optimization** -- fused kernels read/write intermediate values in registers instead of VRAM
- **Operator-level optimization** -- Inductor sometimes generates better code than handwritten PyTorch ops

### Common Pitfalls

**Graph breaks.** If your model has Python control flow that depends on tensor values (e.g., `if x.sum() > 0`), torch.compile cannot trace through it. It "breaks" the graph and falls back to eager mode for that section. You get partial compilation with reduced benefit.

Check for graph breaks:
```python
# Set this before compile to see warnings
import torch._dynamo
torch._dynamo.config.verbose = True
model = torch.compile(model)
```

**Dynamic shapes.** If your batch size or sequence length changes between calls, torch.compile recompiles for each new shape. Keep shapes fixed during training (use `drop_last=True` in DataLoader).

**First-call latency.** The first forward pass is slow (compilation). Do not benchmark the first iteration. Run a few warmup steps first.

**What NOT to compile.** Never compile the optimizer. Only compile the model. Compiling the optimizer can cause massive memory bloat (the optimizer maintains fp32 copies of all parameters, and compiling it tries to trace through all that state).

```python
# CORRECT
model = torch.compile(model)

# WRONG -- do not do this
# optimizer = torch.compile(optimizer)  # memory explosion
```

---

## 2.7 First Training Run

### Running the Training

```bash
# From project root
python training/train.py
```

### What to Expect

**First 30-60 seconds:** No output. torch.compile is compiling the model. Be patient.

**Then, output every 10 steps:**
```
step    10 | loss 10.7832 | lr 1.50e-05 | grad_norm 4.21 | 12543 tok/s | 3.2 GB
step    20 | loss 10.2145 | lr 3.00e-05 | grad_norm 3.87 | 14201 tok/s | 3.2 GB
step    30 | loss  9.6521 | lr 4.50e-05 | grad_norm 3.45 | 15122 tok/s | 3.2 GB
...
step   200 | loss  7.1234 | lr 3.00e-04 | grad_norm 1.82 | 16543 tok/s | 3.2 GB  <- warmup complete
...
step  1000 | loss  5.4321 | lr 2.85e-04 | grad_norm 0.95 | 16800 tok/s | 3.2 GB
...
step  5000 | loss  4.8765 | lr 1.90e-04 | grad_norm 0.72 | 16900 tok/s | 3.2 GB
...
step 10000 | loss  4.5123 | lr 3.00e-05 | grad_norm 0.61 | 16950 tok/s | 3.2 GB
```

### How to Read the Logs

- **loss:** Should decrease monotonically (with noise). Start ~10.8, reach ~4.5-5.0.
- **lr:** Ramps up during warmup (steps 1-200), then decays via cosine.
- **grad_norm:** Should decrease over training. If it spikes to >10, something is wrong. If it is always exactly 1.0, clipping is active on every step (LR might be too high).
- **tok/s:** Tokens processed per second. Should stabilize after warmup. Target: >10K without compile, >15K with compile.
- **mem_gb:** Peak GPU memory. Should be 3-4 GB for 124M model. If it grows over time, you have a memory leak.

### Sanity Checks

1. **Initial loss ~10.8.** If it is much higher or lower, the model or loss function is wrong.
2. **Loss decreasing.** After 100 steps, loss should be below 9.0. After 1000 steps, below 6.0.
3. **No NaN/inf.** If loss becomes NaN, check: is GradScaler enabled? Is gradient clipping active? Is RMSNorm computing in fp32?
4. **Throughput stable.** If tok/s keeps dropping, you may be running out of memory (swapping to CPU).
5. **Validation loss tracks training loss.** If val_loss stops improving while train_loss keeps dropping, you are overfitting (not a problem on BabyLM with 124M params -- the model is too small to memorize it).

### Reading the Log File

```python
"""read_logs.py -- Plot training curves from the JSON log."""
import json
import matplotlib.pyplot as plt

steps, losses, lrs = [], [], []
with open("checkpoints/gpt2_modern/train_log.jsonl") as f:
    for line in f:
        entry = json.loads(line)
        steps.append(entry['step'])
        losses.append(entry['loss'])
        lrs.append(entry['lr'])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(steps, losses)
ax1.set_xlabel("Step")
ax1.set_ylabel("Loss")
ax1.set_title("Training Loss")
ax1.set_ylim(0, 12)

ax2.plot(steps, lrs)
ax2.set_xlabel("Step")
ax2.set_ylabel("Learning Rate")
ax2.set_title("LR Schedule")

plt.tight_layout()
plt.savefig("checkpoints/gpt2_modern/training_curves.png", dpi=150)
plt.show()
```

### Quick Generation Test

After training, verify the model produces coherent text:

```python
"""generate.py -- Generate text from a trained checkpoint."""
import torch
import tiktoken
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.gpt2_modern import GPT2Modern, GPT2Config

device = torch.device('cuda')
enc = tiktoken.get_encoding("gpt2")

# Load checkpoint
ckpt = torch.load("checkpoints/gpt2_modern/best.pt", map_location=device)
model = GPT2Modern(ckpt['config']).to(device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# Generate
prompt = "Once upon a time"
tokens = enc.encode(prompt)
input_ids = torch.tensor([tokens], device=device)

with torch.no_grad():
    for _ in range(100):
        logits = model(input_ids)  # (1, vocab_size)
        # Sample from top-k
        top_k = 40
        values, indices = torch.topk(logits, top_k)
        probs = torch.softmax(values, dim=-1)
        next_token = indices[0, torch.multinomial(probs[0], 1)]
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
        # Truncate to max_seq_len
        if input_ids.shape[1] > 1024:
            input_ids = input_ids[:, -1024:]

generated = enc.decode(input_ids[0].tolist())
print(generated)
```

After 10K steps on BabyLM, expect semi-coherent text. It will not be GPT-4, but it should produce grammatical English with some topical consistency. If it outputs pure garbage, the model did not train correctly.

---

## Checkpoint

Before moving to Part 03, verify:
- [ ] Model builds and reports ~124M parameters
- [ ] Initial loss is ~10.8 (random weights)
- [ ] Training runs without NaN or crashes
- [ ] Loss decreases: <9.0 at step 100, <6.0 at step 1000
- [ ] torch.compile works (>1.3x speedup over no-compile)
- [ ] Throughput >10K tok/s (no compile) or >15K tok/s (with compile)
- [ ] VRAM usage is 3-4 GB
- [ ] Checkpoint saved to `checkpoints/gpt2_modern/`
- [ ] Generated text is semi-coherent English

---

**Previous: [Part 01 -- Environment & Hardware](01_environment_and_hardware.md)**
**Next: [Part 03 -- Profiling: Finding What to Optimize](03_profiling.md)**
