# Part 13: Inference & Deployment

## Goal
Build a fast inference engine with KV-cache, implement multiple decoding strategies, quantize to INT8 for faster generation, and deploy behind an HTTP API that streams tokens to clients.

## Why This Matters
Training is a one-time cost. Inference happens millions of times. A model that generates at 50 tok/s is useless as an interactive assistant; at 300+ tok/s it feels instant. This part closes the gap.

## Prerequisites
- Part 02: You understand the model forward pass.
- Part 04-07: You understand CUDA kernels and torch.compile.
- Part 12: You have a trained SFT model.

---

## 13.1 The Inference Bottleneck

Autoregressive inference has two distinct phases with fundamentally different bottlenecks.

### Prefill Phase (Prompt Processing)

The model processes the entire prompt in parallel. This is compute-bound -- the GPU is doing large matrix multiplications on the full sequence.

```
Prompt: "What is Python?" (5 tokens)
Prefill: process all 5 tokens in one forward pass
Output:  KV-cache populated, first output logit ready
```

Prefill benefits from batch size and Tensor Cores. It is essentially the same workload as training (minus the backward pass).

### Decode Phase (Token Generation)

The model generates one token at a time. Each step reads the entire model's weights to produce a single output token. This is memory-bandwidth-bound -- the GPU spends most of its time reading weights from VRAM.

```
Step 1: Read all weights (700 MB for 350M fp16) -> produce 1 token
Step 2: Read all weights again -> produce 1 token
Step 3: Read all weights again -> produce 1 token
...
```

### The Decode Floor

The minimum time per token is determined by how fast you can read the model weights:

```
Time per token >= model_size_bytes / memory_bandwidth
```

For a 350M parameter model in FP16 on an RTX 4060 Ti:
```
Model size: 350M * 2 bytes = 700 MB
Bandwidth:  288 GB/s
Floor:      700 MB / 288 GB/s = 2.43 ms per token
Max tok/s:  1 / 0.00243 = ~412 tok/s
```

In practice you will achieve 70-85% of this theoretical maximum due to overhead (kernel launch, softmax, sampling). Expect 280-350 tok/s for a 350M model.

### Profiling Inference

```python
"""benchmark_inference.py -- Measure prefill and decode speed."""
import torch
import time


def benchmark_inference(model, vocab_size=50257, seq_len=512, n_new=128, warmup=5):
    device = next(model.parameters()).device
    model.eval()

    # Random prompt
    input_ids = torch.randint(0, vocab_size, (1, seq_len), device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            model(input_ids)
    torch.cuda.synchronize()

    # Prefill benchmark
    start = time.perf_counter()
    n_prefill = 20
    with torch.no_grad():
        for _ in range(n_prefill):
            model(input_ids)
    torch.cuda.synchronize()
    prefill_time = (time.perf_counter() - start) / n_prefill
    prefill_tps = seq_len / prefill_time

    print(f"Prefill: {prefill_time*1000:.1f} ms for {seq_len} tokens = {prefill_tps:.0f} tok/s")

    # Decode benchmark (simple, without KV-cache)
    start = time.perf_counter()
    with torch.no_grad():
        for i in range(n_new):
            logits = model(input_ids)
            next_token = logits[:, -1:, :].argmax(dim=-1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
    torch.cuda.synchronize()
    decode_time = time.perf_counter() - start
    decode_tps = n_new / decode_time

    print(f"Decode (no KV-cache): {decode_time*1000:.1f} ms for {n_new} tokens = {decode_tps:.0f} tok/s")
    print(f"  Note: KV-cache will make this 5-20x faster")

    return {"prefill_tps": prefill_tps, "decode_tps": decode_tps}
```

---

## 13.2 KV-Cache

### Why: Avoid Redundant Computation

Without KV-cache, generating token N requires recomputing attention for ALL previous N-1 tokens. This is O(N^2) total work for N tokens. With KV-cache, each step only computes attention for the new token against cached K, V from previous steps -- O(N) per step.

### How It Works

At each layer, cache the K and V tensors. When generating the next token, only compute Q for the new position and attend to the cached K, V.

```python
class KVCache:
    """Pre-allocated KV cache for efficient autoregressive generation."""

    def __init__(self, batch_size, max_seq_len, n_layers, n_kv_heads, head_dim, device, dtype):
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers

        # Pre-allocate full cache
        shape = (n_layers, 2, batch_size, n_kv_heads, max_seq_len, head_dim)
        self.cache = torch.zeros(shape, device=device, dtype=dtype)
        self.position = 0  # current fill level

    def update(self, layer_idx, k, v):
        """Store new K, V at current position.

        Args:
            layer_idx: which layer
            k: (B, n_kv_heads, 1, head_dim) -- single new position
            v: (B, n_kv_heads, 1, head_dim)
        """
        seq_len = k.size(2)
        self.cache[layer_idx, 0, :, :, self.position:self.position+seq_len, :] = k
        self.cache[layer_idx, 1, :, :, self.position:self.position+seq_len, :] = v

    def get(self, layer_idx):
        """Retrieve cached K, V up to current position.

        Returns:
            k: (B, n_kv_heads, position, head_dim)
            v: (B, n_kv_heads, position, head_dim)
        """
        k = self.cache[layer_idx, 0, :, :, :self.position, :]
        v = self.cache[layer_idx, 1, :, :, :self.position, :]
        return k, v

    def advance(self, n=1):
        """Move position forward after storing new tokens."""
        self.position += n
```

### Memory Cost

KV-cache memory per token per layer:
```
2 * n_kv_heads * head_dim * 2 bytes (fp16)
```

For a 350M model with 24 layers, 4 KV heads, head_dim=64:
```
Per token: 2 * 4 * 64 * 2 * 24 = 24,576 bytes = 24 KB
Per 1K context: 24 MB
Per 4K context: 96 MB
```

This is small relative to 16GB VRAM. KV-cache only becomes a problem for very large models or very long contexts.

### Attention with KV-Cache

Modify the attention forward pass to use the cache during generation:

```python
class AttentionWithCache(nn.Module):
    """Attention with KV-cache support for inference."""

    def __init__(self, dim, n_heads, n_kv_heads, qk_norm=True):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.n_rep = n_heads // n_kv_heads

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    def forward(self, x, freqs_cis, kv_cache=None, layer_idx=None):
        B, T, _ = x.shape

        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        # (apply_rotary_emb would go here)

        if kv_cache is not None:
            # Store new K, V in cache
            kv_cache.update(layer_idx, k, v)
            # Retrieve full K, V history
            k, v = kv_cache.get(layer_idx)

        # GQA expansion
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # Attention (no causal mask needed during decode -- we only attend to past)
        if T == 1 and kv_cache is not None:
            # Decode: single query against full cache
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
            # Prefill: full causal attention
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        return self.wo(y.transpose(1, 2).contiguous().view(B, T, -1))
```

---

## 13.3 Generation Loop

### Complete Generation Function

```python
@torch.no_grad()
def generate(
    model,
    input_ids,
    max_new_tokens=256,
    temperature=1.0,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.3,
    eos_token_id=None,
):
    """Generate tokens autoregressively with KV-cache.

    Args:
        model: Language model
        input_ids: (1, prompt_len) initial token IDs
        max_new_tokens: maximum tokens to generate
        temperature: sampling temperature (0 = greedy)
        top_k: keep only top K logits (0 = disabled)
        top_p: nucleus sampling threshold
        repetition_penalty: penalize repeated tokens
        eos_token_id: stop generation when this token is produced

    Returns:
        List of generated token IDs (not including prompt)
    """
    device = input_ids.device
    generated = []
    prompt_len = input_ids.size(1)

    # Track generated tokens for repetition penalty
    all_tokens = input_ids[0].tolist()

    for step in range(max_new_tokens):
        # Forward pass (full sequence each time -- add KV-cache for speedup)
        logits = model(input_ids)
        next_logits = logits[0, -1, :].float()  # (vocab_size,)

        # Apply repetition penalty
        if repetition_penalty != 1.0:
            for token_id in set(all_tokens[-128:]):  # last 128 tokens
                if next_logits[token_id] > 0:
                    next_logits[token_id] /= repetition_penalty
                else:
                    next_logits[token_id] *= repetition_penalty

        # Temperature scaling
        if temperature > 0 and temperature != 1.0:
            next_logits = next_logits / temperature

        # Top-K filtering
        if top_k > 0:
            top_k_values, _ = torch.topk(next_logits, top_k)
            threshold = top_k_values[-1]
            next_logits[next_logits < threshold] = float('-inf')

        # Top-P (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[sorted_mask] = float('-inf')

            # Scatter back to original order
            next_logits = torch.zeros_like(next_logits).scatter(0, sorted_indices, sorted_logits)

        # Sample or greedy
        if temperature == 0:
            next_token = next_logits.argmax().unsqueeze(0)
        else:
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        # Check for EOS
        token_id = next_token.item()
        if eos_token_id is not None and token_id == eos_token_id:
            break

        generated.append(token_id)
        all_tokens.append(token_id)

        # Append to sequence for next step
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

    return generated
```

### Decoding Strategies Explained

**Greedy (temperature=0):** Always pick the highest-probability token. Deterministic but repetitive. Good for factual questions.

**Temperature Sampling:** Divide logits by temperature before softmax. Lower temperature (0.3) is more conservative; higher (1.5) is more creative but risky.

```
Logits:      [5.0, 3.0, 2.0, 1.0]
temp=0.5:    [10, 6, 4, 2]    -> probs: [0.88, 0.09, 0.02, 0.01]  (peaked)
temp=1.0:    [5, 3, 2, 1]     -> probs: [0.64, 0.18, 0.09, 0.04]  (balanced)
temp=2.0:    [2.5, 1.5, 1, 0.5] -> probs: [0.40, 0.25, 0.18, 0.11] (flat)
```

**Top-K:** Zero out all logits except the top K. Prevents sampling from the long tail of unlikely tokens. K=50 is a reasonable default.

**Top-P (Nucleus):** Keep the smallest set of tokens whose cumulative probability exceeds P. Adapts dynamically -- for confident predictions, only a few tokens are kept; for uncertain predictions, many tokens remain. P=0.9 is standard.

**Repetition Penalty:** Reduce logits of recently generated tokens. This prevents loops like "the the the the". A penalty of 1.3 means repeated tokens have their logits divided by 1.3.

---

## 13.4 Quantization

### Why Quantize

Quantization reduces the precision of model weights from FP16 (2 bytes) to INT8 (1 byte) or INT4 (0.5 bytes). Since decode is memory-bandwidth-bound, smaller weights means faster generation.

| Precision | Bytes/param | 350M Model Size | Expected Speedup |
|-----------|-------------|-----------------|------------------|
| FP16 | 2 | 700 MB | 1x (baseline) |
| INT8 | 1 | 350 MB | ~1.7-2.0x |
| INT4 | 0.5 | 175 MB | ~2.5-3.0x |

### INT8 Quantization with PyTorch

PyTorch's `torch.ao` (Architecture Optimization) module provides quantization utilities:

```python
"""quantize_int8.py -- Post-training INT8 quantization."""
import torch
import torch.ao.quantization as quant


def quantize_model_int8(model):
    """Apply dynamic INT8 quantization to linear layers.

    Dynamic quantization: weights are quantized offline, activations are
    quantized on-the-fly during inference. No calibration data needed.
    """
    model.eval()

    # Quantize all linear layers to INT8
    quantized = torch.ao.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )

    # Report size reduction
    original_size = sum(p.numel() * p.element_size() for p in model.parameters())
    quantized_size = sum(
        p.numel() * p.element_size() for p in quantized.parameters()
    )
    # Note: quantized_size underestimates due to qint8 representation
    print(f"Original: {original_size / 1e6:.1f} MB")
    print(f"Quantized: ~{original_size / 2 / 1e6:.1f} MB (estimated INT8)")

    return quantized


def compare_quality(original, quantized, tokenizer, prompts):
    """Compare generation quality between original and quantized models."""
    for prompt in prompts:
        input_ids = torch.tensor([tokenizer.encode(prompt)], device="cuda")

        orig_out = generate(original, input_ids, max_new_tokens=50, temperature=0)
        quant_out = generate(quantized, input_ids, max_new_tokens=50, temperature=0)

        print(f"\nPrompt: {prompt}")
        print(f"Original:  {tokenizer.decode(orig_out)}")
        print(f"Quantized: {tokenizer.decode(quant_out)}")


# Usage
model = load_model("checkpoints/sft/sft_final.pt")
quantized = quantize_model_int8(model)
```

### When to Quantize

- **Always for deployment.** There is rarely a reason to serve FP16 if INT8 works.
- **Never during training.** Quantization errors accumulate through gradient updates and corrupt the model.
- **INT8 first, INT4 only if needed.** INT8 has minimal quality loss; INT4 can degrade noticeably for small models.
- **Test quality after quantization.** Run your evaluation suite (HellaSwag, ARC) on both versions. If the drop is < 1%, ship INT8.

### GPTQ/AWQ for INT4

For more aggressive quantization, use GPTQ or AWQ. These require a calibration dataset to determine which weights are most important:

```python
"""quantize_gptq.py -- INT4 quantization with GPTQ (requires auto-gptq package)."""
# pip install auto-gptq

from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,       # quantize in groups of 128 weights
    desc_act=False,       # faster inference
)

# You need a calibration dataset (128 examples is usually enough)
calibration_data = load_calibration_data("data/calibration.jsonl", n=128)

# Quantize
model_quantized = AutoGPTQForCausalLM.from_pretrained(
    model, quantize_config=quantize_config
)
model_quantized.quantize(calibration_data)
model_quantized.save_quantized("checkpoints/model_int4/")
```

---

## 13.5 Serving

### Simple HTTP API with FastAPI

```python
"""server.py -- Minimal inference server with streaming."""
import json
import torch
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import asyncio

app = FastAPI()

# Load model at startup
model = None
tokenizer = None


@app.on_event("startup")
async def load():
    global model, tokenizer
    model = load_model("checkpoints/sft/sft_final.pt")
    model.eval()
    tokenizer = ChatMLTokenizer()
    print("Model loaded.")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI-compatible chat endpoint with streaming."""
    body = await request.json()
    messages = body.get("messages", [])
    stream = body.get("stream", False)
    max_tokens = body.get("max_tokens", 256)
    temperature = body.get("temperature", 0.7)

    # Tokenize messages
    input_ids = tokenizer.encode_chatml(messages)
    input_ids.extend(tokenizer.start_assistant_turn())
    input_tensor = torch.tensor([input_ids], device="cuda")

    if stream:
        return StreamingResponse(
            stream_tokens(input_tensor, max_tokens, temperature),
            media_type="text/event-stream",
        )
    else:
        generated = generate(model, input_tensor,
                             max_new_tokens=max_tokens,
                             temperature=temperature,
                             eos_token_id=tokenizer.im_end_id)
        text = tokenizer.decode(generated)
        return {
            "choices": [{"message": {"role": "assistant", "content": text}}],
        }


async def stream_tokens(input_tensor, max_tokens, temperature):
    """Yield Server-Sent Events as tokens are generated."""
    generated_ids = []
    current_input = input_tensor

    for step in range(max_tokens):
        with torch.no_grad():
            logits = model(current_input)
            next_logits = logits[0, -1, :].float()

            if temperature > 0:
                next_logits = next_logits / temperature
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)

        token_id = next_token.item()
        if token_id == tokenizer.im_end_id:
            break

        generated_ids.append(token_id)
        text_chunk = tokenizer.decode([token_id])

        # SSE format
        data = json.dumps({"choices": [{"delta": {"content": text_chunk}}]})
        yield f"data: {data}\n\n"

        current_input = torch.cat([current_input, next_token.unsqueeze(0)], dim=1)

        # Yield control to event loop
        await asyncio.sleep(0)

    yield "data: [DONE]\n\n"
```

Run the server:
```bash
pip install fastapi uvicorn
uvicorn server:app --host 0.0.0.0 --port 8000
```

Test it:
```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is Python?"}
        ],
        "stream": false,
        "max_tokens": 128
    }'
```

### Batched Inference

Process multiple requests simultaneously to maximize GPU utilization:

```python
class InferenceBatcher:
    """Collect requests and process in batches for throughput."""

    def __init__(self, model, tokenizer, max_batch=8, max_wait_ms=50):
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch = max_batch
        self.max_wait_ms = max_wait_ms
        self.queue = asyncio.Queue()

    async def add_request(self, input_ids, max_tokens):
        """Add a request to the batch queue."""
        future = asyncio.Future()
        await self.queue.put((input_ids, max_tokens, future))
        return await future

    async def batch_loop(self):
        """Continuously collect and process batches."""
        while True:
            batch = []
            # Collect up to max_batch requests
            try:
                item = await asyncio.wait_for(
                    self.queue.get(), timeout=self.max_wait_ms / 1000
                )
                batch.append(item)
            except asyncio.TimeoutError:
                continue

            # Drain remaining items
            while len(batch) < self.max_batch:
                try:
                    item = self.queue.get_nowait()
                    batch.append(item)
                except asyncio.QueueEmpty:
                    break

            # Process batch
            if batch:
                await self._process_batch(batch)

    async def _process_batch(self, batch):
        """Process a batch of requests."""
        # Pad to same length
        max_len = max(ids.size(1) for ids, _, _ in batch)
        padded = torch.zeros(len(batch), max_len, dtype=torch.long, device="cuda")
        for i, (ids, _, _) in enumerate(batch):
            padded[i, :ids.size(1)] = ids[0]

        # Forward pass
        with torch.no_grad():
            logits = self.model(padded)

        # Return results
        for i, (ids, max_tokens, future) in enumerate(batch):
            result = logits[i, ids.size(1)-1, :]  # last position logits
            future.set_result(result)
```

---

## 13.6 Benchmarking Inference

### Key Metrics

| Metric | Definition | Target (350M) |
|--------|-----------|---------------|
| TTFT | Time to first token (prefill + 1 decode step) | < 50ms at 512 context |
| Decode tok/s | Tokens generated per second during decode | > 300 |
| Latency P50 | Median end-to-end response time | < 500ms for 100 tokens |
| Latency P99 | 99th percentile response time | < 2000ms for 100 tokens |
| Throughput | Total tokens/sec across concurrent requests | > 500 at batch=4 |

### Benchmarking Script

```python
"""benchmark_serve.py -- Benchmark inference latency and throughput."""
import torch
import time
import statistics


def benchmark_generation(model, tokenizer, n_runs=50):
    """Benchmark decode speed."""
    prompt = "Once upon a time"
    input_ids = torch.tensor([tokenizer.encode(prompt)], device="cuda")

    model.eval()
    latencies = []
    token_counts = []

    for _ in range(n_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()

        tokens = generate(
            model, input_ids.clone(),
            max_new_tokens=128,
            temperature=0.7,
            eos_token_id=tokenizer.eos_token_id,
        )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        latencies.append(elapsed)
        token_counts.append(len(tokens))

    # Report
    avg_tokens = statistics.mean(token_counts)
    avg_latency = statistics.mean(latencies)
    p50 = statistics.median(latencies)
    p99 = sorted(latencies)[int(0.99 * len(latencies))]
    avg_tps = sum(token_counts) / sum(latencies)

    print(f"Decode benchmark ({n_runs} runs):")
    print(f"  Avg tokens/response: {avg_tokens:.0f}")
    print(f"  Avg latency:  {avg_latency*1000:.1f} ms")
    print(f"  P50 latency:  {p50*1000:.1f} ms")
    print(f"  P99 latency:  {p99*1000:.1f} ms")
    print(f"  Avg tok/s:    {avg_tps:.0f}")

    return {
        "avg_tps": avg_tps,
        "p50_ms": p50 * 1000,
        "p99_ms": p99 * 1000,
    }


def benchmark_ttft(model, tokenizer, context_lengths=[128, 256, 512, 1024]):
    """Benchmark time-to-first-token at different context lengths."""
    model.eval()

    print("\nTTFT benchmark:")
    for ctx_len in context_lengths:
        input_ids = torch.randint(0, 50257, (1, ctx_len), device="cuda")

        # Warmup
        with torch.no_grad():
            model(input_ids)
        torch.cuda.synchronize()

        # Measure
        times = []
        for _ in range(20):
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                model(input_ids)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

        avg = statistics.mean(times) * 1000
        print(f"  context={ctx_len:>5}: TTFT = {avg:.1f} ms")
```

### Comparing Your Model to Baselines

Run the same benchmark on HuggingFace's `generate()` to see how your optimized inference compares:

```python
def benchmark_hf_baseline(model_name="gpt2"):
    """Benchmark HuggingFace generate() for comparison."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    hf_tokenizer = AutoTokenizer.from_pretrained(model_name)

    prompt = "Once upon a time"
    inputs = hf_tokenizer(prompt, return_tensors="pt").to("cuda")

    # Warmup
    hf_model.generate(**inputs, max_new_tokens=10, do_sample=False)

    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(20):
        hf_model.generate(**inputs, max_new_tokens=128, do_sample=True,
                          temperature=0.7, top_p=0.9)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    tps = 128 * 20 / elapsed
    print(f"HuggingFace generate(): {tps:.0f} tok/s")
```

---

## Exercises

### Exercise 1: Implement Generation with KV-Cache

Modify your model's attention to support KV-cache (see Section 13.2). Then implement the generation loop:

```python
# Your implementation should achieve:
# - 5-20x speedup over naive generation (no KV-cache)
# - > 300 tok/s for 350M model
# - Identical output to naive generation (verify with greedy decoding)
```

Verify by generating the same prompt with and without KV-cache using temperature=0. The outputs must be identical.

### Exercise 2: Quantize to INT8

```python
# 1. Load your SFT model
model = load_model("checkpoints/sft/sft_final.pt")

# 2. Run evaluation BEFORE quantization
eval_before = evaluate_hellaswag(model)

# 3. Quantize
model_int8 = quantize_model_int8(model)

# 4. Run evaluation AFTER quantization
eval_after = evaluate_hellaswag(model_int8)

# 5. Compare speed
speed_before = benchmark_generation(model)
speed_after = benchmark_generation(model_int8)

# Expected:
# - Quality drop < 1% on HellaSwag
# - Speed improvement 1.5-2.0x
```

### Exercise 3: Build a FastAPI Server

```bash
# 1. Implement server.py from Section 13.5
# 2. Start the server
uvicorn server:app --port 8000

# 3. Test with curl (non-streaming)
curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"messages": [{"role": "user", "content": "Hello!"}]}'

# 4. Test streaming
curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"messages": [{"role": "user", "content": "Tell me about Python."}], "stream": true}'

# 5. Verify: tokens stream character by character, response is coherent
```

---

## Checkpoint

Before moving to Part 14, verify:

- [ ] KV-cache implemented and generates identical output to naive decoding
- [ ] Generation speed > 300 tok/s for 350M model (> 150 tok/s without KV-cache optimization)
- [ ] INT8 quantization working with < 1% quality degradation
- [ ] FastAPI server responds to requests and streams tokens
- [ ] You understand the decode bottleneck: model_size / bandwidth = floor
- [ ] TTFT < 50ms at context=512

**Expected time:** 6 hours. The KV-cache implementation is the hardest part -- getting the position indexing right with RoPE requires careful attention to detail. If your KV-cache generation produces garbage, the first thing to check is that RoPE frequencies are applied to the correct positions (not always starting from 0).
