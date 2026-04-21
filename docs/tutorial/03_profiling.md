# Part 03: Profiling -- Finding What to Optimize

## Goal
Profile your GPT-2 training loop, identify the top performance bottlenecks, and build a prioritized optimization plan. By the end, you will know exactly which operations consume the most wall time and which ones are candidates for custom CUDA kernels.

## Why This Matters
The single most common mistake in performance optimization is guessing. Without profiling data, you will optimize the wrong thing, waste days writing a CUDA kernel for an op that accounts for 2% of runtime, and miss the op that accounts for 30%. Profiling gives you the map. Everything else is driving blind.

---

## 3.1 Why Profile Before Optimizing

### Amdahl's Law

Gene Amdahl formalized a simple truth in 1967: the maximum speedup you can achieve by optimizing one part of a system is limited by how much time that part consumes.

```
                    1
Speedup = ─────────────────────
           (1 - p) + p / s

Where:
  p = fraction of time spent in the optimized part
  s = speedup of that part
```

Concrete example. Suppose your training step takes 100ms:
- Matmuls: 65ms (65%)
- Softmax: 12ms (12%)
- RMSNorm: 8ms (8%)
- SwiGLU activation: 5ms (5%)
- Everything else: 10ms (10%)

If you write a perfect RMSNorm kernel that takes 0ms (impossible, but let's dream):
- New total: 92ms
- Speedup: 100/92 = 1.087x (8.7%)

If you instead optimize matmuls by 10% (possible with better tiling):
- New matmul time: 58.5ms
- New total: 93.5ms
- Speedup: 100/93.5 = 1.070x (7.0%)

The RMSNorm kernel (8% of time, fully eliminated) gives roughly the same benefit as a 10% improvement to matmuls (65% of time). This is Amdahl's law in action.

**The lesson:** Always profile first. Optimize the biggest bottlenecks. Never assume you know what is slow.

### The 80/20 Rule

In practice, 80% of your runtime comes from 20% of your operations. For transformer training, the breakdown typically looks like:

```
Matmuls (Q/K/V proj, attention, FFN)  [====================] 60-70%
Attention softmax + masking           [====]                  8-15%
Normalization (RMSNorm/LayerNorm)     [===]                   5-10%
Activation functions (SwiGLU/GELU)    [==]                    3-5%
Residual additions                    [=]                     1-2%
Memory ops (copy, reshape, transpose) [==]                    3-5%
Optimizer step                        [==]                    3-5%
Data loading                          [=]                     1-3%
```

You cannot beat cuBLAS at matmuls on NVIDIA hardware. Tensor Cores + years of tuning make it near-optimal. The opportunity lies in the remaining 30-40%: norms, activations, softmax, and especially fusing these small ops together to reduce memory traffic.

---

## 3.2 torch.profiler

PyTorch's built-in profiler is the fastest way to get a bottleneck map. It records CPU and CUDA events during execution and produces a sorted table of the slowest operations.

### Basic Usage

Create `scripts/profile_training.py`:

```python
"""
scripts/profile_training.py -- Profile the GPT-2 training loop.

Records 5 training steps under the profiler, then exports results
as a table and Chrome trace file.
"""
import os
import sys
import torch
from torch.profiler import profile, record_function, ProfilerActivity, schedule
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.gpt2_modern import GPT2Modern, GPT2Config
from training.dataset import BabyLMDataset


def main():
    device = torch.device('cuda')
    config = GPT2Config()

    # Build model
    model = GPT2Modern(config).to(device)
    model.train()
    # NOTE: Do NOT use torch.compile here -- it obscures the per-op breakdown.
    # We profile the un-compiled model to see individual operations clearly.

    # Minimal data -- we only need a few batches
    dataset = BabyLMDataset("datasets/babylm_train.txt", block_size=1024)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)
    data_iter = iter(loader)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
    scaler = torch.amp.GradScaler('cuda')

    # Warmup: run a few steps outside the profiler to stabilize CUDA caches
    print("Warming up (3 steps)...")
    for _ in range(3):
        x, y = next(data_iter)
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast('cuda', dtype=torch.float16):
            loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    # Profile: 5 training steps
    print("Profiling (5 steps)...")
    os.makedirs("benchmarks", exist_ok=True)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(
            wait=0,      # start immediately
            warmup=1,    # discard first step (profiler warmup)
            active=4,    # record 4 steps
            repeat=1,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("benchmarks/profiler_logs"),
        record_shapes=True,      # record tensor shapes
        profile_memory=True,     # record memory allocations
        with_stack=True,         # record Python stack traces
    ) as prof:
        for step in range(5):
            x, y = next(data_iter)
            x, y = x.to(device), y.to(device)

            # Annotate sections for readability
            with record_function("forward"):
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    loss = model(x, y)

            with record_function("backward"):
                scaler.scale(loss).backward()

            with record_function("optimizer"):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            prof.step()

    # Print summary tables
    print("\n" + "=" * 80)
    print("TOP 20 CUDA OPERATIONS (sorted by CUDA time)")
    print("=" * 80)
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=20,
        max_name_column_width=50,
    ))

    print("\n" + "=" * 80)
    print("TOP 20 OPERATIONS BY SELF CUDA TIME (excludes children)")
    print("=" * 80)
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total",
        row_limit=20,
        max_name_column_width=50,
    ))

    # Export Chrome trace (open in chrome://tracing)
    trace_path = "benchmarks/training_trace.json"
    prof.export_chrome_trace(trace_path)
    print(f"\nChrome trace exported to: {trace_path}")
    print("Open chrome://tracing in Chrome/Edge and load this file to visualize.")

    # Export stacks for flamegraph
    stacks_path = "benchmarks/profiler_stacks.txt"
    prof.export_stacks(stacks_path, "self_cuda_time_total")
    print(f"Stacks exported to: {stacks_path}")


if __name__ == "__main__":
    main()
```

### Reading the Output

The profiler prints two tables. Here is what a typical output looks like (numbers are illustrative for RTX 4060 Ti):

```
TOP 20 CUDA OPERATIONS (sorted by CUDA time)
-------------------------------------------------------------------------
Name                               Self CPU   Self CUDA   # Calls
-------------------------------------------------------------------------
aten::mm                           12.3ms     38.2ms      192
aten::_scaled_dot_product_flash    4.1ms      15.6ms      12
aten::addmm                        2.1ms      5.4ms       24
aten::_softmax                     0.8ms      3.2ms       12
aten::mul                          0.6ms      2.1ms       96
aten::add_                         0.4ms      1.8ms       48
aten::native_layer_norm            0.5ms      1.6ms       24
aten::silu_                        0.2ms      0.8ms       12
...
```

**Key columns:**
- **Self CPU:** Time the CPU spent dispatching this operation (not waiting for GPU)
- **Self CUDA:** Time the GPU spent executing this kernel
- **# Calls:** How many times this op was called across the profiled steps

**What to look for:**
1. `aten::mm` and `aten::addmm` are matmuls. These should dominate. Do NOT try to replace these with custom kernels -- cuBLAS is already optimal on NVIDIA.
2. `aten::_scaled_dot_product_flash` is the fused attention kernel. Also do not replace this.
3. Everything else (norms, activations, element-wise ops) is your optimization target.

### CPU Time vs CUDA Time

This distinction matters:

- **CPU time** includes Python overhead, kernel launch overhead, and any CPU computation. If CPU time >> CUDA time, your bottleneck is Python/launch overhead, not GPU computation.
- **CUDA time** is how long the GPU was actually computing. If CUDA time >> CPU time, the GPU is busy (good) but the kernel itself could be faster.
- **Gap between CPU and CUDA**: if there is idle time where neither CPU nor CUDA is active, you have a synchronization problem (the CPU is waiting for the GPU or vice versa).

### Grouped by Input Shape

```python
# See which shapes are most expensive
print(prof.key_averages(group_by_input_shape=True).table(
    sort_by="cuda_time_total",
    row_limit=30,
))
```

This tells you, for example, that `aten::mm` with shape `[8192, 768] x [768, 2048]` (FFN up-projection) takes longer than `[8192, 768] x [768, 768]` (attention projection). Useful when deciding which matmul sizes to optimize around.

---

## 3.3 NVIDIA Nsight Systems

While torch.profiler gives you operation-level timing, Nsight Systems gives you a **timeline view** showing exactly what the GPU is doing at every microsecond. This is essential for finding:
- Gaps between kernel launches (idle GPU)
- Memory copy stalls
- CPU-GPU synchronization points
- Kernel launch overhead

### Install

```bash
# Nsight Systems is included with CUDA Toolkit
# Check if installed:
nsys --version

# If not, download from NVIDIA:
# https://developer.nvidia.com/nsight-systems
# The free version is sufficient for everything in this tutorial.
```

### Profiling a Training Run

```bash
# Profile 5 training iterations
nsys profile \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    --output=benchmarks/nsys_training \
    --force-overwrite=true \
    python training/train_profile_nsys.py
```

Create a minimal script for Nsight profiling:

```python
"""training/train_profile_nsys.py -- Minimal script for nsys profiling."""
import os
import sys
import torch
import torch.cuda.nvtx as nvtx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.gpt2_modern import GPT2Modern, GPT2Config
from training.dataset import BabyLMDataset
from torch.utils.data import DataLoader


def main():
    device = torch.device('cuda')
    config = GPT2Config()
    model = GPT2Modern(config).to(device)
    model.train()

    dataset = BabyLMDataset("datasets/babylm_train.txt", block_size=1024)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)
    data_iter = iter(loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
    scaler = torch.amp.GradScaler('cuda')

    # Warmup (not profiled)
    for _ in range(3):
        x, y = next(data_iter)
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast('cuda', dtype=torch.float16):
            loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    # Profiled region -- use NVTX markers so they show in the timeline
    for step in range(5):
        x, y = next(data_iter)
        x, y = x.to(device), y.to(device)

        nvtx.range_push(f"step_{step}")

        nvtx.range_push("forward")
        with torch.amp.autocast('cuda', dtype=torch.float16):
            loss = model(x, y)
        nvtx.range_pop()

        nvtx.range_push("backward")
        scaler.scale(loss).backward()
        nvtx.range_pop()

        nvtx.range_push("optimizer")
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        nvtx.range_pop()

        nvtx.range_pop()  # step

    torch.cuda.synchronize()
    print("Profiling complete. Open the .nsys-rep file in Nsight Systems GUI.")


if __name__ == "__main__":
    main()
```

### Reading the Nsight Timeline

Open the `.nsys-rep` file in the Nsight Systems GUI:

```bash
nsys-ui benchmarks/nsys_training.nsys-rep
```

You will see a timeline with several rows:

```
CPU Thread  ─────[fwd]──────[bwd]──────[opt]────[fwd]──────[bwd]──────[opt]────
CUDA Stream ──▓▓▓▓▓▓▓▓▓▓──▓▓▓▓▓▓▓▓▓▓▓▓▓──▓▓──▓▓▓▓▓▓▓▓▓▓──▓▓▓▓▓▓▓▓▓▓▓▓▓──▓▓──
              ^kernels^     ^kernels^              ^tiny gaps = good^
```

**What to look for:**

1. **Dense kernel execution** (no gaps). This means the GPU is fully utilized. The CPU is launching kernels fast enough to keep the GPU busy.

2. **Gaps between kernels.** If you see white space between CUDA kernels, the GPU is idle. Common causes:
   - Python overhead between ops (torch.compile fixes this)
   - CPU-GPU synchronization (e.g., `.item()` calls that force GPU to finish)
   - Data loading stalls (CPU cannot prepare the next batch fast enough)

3. **Kernel durations.** Hover over individual kernels to see their names and durations. The longest kernels are your optimization targets.

4. **Memory transfers.** Look for `cudaMemcpy` events. In a well-configured training loop, there should be almost none (data stays on GPU). If you see frequent transfers, something is copying tensors between CPU and GPU unnecessarily.

### Finding Kernel Launch Overhead

Each CUDA kernel launch has overhead: ~5-10 microseconds to set up and dispatch. For a model with hundreds of tiny operations, this adds up.

Count kernel launches per training step in the timeline:
- **Un-compiled model:** ~500-1000 kernel launches per step
- **torch.compile:** ~50-150 kernel launches (fusion reduces the count dramatically)

If you see 500+ launches and each kernel takes <10us, launch overhead is your bottleneck. torch.compile or manual kernel fusion is the fix.

---

## 3.4 Manual Timing

Sometimes you need precise timing for individual operations without the overhead of a full profiler. CUDA events are the right tool.

### torch.cuda.Event Timing

```python
"""scripts/time_ops.py -- Measure individual operation timings."""
import torch
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.gpt2_modern import RMSNorm, SwiGLUFFN, GroupedQueryAttention, GPT2Config


def time_op(fn, warmup=10, repeats=100, label="op"):
    """
    Time a CUDA operation using CUDA events.
    
    CUDA events are inserted into the GPU command stream. The elapsed time
    between start and end events is measured on the GPU clock, not the CPU
    clock. This gives accurate kernel timing without CPU synchronization
    artifacts.
    """
    # Warmup: fill caches, trigger JIT compilation
    for _ in range(warmup):
        fn()
    
    torch.cuda.synchronize()
    
    # Time with CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(repeats):
        fn()
    end_event.record()
    
    torch.cuda.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event) / repeats
    
    print(f"{label:40s} {elapsed_ms:.4f} ms")
    return elapsed_ms


def main():
    device = torch.device('cuda')
    config = GPT2Config()
    B, T, D = 8, 1024, config.d_model  # batch=8, seq=1024, dim=768

    # Create test inputs
    x = torch.randn(B, T, D, device=device, dtype=torch.float16)

    print(f"Input shape: ({B}, {T}, {D}) = {B*T*D:,} elements")
    print(f"Input size: {B*T*D*2/1e6:.1f} MB (fp16)")
    print()
    print(f"{'Operation':<40s} {'Time (ms)':>10s}")
    print("-" * 55)

    # --- RMSNorm ---
    norm = RMSNorm(D).to(device).half()
    time_op(lambda: norm(x), label="RMSNorm (768)")

    # --- RMSNorm as separate ops (what PyTorch does internally) ---
    weight = torch.ones(D, device=device, dtype=torch.float16)
    eps = 1e-6
    def rmsnorm_manual():
        x_f = x.float()
        rms = torch.sqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + eps)
        return ((x_f / rms) * weight).half()
    time_op(rmsnorm_manual, label="RMSNorm manual (unfused)")

    # --- SwiGLU FFN ---
    ffn = SwiGLUFFN(config).to(device).half()
    time_op(lambda: ffn(x), label="SwiGLU FFN (768->2048->768)")

    # --- Attention ---
    freqs_cis = torch.randn(T, D // config.n_heads // 2, dtype=torch.cfloat, device=device)
    attn = GroupedQueryAttention(config).to(device).half()
    time_op(lambda: attn(x, freqs_cis), label="GQA Attention (12h, 4kv)")

    # --- Individual matmul sizes ---
    # Q/K/V projection: (B*T, D) @ (D, D) = (B*T, D)
    a = torch.randn(B * T, D, device=device, dtype=torch.float16)
    b = torch.randn(D, D, device=device, dtype=torch.float16)
    time_op(lambda: torch.mm(a, b), label="Matmul: (8192, 768) x (768, 768)")

    # FFN up: (B*T, D) @ (D, FFN) = (B*T, FFN)
    c = torch.randn(D, config.ffn_dim, device=device, dtype=torch.float16)
    time_op(lambda: torch.mm(a, c), label="Matmul: (8192, 768) x (768, 2048)")

    # --- Element-wise ops ---
    time_op(lambda: F.silu(x), label="SiLU activation")
    time_op(lambda: x + x, label="Element-wise add")
    time_op(lambda: x * x, label="Element-wise multiply")

    # --- Softmax ---
    attn_scores = torch.randn(B, config.n_heads, T, T, device=device, dtype=torch.float16)
    time_op(lambda: F.softmax(attn_scores, dim=-1), label="Softmax (8, 12, 1024, 1024)")

    # --- Cross-entropy loss ---
    logits = torch.randn(B * T, config.vocab_size, device=device, dtype=torch.float16)
    targets = torch.randint(0, config.vocab_size, (B * T,), device=device)
    time_op(lambda: F.cross_entropy(logits, targets), label="Cross-entropy (8192, 50257)")

    # --- Full forward pass ---
    from models.gpt2_modern import GPT2Modern
    model = GPT2Modern(config).to(device).half()
    input_ids = torch.randint(0, config.vocab_size, (B, T), device=device)
    targets_full = torch.randint(0, config.vocab_size, (B, T), device=device)
    time_op(lambda: model(input_ids, targets_full), label="FULL FORWARD PASS")

    # --- Full forward + backward ---
    def fwd_bwd():
        loss = model(input_ids, targets_full)
        loss.backward()
        model.zero_grad(set_to_none=True)
    time_op(fwd_bwd, label="FULL FORWARD + BACKWARD")


if __name__ == "__main__":
    main()
```

### Running the Timer

```bash
python scripts/time_ops.py
```

Expected output (approximate, RTX 4060 Ti):
```
Input shape: (8, 1024, 768) = 6,291,456 elements
Input size: 12.0 MB (fp16)

Operation                                Time (ms)
-------------------------------------------------------
RMSNorm (768)                              0.0850
RMSNorm manual (unfused)                   0.1900
SwiGLU FFN (768->2048->768)                0.9200
GQA Attention (12h, 4kv)                   1.8500
Matmul: (8192, 768) x (768, 768)           0.1800
Matmul: (8192, 768) x (768, 2048)          0.3200
SiLU activation                            0.0250
Element-wise add                           0.0200
Element-wise multiply                      0.0200
Softmax (8, 12, 1024, 1024)                0.4500
Cross-entropy (8192, 50257)                0.3500
FULL FORWARD PASS                          8.5000
FULL FORWARD + BACKWARD                   25.5000
```

**Key observations:**
- Matmuls dominate (0.18-0.32ms each, and there are dozens per forward pass)
- RMSNorm's manual unfused version is ~2x slower than the module (PyTorch already partially fuses it)
- A custom CUDA kernel for RMSNorm can match or beat even the module's timing
- Element-wise ops are individually tiny but add up across 12 layers
- Forward+backward is ~3x forward (backward is roughly 2x forward cost)

### Building a Simple Op-Level Profiler

For ongoing optimization work, wrap your model to automatically time each layer:

```python
"""scripts/layer_profiler.py -- Time each layer in the model."""
import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.gpt2_modern import GPT2Modern, GPT2Config


class LayerTimer:
    """Hook-based layer timer. Attaches to every named module."""
    def __init__(self, model):
        self.timings = {}
        self.start_events = {}
        self.end_events = {}
        
        for name, module in model.named_modules():
            if name == '':  # skip root
                continue
            # Only time leaf modules or specific composites
            if len(list(module.children())) == 0 or name.startswith('layers.'):
                self.timings[name] = []
                module.register_forward_pre_hook(self._make_pre_hook(name))
                module.register_forward_hook(self._make_post_hook(name))
    
    def _make_pre_hook(self, name):
        def hook(module, input):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self.start_events[name] = event
        return hook
    
    def _make_post_hook(self, name):
        def hook(module, input, output):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self.end_events[name] = event
        return hook
    
    def synchronize_and_collect(self):
        torch.cuda.synchronize()
        for name in self.start_events:
            if name in self.end_events:
                elapsed = self.start_events[name].elapsed_time(self.end_events[name])
                self.timings[name].append(elapsed)
    
    def report(self, top_k=20):
        print(f"\n{'Layer':<50s} {'Avg (ms)':>10s} {'% Total':>10s}")
        print("-" * 75)
        
        # Compute averages (skip first measurement -- warmup)
        averages = {}
        for name, times in self.timings.items():
            if len(times) > 1:
                averages[name] = sum(times[1:]) / len(times[1:])
        
        total = sum(averages.values())
        
        # Sort by time, print top K
        sorted_layers = sorted(averages.items(), key=lambda x: x[1], reverse=True)
        for name, avg_ms in sorted_layers[:top_k]:
            pct = avg_ms / total * 100 if total > 0 else 0
            print(f"{name:<50s} {avg_ms:>10.4f} {pct:>9.1f}%")
        
        print("-" * 75)
        print(f"{'TOTAL':<50s} {total:>10.4f}")


def main():
    device = torch.device('cuda')
    config = GPT2Config()
    model = GPT2Modern(config).to(device).half()
    timer = LayerTimer(model)

    B, T = 8, 1024
    input_ids = torch.randint(0, config.vocab_size, (B, T), device=device)
    targets = torch.randint(0, config.vocab_size, (B, T), device=device)

    # Run several iterations
    for i in range(12):
        with torch.amp.autocast('cuda', dtype=torch.float16):
            loss = model(input_ids, targets)
        timer.synchronize_and_collect()
        if i == 0:
            print(f"Warmup step loss: {loss.item():.4f}")

    timer.report(top_k=30)


if __name__ == "__main__":
    main()
```

This gives you a clear ranking of which layers consume the most time, without needing external tools.

---

## 3.5 Interpreting Results for GPT-2 124M

### Expected Breakdown

After running the profilers above, you should see a breakdown roughly like this:

```
Category                    % of Forward Time    Optimization Potential
─────────────────────────────────────────────────────────────────────────
Matmuls (Q,K,V,O,FFN)           60-70%           LOW  (cuBLAS is optimal)
Attention (softmax+mask)         10-15%           LOW  (SDPA is already fused)
RMSNorm (24 calls)               5-10%           HIGH (memory-bound, fusable)
SwiGLU activation                 3-5%            MED  (fusable with FFN matmul)
Residual adds                     1-2%            MED  (fusable with norm)
Cross-entropy loss                2-3%            MED  (large vocab, chunking helps)
Other overhead                    3-5%            MED  (compile reduces this)
```

### Which Ops are Memory-Bound vs Compute-Bound?

Recall from Part 01: the ridge point for RTX 4060 Ti is ~611 FLOPS/byte. Any operation below this ratio is memory-bound (limited by how fast data can be read/written, not by how fast the GPU can compute).

```
Operation          Arithmetic Intensity    Bound          Notes
─────────────────────────────────────────────────────────────────
Large matmul       ~500-2000 FLOPS/byte    COMPUTE        Tensor Cores keep up
Small matmul       ~50-200 FLOPS/byte      MEMORY         Too small for Tensor Cores
Attention SDPA     ~100-500 FLOPS/byte     MIXED          Flash attention helps
RMSNorm            ~5 FLOPS/byte           MEMORY ***     Just read, square, mean, scale
SiLU               ~1 FLOP/byte            MEMORY ***     One read, one nonlinearity, one write
Element-wise add   ~0.33 FLOPS/byte        MEMORY ***     One read each, one write
Softmax            ~5 FLOPS/byte           MEMORY ***     Read, exp, sum, normalize
Cross-entropy      ~2 FLOPS/byte           MEMORY ***     Read logits, compute loss
```

Operations marked `***` are heavily memory-bound. These are your custom kernel targets. The strategy is:
1. **Reduce memory traffic** by fusing consecutive operations into a single kernel (one read, one write instead of multiple round-trips to VRAM)
2. **Use vectorized loads** (read 2 or 4 elements at once instead of 1)
3. **Keep intermediate values in registers** (not VRAM)

### Where Custom Kernels Help (and Where They Cannot)

**Custom kernels CAN help:**
- **RMSNorm:** PyTorch launches 3-5 separate kernels (square, mean, add, rsqrt, multiply). A fused kernel does it in 1 launch with 2 reads of the input (sum_sq pass + normalize pass). Expected speedup: 2-3x.
- **Fused RMSNorm + Residual Add:** Combine `x = x + Attention(RMSNorm(x))` into one kernel that reads `x` once, normalizes, and writes `x_normed` while also writing the residual. Saves one full read/write of `x`.
- **SwiGLU activation:** Fuse `silu(gate) * up` into one kernel. Saves writing and re-reading the intermediate.
- **Cross-entropy with chunking:** For large vocabularies, chunk the logits computation to avoid materializing the full `(batch*seq, vocab)` tensor.
- **Fused attention bias/mask:** If using custom attention patterns, fuse the mask application with softmax.

**Custom kernels CANNOT help (on NVIDIA):**
- **Matmuls.** cuBLAS has been optimized for decades. It uses Tensor Cores, tiling, pipelining, and hardware-specific tuning that you cannot replicate in a weekend. Do not write custom matmul kernels.
- **Attention.** PyTorch's `scaled_dot_product_attention` already uses FlashAttention-2 or memory-efficient attention. This is already near-optimal.
- **Optimizer step.** The fused AdamW in PyTorch is already a single kernel. The cost is proportional to parameter count and cannot be reduced without changing the algorithm.

---

## 3.6 Building Your Optimization Priority List

### The Template

After profiling, fill in this table. This becomes your roadmap for Parts 04-07.

```
Optimization Priority List -- GPT-2 124M on RTX 4060 Ti
════════════════════════════════════════════════════════════════════════════
Measured with: [torch.profiler / nsys / manual timing]
Date: [YYYY-MM-DD]
Batch: 8x1024, fp16, no compile

#  Operation         Time(ms)  % Total  Bound    Fusable?  Priority  Part
── ───────────────── ───────── ──────── ──────── ───────── ───────── ─────
1  Matmuls (all)     XX.XX     XX%      Compute  NO        SKIP      --
2  Attention SDPA    XX.XX     XX%      Mixed    NO        SKIP      --
3  RMSNorm (x24)     XX.XX     XX%      Memory   YES(+res) HIGH      04
4  SwiGLU (x12)      XX.XX     XX%      Memory   YES       HIGH      05
5  Softmax (x12)     XX.XX     XX%      Memory   YES(+mask)MED       05
6  Residual add(x24) XX.XX     XX%      Memory   YES(+norm)MED       05
7  Cross-entropy     XX.XX     XX%      Memory   YES(chunk)MED       05
8  Optimizer         XX.XX     XX%      Memory   NO        LOW       --
9  Data loading      XX.XX     XX%      CPU      N/A       LOW       --

TOTAL forward:       XX.XX ms
TOTAL fwd+bwd:       XX.XX ms
Tokens/sec:          XXXXX
```

### How to Fill the Priority Column

Score each operation on three axes:

1. **% of total time** -- Higher percentage = higher priority (Amdahl's law)
2. **Memory-bound?** -- Memory-bound ops benefit most from fusion. Compute-bound ops need algorithmic changes.
3. **Fusable with neighbors?** -- If an op can be merged with the next op in the graph, the combined speedup is greater than optimizing either alone.

Priority = High if (>5% of time) AND (memory-bound) AND (fusable).
Priority = Medium if two of three conditions are met.
Priority = Low or Skip otherwise.

### Save Your Results

```bash
mkdir -p benchmarks
# After running the profiler, save the summary
python scripts/profile_training.py > benchmarks/profile_summary.txt 2>&1
python scripts/time_ops.py > benchmarks/op_timings.txt 2>&1
python scripts/layer_profiler.py > benchmarks/layer_timings.txt 2>&1
```

Keep these files. You will reference them in Part 04 when writing your first kernel, and again in Part 05 when deciding which kernel to fuse next.

---

## 3.7 Profiling the Compiled Model

Once you have profiled the un-compiled model (Section 3.2-3.4), repeat with `torch.compile` to see what the compiler already handles:

```python
"""scripts/profile_compiled.py -- Compare un-compiled vs compiled."""
import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.gpt2_modern import GPT2Modern, GPT2Config
from training.dataset import BabyLMDataset
from torch.utils.data import DataLoader


def benchmark(model, data_iter, device, steps=20, label="model"):
    """Time a model over multiple steps, return avg step time."""
    torch.cuda.synchronize()
    
    # Warmup
    for _ in range(5):
        x, y = next(data_iter)
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast('cuda', dtype=torch.float16):
            loss = model(x, y)
        loss.backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    
    # Timed runs
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(steps):
        x, y = next(data_iter)
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast('cuda', dtype=torch.float16):
            loss = model(x, y)
        loss.backward()
        model.zero_grad(set_to_none=True)
    end.record()
    
    torch.cuda.synchronize()
    avg_ms = start.elapsed_time(end) / steps
    tokens_per_step = 8 * 1024  # batch_size * seq_len
    tok_per_sec = tokens_per_step / (avg_ms / 1000)
    
    print(f"{label:25s} | {avg_ms:.2f} ms/step | {tok_per_sec:.0f} tok/s")
    return avg_ms


def main():
    device = torch.device('cuda')
    config = GPT2Config()
    
    dataset = BabyLMDataset("datasets/babylm_train.txt", block_size=1024)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)

    # Un-compiled
    model_eager = GPT2Modern(config).to(device)
    model_eager.train()
    data_iter = iter(loader)
    t_eager = benchmark(model_eager, data_iter, device, label="Eager (no compile)")
    del model_eager
    torch.cuda.empty_cache()

    # Compiled
    model_compiled = GPT2Modern(config).to(device)
    model_compiled.train()
    model_compiled = torch.compile(model_compiled)
    data_iter = iter(loader)
    t_compiled = benchmark(model_compiled, data_iter, device, label="torch.compile")

    print(f"\nSpeedup: {t_eager / t_compiled:.2f}x")
    print(f"\nWhat torch.compile fused away:")
    print(f"  - Element-wise ops (add, mul, silu) -> fused Triton kernels")
    print(f"  - RMSNorm components -> single fused kernel")
    print(f"  - Reduced kernel launch count from ~500+ to ~100")
    print(f"\nWhat remains unfused (still improvable with custom CUDA):")
    print(f"  - RMSNorm + residual add (compile fuses norm but not the residual)")
    print(f"  - SwiGLU gate*up with the down projection")
    print(f"  - Cross-entropy with large vocabulary")


if __name__ == "__main__":
    main()
```

Expected results:
```
Eager (no compile)        | 52.30 ms/step | 12420 tok/s
torch.compile             | 38.50 ms/step | 16880 tok/s

Speedup: 1.36x
```

torch.compile already captures much of the "easy" fusion. Custom CUDA kernels in Part 04-05 target the remaining gaps that the compiler misses, particularly cross-layer fusions and operations with custom memory access patterns.

---

## Exercises

1. **Profile different batch sizes.** Run the manual timer with batch_size=1, 4, 8, 16. How does the matmul-to-overhead ratio change? At what batch size does the GPU become fully utilized?

2. **Profile inference vs training.** Time just the forward pass (no backward). What percentage of total time is backward? Is it exactly 2x forward, or is the ratio different?

3. **Find the memory bottleneck.** Increase batch size until you run out of VRAM. At what batch size does it OOM? Does gradient checkpointing (`torch.utils.checkpoint`) allow a larger batch?

---

## Checkpoint

Before moving to Part 04, verify:
- [ ] torch.profiler output saved with top-20 ops by CUDA time
- [ ] Chrome trace file generated and viewable in chrome://tracing
- [ ] Manual op timings recorded for: RMSNorm, SwiGLU, attention, matmuls, softmax
- [ ] Layer profiler shows per-layer breakdown
- [ ] Optimization priority list filled in with measured data
- [ ] Eager vs compiled comparison shows >1.2x speedup from torch.compile
- [ ] Top 3-5 optimization targets identified (should include RMSNorm)

---

**Previous: [Part 02 -- Training Stack: GPT-2 on BabyLM](02_training_stack.md)**
**Next: [Part 04 -- Your First CUDA Kernel: RMSNorm](04_first_cuda_kernel.md)**
