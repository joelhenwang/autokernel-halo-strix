#!/usr/bin/env python3
"""
AutoKernel End-to-End Verifier -- Plug optimized kernels back into the model and verify.

Usage:
    uv run verify.py --model models/llama_7b.py --class-name LlamaModel --input-shape 1,2048
    uv run verify.py --module transformers --class-name AutoModelForCausalLM --pretrained meta-llama/Llama-2-7b-hf
    uv run verify.py --model models/llama_7b.py --class-name LlamaModel --input-shape 1,2048 --diagnose

Checks:
  1. Loads the original model
  2. Runs inference with original PyTorch ops -> captures reference output
  3. Replaces bottleneck ops with optimized HIP C++ kernels
  4. Runs inference with optimized kernels -> captures optimized output
  5. Compares outputs (tolerance check)
  6. Benchmarks both paths -> reports end-to-end speedup
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.join(SCRIPT_DIR, "workspace")
ORCHESTRATION_STATE = os.path.join(WORKSPACE_DIR, "orchestration_state.json")

# Benchmarking defaults
WARMUP_RUNS = 10
TIMED_RUNS = 50

# Tolerance defaults by dtype
DEFAULT_TOLERANCES: Dict[torch.dtype, Dict[str, float]] = {
    torch.float16:  {"atol": 1e-3, "rtol": 1e-3},
    torch.bfloat16: {"atol": 2e-3, "rtol": 2e-3},
    torch.float32:  {"atol": 1e-5, "rtol": 1e-5},
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class KernelReplacement:
    """Describes a single kernel replacement: what to replace and with what."""
    kernel_type: str          # e.g. "matmul", "layernorm", "rmsnorm"
    rank: int                 # priority rank from profiling
    speedup: float            # individual kernel speedup
    optimized_path: str       # path to optimized kernel .py file
    module_fn: Optional[Callable] = None  # loaded kernel function


@dataclass
class VerificationResult:
    """Full verification result."""
    model_name: str = ""
    input_shape: str = ""
    dtype_str: str = ""
    gpu_name: str = ""

    # Reference run
    ref_output_shape: str = ""
    ref_latency_ms: float = 0.0

    # Optimized run
    opt_output_shape: str = ""
    opt_latency_ms: float = 0.0
    kernels_replaced: List[Dict[str, Any]] = field(default_factory=list)

    # Comparison
    correctness: str = "UNKNOWN"
    max_abs_error: float = 0.0
    mean_abs_error: float = 0.0
    has_nan: bool = False
    has_inf: bool = False

    # Summary
    end_to_end_speedup: float = 0.0


# ---------------------------------------------------------------------------
# 1. Model Loading
# ---------------------------------------------------------------------------

def load_model_from_file(model_path: str, class_name: str, **kwargs) -> nn.Module:
    """Load a model from a Python file by importing it and instantiating the class."""
    model_path = os.path.abspath(model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    spec = importlib.util.spec_from_file_location("user_model", model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import model from: {model_path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not hasattr(mod, class_name):
        available = [n for n in dir(mod) if not n.startswith("_")]
        raise AttributeError(
            f"Class '{class_name}' not found in {model_path}. "
            f"Available names: {available}"
        )

    cls = getattr(mod, class_name)
    model = cls(**kwargs)
    return model


def load_model_from_module(module_name: str, class_name: str,
                           pretrained: Optional[str] = None, **kwargs) -> nn.Module:
    """Load a model from an installed Python module (e.g. 'transformers')."""
    try:
        mod = importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(
            f"Cannot import module '{module_name}'. Is it installed? Error: {e}"
        )

    if not hasattr(mod, class_name):
        raise AttributeError(
            f"Class '{class_name}' not found in module '{module_name}'."
        )

    cls = getattr(mod, class_name)

    if pretrained:
        # HuggingFace-style: cls.from_pretrained(...)
        if hasattr(cls, "from_pretrained"):
            model = cls.from_pretrained(pretrained, **kwargs)
        else:
            raise AttributeError(
                f"'{class_name}' has no 'from_pretrained' method. "
                f"Cannot load pretrained weights from '{pretrained}'."
            )
    else:
        model = cls(**kwargs)

    return model


def load_model(args) -> nn.Module:
    """Unified model loader from CLI args."""
    dtype = _parse_dtype(args.dtype)

    if args.model:
        print(f"Loading model from file: {args.model} (class: {args.class_name})")
        model = load_model_from_file(args.model, args.class_name)
    elif args.module:
        print(f"Loading model from module: {args.module} (class: {args.class_name})")
        extra_kwargs = {}
        if dtype == torch.float16:
            extra_kwargs["torch_dtype"] = torch.float16
        elif dtype == torch.bfloat16:
            extra_kwargs["torch_dtype"] = torch.bfloat16
        model = load_model_from_module(
            args.module, args.class_name, pretrained=args.pretrained, **extra_kwargs
        )
    else:
        raise ValueError("Must specify either --model (file path) or --module (Python module)")

    # Save complex buffers before dtype cast — model.to(dtype=float16) discards
    # imaginary parts of complex64 buffers (e.g. freqs_cis in LLaMA).
    _complex_buffers = {}
    for name, buf in model.named_buffers():
        if buf.is_complex():
            _complex_buffers[name] = buf.clone()

    model = model.to(dtype=dtype)

    # Restore complex buffers that were destroyed by dtype cast
    if _complex_buffers:
        for name, original_buf in _complex_buffers.items():
            parts = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            # Re-register as buffer on the correct device
            parent.register_buffer(parts[-1], original_buf, persistent=False)

    if torch.cuda.is_available():
        try:
            model = model.cuda()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"WARNING: OOM moving model to GPU. Trying with smaller footprint...")
                torch.cuda.empty_cache()
                model = model.half().cuda()
            else:
                raise

    model.eval()
    return model


# ---------------------------------------------------------------------------
# 2. Input Generation
# ---------------------------------------------------------------------------

def generate_sample_input(
    input_shape: str,
    dtype: torch.dtype,
    device: str = "cuda",
    seed: int = 42,
) -> torch.Tensor:
    """Generate a sample input tensor from a shape string like '1,2048'."""
    dims = [int(d.strip()) for d in input_shape.split(",")]
    torch.manual_seed(seed)

    if dtype in (torch.int32, torch.int64, torch.long):
        # For language models, generate token IDs (assume vocab size ~32000)
        return torch.randint(0, 32000, dims, device=device, dtype=dtype)
    else:
        return torch.randn(dims, device=device, dtype=dtype)


def infer_input_type(model: nn.Module) -> str:
    """Try to determine if the model expects integer token IDs or float tensors."""
    # Check if model has an embedding layer as the first module
    for name, child in model.named_children():
        if isinstance(child, nn.Embedding):
            return "token_ids"
        if isinstance(child, (nn.Linear, nn.Conv2d)):
            return "float"
    return "float"


def make_model_input(
    model: nn.Module,
    input_shape: str,
    dtype: torch.dtype,
    device: str = "cuda",
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """Create an appropriate input for the model."""
    input_type = infer_input_type(model)

    if input_type == "token_ids":
        # Language model: expects integer input_ids
        dims = [int(d.strip()) for d in input_shape.split(",")]
        torch.manual_seed(42)
        input_ids = torch.randint(0, 32000, dims, device=device, dtype=torch.long)

        # Check if model accepts input_ids keyword
        sig = inspect.signature(model.forward)
        if "input_ids" in sig.parameters:
            return {"input_ids": input_ids}
        return input_ids
    else:
        return generate_sample_input(input_shape, dtype, device)


# ---------------------------------------------------------------------------
# 3. Benchmarking
# ---------------------------------------------------------------------------

def benchmark_model(
    model: nn.Module,
    model_input: Union[torch.Tensor, Dict[str, torch.Tensor]],
    warmup: int = WARMUP_RUNS,
    timed: int = TIMED_RUNS,
) -> Tuple[Any, float]:
    """
    Benchmark model inference. Returns (output, median_latency_ms).
    Uses CUDA events for precise GPU timing.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for benchmarking.")

    def _run():
        with torch.no_grad():
            if isinstance(model_input, dict):
                return model(**model_input)
            else:
                return model(model_input)

    # Warmup
    print(f"  Warmup: {warmup} runs...", end="", flush=True)
    for _ in range(warmup):
        output = _run()
    torch.cuda.synchronize()
    print(" done")

    # Timed runs
    print(f"  Timed: {timed} runs...", end="", flush=True)
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(timed)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(timed)]

    torch.cuda.synchronize()
    for i in range(timed):
        start_events[i].record()
        _run()
        end_events[i].record()
    torch.cuda.synchronize()
    print(" done")

    # Compute median
    times_ms = sorted(s.elapsed_time(e) for s, e in zip(start_events, end_events))
    median_ms = times_ms[len(times_ms) // 2]

    # Final reference output (deterministic)
    with torch.no_grad():
        output = _run()
    torch.cuda.synchronize()

    return output, median_ms


# ---------------------------------------------------------------------------
# 4. Kernel Replacement
# ---------------------------------------------------------------------------

def load_orchestration_state() -> Optional[Dict]:
    """Load workspace/orchestration_state.json if it exists."""
    if not os.path.exists(ORCHESTRATION_STATE):
        return None
    with open(ORCHESTRATION_STATE, "r") as f:
        return json.load(f)


def discover_optimized_kernels() -> List[KernelReplacement]:
    """
    Find optimized kernels from the workspace directory.
    Checks orchestration_state.json first, then scans for *_optimized.py files.
    """
    replacements: List[KernelReplacement] = []

    # Strategy 1: Read orchestration state
    state = load_orchestration_state()
    if state and "kernels" in state:
        for k in state["kernels"]:
            ktype = k.get("op_type", k.get("type", "unknown"))
            rank = k.get("rank", 0)
            speedup = k.get("speedup", k.get("best_speedup", 1.0))
            # optimized_path is not written by orchestrate.py, so derive it
            # from the kernel file path if available
            opt_path = k.get("optimized_path", "")

            if not opt_path:
                # Try to derive from the "file" key that orchestrate.py writes
                base_file = k.get("file", "")
                if base_file:
                    stem = Path(base_file).stem
                    opt_path = os.path.join(
                        WORKSPACE_DIR, f"{stem}_optimized.py"
                    )
                else:
                    # Fallback convention: workspace/kernel_{type}_{rank}_optimized.py
                    opt_path = os.path.join(
                        WORKSPACE_DIR, f"kernel_{ktype}_{rank}_optimized.py"
                    )

            if os.path.exists(opt_path) and speedup > 1.0:
                replacements.append(KernelReplacement(
                    kernel_type=ktype,
                    rank=rank,
                    speedup=speedup,
                    optimized_path=opt_path,
                ))
        return replacements

    # Strategy 2: Scan workspace directory for optimized kernel files
    if not os.path.isdir(WORKSPACE_DIR):
        return replacements

    for fname in sorted(os.listdir(WORKSPACE_DIR)):
        if fname.endswith("_optimized.py"):
            # Parse filename: kernel_{type}_{rank}_optimized.py
            # Type can be multi-word (e.g. flash_attention), so the rank
            # is always the last numeric segment before "_optimized.py".
            stem = fname.replace("_optimized.py", "")  # e.g. "kernel_flash_attention_1"
            parts = stem.split("_")
            if len(parts) >= 3 and parts[0] == "kernel":
                # Find the rank: last part that is purely numeric
                rank = 0
                rank_idx = len(parts)
                for i in range(len(parts) - 1, 0, -1):
                    if parts[i].isdigit():
                        rank = int(parts[i])
                        rank_idx = i
                        break
                # Everything between parts[1] and the rank index is the type
                ktype = "_".join(parts[1:rank_idx]) if rank_idx > 1 else parts[1]
                opt_path = os.path.join(WORKSPACE_DIR, fname)
                replacements.append(KernelReplacement(
                    kernel_type=ktype,
                    rank=rank,
                    speedup=0.0,  # Unknown without state file
                    optimized_path=opt_path,
                ))

    return replacements


def load_kernel_module(path: str) -> Any:
    """Dynamically import a kernel .py file and return the module."""
    path = os.path.abspath(path)
    module_name = f"opt_kernel_{os.path.basename(path).replace('.py', '')}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load kernel from: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _LinearWrapper(nn.Module):
    """Wraps nn.Linear to use an optimized matmul kernel_fn."""

    def __init__(self, original: nn.Linear, kernel_fn: Callable):
        super().__init__()
        self.original = original
        self.kernel_fn = kernel_fn
        self.weight = original.weight
        self.bias = original.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape to 2D for kernel_fn, then reshape back
        orig_shape = x.shape
        if x.dim() > 2:
            x_2d = x.reshape(-1, x.shape[-1])
        else:
            x_2d = x

        # kernel_fn expects (A, B) where A @ B = C
        # For nn.Linear: output = input @ weight.T + bias
        # So we call kernel_fn(input, weight.T)
        weight_t = self.weight.t().contiguous()
        out = self.kernel_fn(x_2d, weight_t)

        if self.bias is not None:
            out = out + self.bias

        if len(orig_shape) > 2:
            out = out.reshape(*orig_shape[:-1], out.shape[-1])

        return out


class _LayerNormWrapper(nn.Module):
    """Wraps nn.LayerNorm to use an optimized kernel_fn."""

    def __init__(self, original: nn.LayerNorm, kernel_fn: Callable):
        super().__init__()
        self.original = original
        self.kernel_fn = kernel_fn
        self.weight = original.weight
        self.bias = original.bias
        self.eps = original.eps
        self.normalized_shape = original.normalized_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape if needed: kernel_fn expects (x, weight, bias[, eps])
        orig_shape = x.shape
        if x.dim() > 2:
            x_2d = x.reshape(-1, x.shape[-1])
        else:
            x_2d = x

        try:
            # Try full signature: kernel_fn(x, weight, bias, eps)
            out = self.kernel_fn(x_2d, self.weight, self.bias, self.eps)
        except TypeError:
            try:
                # Try without eps: kernel_fn(x, weight, bias)
                out = self.kernel_fn(x_2d, self.weight, self.bias)
            except TypeError:
                # Fallback: just x
                out = self.kernel_fn(x_2d)

        if len(orig_shape) > 2:
            out = out.reshape(orig_shape)

        return out


class _RMSNormWrapper(nn.Module):
    """Wraps RMSNorm-like modules to use an optimized kernel_fn."""

    def __init__(self, original: nn.Module, kernel_fn: Callable):
        super().__init__()
        self.original = original
        self.kernel_fn = kernel_fn
        # RMSNorm typically has a 'weight' attribute
        self.weight = getattr(original, "weight", None)
        self.eps = getattr(original, "eps", 1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        if x.dim() > 2:
            x_2d = x.reshape(-1, x.shape[-1])
        else:
            x_2d = x

        if self.weight is not None:
            try:
                out = self.kernel_fn(x_2d, self.weight, self.eps)
            except TypeError:
                out = self.kernel_fn(x_2d, self.weight)
        else:
            out = self.kernel_fn(x_2d)

        if len(orig_shape) > 2:
            out = out.reshape(orig_shape)

        return out


class _SiluGateMulWrapper(nn.Module):
    """Wraps SwiGLU FeedForward modules (with w1/w2/w3) to use silu_gate_mul kernel."""

    def __init__(self, original: nn.Module, kernel_fn: Callable):
        super().__init__()
        self.w1 = original.w1   # gate projection
        self.w2 = original.w2   # down projection
        self.w3 = original.w3   # up projection
        self.kernel_fn = kernel_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.w1(x)
        up = self.w3(x)
        activated = self.kernel_fn(gate, up)  # silu(gate) * up
        return self.w2(activated)


class _RotaryAttentionWrapper(nn.Module):
    """Wraps LLaMA-style Attention modules to use rotary_embedding kernel."""

    def __init__(self, original: nn.Module, kernel_fn: Callable,
                 freqs_cis_complex: Optional[torch.Tensor] = None):
        super().__init__()
        self.kernel_fn = kernel_fn
        # Copy attributes from original Attention
        for attr in ("n_heads", "n_kv_heads", "head_dim", "n_rep", "wq", "wk", "wv", "wo"):
            if hasattr(original, attr):
                setattr(self, attr, getattr(original, attr))
        # Store precomputed cos/sin from original complex freqs_cis
        # (model.to(dtype=float16) casts complex64 buffers to real, losing imag)
        if freqs_cis_complex is not None and freqs_cis_complex.is_complex():
            self.register_buffer("_cos", freqs_cis_complex.real.clone(), persistent=False)
            self.register_buffer("_sin", freqs_cis_complex.imag.clone(), persistent=False)
        else:
            self._cos = None
            self._sin = None

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Extract cos/sin — use precomputed if freqs_cis was cast to real by model.to(dtype)
        # Keep as fp32 to avoid precision loss — kernel_fn_fp32 needs fp32 cos/sin,
        # kernel_fn (fp16 native) will cast internally if needed.
        if freqs_cis.is_complex():
            cos = freqs_cis[:T].real
            sin = freqs_cis[:T].imag
        elif self._cos is not None:
            cos = self._cos[:T]
            sin = self._sin[:T]
        else:
            raise RuntimeError("freqs_cis is not complex and no precomputed cos/sin available")

        q = self.kernel_fn(q.contiguous(), cos, sin)
        k = self.kernel_fn(k.contiguous(), cos, sin)

        # GQA key/value repeat
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(y.transpose(1, 2).contiguous().view(B, T, -1))


class _FusedQKVAttentionWrapper(nn.Module):
    """Wraps LLaMA Attention to fuse wq/wk/wv into a single matmul.

    rocBLAS achieves higher TFLOPS utilization on the larger concatenated GEMM
    ([M,K]@[K, Nq+Nk+Nv]) vs three separate calls.  Optionally integrates
    the rotary_embedding kernel so the standalone _RotaryAttentionWrapper is
    not needed.
    """

    def __init__(
        self,
        original: nn.Module,
        rotary_kernel_fn: Optional[Callable] = None,
        freqs_cis_complex: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        # Build fused QKV weight: concatenate wq, wk, wv along output dim
        q_out = original.wq.out_features
        k_out = original.wk.out_features
        v_out = original.wv.out_features
        in_feat = original.wq.in_features

        self.w_qkv = nn.Linear(in_feat, q_out + k_out + v_out, bias=False)
        self.w_qkv.weight.data = torch.cat([
            original.wq.weight.data,
            original.wk.weight.data,
            original.wv.weight.data,
        ], dim=0)

        self.wo = original.wo
        self.n_heads = original.n_heads
        self.n_kv_heads = original.n_kv_heads
        self.head_dim = original.head_dim
        self.n_rep = original.n_rep
        self.q_size = q_out
        self.k_size = k_out

        # Optional rotary kernel integration
        self.rotary_kernel_fn = rotary_kernel_fn
        if freqs_cis_complex is not None and freqs_cis_complex.is_complex():
            self.register_buffer("_cos", freqs_cis_complex.real.clone(), persistent=False)
            self.register_buffer("_sin", freqs_cis_complex.imag.clone(), persistent=False)
        else:
            self._cos = None
            self._sin = None

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        # Single fused matmul for Q, K, V
        qkv = self.w_qkv(x)  # [B, T, q_size + k_size + v_size]

        q = qkv[..., :self.q_size].view(B, T, self.n_heads, self.head_dim)
        k = qkv[..., self.q_size:self.q_size + self.k_size].view(
            B, T, self.n_kv_heads, self.head_dim
        )
        v = qkv[..., self.q_size + self.k_size:].view(
            B, T, self.n_kv_heads, self.head_dim
        )

        # Apply RoPE
        if self.rotary_kernel_fn is not None:
            # Use HIP kernel (operates on [B, H, T, D])
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2)
            if freqs_cis.is_complex():
                cos = freqs_cis[:T].real
                sin = freqs_cis[:T].imag
            elif self._cos is not None:
                cos = self._cos[:T]
                sin = self._sin[:T]
            else:
                raise RuntimeError("freqs_cis is not complex and no precomputed cos/sin")
            q = self.rotary_kernel_fn(q, cos, sin)
            k = self.rotary_kernel_fn(k, cos, sin)
        else:
            # Use model's apply_rotary_emb (complex multiplication in fp32)
            from models.llama_7b import apply_rotary_emb
            q, k = apply_rotary_emb(q, k, freqs_cis)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

        # GQA repeat
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(y.transpose(1, 2).contiguous().view(B, T, -1))


class _FusedResidualRMSNormBlockWrapper(nn.Module):
    """Wraps TransformerBlock to fuse residual_add + ffn_norm using dual-output kernel.

    Replaces the pattern:
        x_new = x + attention(attention_norm(x), freqs_cis)
        x_out = x_new + feed_forward(ffn_norm(x_new))
    With:
        attn_out = attention(attention_norm(x), freqs_cis)
        hidden, normed = kernel_fn_dual(attn_out, x, ffn_norm.weight)  # fused
        x_out = hidden + feed_forward(normed)
    """

    def __init__(self, original_block: nn.Module, kernel_fn_dual: Callable):
        super().__init__()
        self.attention = original_block.attention
        self.attention_norm = original_block.attention_norm
        self.feed_forward = original_block.feed_forward
        self.ffn_norm_weight = original_block.ffn_norm.weight
        self.kernel_fn_dual = kernel_fn_dual

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        attn_out = self.attention(self.attention_norm(x), freqs_cis)
        # Fuse: hidden = attn_out + x, normed = rmsnorm(hidden, ffn_norm.weight)
        hidden, normed = self.kernel_fn_dual(
            attn_out.view(-1, attn_out.shape[-1]),
            x.view(-1, x.shape[-1]),
            self.ffn_norm_weight,
        )
        hidden = hidden.view(attn_out.shape)
        normed = normed.view(attn_out.shape)
        ffn_out = self.feed_forward(normed)
        return hidden + ffn_out


def _patch_llama_for_compile(model: nn.Module) -> nn.Module:
    """Patch LlamaModel forward methods to use registered custom ops.

    Monkey-patches CLASS-level forward methods so torch.compile's tracer sees
    custom op calls as opaque nodes.  Inductor fuses all PyTorch ops between them.
    """
    import kernels.hip._torch_ops  # noqa: F401 — triggers op registration
    from models.llama_7b import RMSNorm, FeedForward, Attention, TransformerBlock

    # 1. RMSNorm → autokernel::rmsnorm
    def _rmsnorm_forward(self, x):
        if x.dtype == torch.float16:
            return torch.ops.autokernel.rmsnorm(
                x.view(-1, x.shape[-1]), self.weight
            ).view(x.shape)
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight

    RMSNorm.forward = _rmsnorm_forward

    # 2. FeedForward → autokernel::silu_gate_mul
    def _ffn_forward(self, x):
        gate = self.w1(x)
        up = self.w3(x)
        if gate.dtype == torch.float16:
            activated = torch.ops.autokernel.silu_gate_mul(
                gate.contiguous(), up.contiguous()
            )
        else:
            activated = F.silu(gate) * up
        return self.w2(activated)

    FeedForward.forward = _ffn_forward

    # 3. Attention → autokernel::rotary_emb_fp32
    def _attn_forward(self, x, freqs_cis):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        if x.dtype == torch.float16 and freqs_cis.is_complex():
            cos = freqs_cis[:T].real   # fp32
            sin = freqs_cis[:T].imag   # fp32
            q = torch.ops.autokernel.rotary_emb_fp32(
                q.transpose(1, 2).contiguous(), cos, sin
            )
            k = torch.ops.autokernel.rotary_emb_fp32(
                k.transpose(1, 2).contiguous(), cos, sin
            )
            v = v.transpose(1, 2)
        else:
            from models.llama_7b import apply_rotary_emb
            q, k = apply_rotary_emb(q, k, freqs_cis)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(y.transpose(1, 2).contiguous().view(B, T, -1))

    Attention.forward = _attn_forward

    # 4. TransformerBlock → autokernel::fused_res_rmsnorm
    def _block_forward(self, x, freqs_cis):
        attn_out = self.attention(self.attention_norm(x), freqs_cis)
        if attn_out.dtype == torch.float16:
            hidden, normed = torch.ops.autokernel.fused_res_rmsnorm(
                attn_out.view(-1, attn_out.shape[-1]),
                x.view(-1, x.shape[-1]),
                self.ffn_norm.weight,
            )
            hidden = hidden.view(attn_out.shape)
            normed = normed.view(attn_out.shape)
        else:
            hidden = x + attn_out
            normed = self.ffn_norm(hidden)
        return hidden + self.feed_forward(normed)

    TransformerBlock.forward = _block_forward

    return model


class OptimizedModelContext:
    """
    Context manager that patches a model's submodules to use optimized HIP C++ kernels.

    Usage:
        with OptimizedModelContext(model, replacements) as patched_model:
            output = patched_model(input)
    """

    def __init__(self, model: nn.Module, replacements: List[KernelReplacement],
                 fused_qkv: bool = False,
                 all_replacements: Optional[List[KernelReplacement]] = None):
        self.model = model
        self.replacements = replacements
        self.fused_qkv = fused_qkv
        # all_replacements: full list of discovered kernels (for finding rotary in fused QKV)
        self._all_replacements = all_replacements or replacements
        self._original_modules: Dict[str, nn.Module] = {}
        self._applied: List[str] = []

    def __enter__(self) -> nn.Module:
        # Apply fused QKV first (before kernel replacements) so that
        # downstream wrappers (block fusion, silu_gate_mul) see the fused attention.
        if self.fused_qkv:
            count = self._replace_fused_qkv_attention()
            if count > 0:
                self._applied.append(f"  fused_qkv: {count} attention modules fused")

        for repl in self.replacements:
            try:
                kernel_mod = load_kernel_module(repl.optimized_path)
                # Special dispatch: some kernel types need non-standard entry points
                if repl.kernel_type == "fused_residual_add_rmsnorm":
                    if hasattr(kernel_mod, "kernel_fn_dual"):
                        repl.module_fn = kernel_mod.kernel_fn_dual
                    else:
                        print(f"  WARNING: {repl.optimized_path} has no kernel_fn_dual, skipping")
                        continue
                elif repl.kernel_type == "rotary_embedding" and hasattr(kernel_mod, "kernel_fn_fp32"):
                    repl.module_fn = kernel_mod.kernel_fn_fp32
                else:
                    if not hasattr(kernel_mod, "kernel_fn"):
                        print(f"  WARNING: {repl.optimized_path} has no kernel_fn, skipping")
                        continue
                    repl.module_fn = kernel_mod.kernel_fn
            except Exception as e:
                print(f"  WARNING: Failed to load {repl.optimized_path}: {e}")
                continue

            replaced = self._apply_replacement(repl)
            if replaced > 0:
                self._applied.append(
                    f"  {repl.kernel_type} (rank {repl.rank}): "
                    f"{repl.speedup:.1f}x -> {repl.optimized_path}"
                )

        return self.model

    def __exit__(self, *exc):
        # Restore all original modules
        for name, original in self._original_modules.items():
            parts = name.split(".")
            parent = self.model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], original)
        self._original_modules.clear()
        self._applied.clear()

    def _apply_replacement(self, repl: KernelReplacement) -> int:
        """
        Replace matching modules in the model. Returns number of modules replaced.
        """
        count = 0

        if repl.kernel_type == "matmul":
            count = self._replace_linear_modules(repl)
        elif repl.kernel_type == "layernorm":
            count = self._replace_layernorm_modules(repl)
        elif repl.kernel_type == "rmsnorm":
            count = self._replace_rmsnorm_modules(repl)
        elif repl.kernel_type == "silu_gate_mul":
            count = self._replace_swiglu_modules(repl)
        elif repl.kernel_type == "rotary_embedding":
            if self.fused_qkv:
                # Rotary is already integrated into the fused QKV wrapper
                return 0
            count = self._replace_rotary_in_attention(repl)
        elif repl.kernel_type == "fused_residual_add_rmsnorm":
            count = self._replace_fused_residual_rmsnorm_blocks(repl)
        else:
            print(f"  NOTE: No replacement strategy for kernel type '{repl.kernel_type}'. "
                  f"Skipping. (Supported: matmul, layernorm, rmsnorm, silu_gate_mul, "
                  f"rotary_embedding, fused_residual_add_rmsnorm)")

        return count

    def _replace_linear_modules(self, repl: KernelReplacement) -> int:
        """Replace all nn.Linear modules with optimized matmul wrapper."""
        count = 0
        for name, module in list(self.model.named_modules()):
            if isinstance(module, nn.Linear):
                # Save original
                self._original_modules[name] = module
                # Create wrapper
                wrapper = _LinearWrapper(module, repl.module_fn)
                # Install wrapper
                parts = name.split(".")
                parent = self.model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], wrapper)
                count += 1
        return count

    def _replace_layernorm_modules(self, repl: KernelReplacement) -> int:
        """Replace all nn.LayerNorm modules with optimized wrapper."""
        count = 0
        for name, module in list(self.model.named_modules()):
            if isinstance(module, nn.LayerNorm):
                self._original_modules[name] = module
                wrapper = _LayerNormWrapper(module, repl.module_fn)
                parts = name.split(".")
                parent = self.model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], wrapper)
                count += 1
        return count

    def _replace_rmsnorm_modules(self, repl: KernelReplacement) -> int:
        """
        Replace RMSNorm modules. Since there is no standard nn.RMSNorm,
        we look for common class names and attributes.
        """
        count = 0
        rmsnorm_names = {"RMSNorm", "LlamaRMSNorm", "T5LayerNorm", "GemmaRMSNorm"}

        for name, module in list(self.model.named_modules()):
            cls_name = type(module).__name__
            # Match by class name or by having 'weight' but no 'bias' and a norm-like name
            is_rmsnorm = (
                cls_name in rmsnorm_names
                or (hasattr(module, "weight")
                    and hasattr(module, "eps")
                    and not hasattr(module, "bias")
                    and cls_name.lower().endswith("norm")
                    and not isinstance(module, nn.LayerNorm))
            )

            if is_rmsnorm:
                self._original_modules[name] = module
                wrapper = _RMSNormWrapper(module, repl.module_fn)
                parts = name.split(".")
                parent = self.model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], wrapper)
                count += 1
        return count

    def _replace_swiglu_modules(self, repl: KernelReplacement) -> int:
        """Replace SwiGLU FeedForward modules (w1/w2/w3 pattern) with silu_gate_mul kernel."""
        count = 0
        for name, module in list(self.model.named_modules()):
            has_swiglu = (
                hasattr(module, "w1") and hasattr(module, "w2") and hasattr(module, "w3")
                and isinstance(getattr(module, "w1", None), nn.Linear)
            )
            if has_swiglu:
                self._original_modules[name] = module
                wrapper = _SiluGateMulWrapper(module, repl.module_fn)
                parts = name.split(".")
                parent = self.model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], wrapper)
                count += 1
        return count

    def _replace_rotary_in_attention(self, repl: KernelReplacement) -> int:
        """Replace LLaMA-style Attention modules to use rotary_embedding kernel."""
        # Find complex freqs_cis. model.to(dtype=float16) casts complex64 → real fp16.
        freqs_cis_complex = None
        for bname, buf in self.model.named_buffers():
            if "freqs_cis" in bname:
                if buf.is_complex():
                    freqs_cis_complex = buf
                else:
                    # Buffer was cast to real — recompute from scratch
                    head_dim = None
                    for _, m in self.model.named_modules():
                        if hasattr(m, "head_dim"):
                            head_dim = m.head_dim
                            break
                    if head_dim:
                        max_len = buf.shape[0]
                        theta = 10000.0
                        freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
                        t = torch.arange(max_len, dtype=torch.float32)
                        freqs_outer = torch.outer(t, freqs)
                        freqs_cis_complex = torch.polar(
                            torch.ones_like(freqs_outer), freqs_outer
                        ).to(buf.device)
                break

        count = 0
        for name, module in list(self.model.named_modules()):
            has_attn = (
                hasattr(module, "wq") and hasattr(module, "wk")
                and hasattr(module, "wv") and hasattr(module, "wo")
                and isinstance(getattr(module, "wq", None), nn.Linear)
            )
            if has_attn:
                self._original_modules[name] = module
                wrapper = _RotaryAttentionWrapper(module, repl.module_fn, freqs_cis_complex)
                parts = name.split(".")
                parent = self.model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], wrapper)
                count += 1
        return count

    def _replace_fused_residual_rmsnorm_blocks(self, repl: KernelReplacement) -> int:
        """Replace TransformerBlock modules to fuse residual_add + ffn_norm."""
        count = 0
        for name, module in list(self.model.named_modules()):
            has_block = (
                hasattr(module, "attention") and hasattr(module, "attention_norm")
                and hasattr(module, "feed_forward") and hasattr(module, "ffn_norm")
                and hasattr(getattr(module, "ffn_norm", None), "weight")
            )
            if has_block:
                self._original_modules[name] = module
                wrapper = _FusedResidualRMSNormBlockWrapper(module, repl.module_fn)
                parts = name.split(".")
                parent = self.model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], wrapper)
                count += 1
        return count

    def _replace_fused_qkv_attention(self) -> int:
        """Replace Attention modules with fused QKV projection wrapper."""
        # Find rotary kernel — search ALL discovered replacements (not just active subset)
        rotary_fn = None
        for repl in self._all_replacements:
            if repl.kernel_type == "rotary_embedding" and repl.module_fn is not None:
                rotary_fn = repl.module_fn
                break
        # If rotary kernel not yet loaded, try to load it now
        if rotary_fn is None:
            for repl in self._all_replacements:
                if repl.kernel_type == "rotary_embedding":
                    try:
                        kernel_mod = load_kernel_module(repl.optimized_path)
                        if hasattr(kernel_mod, "kernel_fn_fp32"):
                            rotary_fn = kernel_mod.kernel_fn_fp32
                            repl.module_fn = rotary_fn
                        elif hasattr(kernel_mod, "kernel_fn"):
                            rotary_fn = kernel_mod.kernel_fn
                            repl.module_fn = rotary_fn
                    except Exception:
                        pass
                    break

        # Find complex freqs_cis
        freqs_cis_complex = None
        for bname, buf in self.model.named_buffers():
            if "freqs_cis" in bname and buf.is_complex():
                freqs_cis_complex = buf
                break

        count = 0
        for name, module in list(self.model.named_modules()):
            has_attn = (
                hasattr(module, "wq") and hasattr(module, "wk")
                and hasattr(module, "wv") and hasattr(module, "wo")
                and isinstance(getattr(module, "wq", None), nn.Linear)
            )
            if has_attn:
                self._original_modules[name] = module
                wrapper = _FusedQKVAttentionWrapper(
                    module, rotary_fn, freqs_cis_complex
                )
                parts = name.split(".")
                parent = self.model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], wrapper)
                count += 1
        return count

    @property
    def applied_summary(self) -> List[str]:
        return self._applied


# ---------------------------------------------------------------------------
# 5. Output Comparison
# ---------------------------------------------------------------------------

def extract_tensor(output: Any) -> torch.Tensor:
    """
    Extract a single tensor from model output, which might be a tuple, dict,
    or ModelOutput-like object.
    """
    if isinstance(output, torch.Tensor):
        return output

    # HuggingFace ModelOutput or similar dataclass-like object
    if hasattr(output, "logits"):
        return output.logits
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state

    # Tuple/list: return first tensor element
    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor):
                return item
        # Recurse into first element
        if len(output) > 0:
            return extract_tensor(output[0])

    # Dict: try common keys
    if isinstance(output, dict):
        for key in ["logits", "last_hidden_state", "output", "hidden_states"]:
            if key in output and isinstance(output[key], torch.Tensor):
                return output[key]
        # Return first tensor value
        for v in output.values():
            if isinstance(v, torch.Tensor):
                return v

    raise ValueError(
        f"Cannot extract tensor from output of type {type(output)}. "
        f"Consider adding support for this output format."
    )


def compare_outputs(
    ref_output: torch.Tensor,
    opt_output: torch.Tensor,
    dtype: torch.dtype,
    custom_atol: Optional[float] = None,
    custom_rtol: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compare reference and optimized outputs. Returns comparison metrics.
    """
    result: Dict[str, Any] = {}

    # Shape check
    result["shapes_match"] = ref_output.shape == opt_output.shape
    result["ref_shape"] = str(list(ref_output.shape))
    result["opt_shape"] = str(list(opt_output.shape))

    if not result["shapes_match"]:
        result["correctness"] = "FAIL"
        result["reason"] = f"Shape mismatch: ref={result['ref_shape']}, opt={result['opt_shape']}"
        return result

    # NaN / Inf check
    ref_float = ref_output.float()
    opt_float = opt_output.float()

    result["ref_has_nan"] = bool(torch.isnan(ref_float).any())
    result["ref_has_inf"] = bool(torch.isinf(ref_float).any())
    result["opt_has_nan"] = bool(torch.isnan(opt_float).any())
    result["opt_has_inf"] = bool(torch.isinf(opt_float).any())

    if result["opt_has_nan"] and not result["ref_has_nan"]:
        result["correctness"] = "FAIL"
        result["reason"] = "Optimized output contains NaN where reference does not"
        return result

    if result["opt_has_inf"] and not result["ref_has_inf"]:
        result["correctness"] = "FAIL"
        result["reason"] = "Optimized output contains Inf where reference does not"
        return result

    # Numerical comparison
    diff = (ref_float - opt_float).abs()

    # Mask out positions where both are NaN (those are fine)
    valid_mask = ~(torch.isnan(ref_float) & torch.isnan(opt_float))
    if valid_mask.any():
        valid_diff = diff[valid_mask]
        result["max_abs_error"] = float(valid_diff.max())
        result["mean_abs_error"] = float(valid_diff.mean())
    else:
        result["max_abs_error"] = 0.0
        result["mean_abs_error"] = 0.0

    # Tolerance check
    tols = DEFAULT_TOLERANCES.get(dtype, {"atol": 1e-4, "rtol": 1e-4})
    atol = custom_atol if custom_atol is not None else tols["atol"]
    rtol = custom_rtol if custom_rtol is not None else tols["rtol"]

    # Use allclose on the valid (non-NaN) elements
    if valid_mask.any():
        passes = torch.allclose(
            ref_float[valid_mask], opt_float[valid_mask], atol=atol, rtol=rtol
        )
    else:
        passes = True

    result["correctness"] = "PASS" if passes else "FAIL"
    result["atol"] = atol
    result["rtol"] = rtol

    if not passes:
        result["reason"] = (
            f"Values exceed tolerance (atol={atol}, rtol={rtol}). "
            f"max_abs_error={result['max_abs_error']:.6e}, "
            f"mean_abs_error={result['mean_abs_error']:.6e}"
        )

    return result


# ---------------------------------------------------------------------------
# 6. Diagnosis Mode (apply kernels one at a time)
# ---------------------------------------------------------------------------

def diagnose_kernel_failures(
    model: nn.Module,
    model_input: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ref_tensor: torch.Tensor,
    replacements: List[KernelReplacement],
    dtype: torch.dtype,
) -> List[Dict[str, Any]]:
    """
    Apply each kernel replacement individually to find which one causes failure.
    """
    results = []

    for repl in replacements:
        print(f"\n  Testing kernel: {repl.kernel_type} (rank {repl.rank})...")
        ctx = OptimizedModelContext(model, [repl])

        try:
            with ctx as patched_model:
                with torch.no_grad():
                    if isinstance(model_input, dict):
                        opt_output = patched_model(**model_input)
                    else:
                        opt_output = patched_model(model_input)
                torch.cuda.synchronize()

            opt_tensor = extract_tensor(opt_output)
            comp = compare_outputs(ref_tensor, opt_tensor, dtype)

            results.append({
                "kernel_type": repl.kernel_type,
                "rank": repl.rank,
                "path": repl.optimized_path,
                "correctness": comp["correctness"],
                "max_abs_error": comp.get("max_abs_error", 0.0),
                "mean_abs_error": comp.get("mean_abs_error", 0.0),
                "reason": comp.get("reason", ""),
            })

            status = comp["correctness"]
            if status == "PASS":
                print(f"    -> PASS (max_err={comp.get('max_abs_error', 0):.6e})")
            else:
                print(f"    -> FAIL: {comp.get('reason', 'unknown')}")

        except Exception as e:
            results.append({
                "kernel_type": repl.kernel_type,
                "rank": repl.rank,
                "path": repl.optimized_path,
                "correctness": "ERROR",
                "max_abs_error": float("inf"),
                "mean_abs_error": float("inf"),
                "reason": str(e),
            })
            print(f"    -> ERROR: {e}")

    return results


# ---------------------------------------------------------------------------
# 7. Output Formatting
# ---------------------------------------------------------------------------

def format_report(result: VerificationResult, diagnose_results: Optional[List] = None) -> str:
    """Format the verification result into a human-readable report."""
    lines = []
    lines.append("")
    lines.append("=== AutoKernel End-to-End Verification ===")
    lines.append("")
    lines.append(f"Model: {result.model_name}")
    lines.append(f"Input: [{result.input_shape}], dtype={result.dtype_str}")
    lines.append(f"GPU: {result.gpu_name}")

    # Reference run
    lines.append("")
    lines.append("--- Reference Run ---")
    lines.append(f"Output shape: {result.ref_output_shape}")
    lines.append(f"Latency: {result.ref_latency_ms:.1f} ms ({TIMED_RUNS} runs, median)")

    # Optimized run
    lines.append("")
    lines.append("--- Optimized Run ---")
    if result.kernels_replaced:
        lines.append("Kernels replaced:")
        for k in result.kernels_replaced:
            lines.append(f"  {k['type']} (rank {k['rank']}): "
                         f"{k['speedup']:.1f}x -> {k['path']}")
    else:
        lines.append("Kernels replaced: none")
    lines.append(f"Output shape: {result.opt_output_shape}")
    lines.append(f"Latency: {result.opt_latency_ms:.1f} ms ({TIMED_RUNS} runs, median)")

    # Verification
    lines.append("")
    lines.append("--- Verification ---")
    lines.append(f"correctness: {result.correctness}")
    lines.append(f"max_abs_error: {result.max_abs_error:.2e}")
    lines.append(f"mean_abs_error: {result.mean_abs_error:.2e}")
    if result.has_nan:
        lines.append("WARNING: NaN detected in optimized output")
    if result.has_inf:
        lines.append("WARNING: Inf detected in optimized output")

    # Summary
    lines.append("")
    lines.append("--- Summary ---")
    lines.append(f"original_latency_ms: {result.ref_latency_ms:.1f}")
    lines.append(f"optimized_latency_ms: {result.opt_latency_ms:.1f}")
    lines.append(f"end_to_end_speedup: {result.end_to_end_speedup:.2f}x")
    lines.append(f"kernels_replaced: {len(result.kernels_replaced)}")

    # Diagnosis
    if diagnose_results:
        lines.append("")
        lines.append("--- Diagnosis (per-kernel) ---")
        for dr in diagnose_results:
            status = dr["correctness"]
            line = f"  {dr['kernel_type']} (rank {dr['rank']}): {status}"
            if status == "PASS":
                line += f" | max_err={dr['max_abs_error']:.2e}"
            if dr.get("reason"):
                line += f" | {dr['reason']}"
            lines.append(line)

    lines.append("")
    return "\n".join(lines)


def save_verification_json(result: VerificationResult, path: str) -> None:
    """Save verification results as JSON for programmatic consumption."""
    data = {
        "model": result.model_name,
        "input_shape": result.input_shape,
        "dtype": result.dtype_str,
        "gpu": result.gpu_name,
        "reference": {
            "output_shape": result.ref_output_shape,
            "latency_ms": round(result.ref_latency_ms, 2),
        },
        "optimized": {
            "output_shape": result.opt_output_shape,
            "latency_ms": round(result.opt_latency_ms, 2),
            "kernels_replaced": result.kernels_replaced,
        },
        "verification": {
            "correctness": result.correctness,
            "max_abs_error": result.max_abs_error,
            "mean_abs_error": result.mean_abs_error,
            "has_nan": result.has_nan,
            "has_inf": result.has_inf,
        },
        "summary": {
            "original_latency_ms": round(result.ref_latency_ms, 2),
            "optimized_latency_ms": round(result.opt_latency_ms, 2),
            "end_to_end_speedup": round(result.end_to_end_speedup, 3),
            "kernels_replaced": len(result.kernels_replaced),
        },
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_dtype(dtype_str: str) -> torch.dtype:
    """Parse a dtype string into a torch.dtype."""
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
    }
    key = dtype_str.lower().strip()
    if key not in mapping:
        raise ValueError(f"Unknown dtype '{dtype_str}'. Choose from: {list(mapping.keys())}")
    return mapping[key]


def _get_gpu_name() -> str:
    """Get current GPU name."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "No GPU"


def _output_shape_str(output: Any) -> str:
    """Get shape string from model output."""
    try:
        t = extract_tensor(output)
        return str(list(t.shape))
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Incremental Verification
# ---------------------------------------------------------------------------

def incremental_verification(
    model: nn.Module,
    model_input: Any,
    ref_tensor: torch.Tensor,
    ref_latency: float,
    replacements: List[KernelReplacement],
    dtype: torch.dtype,
    warmup: int,
    timed: int,
    custom_atol: Optional[float] = None,
    custom_rtol: Optional[float] = None,
    fused_qkv: bool = False,
    all_replacements: Optional[List[KernelReplacement]] = None,
) -> List[Dict[str, Any]]:
    """Apply kernels one by one, measuring cumulative correctness and latency."""
    log: List[Dict[str, Any]] = []
    active: List[KernelReplacement] = []

    print(f"\n{'='*60}")
    print(f"  Incremental Verification ({len(replacements)} kernels)")
    print(f"  Baseline latency: {ref_latency:.2f} ms")
    print(f"{'='*60}\n")

    for repl in replacements:
        active.append(repl)
        step = len(log) + 1
        print(f"--- Step {step}: + {repl.kernel_type} ({len(active)} kernels active) ---")

        entry: Dict[str, Any] = {
            "step": step,
            "kernel_added": repl.kernel_type,
            "kernels_active": [r.kernel_type for r in active],
        }

        ctx = OptimizedModelContext(model, list(active), fused_qkv=fused_qkv,
                                    all_replacements=all_replacements or replacements)
        try:
            with ctx as patched_model:
                if ctx.applied_summary:
                    for line in ctx.applied_summary:
                        print(f"  {line}")
                opt_output, opt_latency = benchmark_model(
                    patched_model, model_input, warmup, timed
                )

            opt_tensor = extract_tensor(opt_output)
            comp = compare_outputs(ref_tensor, opt_tensor, dtype, custom_atol, custom_rtol)

            entry["correctness"] = comp["correctness"]
            entry["max_abs_error"] = comp.get("max_abs_error", 0.0)
            entry["latency_ms"] = round(opt_latency, 3)
            entry["speedup_vs_baseline"] = round(ref_latency / opt_latency, 4)

            status = "PASS" if comp["correctness"] == "PASS" else "FAIL"
            print(f"  correctness: {status} (max_err={comp.get('max_abs_error', 0):.2e})")
            print(f"  latency: {opt_latency:.2f} ms (speedup: {ref_latency / opt_latency:.3f}x)")

            if comp["correctness"] != "PASS":
                active.pop()
                entry["note"] = "removed (failed correctness)"
                print(f"  -> Removed {repl.kernel_type} from active set")

        except Exception as e:
            active.pop()
            entry["correctness"] = "ERROR"
            entry["error"] = str(e)
            print(f"  ERROR: {e}")
            traceback.print_exc()
            print(f"  -> Removed {repl.kernel_type} from active set")

        log.append(entry)
        print()

    # Print summary table
    print(f"{'='*60}")
    print(f"  Incremental Verification Summary")
    print(f"{'='*60}")
    print(f"{'Step':<6}{'Kernel':<25}{'Status':<8}{'Latency':>10}{'Speedup':>10}")
    print(f"{'-'*60}")
    print(f"{'base':<6}{'(PyTorch)':<25}{'':<8}{ref_latency:>9.2f}{'1.000x':>10}")
    for e in log:
        status = e.get("correctness", "?")[:4]
        lat = f"{e.get('latency_ms', 0):>9.2f}" if "latency_ms" in e else "     N/A"
        spd = f"{e.get('speedup_vs_baseline', 0):.3f}x" if "speedup_vs_baseline" in e else "N/A"
        note = " *removed" if "note" in e else ""
        print(f"{e['step']:<6}{e['kernel_added']:<25}{status:<8}{lat}{spd:>10}{note}")
    print()

    return log


def append_to_incremental_log(
    log: List[Dict[str, Any]],
    model_name: str,
    input_shape: str,
    dtype_str: str,
    gpu_name: str,
    baseline_latency: float,
) -> str:
    """Append an incremental verification run to the JSON log. Returns log path."""
    log_path = os.path.join(WORKSPACE_DIR, "incremental_log.json")

    if os.path.exists(log_path):
        with open(log_path) as f:
            data = json.load(f)
    else:
        data = {"model": model_name, "runs": []}

    data["runs"].append({
        "run_id": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "input_shape": input_shape,
        "dtype": dtype_str,
        "gpu": gpu_name,
        "baseline_latency_ms": baseline_latency,
        "steps": log,
    })

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(data, f, indent=2)

    return log_path


def generate_progress_chart(log_path: Optional[str] = None) -> Optional[str]:
    """Generate a progress chart from the incremental verification log."""
    if log_path is None:
        log_path = os.path.join(WORKSPACE_DIR, "incremental_log.json")

    if not os.path.exists(log_path):
        print(f"No incremental log found at {log_path}")
        return None

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping chart generation")
        return None

    with open(log_path) as f:
        data = json.load(f)

    if not data.get("runs"):
        print("No runs in log")
        return None

    # Use the latest run
    run = data["runs"][-1]
    baseline = run["baseline_latency_ms"]
    steps = run["steps"]

    labels = ["PyTorch\n(baseline)"]
    latencies = [baseline]
    speedups = [1.0]
    colors = ["#888888"]

    for s in steps:
        labels.append(s["kernel_added"])
        lat = s.get("latency_ms", baseline)
        latencies.append(lat)
        speedups.append(s.get("speedup_vs_baseline", 1.0))
        colors.append("#4CAF50" if s.get("correctness") == "PASS" else "#F44336")

    fig, ax1 = plt.subplots(figsize=(10, 5))

    x = range(len(labels))
    bars = ax1.bar(x, latencies, color=colors, alpha=0.8, width=0.6)
    ax1.set_ylabel("Latency (ms)", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)

    ax2 = ax1.twinx()
    ax2.plot(x, speedups, "ko-", linewidth=2, markersize=6, label="Cumulative speedup")
    ax2.set_ylabel("Speedup vs PyTorch", fontsize=11)
    ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left")

    model_name = run.get("model", "Model")
    ax1.set_title(f"AutoKernel Incremental Verification — {model_name}", fontsize=13)

    plt.tight_layout()
    chart_path = os.path.join(WORKSPACE_DIR, "verification_progress.png")
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    print(f"Chart saved: {chart_path}")
    return chart_path


# ---------------------------------------------------------------------------
# Decode Benchmark (autoregressive generation with KV-cache)
# ---------------------------------------------------------------------------

def decode_benchmark(
    model: nn.Module,
    input_ids: torch.Tensor,
    n_decode_tokens: int = 32,
    warmup: int = 3,
    timed: int = 10,
) -> Dict[str, Any]:
    """Benchmark autoregressive token generation with KV-cache.

    Implements a simple KV-cache loop: prefill the input sequence, then decode
    one token at a time.  Returns timing statistics.
    """
    from models.llama_7b import apply_rotary_emb

    B, T = input_ids.shape
    device = input_ids.device

    # Access model internals (works for LlamaModel / LlamaModel7B)
    layers = model.layers
    tok_emb = model.tok_embeddings
    norm = model.norm
    output_proj = model.output
    freqs_cis = model.freqs_cis

    def _prefill():
        """Run prefill and return KV-cache + logits."""
        h = tok_emb(input_ids)
        kv_cache = []
        for layer in layers:
            # --- Attention with KV capture ---
            attn = layer.attention
            normed = layer.attention_norm(h)
            B_, T_, _ = normed.shape
            q = attn.wq(normed).view(B_, T_, attn.n_heads, attn.head_dim)
            k = attn.wk(normed).view(B_, T_, attn.n_kv_heads, attn.head_dim)
            v = attn.wv(normed).view(B_, T_, attn.n_kv_heads, attn.head_dim)
            q, k = apply_rotary_emb(q, k, freqs_cis[:T_])
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            if attn.n_rep > 1:
                k_exp = k.repeat_interleave(attn.n_rep, dim=1)
                v_exp = v.repeat_interleave(attn.n_rep, dim=1)
            else:
                k_exp, v_exp = k, v
            y = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=True)
            h = h + attn.wo(y.transpose(1, 2).contiguous().view(B_, T_, -1))
            kv_cache.append((k, v))  # store un-repeated KV

            # --- FFN ---
            h = h + layer.feed_forward(layer.ffn_norm(h))

        logits = output_proj(norm(h))
        return kv_cache, logits

    def _decode_step(kv_cache, token_id, pos):
        """Decode one token, update KV-cache in place, return logits."""
        h = tok_emb(token_id)  # [1, 1, dim]
        for i, layer in enumerate(layers):
            attn = layer.attention
            normed = layer.attention_norm(h)
            q = attn.wq(normed).view(1, 1, attn.n_heads, attn.head_dim)
            k_new = attn.wk(normed).view(1, 1, attn.n_kv_heads, attn.head_dim)
            v_new = attn.wv(normed).view(1, 1, attn.n_kv_heads, attn.head_dim)
            q, k_new = apply_rotary_emb(q, k_new, freqs_cis[pos:pos + 1])
            q = q.transpose(1, 2)
            k_new = k_new.transpose(1, 2)
            v_new = v_new.transpose(1, 2)

            # Append to cache
            k_old, v_old = kv_cache[i]
            k_cat = torch.cat([k_old, k_new], dim=2)
            v_cat = torch.cat([v_old, v_new], dim=2)
            kv_cache[i] = (k_cat, v_cat)

            # Attention with full cache
            if attn.n_rep > 1:
                k_exp = k_cat.repeat_interleave(attn.n_rep, dim=1)
                v_exp = v_cat.repeat_interleave(attn.n_rep, dim=1)
            else:
                k_exp, v_exp = k_cat, v_cat
            y = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=False)
            h = h + attn.wo(y.transpose(1, 2).contiguous().view(1, 1, -1))

            h = h + layer.feed_forward(layer.ffn_norm(h))

        return output_proj(norm(h))

    def _full_generation():
        """Prefill + N decode steps.  Returns per-step decode times."""
        kv_cache, logits = _prefill()
        next_tok = logits[:, -1:].argmax(dim=-1)  # greedy
        pos = T

        decode_times = []
        for _ in range(n_decode_tokens):
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()
            logits = _decode_step(kv_cache, next_tok, pos)
            end_ev.record()
            torch.cuda.synchronize()
            decode_times.append(start_ev.elapsed_time(end_ev))
            next_tok = logits[:, -1:].argmax(dim=-1)
            pos += 1
        return decode_times

    # Warmup
    print(f"  Decode benchmark: prefill {T} tokens, decode {n_decode_tokens} tokens")
    print(f"  Warmup: {warmup} runs...", end="", flush=True)
    with torch.no_grad():
        for _ in range(warmup):
            _full_generation()
    print(" done")

    # Timed runs
    print(f"  Timed: {timed} runs...", end="", flush=True)
    all_decode_times = []
    with torch.no_grad():
        for _ in range(timed):
            dt = _full_generation()
            all_decode_times.append(dt)
    print(" done")

    # Aggregate: median per-token latency
    import numpy as np
    all_times = np.array(all_decode_times)  # [timed, n_decode_tokens]
    median_per_token = np.median(all_times, axis=0)  # [n_decode_tokens]
    avg_decode_ms = float(np.median(median_per_token))
    tokens_per_sec = 1000.0 / avg_decode_ms if avg_decode_ms > 0 else 0.0

    # Also time prefill separately
    prefill_times = []
    with torch.no_grad():
        for _ in range(timed):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            _prefill()
            e.record()
            torch.cuda.synchronize()
            prefill_times.append(s.elapsed_time(e))
    prefill_times.sort()
    prefill_ms = prefill_times[len(prefill_times) // 2]

    result = {
        "prefill_tokens": T,
        "decode_tokens": n_decode_tokens,
        "prefill_ms": round(prefill_ms, 2),
        "avg_decode_ms_per_token": round(avg_decode_ms, 3),
        "tokens_per_sec": round(tokens_per_sec, 1),
        "decode_latencies_ms": [round(float(x), 3) for x in median_per_token],
    }

    print(f"\n  Prefill ({T} tokens): {prefill_ms:.1f} ms")
    print(f"  Decode ({n_decode_tokens} tokens): {avg_decode_ms:.2f} ms/token")
    print(f"  Throughput: {tokens_per_sec:.1f} tokens/sec")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global WORKSPACE_DIR, ORCHESTRATION_STATE, WARMUP_RUNS, TIMED_RUNS

    parser = argparse.ArgumentParser(
        description="AutoKernel End-to-End Verifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model loading
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model", type=str,
        help="Path to a Python file containing the model class"
    )
    model_group.add_argument(
        "--module", type=str,
        help="Python module name (e.g. 'transformers')"
    )

    parser.add_argument(
        "--class-name", type=str, required=True,
        help="Name of the model class to instantiate"
    )
    parser.add_argument(
        "--pretrained", type=str, default=None,
        help="Pretrained model name/path (for HuggingFace models)"
    )
    parser.add_argument(
        "--input-shape", type=str, default="1,2048",
        help="Comma-separated input shape, e.g. '1,2048' (default: 1,2048)"
    )
    parser.add_argument(
        "--dtype", type=str, default="float16",
        help="Data type: float16, bfloat16, float32 (default: float16)"
    )

    # Benchmark tuning
    parser.add_argument(
        "--warmup", type=int, default=WARMUP_RUNS,
        help=f"Number of warmup iterations (default: {WARMUP_RUNS})"
    )
    parser.add_argument(
        "--timed", type=int, default=TIMED_RUNS,
        help=f"Number of timed iterations (default: {TIMED_RUNS})"
    )

    # Tolerance overrides
    parser.add_argument("--atol", type=float, default=None, help="Override absolute tolerance")
    parser.add_argument("--rtol", type=float, default=None, help="Override relative tolerance")

    # Modes
    parser.add_argument(
        "--diagnose", action="store_true",
        help="On failure, test each kernel replacement individually to find the culprit"
    )
    parser.add_argument(
        "--json", type=str, default=None,
        help="Save results to a JSON file at this path"
    )
    parser.add_argument(
        "--workspace", type=str, default=None,
        help="Override workspace directory (default: ./workspace)"
    )
    parser.add_argument(
        "--incremental", action="store_true",
        help="Apply kernels one-by-one, measuring cumulative correctness and latency"
    )
    parser.add_argument(
        "--kernel-order", type=str, default="fused_residual_add_rmsnorm,rmsnorm,silu_gate_mul,rotary_embedding",
        help="Comma-separated kernel application order for --incremental"
    )
    parser.add_argument(
        "--fused-qkv", action="store_true",
        help="Fuse wq/wk/wv into single matmul (rocBLAS utilization optimization)"
    )
    parser.add_argument(
        "--torch-compile", action="store_true",
        help="Apply torch.compile(backend='inductor') before benchmarking"
    )
    parser.add_argument(
        "--compile-with-kernels", action="store_true",
        help="Register HIP kernels as custom ops + torch.compile (combines Inductor fusion with our kernels)"
    )
    parser.add_argument(
        "--compile-mode", type=str, default="default",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode when using --torch-compile or --compile-with-kernels"
    )
    parser.add_argument(
        "--decode-benchmark", action="store_true",
        help="Benchmark autoregressive decode (prefill + N token generation with KV-cache)"
    )
    parser.add_argument(
        "--decode-tokens", type=int, default=32,
        help="Number of tokens to generate in decode benchmark (default: 32)"
    )
    parser.add_argument(
        "--chart", action="store_true",
        help="Generate progress chart from historical incremental log"
    )

    args = parser.parse_args()

    # Override globals if workspace specified
    if args.workspace:
        WORKSPACE_DIR = os.path.abspath(args.workspace)
        ORCHESTRATION_STATE = os.path.join(WORKSPACE_DIR, "orchestration_state.json")

    WARMUP_RUNS = args.warmup
    TIMED_RUNS = args.timed

    # Chart-only mode: generate chart from existing log and exit
    if args.chart:
        generate_progress_chart()
        return

    dtype = _parse_dtype(args.dtype)
    gpu_name = _get_gpu_name()

    print("=" * 60)
    print("  AutoKernel End-to-End Verifier")
    print("=" * 60)
    print()

    # -----------------------------------------------------------------------
    # Step 1: Discover optimized kernels
    # -----------------------------------------------------------------------
    print("Step 1: Discovering optimized kernels...")
    replacements = discover_optimized_kernels()
    if not replacements:
        print()
        print("No optimized kernels found.")
        print(f"  Searched: {WORKSPACE_DIR}")
        print(f"  State file: {ORCHESTRATION_STATE}")
        print()
        print("Run the optimization loop first to produce optimized kernels.")
        print("Expected files: workspace/kernel_<type>_<rank>_optimized.py")
        sys.exit(1)

    print(f"  Found {len(replacements)} optimized kernel(s):")
    for r in replacements:
        print(f"    {r.kernel_type} (rank {r.rank}): speedup={r.speedup:.1f}x -> {r.optimized_path}")
    print()

    # -----------------------------------------------------------------------
    # Step 2: Load model
    # -----------------------------------------------------------------------
    print("Step 2: Loading model...")
    try:
        model = load_model(args)
        model_name = args.class_name
        if args.pretrained:
            model_name = f"{args.class_name} ({args.pretrained})"
        print(f"  Model loaded: {model_name}")
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {param_count:,}")
    except Exception as e:
        print(f"\nERROR: Failed to load model: {e}")
        traceback.print_exc()
        sys.exit(1)
    print()

    # -----------------------------------------------------------------------
    # Step 3: Create input
    # -----------------------------------------------------------------------
    print("Step 3: Creating model input...")
    try:
        model_input = make_model_input(model, args.input_shape, dtype)
        if isinstance(model_input, dict):
            for k, v in model_input.items():
                print(f"  {k}: shape={list(v.shape)}, dtype={v.dtype}")
        else:
            print(f"  Input: shape={list(model_input.shape)}, dtype={model_input.dtype}")
    except Exception as e:
        print(f"\nERROR: Failed to create input: {e}")
        traceback.print_exc()
        sys.exit(1)
    print()

    # -----------------------------------------------------------------------
    # Step 3.5: Decode benchmark mode (early exit)
    # -----------------------------------------------------------------------
    if getattr(args, "decode_benchmark", False):
        print("=" * 60)
        print("  Decode Benchmark (autoregressive generation with KV-cache)")
        print("=" * 60)
        n_tokens = getattr(args, "decode_tokens", 32)
        try:
            model.eval()
            # Extract input_ids tensor
            if isinstance(model_input, dict):
                _ids = model_input.get("input_ids", next(iter(model_input.values())))
            else:
                _ids = model_input
            result = decode_benchmark(
                model, _ids, n_decode_tokens=n_tokens,
                warmup=3, timed=5,
            )
            # Save result
            result_path = os.path.join(WORKSPACE_DIR, "decode_benchmark.json")
            os.makedirs(WORKSPACE_DIR, exist_ok=True)
            with open(result_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\n  Results saved: {result_path}")
        except Exception as e:
            print(f"\nERROR: Decode benchmark failed: {e}")
            traceback.print_exc()
            sys.exit(1)
        return

    # -----------------------------------------------------------------------
    # Step 3.5b: Optional torch.compile
    # -----------------------------------------------------------------------
    if getattr(args, "torch_compile", False):
        print("Step 3.5: Applying torch.compile(backend='inductor')...")
        try:
            # Fix: project's profile.py shadows stdlib profile module used by cProfile.
            # Remove project root from sys.path during torch.compile import/call.
            import sys as _sys
            _cwd = os.getcwd()
            _cwd_was_in_path = _cwd in _sys.path
            if _cwd_was_in_path:
                while _cwd in _sys.path:
                    _sys.path.remove(_cwd)
            # Also clear cached project profile module if it shadows stdlib
            if "profile" in _sys.modules:
                _pm = _sys.modules["profile"]
                if hasattr(_pm, "__file__") and _pm.__file__ and _cwd in str(_pm.__file__):
                    del _sys.modules["profile"]
                    if "cProfile" in _sys.modules:
                        del _sys.modules["cProfile"]

            model = torch.compile(model, backend="inductor")
            # Trigger compilation with a warmup forward
            with torch.no_grad():
                _ = model(model_input) if isinstance(model_input, torch.Tensor) else model(**model_input)
            torch.cuda.synchronize()
            print("  torch.compile: OK (graph compiled)")
        except Exception as e:
            print(f"  WARNING: torch.compile failed: {e}")
            print("  Continuing without compilation...")
        finally:
            if _cwd_was_in_path and _cwd not in _sys.path:
                _sys.path.insert(0, _cwd)
        print()

    # -----------------------------------------------------------------------
    # Step 4: Reference run
    # -----------------------------------------------------------------------
    print("Step 4: Reference run (original PyTorch ops)...")
    try:
        ref_output, ref_latency = benchmark_model(model, model_input, WARMUP_RUNS, TIMED_RUNS)
        ref_tensor = extract_tensor(ref_output)
        ref_shape_str = str(list(ref_tensor.shape))
        print(f"  Output shape: {ref_shape_str}")
        print(f"  Median latency: {ref_latency:.1f} ms")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\nERROR: GPU out of memory during reference run.")
            print("  Try a smaller --input-shape or a smaller model.")
            torch.cuda.empty_cache()
            sys.exit(1)
        else:
            raise
    except Exception as e:
        print(f"\nERROR: Reference run failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    print()

    # -----------------------------------------------------------------------
    # Step 5: compile-with-kernels mode (early exit)
    # -----------------------------------------------------------------------
    if getattr(args, "compile_with_kernels", False):
        print("Step 5: Applying custom ops + torch.compile...")
        try:
            # Patch model forward methods BEFORE removing CWD from sys.path
            # (kernels.hip._torch_ops needs to be importable)
            model = _patch_llama_for_compile(model)

            # Now fix profile.py conflict for torch.compile
            _cwd = os.getcwd()
            _cwd_in = _cwd in sys.path
            if _cwd_in:
                while _cwd in sys.path:
                    sys.path.remove(_cwd)
            if "profile" in sys.modules:
                _pm = sys.modules["profile"]
                if hasattr(_pm, "__file__") and _pm.__file__ and _cwd in str(_pm.__file__):
                    del sys.modules["profile"]
                    if "cProfile" in sys.modules:
                        del sys.modules["cProfile"]
            compile_mode = getattr(args, "compile_mode", "default")
            if compile_mode == "default":
                compiled_model = torch.compile(model, backend="inductor")
            else:
                compiled_model = torch.compile(model, backend="inductor", mode=compile_mode)

            # Warmup (triggers graph compilation)
            print(f"  Compiling graph (mode={compile_mode})...", end="", flush=True)
            with torch.no_grad():
                _ = compiled_model(model_input) if isinstance(model_input, torch.Tensor) else compiled_model(**model_input)
            torch.cuda.synchronize()
            print(" done")

            # Benchmark compiled model
            print(f"  Benchmarking compiled model...", end="", flush=True)
            compiled_output, compiled_latency = benchmark_model(
                compiled_model, model_input, WARMUP_RUNS, TIMED_RUNS
            )
            print(f" {compiled_latency:.1f} ms")

            # Compare
            compiled_tensor = extract_tensor(compiled_output)
            comp = compare_outputs(ref_tensor, compiled_tensor, dtype,
                                   getattr(args, "atol", None), getattr(args, "rtol", None))
            status = comp["correctness"]
            max_err = comp.get("max_abs_error", 0)
            speedup = ref_latency / compiled_latency

            print(f"\n  === compile-with-kernels Results ===")
            print(f"  Reference latency:  {ref_latency:.1f} ms")
            print(f"  Compiled latency:   {compiled_latency:.1f} ms")
            print(f"  Speedup:            {speedup:.3f}x")
            print(f"  Correctness:        {status} (max_err={max_err:.2e})")

        except Exception as e:
            print(f"\n  ERROR: compile-with-kernels failed: {e}")
            traceback.print_exc()
        finally:
            if _cwd_in and _cwd not in sys.path:
                sys.path.insert(0, _cwd)
        return

    # -----------------------------------------------------------------------
    # Step 5b: Incremental or full optimized run
    # -----------------------------------------------------------------------
    if args.incremental:
        # Sort replacements by --kernel-order
        order = [k.strip() for k in args.kernel_order.split(",")]
        ordered: List[KernelReplacement] = []
        for ktype in order:
            for r in replacements:
                if r.kernel_type == ktype:
                    ordered.append(r)
                    break
        # Add any remaining not in the order
        seen_types = {r.kernel_type for r in ordered}
        for r in replacements:
            if r.kernel_type not in seen_types:
                ordered.append(r)

        log = incremental_verification(
            model, model_input, ref_tensor, ref_latency, ordered, dtype,
            WARMUP_RUNS, TIMED_RUNS, args.atol, args.rtol,
            fused_qkv=getattr(args, "fused_qkv", False),
            all_replacements=ordered,
        )

        model_name = args.class_name
        log_path = append_to_incremental_log(
            log, model_name, args.input_shape, args.dtype, gpu_name, ref_latency
        )
        print(f"Log saved: {log_path}")
        chart = generate_progress_chart(log_path)
        return

    print("Step 5: Optimized run (with HIP kernel replacements)...")
    ctx = OptimizedModelContext(model, replacements,
                                fused_qkv=getattr(args, "fused_qkv", False))
    try:
        with ctx as patched_model:
            if ctx.applied_summary:
                print("  Replacements applied:")
                for line in ctx.applied_summary:
                    print(f"  {line}")
            else:
                print("  WARNING: No kernel replacements could be applied to this model.")
                print("  The model may not contain modules matching the optimized kernel types.")

            opt_output, opt_latency = benchmark_model(
                patched_model, model_input, WARMUP_RUNS, TIMED_RUNS
            )
            opt_tensor = extract_tensor(opt_output)
            opt_shape_str = str(list(opt_tensor.shape))
            print(f"  Output shape: {opt_shape_str}")
            print(f"  Median latency: {opt_latency:.1f} ms")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\nERROR: GPU out of memory during optimized run.")
            print("  The optimized kernels may use more memory than expected.")
            torch.cuda.empty_cache()
            sys.exit(1)
        else:
            print(f"\nERROR: Optimized run failed: {e}")
            traceback.print_exc()
            sys.exit(1)
    except Exception as e:
        print(f"\nERROR: Optimized run failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    print()

    # -----------------------------------------------------------------------
    # Step 6: Compare outputs
    # -----------------------------------------------------------------------
    print("Step 6: Comparing outputs...")
    comp = compare_outputs(ref_tensor, opt_tensor, dtype, args.atol, args.rtol)
    print(f"  correctness: {comp['correctness']}")
    print(f"  max_abs_error: {comp.get('max_abs_error', 0):.2e}")
    print(f"  mean_abs_error: {comp.get('mean_abs_error', 0):.2e}")
    if comp.get("reason"):
        print(f"  reason: {comp['reason']}")
    print()

    # -----------------------------------------------------------------------
    # Step 6b: Diagnose failures if requested
    # -----------------------------------------------------------------------
    diagnose_results = None
    if args.diagnose and comp["correctness"] == "FAIL":
        print("Step 6b: Diagnosing failure (testing each kernel individually)...")
        diagnose_results = diagnose_kernel_failures(
            model, model_input, ref_tensor, replacements, dtype
        )
        print()

    # -----------------------------------------------------------------------
    # Step 7: Build and display final report
    # -----------------------------------------------------------------------
    speedup = ref_latency / opt_latency if opt_latency > 0 else 0.0

    result = VerificationResult(
        model_name=model_name if args.pretrained else args.class_name,
        input_shape=args.input_shape,
        dtype_str=args.dtype,
        gpu_name=gpu_name,
        ref_output_shape=ref_shape_str,
        ref_latency_ms=ref_latency,
        opt_output_shape=opt_shape_str,
        opt_latency_ms=opt_latency,
        kernels_replaced=[
            {
                "type": r.kernel_type,
                "rank": r.rank,
                "speedup": r.speedup,
                "path": r.optimized_path,
            }
            for r in replacements
            if r.module_fn is not None
        ],
        correctness=comp["correctness"],
        max_abs_error=comp.get("max_abs_error", 0.0),
        mean_abs_error=comp.get("mean_abs_error", 0.0),
        has_nan=comp.get("opt_has_nan", False),
        has_inf=comp.get("opt_has_inf", False),
        end_to_end_speedup=speedup,
    )

    report = format_report(result, diagnose_results)
    print(report)

    # Save JSON if requested
    if args.json:
        json_path = os.path.abspath(args.json)
        save_verification_json(result, json_path)
        print(f"Results saved to: {json_path}")

    # Default: save to workspace
    default_json = os.path.join(WORKSPACE_DIR, "verification_result.json")
    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    save_verification_json(result, default_json)
    print(f"Results saved to: {default_json}")

    # Exit code: 0 for PASS, 1 for FAIL
    if result.correctness != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
