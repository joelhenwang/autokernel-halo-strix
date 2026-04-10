"""
Benchmark flash_attn vs SDPA vs Hybrid: isolated forward and backward timing.

Measures forward-only, backward-only, and fwd+bwd for each attention backend
to understand exactly where time is spent.

Usage:
    python scripts/bench_attention_backward.py
"""

import os
import time
import torch
import torch.nn.functional as F


def timer(fn, warmup=5, iters=50, label=""):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    elapsed = (time.time() - t0) / iters * 1000
    print(f"  {label}: {elapsed:.2f}ms")
    return elapsed


def make_tensors(B, H, T, D, layout="btHd", requires_grad=True):
    """Create q, k, v tensors. layout='btHd' for flash_attn, 'bHtd' for SDPA."""
    if layout == "btHd":
        q = torch.randn(B, T, H, D, device="cuda", dtype=torch.float16, requires_grad=requires_grad)
        k = torch.randn(B, T, H, D, device="cuda", dtype=torch.float16, requires_grad=requires_grad)
        v = torch.randn(B, T, H, D, device="cuda", dtype=torch.float16, requires_grad=requires_grad)
    else:
        q = torch.randn(B, H, T, D, device="cuda", dtype=torch.float16, requires_grad=requires_grad)
        k = torch.randn(B, H, T, D, device="cuda", dtype=torch.float16, requires_grad=requires_grad)
        v = torch.randn(B, H, T, D, device="cuda", dtype=torch.float16, requires_grad=requires_grad)
    return q, k, v


def bench_sdpa(B, H, T, D):
    print(f"\n--- SDPA (B={B}, H={H}, T={T}, D={D}) ---")
    q, k, v = make_tensors(B, H, T, D, layout="bHtd")
    grad = torch.randn(B, H, T, D, device="cuda", dtype=torch.float16)

    # Forward only
    q2, k2, v2 = make_tensors(B, H, T, D, layout="bHtd", requires_grad=False)
    fwd = timer(lambda: F.scaled_dot_product_attention(q2, k2, v2, is_causal=True), label="forward")

    # Backward only (pre-compute forward, time just backward)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.cuda.synchronize()

    def bwd_only():
        out.backward(grad, retain_graph=True)
        q.grad = k.grad = v.grad = None

    bwd = timer(bwd_only, label="backward")

    # Fwd+Bwd
    def fwd_bwd():
        q3, k3, v3 = make_tensors(B, H, T, D, layout="bHtd")
        o = F.scaled_dot_product_attention(q3, k3, v3, is_causal=True)
        o.sum().backward()

    fwd_bwd = timer(fwd_bwd, label="fwd+bwd")

    return {"fwd": fwd, "bwd": bwd, "fwd_bwd": fwd_bwd}


def bench_flash_attn(B, H, T, D):
    print(f"\n--- flash_attn aiter/Triton (B={B}, H={H}, T={T}, D={D}) ---")
    try:
        os.environ["FLASH_ATTENTION_TRITON_AMD_ENABLE"] = "TRUE"
        from flash_attn import flash_attn_func
    except ImportError:
        print("  SKIP: flash_attn not installed")
        return None

    q, k, v = make_tensors(B, H, T, D, layout="btHd")
    grad = torch.randn(B, T, H, D, device="cuda", dtype=torch.float16)

    # Forward only
    q2, k2, v2 = make_tensors(B, H, T, D, layout="btHd", requires_grad=False)
    fwd = timer(lambda: flash_attn_func(q2, k2, v2, causal=True), label="forward")

    # Backward only
    out = flash_attn_func(q, k, v, causal=True)
    torch.cuda.synchronize()

    def bwd_only():
        out.backward(grad, retain_graph=True)
        q.grad = k.grad = v.grad = None

    bwd = timer(bwd_only, label="backward")

    # Fwd+Bwd
    def fwd_bwd():
        q3, k3, v3 = make_tensors(B, H, T, D, layout="btHd")
        o = flash_attn_func(q3, k3, v3, causal=True)
        o.sum().backward()

    fwd_bwd = timer(fwd_bwd, label="fwd+bwd")

    return {"fwd": fwd, "bwd": bwd, "fwd_bwd": fwd_bwd}


def bench_hybrid(B, H, T, D):
    print(f"\n--- Hybrid: flash_attn fwd + SDPA bwd (B={B}, H={H}, T={T}, D={D}) ---")
    try:
        os.environ["FLASH_ATTENTION_TRITON_AMD_ENABLE"] = "TRUE"
        from flash_attn import flash_attn_func
    except ImportError:
        print("  SKIP: flash_attn not installed")
        return None

    class HybridFlashSDPA(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k, v, causal=True):
            # flash_attn forward (fast: 0.25ms)
            out = flash_attn_func(q, k, v, causal=causal)
            ctx.save_for_backward(q, k, v)
            ctx.causal = causal
            return out

        @staticmethod
        def backward(ctx, grad_output):
            q, k, v = ctx.saved_tensors
            # SDPA backward (fast: ~3ms) — recompute attention with SDPA
            q_s = q.transpose(1, 2).detach().requires_grad_(True)
            k_s = k.transpose(1, 2).detach().requires_grad_(True)
            v_s = v.transpose(1, 2).detach().requires_grad_(True)
            with torch.enable_grad():
                o = F.scaled_dot_product_attention(q_s, k_s, v_s, is_causal=ctx.causal)
                o.backward(grad_output.transpose(1, 2))
            return q_s.grad.transpose(1, 2), k_s.grad.transpose(1, 2), v_s.grad.transpose(1, 2), None

    def hybrid_fn(q, k, v):
        return HybridFlashSDPA.apply(q, k, v, True)

    # Forward only (should be same as flash_attn)
    q2, k2, v2 = make_tensors(B, H, T, D, layout="btHd", requires_grad=False)
    fwd = timer(lambda: flash_attn_func(q2, k2, v2, causal=True), label="forward (flash)")

    # Backward only
    q, k, v = make_tensors(B, H, T, D, layout="btHd")
    out = hybrid_fn(q, k, v)
    grad = torch.randn_like(out)
    torch.cuda.synchronize()

    def bwd_only():
        out.backward(grad, retain_graph=True)
        q.grad = k.grad = v.grad = None

    bwd = timer(bwd_only, label="backward (SDPA recompute)")

    # Fwd+Bwd
    def fwd_bwd():
        q3, k3, v3 = make_tensors(B, H, T, D, layout="btHd")
        o = hybrid_fn(q3, k3, v3)
        o.sum().backward()

    fwd_bwd = timer(fwd_bwd, label="fwd+bwd")

    # Correctness: compare gradients with pure SDPA
    q_h, k_h, v_h = make_tensors(B, H, T, D, layout="btHd")
    o_h = hybrid_fn(q_h, k_h, v_h)
    o_h.sum().backward()
    dq_hybrid = q_h.grad.clone()

    q_s = q_h.data.transpose(1, 2).requires_grad_(True)
    k_s = k_h.data.transpose(1, 2).requires_grad_(True)
    v_s = v_h.data.transpose(1, 2).requires_grad_(True)
    o_s = F.scaled_dot_product_attention(q_s, k_s, v_s, is_causal=True)
    o_s.sum().backward()
    dq_sdpa = q_s.grad.transpose(1, 2)

    diff = (dq_hybrid.float() - dq_sdpa.float()).abs()
    print(f"  Grad diff (hybrid vs SDPA): max={diff.max().item():.6f}, mean={diff.mean().item():.6f}")

    return {"fwd": fwd, "bwd": bwd, "fwd_bwd": fwd_bwd}


def bench_hybrid_v2(B, H, T, D):
    """Approach B2: Zero-recompute hybrid — flash fwd + SDPA aten bwd with shared logsumexp."""
    print(f"\n--- Hybrid v2: zero-recompute (B={B}, H={H}, T={T}, D={D}) ---")
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from kernels.hip.hybrid_attention import hybrid_flash_sdpa_attention
    except Exception as e:
        print(f"  SKIP: hybrid_attention import failed ({e})")
        return None

    q2, k2, v2 = make_tensors(B, H, T, D, layout="btHd", requires_grad=False)
    fwd = timer(lambda: hybrid_flash_sdpa_attention(q2, k2, v2, causal=True), label="forward")

    q, k, v = make_tensors(B, H, T, D, layout="btHd")
    out = hybrid_flash_sdpa_attention(q, k, v, causal=True)
    grad = torch.randn_like(out)
    torch.cuda.synchronize()

    def bwd_only():
        out.backward(grad, retain_graph=True)
        q.grad = k.grad = v.grad = None

    bwd = timer(bwd_only, label="backward (SDPA aten, no recompute)")

    def fwd_bwd():
        q3, k3, v3 = make_tensors(B, H, T, D, layout="btHd")
        o = hybrid_flash_sdpa_attention(q3, k3, v3, causal=True)
        o.sum().backward()

    fwd_bwd = timer(fwd_bwd, label="fwd+bwd")

    # Correctness
    q_h, k_h, v_h = make_tensors(B, H, T, D, layout="btHd")
    o_h = hybrid_flash_sdpa_attention(q_h, k_h, v_h, causal=True)
    o_h.sum().backward()
    dq_v2 = q_h.grad.clone()

    q_s = q_h.data.transpose(1, 2).requires_grad_(True)
    k_s = k_h.data.transpose(1, 2).requires_grad_(True)
    v_s = v_h.data.transpose(1, 2).requires_grad_(True)
    o_s = F.scaled_dot_product_attention(q_s, k_s, v_s, is_causal=True)
    o_s.sum().backward()
    dq_sdpa = q_s.grad.transpose(1, 2)

    diff = (dq_v2.float() - dq_sdpa.float()).abs()
    print(f"  Grad diff (v2 vs SDPA): max={diff.max().item():.6f}, mean={diff.mean().item():.6f}")

    return {"fwd": fwd, "bwd": bwd, "fwd_bwd": fwd_bwd}


def main():
    print(f"PyTorch: {torch.__version__}")
    print(f"HIP: {torch.version.hip}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    B, H, T, D = 8, 8, 512, 128

    results = {}
    results["sdpa"] = bench_sdpa(B, H, T, D)
    results["flash_attn"] = bench_flash_attn(B, H, T, D)
    results["hybrid_v1"] = bench_hybrid(B, H, T, D)
    results["hybrid_v2"] = bench_hybrid_v2(B, H, T, D)

    print("\n" + "=" * 60)
    print("SUMMARY (ms)")
    print("=" * 60)
    print(f"{'Backend':<25} {'Forward':>8} {'Backward':>8} {'Fwd+Bwd':>8}")
    print("-" * 60)
    for name, r in results.items():
        if r:
            print(f"{name:<25} {r['fwd']:>8.2f} {r['bwd']:>8.2f} {r['fwd_bwd']:>8.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
