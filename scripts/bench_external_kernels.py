"""
Benchmark external kernel libraries on gfx1151.

Tests: aule-attention, mamba-ssm, causal-conv1d, flash-linear-attention.
Each section is independent — skips gracefully if a package isn't installed.

Usage:
    python scripts/bench_external_kernels.py
    python scripts/bench_external_kernels.py --test attention
    python scripts/bench_external_kernels.py --test scan
    python scripts/bench_external_kernels.py --test conv
    python scripts/bench_external_kernels.py --test fla
    python scripts/bench_external_kernels.py --test all
"""

import argparse
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def timer(fn, warmup=5, iters=50, label=""):
    """Benchmark a function: warmup, then time `iters` iterations."""
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


def check_close(a, b, name, atol=0.01, rtol=0.01):
    """Check two tensors are close, report max diff."""
    diff = (a.float() - b.float()).abs()
    ok = torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol)
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: max_diff={diff.max().item():.6f}, mean_diff={diff.mean().item():.6f}")
    return ok


# ============================================================================
# Test 1: Flash Attention (AOTriton via flash_attn package)
# ============================================================================

def test_attention():
    print("\n" + "=" * 60)
    print("TEST: Flash Attention (AOTriton backend)")
    print("=" * 60)

    try:
        # Use aiter Triton backend (supports backward on gfx11)
        import os
        os.environ["FLASH_ATTENTION_TRITON_AMD_ENABLE"] = "TRUE"
        from flash_attn import flash_attn_func
    except ImportError as e:
        print(f"  SKIP: flash-attn not installed ({e})")
        return

    # flash_attn_func expects (B, T, H, D) layout, not (B, H, T, D)
    B, H, T, D = 8, 8, 512, 128
    q = torch.randn(B, T, H, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, T, H, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, T, H, D, device="cuda", dtype=torch.float16)

    # flash_attn forward
    y_flash = flash_attn_func(q, k, v, causal=True)

    # SDPA reference (needs B, H, T, D layout)
    q_sdpa = q.transpose(1, 2)
    k_sdpa = k.transpose(1, 2)
    v_sdpa = v.transpose(1, 2)
    y_sdpa = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, is_causal=True)
    y_sdpa = y_sdpa.transpose(1, 2)  # back to (B, T, H, D)

    check_close(y_flash, y_sdpa, "flash_attn vs SDPA")

    # Backward test
    try:
        q_grad = q.clone().detach().requires_grad_(True)
        y = flash_attn_func(q_grad, k, v, causal=True)
        y.sum().backward()
        print(f"  Backward: OK (q.grad shape={q_grad.grad.shape})")
        has_backward = True
    except RuntimeError as e:
        print(f"  Backward: NOT SUPPORTED ({e})")
        has_backward = False

    # Speed — forward only
    q2 = q.clone().detach().requires_grad_(False)
    k2 = k.clone().detach().requires_grad_(False)
    v2 = v.clone().detach().requires_grad_(False)

    timer(lambda: flash_attn_func(q2, k2, v2, causal=True), label="flash_attn forward")
    timer(lambda: F.scaled_dot_product_attention(
        q2.transpose(1, 2), k2.transpose(1, 2), v2.transpose(1, 2), is_causal=True
    ), label="SDPA forward")

    # Forward+backward (only if backward is supported)
    if has_backward:
        def flash_fwd_bwd():
            q3 = q.clone().detach().requires_grad_(True)
            y = flash_attn_func(q3, k, v, causal=True)
            y.sum().backward()

        def sdpa_fwd_bwd():
            q3 = q_sdpa.clone().detach().requires_grad_(True)
            y = F.scaled_dot_product_attention(q3, k_sdpa, v_sdpa, is_causal=True)
            y.sum().backward()

        timer(flash_fwd_bwd, label="flash_attn fwd+bwd")
        timer(sdpa_fwd_bwd, label="SDPA fwd+bwd")
    else:
        print("  (fwd+bwd benchmark skipped — backward not supported on this GPU)")

    print(f"  Shape: ({B},{H},{T},{D}) — GQA-ready")


# ============================================================================
# Test 2: mamba-ssm Selective Scan
# ============================================================================

def test_scan():
    print("\n" + "=" * 60)
    print("TEST: mamba-ssm Selective Scan")
    print("=" * 60)

    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
    except ImportError:
        print("  SKIP: mamba-ssm not installed (run ./scripts/install_mamba_ssm_rocm.sh)")
        return

    B, D, L, N = 8, 384, 512, 64
    u = torch.randn(B, D, L, device="cuda", dtype=torch.float16, requires_grad=True)
    delta = torch.randn(B, D, L, device="cuda", dtype=torch.float16, requires_grad=True)
    A = -torch.rand(D, N, device="cuda", dtype=torch.float32)
    B_mat = torch.randn(B, N, L, device="cuda", dtype=torch.float16)
    C_mat = torch.randn(B, N, L, device="cuda", dtype=torch.float16)
    D_vec = torch.randn(D, device="cuda", dtype=torch.float32)

    # Correctness vs reference
    y_kern = selective_scan_fn(u, delta, A, B_mat, C_mat, D_vec, delta_softplus=True)
    y_ref = selective_scan_ref(u, delta, A, B_mat, C_mat, D_vec, delta_softplus=True)
    check_close(y_kern, y_ref, "mamba kernel vs ref")

    # Backward test
    y_kern.sum().backward()
    print(f"  Backward: OK")

    # Speed
    timer(
        lambda: selective_scan_fn(u, delta, A, B_mat, C_mat, D_vec, delta_softplus=True),
        label="mamba-ssm scan forward",
    )
    timer(
        lambda: selective_scan_ref(u, delta, A, B_mat, C_mat, D_vec, delta_softplus=True),
        label="mamba-ssm ref forward",
    )

    # Compare vs our chunked scan
    try:
        from models.amadeus import selective_scan_chunked
        # Reshape for our API: (B, L, D) not (B, D, L)
        x = u.transpose(1, 2).contiguous()  # (B, L, D)
        dt = F.softplus(delta.transpose(1, 2).contiguous()).clamp(1e-4, 0.5)
        A_log = torch.log(torch.arange(1, D + 1, dtype=torch.float32, device="cuda"))
        B_ours = B_mat[:, :, :].transpose(1, 2).contiguous()[:, :, :N]
        C_ours = C_mat[:, :, :].transpose(1, 2).contiguous()[:, :, :N]
        D_ours = D_vec.float()
        timer(
            lambda: selective_scan_chunked(x, dt, A_log, B_ours, C_ours, D_ours, n_heads=6),
            label="our chunked scan forward",
        )
    except Exception as e:
        print(f"  (skipped our chunked scan comparison: {e})")

    print(f"  Shape: u=({B},{D},{L}), A=({D},{N})")


# ============================================================================
# Test 3: causal-conv1d
# ============================================================================

def test_conv():
    print("\n" + "=" * 60)
    print("TEST: causal-conv1d")
    print("=" * 60)

    try:
        from causal_conv1d import causal_conv1d_fn
    except ImportError:
        print("  SKIP: causal-conv1d not installed (run ./scripts/install_causal_conv1d_rocm.sh)")
        return

    B, D, L, K = 8, 640, 512, 3
    x = torch.randn(B, D, L, device="cuda", dtype=torch.float16, requires_grad=True)
    weight = torch.randn(D, K, device="cuda", dtype=torch.float16, requires_grad=True)

    # causal_conv1d forward
    y_fast = causal_conv1d_fn(x, weight)

    # nn.Conv1d reference
    conv = nn.Conv1d(D, D, K, padding=K - 1, groups=D, bias=False).cuda().half()
    with torch.no_grad():
        conv.weight.copy_(weight.unsqueeze(1))
    x_ref = x.clone().detach()
    y_ref = conv(x_ref)[:, :, :L]

    check_close(y_fast, y_ref, "causal_conv1d vs nn.Conv1d")

    # Backward
    y_fast.sum().backward()
    print(f"  Backward: OK (x.grad shape={x.grad.shape})")

    # Speed
    x2 = torch.randn(B, D, L, device="cuda", dtype=torch.float16)
    w2 = weight.clone().detach()
    timer(lambda: causal_conv1d_fn(x2, w2), label="causal_conv1d forward")
    timer(lambda: conv(x2)[:, :, :L], label="nn.Conv1d forward")

    print(f"  Shape: ({B},{D},{L}), kernel={K}")


# ============================================================================
# Test 4: flash-linear-attention (FLA)
# ============================================================================

def test_fla():
    print("\n" + "=" * 60)
    print("TEST: flash-linear-attention (FLA)")
    print("=" * 60)

    # Test GLA (Gated Linear Attention)
    try:
        from fla.ops.gla import chunk_gla
        print("\n  --- GLA (Gated Linear Attention) ---")
        B, H, T, D = 8, 16, 512, 64
        q = torch.randn(B, H, T, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, T, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, T, D, device="cuda", dtype=torch.float16)
        g = torch.randn(B, H, T, D, device="cuda", dtype=torch.float16)

        o, final_state = chunk_gla(q, k, v, g, output_final_state=True)
        print(f"  Forward OK: output={o.shape}, state={final_state.shape}")

        q2 = q.clone().detach().requires_grad_(True)
        o2, _ = chunk_gla(q2, k, v, g)
        o2.sum().backward()
        print(f"  Backward: OK")

        timer(lambda: chunk_gla(q, k, v, g), label="GLA forward")
    except ImportError:
        print("  SKIP: fla.ops.gla not available")
    except Exception as e:
        print(f"  FAIL: GLA — {e}")

    # Test retention (RetNet)
    try:
        from fla.ops.retention import chunk_retention
        print("\n  --- Retention (RetNet) ---")
        q = torch.randn(B, H, T, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, T, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, T, D, device="cuda", dtype=torch.float16)

        o, final_state = chunk_retention(q, k, v)
        print(f"  Forward OK: output={o.shape}")
        timer(lambda: chunk_retention(q, k, v), label="Retention forward")
    except ImportError:
        print("  SKIP: fla.ops.retention not available")
    except Exception as e:
        print(f"  FAIL: Retention — {e}")

    # Test DeltaNet
    try:
        from fla.ops.delta_rule import chunk_delta_rule
        print("\n  --- DeltaNet ---")
        q = torch.randn(B, H, T, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, T, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, T, D, device="cuda", dtype=torch.float16)
        beta = torch.rand(B, H, T, device="cuda", dtype=torch.float16)

        o, final_state = chunk_delta_rule(q, k, v, beta)
        print(f"  Forward OK: output={o.shape}")
        timer(lambda: chunk_delta_rule(q, k, v, beta), label="DeltaNet forward")
    except ImportError:
        print("  SKIP: fla.ops.delta_rule not available")
    except Exception as e:
        print(f"  FAIL: DeltaNet — {e}")

    # Test HGRN (expects (B, T, D) — no head dim, per-dimension recurrence)
    try:
        from fla.ops.hgrn import chunk_hgrn
        print("\n  --- HGRN ---")
        D_hgrn = H * D  # flatten heads into dim
        x = torch.randn(B, T, D_hgrn, device="cuda", dtype=torch.float16)
        g = torch.randn(B, T, D_hgrn, device="cuda", dtype=torch.float16)

        o, final_state = chunk_hgrn(x, g)
        print(f"  Forward OK: output={o.shape}")
        timer(lambda: chunk_hgrn(x, g), label="HGRN forward")
    except ImportError:
        print("  SKIP: fla.ops.hgrn not available")
    except Exception as e:
        print(f"  FAIL: HGRN — {e}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Benchmark external kernels on gfx1151")
    parser.add_argument(
        "--test",
        choices=["attention", "scan", "conv", "fla", "all"],
        default="all",
        help="Which test to run",
    )
    args = parser.parse_args()

    print(f"PyTorch: {torch.__version__}")
    print(f"HIP: {torch.version.hip}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: No GPU available!")
        sys.exit(1)

    if args.test in ("attention", "all"):
        test_attention()
    if args.test in ("scan", "all"):
        test_scan()
    if args.test in ("conv", "all"):
        test_conv()
    if args.test in ("fla", "all"):
        test_fla()

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
