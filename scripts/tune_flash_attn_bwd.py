"""
Tune flash_attn Triton backward kernel for gfx1151 (Strix Halo).

Tests multiple optimization strategies via runtime monkey-patching:
  Option 1: Different backward block sizes (A/B/C/D configs)
  Option 2: Split vs fused backward mode
  Option 5: Autotune sweep (enables RDNA to use full config space)
  Option 6: exp2 check

Usage:
    python scripts/tune_flash_attn_bwd.py
    python scripts/tune_flash_attn_bwd.py --sweep   # enable full autotune sweep (slow)
"""

import argparse
import os
import shutil
import sys
import time

os.environ["FLASH_ATTENTION_TRITON_AMD_ENABLE"] = "TRUE"
# Disable autotune for manual config testing
os.environ["FLASH_ATTENTION_TRITON_AMD_AUTOTUNE"] = "0"

import torch
import torch.nn.functional as F


def timer(fn, warmup=3, iters=20, label=""):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    elapsed = (time.time() - t0) / iters * 1000
    if label:
        print(f"  {label}: {elapsed:.2f}ms")
    return elapsed


def clear_triton_cache():
    """Clear Triton JIT cache to force recompilation with new configs."""
    cache_dir = os.path.expanduser("~/.triton/cache")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)


def make_fwd_bwd_fn(B, H, T, D):
    """Create a flash_attn fwd+bwd benchmark function."""
    from flash_attn import flash_attn_func

    def fwd_bwd():
        q = torch.randn(B, T, H, D, device="cuda", dtype=torch.float16, requires_grad=True)
        k = torch.randn(B, T, H, D, device="cuda", dtype=torch.float16, requires_grad=True)
        v = torch.randn(B, T, H, D, device="cuda", dtype=torch.float16, requires_grad=True)
        o = flash_attn_func(q, k, v, causal=True)
        o.sum().backward()

    return fwd_bwd


def make_sdpa_fwd_bwd_fn(B, H, T, D):
    """Create an SDPA fwd+bwd benchmark function."""
    def fwd_bwd():
        q = torch.randn(B, H, T, D, device="cuda", dtype=torch.float16, requires_grad=True)
        k = torch.randn(B, H, T, D, device="cuda", dtype=torch.float16, requires_grad=True)
        v = torch.randn(B, H, T, D, device="cuda", dtype=torch.float16, requires_grad=True)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o.sum().backward()

    return fwd_bwd


def patch_bwd_config(config_name, config_dict):
    """Monkey-patch aiter's get_bwd_configs to use a custom config for RDNA."""
    import aiter.ops.triton._triton_kernels.flash_attn_triton_amd.bwd as bwd_mod

    original_get_bwd_configs = bwd_mod.get_bwd_configs

    def patched_get_bwd_configs(mode="off"):
        """Override RDNA backward configs."""
        # Call original to get the structure
        configs = original_get_bwd_configs("off")

        # Override with our test config
        for key in config_dict:
            if key in configs:
                configs[key] = config_dict[key]

        return configs

    bwd_mod.get_bwd_configs = patched_get_bwd_configs
    return original_get_bwd_configs


def restore_bwd_config(original_fn):
    """Restore original get_bwd_configs."""
    import aiter.ops.triton._triton_kernels.flash_attn_triton_amd.bwd as bwd_mod
    bwd_mod.get_bwd_configs = original_fn


def patch_bwd_mode(mode):
    """Monkey-patch BWD_MODE in aiter utils."""
    import aiter.ops.triton._triton_kernels.flash_attn_triton_amd.utils as utils_mod
    original = utils_mod.BWD_MODE
    utils_mod.BWD_MODE = mode
    return original


def restore_bwd_mode(original):
    import aiter.ops.triton._triton_kernels.flash_attn_triton_amd.utils as utils_mod
    utils_mod.BWD_MODE = original


def check_exp2_usage():
    """Check if backward kernels use exp2 or exp."""
    import aiter.ops.triton._triton_kernels.flash_attn_triton_amd.bwd as bwd_mod
    import inspect
    source = inspect.getsource(bwd_mod)

    exp2_count = source.count("tl.exp2(") + source.count("tl.math.exp2(")
    exp_count = source.count("tl.exp(") + source.count("tl.math.exp(")

    # Exclude comments
    print(f"\n--- Option 6: exp2 check ---")
    print(f"  tl.exp2 / tl.math.exp2 calls: {exp2_count}")
    print(f"  tl.exp / tl.math.exp calls: {exp_count}")
    if exp2_count > 0 and exp_count == 0:
        print("  Already using exp2 (optimal)")
    elif exp2_count > 0:
        print("  Mixed exp/exp2 usage")
    else:
        print("  Using exp only — exp2 could save cycles")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true", help="Enable full autotune sweep (slow)")
    args = parser.parse_args()

    print(f"PyTorch: {torch.__version__}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    B, H, T, D = 8, 8, 512, 128
    print(f"Shape: B={B}, H={H}, T={T}, D={D}\n")

    # ================================================================
    # Baselines
    # ================================================================
    print("=" * 60)
    print("BASELINES")
    print("=" * 60)

    sdpa_time = timer(make_sdpa_fwd_bwd_fn(B, H, T, D), label="SDPA fwd+bwd")
    flash_default = timer(make_fwd_bwd_fn(B, H, T, D), label="flash_attn default fwd+bwd")

    results = [
        ("SDPA", sdpa_time),
        ("flash_attn default", flash_default),
    ]

    # ================================================================
    # Option 1: Block size configs
    # ================================================================
    print("\n" + "=" * 60)
    print("OPTION 1: Block size configs")
    print("=" * 60)

    # Test configs — each overrides the backward block sizes
    # Format matches what get_bwd_configs returns
    test_configs = {
        "A: CDNA-style (32/128/128/64)": {
            "BLOCK_M1": 32, "BLOCK_N1": 128, "BLOCK_M2": 128, "BLOCK_N2": 64,
            "PRE_BLOCK": 64, "num_warps_pre": 8,
            "num_warps1": 4, "num_warps2": 4,
            "num_stages1": 1, "num_stages2": 1,
        },
        "B: Medium asym (32/64/64/32)": {
            "BLOCK_M1": 32, "BLOCK_N1": 64, "BLOCK_M2": 64, "BLOCK_N2": 32,
            "PRE_BLOCK": 64, "num_warps_pre": 8,
            "num_warps1": 4, "num_warps2": 4,
            "num_stages1": 1, "num_stages2": 1,
        },
        "C: Large sym (64/64/64/64)": {
            "BLOCK_M1": 64, "BLOCK_N1": 64, "BLOCK_M2": 64, "BLOCK_N2": 64,
            "PRE_BLOCK": 64, "num_warps_pre": 8,
            "num_warps1": 8, "num_warps2": 8,
            "num_stages1": 1, "num_stages2": 1,
        },
        "D: Triton ref (32/128/128/32)": {
            "BLOCK_M1": 32, "BLOCK_N1": 128, "BLOCK_M2": 128, "BLOCK_N2": 32,
            "PRE_BLOCK": 128, "num_warps_pre": 8,
            "num_warps1": 4, "num_warps2": 4,
            "num_stages1": 1, "num_stages2": 1,
        },
        "E: Double (64/32/32/64)": {
            "BLOCK_M1": 64, "BLOCK_N1": 32, "BLOCK_M2": 32, "BLOCK_N2": 64,
            "PRE_BLOCK": 64, "num_warps_pre": 4,
            "num_warps1": 4, "num_warps2": 4,
            "num_stages1": 1, "num_stages2": 1,
        },
    }

    for name, config in test_configs.items():
        print(f"\n  Config {name}")
        try:
            clear_triton_cache()
            orig = patch_bwd_config(name, config)
            t = timer(make_fwd_bwd_fn(B, H, T, D), label=f"fwd+bwd")
            results.append((f"Opt1: {name}", t))
            restore_bwd_config(orig)
        except Exception as e:
            print(f"    FAILED: {e}")
            try:
                restore_bwd_config(orig)
            except:
                pass

    # ================================================================
    # Option 2: Split vs Fused mode
    # ================================================================
    print("\n" + "=" * 60)
    print("OPTION 2: Backward mode (fused vs split)")
    print("=" * 60)

    for mode in ["fused", "split", "fused_atomic"]:
        try:
            clear_triton_cache()
            orig_mode = patch_bwd_mode(mode)
            t = timer(make_fwd_bwd_fn(B, H, T, D), label=f"BWD_MODE={mode}")
            results.append((f"Opt2: {mode}", t))
            restore_bwd_mode(orig_mode)
        except Exception as e:
            print(f"  BWD_MODE={mode}: FAILED ({e})")
            try:
                restore_bwd_mode(orig_mode)
            except:
                pass

    # ================================================================
    # Option 6: exp2 check
    # ================================================================
    check_exp2_usage()

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Config':<40} {'Time':>8} {'vs SDPA':>10} {'vs flash':>10}")
    print("-" * 70)
    for name, t in sorted(results, key=lambda x: x[1]):
        vs_sdpa = ((t - sdpa_time) / sdpa_time) * 100
        vs_flash = ((t - flash_default) / flash_default) * 100
        sdpa_marker = "FASTER" if t < sdpa_time else "slower"
        flash_marker = "FASTER" if t < flash_default else "slower"
        print(f"  {name:<38} {t:>6.2f}ms {vs_sdpa:>+7.1f}% {vs_flash:>+7.1f}%")

    best_name, best_time = min(results, key=lambda x: x[1])
    print(f"\n  BEST: {best_name} at {best_time:.2f}ms")
    if best_time < sdpa_time:
        print(f"  BEATS SDPA by {((sdpa_time - best_time) / sdpa_time) * 100:.1f}%!")
    else:
        print(f"  Still {((best_time - sdpa_time) / sdpa_time) * 100:.1f}% slower than SDPA ({sdpa_time:.2f}ms)")


if __name__ == "__main__":
    main()
