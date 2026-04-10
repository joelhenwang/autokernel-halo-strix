"""
Patch aiter's flash_attn Triton backward configs for RDNA (gfx1151).

Replaces the conservative single 32x32x32x32 config with multiple configs
for Triton autotuning. Also supports changing BWD_MODE and restoring originals.

Usage:
    python scripts/patch_aiter_bwd_rdna.py                  # Patch RDNA configs (fused mode)
    python scripts/patch_aiter_bwd_rdna.py --mode split      # Patch configs + set split mode
    python scripts/patch_aiter_bwd_rdna.py --mode fused_atomic
    python scripts/patch_aiter_bwd_rdna.py --restore         # Restore originals
"""

import argparse
import os
import shutil
import sys
from pathlib import Path


AITER_DIR = Path(os.environ.get(
    "AITER_DIR",
    os.path.expanduser("~/Desktop/ai_lab/autokernel-halo-strix/aiter")
))

BWD_FILE = AITER_DIR / "aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/bwd.py"
UTILS_FILE = AITER_DIR / "aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/utils.py"

# The original RDNA config block (appears twice: in "off" and "on" mode)
ORIGINAL_RDNA_CAUSAL = '''            causal_configs = [
                triton.Config(
                    {
                        "BLOCK_M1": 32,
                        "BLOCK_N1": 32,
                        "BLOCK_M2": 32,
                        "BLOCK_N2": 32,
                        "BLK_SLICE_FACTOR": 2,
                    },
                    num_stages=1,
                    num_warps=4,
                ),
            ]'''

ORIGINAL_RDNA_NONCAUSAL = '''            noncausal_configs = [
                triton.Config(
                    {
                        "BLOCK_M1": 32,
                        "BLOCK_N1": 32,
                        "BLOCK_M2": 32,
                        "BLOCK_N2": 32,
                        "BLK_SLICE_FACTOR": 2,
                    },
                    num_stages=1,
                    num_warps=4,
                ),
            ]'''

ORIGINAL_RDNA_PREPROCESS = '''            preprocess_configs = [
                triton.Config({"PRE_BLOCK": 32}, num_stages=1, num_warps=4),
            ]'''

# New RDNA configs with multiple options for autotuning
NEW_RDNA_CAUSAL = '''            causal_configs = [
                # Original conservative (32x32)
                triton.Config(
                    {"BLOCK_M1": 32, "BLOCK_N1": 32, "BLOCK_M2": 32, "BLOCK_N2": 32, "BLK_SLICE_FACTOR": 2},
                    num_stages=1, num_warps=4,
                ),
                # Medium asymmetric (32x64 / 64x32)
                triton.Config(
                    {"BLOCK_M1": 32, "BLOCK_N1": 64, "BLOCK_M2": 64, "BLOCK_N2": 32, "BLK_SLICE_FACTOR": 2},
                    num_stages=1, num_warps=4,
                ),
                # Larger symmetric (64x64)
                triton.Config(
                    {"BLOCK_M1": 64, "BLOCK_N1": 64, "BLOCK_M2": 64, "BLOCK_N2": 64, "BLK_SLICE_FACTOR": 2},
                    num_stages=1, num_warps=8,
                ),
                # CDNA-style asymmetric (32x128 / 128x64)
                triton.Config(
                    {"BLOCK_M1": 32, "BLOCK_N1": 128, "BLOCK_M2": 128, "BLOCK_N2": 64, "BLK_SLICE_FACTOR": 2},
                    num_stages=1, num_warps=4,
                ),
            ]'''

NEW_RDNA_NONCAUSAL = '''            noncausal_configs = [
                triton.Config(
                    {"BLOCK_M1": 32, "BLOCK_N1": 32, "BLOCK_M2": 32, "BLOCK_N2": 32, "BLK_SLICE_FACTOR": 2},
                    num_stages=1, num_warps=4,
                ),
                triton.Config(
                    {"BLOCK_M1": 32, "BLOCK_N1": 64, "BLOCK_M2": 64, "BLOCK_N2": 32, "BLK_SLICE_FACTOR": 2},
                    num_stages=1, num_warps=4,
                ),
                triton.Config(
                    {"BLOCK_M1": 64, "BLOCK_N1": 64, "BLOCK_M2": 64, "BLOCK_N2": 64, "BLK_SLICE_FACTOR": 2},
                    num_stages=1, num_warps=8,
                ),
                triton.Config(
                    {"BLOCK_M1": 32, "BLOCK_N1": 128, "BLOCK_M2": 128, "BLOCK_N2": 64, "BLK_SLICE_FACTOR": 2},
                    num_stages=1, num_warps=4,
                ),
            ]'''

NEW_RDNA_PREPROCESS = '''            preprocess_configs = [
                triton.Config({"PRE_BLOCK": 32}, num_stages=1, num_warps=4),
                triton.Config({"PRE_BLOCK": 64}, num_stages=1, num_warps=8),
            ]'''


def patch_bwd_configs():
    """Replace RDNA configs in bwd.py with multi-config autotuning."""
    text = BWD_FILE.read_text()

    # Count replacements for verification
    n_causal = text.count(ORIGINAL_RDNA_CAUSAL)
    n_noncausal = text.count(ORIGINAL_RDNA_NONCAUSAL)
    n_pre = text.count(ORIGINAL_RDNA_PREPROCESS)

    if n_causal == 0 and n_noncausal == 0:
        print("  WARNING: Original RDNA configs not found — already patched or format changed?")
        return False

    text = text.replace(ORIGINAL_RDNA_CAUSAL, NEW_RDNA_CAUSAL)
    text = text.replace(ORIGINAL_RDNA_NONCAUSAL, NEW_RDNA_NONCAUSAL)
    text = text.replace(ORIGINAL_RDNA_PREPROCESS, NEW_RDNA_PREPROCESS)

    BWD_FILE.write_text(text)
    print(f"  Patched bwd.py: {n_causal} causal, {n_noncausal} noncausal, {n_pre} preprocess blocks")
    return True


def patch_bwd_mode(mode: str):
    """Change BWD_MODE in utils.py."""
    text = UTILS_FILE.read_text()
    import re
    new_text = re.sub(
        r'BWD_MODE: Literal\["fused", "fused_atomic", "split"\] = "[^"]*"',
        f'BWD_MODE: Literal["fused", "fused_atomic", "split"] = "{mode}"',
        text,
    )
    if new_text == text:
        print(f"  WARNING: BWD_MODE pattern not found in utils.py")
        return False
    UTILS_FILE.write_text(new_text)
    print(f"  Set BWD_MODE = \"{mode}\"")
    return True


def restore():
    """Restore original files from git."""
    bwd_dir = BWD_FILE.parent
    os.system(f"cd {AITER_DIR} && git checkout -- {BWD_FILE.relative_to(AITER_DIR)} {UTILS_FILE.relative_to(AITER_DIR)}")
    print("  Restored originals from git")


def clear_triton_cache():
    cache_dir = os.path.expanduser("~/.triton/cache")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print("  Cleared ~/.triton/cache/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fused", "split", "fused_atomic"], default="fused",
                        help="BWD_MODE to set")
    parser.add_argument("--restore", action="store_true", help="Restore originals from git")
    args = parser.parse_args()

    if not BWD_FILE.exists():
        print(f"ERROR: {BWD_FILE} not found")
        sys.exit(1)

    if args.restore:
        restore()
        clear_triton_cache()
        return

    print(f"Patching aiter backward for RDNA (gfx1151)...")
    print(f"  BWD_MODE: {args.mode}")

    # Restore first to ensure clean state
    restore()

    # Apply patches
    patch_bwd_configs()
    patch_bwd_mode(args.mode)
    clear_triton_cache()

    print("Done. Run benchmark with: FLASH_ATTENTION_TRITON_AMD_AUTOTUNE=1 python scripts/bench_attention_backward.py")


if __name__ == "__main__":
    main()
