"""Pre-compile autokernel HIP kernels for a model before DDP launch.

Run this ONCE per model architecture to warm the kernel cache (~5-10 min).
After this, autokernel.optimize() returns instantly from cache.

Usage:
    python scripts/precompile_kernels.py --model models/vidar_halo.py --class-name VidarHalo
    # Then launch DDP with --optimize-kernels
"""

import argparse
import importlib.util
import sys
import time

import torch


def main():
    parser = argparse.ArgumentParser(description="Pre-compile autokernel HIP kernels")
    parser.add_argument("--model", required=True, help="Path to model .py file")
    parser.add_argument("--class-name", required=True, help="Model class name")
    args = parser.parse_args()

    sys.path.insert(0, ".")

    spec = importlib.util.spec_from_file_location("user_model", args.model)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["user_model"] = mod
    spec.loader.exec_module(mod)
    cls = getattr(mod, args.class_name)

    model = cls()
    model = model.cuda()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.class_name} ({n_params/1e6:.1f}M params)")

    import autokernel
    t0 = time.time()
    model = autokernel.optimize(model, training=True)
    elapsed = time.time() - t0
    print(f"autokernel.optimize() completed in {elapsed:.1f}s")

    report = autokernel.report(model)
    for name, info in report.get("patterns", {}).items():
        print(f"  {name}: {info['modules_replaced']} modules ({info['op_speedup']} speedup)")

    print("\nKernel cache warmed. DDP with --optimize-kernels will use cached kernels.")


if __name__ == "__main__":
    main()
