"""
AutoKernel — One-liner GPU kernel optimization for PyTorch models on AMD ROCm.

Usage:
    import autokernel

    model = autokernel.optimize(model)                    # auto-detect & apply all
    model = autokernel.optimize(model, compile=True)      # + torch.compile fusion
    print(autokernel.report(model))                       # inspect what was applied
    autokernel.restore(model)                             # revert to original
    autokernel.list_patterns()                            # available optimizations
"""

from autokernel._registry import list_patterns, optimize, report, restore

__all__ = ["optimize", "report", "restore", "list_patterns"]
