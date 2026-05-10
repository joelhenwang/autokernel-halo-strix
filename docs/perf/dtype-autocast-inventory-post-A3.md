# v3 T-0.7 Dtype/Autocast Inventory

**Source:** `scripts/autokernel_dtype_inventory.py` (Tier 0 cheap inventory).

**Schema:** v3 §9.2 (docs/research/autokernel-40k-v3-execution-plan.md).


This is a static inventory of registration metadata. Tier 1 fixes graph-break ops; Tier 2 runs deep parity with actual tensors.


## Headline

- 7/9 ops are Phase-B-fixed (autograd-safe).
- 2/9 ops cause graph breaks (T-3.2 targets).
- 6/9 ops have `register_autocast` rules registered.


## Per-op table

| op | phase_b_fixed | graph_break | has_autograd | has_fake | has_autocast | flags |
|---|---:|---:|---:|---:|---:|---|
| `autokernel::silu_gate_mul` | True | False | True | True | True | --ak-swiglu-fwd,--ak-swiglu-bwd |
| `autokernel::rmsnorm` | True | False | True | True | True | --ak-rmsnorm |
| `autokernel::fused_res_rmsnorm` | True | False | True | True | True | --ak-res-rmsnorm |
| `autokernel::rotary_emb_fp32` | True | False | True | True | True | --ak-rope |
| `autokernel::fused_ple_gate` | True | False | True | True | False | --ak-ple-gate |
| `kernels.hip.fused_rope_gate_mul.kernel_fn` | False | True | False | False | False | --ak-rope-gate,--ak-fix-rope-gate-op |
| `DaoAILab::causal_conv1d_fn` | False | True | None | None | None | --ak-causal-conv,--ak-causal-conv-shim |
| `autokernel::fused_rope_gate_mul` | True | False | True | True | True | --ak-fix-rope-gate-op |
| `autokernel::causal_conv1d` | True | False | True | True | True | --ak-causal-conv-shim |

## Graph-break details

- **`kernels.hip.fused_rope_gate_mul.kernel_fn`**: @torch.compiler.disable on wrapper. Fix target: T-3.2.
- **`DaoAILab::causal_conv1d_fn`**: older custom_op semantics (external extension). Fix target: T-3.2.

## Tier 2 targets

Deep parity (forward/backward rel_err, grad cosine, post-NorMuon update cosine) required for ops in the training path. From v3 §4.2 Tier 2 list:

- `autokernel::silu_gate_mul`
- `autokernel::rmsnorm`
- `autokernel::fused_res_rmsnorm`
- `kernels.hip.fused_rope_gate_mul.kernel_fn`
- `DaoAILab::causal_conv1d_fn`
