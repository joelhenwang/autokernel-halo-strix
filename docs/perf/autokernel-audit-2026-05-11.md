# Autokernel Audit Phase A (2026-05-11)

Full 14-model × 3-config diagnostic truth matrix from `scripts/audit_phase_a3_batch.sh`. Each probe ran 50 opt steps with `--diag-frozen-params`, recording every parameter's `.grad.norm()` on every step.

Configurations:

- **V0**: baseline (no `--optimize-kernels`)
- **V1**: `--optimize-kernels` (all patterns active)
- **V3**: `--optimize-kernels --autokernel-exclude fused_silu_gate_mul`

## Summary: newly-frozen params vs V0 baseline

| Model | V1 newly frozen | V3 newly frozen | V1 total frozen | V3 total frozen | Verdict |
|---|---:|---:|---:|---:|---|
| `odin_flat` (120p) | 0 | 0 | 1 | 1 | clean |
| `odin_flat_30m` (72p) | 0 | 0 | 1 | 1 | clean |
| `odin_flat_ablation` (70p) | 0 | 0 | 1 | 1 | clean |
| `odin_flat_mini` (0p) | 0 | 0 | 0 | 0 | clean |
| `odin_halo` (61p) | 0 | 0 | 2 | 2 | clean |
| `odin_halo_ablation` (61p) | 0 | 0 | 2 | 2 | clean |
| `odin_halo_mini` (0p) | 0 | 0 | 0 | 0 | clean |

## Per-model detail

### `odin_flat`

- **V0**: 119 always_finite, 1 always_none, 0 always_zero, 0 occasionally_finite
- **V1**: 1 always_finite, 1 always_none, 0 always_zero, 118 occasionally_finite
  - frozen examples: `module.layers.6._orig_mod.attn.v_res_scale`
- **V3**: 1 always_finite, 1 always_none, 0 always_zero, 118 occasionally_finite
  - frozen examples: `module.layers.6._orig_mod.attn.v_res_scale`

### `odin_flat_30m`

- **V0**: 71 always_finite, 1 always_none, 0 always_zero, 0 occasionally_finite
- **V1**: 1 always_finite, 1 always_none, 0 always_zero, 70 occasionally_finite
  - frozen examples: `module.layers.3._orig_mod.attn.v_res_scale`
- **V3**: 71 always_finite, 1 always_none, 0 always_zero, 0 occasionally_finite
  - frozen examples: `module.layers.3._orig_mod.attn.v_res_scale`

### `odin_flat_ablation`

- **V0**: 69 always_finite, 1 always_none, 0 always_zero, 0 occasionally_finite
- **V1**: 1 always_finite, 1 always_none, 0 always_zero, 68 occasionally_finite
  - frozen examples: `module.layers.4._orig_mod.attn.v_res_scale`
- **V3**: 69 always_finite, 1 always_none, 0 always_zero, 0 occasionally_finite
  - frozen examples: `module.layers.4._orig_mod.attn.v_res_scale`

### `odin_flat_mini`

- **V0**: 0 always_finite, 0 always_none, 0 always_zero, 0 occasionally_finite
- **V1**: 0 always_finite, 0 always_none, 0 always_zero, 0 occasionally_finite
- **V3**: 0 always_finite, 0 always_none, 0 always_zero, 0 occasionally_finite

### `odin_halo`

- **V0**: 58 always_finite, 2 always_none, 0 always_zero, 1 occasionally_finite
- **V1**: 59 always_finite, 2 always_none, 0 always_zero, 0 occasionally_finite
  - frozen examples: `module.shared_layers.3._orig_mod.attn.v_res_scale`, `module.shared_layers.3._orig_mod.attn.head_gate`
- **V3**: 59 always_finite, 2 always_none, 0 always_zero, 0 occasionally_finite
  - frozen examples: `module.shared_layers.3._orig_mod.attn.v_res_scale`, `module.shared_layers.3._orig_mod.attn.head_gate`

### `odin_halo_ablation`

- **V0**: 58 always_finite, 2 always_none, 0 always_zero, 1 occasionally_finite
- **V1**: 59 always_finite, 2 always_none, 0 always_zero, 0 occasionally_finite
  - frozen examples: `module.shared_layers.3._orig_mod.attn.v_res_scale`, `module.shared_layers.3._orig_mod.attn.head_gate`
- **V3**: 59 always_finite, 2 always_none, 0 always_zero, 0 occasionally_finite
  - frozen examples: `module.shared_layers.3._orig_mod.attn.v_res_scale`, `module.shared_layers.3._orig_mod.attn.head_gate`

### `odin_halo_mini`

- **V0**: 0 always_finite, 0 always_none, 0 always_zero, 0 occasionally_finite
- **V1**: 0 always_finite, 0 always_none, 0 always_zero, 0 occasionally_finite
- **V3**: 0 always_finite, 0 always_none, 0 always_zero, 0 occasionally_finite

## Phase B fix map

Maps each observed freeze pattern to the Replacement class that caused it:

| Observed freeze | Root cause (from static audit) | Phase B task |
|---|---|---|
| `ffn.w_gate_up.weight` always_none + `ffn_norm.weight` always_zero | `_FusedSwiGLUReplacement` raw kernel_fn | B.1 |
| `norm.weight` always_zero | `_LayerNormReplacement` raw kernel_fn | B.3 |
| upstream of attention output always_zero | `_FusedQKVAttentionReplacement` raw rotary_fn | B.4 |
| `ffn_norm.weight` always_zero when block replaced | `_FusedResidualRMSNormBlockReplacement` kernel_fn_dual | B.4b |
