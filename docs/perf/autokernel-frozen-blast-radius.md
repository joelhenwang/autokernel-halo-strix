# Autokernel Frozen-Params Blast Radius (Track 3.A, 2026-05-10)

Per-parameter `.grad.norm()` recorded on every optimizer step across three configurations on OdinFlat. Identifies which parameters are frozen (grad=None or grad=0) under each `--optimize-kernels` variant.

Sources:
  - `docs/perf/odinflat-profile-2026-05-10/diag-V0.jsonl`
  - `docs/perf/odinflat-profile-2026-05-10/diag-V1.jsonl`
  - `docs/perf/odinflat-profile-2026-05-10/diag-V3.jsonl`

## Configurations

| Label | Flags | Expected |
|---|---|---|
| V0 | (none) | All params get finite nonzero grads (baseline) |
| V1 | `--optimize-kernels` | `w_gate_up` should be frozen (silu HIP missing autograd) |
| V3 | `--optimize-kernels --autokernel-exclude fused_silu_gate_mul` | All params should get grads if Phase III rmsnorm fix is complete |

## V0: status tally

| Status | Count |
|---|---:|
| always_finite | 119 (99.2%) |
| occasionally_finite | 0 (0.0%) |
| always_zero | 0 (0.0%) |
| always_none | 1 (0.8%) |
| always_zero_or_none | 0 (0.0%) |

### V0: grouped by module

| Module pattern | Param count | Status distribution |
|---|---:|---|
| `module.final_norm.weight` | 1 | always_finite:1 |
| `module.layers.*._orig_mod.attn.head_gate` | 2 | always_finite:2 |
| `module.layers.*._orig_mod.attn.k_scale` | 2 | always_finite:2 |
| `module.layers.*._orig_mod.attn.q_scale` | 2 | always_finite:2 |
| `module.layers.*._orig_mod.attn.v_res_scale` | 2 | always_finite:1; always_none:1 |
| `module.layers.*._orig_mod.attn.wo.weight` | 2 | always_finite:2 |
| `module.layers.*._orig_mod.attn.wqkv.weight` | 2 | always_finite:2 |
| `module.layers.*._orig_mod.conv_bias` | 12 | always_finite:12 |
| `module.layers.*._orig_mod.conv_weight` | 12 | always_finite:12 |
| `module.layers.*._orig_mod.ffn.w_down.weight` | 14 | always_finite:14 |
| `module.layers.*._orig_mod.ffn.w_gate_up.weight` | 14 | always_finite:14 |
| `module.layers.*._orig_mod.ffn_norm.weight` | 14 | always_finite:14 |
| `module.layers.*._orig_mod.out_proj.weight` | 12 | always_finite:12 |
| `module.layers.*._orig_mod.pre_norm.weight` | 14 | always_finite:14 |
| `module.layers.*._orig_mod.proj.weight` | 12 | always_finite:12 |
| `module.lm_head.proj_down.weight` | 1 | always_finite:1 |
| `module.tok_embeddings.embed.weight` | 1 | always_finite:1 |
| `module.tok_embeddings.proj_up.weight` | 1 | always_finite:1 |

## V1: status tally

| Status | Count |
|---|---:|
| always_finite | 91 (75.8%) |
| occasionally_finite | 0 (0.0%) |
| always_zero | 14 (11.7%) |
| always_none | 15 (12.5%) |
| always_zero_or_none | 0 (0.0%) |

### V1: grouped by module

| Module pattern | Param count | Status distribution |
|---|---:|---|
| `module.final_norm.weight` | 1 | always_finite:1 |
| `module.layers.*._orig_mod.attn.head_gate` | 2 | always_finite:2 |
| `module.layers.*._orig_mod.attn.k_scale` | 2 | always_finite:2 |
| `module.layers.*._orig_mod.attn.q_scale` | 2 | always_finite:2 |
| `module.layers.*._orig_mod.attn.v_res_scale` | 2 | always_finite:1; always_none:1 |
| `module.layers.*._orig_mod.attn.wo.weight` | 2 | always_finite:2 |
| `module.layers.*._orig_mod.attn.wqkv.weight` | 2 | always_finite:2 |
| `module.layers.*._orig_mod.conv_bias` | 12 | always_finite:12 |
| `module.layers.*._orig_mod.conv_weight` | 12 | always_finite:12 |
| `module.layers.*._orig_mod.ffn.w_down.weight` | 14 | always_finite:14 |
| `module.layers.*._orig_mod.ffn.w_gate_up.weight` | 14 | always_none:14 |
| `module.layers.*._orig_mod.ffn_norm.weight` | 14 | always_zero:14 |
| `module.layers.*._orig_mod.out_proj.weight` | 12 | always_finite:12 |
| `module.layers.*._orig_mod.pre_norm.weight` | 14 | always_finite:14 |
| `module.layers.*._orig_mod.proj.weight` | 12 | always_finite:12 |
| `module.lm_head.proj_down.weight` | 1 | always_finite:1 |
| `module.tok_embeddings.embed.weight` | 1 | always_finite:1 |
| `module.tok_embeddings.proj_up.weight` | 1 | always_finite:1 |

## V3: status tally

| Status | Count |
|---|---:|
| always_finite | 119 (99.2%) |
| occasionally_finite | 0 (0.0%) |
| always_zero | 0 (0.0%) |
| always_none | 1 (0.8%) |
| always_zero_or_none | 0 (0.0%) |

### V3: grouped by module

| Module pattern | Param count | Status distribution |
|---|---:|---|
| `module.final_norm.weight` | 1 | always_finite:1 |
| `module.layers.*._orig_mod.attn.head_gate` | 2 | always_finite:2 |
| `module.layers.*._orig_mod.attn.k_scale` | 2 | always_finite:2 |
| `module.layers.*._orig_mod.attn.q_scale` | 2 | always_finite:2 |
| `module.layers.*._orig_mod.attn.v_res_scale` | 2 | always_finite:1; always_none:1 |
| `module.layers.*._orig_mod.attn.wo.weight` | 2 | always_finite:2 |
| `module.layers.*._orig_mod.attn.wqkv.weight` | 2 | always_finite:2 |
| `module.layers.*._orig_mod.conv_bias` | 12 | always_finite:12 |
| `module.layers.*._orig_mod.conv_weight` | 12 | always_finite:12 |
| `module.layers.*._orig_mod.ffn.w_down.weight` | 14 | always_finite:14 |
| `module.layers.*._orig_mod.ffn.w_gate_up.weight` | 14 | always_finite:14 |
| `module.layers.*._orig_mod.ffn_norm.weight` | 14 | always_finite:14 |
| `module.layers.*._orig_mod.out_proj.weight` | 12 | always_finite:12 |
| `module.layers.*._orig_mod.pre_norm.weight` | 14 | always_finite:14 |
| `module.layers.*._orig_mod.proj.weight` | 12 | always_finite:12 |
| `module.lm_head.proj_down.weight` | 1 | always_finite:1 |
| `module.tok_embeddings.embed.weight` | 1 | always_finite:1 |
| `module.tok_embeddings.proj_up.weight` | 1 | always_finite:1 |

## Blast-radius comparison

Summary of which parameter groups differ in status between V0 and V1.

| Parameter | V0 status | V1 status |
|---|---|---|
| `module.layers.0._orig_mod.ffn.w_gate_up.weight` | always_finite | always_none |
| `module.layers.0._orig_mod.ffn_norm.weight` | always_finite | always_zero |
| `module.layers.1._orig_mod.ffn.w_gate_up.weight` | always_finite | always_none |
| `module.layers.1._orig_mod.ffn_norm.weight` | always_finite | always_zero |
| `module.layers.10._orig_mod.ffn.w_gate_up.weight` | always_finite | always_none |
| `module.layers.10._orig_mod.ffn_norm.weight` | always_finite | always_zero |
| `module.layers.11._orig_mod.ffn.w_gate_up.weight` | always_finite | always_none |
| `module.layers.11._orig_mod.ffn_norm.weight` | always_finite | always_zero |
| `module.layers.12._orig_mod.ffn.w_gate_up.weight` | always_finite | always_none |
| `module.layers.12._orig_mod.ffn_norm.weight` | always_finite | always_zero |
| `module.layers.13._orig_mod.ffn.w_gate_up.weight` | always_finite | always_none |
| `module.layers.13._orig_mod.ffn_norm.weight` | always_finite | always_zero |
| `module.layers.2._orig_mod.ffn.w_gate_up.weight` | always_finite | always_none |
| `module.layers.2._orig_mod.ffn_norm.weight` | always_finite | always_zero |
| `module.layers.3._orig_mod.ffn.w_gate_up.weight` | always_finite | always_none |
| `module.layers.3._orig_mod.ffn_norm.weight` | always_finite | always_zero |
| `module.layers.4._orig_mod.ffn.w_gate_up.weight` | always_finite | always_none |
| `module.layers.4._orig_mod.ffn_norm.weight` | always_finite | always_zero |
| `module.layers.5._orig_mod.ffn.w_gate_up.weight` | always_finite | always_none |
| `module.layers.5._orig_mod.ffn_norm.weight` | always_finite | always_zero |
| `module.layers.6._orig_mod.ffn.w_gate_up.weight` | always_finite | always_none |
| `module.layers.6._orig_mod.ffn_norm.weight` | always_finite | always_zero |
| `module.layers.7._orig_mod.ffn.w_gate_up.weight` | always_finite | always_none |
| `module.layers.7._orig_mod.ffn_norm.weight` | always_finite | always_zero |
| `module.layers.8._orig_mod.ffn.w_gate_up.weight` | always_finite | always_none |
| `module.layers.8._orig_mod.ffn_norm.weight` | always_finite | always_zero |
| `module.layers.9._orig_mod.ffn.w_gate_up.weight` | always_finite | always_none |
| `module.layers.9._orig_mod.ffn_norm.weight` | always_finite | always_zero |
