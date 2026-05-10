# Autokernel Static Audit (Phase A.1, 2026-05-10)

AST-based scan of every `_*Replacement` class in `autokernel/_patterns.py`. Tags each `forward()` call-site as SAFE (goes through autograd) or UNSAFE (raw pybind → severs gradient flow).

Source: `autokernel/_patterns.py`; generator: `scripts/audit_autokernel_replacements.py`.

## Summary: replacement class verdicts

| Class | Line | Overall | Custom ops referenced | Call count |
|---|---:|:---:|---|---:|
| `_RMSNormReplacement` | 148 | **CONDITIONAL-SAFE** | `rmsnorm` | 3 |
| `_LayerNormReplacement` | 181 | **UNSAFE** | _(none)_ | 2 |
| `_SiluGateMulReplacement` | 196 | **UNSAFE** | _(none)_ | 2 |
| `_FusedQKVAttentionReplacement` | 218 | **UNSAFE** | _(none)_ | 2 |
| `_FusedResidualRMSNormBlockReplacement` | 344 | **UNSAFE** | _(none)_ | 1 |
| `_FusedGriffinBlockReplacement` | 372 | **UNKNOWN** | _(none)_ | 0 |
| `_FusedSwiGLUReplacement` | 554 | **UNSAFE** | _(none)_ | 2 |

## Registered custom ops (`kernels/hip/_torch_ops.py`)

| Op name | Defined | Autograd registered |
|---|---:|:---:|
| `autokernel::fused_gated_conv` | L571 | yes |
| `autokernel::fused_ple_gate` | L486 | yes |
| `autokernel::fused_res_rmsnorm` | L210 | yes |
| `autokernel::griffin_scan` | L430 | yes |
| `autokernel::rmsnorm` | L56 | yes |
| `autokernel::rotary_emb_fp32` | L108 | yes |
| `autokernel::selective_scan` | L274 | yes |
| `autokernel::silu_gate_mul` | L161 | yes |

## Per-class call-site breakdown

### `_RMSNormReplacement` (line 148): **CONDITIONAL-SAFE**

| Line | Verdict | Source |
|---:|:---:|---|
| 176 | **UNSAFE** | `self.kernel_fn(x.view(-1, x.shape[-1]), self.weight).view(orig_shape)` |
| 172 | **SAFE** | `self._autograd_op(flat, self.weight)` |
| 176 | **UNSAFE** | `self.kernel_fn(x.view(-1, x.shape[-1]), self.weight)` |

### `_LayerNormReplacement` (line 181): **UNSAFE**

| Line | Verdict | Source |
|---:|:---:|---|
| 193 | **SAFE** | `F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)` |
| 192 | **UNSAFE** | `self.kernel_fn(x, self.weight, self.bias)` |

### `_SiluGateMulReplacement` (line 196): **UNSAFE**

| Line | Verdict | Source |
|---:|:---:|---|
| 212 | **UNSAFE** | `self.kernel_fn(gate.contiguous(), up.contiguous())` |
| 214 | **SAFE** | `F.silu(gate)` |

### `_FusedQKVAttentionReplacement` (line 218): **UNSAFE**

| Line | Verdict | Source |
|---:|:---:|---|
| 276 | **UNSAFE** | `self.rotary_fn(q, cos, sin)` |
| 277 | **UNSAFE** | `self.rotary_fn(k, cos, sin)` |

### `_FusedResidualRMSNormBlockReplacement` (line 344): **UNSAFE**

| Line | Verdict | Source |
|---:|:---:|---|
| 358 | **UNSAFE** | `self.kernel_fn_dual(` |

### `_FusedGriffinBlockReplacement` (line 372): **UNKNOWN**

_(no classifiable calls in forward)_

### `_FusedSwiGLUReplacement` (line 554): **UNSAFE**

| Line | Verdict | Source |
|---:|:---:|---|
| 585 | **UNSAFE** | `self.kernel_fn(gate.contiguous(), up.contiguous())` |
| 587 | **SAFE** | `F.silu(gate)` |

## Conclusions

- **UNSAFE replacements (require Phase B fix):** 5
  - `_LayerNormReplacement`
  - `_SiluGateMulReplacement`
  - `_FusedQKVAttentionReplacement`
  - `_FusedResidualRMSNormBlockReplacement`
  - `_FusedSwiGLUReplacement`
- **UNKNOWN replacements (require manual review):** 1
  - `_FusedGriffinBlockReplacement`
- **Total registered custom ops with autograd:** 8 / 8

_A UNSAFE replacement calls raw pybind (or equivalent) inside forward(), which returns a tensor with `grad_fn=None`. Under training mode, this severs gradient flow to upstream parameters. See `docs/perf/autokernel-deep-analysis.md` for mechanism._
