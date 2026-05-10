# Autokernel Static Audit (Phase A.1, 2026-05-10)

AST-based scan of every `_*Replacement` class in `autokernel/_patterns.py`. Tags each `forward()` call-site as SAFE (goes through autograd) or UNSAFE (raw pybind → severs gradient flow).

Source: `autokernel/_patterns.py`; generator: `scripts/audit_autokernel_replacements.py`.

## Summary: replacement class verdicts

| Class | Line | Overall | Custom ops referenced | Call count |
|---|---:|:---:|---|---:|
| `_RMSNormReplacement` | 148 | **CONDITIONAL-SAFE** | `rmsnorm` | 3 |
| `_LayerNormReplacement` | 181 | **SAFE** | _(none)_ | 1 |
| `_SiluGateMulReplacement` | 206 | **SAFE** | `silu_gate_mul` | 2 |
| `_FusedQKVAttentionReplacement` | 236 | **CONDITIONAL-SAFE** | `rotary_emb_fp32` | 4 |
| `_FusedResidualRMSNormBlockReplacement` | 380 | **SAFE** | `fused_res_rmsnorm` | 1 |
| `_FusedGriffinBlockReplacement` | 418 | **SAFE** | _(none)_ | 0 |
| `_FusedSwiGLUReplacement` | 600 | **SAFE** | `silu_gate_mul` | 2 |

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

### `_LayerNormReplacement` (line 181): **SAFE**

| Line | Verdict | Source |
|---:|:---:|---|
| 201 | **SAFE** | `F.layer_norm(` |

### `_SiluGateMulReplacement` (line 206): **SAFE**

| Line | Verdict | Source |
|---:|:---:|---|
| 230 | **SAFE** | `self._autograd_op(gate.contiguous(), up.contiguous())` |
| 232 | **SAFE** | `F.silu(gate)` |

### `_FusedQKVAttentionReplacement` (line 236): **CONDITIONAL-SAFE**

| Line | Verdict | Source |
|---:|:---:|---|
| 309 | **SAFE** | `self._autograd_rotary(q, cos, sin)` |
| 310 | **SAFE** | `self._autograd_rotary(k, cos, sin)` |
| 312 | **UNSAFE** | `self.rotary_fn(q, cos, sin)` |
| 313 | **UNSAFE** | `self.rotary_fn(k, cos, sin)` |

### `_FusedResidualRMSNormBlockReplacement` (line 380): **SAFE**

| Line | Verdict | Source |
|---:|:---:|---|
| 404 | **SAFE** | `self._autograd_dual(` |

### `_FusedGriffinBlockReplacement` (line 418): **SAFE**

_(no classifiable calls in forward)_

### `_FusedSwiGLUReplacement` (line 600): **SAFE**

| Line | Verdict | Source |
|---:|:---:|---|
| 635 | **SAFE** | `self._autograd_op(gate.contiguous(), up.contiguous())` |
| 638 | **SAFE** | `F.silu(gate)` |

## Conclusions

- **UNSAFE replacements (require Phase B fix):** 0
- **UNKNOWN replacements (require manual review):** 0
- **Total registered custom ops with autograd:** 8 / 8

_A UNSAFE replacement calls raw pybind (or equivalent) inside forward(), which returns a tensor with `grad_fn=None`. Under training mode, this severs gradient flow to upstream parameters. See `docs/perf/autokernel-deep-analysis.md` for mechanism._
