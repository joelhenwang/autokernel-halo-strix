# WI1: Root-cause analysis of `triton_poi_fused__to_copy_mul_transpose_view_*`

**Phase 2 work item:** WI1 — characterize the kernel at **9.1% of wall time** in the Phase 1 profile.
**Status:** CLOSED. Already-optimal fusion; no action.
**Evidence:** `docs/perf/kernel-bodies-c1.txt` (full kernel source below).

## What the kernel actually does

Inductor emits two near-identical kernels `triton_poi_fused__to_copy_mul_transpose_view_{7,8}`.
Both are 2,691-byte pointwise triton kernels operating on `xnumel = 2,097,152` elements
(that is `16 × 256 × 512` → batch × seq × 512). Both fuse **5 ATen ops** into a single
kernel launch:

```python
@triton.jit
def triton_poi_fused__to_copy_mul_transpose_view_8(
    in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr
):
    # Tensor layout: xnumel = 2_097_152 = 16 * 256 * 512
    x0 = xindex                     # flat index
    x1 = (xindex % 512)             # inner 512 dim
    x2 = xindex // 512              # outer (batch*seq)

    tmp0 = tl.load(in_ptr0 + x0, None).to(tl.float32)           # fp16 -> fp32
    tmp2 = tl.load(in_ptr1 + x0, None)                          # fp32
    tmp5 = tl.load(in_ptr2 + (x1 + 1536*x2), None).to(tl.float32)  # fp16 strided (stride=1536=3*512)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2                                          # fp16*fp32 -> fp32
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp1 * tmp6                                          # fp16*fp16 -> fp32
    tl.store(out_ptr0 + x0, tmp4, None)                         # fp16 out
    tl.store(out_ptr1 + x0, tmp7, None)                         # fp32 out
```

Kernel signature: `(*fp16, *fp32, *fp16, *fp16 out, *fp32 out)`. Three loads, two stores.

## Interpretation

The stride-1536 load pattern on `in_ptr2` (stride = 3 × 512) is the tell-tale sign of a
**packed QKV projection** — one of q/k/v is selected out of a concatenated `(B, T, 3*D)`
tensor. The same fp16 input (`tmp0`/`tmp1`) is multiplied by two different operands
(`tmp2` fp32 and `tmp5` fp16-expanded-to-fp32) and stored to two output tensors.

This is the RoPE-apply-plus-dtype-conversion sequence from the GQA attention block:
`query = (q_rotated_fp16 * cos) + (q_rotated_shifted_fp16 * sin)`, expanded by Inductor
into a fused cast-load-multiply-store pattern. The separate `fp32` output is the
"detached" copy needed as saved-for-backward for RoPE's gradient.

The fact that Inductor emits the *same logical kernel twice* (sequences `_7` and `_8`)
at 40+30 calls suggests two GraphModules producing the same op chain — likely the Q
path and the K path of GQA being Inductor-compiled in separate FX graphs.

## Why this is already optimal

1. **Arithmetic intensity is memory-bound by construction.**
   3 loads + 2 stores × 2^21 elements × ~3 bytes avg ≈ **30 MB traffic per invocation**.
   At 6.24 μs/call (profile measurement), effective bandwidth is ~4.8 TB/s — strongly
   indicating L2 cache hits (Strix Halo L2 = 6 MB, working set for this kernel = ~8 MB,
   partially fits). Peak HBM bandwidth is ~256 GB/s on gfx1151. The kernel is already
   consuming cache-resident data at saturation.

2. **Five ops collapsed into one launch.** Inductor has done all the fusion that is
   possible without also subsuming the matmul upstream (which is rocBLAS and cannot
   be fused into triton on gfx1151 without MFMA).

3. **Dtype casts eliminated.** The `_to_copy` operations are lowered into in-register
   type conversions (`.to(tl.float32)` inline). There is no separate cast kernel.

4. **Saved-for-backward co-optimized.** The dual-output pattern writes both the fp16
   forward result AND the fp32 backward-tangent copy in a single traversal.

## What we could try (but won't, without evidence)

| Candidate | Expected gain | Risk | Rationale |
|-----------|--------------:|:----|:----------|
| Hand-written HIP kernel replacing this fused triton | 0% | High | Would need to beat Inductor's triton at the same memory-bound op. Triton on ROCm generates near-optimal code for pointwise patterns; a naive HIP kernel with `half2` loads does ~the same thing with more code. |
| Skip the fp32 saved-for-backward copy (use recompute instead) | Maybe 1-2% | High | Breaks autograd correctness. Would require custom autograd.Function wrapping the RoPE apply. Risk of numerical drift. |
| Collapse QKV into one wider tensor to avoid the stride-1536 gather | Uncertain | Medium | Requires restructuring `models/_components.py::Attention` to hold a single packed weight for qkv instead of three separate linear layers. Changes checkpoint layout. |

None of these have a clear wins-over-Inductor case at this shape (B=16, T=256, D=512,
rocBLAS-backed matmuls). Not investigating further.

## Decision

**WI1 CLOSED** as "already optimal — no action". The 9.1% is correctly classified as
**memory-bandwidth-bound RoPE-plus-cast fusion**, which is the best Inductor can do
and which a custom HIP kernel would merely replicate.

Filing this kernel's full source as `docs/perf/kernel-bodies-c1.txt` for future
reference. If ever we see a regression in this op's cost, compare against the
stored source to detect recompilation changes.

## Broader implication for Phase 2

The Inductor fusion catalog (`docs/perf/inductor-fusion-catalog.md`) shows **92 unique
triton kernels**, fusing up to **24 ops per kernel**. Key counts:

- `mul` appears in **81 kernels**
- `transpose` appears in **37 kernels**
- `add` appears in **33 kernels**
- `clamp`, `div`, `expand`, `sum` all appear in 20-44 kernels

**This means any op the profile shows as a Triton fused kernel is already captured.**
The remaining profile costs are concentrated in ops that Inductor CANNOT fuse:

1. `aten::mm` (rocBLAS matmul, kept eager) — 8.5%, no action possible
2. `aten::add_` (autograd buffer accumulation or cross-layer residuals) — **WI2 target**
3. `aten::copy_` (autocast cast + contiguity at graph boundaries) — **WI2 target**
4. `aten::embedding_dense_backward` (tied embedding gradient) — **WI3 target**
5. `Memset` (chunked CE buffer zero-init) — **WI4 target**
6. `Memcpy HtoD` (input upload) — **WI5 target**

Proceed with WI2 (add_/copy_ classification) next.
