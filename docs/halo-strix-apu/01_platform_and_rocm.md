# 01 — Platform, ROCm, and the gfx1151 Reality

## 1. The correct mental model of Strix Halo

Your Ryzen AI Max+ 395 system is best thought of as three separate compute domains inside one platform:

- **CPU**
- **RDNA 3.5 iGPU** (`gfx1151`)
- **XDNA 2 NPU**

For LLM training in PyTorch/ROCm, the relevant domain is primarily the **GPU**, not the NPU.

The recurring confusion in our discussion was the phrase “matrix multiplication chip.” That phrase is not very useful here. There is no separate user-visible “matmul chip” that training code targets directly. The meaningful question is:

> Does gfx1151 expose fast and mature matrix backends for the exact kernel path I care about?

The answer is:

- **yes**, for standard GEMM paths through ROCm libraries
- **less reliably**, for some specialized low-level or experimental paths

## 2. Official ROCm support status

AMD’s ROCm Linux support matrices list **gfx1151** and specifically include **AMD Ryzen AI Max+ 395** in the supported hardware list for ROCm 7.2.1. They also list **PyTorch 2.9.1 + ROCm 7.2.1 + Python 3.12** as officially supported, with FP16 explicitly validated.  
Source: AMD ROCm Linux support matrix  
<https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityryz/native_linux/native_linux_compatibility.html>

That matters because it rules out the simplistic idea that your APU is outside the supported ROCm world. It is supported.

## 3. Where GEMM performance actually comes from on this platform

For dense matrix multiplication, the most important performance components in your software stack are:

- **rocBLAS**
- **hipBLASLt**
- **Tensile**
- optionally **PyTorch TunableOp** to choose among GEMM kernels

AMD’s model-acceleration docs say ROCm PyTorch can automatically choose the best-performing GEMM kernels from **rocBLAS** and **hipBLASLt** through **TunableOp**, and that this machinery can substitute `torch.nn.functional.linear(...)` with the best kernel discovered during profiling or warm-up.  
Source: AMD model acceleration libraries doc  
<https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/model-acceleration-libraries.html>

The practical conclusion is straightforward:

> On gfx1151, the safest bet is to shape your hot path so that PyTorch can hand it to rocBLAS / hipBLASLt.

## 4. AOTriton on gfx1151 is still not a fully boring, mature path

AMD’s PyTorch compatibility docs say **AOTriton 0.10b** has:

- **official support** for `gfx950` and `gfx1201`
- **experimental support** for `gfx1101`, `gfx1151`, `gfx1150`, and `gfx1200`

Source: AMD PyTorch compatibility doc  
<https://rocm.docs.amd.com/en/latest/compatibility/ml-compatibility/pytorch-compatibility.html>

That single line explains a lot of the behavior you have seen. It means that when your optimization depends on Triton/AOTriton maturity, you are on a target where the stack itself still labels support as experimental.

So when a kernel path behaves unpredictably, underperforms, or fails on gfx1151, that does **not** automatically imply a silicon limitation. It can simply mean the software path is not mature.

## 5. rocWMMA: “supported” does not automatically mean “mature fast path everywhere”

The current rocWMMA repository states that the library supports RDNA-class GPUs including:

- `gfx1100`
- `gfx1101`
- `gfx1102`
- `gfx1151`
- `gfx1200`
- `gfx1201`

and its listed build targets also include `gfx1151`.  
Source: rocWMMA repository  
<https://github.com/ROCm/rocWMMA>

This is important because it means the story is **not** “rocWMMA completely excludes gfx1151.”

But you should still be careful with the interpretation:

- being in a support/build list does **not** guarantee first-class performance for your exact workload
- it does **not** mean every higher-level stack that depends on WMMA/MFMA-like capabilities is equally mature on gfx1151
- it does **not** mean your own custom kernels will outperform rocBLAS-backed GEMMs

The right engineering stance is:

> rocWMMA support is evidence that gfx1151 is not fundamentally missing matrix acceleration features, but real-world maturity and coverage can still lag.

## 6. Why the llama.cpp issue matters

The llama.cpp issue you linked is useful because it provides a real-world symptom on the same architecture. In that issue, the HIP backend on Strix Halo / `gfx1151` performed much worse than expected and was substantially behind the Vulkan backend for prompt processing in the benchmark shown.  
Source: llama.cpp issue #13565  
<https://github.com/ggml-org/llama.cpp/issues/13565>

That issue does **not** prove that gfx1151 cannot do matrix multiplication.

What it does suggest is this:

- some HIP / ROCm code paths on gfx1151 were weaker than expected in at least that workload
- backend maturity and kernel selection matter a lot
- “supported” and “fast” are not the same thing

This lines up with your own finding that your hand optimization attempts for matmuls were worse than PyTorch’s rocBLAS-backed path.

## 7. The useful interpretation of your matmul experiments

Your observation was:

- when you tried to optimize matmuls yourself, you could not beat PyTorch
- PyTorch was faster because it was using rocBLAS

That is a very plausible and useful conclusion.

The practical meaning is:

- **do not assume a custom low-level matmul kernel will beat rocBLAS on gfx1151**
- if the architecture is friendly to standard dense GEMM, let the library win
- spend your time shaping the graph so the library can see large friendly GEMMs

Put differently:

> On this target, the optimization game is often not “beat rocBLAS,” but “present the workload to rocBLAS in the most favorable form.”

## 8. The actual engineering conclusion

### Wrong conclusion
“RDNA 3.5 does not have a matmul chip, so matmul cannot be optimized.”

### Better conclusion
“gfx1151 supports standard GEMM through ROCm libraries, but some specialized fast paths and compiler/kernel stacks are still weaker or less mature than on CUDA or on other AMD targets.”

That framing leads to the right actions:

- rely more on library GEMM
- restructure custom models around larger dense ops
- test rocBLAS vs hipBLASLt selection
- treat specialized custom kernels as something to prove with profiling, not assume

## 9. What this means for future optimization work

If you keep optimizing on Halo Strix, the best order of operations is:

1. **Standardize the hot math**  
   Prefer `nn.Linear`, `F.linear`, `torch.mm`, `torch.matmul`, `torch.bmm`, `torch.addmm`.

2. **Use TunableOp and backend selection**  
   Let ROCm explore rocBLAS / hipBLASLt choices where possible.

3. **Profile first, optimize second**  
   If a custom kernel does not clearly beat the library path, stop investing in it.

4. **Accept that “support” and “peak maturity” are different**  
   gfx1151 is supported; that does not mean every advanced path is fully optimized yet.

## References

- AMD ROCm Linux support matrix:  
  <https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityryz/native_linux/native_linux_compatibility.html>
- AMD PyTorch compatibility doc:  
  <https://rocm.docs.amd.com/en/latest/compatibility/ml-compatibility/pytorch-compatibility.html>
- AMD model acceleration libraries doc:  
  <https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/model-acceleration-libraries.html>
- AMD llama.cpp on ROCm install page:  
  <https://rocm.docs.amd.com/projects/llama-cpp/en/docs-26.02/install/llama-cpp-install.html>
- rocWMMA repository:  
  <https://github.com/ROCm/rocWMMA>
- llama.cpp issue #13565:  
  <https://github.com/ggml-org/llama.cpp/issues/13565>
