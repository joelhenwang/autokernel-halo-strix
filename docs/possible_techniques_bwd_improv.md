---
title: "Possible Techniques for Backward Pass Improvement"
domain: kernels
type: reference
status: active
related:
  - docs/superpowers/specs/2026-04-10-backward-pass-optimization-design.md
  - docs/superpowers/specs/2026-04-10-training-pipeline-optimization-design.md
tags: [%backward, %techniques, %survey]
---

Here’s the landscape for **improving or replacing matrix multiplication in the backward pass of DL training**.

For a linear layer (Y = XW), the expensive backward-pass products are:

* **weight gradient:** (dW = X^\top dY)
* **activation gradient:** (dX = dY W^\top)

So most work falls into one of two goals:

1. **keep those matmuls, but make them much cheaper**, or
2. **approximate/avoid them**, accepting some bias or architectural constraints. ([OpenReview][1])

## What is actually working best today

### 1) IO-aware fused kernels, not new algebra

The biggest practical wins still come from **better kernel design**, especially where the backward pass is bandwidth-bound rather than pure-FLOP-bound. FlashAttention is the canonical example: it does **not remove** backward matmuls, but reorganizes the computation to reduce HBM traffic and avoids storing large intermediates for backward, which is why it speeds up training in practice. Later work on attention backward IO complexity argues this IO-aware style is essentially optimal in the large-cache regime. ([arXiv][2])

What this means in practice:

* If your bottleneck is attention backward, **kernel fusion / tiling / recomputation / IO reduction** is currently more proven than exotic replacements for GEMM.
* For dense MLP backward, vendor-tuned GEMMs are still hard to beat unless you can exploit structure.

### 2) Lower precision for backward GEMMs

A very active line is to keep backward GEMMs but run them in **FP8** or even **FP4**-style formats. Recent work reports transformer designs and training recipes that support **FP8 for all GEMMs in transformer blocks during both forward and backward passes**, with throughput gains while matching BF16-quality training at scale. Other work reports mostly-4-bit training for weights, activations, and gradients over very large token budgets, and NVIDIA’s NVFP4 pretraining work claims FP4-like training can approach FP8-level accuracy with the right format/scaling recipe. ([arXiv][3])

This is currently one of the strongest answers to your question because it improves backward matmul **without changing the learning rule**.

### 3) Structured sparsity and block sparsity

Another practical direction is to make backward products sparse enough that you can use **SpMM** instead of dense GEMM. Recent work on block-wise sparse training, diagonal sparsity, and sparse transformers shows meaningful savings when the sparsity pattern is hardware-friendly enough to survive both forward and backward execution. The catch is that unstructured sparsity usually loses on real hardware; structured sparsity is the version that can actually help wall-clock time. ([OpenReview][4])

The rule here is simple:

* **Sparse math only wins if the kernel and sparsity pattern are co-designed.**
* Otherwise you just trade dense GEMM for worse utilization.

---

## Methods that partially replace backward matmuls

### 4) Low-rank gradient projection

This line assumes the full gradient matrix is overkill and can be handled in a **low-rank subspace**. GaLore projects gradients to reduce optimizer-state memory while keeping full-parameter training, and GaLore 2 improves scalability and integration with modern large-scale training. More recent work on unbiased low-rank projection tries to fix the bias/convergence issues that earlier low-rank projection methods introduced. ([arXiv][5])

Important nuance:

* GaLore-style methods mostly reduce **optimizer-state and training memory**, not the raw backward GEMM cost by themselves.
* But they are relevant because they attack the same bottleneck region and can combine well with low precision or checkpointing.

### 5) Low-rank backprop itself

A newer step beyond low-rank optimizer states is to approximate the **backward multiplications themselves** in a compressed space. INSTANT is a good example: it projects activations and gradients into a low-rank subspace and performs backpropagation computations there, explicitly aiming to reduce both memory and computation during backward. ([OpenReview][1])

This is closer to “replace backward matmul” than GaLore is. The downside is that now you are approximating the actual gradient path more aggressively, so stability and fidelity matter a lot more.

### 6) Approximate matrix multiplication during backprop

Some recent LLM-training work explicitly approximates the activation-gradient multiplication used in backprop rather than computing it exactly. The “QKV Projections Require a Fraction of Their Memory” paper introduces **PAMM** (Point-Approximate Matrix Multiplication), which compresses Q/K/V projection activations heavily and is designed to be composable with FlashAttention. The same paper also situates itself among prior approaches that approximate backward activation-gradient products by sampling rows and columns. ([OpenReview][6])

This is promising because it targets a very specific expensive part of training. But it is still more specialized and less “drop-in universal” than low-precision GEMM.

### 7) Activation compression for backward

CompAct reduces peak memory by compressing activations stored for backward; it is mostly about memory rather than eliminating GEMM, but in practice this matters because memory pressure and recomputation often dominate the overall training budget. Older work on approximate activations for backprop made the same point earlier: exact forward, approximate stored activations for backward, with modest impact on training quality. ([ACL Anthology][7])

This will not “replace” backward matmul, but it can make the whole backward pass materially cheaper or more scalable.

---

## More radical “replacement” ideas

### 8) Replace dense matrices with structured transforms

Instead of a generic dense (W), you can parameterize layers with **structured matrices** that admit fast multiplication, such as butterfly/Monarch-style factorizations. Monarch matrices are proposed as expressive structured matrices for efficient training and inference, and earlier butterfly work shows such parameterizations can learn FFT-like fast transforms or replace dense/pointwise layers in some settings. ([Proceedings of Machine Learning Research][8])

This helps both forward and backward because if the layer structure changes, the backward products inherit the cheaper structure. But:

* it is an **architectural change**, not a simple training trick;
* the tradeoff is between expressivity, implementation complexity, and hardware efficiency.

### 9) Fast matrix multiplication algorithms like Strassen

In principle, you can reduce multiplication counts with Strassen-like algorithms. StrassenNets explicitly learns low-multiplication approximations of DNN matrix multiplications and even recovers Strassen’s (2\times2) algorithm. More recent work continues studying learned fast matrix multiplication algorithms. ([arXiv][9])

Why this is not mainstream for training:

* lower asymptotic multiplication count does **not** automatically mean faster GPU training;
* these methods usually increase additions, data movement, and kernel complexity;
* backward stability and implementation overhead are serious issues.

So this is intellectually interesting, but currently not the first place to look for practical DL backward-pass speedups.

### 10) Backprop-free or backprop-lite learning rules

Feedback Alignment, Direct Feedback Alignment, forward-gradient methods, and related “backprop-free” approaches reduce or remove the need to propagate exact transposed weight products during backward. Classical feedback-alignment work replaces weight transposes with fixed random matrices, and newer work continues to explore DFA/forward-gradient hybrids and approximate-gradient approaches. ([arXiv][10])

This is the clearest way to “replace” backward matmuls conceptually. But for mainstream deep learning, especially large transformers, it is still much less mature than standard backpropagation. Accuracy, scaling, and stability remain the main barriers.

---

## My honest take on the field

If your goal is **practical training speed today**, the ranking is roughly:

1. **IO-aware backward kernels / fused kernels / recomputation design**
2. **Lower precision backward GEMMs (FP8 first, FP4 cautiously)**
3. **Structured sparsity if your hardware can exploit it**
4. **Activation compression / selective approximate backward for memory-bound regimes**
5. **Low-rank backprop approximations**
6. **Structured matrix replacements as an architectural redesign**
7. **Feedback alignment / backprop-free learning rules**
8. **Strassen-like algebraic replacements for mainstream GPU training**

That ordering is based on how close the methods are to current large-scale practice versus how experimental they still are. The strongest “real” improvements in 2024–2026 are still mostly **making GEMM-like backward computation cheaper**, not abolishing it. ([arXiv][2])

---

## If you want to research this seriously, split it into these subtopics

### A. Kernel/system level

Study:

* FlashAttention backward and IO complexity
* fused backward kernels
* recomputation vs activation storage
* dense GEMM vs sparse SpMM crossover points ([arXiv][2])

### B. Numerics

Study:

* FP8 and FP4 training stability
* scaling formats, block scaling, outlier management
* which backward tensors are most precision-sensitive ([arXiv][3])

### C. Gradient compression / approximation

Study:

* GaLore / GaLore 2 / unbiased low-rank projection
* INSTANT
* PAMM and approximate activation-gradient multiplication
* activation compression methods like CompAct ([arXiv][5])

### D. Architectural alternatives

Study:

* Monarch / butterfly / learned fast transforms
* sparse architectures whose backward stays hardware-friendly ([Proceedings of Machine Learning Research][8])

### E. Learning-rule replacements

Study:

* Feedback Alignment / DFA
* forward-gradient and hybrid methods
* why they still lag exact reverse-mode AD at scale ([arXiv][10])

---

## Best concrete starting papers by category

For practical backward-pass improvement:

* **FlashAttention** ([arXiv][2])
* **On Fine-Grained I/O Complexity of Attention Backward** ([OpenReview][11])
* **Towards Fully FP8 GEMM LLM Training at Scale** ([arXiv][3])
* **FP4 All the Way** / **Pretraining LLMs with NVFP4** ([arXiv][12])

For approximate/compressed backward:

* **GaLore** and **GaLore 2** ([arXiv][5])
* **INSTANT** ([OpenReview][13])
* **QKV Projections Require a Fraction of Their Memory / PAMM** ([OpenReview][6])
* **CompAct** ([ACL Anthology][7])

For genuine “replacement” directions:

* **Monarch** / **butterfly transforms** ([Proceedings of Machine Learning Research][8])
* **StrassenNets** ([arXiv][9])
* **Random Backpropagation / Feedback Alignment** ([arXiv][10])

If you want, I can turn this into a **research map** with:

* exact formulas for where matmuls appear in transformer backward,
* which methods reduce **FLOPs** vs **memory** vs **IO**,
* and which ones are realistic on your ROCm Halo Strix setup.

[1]: https://openreview.net/pdf?id=P2q6Y7UweV&utm_source=chatgpt.com "INSTANT: COMPRESSING GRADIENTS AND ACTIVA"
[2]: https://arxiv.org/pdf/2205.14135?utm_source=chatgpt.com "FlashAttention: Fast and Memory-Efficient Exact Attention ..."
[3]: https://arxiv.org/abs/2505.20524?utm_source=chatgpt.com "Towards Fully FP8 GEMM LLM Training at Scale"
[4]: https://openreview.net/forum?id=nay3Kvw8BD&utm_source=chatgpt.com "An Efficient Training Algorithm for Models with Block-wise ..."
[5]: https://arxiv.org/abs/2403.03507?utm_source=chatgpt.com "GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection"
[6]: https://openreview.net/forum?id=Xvpk1g02u2&utm_source=chatgpt.com "QKV Projections Require a Fraction of Their Memory"
[7]: https://aclanthology.org/2025.naacl-long.71/?utm_source=chatgpt.com "Compressed Activations for Memory-Efficient LLM Training"
[8]: https://proceedings.mlr.press/v162/dao22a/dao22a.pdf?utm_source=chatgpt.com "Monarch: Expressive Structured Matrices for Efficient and ..."
[9]: https://arxiv.org/abs/1712.03942?utm_source=chatgpt.com "StrassenNets: Deep Learning with a Multiplication Budget"
[10]: https://arxiv.org/pdf/1612.02734?utm_source=chatgpt.com "Random Backpropagation and the Deep Learning Channel"
[11]: https://openreview.net/forum?id=lBBtmSu5Q2&utm_source=chatgpt.com "On Fine-Grained I/O Complexity of Attention Backward ..."
[12]: https://arxiv.org/abs/2505.19115?utm_source=chatgpt.com "FP4 All the Way: Fully Quantized Training of LLMs"
[13]: https://openreview.net/forum?id=P2q6Y7UweV&utm_source=chatgpt.com "INSTANT: Compressing Gradients and Activations for..."
