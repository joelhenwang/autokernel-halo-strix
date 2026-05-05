# LLM Novelties (2025–2026) Worth Testing in AutoKernel-Halo-Strix
Date: 2026-05-04 | POC: AdaL

## Executive summary

For this project, the highest-ROI novelties are not generic “bigger LLM” ideas. They are the ones that directly attack the repo’s known bottlenecks: memory-bandwidth-limited decode, the need for low-token-count architectural validation, and the project’s existing strength in hybrid/looped/SSM-style models.

My top recommendation is to prioritize **hybrid long-context architectures and transformer→hybrid conversion methods** over pure-transformer scaling. HALO/HypeNet shows that a pretrained Transformer can be converted into an attention-RNN hybrid in just 2.3B tokens while preserving comparable core performance and improving long-context efficiency. [1, lines 14-15] That token budget is materially smaller than prior hybrid-distillation approaches listed in the paper, including RAD (20B), KL-LS (25B), and Jet-Nemotron (400B). [1, lines 54-60] This is a strong fit for this repository because the project is already exploring looped hybrids, custom kernels, and long-context-efficient architectures.

The second recommendation is to test **curriculum learning as a warm-start for pretraining**. A 2026 EACL study reports that curriculum learning reduced the training steps needed to reach baseline performance by 18–45% in early and mid-training, and that using curriculum only as a warmup before returning to random sampling produced sustained gains up to 3.5%. [2, lines 48-49] For a resource-constrained research setup, this is unusually attractive because it does not require model surgery, only changes to sample ordering and bookkeeping.

The third recommendation is **training-free self-speculative decoding with adaptive cascades**, but I rank it below the first two because it addresses inference speed only, not model quality. CAS-Spec builds draft models on the fly using layer sparsity and activation quantization, then uses an adaptive Dynamic Tree Cascade (DyTC) to route between them. [3, lines 35-35] The paper reports average speedups of 1.1× to 2.3× over autoregressive decoding, with DyTC improving average speedup by 47–48% over static cascade and tree baselines. [3, lines 35-35] This is highly relevant because the repo has already identified decode as weight-bandwidth-bound; speculative decoding is one of the few credible ways to amortize those weight reads.

## Ranked shortlist of the most promising directions

### 1. HALO-style transformer→hybrid conversion
**Why it matters:** This is the best available “cheap architecture search at scale” mechanism I found. Instead of pretraining a new hybrid from scratch, the method converts a pretrained Transformer into a hybrid model with a much smaller adaptation budget. HALO combines attention-layer selection, distillation, and a hybrid architecture called HypeNet with the explicit goal of improving performance-throughput tradeoffs for long-context use. [1, lines 37-48]

**Why it fits this repo:** AutoKernel-Halo-Strix already has multiple custom hybrid designs, a strong interest in looped/recurrent blocks, and a hard constraint against expensive brute-force pretraining. This method directly addresses that.

**Concrete experiment in this repo:** Start with one of the repo’s existing transformer-like baselines or an imported small dense model, replace roughly 75% of attention layers with a simple recurrent/linear-attention mixer, and distill back using a reduced version of HALO. The point would not be to reproduce Qwen3-scale results, but to test whether the conversion recipe transfers to 50M–200M models on Strix Halo.

**Expected upside:** Faster prefill, lower KV/cache pressure, and a much cheaper way to validate new hybrid designs than training them from scratch.

**Difficulty:** Medium-high. The training pipeline work is non-trivial, but the architectural ingredients already overlap with this repo.

**Main risk:** Distilled hybrids can preserve short-context ability while still degrading on recall-heavy tasks if layer selection is poor. HALO’s key claim is that attention-layer selection should prioritize layers whose replacement hurts recall more than common-sense reasoning, and that only about 25% of layers are kept as attention. [1, lines 142-152]

### 2. HyPE-style positional design for hybrids
**Why it matters:** Positional strategy is emerging as a first-class design lever for long-context models. HypeNet’s HyPE uses RoPE in RNN layers and NoPE in attention layers, plus position-dependent attention-logit scaling at inference, to combine stronger local positional modeling with better long-range length generalization. [1, lines 170-184] SWAN-GPT independently reports that interleaving NoPE layers with sliding-window RoPE layers, plus dynamic scaling of attention scores at inference, produces strong extrapolation to much longer sequences without extra long-context training. [4, lines 49-49]

**Why it fits this repo:** This repo already cares about long-context efficiency and hybrid mixers. HyPE is much easier to test than inventing a whole new model family.

**Concrete experiment in this repo:** Run an ablation on VIDAR/FENRIR/TYR-style hybrids with:
1. all-RoPE,
2. attention-NoPE + recurrent-RoPE,
3. attention-NoPE + recurrent-RoPE + inference-time logit scaling.

**Expected upside:** Better length extrapolation with minimal runtime cost.

**Difficulty:** Medium.

**Main risk:** The gains may depend on exact mixer choice and training length. But even a negative result would still sharpen the project’s positional-design playbook.

### 3. Lightning Attention as the default recurrent/linear mixer baseline
**Why it matters:** The strongest consistent architecture signal across the gathered sources is that simple, hardware-friendly linear/recurrent mixers are often better research defaults than more complex state updates. HypeNet specifically argues that its long-context performance depends heavily on the interaction between hybrid PE choices and the RNN mixer. [1, lines 166-184] MiniMax-01 is cited as one of the major 2025 systems showing the practical viability of Lightning Attention at scale. [1, lines 27-27] MiniMax-01’s abstract says its long-context scaling is built around Lightning Attention and efficient scaling of that mechanism to million-token contexts. [5]

**Why it fits this repo:** The repo already concluded that memory movement and kernel simplicity dominate on Strix Halo. Lightning-style updates are mechanically simpler than full attention and appear more hardware-aligned than more elaborate data-dependent state transitions.

**Concrete experiment in this repo:** Make Lightning Attention the “boring but strong” recurrent baseline for all future hybrid experiments, and compare it directly against any Mamba/GLA/GDN-inspired alternatives inside the same model shell.

**Expected upside:** Better throughput, easier kernelization, and fewer unstable training dynamics than more expressive but fussier recurrent blocks.

**Difficulty:** Medium.

**Main risk:** It may leave some quality on the table for tasks requiring more expressive memory dynamics.

### 4. Curriculum learning as a pretraining warm-start
**Why it matters:** The 2026 curriculum-learning study is unusually practical: it reports that curriculum ordering accelerates early and mid-phase convergence, and that curriculum-warmup followed by random sampling can outperform pure random training by up to 3.5%. [2, lines 48-49] It also identifies compression ratio, MTLD lexical diversity, and Flesch Reading Ease as the strongest difficulty signals. [2, lines 48-49]

**Why it fits this repo:** This can be layered on top of the existing dataset pipeline with almost no architecture changes. It is a low-risk training-efficiency improvement.

**Concrete experiment in this repo:** Build a three-bin curriculum for the existing corpora using gzip compression ratio, readability, and lexical diversity approximations. Train:
- random baseline,
- easy→hard for first 10–20% of steps then random,
- interleaved curriculum.

**Expected upside:** Fewer steps to hit smoke-test quality thresholds and potentially better final BPB for fixed token budgets.

**Difficulty:** Low-medium.

**Main risk:** If the proxy difficulty metrics are noisy for code/mixed technical corpora, the curriculum may help less than the paper suggests.

### 5. RAD-style redundancy-aware layer replacement
**Why it matters:** RAD proposes using self-speculative decoding to identify redundant attention layers, replace them with SSM components, and then apply targeted distillation; it reports significantly stronger math/coding results than a baseline and up to ~2× faster convergence in standard knowledge distillation. [6, lines 48-49]

**Why it fits this repo:** This is a more targeted alternative to whole-model hybrid conversion. It aligns with the repo’s empirical mindset: diagnose, replace, measure.

**Concrete experiment in this repo:** Use a cheap redundancy signal for small models—ideally one derived from acceptance or agreement under shallow/early-exit drafting—to rank attention blocks, replace only the most redundant subset with a linear/recurrent mixer, then distill.

**Expected upside:** Smaller architectural changes than HALO, potentially faster to validate.

**Difficulty:** Medium-high.

**Main risk:** The published gains are strongest in distillation-heavy settings with external teachers; the benefit may shrink in a fully in-house small-model setup.

### 6. Training-free self-speculative decoding with adaptive cascades
**Why it matters:** CAS-Spec constructs draft models from the target model itself using DSIA strategies such as layer sparsity and activation quantization, then chooses the best cascade and draft length online. [3, lines 31-35] The authors argue that the main contribution is the dynamic, step-wise optimization of the cascade, and report negligible scheduling overhead below 0.1 ms, or about 0.5–2% of total compute time. [3, lines 87-93]

**Why it fits this repo:** The repo already knows that decode is bandwidth-bound. Adaptive self-speculation is one of the few paths that can improve tokens/sec without requiring a separate trained drafter.

**Concrete experiment in this repo:** Start with a much simpler version than CAS-Spec: one draft path using either layer skipping or weight-only quantized early layers, plus a basic acceptance-rate controller.

**Expected upside:** Better decode speed on local inference workloads, especially for 7B-class models.

**Difficulty:** Medium.

**Main risk:** Even the paper’s reviewers flagged that the method is clearest and strongest in the training-free regime, and that transfer to trained drafters like EAGLE-style systems remains less explored. [3, lines 69-71] So it is promising, but still more of an inference-side engineering bet than a clean architectural result.

### 7. Minimal SWAN-style long-context conversion
**Why it matters:** SWAN-GPT reports that existing pretrained decoder-only models can be converted to the SWAN architecture with minimal continued training, while also improving efficiency and long-context robustness. [4, lines 49-49]

**Why it fits this repo:** This is a lighter-weight alternative to full hybridization. If HALO is “replace most attention with recurrent layers,” SWAN is “keep transformer structure but change where positional encoding lives.”

**Concrete experiment in this repo:** Convert one existing transformer baseline to a SWAN-like structure by replacing a subset of full-attention layers with local/sliding-window attention and removing positional encoding from selected layers.

**Expected upside:** Better long-context scaling without the implementation cost of new recurrent kernels.

**Difficulty:** Medium.

**Main risk:** The effect size may be smaller than hybrid conversion on this specific hardware.

### 8. Better small-model distillation from stronger teachers
**Why it matters:** RAD’s results suggest that careful hybrid-aware distillation can outperform naïve baselines even with much smaller teachers than expected. [6, lines 48-49] Qwen3 explicitly frames one of its contributions as transferring knowledge from flagship models so that smaller models can remain highly competitive while using less compute to build. [7]

**Why it fits this repo:** This repo is already small-model-first. Better teacher transfer is probably more impactful than adding raw parameters.

**Concrete experiment in this repo:** Distill a 50M–100M project-native model from a stronger open model on a narrow domain mixture, but use hybrid-aware losses or layer-matching instead of plain token KL alone.

**Expected upside:** Better sample efficiency and stronger small-model reasoning.

**Difficulty:** Medium.

**Main risk:** Distillation data quality becomes the main bottleneck.

## Lower-priority or weaker-fit directions

### Massive MoE research
I would deprioritize frontier-scale MoE ideas for now. MiniMax-01 combines Lightning Attention with a 32-expert MoE and trains at hundred-billion scale. [5] That is important as proof that Lightning Attention works at scale, but the full MoE recipe is poorly matched to this repository’s hardware and experimentation budget.

### Fully training-based speculative systems as the first decode bet
Methods like EAGLE-style trained drafters may ultimately outperform training-free self-speculation, but they are a worse first step here because they add training and systems complexity before the repo has established a simple speculative-decoding baseline.

### Chasing highly expressive recurrent blocks first
The sources collectively suggest that some of the more expressive recurrent alternatives can be less stable or less length-generalizable than simpler mixers in practice. For this project, simpler, faster recurrent blocks should be the default baseline to beat.

## The three best next experiments

### Experiment A: Curriculum warm-start on VIDAR-HALO
- **Cost:** low
- **Change scope:** data loader / sampler only
- **Success metric:** steps-to-loss target, BPB after fixed tokens, tok/s unchanged
- **Why first:** easiest high-signal experiment

### Experiment B: HyPE-style positional ablation in an existing hybrid
- **Cost:** medium
- **Change scope:** model positional encoding + inference-time scaling
- **Success metric:** long-context eval and perplexity under extrapolation
- **Why second:** directly aligned with the repo’s architecture agenda

### Experiment C: Small-scale HALO/RAD-inspired hybrid conversion
- **Cost:** medium-high
- **Change scope:** teacher/student distillation + block replacement
- **Success metric:** quality retention vs throughput/prefill gain
- **Why third:** highest long-term upside if it works

## Bottom line

If I were steering the roadmap, I would sequence the work as:

1. **Curriculum warm-start** for immediate training-efficiency gains  
2. **HyPE / SWAN positional experiments** for fast architectural signal  
3. **HALO/RAD-style hybrid conversion** as the main medium-term research bet  
4. **CAS-Spec-lite** as the first decode-speed research branch  

That ordering matches the repo’s strengths: empirical iteration, hardware-aware architecture design, and a willingness to validate new ideas on smaller models before scaling them up.

## References

- [1] Efficient Distillation and Effective Architectures for Extremely Long Contexts | https://arxiv.org/html/2601.22156v1
- [2] Beyond Random Sampling: Efficient Language Model Pretraining via Curriculum Learning | https://aclanthology.org/2026.eacl-long.271/
- [3] CAS-Spec: Cascade Adaptive Self-Speculative Decoding for On-the-Fly... | https://openreview.net/forum?id=m0bR0sxhfL&referrer=%5Bthe%20profile%20of%20Xuelong%20Li%5D(%2Fprofile%3Fid%3D~Xuelong_Li1)
- [4] SWAN-GPT: An Efficient and Scalable Approach for Long-Context Language Modeling | https://arxiv.org/abs/2504.08719
- [5] MiniMax-01: Scaling Foundation Models with Lightning Attention | https://arxiv.org/abs/2501.08313
- [6] RAD: Redundancy-Aware Distillation for Hybrid Models via Self-Speculative Decoding | https://arxiv.org/abs/2505.22135
- [7] Qwen3 Technical Report | https://arxiv.org/abs/2505.09388