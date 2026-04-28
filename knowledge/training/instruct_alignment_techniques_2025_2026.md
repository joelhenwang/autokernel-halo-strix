---
title: "Instruct Alignment Techniques: 2025-2026 Survey"
domain: training
type: reference
status: active
tags: [alignment, sft, dpo, preference-optimization, instruction-tuning, small-models]
related:
  - sft_pipeline.md
  - alignment_implementation_details.md
  - ../architectures/reliable_small_lm_insights.md
---

# Instruct Alignment Techniques: Late 2025 -- April 2026

Survey of the most recent and efficient methods for turning a base LM into an instruction-following model. Focused on actionability for ~80-170M parameter models on limited compute (2 GPUs, AMD Strix Halo).

---

## 1. Efficient SFT Alternatives

### 1a. Importance-Aware Data Selection (MIWV)
- **Paper:** "Importance-Aware Data Selection for Efficient LLM Instruction Tuning" (Nov 2025)
- **Authors:** Tingyu Jiang, Shen Li, Yiyao Song et al.
- **Core idea:** Model Instruction Weakness Value (MIWV) identifies which instruction samples provide the greatest benefit by measuring discrepancies in the model's ICL responses. Achieves superior results with **only the top 1% of training data** compared to full-dataset SFT.
- **Cost:** Dramatically lower -- trains on 1% of data. Requires one inference pass to score data.
- **Practical for 80M?** YES. Especially useful when compute is limited. Run one scoring pass, then SFT on the most impactful subset.

### 1b. DataFlow: Quality Over Quantity
- **Paper:** "DataFlow: An LLM-Driven Framework for Unified Data Preparation" (Dec 2025)
- **Authors:** Hao Liang, Xiaochen Ma et al. (30+ authors)
- **Core idea:** ~200 reusable operators for data curation. Key finding: **a unified 10K-sample dataset produced by DataFlow beats models trained on 1M Infinity-Instruct data**. Quality trumps quantity by 100x.
- **Cost:** Upfront data curation cost, but training cost drops 100x.
- **Practical for 80M?** YES. The 10K-beats-1M finding is exactly what tiny models need.

### 1c. UFT: Unifying Supervised and Reinforcement Fine-Tuning
- **Paper:** "UFT: Unifying Supervised and Reinforcement Fine-Tuning" (May 2025, updated Oct 2025)
- **Authors:** Mingyang Liu, Gabriele Farina, Asuman Ozdaglar
- **Core idea:** Single framework combining SFT and RFT. Breaks RFT's exponential sample complexity bottleneck. Outperforms both SFT and RFT regardless of model size.
- **Cost:** Similar to SFT -- single training pass.
- **Practical for 80M?** LIKELY YES. Claims model-size-agnostic improvements.

---

## 2. DPO Improvements and Replacements

### 2a. SimPO -- Reference-Free Preference Optimization (BEST PICK)
- **Paper:** "SimPO: Simple Preference Optimization with a Reference-Free Reward" (May 2024, NeurIPS 2024)
- **Authors:** Yu Meng, Mengzhou Xia, Danqi Chen
- **Core idea:** Uses average log probability as implicit reward. No reference model needed. Adds target reward margin to Bradley-Terry objective. Outperforms DPO by 6.4 points on AlpacaEval 2, 7.5 on Arena-Hard.
- **Cost:** CHEAPER than DPO -- eliminates reference model (saves ~50% memory).
- **Practical for 80M?** YES. No reference model = half the GPU memory. Ideal for memory-constrained setups.

### 2b. ORPO -- Monolithic SFT+Preference in One Step (BEST PICK)
- **Paper:** "ORPO: Monolithic Preference Optimization without Reference Model" (Mar 2024)
- **Authors:** Jiwoo Hong, Noah Lee, James Thorne
- **Core idea:** Combines SFT and preference optimization into a single training phase using odds ratios. No separate SFT stage, no reference model. Phi-2 (2.7B) hit 12.20% AlpacaEval 2, 7.32 MT-Bench.
- **Cost:** ~50% cheaper than SFT+DPO pipeline (eliminates entire SFT stage + reference model).
- **Practical for 80M?** HIGHLY RECOMMENDED. Single-stage training = maximum compute efficiency. Perfect for tiny models.

### 2c. KTO -- Unpaired Preference Learning (BEST PICK)
- **Paper:** "KTO: Model Alignment as Prospect Theoretic Optimization" (Feb 2024, ICML 2024)
- **Authors:** Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, Douwe Kiela
- **Core idea:** Uses Kahneman-Tversky prospect theory. Only needs binary signal (good/bad), NOT paired preferences. Matches DPO performance from 1B to 30B. Higher sample efficiency than DPO for safety tasks.
- **Cost:** Similar to DPO but works with UNPAIRED data (much easier to collect).
- **Practical for 80M?** YES. Data collection is dramatically easier -- just label outputs as good or bad. SFT+KTO two-stage shown effective for Qwen2-0.5B.

### 2d. RePO -- ReLU-Based Preference Optimization
- **Paper:** "RePO: Understanding Preference Learning Through ReLU-Based Optimization" (Mar 2025)
- **Authors:** Junkang Wu et al.
- **Core idea:** SimPO's limiting case (beta->infinity). ReLU max-margin loss auto-discards trivial pairs. Only ONE hyperparameter to tune (vs SimPO's two, DPO's two).
- **Cost:** Same as SimPO. Simpler tuning.
- **Practical for 80M?** YES. Fewer hyperparameters = less tuning compute.

### 2e. AlphaPO -- Reward Shape Matters
- **Paper:** "AlphaPO: Reward Shape Matters for LLM Alignment" (Jan 2025, ICML 2025)
- **Authors:** Aman Gupta et al.
- **Core idea:** Alpha parameter modifies reward function shape beyond standard log. Addresses likelihood displacement. 7-10% improvement over SimPO, 15-50% over DPO.
- **Cost:** Same as base method + alpha parameter.
- **Practical for 80M?** YES. Drop-in improvement on top of SimPO/DPO.

### 2f. D2PO -- Temporal Decay for Token Weighting
- **Paper:** "Earlier Tokens Contribute More" (Feb 2025, ICLR 2025)
- **Authors:** Ruichen Shao et al.
- **Core idea:** Gamma decay factor weights earlier tokens more heavily. 5.9-8.8 point gains on AlpacaEval 2, 3.3-9.7 on Arena-Hard over vanilla DPO.
- **Cost:** Negligible overhead -- adds one decay parameter.
- **Practical for 80M?** YES. Free improvement on any DPO/SimPO training.

### 2g. RCPO -- Beyond Pairwise Preferences
- **Paper:** "Beyond Pairwise: Empowering LLM Alignment With Ranked Choice Modeling" (Oct 2025, ICLR 2026)
- **Authors:** Yuxuan Tang, Yifan Feng
- **Core idea:** Subsumes DPO and SimPO as special cases. Supports ranked (not just pairwise) feedback.
- **Practical for 80M?** Possible but likely overkill -- pairwise is sufficient at this scale.

---

## 3. Self-Play / Self-Improvement

### 3a. SPIN -- Self-Play Fine-Tuning
- **Paper:** "Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models" (Jan 2024, ICML 2024)
- **Authors:** Zixiang Chen, Yihe Deng, Huizhuo Yuan, Kaixuan Ji, Quanquan Gu
- **Core idea:** Model plays against previous versions of itself. Generates training data from prior iterations, iteratively self-improving without new human data.
- **Cost:** Multiple training iterations (typically 3), each similar to DPO.
- **Practical for 80M?** MAYBE. 3x training cost. But no need for additional human-labeled data.

### 3b. T-SPIN -- Triplets for Stable Self-Play
- **Paper:** "Triplets Better Than Pairs: Towards Stable and Effective Self-Play" (Jan 2026, NeurIPS 2025)
- **Authors:** Yibo Wang et al.
- **Core idea:** Incorporates historical advantages alongside current ones for stability. Achieves same results with **only 25% of samples**.
- **Cost:** 3x iterations but 4x more sample-efficient per iteration.
- **Practical for 80M?** YES. The 25% sample reduction helps a lot at small scale.

### 3c. SAO -- Self-Alignment Optimization
- **Paper:** "Aligning Large Language Models via Fully Self-Synthetic Data" (Oct 2025)
- **Authors:** Shangjian Yin et al.
- **Core idea:** Model generates ALL training data: prompts, responses, AND preferences through persona role-play. No external data or reward model needed. Enhances chat on AlpacaEval 2.0 while maintaining downstream task performance.
- **Cost:** Inference cost for data generation + standard preference training.
- **Practical for 80M?** UNCERTAIN. An 80M model may not generate high-quality self-evaluations. Works better at larger scales where the model can meaningfully judge its own outputs.

### 3d. ScPO -- Self-Consistency Preference Optimization
- **Paper:** "Self-Consistency Preference Optimization" (Nov 2024, ICML 2025)
- **Authors:** Archiki Prasad et al.
- **Core idea:** Iteratively trains consistent answers to be preferred over inconsistent ones on unsupervised problems. No reward model needed.
- **Cost:** Multiple sampling + iterative training. Tested on 8B+.
- **Practical for 80M?** UNLIKELY at this scale. Requires model consistency signal which needs sufficient capability.

### 3e. Self-Rewarding Language Models
- **Paper:** "Self-Rewarding Language Models" (Jan 2024, ICML 2024)
- **Authors:** Weizhe Yuan et al. (Meta)
- **Core idea:** Model acts as its own judge via LLM-as-a-Judge prompting. Iterative DPO where model improves instruction-following AND reward-generation simultaneously.
- **Cost:** 3 iterations of DPO. Tested only at 70B.
- **Practical for 80M?** NO. An 80M model cannot reliably judge its own outputs.

---

## 4. Distillation-Based Instruction Tuning

### 4a. Curriculum SFT + On-Policy Knowledge Distillation
- **Paper:** "Revealing the Power of Post-Training for Small Language Models via Knowledge Distillation" (Sep 2025)
- **Authors:** Miao Rang et al. (Huawei)
- **Core idea:** Curriculum-based SFT progressing from easy to hard, followed by offline on-policy KD. State-of-the-art among billion-parameter models.
- **Cost:** Requires teacher model inference + standard student training.
- **Practical for 80M?** YES. Generate teacher outputs offline (e.g., from Llama-3-8B-Instruct), then SFT the 80M student.

### 4b. TAID -- Temporally Adaptive Interpolated Distillation
- **Paper:** "TAID" (Jan 2025, ICLR 2025 Spotlight)
- **Authors:** Makoto Shing et al.
- **Core idea:** Dynamically interpolates student and teacher distributions through adaptive intermediate distribution. Tested at 1.5B-2B.
- **Cost:** Standard KD cost with adaptive scheduling.
- **Practical for 80M?** YES. Adaptive interpolation should help at any scale.

### 4c. ORPO-Distill -- Cross-Architecture Distillation
- **Paper:** "ORPO-Distill" (Sep 2025)
- **Authors:** Aasheesh Singh, Vishal Vaddina, Dagnachew Birru
- **Core idea:** Applies ORPO objective for cross-architecture distillation. Contrasts teacher and student reasoning traces.
- **Cost:** ORPO training cost (single stage, no reference model).
- **Practical for 80M?** YES. Combines distillation + alignment in one step.

### 4d. VibeThinker-1.5B -- Reasoning Distillation
- **Paper:** "VibeThinker-1.5B" (Nov 2025)
- **Authors:** Sen Xu et al.
- **Core idea:** Two-stage distillation from DeepSeek-R1 (671B) to 1.5B: diversity-exploring SFT + MaxEnt RL. Beats 400x larger teacher on some math benchmarks. Training cost: $7,800.
- **Cost:** SFT + RL. But 1.5B is 20x larger than our target.
- **Practical for 80M?** The SFT distillation stage is applicable. The RL stage may be too expensive for 80M's capability level.

---

## 5. Alignment at Small Scale (<500M)

### 5a. SmolLM Alignment Recipe (PROVEN BASELINE)
- **Paper/Blog:** SmolLM (HuggingFace, 2024)
- **Model sizes:** 135M, 360M, 1.7B
- **Method:** SFT (1 epoch) on WebInstructSub + StarCoder2-Self-OSS-Instruct, then DPO (1 epoch) on HelpSteer (135M/1.7B) or dpo-mix-7k (360M). Used Zephyr-Gemma recipe. LR: 3e-4 for SFT.
- **Results:** Competitive IFEval across all sizes including 135M.
- **Cost:** 1 epoch SFT + 1 epoch DPO. Very cheap.
- **Practical for 80M?** THIS IS THE BASELINE TO BEAT. Proven at 135M. Simple, cheap, reproducible.

### 5b. SFT-KTO Two-Stage for 0.5B
- **Reference:** "Post-Training Enhanced Optimization for Small Language Models" (Nov 2024, Zhai)
- **Model:** Qwen2-0.5B
- **Method:** SFT followed by KTO. KTO especially effective at small scale because it needs only binary feedback.
- **Cost:** SFT + KTO (similar to SFT + DPO but easier data collection).
- **Practical for 80M?** YES. KTO's unpaired data requirement is key -- no need for costly preference pairs.

### 5c. Scale-Dependent Hallucination Threshold
- **Paper:** "Before the First Token: Scale-Dependent Emergence of Hallucination Signals" (Apr 2026)
- **Authors:** Dip Roy et al.
- **Key finding:** Models below 400M show chance-level factuality probe accuracy. Reliable factuality signals emerge above ~1B. This means **alignment at 80-170M will primarily improve format/style, not factual accuracy**.
- **Practical implication:** Focus alignment on instruction format, EOS behavior, and style. Don't expect factual grounding improvements at this scale.

### 5d. Learning to Reason via Self-Iterative Process Feedback
- **Paper:** (Dec 2024)
- **Authors:** Kaiyuan Chen, Jin Wang, Xuejie Zhang
- **Core idea:** ORPO + self-generated feedback for reasoning. Gemma-2B improved 12.43 points on GSM8K.
- **Practical for 80M?** The ORPO component is applicable. Self-generated feedback unlikely to work at 80M.

---

## 6. RLHF-Free Alignment

### 6a. GRPO -- Group Relative Policy Optimization (BEST PICK for RL)
- **Paper:** DeepSeekMath (Feb 2024) + DAPO (Mar 2025)
- **Core idea:** Variant of PPO that eliminates the value model entirely. Groups responses, computes relative advantage within each group. Saves ~50% memory vs PPO.
- **Cost:** Cheaper than PPO. Needs reward signal but no value network or reward model.
- **Practical for 80M?** YES for verifiable tasks (math, code). Not applicable for general chat alignment without a reward signal.

### 6b. DAPO -- Decoupled Clip and Dynamic Sampling
- **Paper:** "DAPO: An Open-Source LLM Reinforcement Learning System at Scale" (Mar 2025)
- **Authors:** 35 researchers
- **Core idea:** Improves GRPO with decoupled clipping, dynamic sampling, token-level loss, overlong filtering. Achieves 50 on AIME 2024 with Qwen2.5-32B.
- **Cost:** More efficient than GRPO through better sampling.
- **Practical for 80M?** Only for verifiable-reward tasks.

### 6c. RLVR -- Reinforcement Learning with Verifiable Rewards
- **Paper:** Tulu 3 (Nov 2024, revised Apr 2025)
- **Authors:** Nathan Lambert et al. (AI2)
- **Core idea:** Third stage after SFT+DPO. Uses programmatic/verifiable rewards (math correctness, code execution) instead of learned reward models.
- **Cost:** Standard RL cost but no reward model training.
- **Practical for 80M?** YES for math/code. Tulu 3's SFT+DPO+RLVR pipeline is the gold standard.

### 6d. PPPO -- Progressive Prefix-token Policy Optimization
- **Paper:** (Dec 2025)
- **Authors:** Sun, Zhao, Wei et al.
- **Core idea:** Optimizes only prefix reasoning segments. 18% accuracy improvement using only 26% of training tokens.
- **Cost:** ~4x cheaper in token compute.
- **Practical for 80M?** Interesting for token-efficient RL if applicable.

### 6e. SSB -- Semantic Soft Bootstrapping (Self-Distillation)
- **Paper:** (Dec 2025)
- **Authors:** Mitra, Ulukus
- **Core idea:** Same model acts as teacher-student with contextual correctness feedback. 10.6% improvement on GSM8K over GRPO with Qwen2.5-3B.
- **Cost:** No separate reward model. Standard training.
- **Practical for 80M?** MAYBE. Tested at 3B, but the self-distillation principle might scale down.

---

## 7. Synthetic Data for Instruction Tuning

### 7a. Magpie -- Data from Nothing (BEST PICK)
- **Paper:** "Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing" (Jun 2024)
- **Authors:** Zhangchen Xu et al.
- **Core idea:** Feed only the chat template prefix to an aligned model (e.g., Llama-3-Instruct) and it auto-generates user queries. 300K curated instances match the official Llama-3-8B-Instruct trained on 10M data points.
- **Cost:** Only inference cost on the teacher model. No human annotation.
- **Practical for 80M?** YES for data generation (use a larger aligned model to generate). The 80M model is the STUDENT, not the generator.

### 7b. SAO -- Self-Generated Everything
- **Paper:** (Oct 2025, described in Section 3c above)
- **Practical for 80M?** NO. Requires model capability to generate quality data.

### 7c. DataFlow (described in Section 1b above)
- Key: 10K curated samples > 1M generic samples.

### 7d. QueST -- Difficulty-Aware Synthetic Data
- **Paper:** (Oct 2025)
- **Authors:** Hu et al.
- **Core idea:** Difficulty-aware graph sampling + rejection fine-tuning for coding. 8B model matches 671B DeepSeek-R1 on competitive coding with 100K generated problems.
- **Practical for 80M?** The difficulty-aware selection principle applies.

---

## 8. Multi-Stage Alignment Pipelines

### 8a. Tulu 3 Pipeline (GOLD STANDARD)
- **Paper:** "Tulu 3: Pushing Frontiers in Open Language Model Post-Training" (Nov 2024, revised Apr 2025)
- **Authors:** Nathan Lambert et al. (AI2)
- **Pipeline:** SFT -> DPO -> RLVR (3 stages)
- **Key insights:** Substantial data decontamination. Released complete recipes. Documents what DIDN'T work. Surpasses Llama-3.1-Instruct, Qwen-2.5, Mistral.
- **Practical for 80M?** SFT+DPO stages YES. RLVR stage only for verifiable tasks.

### 8b. Nanbeige4-3B Pipeline
- **Paper:** (Dec 2025)
- **Authors:** Chen Yang et al.
- **Pipeline:** Fine-grained warmup-stable-decay -> SFT with deliberative refinement -> Dual Preference Distillation -> Multi-stage RL
- **At 3B:** Rivals much larger models.
- **Practical for 80M?** Too complex. Stick to simpler pipelines.

### 8c. SmolLM Pipeline (SIMPLEST PROVEN)
- **Pipeline:** SFT (1 epoch) -> DPO (1 epoch)
- **Proven at 135M.** Total: 2 epochs of training.

### 8d. EVA Pipeline
- **Paper:** "EVA: Efficient Video Agent" (Mar 2026)
- **Pipeline:** SFT -> KTO -> GRPO (3 stages)
- **Key insight:** KTO as middle stage is interesting -- easier data than DPO.

---

## 9. Lightweight Alternatives to Full Fine-Tuning

### 9a. LoRA + Preference Optimization
- Standard LoRA (rank 16-64) with any preference method reduces memory 4-8x.
- For 80M params, LoRA is likely unnecessary -- full fine-tuning fits in memory.

### 9b. Context Distillation
- Distill system prompt behavior into weights via SFT on (prompt+system_instruction, response) pairs.
- **Practical for 80M?** YES. Simple and effective. Train on conversations where the system prompt constrains behavior, then remove the system prompt at inference.

### 9c. PEFT under RLVR
- **Paper:** "Parameter Efficiency Evaluation" (Dec 2025)
- **Finding:** DoRA > AdaLoRA > standard LoRA for RLVR. Extreme rank reduction (rank-1) severely bottlenecks reasoning.
- **Practical for 80M?** Full fine-tuning is better at this scale. LoRA only if memory-constrained.

---

## 10. Token-Level vs Sequence-Level Optimization

### 10a. TGDPO -- Token-Level Reward Guidance
- **Paper:** "Harnessing Token-Level Reward Guidance for Enhancing DPO" (Jun 2025)
- **Authors:** Mingkang Zhu et al.
- **Core idea:** Decomposes sequence-level PPO into token-level. Different tokens deviate from reference policy at different rates. +7.5 on MT-Bench, +6.2 on AlpacaEval 2 over DPO.
- **Cost:** Requires token-level rewards (additional overhead).
- **Practical for 80M?** Marginal benefit vs. complexity.

### 10b. ConfPO -- Confidence-Based Token Selection
- **Paper:** "Exploiting Policy Model Confidence for Critical Token Selection" (Jun 2025)
- **Authors:** Hee Suk Yoon et al.
- **Core idea:** Identifies preference-critical tokens using policy model confidence. No auxiliary models needed. Zero additional computational overhead.
- **Cost:** FREE improvement on top of DPO/SimPO.
- **Practical for 80M?** YES. Zero-cost improvement.

### 10c. TI-DPO -- Token-Importance Guided DPO
- **Paper:** (May 2025)
- **Authors:** Ning Yang et al.
- **Core idea:** Gradient attribution + Gaussian priors for token importance. Higher accuracy and diversity.
- **Cost:** Small overhead for importance computation.

### 10d. D2PO (described in Section 2f)
- Temporal decay -- earlier tokens weighted more. Free improvement.

---

## RECOMMENDED PIPELINE FOR 80-170M MODELS

Given the constraints (80-170M params, 2 AMD GPUs, limited compute), here is the recommended pipeline ranked by practicality:

### Option A: Simplest (Proven at 135M -- SmolLM recipe)
```
1. SFT (1 epoch) on curated 10-50K instruction data
   - Use Magpie-generated or DataFlow-curated data
   - LR: 2e-5 to 3e-4, AdamW
2. DPO or SimPO (1 epoch) on preference data
   - SimPO preferred (no reference model, saves memory)
   - HelpSteer or dpo-mix-7k
```
Total: 2 epochs. ~1-2 hours on 2 GPUs.

### Option B: Most Compute-Efficient (Single Stage)
```
1. ORPO (1-2 epochs) on mixed instruction+preference data
   - Combines SFT + preference in single pass
   - No reference model, no separate SFT stage
```
Total: 1-2 epochs. ~30-60 minutes on 2 GPUs.

### Option C: Best Quality (Multi-Stage with Unpaired Data)
```
1. Curate data: MIWV scoring or DataFlow to select top 10K instructions
2. SFT (1-3 epochs) on curated data
3. KTO (1 epoch) with binary good/bad labels
   - Easier to collect than paired DPO data
   - Apply D2PO temporal decay + ConfPO token selection (free)
4. [Optional] RLVR on math/code if verifiable tasks needed
```
Total: 3-5 epochs across stages.

### Option D: Distillation-First
```
1. Generate 50-100K instruction responses from Llama-3-8B-Instruct (or similar)
   - Use Magpie for diverse queries
2. SFT (1 epoch) on teacher outputs
3. SimPO (1 epoch) with teacher-vs-student preference pairs
```
Total: 2 epochs + teacher inference.

### Key Findings for Small Scale

1. **Data quality >> data quantity**: 10K curated > 1M generic (DataFlow)
2. **ORPO is the efficiency king**: single-stage, no reference model, proven on Phi-2
3. **SimPO > DPO**: reference-free saves half the memory, better results
4. **KTO for easy data**: binary feedback is 10x easier to collect than paired preferences
5. **Factual accuracy won't improve at <400M** (Roy et al. 2026): focus on format/style
6. **Self-play/self-rewarding don't work at <1B**: model can't judge itself
7. **Distillation from larger models is the highest-ROI approach** for capabilities
8. **D2PO temporal decay and ConfPO are free improvements** on any preference method
9. **AlphaPO gives 7-50% improvement** as drop-in alpha parameter on SimPO/DPO
10. **SmolLM recipe (SFT+DPO, 2 epochs)** is the proven baseline at 135M

### What To Skip

- Self-rewarding / SAO / ScPO: requires model capability that 80M doesn't have
- Complex multi-stage RL pipelines: cost/benefit ratio too high at this scale
- RLHF with learned reward models: overkill, use SimPO/ORPO/KTO instead
- LoRA/PEFT: unnecessary at 80M -- full fine-tuning fits in memory
- Token-level methods requiring auxiliary models (TGDPO): marginal benefit
