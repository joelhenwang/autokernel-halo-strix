# AutoKernel-Halo-Strix: Broader Deep Dive and Research Roadmap
Date: 2026-05-04 | POC: AdaL

## Executive summary

AutoKernel-Halo-Strix is unusually well-positioned for a specific kind of LLM research: **hardware-aware small-to-mid-scale model research where architecture, training recipe, and inference system are co-designed rather than optimized in isolation**. The project’s strongest unique advantage is not raw scale. It is the combination of (1) custom HIP kernel capability, (2) empirical evidence that Strix Halo is memory-bandwidth-limited rather than matmul-accelerator-rich, and (3) an existing culture of testing hybrid, looped, and recurrent alternatives rather than defaulting to plain Transformer scaling.

That combination changes what “good research taste” looks like here. The best opportunities are the ones that either reduce memory traffic, amortize weight reads, improve sample efficiency, or let the project validate new ideas without frontier-scale pretraining. The weakest opportunities are the ones whose benefits rely on tensor-core-style compute abundance, giant expert routing systems, or expensive teacher/student pipelines that only pay off at much larger scale.

The highest-value directions remain:
1. **Hybrid/looped architecture research with cheap adaptation methods**
2. **Pretraining efficiency work: curriculum, tokenization, and data mixture**
3. **Inference-time methods that directly attack the bandwidth wall**
4. **Evaluation methodology that measures deployability, not just benchmark scores**

The report’s bottom line is simple: this project should lean into being a **research platform for efficient local language models**, not a smaller imitation of frontier dense-model labs.

## What this project is uniquely suited for

This repo is a particularly strong environment for ideas where model design and systems design must interact tightly. Its own optimization report already shows that on Strix Halo, the biggest gains come from **fusion and overhead elimination**, not from trying to out-matmul vendor libraries. Hybrid architectures are attractive because they trade some softmax attention for recurrent or linear-time mechanisms that are more efficient on long contexts. [1, lines 27-27] The HypeNet results emphasize that reducing attention layers can improve long-context efficiency enough that the original Transformer can run out of memory where the hybrid still works. [1, lines 31-31]

The project is therefore uniquely suited to:
- validating **hybrid attention/RNN/SSM** designs;
- testing **cheap conversion or distillation** from existing models into more efficient ones;
- exploring **decode-time acceleration** methods that amortize weight reads;
- co-optimizing **tokenization, data ordering, and small-model training**;
- building **real deployability benchmarks** where latency and throughput matter as much as accuracy.

It is much less suited to:
- inventing gigantic compute-bound kernels that depend on accelerators this hardware does not have;
- massive MoE research that assumes large multi-node clusters and high-bandwidth interconnects;
- methods whose only real advantage appears after training on tens or hundreds of billions of tokens.

So the key strategic principle is: **prefer ideas whose gains survive when bandwidth, not FLOPs, is the bottleneck**.

## Architecture opportunities

### 1. Transformer-to-hybrid conversion is the best medium-term architecture bet

The strongest architecture-side opportunity remains HALO-style conversion and its relatives. HALO/HypeNet claims a pretrained Transformer can be converted into an attention-RNN hybrid using only 2.3B tokens while preserving comparable core performance and improving long-context efficiency. [1, lines 14-15] The paper explicitly positions this as far cheaper than prior hybrid conversion methods such as RAD (20B), KL-LS (25B), and Jet-Nemotron (400B). [1, lines 54-60]

This matters because it turns architecture research into something more affordable. Instead of training every hybrid from scratch, this repo could:
- start from a strong small dense teacher,
- replace a controlled fraction of attention blocks,
- use a reduced distillation recipe,
- then compare quality, prefill speed, and decode characteristics on the target hardware.

That is a much more efficient way to probe architectural hypotheses than fully retraining each candidate.

### 2. Positional strategy is now a first-class architecture lever

The evidence collected here strongly suggests that positional design is not just a detail anymore. HyPE combines RoPE in recurrent layers, NoPE in attention layers, and position-dependent attention-logit scaling to improve long-range length generalization with negligible runtime cost. [1, lines 170-184] SWAN-GPT independently shows that interleaving NoPE layers with sliding-window RoPE layers, plus dynamic scaling of attention scores during inference, can produce strong length extrapolation without additional long-context training. [2, lines 49-49]

That makes positional ablations a very attractive short-term research branch:
- all-RoPE baseline,
- attention-NoPE + recurrent-RoPE,
- inference-time scaling on top,
- maybe local-vs-global layer partitioning.

These experiments are much cheaper than full architecture redesign and may still unlock meaningful long-context gains.

### 3. Lightning-style recurrent mixers should be the default baseline to beat

A recurring pattern across the sources is that **simpler recurrent or linear-attention mixers may be better research defaults than more expressive but finicky ones**. HypeNet makes the interaction between mixer choice and length-generalization central to performance. [1, lines 166-184] MiniMax-01’s core claim is that Lightning Attention is the scaling mechanism that makes million-token contexts practical at large scale. [3]

For this project, that means Lightning-style updates should probably be the “boring strong baseline” inside hybrid shells, especially because simpler updates are easier to kernelize and more aligned with the project’s memory-centric optimization profile.

### 4. RAD-style selective replacement is a strong lower-risk alternative to full conversion

Full conversion is powerful, but selective block replacement may be more practical in some experiments. RAD proposes identifying redundant attention layers using self-speculative decoding, replacing only those layers with SSM components, and then distilling; it reports better math/coding performance than a baseline and up to about 2× faster distillation convergence. [4, lines 48-49]

For this repo, RAD’s real value is methodological: it suggests a **diagnose → replace → distill** loop that fits the project’s experimental style. Even if the exact redundancy signal changes, the broader pattern is compelling.

### 5. What not to chase architecturally

The weakest-fit architectural directions are:
- large-scale MoE as a primary research pillar,
- very compute-heavy full-attention alternatives,
- highly expressive recurrent blocks that require substantial stability babysitting before they show any value.

These may still matter later, but they are poor first-order bets for this hardware and workflow.

## Pretraining and data opportunities

### 1. Curriculum warm-start is the clearest quick win

Among all pretraining ideas surveyed, curriculum learning is the most obviously actionable. The EACL 2026 curriculum-learning paper reports 18–45% faster convergence to baseline performance in early and mid-training and sustained gains up to 3.5% when curriculum is used as a warmup before switching back to random sampling. [5, lines 48-49] It also identifies compression ratio, lexical diversity (MTLD), and readability as the most effective difficulty signals. [5, lines 48-49]

This is a strong fit because:
- it requires no major model surgery,
- it can sit on top of the existing data pipeline,
- it naturally complements the repo’s small-model, limited-budget training style.

This should be treated as immediate roadmap material, not just background reading.

### 2. Tokenization is strategically important, not just preprocessing

This project already understands tokenization as a systems issue, not merely a language modeling issue. Smaller token counts mean:
- fewer decode steps,
- less activation traffic,
- smaller sequence lengths during training,
- cheaper data pipelines.

The prior report already noted that VIDAR’s custom tokenizer reduced token counts significantly on its target corpora. The broader lesson is that **tokenization work is especially valuable in this repo because every saved token multiplies through both training and inference**.

The practical research direction here is not “invent a universal tokenizer.” It is:
- test domain-specific or mixed-domain tokenizers,
- measure token count changes on the corpora that actually matter here,
- then measure the downstream effect on throughput, loss, and memory.

### 3. Data mixture optimization belongs in the core research portfolio

The repo already has some evidence that data mixture optimization matters. The broader literature and project context together suggest this should remain central. Data-mixture work fits the project well because it is:
- sample-efficiency-oriented,
- architecture-agnostic,
- cheap relative to model redesign,
- and easy to combine with curriculum experiments.

The most attractive experiments are:
- curriculum × data-mixture interaction,
- technical-domain emphasis vs broader general data,
- teacher-distilled or synthetic augmentation only where it compresses token budgets effectively.

### 4. Distillation-first is often a better research strategy than scratch training

For this repo, “can we cheaply reshape a strong model into the one we want?” is often a better question than “can we train the desired model from zero?” HALO and RAD both support that general view. HALO’s recipe is explicitly multi-stage: weight transfer, hidden-state alignment, end-to-end KL distillation, then long-context finetuning. [1, lines 122-164]

That is attractive because this repo’s actual comparative advantage is experimentation, not massive-scale scratch pretraining. Distillation-first should be a default mindset for medium-term architecture work.

## Inference and systems opportunities

### 1. Speculative decoding remains one of the few real answers to the decode wall

This project has already identified that decode is fundamentally limited by weight reads. That means methods that simply make individual layers slightly cheaper may not move the needle enough. The attractive inference-side ideas are the ones that **amortize those reads across more accepted tokens**.

CAS-Spec constructs draft models on the fly using training-free methods such as layer sparsity and activation quantization, then uses an adaptive cascade to choose the most promising drafting path and length. [6, lines 31-35] The paper was accepted as a NeurIPS 2025 poster, though reviewers still noted that its broader applicability beyond training-free drafters needed more clarification. [6, lines 69-71] The authors argue that the dynamic routing is the key contribution, and report scheduling overhead below 0.1ms, roughly 0.5–2% of total compute time. [6, lines 87-93]

This is still promising even if the first implementation here is much simpler than the paper’s full DyTC framework.

### 2. Adaptive test-time compute is a good fit only in narrow settings

There is an important distinction between “reasoning-time compute” in general and what this repo should pursue. Adaptive test-time compute allocation via constrained policy optimization reports up to 12.8% relative accuracy improvement on MATH under matched compute budgets by learning which inputs deserve more repeated-sampling compute. [7, lines 30-30] Its core observation is that easy questions should be answered cheaply, responsive questions should receive extra compute, and intractable questions should not waste budget. [7, lines 38-40]

This is intellectually attractive, but it is not universally aligned with the repo’s core workloads. It fits best when:
- the target task is reasoning-heavy,
- repeated sampling is acceptable,
- and latency budgets are flexible.

It fits less well for plain autoregressive local generation where output quality is dominated by base-model competence rather than deliberation.

So this is a **conditional-fit** direction: worth testing for math/reasoning variants, but not a universal default.

### 3. Retrieval and external memory are promising, but model-specific

The system-log benchmark is highly informative here because it focuses on small deployable models under latency constraints. The benchmark found strong stratification across SLMs and SRLMs under zero-shot, few-shot, and RAG settings, with Qwen3-4B reaching 95.64% accuracy with RAG and some tiny models improving dramatically under retrieval. [8, lines 18-18] But it also found that not all models benefit from retrieval: some reasoning-focused small models degraded substantially when paired with RAG. [8, lines 208-212] Reducing top-k retrieval for a struggling Qwen3-1.7B variant did not recover performance, suggesting the issue was not context volume alone but poor retrieval integration. [8, lines 226-228]

This is highly relevant to AutoKernel-Halo-Strix. It suggests that:
- retrieval should not be treated as automatically beneficial,
- small models may vary radically in how well they use retrieved evidence,
- “reasoning-optimized” models may sometimes be worse retrieval consumers than simpler ones.

That means retrieval experiments here should always be architecture-specific and latency-aware.

### 4. Hardware-fit principle for inference work

Inference work that best fits this repo usually has one of four properties:
1. it reduces the number of decode steps,
2. it increases accepted tokens per expensive weight read,
3. it reduces sequence length or KV footprint,
4. it uses lightweight routing or control logic with negligible overhead.

That principle should guide future decode-side decisions more than benchmark fashion does.

## Optimization and training mechanics

This was the least well-covered area in the gathered primary evidence, so conclusions here should be treated as more tentative.

The main meta-finding is that this repo should prefer **low-complexity training-mechanics changes that interact well with hybrid/recurrent stability** over optimizer novelty for its own sake. The project already has enough moving parts at the architecture and kernel level.

The most promising mechanics directions are:
- selective fp32 islands where instability concentrates,
- tighter defaults around gradient clipping and recurrence stabilization,
- schedule design that favors small-model convergence over frontier-style long ramps,
- compile/custom-op/autograd co-design when it enables throughput without degrading correctness.

From the gathered evidence, I would frame optimizer work as a second-tier priority unless there is a very concrete hypothesis tied to this repo’s model family.

## Evaluation methodology upgrades

### 1. This project should measure deployability, not just leaderboard accuracy

The strongest evaluation lesson from the small-log benchmark is methodological, not domain-specific. The paper explicitly treats its task as a probe of runtime competence and deployability rather than as a narrow end task. [8, lines 18-18] Its stated contribution includes jointly measuring accuracy and inference latency for small and small-reasoning models under realistic prompting settings. [8, lines 88-90]

That is exactly the mindset this repo should adopt. For AutoKernel-Halo-Strix, evaluation should routinely include:
- quality metric,
- prefill latency,
- decode latency or tok/s,
- memory footprint,
- failure mode profile,
- stability under prompt or context variation.

### 2. Fast falsification benchmarks matter more than giant benchmark suites

This repo needs quick ways to reject bad ideas. Good evaluation for this project should include:
- one long-context stress test,
- one retrieval-conditioned task,
- one reasoning-ish task,
- one plain language modeling or perplexity metric,
- one latency-throughput profile.

The goal is not comprehensive benchmark coverage first. It is fast, repeatable, architecture-sensitive signal.

### 3. Evaluate retrieval compatibility explicitly

The retrieval findings are strong enough that every future “small model + external memory” experiment should include:
- no retrieval,
- few-shot,
- retrieval with fixed top-k,
- latency measurement,
- output-format adherence.

That prevents false optimism from architectures that look promising until retrieval enters the picture.

## Ranked experiment backlog

Below is the broader backlog, ranked roughly by project fit.

### Tier 1: immediate
1. **Curriculum warm-start on VIDAR-HALO**
   - Upside: high
   - Cost: low
   - Kill criterion: no step-to-target improvement after 10–20% of training

2. **HyPE-style positional ablation**
   - Upside: medium-high
   - Cost: medium
   - Kill criterion: no long-context gain or too much short-context regression

3. **Lightning Attention baseline shell**
   - Upside: medium-high
   - Cost: medium
   - Kill criterion: no throughput or stability advantage over current recurrent baseline

4. **Tokenizer comparison on active corpora**
   - Upside: high
   - Cost: low-medium
   - Kill criterion: token savings don’t translate into throughput or loss improvements

5. **Deployability benchmark harness**
   - Upside: high
   - Cost: medium
   - Kill criterion: none; this is infrastructure work

### Tier 2: medium-term bets
6. **HALO-lite conversion for 50M–200M models**
7. **RAD-style selective block replacement**
8. **Curriculum × data-mixture interaction study**
9. **Retrieval compatibility benchmark across project models**
10. **CAS-Spec-lite self-speculative decode**
11. **Adaptive budgeted reasoning on reasoning-specific tasks only**
12. **Teacher-guided small-model distillation with hybrid-aware losses**

### Tier 3: ambitious
13. **Integrated hybrid + retrieval-aware small model**
14. **Dynamic mode-switching small model inspired by Qwen3 thinking/non-thinking**
15. **Adaptive depth or early-exit hybrid**
16. **Quantization-first decode stack coupled with speculative accept/reject control**
17. **Model family explicitly co-designed around kernel-fusible hot paths**
18. **A project-native benchmark suite for local deployable LMs on Strix Halo**

## Recommended roadmap

### Next 1–2 weeks
- curriculum warm-start
- positional ablation
- tokenizer comparison
- evaluation harness cleanup

### Next 1–2 months
- Lightning baseline
- retrieval compatibility study
- HALO-lite feasibility work
- first simple speculative decode prototype

### Quarter-scale
- full hybrid conversion branch
- selective replacement/distillation branch
- deployability benchmark release
- one flagship paper-worthy architecture/system co-design result

## Anti-roadmap: what not to chase

Do **not** make these central:
- giant MoE scaling
- compute-bound kernel heroics with poor hardware fit
- optimizer novelty without a concrete architecture-specific hypothesis
- retrieval as a blanket default for all small models
- reasoning-time scaling as a universal solution rather than a task-specific tool

## Final take

The most important strategic insight from this broader deep dive is that AutoKernel-Halo-Strix should think of itself as a **co-design lab**.

The best work here will come from questions like:
- Which architectures create the right memory-access pattern for this hardware?
- Which training recipes let small hybrids converge faster and more reliably?
- Which inference tricks actually beat the bandwidth wall instead of just moving it?
- Which evaluation setups tell us early whether a model is deployable, not just impressive on paper?

The repo should aim to produce research that would be hard for a generic large-scale lab to notice, but easy for this environment to expose: **small-model, local, hardware-aware truths about what really works**.

## References

- [1] Efficient Distillation and Effective Architectures for Extremely Long Contexts | https://arxiv.org/html/2601.22156v1
- [2] SWAN-GPT: An Efficient and Scalable Approach for Long-Context Language Modeling | https://arxiv.org/abs/2504.08719
- [3] MiniMax-01: Scaling Foundation Models with Lightning Attention | https://arxiv.org/abs/2501.08313
- [4] RAD: Redundancy-Aware Distillation for Hybrid Models via Self-Speculative Decoding | https://arxiv.org/abs/2505.22135
- [5] Beyond Random Sampling: Efficient Language Model Pretraining via Curriculum Learning | https://aclanthology.org/2026.eacl-long.271/
- [6] CAS-Spec: Cascade Adaptive Self-Speculative Decoding for On-the-Fly... | https://openreview.net/forum?id=m0bR0sxhfL&referrer=%5Bthe%20profile%20of%20Xuelong%20Li%5D(%2Fprofile%3Fid%3D~Xuelong_Li1)
- [7] Adaptive Test-Time Compute Allocation for Reasoning LLMs via Constrained Policy Optimization | https://arxiv.org/html/2604.14853v1
- [8] Benchmarking Small Language Models and Small Reasoning Language Models on System Log Severity Classification | https://arxiv.org/html/2601.07790v1