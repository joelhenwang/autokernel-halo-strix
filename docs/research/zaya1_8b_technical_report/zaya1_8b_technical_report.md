# ZAYA1-8B Technical Report

Robert Washbourne\*, Rishi Iyer, Tomas Figliolia, Henry Zheng, Ryan Lorig-Roach, Sungyeon Yang, Pritish Yuvraj, Quentin Anthony, Yury Tokpanov, Xiao Yang, Ganesh Nanduru, Stephen Ebert, Praneeth Medepalli, Skyler Szot, Srivatsan Rajagopal, Alex Ong, Bhavana Mehta, Beren Millidge\*

Zyphra

San Francisco, CA

\*Corresponding authors: rob@zyphra.com, beren@zyphra.com

Abstract—We present ZAYA1-8B, a reasoning-focused mixtureof-experts (MoE) model with 700M active and 8B total parameters, built on Zyphra’s MoE++ architecture. ZAYA1-8B’s core pretraining, midtraining, and supervised fine-tuning (SFT) were performed on a full-stack AMD compute, networking, and software platform. With under 1B active parameters, ZAYA1-8B matches or exceeds DeepSeek-R1-0528 on several challenging mathematics and coding benchmarks, and remains competitive with substantially larger openweight reasoning models. ZAYA1-8B was trained from scratch for reasoning, with reasoning data included from pretraining onward using an answer-preserving trimming scheme. Post-training uses a four-stage RL cascade: reasoning warmup on math and puzzles; a 400-task RLVE-Gym curriculum; math and code RL with testtime compute traces and synthetic code environments built from competitive-programming references; and behavioral RL for chat and instruction following. We also introduce Markovian RSA, a test-time compute method that recursively aggregates parallel reasoning traces while carrying forward only bounded-length reasoning tails between rounds. In TTC evaluation, Markovian RSA raises ZAYA1-8B to 91.9% on AIME’25 and 89.6% on HMMT’25 while carrying forward only a 4K-token tail, narrowing the gap to much larger reasoning models including Gemini-2.5 Pro, DeepSeek-V3.2, and GPT-5-High.

## I. INTRODUCTION

In this paper, we introduce ZAYA1-8B, a 700M-active, 8B-total parameter mixture-of-experts (MoE) model. With under 1B active parameters, ZAYA1-8B matches or exceeds DeepSeek-R1-0528 on several challenging mathematics and coding benchmarks, while remaining competitive with substantially larger open-weight reasoning models including OLMo-3.1-32B-Think, Nemotron-3-Nano-30B-A3B, Mistral-Small-4-119B-2603, and Intellect-3-12A-106B (NVIDIA, 2025; Team et al., 2025c; Team, 2025a; Mistral AI, 2026).

Moreover, using our test-time compute scheme, Markovian RSA, ZAYA1-8B narrows the gap on AIME’25 and HMMT’25 to substantially larger reasoning models including Gemini-2.5 Pro, DeepSeek-V3.2, Qwen3-235B-A22B-Thinking-2507, and GPT-5-High (Comanici et al., 2025; DeepSeek-AI, 2025c; Team, 2025b; OpenAI, 2025). These results suggest that competitive mathematical reasoning can be reached with under 1B active parameters when model architecture, reasoning-heavy training, verifiable RL, and testtime aggregation are co-designed.

The system combines five design choices that we found important in practice:

Architecture: ZAYA1-8B builds on Zyphra’s MoE++ architecture (Anthony et al., 2025), with three main changes relative to standard transformer MoE designs. First, ZAYA1- 8B uses Compressed Convolutional Attention (CCA) (Figliolia et al., 2025), a FLOP- and memory-efficient attention variant that performs sequence mixing in a compressed latent space. Prior work showed that CCA performs well on perplexity and standard language modeling at small scale; ZAYA1-8B evaluates its behavior at larger scale and on more challenging reasoning and long-context tasks. Second, ZAYA1-8B uses the ZAYA1 router, which replaces the standard linear MoE router with a multi-layer MLP-based design, substantially increasing its expressiveness. In our experiments we find that increasing router capacity and expressiveness is a strong use of marginal parameters. A small number of router parameters controls a much larger number of expert parameters, and better routing decisions significantly reduce balancing instability and improve model quality. Third, ZAYA1-8B applies learned residual scaling to both the residual stream and the layer input at each block, which controls residual-norm growth through depth at negligible parameter and FLOP cost.

Reasoning-aware training across stages: We designed ZAYA1-8B from scratch for reasoning. Motivated by evidence that including reasoning data during pretraining can produce gains that post-training alone does not recover (Akter et al., 2025), we include long chain-of-thought (CoT) data in all pretraining phases and during midtraining. To train on reasoning traces that exceed the pretraining context length, we introduce a novel answer-preserving trimming methodology, which truncates the tail of the reasoning trace while preserving the final answer, or drops the example if the answer alone does not fit. Unlike prior length-control methods that operate during inference or RL rollout generation (Khatri et al., 2025; Yang et al., 2025), AP-trimming is applied during training-data construction.

Cascaded reinforcement learning pipeline: Post-training for ZAYA1-8B uses a four-stage RL cascade: reasoning warmup, a 400-task adaptive difficulty curriculum over the RLVE-Gym environment suite (Zeng et al., 2025b), math and code RL with test-time compute traces, and a final behavioral

![](images/51b0d58e4dea9a2cf96e0c649472b7969b6793bd959d33f448dafa86830daa20.jpg)  
Fig. 1: ZAYA1-8B with Markovian RSA test-time compute vs. substantially larger reasoning models on AIME’25, HMMT’25, and LCB-v6. Hatched bars show the boost from Markovian RSA over single-rollout ZAYA1-8B. With 0.7B active parameters and the 40K/4K Markovian RSA configuration (Section VI-C), ZAYA1-8B reaches 91.9% on AIME’25 and 89.6% on HMMT’25, narrowing the gap to larger proprietary and open-weight reasoning models. ZAYA1-8B numbers (single-rollout and TTC) are evaluated in the Zyphra harness on the pre-behavioral checkpoint after math+code+TTC RL and before the final lightweight behavioral-RL polishing stage; comparator numbers are taken from official release materials (see Table XI for sources). The final behavioral stage targets chat style, instruction following, and preference behavior rather than math/code/TTC capability.

RL stage. The cascade uses asynchronous PipelineRL (Piché et al., 2025; Khatri et al., 2025) with DPPO Binary-TV trustregion masking (Qi et al., 2026), Dr-GRPO sequence-level loss aggregation (Liu et al., 2024), MaxRL advantage estimation (Tajwar et al., 2026), and no KL regularization in the reward. Stable training required substantial precision, verifier, and data curation work, which we document throughout the report.

Test-time compute methods: We introduce Markovian RSA, a novel test-time compute method that combines the recursive candidate-aggregation structure of RSA (Venkatraman et al., 2025) with the bounded-workspace principle of Markovian Thinking (Aghajohari et al., 2025). Markovian RSA turns long reasoning into staged batched inference: each stage generates N candidates in parallel, each candidate has bounded decode length β, and aggregation prefill depends only on C carried-forward tails of length τ , not on the full reasoning history. Crucially, we also integrate Markovian RSA into training: SFT data is constructed by reshuffling expertmodel rollouts into aggregation examples, and RL stages train both expert-model and policy-self-aggregation variants. The resulting model is trained for the Markovian RSA workflow at inference and we achieve substantial performance uplift by doing so.

AMD training stack: Building on our prior work with AMD MI300X GPUs and AMD Pensando Pollara 400 networking for large-scale pretraining (Anthony et al., 2025), ZAYA1-8B was pretrained, midtrained, and supervised finetuned on this GPU/networking stack. This provides evidence that the stack can support sustained pretraining, longcontext midtraining, and supervised fine-tuning for an 8Btotal-parameter MoE reasoning model. We validate this stack at the ZAYA1-8B scale; validation for substantially larger models and broader parallelism regimes remains future work.

The remainder of this report is organized as follows: Section II describes the ZAYA1-8B architecture. Section III describes pretraining, midtraining, and answer-preserving trimming. Section IV describes the SFT stage and RL cascade, including infrastructure, precision, optimizer, and stabilitymonitoring choices. Section V reports benchmark results and comparisons. Section VI describes our test-time compute approach. Section VII concludes with observations from training and open questions.

![](images/fce9b1e6dfbc1ffdad8b2ae8de96fd128f44d4ed746a451772458c65012b432c.jpg)

<details>
<summary>scatter</summary>

| Model | Active parameters, B, log scale | HMMT26 | Total params |
| --- | --- | --- | --- |
| ZAYA1-8B | 0.7 | 71.5 | 8 |
| Nemotron-3-Nano-A3-30B | 3.0 | 75.5 | 30 |
| Mistral-4-Small-6B-119B | 6.0 | 70.5 | 80 |
| Qwen3.5-4B | 4.0 | 63.5 | 8 |
| Qwen3-Next-80B-A3B-Think | 3.8 | 79.5 | 119 |
| Intellect-3-12B-106B | 12.0 | 72.0 | 119 |
</details>

AIME'26 vs. Active Parameter Count

![](images/8f94a08c27e277359b08459f53e4e8435106c314c2a22bd8aadb902e48d98dd3.jpg)

<details>
<summary>bubble</summary>

| Model | Active parameters, B, log scale | AIME26 |
| :--- | :--- | :--- |
| ZAYA1-8B | 0.7 | 89 |
| Qwen3-Next-80B-A3B-Think | 4.5 | 90 |
| Nemotron-3-Nano-A3-30B | 4.5 | 90 |
| Mistral-4-Small-6B-119B | 6 | 86.5 |
| Intellect-3-12B-106B | 12 | 86.2 |
| Qwen3.5-4B | 4 | 84.5 |
</details>

LCB-v6 vs. Active Parameter Count   
![](images/48cd01c4db887a51aa1dde25f35d1e69d907d488fbd1aa5869113db01e2a9c3c.jpg)

<details>
<summary>bubble</summary>

| Model | Active parameters, B, log scale | LCB-v6 |
| :--- | :--- | :--- |
| ZAYA1-8B | 0.7 | 64.8 |
| Nemotron-3-Nano-A3-30B | 3.0 | 64.5 |
| Qwen3-Next-80B-A3B-Think | 3.7 | 67.8 |
| Intellect-3-12B-106B | 12.0 | 66.7 |
| Mistral-4-Small-6B-119B | 6.0 | 57.9 |
| Qwen3.5-4B | 4.0 | 55.6 |
</details>

Fig. 2: Active-parameter scaling across HMMT’26, AIME’26, and LiveCodeBench-v6. ZAYA1-8B is shown at 0.7B active parameters and compared against larger open-weight and frontier models where available. Bubble area denotes total parameter count where available.

## II. MODEL

### A. Architecture

ZAYA1-8B uses an MoE architecture with three changes relative to contemporary MoE models: (1) CCA for the attention block, (2) the ZAYA1 router, and (3) residual scaling. In our ablations, these changes improve per-parameter perplexity relative to classical MoE architectures (Shazeer et al., 2016; Fedus et al., 2022) using MLA or GQA attention and a linear router (Dai et al., 2024). CCA also improves training speed relative to GQA and MLA and reduces prefill FLOPs while maintaining comparable KV-cache compression rates.

1) Compressed Convolutional Attention (CCA): CCA performs sequence mixing in a compressed latent space using a lightweight convolutional downprojector. This reduces compute requirements for training and prefill and reduces KVcache size for long-context decoding. CCA is competitive with attention variants such as MLA and GQA (Ainslie et al., 2023; DeepSeek-AI, 2025b). ZAYA1-8B’s reasoning and longcontext performance provides evidence that CCA remains effective at this scale and can support reasoning, in-context learning (ICL), and long-range recall. CCA also supports our long-context midtraining workloads at lower compute and communication cost, which was important for training ZAYA1-8B during midtraining and RL phases. Appendix C provides additional details.

2) ZAYA1 Router: We replace the standard linear router used in many large-scale MoE models with a more expressive router. First, we use an MLP in place of the linear router. Second, we mix the router representation with the previous layer’s routing representation using Exponential Depth Averaging (EDA), a variant of Depth-Weighted Averaging (Pagliardini et al., 2024).

<table><tr><td>Property</td><td>ZAYA1-8B configuration</td></tr><tr><td>Architecture family</td><td>Decoder-only MoE Transformer, Zyphra MoE++</td></tr><tr><td>Active parameters</td><td>0.76B</td></tr><tr><td>Total parameters</td><td>8.4B</td></tr><tr><td>Transformer layers</td><td>40</td></tr><tr><td>Hidden dimension</td><td>2048</td></tr><tr><td>CCA query heads</td><td>8</td></tr><tr><td>KV heads</td><td>2</td></tr><tr><td>Head dimension</td><td>128</td></tr><tr><td>Attention variant</td><td>CCGQA with CCA preconditioner</td></tr><tr><td>Query compression</td><td>2×</td></tr><tr><td>KV-cache compression</td><td>8× relative to full multi-head attention</td></tr><tr><td>Experts per MoE layer</td><td>16</td></tr><tr><td>Routing</td><td>Top-1, no residual expert</td></tr><tr><td>Expert FFN width</td><td>4096 pre-activation / 2048 post-activation</td></tr><tr><td>Router latent dimension</td><td>256</td></tr><tr><td>Position embeddings</td><td>50% RoPE on each head</td></tr><tr><td>Tokenizer</td><td>Gemma3 tokenizer, 262,272 vocabulary size</td></tr><tr><td>Primary training hardware</td><td>AMD MI300X with Pollara networking</td></tr></table>

TABLE I: ZAYA1-8B model configuration. Exact parameter counts are shown; the rounded release convention refers to the model as 0.7B active and 8B total. Architectural constants follow the ZAYA1 base configuration used for pretraining and continued post-training.

![](images/697c575efad53f0c713deb232c275e48f98a44480f2a704b4fda2f2b37a23719.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph Input
        A["MLP"] --> B["RMSNorm"]
        C["EDA"] --> D["↓Proj"]
    end

    subgraph X
        E["QK Mean"] --> F["Headwise Conv"]
        F --> G["Depthwise Conv"]
        G --> H["Depthwise Conv"]
        H --> I["Query ↓Proj"]
        I --> J["Key ↓Proj"]
        J --> K["Value ↓Proj"]
        K --> L["Embedding"]
        M["MLP₁"] --> N["MLP₂"] --> O["MLP₃"] --> P["MLP₄"] --> Q["MLP₅"]
        R["MLP₁₆"] --> S["MLP₂ₙ"] --> T["MLP₃ₙ"] --> U["MLP₄ₙ"]
    end

    V["Input: open"] --> W["x L"]
    X["Input: the"] --> Y["Embedding"]
    Z["Output: door"] --> AA["Softmax"]
    AA --> AB["Linear"]
    AB --> AC["RMSNorm"]
    AC --> AD["⊕"]
    AD --> AE["MLP₁"]
    AD --> AF["MLP₂"]
    AD --> AG["MLP₃"]
    AD --> AH["MLP₄ₙ"]
    AD --> AI["MLP₁₆"]
    AE --> AJ["Router"]
    AF --> AJ
    AG --> AJ
    AH --> AJ
    AI --> AJ
    AJ --> AK["Self-Attention"]
    AK --> AL["CCA"]
    AL --> AM["RMSNorm"]
    AM --> AN["Embedding"]
    AO["x"] --> AP["Qk Mean"]
    AP --> AQ["Headwise Conv"]
    AP --> AR["Headwise Conv"]
    AP --> AS["Depthwise Conv"]
    AP --> AT["Depthwise Conv"]
    AP --> AU["Query ↓Proj"]
    AP --> AV["Key ↓Proj"]
    AP --> AW["Value ↓Proj"]
    AP --> AX["Time Delay"]
    AX --> AY["Height k̃"]
    AY --> AZ["RMSNorm"]
    AZ --> BA["RMSNorm"]
    BA --> BB["x L"]
```
</details>

Fig. 3: ZAYA1-8B model architecture. Two of the three main architectural changes are shown here: CCA for the attention block and the ZAYA1 router. The ZAYA1 router replaces the linear router with an MLP-based router consisting of a down-projection, EDA, and a three-layer MLP.

Given the residual stream input \(\boldsymbol { x } _ { l } \in \mathbb { R } ^ { B \times S \times D }\) , where D is the residual stream dimension, the \(\mathrm { Z A Y A 1 }\) router first downprojects the residual stream to a smaller router dimension R using a learned weight matrix \(W _ { \mathrm { d o w n } } \in \mathbb { R } ^ { R \times D }\)

\[
r _ {l} = W _ {\text { down }} x _ {l}, \tag {1}
\]

such that \(\boldsymbol { r } _ { l } \in \mathbb { R } ^ { B \times S \times R }\) . For ZAYA1-8B we set R = 256. We then apply EDA, which combines the representation with that of the previous layer using a learned coefficient \(\gamma \colon\)

\[
r _ {l} = r _ {l} + \gamma r _ {l - 1}. \tag {2}
\]

The EDA operation is followed by a three-layer MLP with GeLU activations to produce the final router scores \(s \in\) \(\mathbb { R } ^ { B \times S \times E }\) , where E is the number of experts:

\[
s _ {l} = \text { softmax } (\text { MLP } (\text { RMSnorm } (r _ {l}))). \tag {3}
\]

The scores are then used to select experts through a top-k operation:

\[
e _ {\mathrm{idx}} = \operatorname{topk} (s _ {l} + b _ {l})  , \tag {4}
\]

where \(b _ { l }\) are learned bias-balancing vectors and topk selects the \(k\) experts with the largest biased router scores for each token. In ZAYA1-8B, \(k \ = \ 1\) , so (4) reduces to selecting arg maxe \(\left( { { s _ { l , e } } + { b _ { l , e } } } \right)\) for each token. The ZAYA1 router uses a bias-balancing scheme building on (DeepSeek-AI, 2025b). Routing biases are updated using a scheme inspired by proportional–integral–derivative (PID) controllers from classical control theory (Åström & Hägglund, 2006). The router enforces balancing across a global batch of expert choices. Our PID optimizer uses AdamW internally, where the error signal passed to the optimizer is the difference between the empirical routing probability distribution and the uniform distribution. Specifically, the gradient \(\nabla b _ { l , e }\) , for expert e at layer l, is computed as:

\[
\nabla b _ {l, e} = p _ {l, e} - \frac {1}{E}, \tag {5}
\]

where \(p _ { l , e }\) is the actual fraction of tokens routed to expert e in the current batch, and \(E\) is the total number of experts. This gradient signal is then used by AdamW to update the bias terms, penalizing over-utilized experts and boosting under-utilized ones. This improved the convergence speed and stability of the PID loop relative to the classical DeepSeek implementation.

In our experiments, the MLP router and EDA improve MoE performance and make balancing (Figure 4) and expert specialization easier. The additional MLP adds some FLOPs and parameters, but parameter-matched ablations show that the router is a strong target for marginal parameters compared with the experts or attention. The added router parameters and FLOPs remain small because the MLP operates in the downprojected latent space rather than in the full embedding dimension. Figure 4 illustrates the average balancing across layers from initialization of an experiment-sized model. Empirically, reduced time to convergence translated to increased recovery speed in the face of perturbations such as data distribution shifts throughout phases of training. This yields an improved router-load entropy convergence in the reported 1.8B ablation and reduced balancing failures in our training runs compared to linear routers.

3) ZAYA1 Residual Scaling: The final architectural change in ZAYA1-8B is residual scaling. We apply a learned bias \(b _ { l }\) and gating coefficient \(\boldsymbol { \alpha } \in \mathbb { R } ^ { D }\) both to the residual stream and to the output of each layer before the residual connection:

![](images/bd16740190a98af9793eaa5135868eeb96e6b062b277ead5ba81b80f33442693.jpg)

<details>
<summary>line</summary>

| Training step | Linear | MLP + EDA | Threshold (H=0.99) |
| ------------- | ------ | --------- | ------------------ |
| 0             | 0.7    | 0.8       | 1.0                |
| 100           | 0.1    | 0.9       | 1.0                |
| 200           | 0.3    | 0.95      | 1.0                |
| 300           | 0.6    | 0.98      | 1.0                |
| 400           | 0.8    | 0.99      | 1.0                |
| 500           | 0.9    | 0.995     | 1.0                |
| 600           | 0.95   | 0.998     | 1.0                |
| 700           | 0.97   | 0.999     | 1.0                |
| 800           | 0.98   | 0.9995    | 1.0                |
| 900           | 0.99   | 0.9998    | 1.0                |
| 1000          | 0.995  | 0.9999    | 1.0                |
</details>

Fig. 4: Normalized router-load entropy, averaged over MoE layers, as a function of training step from initialization of a 1.8B-parameter experimental model. For each global batch and layer, let \(p _ { i }\) denote the fraction of routed tokens assigned to expert i, with \(E\) total experts. We report \(H ( p ) /\) ln \(E _ { \mathrm { { i } } }\) , where \(\begin{array} { r } { H ( p ) = - \sum _ { i = 1 } ^ { E } p _ { i } } \end{array}\) ln \(p _ { i }\) is the Shannon entropy of the empirical expert-load distribution.

\[
\text { Res - scale } (x) = \alpha x + \beta , \tag {6}
\]

\[
x _ {l + 1} = \operatorname{Res-scale} _ {\text { res }} (x _ {l}) + \operatorname{Res-scale} _ {\text { out }} (\text { Layer } (\text { RMSnorm } (x _ {l})))
\]

Different gating coefficients and biases are applied to the residual stream and to the layer outputs. Residual scaling lets the model downweight parts of the residual stream and control how much prior residual information is retained. In our experiments, residual scaling provides similar benefits to Qwen’s attention gating scheme (Qiu et al., 2025), without the parameter or FLOP overhead of an explicit gating matrix. Residual scaling also helps control residual-norm growth through network depth, without observing any gradient vanishing. We initialize α to ones and \(\beta\) to zeros, as this initializes the model with default residual connections. Because residual scaling adds only \(4 \times L \times D\) parameters, its parameter and FLOP overhead are comparable to LayerNorm and are negligible.

Beyond these architectural changes, we trained with 16 experts and a hidden-dimension expansion factor of 2. This relatively fine-grained expert configuration improved performance at fixed parameter count, consistent with prior work (Team et al., 2025b; DeepSeek-AI, 2025a; Dai et al., 2024; Tian et al., 2025).

Unlike many contemporary MoEs, we trained with top-k equal to 1 and without residual experts (Rajbhandari et al., 2022; DeepSeek-AI, 2025b). In our experiments, the improved routing expressiveness of the ZAYA1 router and the resulting expert specialization make a residual expert unnecessary. FLOP-matched experiments also favored top-1 over higher top-k when using the ZAYA1 router. We hypothesize that the ZAYA1 router assigns more certain expert choices, with better expert specialization, so additional experts in parallel via topk are less useful. When larger values of k are used, their contribution is further reduced by multiplication with the routing probability. ZAYA1-8B produces lower-entropy routing probabilities per token than linear routers, consistent with more confident routing. As a sanity check on expert redundancy, Appendix D measures within-layer expert subspace overlap for ZAYA1-8B and public MoE baselines. ZAYA1-8B is not an outlier toward higher expert overlap: its first-projection input overlap is 1.45× the random-subspace baseline, close to Qwen3-30B-A3B’s 1.48×, while its output-projection overlap is intermediate among the compared MoEs. For attention, we used CCGQA with a query compression rate of 2× and a KV compression rate of 8×. We applied RoPE (Su et al., 2023) to half the channels in each head, leaving the other half without position embeddings. ZAYA1-8B was trained with the Gemma3 tokenizer.

Table I summarizes core architectural hyperparameters of the final release configuration.

## III. PRETRAINING AND MIDTRAINING

ZAYA1-8B was initialized from Zyphra’s ZAYA1 base architecture and trained through pretraining, context-extension midtraining, and SFT on an AMD MI300X cluster equipped with the AMD Pensando Pollara networking stack. Full details of the base-model pretraining system, hardware, checkpointing, context parallelism, and AMD-specific optimizer and kernel work are provided in (Anthony et al., 2025).

Table II summarizes the main phases. Base pretraining used a broad web-crawl distribution with code, math, multilingual, and reasoning data mixed in progressively. The second base pretraining phase upweighted code, math, reasoning, and instruction-formatted data while still training at 4K context length. We then ran a reasoning-focused midtrain phase at 32K context for 1.2T tokens at a RoPE base frequency of 1M. This was followed by an SFT phase at 131K context for 660B tokens at a RoPE base frequency of 5M. We believe that training for a large number of tokens at longer contexts significantly improves the model’s native long-context capabilities and thus provides a stronger base for post-training and RL. The substantial reduction in prefill FLOPs we obtained through using CCA was instrumental in making this feasible at our compute scale.

Table III reports coarse data categories for the reasoningfocused midtrain and SFT. Percentages are normalized over the nonzero mixture weights in the data cards; we report only category-level proportions rather than individual dataset names. To specialize the model for reasoning and provide as strong a base for RL as possible we utilized a very high fraction of long-CoT reasoning traces in the midtrain and SFT.

For context extension, we used all-gather KV context parallelism with two ranks at 32K and eight ranks at 131K. CCA’s compressed KV representation kept activation and KVcache memory overhead low, while short asynchronous pointto-point exchanges handled the convolution and value-shift boundary conditions introduced by CCA. Across these phases, we trained with the Muon optimizer using AdamW RMS matching (Jordan et al., 2024; Liu et al., 2025).

### A. Reasoning-aware pretraining and answer-preserving trimming

Recent work suggests that introducing long chain-ofthought reasoning data during pretraining and midtraining, rather than only during post-training, can produce gains that subsequent fine-tuning does not recover (Akter et al., 2025). We follow this approach throughout ZAYA1-8B’s training pipeline: every pretraining and midtraining phase included long-CoT data and it was a majority of the mix for the midtraining phases.

Including reasoning data at short pretraining contexts creates a practical challenge: reasoning traces from strong teacher models often exceed 10K tokens, with a long tail beyond 30K. At the initial 4K context length, each example must be handled in one of three ways: (i) drop it entirely, losing the reasoning signal; (ii) truncate naively, often preserving the reasoning prefix while losing the answer and thereby training the model on reasoning that never reaches a conclusion; or (iii) preserve the answer while truncating part of the reasoning. We use the third option and call the resulting scheme answer-preserving (AP) trimming.

Given a sample containing one or more assistant messages with <think>...</think> reasoning blocks followed by a final-answer section, AP-trimming applies the following procedure to fit the sample within a target context budget C:

1) Keep unchanged. If the full conversation fits within C, retain it as-is.   
2) Trim the tail of the last reasoning block. If the conversation does not fit, truncate the final assistant turn’s reasoning trace from the tail, immediately before the answer. This preserves the start of the reasoning trace and the full answer section. The retained reasoning length is chosen so that the full sample fits within C.   
3) Drop prior reasoning blocks. For multi-turn conversations, if step 2 is insufficient, remove the <think> blocks of earlier assistant turns while preserving their answer sections, then re-apply step 2.   
4) Drop the sample. If the answer sections alone exceed C, discard the sample.

The core idea is to truncate from the tail of the reasoning trace rather than from the middle. The beginning of a reasoning trace often contains problem decomposition, planning, and exploration of multiple approaches. The tail is usually more local, consolidating the selected approach into the final answer. Removing tail tokens therefore preserves more of the planning and decomposition signal while producing partial but coherent reasoning sequences whose beginning, truncated end, and final answer remain causally aligned. The transition between truncated reasoning and the answer is distributionally artificial, but in practice we did not observe obvious artifacts: passrate evaluations on reasoning benchmarks after pretraining and midtraining remained strong, and we did not identify a truncation-specific failure mode in downstream evaluations.

<table><tr><td>Phase</td><td>Context</td><td>RoPE base</td><td>Token budget</td><td>Main emphasis</td></tr><tr><td>Base pretraining, phase 1</td><td>4K</td><td>10K</td><td>8T</td><td>Broad web, code, math, multilingual</td></tr><tr><td>Base pretraining, phase 2</td><td>4K</td><td>10K</td><td>4T</td><td>More code, math, reasoning, instruction data</td></tr><tr><td>32K midtraining</td><td>32K</td><td>1M</td><td>1.2T</td><td>Long-CoT reasoning, code, math, long-context data</td></tr><tr><td>SFT</td><td>131K</td><td>5M</td><td>660B</td><td>Chat template, reasoning, code, IF, TTC traces</td></tr></table>

TABLE II: Training recipe summary. Base-pretraining details are summarized here for context and described in detail in (Anthony et al., 2025).

<table><tr><td>Category</td><td>32K midtraining</td><td>131K SFT</td></tr><tr><td>Long-CoT reasoning traces</td><td>86.1%</td><td>75.0%</td></tr><tr><td>Web, synthetic web, multilingual</td><td>5.7%</td><td>9.8%</td></tr><tr><td>Natively long-context data</td><td>0.8%</td><td>6.4%</td></tr><tr><td>Code corpus / code SFT</td><td>3.0%</td><td>5.0%</td></tr><tr><td>Math/STEM corpus</td><td>3.0%</td><td>2.6%</td></tr><tr><td>Short instruction / few-shot data</td><td>1.4%</td><td>1.2%</td></tr></table>

TABLE III: Coarse midtraining data mixtures. The 32K context-extension mixture was trained for approximately 1.2T tokens, while SFT was trained for approximately 660B tokens; percentages denote normalized mixture weights. Individual source datasets are omitted.

a) Stage-aware re-trimming: AP-trimming is applied offline to each dataset at each context length where the data is used. As the training pipeline advances through 4K pretraining, 32K midtraining, and 131K context-extension SFT, we re-trim each dataset to the corresponding context length and progressively retain longer reasoning traces. Most reasoning datasets fit fully at 131K context, so late midtraining operates on nearcomplete traces; early pretraining uses the most aggressive trimming.

b) Relation to prior work: The closest related techniques operate during inference or RL rollout generation rather than during pretraining data construction. (Khatri et al., 2025) use forced length interruptions during RL rollouts: when a thinking trace approaches the budget, the environment appends an endof-thinking phrase that forces the model to produce a final answer. (Yang et al., 2025) use a similar mechanism for inference-time thinking-budget control. Both methods operate on rollouts during training or generation, not on training data before consumption. The closest training-data analogue is the answer-length-filtered subset of (Akter et al., 2025), which retains examples whose answer length exceeds 4K tokens as a proxy for reasoning depth. That is a selection strategy rather than a truncation strategy. AP-trimming addresses the complementary problem of using long-CoT reasoning data at training contexts shorter than the natural trace length by truncating reasoning while preserving the answer section.

## IV. POST-TRAINING

Post-training begins with SFT, followed by a four-stage RL cascade. The first three RL stages are almost entirely verifiable reasoning: a math/puzzle/TTC warmup, an RLVE-Gym adaptive difficulty curriculum, and a two-phase math+code+TTC stage. We defer general chat, style, and instruction-following optimization to the final behavioral RL stage. This ordering prioritizes capability extraction from verifiable signals before applying preference and instruction-following rewards.

Two aspects of this ordering differ from common posttraining recipes. First, reasoning RL is front-loaded: most RL compute before behavioral RL is spent on verifiable math, puzzles, synthetic environments, and code. Second, the code stage uses several synthetic auxiliary environments constructed from competitive-programming references, including code input/output prediction, code reconstruction from test cases, and falsification.

### A. Supervised Fine-Tuning

The SFT phase establishes the chat template used in subsequent post-training, improves instruction following, and continues reasoning supervision at 131K context. The stage consumed 660B tokens. We use a supervised mixture spanning chat, instruction following, code, math, reasoning, tool-calling traces, and TTC aggregation examples, but do not report individual dataset details.

Because the SFT stage trains at 131K context, packing strategy mattered. We use optimized best-fit decreasing bin packing (Ding et al., 2024) rather than naively streaming examples into fixed-length windows and truncating at arbitrary boundaries. The packer fills each 131K window with complete examples whenever possible; over-length examples are handled by dataset-specific preprocessing before packing rather than by training on arbitrary suffixes created by a fixed-boundary truncation pass. This avoided hallucination artifacts we observed when models were trained on endings of mechanically truncated packed sequences.

SFT also introduces aggregation-based examples used by Markovian RSA. These examples present the model with a problem and several candidate reasoning tails, then train it to produce a single improved solution. Section VI-B describes this construction in detail.

![](images/8d9978fcd3433f6aeac1a37b3daaf4a32c909a25ab414aeb01f30e5cce06e3d6.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["1 SFT\nCE loss\n660B tokens\n131K context\n• Sets chat template and supervised skills\n• Mixture: chat, IF, code, math, reasoning, tool traces\n• Self aggregation cold start examples\n• Answer-Preserving trimming"] --> B["2 Reasoning warmup\nverifiable reward\n232 RL steps"]
    B --> C["3 RLVE-Gym curriculum\nverifiable reward\n400 steps\n• 400 adaptive puzzle-like environments\n• Thompson/IRT calibration to 0.5 pass rate\n• Online scheduler moves difficulty up or down\n• Sampler favors least-sampled environments"]
    C --> D["4 Math + Code + TTC\nverifiable reward\n384 steps\n• 18,656 rows\n• Standard math 33.8%\n• Standard code 20.3%\n• Code aux/TTC 33.4%\n• Math TTC/RSA/PaCoRe 12.5%"]
    D --> E["5 Math + Code + TTC\nverifiable reward\n464 steps\n• 12,092 rows\n• Code-focused mix\n• Standard code 31.4%\n• Code aux/TTC 26.8%\n• Standard math 22.5%\n• Math TTC/RSA/PaCoRe 19.4%"]
    E --> F["6 Behavioral RL\nRLAIF + verifiable\n384 steps\n• 80K RM prompts × 1 epoch\n• Simple IF prompts × 1 epoch\n• Hard IFBench-like prompts × 1 epoch\n• RM score gated by binary IF checker"]
```
</details>

Fig. 5: Schematic of our post-training process for ZAYA1-8B. Post-training progressed through SFT followed by four sequential RL stages. The first stage built general reasoning capabilities on math and puzzles and was then followed by two stages of code + TTC training. The model was then polished through a short behavioral RLHF phase which focused more on chat and user interaction.

### B. Reinforcement Learning Cascade

Post-training reinforcement learning is organized as a fourstage cascade. The cascade uses a shared algorithmic spine described in Section IV-B1; individual stages differ in data, reward signal, and stage length.

1) Algorithmic Spine: All RL stages and subphases share a common algorithmic spine. Per-stage differences are confined to data, reward signal, and a small number of hyperparameters.   
a) PipelineRL: Rollout generation and gradient updates run fully asynchronously on disjoint GPU pools (Piché et al., 2025; Khatri et al., 2025). We allocate 2–5× more rollout workers than trainer workers, balanced per workload to match average response length to actor update time so that neither pool stalls. Trainer-to-rollout weight sync happens in place every 2 trainer iterations; in steady state, the rollout policy is bounded at 2 trainer updates behind the trainer policy.   
b) Trust region: DPPO Binary-TV. We replace PPO’s per-token ratio clipping with Binary Total-Variation trustregion masking (Qi et al., 2026). Tokens for which the policydivergence estimate exceeds a threshold δ are masked from the gradient while remaining tokens contribute as in standard policy-gradient updates. We use δ = 0.1 in production. We tune this threshold against preserving the reward-growth trajectory of an unconstrained baseline, selecting the largest value that did not produce unconstrained reward growth in early training. The Binary-TV variant uses a deterministic indicator over a single divergence threshold rather than the continuous TV penalty or the Top-K approximation, and adds negligible overhead relative to standard PPO.   
c) Loss aggregation: Dr-GRPO SMTSN. Loss aggregation follows Dr-GRPO (Liu et al., 2024): sequence-mean over token-sum-norm (SMTSN). Token-level losses are summed

within each rollout and then averaged across rollouts in the batch, rather than averaged per-token. This avoids the implicit length normalization in standard GRPO, which biases the gradient toward longer responses.

d) Advantage estimation: MaxRL. Advantages are computed as in (Tajwar et al., 2026). For each prompt, we sample a group of G rollouts with task rewards \(r _ { i } ~ \in ~ \{ 0 , 1 \}\) using dynamic sampling (Yu et al., 2025). The advantage normalizes by the per-prompt mean reward rather than the per-prompt reward standard deviation:

\[
\hat {A} _ {i} = \frac {r _ {i} - \bar {r}}{\bar {r}}, \tag {7}
\]

where \(\begin{array} { r c l } { \bar { r } } & { = } & { \frac { 1 } { G } \sum _ { j = 1 } ^ { G } r _ { j } } \end{array}\) 1G j=1 rj is the group mean reward. This PG corresponds to the variance-reduced MaxRL estimator (Tajwar et al. (2026), Algorithm 1), which is unbiased for a truncated maximum-likelihood objective rather than for expected reward and produces stronger gradient signal on harder prompts. We use this normalization in all RL stages except the final behavioral RL stage (Section IV-B7), which uses standard GRPO with reward standard-deviation normalization.

e) Reward shape and length reward: Task rewards are binary across the cascade, with the exception of (i) RLVE-Gym environments that yield continuous solve rates near difficulty thresholds (Section IV-B4) and (ii) the behavioral RL stage, which uses a normalized reward-model score (Section IV-B7). All RL stages except behavioral RL also include the difficultyscaled length reward of Section IV-B2, applied additively as \(\Delta r _ { i }\) to the task reward at the numerator of the advantage only — the denominator r¯ uses the unmodified task reward to avoid scale blow-up. The length-reward coefficient c ramps from a small initial value during reasoning warmup to \(c = 1 . 0\) during the math+code+TTC stage, where the production reasoning length is established.   
f) No KL in reward: The cascade applies no KL regularization to the reward; the trust region is enforced entirely by DPPO Binary-TV. In stress-testing with high KL-penalty

<table><tr><td>Stage / phase</td><td>Length</td><td>Main data</td><td>Reward signal</td></tr><tr><td>Reasoning warmup</td><td>232 steps</td><td>Math, puzzles, TTC traces</td><td>Verifiable task reward</td></tr><tr><td>RLVE-Gym curriculum</td><td>400 steps</td><td>400 adaptive environments</td><td>Environment verifier / solve rate</td></tr><tr><td>Math+Code+TTC, phase 1</td><td>384 steps</td><td>General math, code, TTC</td><td>Verifiable task reward</td></tr><tr><td>Math+Code+TTC, phase 2</td><td>464 steps</td><td>Code-focused mix</td><td>Verifiable task reward</td></tr><tr><td>Behavioral RL</td><td>384 steps</td><td>Chat/RM, simple IF, hard IF</td><td>RM score with IF gate for IF stages</td></tr></table>

TABLE IV: RL cascade summary. The first three stages emphasize verifiable reasoning. Behavioral RL is run last to tune chat, style, and instruction-following behavior.

coefficients, we observed a length-dependent bias attributable to applying a signed sequence-level log-ratio reward term to stale or mixed-policy rollouts under in-flight weight sync; Section IV-F describes the mechanism and possible mitigations. The production cascade avoids this configuration entirely by relying on the DPPO trust region alone.

g) Optimizer: All RL stages use Muon with momentum set to zero (GLM-5-Team et al., 2026), extending the GLM-5 prescription of resetting the optimizer at each weight-sync boundary into a fully momentum-free regime. Section IV-E describes this choice in detail and discusses its motivation and memory implications.

h) Hyperparameters: Across all five stages, the cascade uses minibatches of 128 prompts with rollout group size \(G ~ = ~ 1 6\) responses per prompt. Per-rollout maximum response length is 81,920 tokens, except for the first half of the reasoning-warmup stage, which uses 65K. The maximum aggregation-prompt length is 20,480 tokens, sized to fit Markovian RSA round-1 prompts containing \(C = 4\) candidate tails. Trainer-to-rollout weight sync occurs every 2 trainer iterations. The trainer requires 2 batches of completed rollouts to be available in the buffer before pulling, and the buffer is capped with oldest-sample eviction; in steady state, on-policy staleness is bounded at 2 trainer updates. Learning rates are set per stage in the range \(2 \times 1 0 ^ { - 6 }\) to \(1 \times 1 0 ^ { - 5 }\) , with the smallest values used during behavioral RL.

2) Token efficiency: To encourage concise reasoning, we combine aspects from ALP (Xiang et al., 2025) and ShortRL (Yuan et al., 2025) to create a group-relative, difficulty-scaled length reward.

Given a prompt with rollout group size \(G ,\) response reward \(r _ { i } \in \{ 0 , 1 \}\) , and response length \(\ell _ { i } ,\) we compute the group solve rate \(\begin{array} { r } { { \mathrm {  ~ \nabla ~ } } p = \frac { 1 } { G } \sum _ { i = 1 } ^ { G } r _ { i } } \end{array}\) and the shortest correct response length \(\ell _ { \mathrm { m i n } } .\) . Similar to ShortRL, we define a linear length interpolation, with the distinction that \(\ell _ { \mathrm { m a x } }\) is a constant:

\[
\begin{array}{l} \lambda_ {i} = \frac {1}{2} - \operatorname{clamp} \left(\frac {\ell_ {i} - \ell_ {\min}}{\ell_ {\max} - \ell_ {\min}}, 0, 1\right), \\ \tilde {\lambda} _ {i} = \left\{ \begin{array}{l l} \frac {1}{2} & \text { if } \ell_ {i} \leq \ell_ {\min} + T _ {\ell}, \\ \lambda_ {i} & \text { otherwise }. \end{array} \right. \tag {8} \\ \end{array}
\]

Let \(k = \textstyle \sum _ { i } r _ { i }\) denote the number of correct responses in the group. We apply the length reward only when at least two responses in the group are correct, so that there is a nontrivial comparison among correct response lengths. We adopt the following correctness and difficulty gate:

\[
m _ {i} = \mathbb {1} [ p > p ^ {*} - T _ {\mathrm{acc}} ] \cdot \mathbb {1} \left[ p > \frac {1}{G} \right] \cdot \mathbb {1} [ r _ {i} = 1 ], \tag {9}
\]

where the term \(p ^ { * }\) denotes a running maximum solve rate for the corresponding data source or environment, the condition \(p > 1 / G\) is equivalent to \(k \geq 2\) for integer-valued binary rewards, and \(T _ { \mathrm { a c c } }\) is a tolerance that prevents the length reward from activating far below the current observed capability frontier. We additionally scale the bonus by the solve rate \(p ,\) attenuating the length penalty on difficult problems and amplifying it on easier problems. The final additive reward is:

\[
\Delta r _ {i} = c \cdot p \cdot m _ {i} \cdot \tilde {\lambda} _ {i}, \tag {10}
\]

where c is a scaling coefficient. This reward \(\Delta r _ { i }\) is added to the task reward, biasing the policy toward shorter correct solutions while preserving task accuracy in our production runs.

3) Reasoning Warmup: The first RL stage is a 232-step reasoning warmup on math, puzzle, and TTC reasoning prompts. Its purpose is to adapt the SFT model to long verifiable rollouts before the broader RLVE and math+code stages. The warmup set contains 84,604 rows and is deliberately biased toward hard prompts: retained examples have prior pass rate at most 0.75, with most examples at substantially lower pass rates. Responses in this stage are long, with median replay response length around 17.6K tokens and a p90 near 30K tokens.

Rewards are verifiable and task-specific. For math problems, the reward is based on final-answer correctness after normalization. For puzzle environments, the reward is supplied by the environment verifier. TTC prompts are formatted to match the Markovian RSA workflow described in Section VI, so the model begins RL already seeing aggregation-based reasoning prompts.

4) RLVE-Gym Difficulty Curriculum: The second RL stage trains for 400 steps on 400 adaptive and verifiable problem generators from RLVE-Gym (Zeng et al., 2025a). We integrated RLVE as a dataset in VeRL (Sheng et al., 2025). Although this stage has fewer optimizer steps than the later math+code stage, the average step is roughly twice as long because responses are long, with reasoning lengths around 50K tokens. We use this stage to expose the model to a broad distribution of puzzle-like verifiable environments while keeping each environment near the model’s current difficulty boundary.

<table><tr><td>Category</td><td>Mix share</td></tr><tr><td>Reasoning-gym and reasoning-core puzzles</td><td>54.4%</td></tr><tr><td>Competition level math reasoning</td><td>31.2%</td></tr><tr><td>Enigmata puzzles</td><td>14.4%</td></tr></table>

TABLE V: Coarse composition of the reasoning-warmup RL data. Percentages are computed over 84,604 warmup rows.

During training, we used an online scheduler for problem difficulty, and we balanced environment selection using a weighted sampler for which the least sampled environments get the highest weight. Our difficulty scheduler differs slightly from the authors’ in that it uses a tighter bound on the difficulty d and allows regressions.

Let yˆ denote either the optimal solution or a reasonable heuristic when optimal solutions are intractable. We define

\[
r _ {i} = \left\{ \begin{array}{l l} 1 & \text { if } | \hat {y} - y | <   \epsilon , \\ 0 & \text { otherwise }, \end{array} \right.
\]

\[
\delta_ {i} = \left\{ \begin{array}{l l} + 1 & \text { if } \bar {r} > 0. 7 \text { and } d _ {\text { group }} = d, \\ - 1 & \text { if } \bar {r} = 0, \\ 0 & \text { otherwise }, \end{array} \right. \tag {11}
\]

\[
d \leftarrow d + \delta_ {i},
\]

where \(r _ { i }\) is reward per rollout in a group, \(\begin{array} { r } { \bar { r } = \frac { 1 } { G } \sum _ { i } r _ { i } } \end{array}\) is the group pass rate, ϵ is an environment-specific numerical tolerance used to determine whether the rollout answer \(y\) is close enough to the target yˆ to receive reward 1, d is the current difficulty setting, and \(d _ { \mathrm { g r o u p } }\) is the observed difficulty of the last computed group. We constrain updates to \(d _ { \mathrm { g r o u p } } = d\) to prevent stale difficulties from affecting the pass rate estimate.

Crucially, we use an initial tuning step to avoid training on difficulties that are too easy for the model, and we aim to maximize the information content during training by initializing all environments to a difficulty that gives a 0.5 pass rate for the model. This tuning process is an adaptive search problem, and the search space is essentially unbounded. Some environments are solvable into the range of \(d > 1 0 0\) , while others are rarely solvable even at 0. For this reason, we rely on Thompson Sampling (Thompson, 1933) as a reasonably efficient method to determine the 0.5 solve rate crossing point for every environment. We model the pass rate using the complement of the logistic curve as is commonly done in Item-Response-Theory (IRT) (Lord, 1980) and we sample from the midpoint with an ε-greedy approach. Each verified response group yields an estimate of the pass rate. A parameter pool is maintained as with Thompson Sampling and a single Gaussian prior on \(\mu\) and s is used for all environments based on empirically observed ranges.

\[
p _ {\text { success }} = \sigma \left(- \frac {d - \mu}{s}\right) = \frac {1}{1 + e ^ {(d - \mu) / s}}, \tag {12}
\]

\[
\Theta = \{(\mu_ {m}, s _ {m}) \} _ {m = 1} ^ {M}.
\]

Given a parameter pool, we perform weighted sampling proportional to the posterior weight of the candidates (initialized as uniform) in Θ and for each iteration we sample a candidate and use it to compute a difficulty d at \(p _ { \mathrm { t a r g e t } }\) :

\[
\begin{array}{l} \boldsymbol {w} = (w _ {1}, \dots , w _ {M}), \quad \sum_ {m = 1} ^ {M} w _ {m} = 1, \\ j \sim \text { Categorical } (\boldsymbol {w}), \tag {13} \\ \end{array}
\]

\[
d _ {j} = \mu_ {j} + s _ {j} \cdot \log \left(\frac {1 - p _ {\text { target }}}{p _ {\text { target }}}\right),
\]

where w is the normalized posterior-weight vector over the candidate logistic-curve parameters in Θ.

We use \(p _ { \mathrm { t a r g e t } } = 0 . 5 ,\) , the maximum Fisher information point of the logistic model (Lord, 1980), with ε-greedy exploration to \(0 . 5 \pm 0 . 2 5\) . We then perform rollouts and verification at the sampled difficulty and close the loop by updating and renormalizing the posterior:

\[
w _ {j} \propto w _ {j} \cdot \text { Binomial } (k; G, p _ {\text { success }, j}), \tag {14}
\]

\[
j \in \{1, \dots , M \}.
\]

where \(p _ { \mathrm { s u c c e s s } , j }\) is the current estimate for candidate \(j .\) . Groups are generated asynchronously using vLLM with the previous phase’s frozen model weights. If the effective sample size of the pool falls below a threshold, we resample with replacement from Θ, aggregate the observation history as a recencyweighted sum of successes and failures, then re-initialize likelihoods.

This curriculum is intended to maximize useful verifier signal. Environments that are too easy produce mostly positive groups and little policy-gradient information; environments that are too hard produce mostly negative groups. The initial calibration and online difficulty updates keep each environment near a solvable but non-saturated regime, making the stage a bridge between the narrower reasoning warmup and the broader math+code+TTC RL stage.

5) Math, Code, and Test-Time Compute RL: The third RL stage is the main capability-building stage of the cascade. It combines olympiad-level math, competitive-programming code, Markovian RSA aggregation prompts, PaCoRe continuation prompts, and synthetic auxiliary code environments. We run this stage in two phases: a 384-step general math+code+TTC phase, followed by a 464-step code-focused phase.

Table VI summarizes the two data mixtures. Phase 1 contains 18,656 rows and balances math and code while introducing TTC and PaCoRe variants. Phase 2 contains 12,092 rows and increases the code share while retaining math TTC data.

The auxiliary code environments are constructed by transforming competitive-programming references into multiple verifiable tasks per source problem. Each seed problem contains a problem statement, input/output specification, accepted reference implementations, rejected or incorrect implementations when available, and test cases. From these seeds, we construct three auxiliary task families:

<table><tr><td>Category</td><td>Phase 1: general</td><td>Phase 2: code-focused</td></tr><tr><td>Standard code prompts</td><td>20.3%</td><td>31.4%</td></tr><tr><td>Code auxiliary / code TTC prompts</td><td>33.4%</td><td>26.8%</td></tr><tr><td>Standard math prompts</td><td>33.8%</td><td>22.5%</td></tr><tr><td>Math TTC / RSA / PaCoRe prompts</td><td>12.5%</td><td>19.4%</td></tr></table>

TABLE VI: Coarse composition of the math+code+TTC RL stage. Phase 1 uses 18,656 rows; phase 2 uses 12,092 rows. Percentages are grouped from source tags and rounded.

1) CodeI/O prediction (Li et al., 2025). Given code and a set of inputs, the model predicts the outputs; in the reverse direction, given code and outputs, the model proposes inputs that produce them. Output-prediction rewards use exact normalized agreement with the reference execution. Input-prediction rewards execute the reference program on the generated input and check that the target output is produced while satisfying the input schema.   
2) CodeARC reconstruction (Wei et al., 2025). Given a problem description, input/output specification, and example test cases, the model synthesizes code. The verifier compiles or executes the generated solution and checks it against held-out tests.   
3) Falsification. Given a specification and a candidate implementation, the model must find an input that falsifies the implementation relative to the specification or a trusted correct implementation. The verifier checks that the generated input is valid and that it induces a disagreement or specification violation.

These tasks target algorithmic reasoning primitives rather than only end-to-end competitive-programming solving. CodeI/O emphasizes execution tracing and inverse reasoning over program behavior. CodeARC emphasizes synthesis from sparse behavioral evidence. Falsification emphasizes adversarial test construction and spec-implementation comparison. All three are binary-verifiable and therefore fit the same RL objective as math and puzzle prompts.

TTC prompts are included in the same RL stream. For Markovian RSA examples, the prompt contains the original problem and a small set of candidate reasoning tails. The policy generates a single aggregated solution and receives the standard verifiable reward for the final answer or produced code. This lets the stage train both ordinary single-rollout problem solving and the aggregation workflow used at inference time.

6) Agentic task scope: ZAYA1-8B does not include a dedicated multi-turn agentic RL stage in this release. We include some supervised agent, tool, and SWE traces during SFT, but the RL cascade is primarily optimized for verifiable reasoning, math, code, and instruction-following behavior. As a result, we expect agentic benchmarks such as BFCL-v4 and \(\tau ^ { 2 }\) to lag models whose post-training explicitly emphasizes multi-turn tool use. Scaling agentic data and agentic RL is left for future releases.

7) Behavioral RL: The final RL stage tunes general chat behavior, style, and instruction following after the verifiablereasoning stages have established the model’s math and code capabilities. Behavioral RL uses standard GRPO with reward standard-deviation normalization rather than the MaxRL normalization used in the verifiable-reasoning stages. It also does not use the length reward from Section IV-B2.

We first train for one epoch on 80K behavioral prompts (Wang et al., 2024, 2025). This stage improves general response quality and chat behavior without changing the reasoning-focused data distribution of the earlier stages.

We then run two instruction-following stages, each for one epoch. The first uses simpler instruction-following prompts; the second uses more difficult IFBench-like prompts. For these IF stages, the reward is gated by a binary instruction-following checker. If the completion fails the IF gate, its reward is set to zero. If it passes the gate, the completion is scored by the reward model. This prevents the reward model from assigning positive reward to fluent responses that fail the explicit instruction constraints.

### C. RL Infrastructure

a) Router replay: The single most important MoEspecific change for RL stability is router replay: the trainer reuses the expert routing assignments produced by vLLM at rollout time during its own forward pass over the rollout, rather than recomputing routing decisions from scratch. Even with the precision settings in Section IV-D, small numerical differences between the rollout engine and the trainer can produce different routing decisions for tokens near a router decision boundary. In an MoE with top-1 routing, a token routed to expert einference at rollout time but to a different expert \(e _ { \mathrm { t r a i n } } \neq e _ { \mathrm { i n f e r e n c e } }\) at gradient time produces different per-token logits, which corrupts the on-policy gradient. Router replay eliminates this source of mismatch: by pinning the trainer’s expert selection to the rollout-time decision, enforcing \(e _ { \mathrm { t r a i n } } \equiv e _ { \mathrm { i n f e r e n c e } } ,\) , the gradient is computed against the same expert sequence that produced the rollout. We discuss the SNR view of this mismatch in Section VII-C.

In practice, vLLM writes per-token and per-layer expert assignment indices to a shared memory buffer during decode. The write is overlapped with decode work to avoid slowing rollout generation. Assignments are then packed alongside the rest of the rollout batch (token IDs, masks, etc.) when the batch is shipped to the trainer, so router replay introduces no separate transport step.

b) Memory and recompute strategy: For long-rollout training the dominant memory pressure comes from activations. We combine host-side activation offloading with gradient checkpointing: the hidden state tensors from each layer that autograd must retain for backward are temporarily offloaded to CPU memory during the forward pass, while checkpointed layer interiors discard their forward activations and reconstruct them by rerunning the layer forward during backward. This trades extra backward-time compute and host-to-device traffic for substantially lower peak GPU activation memory. In this configuration we use FSDP shard size 4 under the FSDP2 sharding strategy with sequence parallelism disabled; at this model size and per-rank rollout length, the extra cross-rank communication from sequence parallelism and ring attention is not worth the memory savings.

c) Packing and dynamic batching: We use sequence packing and variable length attention for the trainer. This allows the trainer to run with dynamic microbatch sizing: rather than fixing the number of rollouts per microbatch, we fix a token budget of 131,072 tokens per GPU per microbatch and pack rollouts into microbatches up to this budget. This avoids paying for the longest rollout in a fixedrollout-count microbatch when most rollouts are shorter, and keeps GPU memory utilization stable across batches even when rollout-length distributions shift between training stages. We additionally rebalance pack assignments across GPUs so that microbatches on different ranks contain comparable token counts; without this, the slowest rank gates the entire step, since synchronous gradient accumulation must wait for all ranks to finish. With balancing, per-step variance across ranks is small enough that no rank consistently bottlenecks training.

d) Buffer management: The trainer pulls completed rollouts from a shared buffer with a maximum capacity bound and oldest-sample eviction. As described in Section IV-B1, the trainer requires 2 batches of completed rollouts to be available before pulling. This combination keeps the rollout pool from running ahead of the trainer (which would inflate staleness and waste rollout compute), while ensuring the trainer is never blocked waiting for fresh rollouts under our 2–5× rollout-totrainer ratio.

### D. Precision

The default precision regime for ZAYA1-8B RL is BF16 weights and activations, with a small set of operations promoted to FP32. The subset of operations in FP32 is identical between the trainer and vLLM, which is necessary for enginetrainer log-prob agreement within the regime needed for stable PipelineRL training (see Figure 6).

a) FP32 operation set: The following operations run with FP32 numerics on both trainer and inference paths:

Loss/output: fused cross-entropy accumulation and LMhead matmul.

• Attention/normalization: CCA cache state, QK-norm, QKmean, and RMSNorm; see Section II.

Routing/residuals: router softmax and residual stream additions.

The LM-head FP32 promotion follows precedents in (Khatri et al., 2025) and (Chen et al., 2025). The remaining FP32 ops were added incrementally to close engine-trainer logprob mismatch observed in early training runs; without them, mismatch produces grad-norm spikes and stale-policy artifacts under PipelineRL.

b) FP16 detour: Recent work argues that training– inference mismatch in RL fine-tuning can arise directly from floating-point precision, and proposes using FP16 uniformly rather than BF16 as a simple way to reduce mismatch (Qi et al., 2025). However, in our comparisons, we found that a hardened BF16 path with a small matched FP32 operation set on both the rollout engine and trainer achieved the engine– trainer agreement needed for stable PipelineRL training, while retaining BF16’s dynamic-range advantages. We therefore use BF16 weights and activations by default, promote only the operations listed above to FP32, as described previously.

c) Rollout Engine-trainer match: Figure 6 compares per-token log-probabilities computed by vLLM (used during rollout generation) and by the trainer’s prefill (used to compute gradients). At our default precision setup, the two distributions are nearly identical: KL divergence \(: = 1 . 3 \times 1 0 ^ { - 4 }\) and Pearson \(r > 0 . 9 9 9 6\) over a 128-prompt, G = 16 batch with 4K-token completions. This level of agreement is a precondition for stable PipelineRL training under our staleness regime; without the FP32 op set above, agreement degrades substantially and downstream training is unstable.

### E. Optimizer

Let \(\mathcal { L } _ { t } ( W )\) denote the actor training loss for the parameter matrix W on rollout batch t, and let \(g _ { t } = \nabla _ { W } \mathcal { L } _ { t } ( W )\) be the corresponding actor gradient. Let \(m _ { t }\) denote Muon’s first moment buffer, and let M(·) denote the Muon orthogonalization step via Newton-Schulz (Jordan et al., 2024). Standard Muon uses the update

\[
m _ {t} = \mu m _ {t - 1} + g _ {t},
\]

\[
\Delta W _ {t} = - \eta_ {t} \mathcal {M} (m _ {t}), \tag {15}
\]

where \(\eta _ { t }\) is the learning rate at optimizer step t. For actor updates, we set \(\mu = 0 \mathrm { \ s o \ m } _ { t } = g _ { t }\) and \(\Delta W _ { t } = - \eta _ { t } \mathcal { M } ( g _ { t } )\) . Thus each actor update depends on the current rollout batch and does not carry first-moment optimizer state across rollout batches. For embedding and output-head parameters, including the word embedding and LM head, we use AdamW rather than Muon. For the remaining matrix-valued actor weights, we use momentum-free Muon.

The motivation differs from pretraining. Compared to AdamW, Muon stands as a more compute efficient optimizer that is well suited to the RL setting where updates to parameters are sparse (Mukherjee et al., 2026b). Furthermore, in next-token pretraining, adjacent minibatches are drawn from a comparatively stationary data distribution, so momentum can average compatible gradient directions across steps. In RL, each actor update is tied to a rollout batch whose prompts, sampled trajectories, rewards, and generating policy snapshot may differ from neighboring batches. Following (GLM-5- Team et al., 2026), we view optimizer-state reset as a useful stability heuristic for asynchronous RL. Our setting extends this idea: instead of resetting the optimizer state only at rolloutengine weight-sync boundaries, we make every actor update momentum-free. This makes each update depend only on the current rollout batch while retaining Muon’s normalized matrix update, \(\Delta W _ { t } = - \eta _ { t } \mathcal { M } ( g _ { t } )\) , rather than a raw SGD step \(( \Delta W _ { t } ~ = ~ - \eta _ { t } g _ { t } )\) . We treat this as a practical stability and memory choice, not as evidence that zero momentum is generally optimal for RL.

![](images/b0c51ce7915751bfcad1be94a4d5fd2a077b9f382ff8610d6341e889de88bc70.jpg)

<details>
<summary>scatter</summary>

| Engine probability | Prefill probability |
| ------------------ | ------------------- |
| 0.0                | 0.0                 |
| 0.2                | 0.2                 |
| 0.4                | 0.4                 |
| 0.6                | 0.6                 |
| 0.8                | 0.8                 |
| 1.0                | 1.0                 |
</details>

![](images/d2aae21acf7e10f7e74830c56d75a995c47d15c5d20c075faceeccd0d6f65cf6.jpg)

<details>
<summary>scatter</summary>

| Engine probability | BF16+FP32 |
| ------------------ | --------- |
| 0.0                | 0.0       |
| 0.2                | 0.2       |
| 0.4                | 0.4       |
| 0.6                | 0.6       |
| 0.8                | 0.8       |
| 1.0                | 1.0       |
</details>

![](images/e6d57f85a35727d6ea32d2547ab01fccb766e05d49d54243c908458e037148e7.jpg)

<details>
<summary>scatter</summary>

| Engine probability | BF16+FP32+RR |
| ------------------ | ------------ |
| 0.0                | 0.0          |
| 0.2                | 0.2          |
| 0.4                | 0.4          |
| 0.6                | 0.6          |
| 0.8                | 0.8          |
| 1.0                | 1.0          |
</details>

![](images/0cfc073be3ee78c3a7feab7cc9df6ba837b488a0cb2ee14953795863dc150a43.jpg)  
Fig. 6: Per-token probability comparison (log scaled frequency): vLLM (engine, used for rollout generation) vs. trainer prefill (used for gradients) with incremental precision improvements. BF16: naive uniform BF16 implementation in inference and prefill. BF16+FP32: addition of selective upcasting of a subset of operations to FP32. BF16+FP32+RR: additional improvement from implementing router replay on trainer prefill from cached indices of rollout. Each point is a token from a 128-prompt, \(G = 1 6\) evaluation batch with 4K-token completions on ZAYA1-8B. Identity line shown (dashes). For BF16+FP32+RR, KL divergence = 1.3 × 10−4, Pearson \(r > 0 . 9 9 9 6 .\) .

This choice also avoids maintaining a persistent firstmoment buffer for the Muon-updated actor weights during RL, reducing optimizer-state memory relative to momentum Muon. We did not include a controlled optimizer ablation in this report. A direct comparison against momentum Muon, AdamW, and SGD updates is left for future work.

### F. Monitoring and maintaining stability

Reward and KL diagnostics describe the policy’s optimization dynamics but do not reflect the content of generated rollouts. We monitor a small set of auxiliary rollout-level statistics during RL training to fill this gap. A subset of these statistics also act as reward gates, zeroing a rollout’s task reward when its content is flagged as degenerate.

a) Streaming compressibility: Our primary canary is a sliding-window LZ77 compressibility metric computed per chunk on the raw token-ID bytes of each rollout. Compression uses zlib with a \(\mathsf { 2 ^ { 1 0 } } ^ { \mathsf { ^ { - } } } = \mathsf { 1 0 2 4 { - } b y t e }\) LZ77 window \(( \mathrm { w b i t s } = - 1 0 )\) , level-1 deflate, and Z\_SYNC\_FLUSH between chunks; the compressor is stateful, so each chunk’s compression ratio reflects compressibility relative to recent history bounded by the LZ77 window rather than whole-sequence redundancy. Each rollout is divided into fixed-size chunks of C tokens (with the final short chunk merged into its predecessor to avoid \(\Sigma \_ S \Upsilon \mathrm { N C } _ { - }\) \_FLUSH overhead inflating short-tail ratios), and the per-chunk compression ratio

\[
r _ {c} = \frac {\text { compressed   bytes } _ {c} - \text { flush   overhead }}{\text { raw   token - ID   bytes } _ {c}} \tag {16}
\]

is computed for each chunk c.

A small \(r _ { c }\) indicates a chunk that compresses well against its preceding context, which is the signature of degenerate repetition or copying: the model has emitted a span of tokens already present in the LZ77 window. More generally, as is noted by (Lee et al., 2026), an effective compression algorithm also serves as a computable upper bound on Kolmogorov Complexity (Li & Vitányi, 2019), and both ends of the compressibility spectrum could arguably be filtered as either low information content or purely random. We choose LZ77 in particular over simpler n-gram or token counting methods because it takes into account sequence-level matching within the window, whereas language and domain-level n-gram biases can complicate simpler presence/frequency metrics. We flag a rollout if any chunk satisfies \(r _ { c } < \tau _ { \mathrm { r e p e a t } } .\) with a conservative \(\tau _ { \mathrm { r e p e a t } } = 0 . 0 5\) in production. Flagged rollouts have their task reward zeroed before advantage computation, so the policy receives no positive learning signal for producing degenerate text even when the verifier accepts the (technically correct) final answer at the end of a long repetitive trace. The per-chunk granularity allows reward zeroing on rollouts where degenerate spans appear at any position rather than attempting to rely on coarser, full response compressibility.

b) Rare-token monitoring: As an independent signal, we track the fraction of tokens in each rollout whose token IDs fall in the top X% of the tokenizer’s ID range. This is a lightweight proxy for unusual or rarely used tokens in our tokenizer. In production monitoring we track several cutoffs, including 10%, 5%, 2%, and 1%, and use the top-10% token-

ID region for gibberish canaries. A rising rare-token fraction often precedes other failure indicators and is cheap to compute.

c) Operational use: The low-ratio repetition canary and rare-token-fraction statistics are computed per batch during RL training and visible alongside reward and KL in WandB. The repetition canary additionally runs as a reward-zeroing gate: rollouts that exceed the low-ratio threshold have their rewards zeroed before advantage computation, regardless of verifier outcome. Canary signals do not adjust learning rate or any other optimizer setting.

d) Length bias from signed KL-in-reward under pipeline RL: Beyond rollout-level canaries, we also monitored response-length growth, which exposed an interaction between PipelineRL training and a sequence-level signed logratio reward penalty. In early stress tests combining the two, we observed runaway response-length growth: rollouts grew progressively longer over training without corresponding reward improvement. Our working explanation is specific to this estimator and aggregation choice. In pipeline RL, long completions can span multiple generator-policy snapshots: early tokens may be sampled from a stale generator policy \(\pi _ { \mathrm { g e n } , c }\) that is \(\Delta _ { c }\) trainer updates behind the current actor πθ, while later tokens may be sampled from fresher snapshots with smaller \(\Delta _ { c }\) .

The commonly used \(K _ { 1 }\) -estimator log-ratio KL term is

\[
l _ {t} = \log \pi_ {\theta} (y _ {t} \mid h _ {t}) - \log \pi_ {\text { gen }, c (t)} (y _ {t} \mid h _ {t}). \tag {17}
\]

For tokens sampled from \(\pi _ { \mathrm { g e n } , c ( t ) }\) , this signed log-ratio is negative in expectation whenever the current policy differs from the generator policy:

\[
\mathbb {E} _ {y _ {t} \sim \pi_ {\text { gen }, c (t)}} [ l _ {t} ] = - D _ {\mathrm{KL}} \left(\pi_ {\text { gen }, c (t)} (\cdot | h _ {t}) \| \pi_ {\theta} (\cdot | h _ {t})\right) \leq 0. \tag {18}
\]

For fresh tokens with small policy lag, \(l _ { t } \approx 0\) . If these terms are aggregated into a sequence-level scalar,

\[
S _ {\text { seq }} = \sum_ {t} l _ {t}, \tag {19}
\]

and subtracted from reward as

\[
A = r - \beta_ {\mathrm{KL}} S _ {\text { seq }}, \tag {20}
\]

then stale off-policy tokens can create a positive reward offset. Longer completions can accumulate more negative signed logratio terms, and when the resulting sequence-level adjusted advantage is broadcast back to all tokens, stale-prefix terms can affect the learning signal assigned to later suffix tokens.

This produces a length-dependent bias through two interacting effects.

Stale-prefix contamination. Longer sequences can contain more stale prefix tokens contributing negative \({ { l } _ { t } } ,\) making \(S _ { \mathrm { s e q } }\) more negative. Since the signed log-ratio term enters as \(- \beta _ { \mathrm { K L } } S _ { \mathrm { s e q } }\) , the negative sequence sum acts as a positive reward offset, inflating the advantage for longer sequences independent of task quality.

Staleness-dependent penalty scale. The magnitude of the signed log-ratio can also depend on chunk staleness \(\Delta _ { c }\) . Relatedly, Bartoldson derives a first-order EMA-reference approximation for asynchronous RL in which the log-ratio between the current policy and a ∆-old inference policy can be interpreted as a surrogate for KL regularization against an EMA reference, under local linearity and first-order Taylor assumptions (Bartoldson, 2026). This suggests that \(\Delta\) can change the effective scale of a stale-policy log-ratio penalty. In our setting, we use this only as intuition for lag-dependent penalty strength; the length-bias mechanism itself follows from applying a signed off-policy log-ratio at sequence level and subtracting it from reward.

This mechanism should be distinguished from true KL regularization. A KL divergence is non-negative by construction, whereas the sampled signed log-ratio \(l _ { t }\) can be arbitrarily negative on individual samples and is negative in expectation under the generator distribution when \(\pi _ { \mathrm { g e n } , c ( t ) } \neq \pi _ { \theta }\) . The severity depends on both absolute staleness and withinsequence policy heterogeneity. In-flight synchronization can create prefix–suffix heterogeneity by allowing one completion to span multiple generator snapshots; holding the generator fixed for the entire completion removes this specific coupling, but does not remove the off-policy signed-log-ratio length offset if the fixed generator is stale relative to the trainer.

e) Possible mitigations: Two practical mitigations target the specific stale-prefix coupling described above. Chunklocal signed-log-ratio isolation aggregates the signed log-ratio within each chunk rather than across the full sequence, so stale-prefix terms do not directly contaminate the advantage assigned to fresher suffix chunks:

\[
\mathcal {A} _ {c} = A _ {\text { reward }} - \beta_ {\mathrm{KL}} S _ {c}, \quad S _ {c} = \sum_ {t \in c} l _ {t}. \tag {21}
\]

This localizes the bias but does not by itself turn the off-policy signed log-ratio into a true KL penalty.

Staleness rescaling is an additional heuristic: divide the chunk term by an empirical staleness scale \(g ( \Delta _ { c } )\) , with \(g ( \Delta _ { c } ) > 0\) , to reduce variation in effective penalty strength across chunks generated at different lags:

\[
\mathcal {A} _ {c} = A _ {\text { reward }} - \beta_ {\mathrm{KL}} \cdot \frac {1}{g (\Delta_ {c})} \sum_ {t \in c} l _ {t}. \tag {22}
\]

A simple first-order choice is \(g ( \Delta _ { c } ) = \operatorname* { m a x } ( 1 , \Delta _ { c } )\) , motivated by the local-linear lag dependence in Bartoldson’s EMA approximation (Bartoldson, 2026), but the correct scale is implementation- and dynamics-dependent.

For ZAYA1-8B we did not implement either mitigation in production. Instead, we removed KL-in-reward entirely and rely on the DPPO Binary-TV trust region (Section IV-B1) for trust-region enforcement. This was sufficient for the trainingstability properties we required and avoided tracking chunk boundaries and per-chunk generator staleness. We document the mechanism here because it may arise in asynchronous or pipeline RL systems that combine stale or mixed-policy rollouts, a signed K1-estimator log-ratio in the reward, sequencelevel aggregation, and broadcast of the resulting adjusted advantage.

## V. RESULTS

Results are organized into three tables. Table VII compares ZAYA1-8B against open-weight reasoning models at comparable scale. Table VIII extends to open-weight models in the 26B–119B total-parameter range. Table XI reports testtime compute comparisons against open-weight models in the 235B–671B range plus Gemini-2.5 Pro and GPT-5-High.

### A. Evaluation Protocol

Unless otherwise noted, ZAYA1-8B results are measured with the Zyphra evaluation harness. In-class comparator models are run in the same harness when feasible, using each model’s recommended sampling settings from its model card. For ZAYA1-8B single-rollout reasoning evaluations, we use temperature 1.0, top-p 0.95, top-k -1, and benchmark-specific maximum generation lengths. For thinking-mode Qwen comparators, we mirror the recommended thinking-mode settings from the corresponding model card. Results reported from external release materials are marked with †. TTC evaluations use the checkpoint immediately following the math+code+TTC RL stage and before the final behavioral-RL polishing stage; the latter targets chat style, instruction following, and preference behavior rather than additional math/code/TTC capability.

For pass-rate evaluations in the Zyphra harness, we report averages over multiple samples per problem. Math benchmarks, including AIME, HMMT, IMO-AnswerBench, and APEX-shortlist, are reported as avg@64. Code benchmarks, including LiveCodeBench-family tasks, are reported as avg@16. GPQA-Diamond and \(\tau ^ { 2 }\) are reported as avg@16 unless otherwise noted. MMLU-Pro, BFCL-v4, HLE, IFEval, IFBench, EQBench, and Creative Writing are reported as mean@1 or as the benchmark’s standard single-run score. We use avg@k to mean the mean correctness over k independently sampled completions, estimating single-sample pass rate under the stated sampler; it is not best-of-k/pass@k unless explicitly stated. Markovian RSA results use the TTC protocol in Section VI; its token counts are total newly generated decode tokens and exclude prompt/prefill tokens. Results from external release materials may use different sampling and reporting protocols and are marked with †.

### B. Main Results: In-Class Comparison

Table VII compares ZAYA1-8B against Qwen3-4B-Thinking-2507, Qwen3.5-4B, and Gemma-4-E4B-it.

### C. Scaling Comparison: Larger Open-Weight Models

Table VIII compares ZAYA1-8B against larger openweight reasoning models: Arcee-Trinity-Mini, Nemotron-3- Nano, OLMo-3.1-32B-Think, Qwen3-Next-80B-A3B-Think, Intellect-3, and Mistral-Small-4-119B-2603.

### D. Test-Time Compute Scaling

Table XI compares ZAYA1-8B with Markovian RSA testtime compute against substantially larger reasoning models. With the headline Markovian RSA configuration \(( \beta = 4 0 \mathsf { K } ,\) τ = 4K, T = 2, N = 16, C = 4), ZAYA1-8B reaches 91.9 on AIME’25 and 89.6 on HMMT’25 Feb.

### E. Effect of Post-Training

To quantify the effect of post-training, we compare the 131K SFT checkpoint against the final ZAYA1-8B checkpoint using the same evaluation harness and sampling settings in Table IX. This comparison measures the aggregate effect of the RL cascade rather than isolating the contribution of each individual stage. We do not report per-stage ablations in this release.

## VI. TEST-TIME COMPUTE

Test-time compute (TTC) scaling — increasing inference compute per problem to improve answer quality — has become an important axis of capability scaling for reasoning models, alongside model scale and training compute. Two recent lines of work motivate the design space considered here. (Venkatraman et al., 2025) introduce Recursive Self-Aggregation (RSA), a TTC scheme that maintains a population of candidate reasoning chains and refines them through repeated aggregation: at each iteration, the model is shown a random subset of candidates and produces an improved candidate, which seeds the next iteration’s population. Empirically, RSA allows smaller open-weight models to approach the performance of larger reasoning models when given sufficient inference compute. (Aghajohari et al., 2025) introduce Markovian Thinker, a reformulation of the RL thinking environment in which the policy reasons in fixed-size chunks with bounded carryover state between chunks, decoupling thinking length from context size. Their key observation is that longcontext reasoning can be factorized in a Markovian way: with sufficient training, a model can sometimes carry forward only the information needed in a bounded textual state and continue reasoning indefinitely.

We introduce Markovian RSA, a TTC method that combines RSA’s recursive candidate aggregation with the boundedworkspace principle of Markovian Thinker. We integrate it into ZAYA1-8B’s training pipeline so the model is trained to use the same workflow at inference. The method has three components: an algorithm that includes both RSA and Markovian-Thinker for chunked reasoning as special cases (Section VI-A), a training-time integration that supplies verifier-free aggregation examples for SFT and verifiable aggregation prompts for RL (Section VI-B), and an inference-time scaling profile with bounded per-iteration aggregation context, capped attention costs, and predictable throughput (Section VI-C).

### A. Markovian RSA

a) Algorithm: Given a problem q and a base policy π, Markovian RSA proceeds over T aggregation rounds, indexed \(t = 0 , 1 , \ldots , T .\) Each round maintains a population of N candidate reasoning traces. At round \(t \ = \ 0 ,\) , the model generates N independent rollouts directly from q, each with a per-rollout thinking budget β. Each rollout’s reasoning trace is then reduced to its last τ tokens, which we call the tail. We

![](images/c4b2c77b0b66db62f5f14640d75698b094a325fe72183af4eab8d295bdd127ca.jpg)  
ZAYA1-8B (0.7/8B)   
Arcee-Trinity-Mini (3/26B)   
active params total params   
NVIDIA Nemotron 3 Nano (3/30B)   
M Mistral-4-Small (6/119B)   
M Intellect-3 (12/106B)

Fig. 7: Comparison of ZAYA1-8B performance against open-weight reasoning models on various evaluations. The under-bar plots model sizes in active and total parameters on a log scale to give a sense of the scale of the various models. 

<table><tr><td colspan="2">Active / Total</td><td>0.7B / 8.0B</td><td>4.0B / 4.0B</td><td>4.0B / 4.0B</td><td>4.0B / *8.0B</td></tr><tr><td>Category</td><td>Benchmark</td><td>ZAYA1-8B</td><td>Qwen3-4B-Thinking-2507</td><td>Qwen3.5-4B</td><td>Gemma-4-E4B-it</td></tr><tr><td rowspan="4">Math</td><td>AIME&#x27;26</td><td>89.1</td><td>79.0</td><td>84.5</td><td>50.3</td></tr><tr><td>HMMT&#x27;26 Feb.</td><td>71.6</td><td>53.6</td><td>63.6</td><td>32.1</td></tr><tr><td>IMO-AnswerBench</td><td>59.3</td><td>51.6</td><td>48.7</td><td>27.3</td></tr><tr><td>APEX-shortlist</td><td>32.2</td><td>17.1</td><td>21.35</td><td>6.1</td></tr><tr><td>Code</td><td>LiveCodeBench-v6</td><td>64.8</td><td>54.9</td><td> \(55.8^{\dagger}\) </td><td>54.2</td></tr><tr><td rowspan="2">Knowledge</td><td>GPQA-Diamond</td><td>71.0</td><td>66.1</td><td>76.2</td><td>57.4</td></tr><tr><td>MMLU-Pro</td><td>74.2</td><td>74.3</td><td>79.7</td><td>70.2</td></tr><tr><td rowspan="2">Instruction</td><td>IFEval</td><td>85.6</td><td>86.8</td><td>89.8</td><td>88.5</td></tr><tr><td>IFBench</td><td>52.6</td><td>52.9</td><td>59.2</td><td>42.7</td></tr><tr><td rowspan="2">Style &amp; chat</td><td>EQBench</td><td>73.0</td><td>79.6</td><td>79.5</td><td>80.2</td></tr><tr><td>Creative Writing v3</td><td>63.0</td><td>58.6</td><td>72.9</td><td>83.8</td></tr><tr><td rowspan="2">Agentic</td><td>BFCL-v4</td><td>40.5</td><td>49.7</td><td>45.2</td><td>31.7</td></tr><tr><td> \(\tau^{2}\) </td><td>36.3</td><td>52.9</td><td>82.1</td><td>37.7</td></tr></table>

∗Gemma4 includes 4B additional embedding parameters as a part of its total.   
†Qwen3.5-4B LiveCodeBench-v6 scores taken from release materials.

TABLE VII: In-class comparison against models of comparable sizes. ZAYA1-8B used the following sampling settings: T =1.0, top-p=0.95, top-k disabled for math, knowledge, and instruction; T =0.6, top-p=0.95, top-k=20 for code, agentic, and style. We used the recommended sampling settings in the model cards for the other models in this table. EQBench and Creative Writing v3 use the official judge, anthropic/claude-3.7-sonnet.

<table><tr><td>Model</td><td>Active</td><td>Total</td><td>AIME&#x27;26</td><td>HMMT&#x27;26 Feb.</td><td>LCB-v6*</td><td>IFEval</td><td>GPQA-D</td><td>MMLU-Pro</td></tr><tr><td>ZAYA1-8B</td><td>0.7B</td><td>8B</td><td>89.1</td><td>71.6</td><td>64.8</td><td>85.6</td><td>71.0</td><td>74.2</td></tr><tr><td>Arcee-Trinity-Mini</td><td>3B</td><td>26B</td><td>59.6</td><td>36.9</td><td>33.3</td><td>62.0</td><td>46.8</td><td>70.6</td></tr><tr><td>Nemotron-3-Nano-30B-A3B</td><td>3B</td><td>30B</td><td>90.1</td><td>75.5</td><td>64.6</td><td>92.8</td><td>75.1</td><td>78.9</td></tr><tr><td>OLMo-3.1-32B-Think</td><td>32B</td><td>32B</td><td>78.9</td><td>50.6</td><td>58.3</td><td>93.2</td><td>59.6</td><td>75.8</td></tr><tr><td>Qwen3-Next-80B-A3B-Think</td><td>3B</td><td>80B</td><td>90.2</td><td>79.3</td><td>67.8</td><td>88.5</td><td>76.7</td><td>82.6</td></tr><tr><td>Intellect-3</td><td>12B</td><td>106B</td><td>86.3</td><td>72.3</td><td>66.8</td><td>81.2</td><td>74.6</td><td>82.3</td></tr><tr><td>Mistral-Small-4-119B</td><td>6B</td><td>119B</td><td>86.4</td><td>70.6</td><td>57.9</td><td>84.0</td><td>77.2</td><td>81.6</td></tr></table>

TABLE VIII: Scaling comparison against larger open-weight reasoning models, ordered by total parameter count. All numbers are run on the Zyphra evaluation harness.

∗LCB-v6 denotes the 2025-02–2025-05 LiveCodeBench-v6 split. 

<table><tr><td>Benchmark</td><td>SFT checkpoint</td><td>Final ZAYA1-8B</td><td> \(\Delta\) </td></tr><tr><td>AIME&#x27;26</td><td>68.30</td><td>89.10</td><td>+20.80</td></tr><tr><td>HMMT&#x27;26 Feb.</td><td>39.20</td><td>71.60</td><td>+32.40</td></tr><tr><td>LiveCodeBench-v6</td><td>54.80</td><td>64.84</td><td>+10.04</td></tr><tr><td>GPQA-Diamond</td><td>59.30</td><td>71.00</td><td>+11.70</td></tr><tr><td>MMLU-Pro</td><td>70.10</td><td>74.20</td><td>+4.10</td></tr><tr><td>IFEval</td><td>66.60</td><td>85.58</td><td>+18.98</td></tr><tr><td>IFBench</td><td>30.20</td><td>52.56</td><td>+22.36</td></tr><tr><td>EQBench</td><td>57.80</td><td>72.95</td><td>+15.15</td></tr><tr><td>Creative Writing v3</td><td>46.72</td><td>62.97</td><td>+16.25</td></tr><tr><td>BFCL-v4</td><td>33.41</td><td>40.50</td><td>+7.09</td></tr><tr><td> \(\tau^{2}\) </td><td>32.88</td><td>36.30</td><td>+3.42</td></tr></table>

TABLE IX: Aggregate effect of post-training. SFT and final ZAYA1-8B checkpoints are evaluated with the same harness and benchmark-specific sampling settings. This table reports the aggregate effect of the post-training recipe; it is not a per-stage ablation.

write tailτ \(. ( y )\) for the operation that returns the final τ tokens of reasoning trace y, with \(\tau \leq \beta\) .

For rounds \(t \geq 1\) , the algorithm operates on tails from the previous population. To generate each new candidate, it samples \(C \leq N\) tails uniformly at random, concatenates them into an aggregation prompt, and asks the model to reason over the candidate solutions and produce a single improved solution. The model generates a new reasoning trace under the same per-rollout budget \(\beta .\) The trace is again reduced to its final \(\tau\) tokens, and the resulting tail enters the population for round t. This process repeats until round T , after which the final answer is extracted from the final round’s outputs using the standard answer-extraction procedure. The aggregation prompt simply asks the model to consider the candidates and produce the best solution; it does not require specialized parsing or verifier feedback.

Both Markovian RSA and full-chain RSA bound per-rollout generation cost: \(\beta\) caps the number of tokens any single candidate generates. The difference is what gets passed forward. Full-chain RSA passes the full reasoning chain\*, so the aggregation prompt at round \(t \geq 1\) contains \(C\) chains, each

with length up to \(\beta .\) Markovian RSA passes only the final τ tokens of each chain, with \(\tau \le \beta\) chosen independently. This decouples per-rollout thinking depth from aggregationcontext size: \(\beta\) controls how long each candidate may reason, while τ controls how much of that reasoning is carried into the next round. Setting \(\tau \ \ll \ \beta\) allows larger per-rollout thinking budgets while keeping aggregation prompts small. As a result, decode-attention cost, prefill-attention cost, and KVcache footprint are bounded by configuration constants rather than by reasoning length.

b) Default configuration: For ZAYA1-8B, we use \(( N , C , T ) = ( 1 6 , 4 , 2 )\) with \(\beta\) set per workload and τ chosen as a fraction of \(\beta\) (typically \(\tau \leq \beta / 2 )\) . Both β and τ can be tuned per deployment to trade off per-round thinking depth against total inference budget.

c) Inference profile: Markovian RSA changes the inference workload from a single long, position-growing decode into a sequence of bounded-context batched decoding stages. At round 0, the model generates N independent candidates from the original problem, so decode runs at batch size N rather than batch size 1. At each later aggregation round, the model again generates N candidates, but each candidate conditions only on the problem and C carried-forward tails of length at most τ . Thus the aggregation prefill length is

bounded by

\[
L _ {\text { prefill }} \leq | q | + C \tau + O (1), \tag {23}
\]

and the per-candidate decode length is bounded by β, independent of the total amount of reasoning generated across all rounds. This gives a stable serving profile: prefill is short and predictable at every stage, decode uses high-throughput batched generation, and no stage attends over the full reasoning history.

This profile differs from both single-rollout long-CoT and full-chain RSA. A single long rollout has batch size 1 and a decode position that grows with the full reasoning length. Full-chain RSA supports batched candidate generation, but its aggregation prefill grows with Cβ because it forwards full reasoning chains. Markovian RSA keeps the batched candidate-generation structure of RSA while replacing fullchain forwarding with bounded tail forwarding, so increasing β increases per-candidate thinking depth without increasing aggregation-context length.

d) Special cases: Markovian RSA contains several common TTC regimes as special cases:

• Parallel sampling / N responses. Setting T = 0 removes aggregation and produces N independent responses. If a verifier, answer-selection rule, or external scoring model is applied to these responses, this reduces to a Best-of-N evaluation; otherwise it is simply parallel sampling.   
• Full-chain RSA. Setting \(\tau = \beta\) forwards each full reasoning chain between rounds, recovering RSA. In this limit, aggregation prefill grows with the full reasoning budget.   
• Delethink bounded continuation. Setting C = 1 removes cross-candidate aggregation while retaining bounded carryover. Each candidate continues from its own tail, giving a parallel version of Markovian/Delethink chunked reasoning. This isolates the effect of bounded continuation from the additional effect of cross-candidate aggregation.

e) Comparison with PaCoRe: (Hu et al., 2026) introduced PaCoRe, a related multi-round parallel-reasoning scheme that also bounds per-round aggregation context. Pa-CoRe compacts each trajectory by extracting its final-answer or conclusion section and passing this extracted message forward between rounds. Markovian RSA instead passes the final τ tokens of the reasoning trace itself as the carryforward state, regardless of whether the trajectory reached a final conclusion. The two methods share the same goal of bounding aggregation context across rounds and differ in compaction mechanism: PaCoRe uses model-structured finalanswer extraction, while Markovian RSA uses a fixed-size suffix of generated reasoning.

In practice, we also evaluate a PaCoRe hybrid compaction variant: when a candidate reaches a post-think answer section, we pass that compact answer forward; otherwise, we fall back to passing the partial reasoning chain. This hybrid keeps the compact-message advantage of PaCoRe when candidates finish, while avoiding the need to set \(\beta\) large enough for every branch to reach a final answer.

### B. Training-Time Integration

A TTC method may be more effective when the model is trained on the workflow it uses at inference. Markovian RSA’s aggregation prompt presents the model with a problem and several candidate reasoning tails, then asks it to produce a single improved solution. This behavior is rare in standard reasoning-model training data, where each example typically consists of one problem and one solution. To train ZAYA1- 8B for Markovian RSA scaling, we construct aggregationbased examples from existing expert-model reasoning data and include them in SFT and RL.

a) SFT data construction from expert rollouts: Many open-source reasoning datasets used during midtraining and SFT include multiple expert-model rollouts per problem, often with \(n \ = \ 8\) rollouts (e.g., OpenMathReasoning, rStar-Coder, internal reasoning gym and enigmata data). For each problem q with rollouts \(\{ y _ { 1 } , \ldots , y _ { n } \}\) from a teacher model, we construct a round-0-to-round-1 aggregation example as follows: sample C rollouts from the n available; extract their tails \(\left\{ \operatorname { t a i l } _ { \tau } ( y _ { i _ { 1 } } ) , \dots , \operatorname { t a i l } _ { \tau } ( y _ { i _ { C } } ) \right\}\) ; form an aggregation prompt containing q and the C tails; and condition the teacher to produce a new aggregated rollout under the same prompt. The resulting aggregated rollout, including its reasoning trace and final answer, becomes the SFT target.

This construction has two practical advantages. It is offline and reuses existing rollout pools: no new expert-model inference is needed for each round-0 sample beyond the aggregation step. It also does not require a verifier: the teacher’s aggregated rollout is used as the target regardless of whether the underlying answer is verifiable. This makes the technique applicable to puzzle, code, and reasoning domains where the post-think content is itself the answer and where finalanswer-only aggregation strategies such as PaCoRe’s message compaction are not directly applicable.

b) RL stage integration: During RL, Markovian RSA examples are folded into the standard prompt distribution and treated like other RL prompts. Two variants are used during the math+code+TTC stage (Section IV-B5):

• Expert-aggregation. Round-1 prompts are constructed from expert-model rollouts as described above. The policy generates an aggregated rollout and is rewarded against the verifiable target.   
• Self-aggregation. For prompts where rollouts from the current SFT checkpoint or a prior-stage RL checkpoint are available, round-1 prompts are constructed from those self-rollouts. The policy aggregates over its own reasoning traces, or over traces from its predecessor.

In both variants, the aggregation example is a standard RL prompt: the policy generates a single rollout, and verifiable reward is applied to its final answer. No special multi-round RL machinery is required; the round structure is encoded in the prompt construction rather than in the gradient update. We currently train on round-1 self-aggregation. Round-2-andbeyond self-aggregation, where the policy aggregates rollouts from a prior-stage version of itself in an online buffer, is a natural extension left for future work.

<table><tr><td>Method</td><td>Decode profile</td><td>Aggregation/context state</td><td>Special case</td></tr><tr><td>Single long-CoT</td><td>BS=1, position grows with total reasoning</td><td>Full prior trace</td><td>-</td></tr><tr><td>Parallel sampling / N responses</td><td>BS=N, one round</td><td>No aggregation</td><td>Markovian RSA with T = 0; becomes Best-of-N only with an external selector or verifier</td></tr><tr><td>Delethink continuation</td><td>BS=N, chunked rounds</td><td>One bounded tail</td><td>Markovian RSA with C = 1</td></tr><tr><td>Full-chain RSA</td><td>BS=N, aggregation rounds</td><td>C full chains, length up to Cβ</td><td>Markovian RSA with τ = β</td></tr><tr><td>Markovian RSA</td><td>BS=N, aggregation rounds</td><td>C tails, length up to Cτ</td><td>General case</td></tr></table>

TABLE X: Inference-profile view of Markovian RSA. Markovian RSA preserves the batched candidate-generation structure of RSA while bounding the state forwarded between rounds. Setting \(T = 0\) gives N independent responses with no aggregation; this becomes Best-of-N only if an external selector, verifier, or answer-selection rule is applied. Setting \(C = 1\) gives the Delethink bounded-continuation regime, and setting \(\tau = \beta\) recovers full-chain RSA.

![](images/9d469e8ab7c2f7a2f4b144fd2c0d07d2653eb4b99003f36c541aa3a9248a42de.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    subgraph LLM
        A["Population P_t N candidates"]
        B["..."]
        C["..."]
        D["..."]
        E["..."]
        F["..."]
    end

    subgraph Tails
        G["Extract tail (τ)"]
        H["Aggregation Prompt 1<br>Question + C candidate tails<br>Question tail_{1:1} tail_{1:2} ... tail_{1:C}"] --> I["LLM"]
        J["Aggregation Prompt 2<br>Question + C candidate tails<br>Question tail_{2:1} tail_{2:2} ... tail_{2:C}"] --> I
        K["Aggregation Prompt N<br>Question + C candidate tails<br>Question tail_{N:1} tail_{N:2} ... tail_{N:C}"] --> I
    end

    I --> L["Extract tail (τ)"]
    I --> M["Population P_{t+1} N candidates"]
    I --> N["N prompts, each with C tails<br>(context size = C_τ)"]
    style L fill:#000,stroke:#000,color:#fff
    style M fill:#000,stroke:#000,color:#fff
    style N fill:#000,stroke:#000,color:#fff
```
</details>

Fig. 8: One round of Markovian RSA. From a population of N candidate reasoning traces (left), we extract the final τ tokens of each trace as its tail. To produce each new candidate for the next round, we sample C tails uniformly at random and present them to the model as candidate solutions in an aggregation prompt. The model produces a new reasoning trace, whose tail joins the next round’s population. Aggregation context size and per-round attention cost depend only on C and \(\tau ,\) and are independent of the per-rollout thinking budget \(\beta .\) .

c) Domain coverage: Aggregation-based training data is included for math, code, reasoning gym, and enigmata puzzle problems. Directly aggregating over reasoning tails is useful in domains where the post-think content is the answer rather than a separate boxed result. This allows the same approach to apply across domains regardless of answer format.

### C. Inference-Time Scaling

We evaluate Markovian RSA’s inference-time scaling along two axes — per-rollout reasoning budget \(\beta\) and tail size \(\tau -\) with the configuration described in Section \(\operatorname { V I - A } \left( N , C , T \right) =\) (16, 4, 2). The sampling settings are reported in Table XII. We also compare against full-chain RSA (Venkatraman et al., 2025), recovered as the \(\tau = \beta\) limit of our algorithm. The final scores reported in this section are mean correctness over the final-round candidate outputs, not best-of-N unless explicitly stated.

a) Headline result: With Markovian RSA at \(( \beta , \tau , T , N , C ) ~ = ~ ( 4 0 \mathrm { K } , 4 \mathrm { K } , 2 , 1 6 , 4 )\) , ZAYA1-8B reaches 91.9% on AIME’25 and 89.6% on HMMT’25 Feb. These runs use temperature 1.0, top-p 1.0, and a 40K-token finalresponse budget. The result holds while carrying forward only a 4K-token tail between aggregation rounds, approximately one-tenth of the per-rollout reasoning budget.

b) Configuration sweep: Table XII reports accuracy across four Markovian RSA configurations, alongside the

<table><tr><td>Model</td><td>Active</td><td>Total</td><td>AIME&#x27;25</td><td>HMMT&#x27;25 Feb.</td><td>LCB-v6*</td></tr><tr><td>ZAYA1-8B (single rollout)</td><td>0.7B</td><td>8.0B</td><td>88.3</td><td>82.7</td><td>65.0</td></tr><tr><td>ZAYA1-8B + Markovian RSA (40K/4K)</td><td>0.7B</td><td>8.0B</td><td>91.9</td><td>89.6</td><td> \(69.2^‡\) </td></tr><tr><td>DeepSeek-R1-0528 \(^†\) </td><td>37B</td><td>671B</td><td>87.5</td><td>79.4</td><td>68.7</td></tr><tr><td>Qwen3-235B-A22B-Thinking-2507 \(^†\) </td><td>22B</td><td>235B</td><td>92.3</td><td>83.9</td><td>74.1</td></tr><tr><td>Gemini-2.5 Pro \(^†\) </td><td>-</td><td>-</td><td>88.0</td><td>82.5</td><td>72.5</td></tr><tr><td>DeepSeek-V3.2 \(^†\) </td><td>37B</td><td>671B</td><td>93.1</td><td>92.5</td><td>-</td></tr><tr><td>GPT-5-High \(^†\) </td><td>-</td><td>-</td><td>94.6</td><td>88.3</td><td>-</td></tr></table>

∗LCB-v6 denotes the 2025-02–2025-05 LiveCodeBench-v6 split.   
‡ For LCB-v6 on the same pre-behavioral checkpoint after math+code+TTC RL, Markovian RSA improves ZAYA1-8B from 65% single-rollout to 69.2%, while our PaCoRe hybrid compaction variant reaches 71.1%. This variant is not an exact implementation of PaCoRe: when a candidate reaches a post-think answer section, we pass that compact answer forward; when it does not, we fall back to passing the partial reasoning chain rather than dropping the candidate. Since the model was trained with both RSA and PaCoRe-type aggregation examples, we do not attribute the gap to training exposure alone.   
TABLE XI: ZAYA1-8B single-rollout and TTC numbers in this table are evaluated on the pre-behavioral checkpoint after math+code+TTC RL and before the final lightweight behavioral-RL polishing stage, using the Zyphra evaluation harness. The final behavioral stage targets chat style, instruction following, and preference behavior rather than math/code/TTC capability. ZAYA1-8B + Markovian RSA uses the 40K/4K configuration from Section VI-C. Numbers for comparator models marked † are taken from external sources. DeepSeek-R1-0528, Qwen3-235B-A22B-Thinking-2507, and Gemini-2.5 Pro are from the Qwen3-235B-A22B-Thinking-2507 model card (Qwen Team, 2025). DeepSeek-V3.2 is from the DeepSeek-V3.2 technical report (DeepSeek-AI, 2025c); the GPT-5-High row is reproduced from the comparison table in that report rather than from an OpenAI release table.

C = 1 Markovian Thinker baseline described in Section VI-A. At \(T = 2 , N = 1 6 ,\) and \(C = 4 ,\) increasing the per-rollout reasoning budget \(\beta\) from 8K to 16K to 40K improves both benchmarks at fixed tail size: AIME’25 advances from 86.5% to 88.8% to 91.9%, and HMMT’25 advances from 80.8% to 87.1% to 89.6%. HMMT’25 is especially responsive to longer per-rollout reasoning, with a 6.3-point gain from \(\beta = 8 { \sf K }\) to \(\beta = 1 6 \mathsf { K }\) .

c) Evaluation scope: The algorithmic definition, training construction, and serving profile above describe the Markovian RSA method. The remainder of this subsection reports empirical TTC scaling results, measuring how accuracy changes with the per-rollout reasoning budget \(\beta ,\) carried-forward tail length τ , aggregation depth \(T ,\) , and aggregation method under a fixed population size.

d) Generated-token accounting: We report realized total generated decode tokens separately from per-worker trajectory lengths. Markovian RSA generates multiple candidates in parallel at each non-final stage, so a per-worker or per-stage average length is not the total decode cost of the method. For a problem \(q ,\) let \(g _ { s , j } ( q )\) be the number of newly generated tokens from worker j in generation stage s, and let \(n _ { s }\) be the number of workers in that stage. The realized generated-token cost is

\[
D (q) = \sum_ {s} \sum_ {j = 1} ^ {n _ {s}} g _ {s, j} (q) = \sum_ {s} n _ {s} \bar {g} _ {s} (q), \tag {24}
\]

where \(\bar { g } _ { s } ( q )\) is the average generated length per worker in stage s. This count includes newly generated candidate and aggregation tokens across all workers, but excludes the original problem prompt, aggregation-prompt prefill, and copied carryforward tails.

Under this accounting, with the final response budget of 40K, the reported AIME’25/HMMT’25 Markovian RSA evaluations use approximately 440K generated decode tokens per problem for the 16K/4K configuration and approximately 740K generated decode tokens per problem for the 40K/4K configuration. These are realized averages from the evaluation runs, not worst-case caps; they depend on early stopping, benchmark, sampling settings, and implementation details. We include them to avoid conflating per-worker trajectory length with total TTC cost.

e) Tail size and iteration depth: Increasing the tail size \(\tau\) from 4K to 8K at fixed \(\beta ~ = ~ 4 0 \mathsf { K }\) does not improve accuracy on AIME’25 or HMMT’25: the 4K-tail configuration reaches 91.9%/89.6%, while the 8K-tail configuration reaches 90.8%/89.2%. Because these scores average over multiple generated candidates per problem, the comparison is less sensitive to a single unlucky rollout than a one-sample evaluation. We treat this as empirical evidence that, for these configurations and benchmarks, aggregation is not limited by retaining more than a 4K reasoning tail.

This should not be read as a general claim that tail length never matters. On harder benchmarks, the aggregation depth (T ), the diversity of the candidate responses (N and C), β, and \(\tau\) can all contribute to saturating the model’s capacity. We also evaluate higher-compute Markovian RSA settings on APEX-shortlist, a harder capacity-ceiling benchmark. Table XIII reports the three APEX configurations used for the light, high, and extra-high modes shown in Figure 10. The strongest setting, with \(T = 8 , N = 3 2 , C = 4 , \beta = 3 2 \mathrm { K }\) , and \(\tau = 4 \mathrm { K }\) , reaches 51.8% on APEX-shortlist. This setting uses approximately 5.5M newly generated decode tokens per problem, so we treat it as a capability-ceiling evaluation rather than a recommended deployment setting.