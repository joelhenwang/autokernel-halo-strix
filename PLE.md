Per-Layer Embeddings (PLE) is an architecture technique used in Google's Gemma 3n and Gemma 4 models to enhance efficiency by providing each Transformer decoder layer with a dedicated, smaller embedding vector rather than using one single global embedding. This allows models with higher effective parameter counts to function within strict on-device VRAM constraints. 
Hugging Face
Hugging Face
 +4
Key Aspects of PLE:
Memory Efficiency: PLE offloads large embedding weights from high-speed GPU VRAM to slower CPU RAM, allowing larger models to fit into smaller memory footprints.
Structure: As described in this GitHub implementation guide, PLE creates a parallel, low-dimensional pathway. Each token lookup combines a token-identity component and a context-aware component (learned projection) to generate a specialized vector for every layer.
Operation: According to a Reddit post, instead of standard dense matrix multiplication, PLE often uses a "lookup" mechanism to modulate hidden states within a residual block.
Gemma Application: In Gemma 4 models, PLE allows 2.3B-active models to achieve performance comparable to much larger dense models by maximizing parameter utility. 
Hugging Face
Hugging Face
 +7
PLE works alongside other efficiency techniques like MatFormer (a flexible, nested compute structure) and Shared KV Cache

---

The small Gemma 4 models make use of Per-Layer Embeddings (PLE): Instead of a single large embedding matrix that is applied right after the tokenizer at the beginning of processing, there are additional (smaller) embedding matrices for each layer. Through training, they acquire specialized knowledge that can re-contextualize the token for the semantic specialization of each layer, which greatly improves processing quality. The layer-based embedding vectors are combined with the residuals through a series of operations, adding locally relevant information. 

---

`https://github.com/huggingface/transformers/issues/45206`
`https://github.com/huggingface/transformers/pull/45207`
`https://github.com/w4nderlust/transformers/commit/edaac7db98e34208209fd67d8c66781b8c2e4a53`
`https://github.com/w4nderlust/transformers/commit/aa678856e44b3a149338bbe62477f5e9f933bf39`
`https://github.com/w4nderlust/transformers/commit/2e0370e024a01fbe7dae542edbb5f38c5d77ee90`
