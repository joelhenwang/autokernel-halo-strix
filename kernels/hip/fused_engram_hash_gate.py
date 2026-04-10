"""
AutoKernel -- HIP C++ Fused Engram Hash + Gather + Gate kernel (Variant A).

Fuses: XOR hash computation → embedding table gather → key projection →
       RMSNorm query → dot-product gate → DeepSeek magnitude-preserving sigmoid
into a single kernel, eliminating intermediate tensors for hash indices and
multi-head embeddings.

Input:  hidden_states  (M, D)           -- hidden state, fp16
        input_ids      (M,)             -- token IDs, int64
        input_ids_prev (M, max_ngram-1) -- previous token IDs for n-grams, int64
        emb_weight     (n_heads*table_size, d_head) -- embedding table, fp16
        key_proj_w     (D, d_engram)    -- key projection weight, fp16
        norm_weight    (D,)             -- RMSNorm weight, fp16
        hash_mults     (n_ngrams, n_heads, max_ngram) -- hash multipliers, int64
        table_size     int
        n_heads        int
        n_ngrams       int

Output: gate       (M, 1)        -- gate scalar per position, fp16
        embs_flat  (M, d_engram) -- flattened embeddings for value projection, fp16

One block per row (B*T position). Each block:
1. Compute XOR hash for each ngram × head (integer math, registers)
2. Gather from embedding table (irregular access, L2-cached for small tables)
3. Flatten multi-head embeddings in LDS
4. Tile-based mat-vec: key = embs_flat @ key_proj_w.T
5. RMSNorm hidden state → query
6. Dot product, DeepSeek gating, sigmoid
"""

KERNEL_TYPE = "fused_engram_hash_gate"
BACKEND = "hip"

import torch
from kernels.hip._compile import compile_hip

HIP_SRC = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

constexpr int WARP_SIZE = 32;
constexpr float EPS = 1e-6f;
constexpr int MAX_NGRAM = 4;     // max n-gram order
constexpr int MAX_HEADS = 16;    // max hash heads
constexpr int MAX_NGRAMS = 4;    // max n-gram types

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down(val, offset);
    }
    return val;
}

__global__ void fused_engram_hash_gate_kernel(
    const half* __restrict__ H,             // (M, D) hidden states
    const int64_t* __restrict__ IDS,      // (M,) current token IDs
    const int64_t* __restrict__ PREV_IDS, // (M, max_ngram-1) previous IDs
    const half* __restrict__ EMB_W,         // (n_heads*table_size, d_head) embedding
    const half* __restrict__ KEY_W,         // (D, d_engram) key projection
    const half* __restrict__ NORM_W,        // (D,) RMSNorm weight
    const int64_t* __restrict__ HASH_MULTS, // (n_ngrams, n_heads, max_ngram)
    half* __restrict__ GATE_OUT,            // (M, 1) gate output
    half* __restrict__ EMBS_OUT,            // (M, d_engram) flattened embeddings
    int M,             // B*T
    int D,             // d_model
    int d_engram,      // total engram dim
    int d_head,        // per-head dim
    int table_size,
    int n_heads,
    int n_ngrams,
    int max_ngram      // max n-gram order
) {
    const int row = blockIdx.x;
    if (row >= M) return;

    const int tid = threadIdx.x;
    const int blockSize = blockDim.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int num_warps = blockSize / WARP_SIZE;

    const int n_total_heads = n_ngrams * n_heads;

    // Shared memory layout:
    //   [0 .. d_engram-1]: flattened embeddings
    //   [d_engram .. d_engram+31]: warp reduction scratch
    extern __shared__ float smem[];
    float* s_embs = smem;                      // d_engram floats
    float* s_scratch = smem + d_engram;        // 32 floats for reductions

    // =============================================
    // Phase 1: XOR hash + gather + flatten
    // =============================================

    // Each thread handles a subset of (ngram_type, head) pairs
    int64_t cur_id = IDS[row];
    const int64_t* prev_row = PREV_IDS + (int64_t)row * (max_ngram - 1);

    for (int idx = tid; idx < n_total_heads; idx += blockSize) {
        int ng_idx = idx / n_heads;
        int h_idx = idx % n_heads;

        // Build n-gram hash: XOR of (token[t-j] * mult[j])
        int ng_size = ng_idx + 2;  // ngram_sizes[0]=2, [1]=3, etc.
        const int64_t* mults = HASH_MULTS +
            ((int64_t)ng_idx * n_heads + h_idx) * max_ngram;

        int64_t mix = cur_id * mults[0];
        for (int k = 1; k < ng_size && k < max_ngram; k++) {
            // prev_ids[k-1] = token at position t-(k)
            int64_t prev_id = (k - 1 < max_ngram - 1) ? prev_row[k - 1] : 0;
            mix = mix ^ (prev_id * mults[k]);
        }
        int64_t hash_idx = ((mix % table_size) + table_size) % table_size;

        // Gather from embedding table with per-head offset
        int64_t emb_offset = ((int64_t)idx * table_size + hash_idx) * d_head;

        // Copy d_head values into shared memory at the right position
        int emb_start = idx * d_head;
        for (int e = 0; e < d_head; e++) {
            s_embs[emb_start + e] = __half2float(EMB_W[emb_offset + e]);
        }
    }
    __syncthreads();

    // Write flattened embeddings to global output
    for (int e = tid; e < d_engram; e += blockSize) {
        EMBS_OUT[(int64_t)row * d_engram + e] = __float2half(s_embs[e]);
    }

    // =============================================
    // Phase 2: Key projection (embs @ key_proj_w.T → key vector)
    // =============================================

    // Compute key[d] = sum_e(embs[e] * key_w[d, e]) for each d
    // We need the full key vector for dot product with query
    // Store dot product accumulator directly (query * key summed)

    // First compute RMSNorm of hidden state → query
    const half* h_row = H + (int64_t)row * D;

    float local_sum_sq = 0.0f;
    for (int d = tid; d < D; d += blockSize) {
        float hv = __half2float(h_row[d]);
        local_sum_sq += hv * hv;
    }
    local_sum_sq = warp_reduce_sum(local_sum_sq);

    if (lane_id == 0) s_scratch[warp_id] = local_sum_sq;
    __syncthreads();

    float rms_inv;
    if (warp_id == 0) {
        float s = (lane_id < num_warps) ? s_scratch[lane_id] : 0.0f;
        s = warp_reduce_sum(s);
        if (lane_id == 0) {
            s_scratch[0] = rsqrtf(s / (float)D + EPS);
        }
    }
    __syncthreads();
    rms_inv = s_scratch[0];

    // Now compute dot(query, key) where:
    //   query[d] = h[d] * rms_inv * norm_w[d]
    //   key[d] = sum_e(embs[e] * key_w[d, e])
    // Fuse into: sum_d(query[d] * key[d])

    float local_dot = 0.0f;
    for (int d = tid; d < D; d += blockSize) {
        float q = __half2float(h_row[d]) * rms_inv * __half2float(NORM_W[d]);

        // Compute key[d] inline: sum over d_engram
        float k = 0.0f;
        const half* kw_row = KEY_W + (int64_t)d * d_engram;
        for (int e = 0; e < d_engram; e++) {
            k += s_embs[e] * __half2float(kw_row[e]);
        }

        local_dot += q * k;
    }

    // Reduce dot product
    local_dot = warp_reduce_sum(local_dot);
    if (lane_id == 0) s_scratch[warp_id] = local_dot;
    __syncthreads();

    if (tid == 0) {
        float total = 0.0f;
        for (int w = 0; w < num_warps; w++) total += s_scratch[w];

        float gate_raw = total / sqrtf((float)D);
        float abs_g = (gate_raw >= 0.0f) ? gate_raw : -gate_raw;
        abs_g = (abs_g < 1e-6f) ? 1e-6f : abs_g;
        float sign_g = (gate_raw >= 0.0f) ? 1.0f : -1.0f;
        float mp_gate = sqrtf(abs_g) * sign_g;
        float gate = 1.0f / (1.0f + __expf(-mp_gate));

        GATE_OUT[row] = __float2half(gate);
    }
}

std::vector<torch::Tensor> fused_engram_hash_gate_hip(
    torch::Tensor hidden_states,
    torch::Tensor input_ids,
    torch::Tensor prev_ids,
    torch::Tensor emb_weight,
    torch::Tensor key_proj_w,
    torch::Tensor norm_weight,
    torch::Tensor hash_mults,
    int64_t table_size,
    int64_t n_heads,
    int64_t n_ngrams,
    int64_t max_ngram
) {
    int M = hidden_states.size(0);
    int D = hidden_states.size(1);
    int d_head = emb_weight.size(1);
    int d_engram = n_ngrams * n_heads * d_head;

    auto gate_out = torch::empty({M, 1}, hidden_states.options());
    auto embs_out = torch::empty({M, d_engram}, hidden_states.options());

    int threads = min(512, max(32, (D + 31) / 32 * 32));
    int n_total_heads = n_ngrams * n_heads;
    int smem_bytes = (d_engram + 32) * sizeof(float);

    fused_engram_hash_gate_kernel<<<M, threads, smem_bytes>>>(
        reinterpret_cast<const half*>(hidden_states.data_ptr<at::Half>()),
        input_ids.data_ptr<int64_t>(),
        prev_ids.data_ptr<int64_t>(),
        reinterpret_cast<const half*>(emb_weight.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(key_proj_w.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(norm_weight.data_ptr<at::Half>()),
        hash_mults.data_ptr<int64_t>(),
        reinterpret_cast<half*>(gate_out.data_ptr<at::Half>()),
        reinterpret_cast<half*>(embs_out.data_ptr<at::Half>()),
        M, D, d_engram, d_head, table_size, n_heads, n_ngrams, max_ngram
    );

    return {gate_out, embs_out};
}
"""

_module = None


def _get_module():
    global _module
    if _module is None:
        # Multi-output: need custom cpp_src for vector<Tensor> return
        cpp_src = r"""
#include <torch/extension.h>
std::vector<torch::Tensor> fused_engram_hash_gate_hip(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor,
    int64_t, int64_t, int64_t, int64_t);
"""
        _module = compile_hip(HIP_SRC, "fused_engram_hash_gate_hip", cpp_src=cpp_src)
    return _module


def kernel_fn(
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    prev_ids: torch.Tensor,
    emb_weight: torch.Tensor,
    key_proj_w: torch.Tensor,
    norm_weight: torch.Tensor,
    hash_mults: torch.Tensor,
    table_size: int,
    n_heads: int,
    n_ngrams: int,
    max_ngram: int,
) -> tuple:
    """Fused Engram hash + gather + gate (Variant A).

    Returns:
        (gate, embs_flat): gate (M, 1), embs_flat (M, d_engram)
    """
    assert hidden_states.is_cuda
    orig_dtype = hidden_states.dtype

    if orig_dtype != torch.float16:
        return reference_fn(hidden_states, input_ids, prev_ids, emb_weight,
                          key_proj_w, norm_weight, hash_mults,
                          table_size, n_heads, n_ngrams, max_ngram)

    h_2d = hidden_states.reshape(-1, hidden_states.shape[-1]).contiguous()
    ids_1d = input_ids.reshape(-1).contiguous()
    prev_2d = prev_ids.reshape(-1, prev_ids.shape[-1]).contiguous()

    mod = _get_module()
    gate, embs = mod.fused_engram_hash_gate_hip(
        h_2d, ids_1d, prev_2d,
        emb_weight.contiguous(), key_proj_w.contiguous(),
        norm_weight.contiguous(), hash_mults.contiguous(),
        table_size, n_heads, n_ngrams, max_ngram,
    )
    return gate, embs


def reference_fn(
    hidden_states, input_ids, prev_ids, emb_weight,
    key_proj_w, norm_weight, hash_mults,
    table_size, n_heads, n_ngrams, max_ngram,
):
    """Pure PyTorch reference — delegates to EngramLayer components."""
    import torch.nn.functional as F

    M, D = hidden_states.shape[-2], hidden_states.shape[-1]
    h_2d = hidden_states.reshape(-1, D)
    d_head = emb_weight.shape[1]
    n_total = n_ngrams * n_heads
    d_engram = n_total * d_head

    # Hash + gather (simplified reference)
    ids = input_ids.reshape(-1)
    prev = prev_ids.reshape(-1, prev_ids.shape[-1])

    all_embs = []
    for ng_idx in range(n_ngrams):
        ng_size = ng_idx + 2
        for h_idx in range(n_heads):
            mults = hash_mults[ng_idx, h_idx, :max_ngram]
            mix = ids * mults[0]
            for k in range(1, min(ng_size, max_ngram)):
                mix = mix ^ (prev[:, k - 1] * mults[k])
            hash_idx = mix % table_size
            head_offset = (ng_idx * n_heads + h_idx) * table_size
            flat_idx = (hash_idx + head_offset).long()
            emb = emb_weight[flat_idx]  # (M, d_head)
            all_embs.append(emb)

    embs_flat = torch.cat(all_embs, dim=-1)  # (M, d_engram)

    # Key projection
    key = F.linear(embs_flat.to(key_proj_w.dtype), key_proj_w)

    # RMSNorm query
    rms = (h_2d.float().pow(2).mean(-1, keepdim=True) + 1e-6).rsqrt()
    query = h_2d.float() * rms * norm_weight.float()

    # Gate
    gate_raw = (query * key.float()).sum(-1, keepdim=True) / (D ** 0.5)
    gate = gate_raw.abs().clamp(min=1e-6).sqrt() * gate_raw.sign()
    gate = torch.sigmoid(gate)

    return gate.to(hidden_states.dtype), embs_flat.to(hidden_states.dtype)
