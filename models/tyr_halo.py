"""
TYR-HALO: Small Efficient Productive LLM for Limited Resources.

Architecture: FactorizedEmbed -> Prelude GQA -> [6-layer Shared Block, Parcae loop, mHC mixing] -> Coda GQA -> FactorizedLMHead + MTP
  - 6 unique shared layers (4 ShortConv + 2 MoDA-GQA), Parcae loop mean=2
  - MoDA depth-attention: each GQA head attends to prior-iteration KVs
  - mHC (Manifold-Constrained Hyper-Connections): 4-branch residual with Sinkhorn mixing
  - MTP auxiliary head (depth=1) for improved representations during training
  - ~58M unique params, ~115M Parcae-equivalent

Novel mechanisms from: DeepSeek-V4 (mHC), MoDA paper (depth attention), Meta (MTP).

Targets: Beat Portimbria-150M, compete with SmolLM2-135M on 12B synthetic tokens.
Throughput target: 60K tok/s DDP (2-machine Strix Halo) or ~60-80K on RTX 4060 Ti.

Usage:
    python -m halo_training --model models/tyr_halo.py --class-name TyrHaloMini --smoke
    python -m halo_training --model models/tyr_halo.py --class-name TyrHalo --dataset babylm --compile --muon --mtp
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.amadeus import RMSNorm, SwiGLU, GatedConv
from models.argus import precompute_freqs_cis, apply_rotary_emb
from models.argus_prime import Attention, ShortConvBlock
from models.chimera_halo import FactorizedEmbedding, FactorizedLMHead
from models.griffin_halo import SimpleParcaeInjection
from models.jormungandr_halo import CodaAttention

try:
    from kernels.hip.hybrid_attention import hybrid_flash_sdpa_attention
    _HAS_HYBRID_ATTN = True
except ImportError:
    _HAS_HYBRID_ATTN = False


# ---------------------------------------------------------------------------
# Hyperloop Hyper-Connections (MIT, 2604.21254 — replaces mHC Sinkhorn)
# ---------------------------------------------------------------------------

class HyperloopHC(nn.Module):
    """Diagonal hyper-connections for looped transformers.

    n parallel residual streams with diagonal sigmoid gating (not Sinkhorn).
    Applied at loop boundaries only. Hyperloop paper proves diagonal > Sinkhorn.
    """

    def __init__(self, d_model: int, n_streams: int = 4, n_iters: int = 2):
        super().__init__()
        self.n_streams = n_streams
        self.d_model = d_model
        nd = n_streams * d_model
        self.norm = RMSNorm(nd)
        self.hc = nn.ModuleList([
            nn.ModuleDict({
                'w_pre': nn.Linear(nd, n_streams, bias=True),
                'w_post': nn.Linear(nd, n_streams, bias=True),
                'w_res': nn.Linear(nd, n_streams, bias=True),
                'alpha_pre': nn.ParameterList([nn.Parameter(torch.ones(1))]),
                'alpha_post': nn.ParameterList([nn.Parameter(torch.ones(1))]),
                'alpha_res': nn.ParameterList([nn.Parameter(torch.ones(1))]),
            })
            for _ in range(n_iters)
        ])
        self._init_near_identity()

    def _init_near_identity(self):
        for hc in self.hc:
            nn.init.zeros_(hc['w_pre'].weight)
            nn.init.zeros_(hc['w_pre'].bias)
            nn.init.zeros_(hc['w_post'].weight)
            nn.init.zeros_(hc['w_post'].bias)
            nn.init.zeros_(hc['w_res'].weight)
            nn.init.zeros_(hc['w_res'].bias)

    def expand(self, h: torch.Tensor) -> torch.Tensor:
        """(B, T, d) -> (B, T, n, d) by copying."""
        return h.unsqueeze(2).expand(-1, -1, self.n_streams, -1).clone()

    def contract(self, streams: torch.Tensor) -> torch.Tensor:
        """(B, T, n, d) -> (B, T, d) by averaging."""
        return streams.mean(dim=2)

    def read_write(self, streams: torch.Tensor, block_output: torch.Tensor,
                   iter_idx: int) -> torch.Tensor:
        """Apply hyper-connection at loop boundary.

        streams: (B, T, n, d)
        block_output: (B, T, d)
        returns: (B, T, n, d)
        """
        B, T, n, d = streams.shape
        z = self.norm(streams.reshape(B, T, n * d))
        hc = self.hc[iter_idx]

        H_pre = torch.sigmoid(hc['alpha_pre'][0] * hc['w_pre'](z))       # (B, T, n)
        H_post = 2.0 * torch.sigmoid(hc['alpha_post'][0] * hc['w_post'](z))  # (B, T, n)
        H_res = torch.sigmoid(hc['alpha_res'][0] * hc['w_res'](z))        # (B, T, n)

        gated = streams * H_res.unsqueeze(-1)
        written = H_post.unsqueeze(-1) * block_output.unsqueeze(2)
        return gated + written

    def read(self, streams: torch.Tensor, iter_idx: int) -> torch.Tensor:
        """Read from n streams -> single d-dim tensor for block input."""
        B, T, n, d = streams.shape
        z = self.norm(streams.reshape(B, T, n * d))
        hc = self.hc[iter_idx]
        H_pre = torch.sigmoid(hc['alpha_pre'][0] * hc['w_pre'](z))  # (B, T, n)
        return (streams * H_pre.unsqueeze(-1)).sum(dim=2)


# Legacy alias
mHCBranchManager = HyperloopHC


# ---------------------------------------------------------------------------
# MoDA-GQA Block (GQA + Depth Attention + XSA + momentum + SwiGLU)
# ---------------------------------------------------------------------------

class MoDAGQABlock(nn.Module):
    """GQA with MoDA depth-attention, XSA, momentum, SwiGLU FFN.

    MoDA: each attention head attends to both sequence KVs and depth KVs
    (representations from same token position in prior loop iterations).
    Depth KV projection produces position-independent cross-layer retrieval keys.
    """

    def __init__(self, d_model: int, ffn_inner: int,
                 n_heads: int = 10, n_kv_heads: int = 2,
                 momentum_beta: float = 0.5, use_xsa: bool = True):
        super().__init__()
        self.head_dim = d_model // n_heads
        self.n_kv_heads = n_kv_heads
        self.pre_norm = RMSNorm(d_model)
        if use_xsa:
            self.attn = CodaAttention(d_model, n_heads, n_kv_heads,
                                      qk_norm=True, exclusive=True)
        else:
            self.attn = Attention(d_model, n_heads, n_kv_heads, qk_norm=True)
        self.use_xsa = use_xsa
        self.depth_kv_proj = nn.Linear(d_model, n_kv_heads * self.head_dim * 2, bias=False)
        self.log_beta = nn.Parameter(
            torch.tensor(math.log(momentum_beta / (1 - momentum_beta)))
        )
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, ffn_inner)

    def forward(self, x: torch.Tensor, velocity: torch.Tensor,
                freqs_cis: torch.Tensor,
                depth_kvs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        normed = self.pre_norm(x)
        if self.use_xsa:
            attn_out = self.attn(normed, freqs_cis, depth_kvs=depth_kvs)
        else:
            attn_out = self.attn(normed, freqs_cis)

        beta = torch.sigmoid(self.log_beta)
        velocity = beta * velocity + attn_out
        velocity = velocity.clamp(-8.0, 8.0)

        x = x + velocity
        x = x + self.ffn(self.ffn_norm(x))
        return x, velocity

    def compute_depth_kv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project hidden state into depth KV pair for future iterations."""
        B, T, _ = x.shape
        kv = self.depth_kv_proj(x)
        k, v = kv.chunk(2, dim=-1)
        k = k.reshape(B, T, self.n_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        return k, v


# ---------------------------------------------------------------------------
# MTP Auxiliary Head
# ---------------------------------------------------------------------------

class MTPHead(nn.Module):
    """Multi-Token Prediction auxiliary head. Shares embedding table (tied weights).

    Predicts token at position +depth+1 from hidden states.
    Discarded after training; improves backbone representations via auxiliary loss.
    """

    def __init__(self, d_model: int, embed_rank: int, embed_table: nn.Embedding,
                 depth: int = 1):
        super().__init__()
        self.depth = depth
        self.proj = nn.Linear(d_model, embed_rank, bias=False)
        self.embed_table = embed_table

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Returns logits for token at position +(depth+1).

        h: (B, T, d_model)
        returns: (B, T - depth - 1, vocab_size)
        """
        trimmed = h[:, :-(self.depth + 1)]
        return F.linear(self.proj(trimmed), self.embed_table.weight)


# ---------------------------------------------------------------------------
# DS2D Forecast Embeddings (Samsung, 2026 — self-speculative decoding)
# ---------------------------------------------------------------------------

class ForecastEmbeddings(nn.Module):
    """Learned prefix embeddings that prime the model for multi-token prediction.

    Appended to the input embedding sequence during inference. The model learns
    to use these positions as "forecast slots" that predict future tokens.
    Trained via prefix tuning on frozen backbone.
    Ref: "Dynamic Self-Speculative Decoding" (Samsung, 2026)
    """

    def __init__(self, d_model: int, n_forecast: int = 4):
        super().__init__()
        self.n_forecast = n_forecast
        self.embeds = nn.Parameter(torch.randn(1, n_forecast, d_model) * 0.02)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Append forecast embeddings to hidden state sequence.

        h: (B, T, d) -> returns (B, T + n_forecast, d)
        """
        return torch.cat([h, self.embeds.expand(h.shape[0], -1, -1)], dim=1)


# ---------------------------------------------------------------------------
# Draft Heads (inference-time parallel token prediction from h_iter0)
# ---------------------------------------------------------------------------

class DraftHeads(nn.Module):
    """K parallel linear probes on intermediate hidden state for speculative drafting.

    Each head predicts token at position t+k from h_iter0. Shares embedding table.
    Grounded in linear probing theorem: h_{L-k} has ~(1-k/L)^2 of final accuracy.
    Validated by DFlash ablation: no-KV-injection baseline gives 2.83x speedup.
    """

    def __init__(self, d_model: int, embed_rank: int, embed_table: nn.Embedding,
                 n_drafts: int = 4):
        super().__init__()
        self.n_drafts = n_drafts
        self.probes = nn.ModuleList([
            nn.Linear(d_model, embed_rank, bias=False) for _ in range(n_drafts)
        ])
        self.embed_table = embed_table

    def forward(self, h_iter0: torch.Tensor) -> List[torch.Tensor]:
        """Generate K draft token logits from intermediate hidden state.

        h_iter0: (B, T, d) — hidden state after first loop iteration
        returns: list of K tensors, each (B, 1, vocab_size) — logits for t+1..t+K
        """
        last_h = h_iter0[:, -1:, :]  # only last position for decode
        return [F.linear(probe(last_h), self.embed_table.weight)
                for probe in self.probes]

    def draft_tokens(self, h_iter0: torch.Tensor) -> torch.Tensor:
        """Greedy-decode K draft tokens.

        returns: (B, K) — draft token IDs
        """
        logits_list = self.forward(h_iter0)
        return torch.cat([lg.argmax(dim=-1) for lg in logits_list], dim=-1)


# ---------------------------------------------------------------------------
# CTG: Concurrent Token Generation (for batched ES rollouts)
# ---------------------------------------------------------------------------

def concurrent_generate(
    model: nn.Module,
    input_ids: torch.Tensor,
    n_streams: int = 8,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Generate n_streams independent completions in parallel via partitioned KV.

    Each stream gets independent sampling from the same prefill. Useful for
    ES alignment where K=16 rollouts are needed per prompt.
    Ref: "Concurrent Token Generation" (Samsung, 2026)

    Args:
        model: TyrHaloBase instance (eval mode)
        input_ids: (1, T) — single prompt
        n_streams: number of parallel generation streams
        max_new_tokens: tokens to generate per stream
        temperature: sampling temperature

    Returns:
        (n_streams, T + max_new_tokens) — all completions
    """
    assert input_ids.shape[0] == 1, "CTG expects single prompt"
    device = input_ids.device

    # Expand prompt to n_streams copies
    ids = input_ids.expand(n_streams, -1).clone()

    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(ids)
            if isinstance(logits, dict):
                logits = logits["logits"]
            next_logits = logits[:, -1, :] / max(temperature, 1e-8)
            probs = F.softmax(next_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_tokens], dim=1)

    return ids


# ---------------------------------------------------------------------------
# TYR-HALO Model
# ---------------------------------------------------------------------------

class TyrHaloBase(nn.Module):
    """TYR-HALO: Small efficient productive LLM.

    Prelude (1 GQA) -> mHC init -> Parcae loop [6-layer shared block, mHC mix] -> Coda (1 GQA) -> output.
    Core block: 4 ShortConv + 2 MoDA-GQA. Parcae loop provides depth, mHC provides residual richness.
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 640,
        embed_rank: int = 256,
        n_shared_layers: int = 6,
        gqa_positions: Tuple[int, ...] = (2, 5),
        d_conv: int = 512,
        ffn_inner: int = 2304,
        n_heads: int = 10,
        n_kv_heads: int = 2,
        conv_kernel: int = 3,
        mean_recurrence: int = 2,
        backprop_depth: int = 2,
        curriculum_steps: int = 5000,
        use_xsa: bool = True,
        use_moda: bool = True,
        use_mhc: bool = True,
        use_mtp: bool = True,
        mtp_depth: int = 1,
        use_draft_heads: bool = False,
        n_draft_heads: int = 4,
        use_forecast: bool = False,
        n_forecast: int = 4,
        n_branches: int = 2,
        sinkhorn_iters: int = 5,
        momentum_beta_init: float = 0.5,
        max_seq_len: int = 1024,
        use_prelude: bool = True,
        use_coda: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.mean_recurrence = mean_recurrence
        self.backprop_depth = backprop_depth
        self.curriculum_steps = curriculum_steps
        self.max_seq_len = max_seq_len
        self.gqa_positions = set(gqa_positions)
        self.use_prelude = use_prelude
        self.use_coda = use_coda
        self.use_moda = use_moda
        self.use_mhc = use_mhc
        self.use_mtp = use_mtp
        self.use_draft_heads = use_draft_heads
        self.use_forecast = use_forecast

        # === FACTORIZED EMBEDDINGS ===
        self.tok_embeddings = FactorizedEmbedding(vocab_size, embed_rank, d_model)
        self.norm = RMSNorm(d_model)
        self.lm_head = FactorizedLMHead(d_model, embed_rank, self.tok_embeddings.embed)

        # RoPE
        head_dim = d_model // n_heads
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(head_dim, max_seq_len * 2),
            persistent=False,
        )

        self.register_buffer("step_counter", torch.tensor(0, dtype=torch.long))

        # === PRELUDE ===
        if use_prelude:
            self.prelude = MoDAGQABlock(
                d_model, ffn_inner, n_heads, n_kv_heads,
                momentum_beta_init, use_xsa=use_xsa,
            )

        # === Hyperloop HC or SimpleParcaeInjection ===
        if use_mhc:
            self.mhc = HyperloopHC(d_model, n_branches, mean_recurrence)
        self.injection = SimpleParcaeInjection(d_model)

        # === Loop position embeddings (Hyperloop paper) ===
        self.loop_pos_embeds = nn.Parameter(
            torch.zeros(mean_recurrence, d_model)
        )

        # === SHARED BLOCK ===
        self.shared_layers = nn.ModuleList()
        self._is_gqa = []
        self._gqa_indices = []
        for i in range(n_shared_layers):
            if i in self.gqa_positions:
                self.shared_layers.append(MoDAGQABlock(
                    d_model, ffn_inner, n_heads, n_kv_heads,
                    momentum_beta_init, use_xsa=use_xsa,
                ))
                self._is_gqa.append(True)
                self._gqa_indices.append(i)
            else:
                self.shared_layers.append(ShortConvBlock(
                    d_model, d_conv, ffn_inner, conv_kernel, momentum_beta_init,
                ))
                self._is_gqa.append(False)

        self.iter_norm = RMSNorm(d_model)

        # === CODA ===
        if use_coda:
            self.coda = MoDAGQABlock(
                d_model, ffn_inner, n_heads, n_kv_heads,
                momentum_beta_init, use_xsa=use_xsa,
            )

        # === MTP HEAD ===
        if use_mtp:
            self.mtp_head = MTPHead(d_model, embed_rank, self.tok_embeddings.embed,
                                    depth=mtp_depth)

        # === DRAFT HEADS (inference only) ===
        if use_draft_heads:
            self.draft_heads = DraftHeads(d_model, embed_rank,
                                          self.tok_embeddings.embed, n_draft_heads)

        # === DS2D FORECAST EMBEDDINGS (inference only) ===
        if use_forecast:
            self.forecast = ForecastEmbeddings(d_model, n_forecast)

        self._init_weights()
        n_params = sum(p.numel() for p in self.parameters())
        print(f"{self.__class__.__name__}: {n_params / 1e6:.1f}M parameters")

    def _init_weights(self):
        n_layers = len(self.shared_layers)
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)
                if "wo." in name or "w_down." in name or "out_proj." in name:
                    with torch.no_grad():
                        p.div_(math.sqrt(2 * n_layers))
            elif p.dim() == 1 and "bias" in name:
                nn.init.zeros_(p)

    def compile_zones(self):
        """Compile each layer independently for per-zone fusion.

        On NVIDIA: prefer torch.compile(model) over this for full-model fusion.
        On AMD: call AFTER autokernel.optimize() and BEFORE training.
        """
        try:
            from kernels.hip._torch_ops import disable_hip_backward
            disable_hip_backward()
        except ImportError:
            pass
        if self.use_prelude:
            self.prelude = torch.compile(self.prelude, mode="default")
        for i in range(len(self.shared_layers)):
            self.shared_layers[i] = torch.compile(self.shared_layers[i], mode="default")
        if self.use_coda:
            self.coda = torch.compile(self.coda, mode="default")
        return self

    def sample_loop_depth(self, step: int) -> Tuple[int, int]:
        """Parcae-style Poisson depth sampling with 1-sqrt curriculum."""
        progress = min(step / max(self.curriculum_steps, 1), 1.0)
        effective_progress = 1 - math.sqrt(1 - progress)

        t_full = max(self.mean_recurrence - self.backprop_depth, 0)
        t = max(math.ceil(effective_progress * t_full), 0)

        n_detached = torch.poisson(torch.tensor([float(t)])).long().item()
        n_detached = min(n_detached, 2 * max(t, 1))
        n_grad = self.backprop_depth

        return n_detached, n_grad

    def _run_shared_block(
        self, h: torch.Tensor, velocity: torch.Tensor,
        freqs_cis: torch.Tensor,
        depth_kv_buffer: List[Dict[int, Tuple[torch.Tensor, torch.Tensor]]],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, Tuple[torch.Tensor, torch.Tensor]]]:
        """Run all 6 shared layers once. Returns updated h, velocity, and new depth KVs."""
        current_kvs: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        for layer_idx, (layer, is_gqa) in enumerate(zip(self.shared_layers, self._is_gqa)):
            if is_gqa:
                prior_kvs = None
                if self.use_moda and depth_kv_buffer:
                    prior_kvs = [buf[layer_idx] for buf in depth_kv_buffer
                                 if layer_idx in buf]
                    if not prior_kvs:
                        prior_kvs = None
                h, velocity = layer(h, velocity, freqs_cis, depth_kvs=prior_kvs)
                if self.use_moda:
                    current_kvs[layer_idx] = layer.compute_depth_kv(h.detach())
            else:
                h, velocity = layer(h, velocity)
        return h, velocity, current_kvs

    @property
    def _is_deterministic_depth(self) -> bool:
        return self.mean_recurrence == self.backprop_depth

    def _forward_unrolled(self, input_ids: torch.Tensor, targets=None) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Unrolled forward for deterministic depth (mean_recurrence == backprop_depth).

        Writes out each iteration explicitly so torch.compile sees one continuous
        graph with no Python loop → full cross-iteration fusion.
        Uses Hyperloop-style diagonal HC + loop position embeddings.
        """
        B, T = input_ids.shape
        h = self.tok_embeddings(input_ids)
        freqs_cis = self.freqs_cis[:T]

        velocity = torch.zeros_like(h)
        if self.use_prelude:
            h, velocity = self.prelude(h, velocity, freqs_cis)
        input_embed = h

        if self.use_mhc:
            streams = self.mhc.expand(h)

        depth_kv_buffer: List[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = []

        # === ITERATION 1: no re-injection ===
        if self.use_mhc:
            h = self.mhc.read(streams, 0)
        h, velocity, current_kvs = self._run_shared_block(h, velocity, freqs_cis, depth_kv_buffer)
        h = self.iter_norm(h) + self.loop_pos_embeds[0]
        if self.use_mhc:
            streams = self.mhc.read_write(streams, h, 0)
        if current_kvs:
            depth_kv_buffer.append(current_kvs)

        # === ITERATIONS 2..N: with re-injection ===
        for i in range(1, self.mean_recurrence):
            if self.use_mhc:
                h = self.mhc.read(streams, i)
            h = self.injection(h, input_embed)
            h, velocity, current_kvs = self._run_shared_block(
                h, velocity, freqs_cis, depth_kv_buffer)
            h = self.iter_norm(h) + self.loop_pos_embeds[i]
            if self.use_mhc:
                streams = self.mhc.read_write(streams, h, i)
            if current_kvs:
                depth_kv_buffer.append(current_kvs)

        # === CONTRACT STREAMS ===
        if self.use_mhc:
            h = self.mhc.contract(streams)

        # === CODA ===
        velocity = torch.zeros_like(h)
        if self.use_coda:
            h, velocity = self.coda(h, velocity, freqs_cis)

        logits = self.lm_head(self.norm(h))

        if self.training and self.use_mtp and hasattr(self, 'mtp_head'):
            return {"logits": logits, "mtp1": self.mtp_head(h)}
        return logits

    def _forward_dynamic(self, input_ids: torch.Tensor, targets=None) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Dynamic-depth forward with Poisson sampling + detached/grad split."""
        B, T = input_ids.shape
        h = self.tok_embeddings(input_ids)
        freqs_cis = self.freqs_cis[:T]

        velocity = torch.zeros_like(h)
        if self.use_prelude:
            h, velocity = self.prelude(h, velocity, freqs_cis)
        input_embed = h

        if self.use_mhc:
            streams = self.mhc.expand(h)

        if self.training:
            step = self.step_counter.item()
            n_detached, n_grad = self.sample_loop_depth(step)
        else:
            n_detached = 0
            n_grad = self.mean_recurrence

        total_iters = n_detached + n_grad
        depth_kv_buffer: List[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = []
        iter_count = 0
        n_hc = len(self.mhc.hc) if self.use_mhc else self.mean_recurrence

        if self.use_mhc:
            h = self.mhc.read(streams, 0)
        h, velocity, current_kvs = self._run_shared_block(h, velocity, freqs_cis, depth_kv_buffer)
        h = self.iter_norm(h) + self.loop_pos_embeds[0]
        if self.use_mhc:
            streams = self.mhc.read_write(streams, h, 0)
        if current_kvs:
            depth_kv_buffer.append(current_kvs)
        iter_count = 1

        if total_iters <= 1:
            remaining_detached = 0
            remaining_grad = 0
        elif n_detached > 0:
            remaining_detached = n_detached - 1
            remaining_grad = n_grad
        else:
            remaining_detached = 0
            remaining_grad = n_grad - 1

        for _ in range(remaining_detached):
            hc_idx = min(iter_count, n_hc - 1)
            lp_idx = min(iter_count, self.mean_recurrence - 1)
            with torch.no_grad():
                if self.use_mhc:
                    h = self.mhc.read(streams, hc_idx)
                h = self.injection(h, input_embed)
                h, velocity, current_kvs = self._run_shared_block(
                    h, velocity, freqs_cis, depth_kv_buffer)
                h = self.iter_norm(h) + self.loop_pos_embeds[lp_idx]
                if self.use_mhc:
                    streams = self.mhc.read_write(streams, h, hc_idx)
                if current_kvs:
                    depth_kv_buffer.append({k: (dk.detach(), dv.detach())
                                            for k, (dk, dv) in current_kvs.items()})
            iter_count += 1

        for _ in range(remaining_grad):
            hc_idx = min(iter_count, n_hc - 1)
            lp_idx = min(iter_count, self.mean_recurrence - 1)
            if self.use_mhc:
                h = self.mhc.read(streams, hc_idx)
            h = self.injection(h, input_embed)
            h, velocity, current_kvs = self._run_shared_block(
                h, velocity, freqs_cis, depth_kv_buffer)
            h = self.iter_norm(h) + self.loop_pos_embeds[lp_idx]
            if self.use_mhc:
                streams = self.mhc.read_write(streams, h, hc_idx)
            if current_kvs:
                depth_kv_buffer.append(current_kvs)
            iter_count += 1

        if self.use_mhc:
            h = self.mhc.contract(streams)

        velocity = torch.zeros_like(h)
        if self.use_coda:
            h, velocity = self.coda(h, velocity, freqs_cis)

        logits = self.lm_head(self.norm(h))

        if self.training and self.use_mtp and hasattr(self, 'mtp_head'):
            return {"logits": logits, "mtp1": self.mtp_head(h)}
        return logits

    def forward(self, input_ids: torch.Tensor, targets=None) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if self.training:
            self.step_counter += 1
        if self._is_deterministic_depth:
            return self._forward_unrolled(input_ids, targets)
        return self._forward_dynamic(input_ids, targets)

    def speculative_decode(
        self, input_ids: torch.Tensor, max_new_tokens: int = 128,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Autoregressive generation with Parcae-native speculative decoding.

        Uses iter0 as draft phase, iter1 as verify phase.
        Draft heads predict K tokens from h_iter0 in parallel.
        DS2D forecast embeddings optionally prime the model for multi-token slots.
        """
        assert not self.training
        B = input_ids.shape[0]
        device = input_ids.device
        ids = input_ids.clone()
        has_drafts = self.use_draft_heads and hasattr(self, 'draft_heads')

        for _ in range(max_new_tokens):
            h = self.tok_embeddings(ids)
            T = ids.shape[1]
            freqs_cis = self.freqs_cis[:T]

            # Prelude
            velocity = torch.zeros_like(h)
            if self.use_prelude:
                h, velocity = self.prelude(h, velocity, freqs_cis)
            input_embed = h

            if self.use_mhc:
                nb = self.mhc.n_branches
                streams = h.unsqueeze(1).expand(-1, nb, -1, -1).clone()

            depth_kv_buffer: List[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = []

            # === ITER 0: DRAFT PHASE ===
            if self.use_mhc:
                h = self.mhc.read(streams)
            h, velocity, current_kvs = self._run_shared_block(
                h, velocity, freqs_cis, depth_kv_buffer)
            h_iter0 = self.iter_norm(h)
            if self.use_mhc:
                streams = self.mhc.write(streams, h_iter0)
            if current_kvs:
                depth_kv_buffer.append(current_kvs)

            if has_drafts:
                draft_ids = self.draft_heads.draft_tokens(h_iter0)  # (B, K)
                # Append drafts to sequence for batched verification
                candidate_ids = torch.cat([ids, draft_ids], dim=1)

                # === ITER 1: VERIFY PHASE on extended sequence ===
                h_full = self.tok_embeddings(candidate_ids)
                T_full = candidate_ids.shape[1]
                freqs_full = self.freqs_cis[:T_full]
                vel_full = torch.zeros_like(h_full)
                if self.use_prelude:
                    h_full, vel_full = self.prelude(h_full, vel_full, freqs_full)

                if self.use_mhc:
                    streams_v = h_full.unsqueeze(1).expand(-1, nb, -1, -1).clone()
                    h_full = self.mhc.read(streams_v)
                # Skip re-injection for verify (single effective iter on extended seq)
                h_full, vel_full, _ = self._run_shared_block(
                    h_full, vel_full, freqs_full, [])
                h_full = self.iter_norm(h_full)
                if self.use_mhc:
                    streams_v = self.mhc.write(streams_v, h_full)
                # Second iter (verify)
                if self.use_mhc:
                    h_full = self.mhc.read(streams_v)
                h_full = self.injection(h_full, h_full)
                h_full, vel_full, _ = self._run_shared_block(
                    h_full, vel_full, freqs_full, [])
                h_full = self.iter_norm(h_full)

                vel_full = torch.zeros_like(h_full)
                if self.use_coda:
                    h_full, vel_full = self.coda(h_full, vel_full, freqs_full)
                verify_logits = self.lm_head(self.norm(h_full))

                # Accept longest correct prefix
                K = draft_ids.shape[1]
                accepted = 0
                for k in range(K):
                    pos = T - 1 + k
                    if pos >= verify_logits.shape[1]:
                        break
                    verified_token = verify_logits[:, pos, :].argmax(dim=-1)
                    if (verified_token == draft_ids[:, k]).all():
                        accepted += 1
                    else:
                        break

                if accepted > 0:
                    ids = torch.cat([ids, draft_ids[:, :accepted]], dim=1)
                # Always add one token (verified or freshly sampled)
                last_pos = T - 1 + accepted
                if last_pos < verify_logits.shape[1]:
                    next_logits = verify_logits[:, last_pos, :] / max(temperature, 1e-8)
                    if temperature == 0 or temperature == 1.0:
                        next_token = next_logits.argmax(dim=-1, keepdim=True)
                    else:
                        probs = F.softmax(next_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                    ids = torch.cat([ids, next_token], dim=1)
            else:
                # No draft heads — standard autoregressive with full loop
                h = h_iter0
                if self.use_mhc:
                    h = self.mhc.read(streams)
                h = self.injection(h, input_embed)
                h, velocity, _ = self._run_shared_block(
                    h, velocity, freqs_cis, depth_kv_buffer)
                h = self.iter_norm(h)

                velocity = torch.zeros_like(h)
                if self.use_coda:
                    h, velocity = self.coda(h, velocity, freqs_cis)
                logits = self.lm_head(self.norm(h))
                next_logits = logits[:, -1, :] / max(temperature, 1e-8)
                if temperature == 0 or temperature == 1.0:
                    next_token = next_logits.argmax(dim=-1, keepdim=True)
                else:
                    probs = F.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                ids = torch.cat([ids, next_token], dim=1)

            if ids.shape[1] >= input_ids.shape[1] + max_new_tokens:
                break

        return ids


# ---------------------------------------------------------------------------
# Variant classes
# ---------------------------------------------------------------------------

class TyrHalo(TyrHaloBase):
    """Default: mean=3, mHC off, MoDA on, MTP on. 6 shared × 3 iters = 18 effective layers."""
    def __init__(self, **kw):
        super().__init__(mean_recurrence=3, backprop_depth=3, use_mhc=False, **kw)


class TyrHaloLight(TyrHaloBase):
    """Lean: mean=2, loop pos embeds, no HC streams, no prelude/coda. 58.5M unique, ~117M effective."""
    def __init__(self, **kw):
        kw.setdefault('embed_rank', 448)
        kw.setdefault('n_heads', 8)
        kw.setdefault('n_kv_heads', 4)
        super().__init__(mean_recurrence=2, backprop_depth=2, use_mhc=False,
                         use_prelude=False, use_coda=False, **kw)


class TyrHaloMHC(TyrHaloBase):
    """With mHC: mean=2, mHC on, MoDA on, MTP on. Slower but richer residual flow."""
    def __init__(self, **kw):
        super().__init__(mean_recurrence=2, use_mhc=True, **kw)


class TyrHaloFast(TyrHaloBase):
    """Max throughput: mean=3, MoDA on, mHC off, MTP off."""
    def __init__(self, **kw):
        super().__init__(
            mean_recurrence=3, backprop_depth=3,
            use_mhc=False, use_mtp=False, **kw,
        )


class TyrHaloBare(TyrHaloBase):
    """Ablation: no MoDA, no mHC, no MTP."""
    def __init__(self, **kw):
        super().__init__(
            mean_recurrence=2,
            use_xsa=False, use_moda=False, use_mhc=False, use_mtp=False, **kw,
        )


class TyrHaloNoLoop(TyrHaloBase):
    """Ablation: single pass, no Parcae loop, no novel mechanisms."""
    def __init__(self, **kw):
        super().__init__(
            mean_recurrence=1, backprop_depth=1,
            use_xsa=False, use_moda=False, use_mhc=False, use_mtp=False, **kw,
        )


class TyrHaloMini(TyrHaloBase):
    """Tiny config for smoke testing + CLIMB proxy search.

    Full vocab (50257) — lesson from ChimeraHaloMini real-data crash.
    """
    def __init__(self):
        super().__init__(
            vocab_size=50257,
            d_model=128,
            embed_rank=32,
            n_shared_layers=4,
            gqa_positions=(1, 3),
            d_conv=128,
            ffn_inner=256,
            n_heads=4,
            n_kv_heads=2,
            conv_kernel=3,
            mean_recurrence=2,
            backprop_depth=2,
            curriculum_steps=100,
            use_xsa=False,
            use_moda=True,
            use_mhc=True,
            use_mtp=True,
            mtp_depth=1,
            n_branches=2,
            sinkhorn_iters=5,
            momentum_beta_init=0.5,
            max_seq_len=512,
            use_prelude=False,
            use_coda=False,
        )
