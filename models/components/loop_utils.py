"""Loop utilities: HyperloopHC, DepthMemoryCache, iteration helpers.

HyperloopHC:       Diagonal hyper-connections for looped transformers (from tyr_halo).
DepthMemoryCache:  Content-dependent gated aggregation over loop iteration states (from jormungandr_halo).
"""

from typing import List

import torch
import torch.nn as nn

from models._components import RMSNorm


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
        """Apply hyper-connection at loop boundary."""
        B, T, n, d = streams.shape
        z = self.norm(streams.reshape(B, T, n * d))
        hc = self.hc[iter_idx]

        H_pre = torch.sigmoid(hc['alpha_pre'][0] * hc['w_pre'](z))
        H_post = 2.0 * torch.sigmoid(hc['alpha_post'][0] * hc['w_post'](z))
        H_res = torch.sigmoid(hc['alpha_res'][0] * hc['w_res'](z))

        gated = streams * H_res.unsqueeze(-1)
        written = H_post.unsqueeze(-1) * block_output.unsqueeze(2)
        return gated + written

    def read(self, streams: torch.Tensor, iter_idx: int) -> torch.Tensor:
        """Read from n streams -> single d-dim tensor for block input."""
        B, T, n, d = streams.shape
        z = self.norm(streams.reshape(B, T, n * d))
        hc = self.hc[iter_idx]
        H_pre = torch.sigmoid(hc['alpha_pre'][0] * hc['w_pre'](z))
        return (streams * H_pre.unsqueeze(-1)).sum(dim=2)


class DepthMemoryCache(nn.Module):
    """Content-dependent gated aggregation over cached loop iteration states.

    Instead of using only the final loop state, caches h at each iteration
    and lets each sequence position select its own weighted mix of depths.
    """

    def __init__(self, d_core: int, d_gate: int = 64):
        super().__init__()
        self.W_u = nn.Linear(d_core, d_gate, bias=False)

    def forward(self, cached_states: List[torch.Tensor]) -> torch.Tensor:
        if len(cached_states) == 1:
            return cached_states[0]

        u = self.W_u(cached_states[-1])

        gates = []
        for state in cached_states:
            key = self.W_u(state).mean(dim=1)
            gate = (u * key.unsqueeze(1)).sum(dim=-1)
            gates.append(gate)

        gates = torch.softmax(torch.stack(gates, dim=-1), dim=-1)

        stacked = torch.stack(cached_states, dim=-1)
        return (stacked * gates.unsqueeze(2)).sum(dim=-1)