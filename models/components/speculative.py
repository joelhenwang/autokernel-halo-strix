"""Speculative decoding utilities: DraftHeads, ForecastEmbeddings (from tyr_halo).

DraftHeads:          K parallel linear probes on intermediate hidden state for speculative drafting.
ForecastEmbeddings:  Learned prefix embeddings for multi-token prediction.
concurrent_generate: Batched generation with n independent streams via partitioned KV.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ForecastEmbeddings(nn.Module):
    """Learned prefix embeddings that prime the model for multi-token prediction.

    Appended to the input embedding sequence during inference.
    Trained via prefix tuning on frozen backbone.
    """

    def __init__(self, d_model: int, n_forecast: int = 4):
        super().__init__()
        self.n_forecast = n_forecast
        self.embeds = nn.Parameter(torch.randn(1, n_forecast, d_model) * 0.02)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.cat([h, self.embeds.expand(h.shape[0], -1, -1)], dim=1)


class DraftHeads(nn.Module):
    """K parallel linear probes on intermediate hidden state for speculative drafting.

    Each head predicts token at position t+k from h_iter0. Shares embedding table.
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
        last_h = h_iter0[:, -1:, :]
        return [F.linear(probe(last_h), self.embed_table.weight)
                for probe in self.probes]

    def draft_tokens(self, h_iter0: torch.Tensor) -> torch.Tensor:
        logits_list = self.forward(h_iter0)
        return torch.cat([lg.argmax(dim=-1) for lg in logits_list], dim=-1)


def concurrent_generate(
    model: nn.Module,
    input_ids: torch.Tensor,
    n_streams: int = 8,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Generate n_streams independent completions in parallel via partitioned KV."""
    assert input_ids.shape[0] == 1, "CTG expects single prompt"
    device = input_ids.device

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