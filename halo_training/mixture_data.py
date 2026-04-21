"""MixtureDataset: weighted sampling from per-cluster .bin files.

Two modes:
  1. Pre-mixed: single .bin file → wraps BabyLMDataset (Phase 5 output)
  2. Online-mixed: per-cluster .bin files + mixture_config.json → dynamic sampling

Usage:
    # Pre-mixed (recommended)
    ds = MixtureDataset("datasets/datamix_state/phase5_final/train.bin", block_size=1024)

    # Online-mixed (experimentation — change weights without re-assembling)
    ds = MixtureDataset("datasets/datamix_state/phase3_proxy_results/mixture_config.json", block_size=1024)
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from halo_training.data import BabyLMDataset


class MixtureDataset(Dataset):
    """Weighted mixture dataset compatible with halo_training trainer."""

    def __init__(self, path: str, block_size: int = 1024):
        self.block_size = block_size
        p = Path(path)

        if p.suffix == ".bin":
            self._mode = "premixed"
            self._inner = BabyLMDataset(root=str(p), block_size=block_size)
            self.vocab_size = self._inner.vocab_size
        elif p.suffix == ".json":
            self._mode = "online"
            self._init_online(p, block_size)
        else:
            raise ValueError(f"MixtureDataset expects .bin or .json, got {p.suffix}")

    def _init_online(self, config_path: Path, block_size: int):
        config = json.loads(config_path.read_text())
        weights = {int(k): v for k, v in config["weights"].items()}
        n_clusters = config["n_clusters"]

        cluster_dir = config_path.parent / "cluster_bins"
        self._clusters = {}
        self._weights = []
        self._cluster_ids = []

        for k in range(n_clusters):
            w = weights.get(k, 0.0)
            bin_path = cluster_dir / f"cluster_{k}.bin"
            if w < 1e-6 or not bin_path.exists():
                continue
            arr = np.fromfile(str(bin_path), dtype=np.uint16).astype(np.int64)
            n_chunks = len(arr) // (block_size + 1)
            if n_chunks == 0:
                continue
            chunks = torch.from_numpy(arr[:n_chunks * (block_size + 1)].copy()).view(n_chunks, block_size + 1)
            self._clusters[k] = chunks
            self._weights.append(w)
            self._cluster_ids.append(k)

        total_w = sum(self._weights)
        self._weights = [w / total_w for w in self._weights]
        self._cum_weights = np.cumsum(self._weights)
        self._total_chunks = sum(c.shape[0] for c in self._clusters.values())

        import tiktoken
        self.vocab_size = tiktoken.get_encoding("gpt2").n_vocab

    def __len__(self):
        if self._mode == "premixed":
            return len(self._inner)
        return self._total_chunks

    def __getitem__(self, idx):
        if self._mode == "premixed":
            return self._inner[idx]

        r = np.random.random()
        cluster_idx = int(np.searchsorted(self._cum_weights, r))
        cluster_idx = min(cluster_idx, len(self._cluster_ids) - 1)
        k = self._cluster_ids[cluster_idx]
        chunks = self._clusters[k]
        chunk_idx = np.random.randint(0, chunks.shape[0])
        chunk = chunks[chunk_idx]
        return chunk[:-1], chunk[1:]
