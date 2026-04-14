"""Data loading for BabyLM and other text datasets."""

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class BabyLMDataset(Dataset):
    """BabyLM dataset: tokenized text chunks for language modeling.

    Supports pre-tokenized .bin files, parquet (HuggingFace), .jsonl.zst,
    and plain text files.
    """

    def __init__(
        self,
        root: str = "datasets/babylm-strict-small",
        tokenizer_name: str = "gpt2",
        block_size: int = 1024,
    ):
        self.block_size = block_size

        root = Path(root)

        # Support .bin files directly (pre-tokenized with scripts/pretokenize.py)
        if root.suffix == ".bin" and root.is_file():
            import tiktoken
            enc = tiktoken.get_encoding(tokenizer_name)
            self.vocab_size = enc.n_vocab
            raw = np.fromfile(str(root), dtype=np.uint16)
            tokens = raw.astype(np.int64)
            n_tokens = len(tokens)
            n_chunks = n_tokens // (block_size + 1)
            self.tokens = torch.from_numpy(tokens[: n_chunks * (block_size + 1)].copy()).view(n_chunks, block_size + 1)
            print(f"BabyLMDataset: {n_tokens:,} tokens (pre-tokenized .bin) -> {n_chunks:,} chunks of {block_size}")
            return

        if not root.exists():
            raise FileNotFoundError(
                f"Dataset not found at {root}. "
                f"Expected .bin, parquet, .jsonl.zst, or text files."
            )

        # Load pre-tokenized or raw text
        pre_tokens, texts = self._load_tokens_or_texts(root)

        import tiktoken
        enc = tiktoken.get_encoding(tokenizer_name)
        self.vocab_size = enc.n_vocab

        eos_token = enc.n_vocab - 1  # 50256 for GPT-2 (<|endoftext|>)

        if pre_tokens is not None:
            tokens = pre_tokens
        else:
            tokens = []
            for text in texts:
                tokens.extend(enc.encode_ordinary(text))
                tokens.append(eos_token)  # EOS between documents
        self.tokens = torch.tensor(tokens, dtype=torch.long)

        # Chunk into block_size sequences
        n_chunks = len(self.tokens) // (block_size + 1)
        self.tokens = self.tokens[: n_chunks * (block_size + 1)]
        self.tokens = self.tokens.view(n_chunks, block_size + 1)

        print(f"BabyLMDataset: {len(tokens):,} tokens -> {n_chunks:,} chunks of {block_size}")

    def _load_tokens_or_texts(self, root: Path):
        """Load pre-tokenized ints or raw text from parquet/text files.

        Returns (tokens: list[int] | None, texts: list[str] | None).
        Exactly one will be non-None.
        """
        parquet_files = sorted(root.glob("*.parquet"))
        if parquet_files:
            import pyarrow.parquet as pq
            for f in parquet_files:
                table = pq.read_table(f)

                # Pre-tokenized format: input_ids column with list<int>
                if "input_ids" in table.column_names:
                    all_tokens = []
                    for row in table["input_ids"].to_pylist():
                        all_tokens.extend(row)
                        all_tokens.append(50256)  # EOS between documents
                    return all_tokens, None

                # Raw text format: text column with strings
                if "text" in table.column_names:
                    return None, table["text"].to_pylist()

                # Fallback: first column
                col = table.column_names[0]
                data = table[col].to_pylist()
                if data and isinstance(data[0], str):
                    return None, data
                elif data and isinstance(data[0], list):
                    all_tokens = []
                    for row in data:
                        all_tokens.extend(row)
                        all_tokens.append(50256)  # EOS between documents
                    return all_tokens, None

        # Zstandard-compressed JSONL files (recursive — supports nested category dirs)
        zst_files = sorted(root.rglob("*.jsonl.zst"))
        if zst_files:
            import zstandard, json, io
            texts = []
            for f in zst_files:
                dctx = zstandard.ZstdDecompressor()
                with open(f, "rb") as fh:
                    reader = dctx.stream_reader(fh)
                    text_reader = io.TextIOWrapper(reader, encoding="utf-8")
                    for line in text_reader:
                        row = json.loads(line)
                        text = row.get("text", "")
                        if text.strip():
                            texts.append(text)
            if texts:
                return None, texts

        # Plain text files
        texts = []
        for ext in ("*.train", "*.txt"):
            for f in sorted(root.glob(ext)):
                texts.append(f.read_text(encoding="utf-8"))

        if not texts:
            raise FileNotFoundError(
                f"No parquet, .jsonl.zst, .train, or .txt files found in {root}"
            )
        return None, texts

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = self.tokens[idx]
        x = chunk[:-1]       # input_ids
        y = chunk[1:]         # targets (shifted by 1)
        return x, y


def build_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    """Create a DataLoader with sensible defaults for training."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,  # unified memory — no pinning needed
        drop_last=True,
    )
