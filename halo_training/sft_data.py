"""SFT dataset with format adapters, ChatML formatting, loss masking, and packing.

Supports Alpaca, ShareGPT, and ChatML dataset formats. Converts all to ChatML
template, tokenizes, builds loss-masked labels, and optionally packs short
conversations into full-length sequences.

Returns (input_ids, labels) in the same format as BabyLMDataset — labels use
-100 for masked positions, compatible with nn.CrossEntropyLoss(ignore_index=-100).
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from halo_training.chat_template import (
    ChatMLTokenizer, IGNORE_INDEX, IM_START, IM_END, PAD,
)

# ---------------------------------------------------------------------------
# Format Adapters
# ---------------------------------------------------------------------------


def convert_alpaca(example: dict, system_prompt: str) -> List[Dict[str, str]]:
    """Convert Alpaca format to ChatML messages."""
    messages = [{"role": "system", "content": system_prompt}]
    user_content = example["instruction"]
    if example.get("input"):
        user_content += "\n" + example["input"]
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": example["output"]})
    return messages


def convert_sharegpt(example: dict, system_prompt: str) -> List[Dict[str, str]]:
    """Convert ShareGPT/OpenHermes format to ChatML messages."""
    role_map = {"human": "user", "gpt": "assistant", "system": "system"}
    conversations = example.get("conversations", [])
    messages = []
    has_system = any(c.get("from") == "system" for c in conversations)
    if not has_system:
        messages.append({"role": "system", "content": system_prompt})
    for turn in conversations:
        role = role_map.get(turn["from"], turn["from"])
        messages.append({"role": role, "content": turn["value"]})
    return messages


def convert_chatml(example: dict, system_prompt: str) -> List[Dict[str, str]]:
    """Pass through ChatML format (messages already in {role, content} form)."""
    messages = example.get("messages", [])
    has_system = any(m.get("role") == "system" for m in messages)
    if not has_system:
        messages = [{"role": "system", "content": system_prompt}] + messages
    return messages


FORMAT_ADAPTERS = {
    "alpaca": convert_alpaca,
    "sharegpt": convert_sharegpt,
    "chatml": convert_chatml,
}

# ---------------------------------------------------------------------------
# Well-known HuggingFace dataset mappings
# ---------------------------------------------------------------------------

HF_DATASETS = {
    "alpaca": ("yahma/alpaca-cleaned", "alpaca"),
    "openhermes": ("teknium/OpenHermes-2.5", "sharegpt"),
}

# Local dataset shortcuts (pre-processed by scripts/prepare_swe_data.py)
LOCAL_DATASETS = {
    "swe-repair": ("datasets/swe_prepared/swe_code_repair.jsonl", "chatml"),
    "swe-explain": ("datasets/swe_prepared/swe_bug_explain.jsonl", "chatml"),
    "swe-localize": ("datasets/swe_prepared/swe_localize.jsonl", "chatml"),
}

# ---------------------------------------------------------------------------
# SFT Dataset
# ---------------------------------------------------------------------------


class SFTDataset(Dataset):
    """Instruction-tuning dataset with ChatML formatting and loss masking.

    Returns (input_ids, labels) where:
        - input_ids: shape (block_size,), full token sequence
        - labels: shape (block_size,), with -100 for non-assistant tokens

    Compatible with the existing BabyLMDataset interface — trainer needs no changes.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: ChatMLTokenizer,
        format: str = "alpaca",
        block_size: int = 2048,
        system_prompt: str = "You are a helpful assistant.",
        pack: bool = True,
        max_examples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.system_prompt = system_prompt
        self.pack = pack

        adapter = FORMAT_ADAPTERS.get(format)
        if adapter is None:
            raise ValueError(f"Unknown format '{format}'. Choose from: {list(FORMAT_ADAPTERS)}")

        # Load raw examples
        raw_examples = self._load_data(data_path, format)
        if max_examples:
            raw_examples = raw_examples[:max_examples]

        # Convert to ChatML messages, tokenize, build labels
        tokenized = []
        skipped = 0
        for ex in raw_examples:
            messages = adapter(ex, system_prompt)
            tokens, labels = self._build_example(messages)
            if len(tokens) > block_size + 1:
                # Truncate to block_size + 1
                tokens = tokens[:block_size + 1]
                labels = labels[:block_size + 1]
            if len(tokens) < 4:  # too short to be useful
                skipped += 1
                continue
            tokenized.append((tokens, labels))

        if skipped:
            print(f"  SFTDataset: skipped {skipped} examples (too short)")

        # Pack or pad to uniform length
        if pack:
            self.sequences = self._pack_conversations(tokenized)
        else:
            self.sequences = self._pad_conversations(tokenized)

        print(f"  SFTDataset: {len(raw_examples)} examples -> {len(self.sequences)} sequences "
              f"(block_size={block_size}, packed={pack})")

    def _load_data(self, data_path: str, format: str) -> list:
        """Load raw examples from HuggingFace, JSONL, or parquet."""
        # Check if it's a well-known HF dataset name
        if data_path in HF_DATASETS:
            hf_name, _ = HF_DATASETS[data_path]
            return self._load_hf(hf_name)

        # Check local dataset shortcuts
        if data_path in LOCAL_DATASETS:
            local_path, _ = LOCAL_DATASETS[data_path]
            return self._load_jsonl(Path(local_path))

        path = Path(data_path)

        # Local JSONL
        if path.suffix in (".jsonl", ".json"):
            return self._load_jsonl(path)

        # Local parquet
        if path.suffix == ".parquet":
            return self._load_parquet(path)

        # Directory — try JSONL files inside
        if path.is_dir():
            jsonl_files = sorted(path.glob("*.jsonl"))
            if jsonl_files:
                examples = []
                for f in jsonl_files:
                    examples.extend(self._load_jsonl(f))
                return examples
            # Try parquet
            pq_files = sorted(path.glob("*.parquet"))
            if pq_files:
                examples = []
                for f in pq_files:
                    examples.extend(self._load_parquet(f))
                return examples

        # Try as HuggingFace dataset name
        return self._load_hf(data_path)

    def _load_hf(self, name: str) -> list:
        """Load from HuggingFace datasets library."""
        try:
            from datasets import load_dataset
            ds = load_dataset(name, split="train")
            return list(ds)
        except ImportError:
            raise ImportError("Install 'datasets' package: pip install datasets")

    def _load_jsonl(self, path: Path) -> list:
        examples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))
        return examples

    def _load_parquet(self, path: Path) -> list:
        try:
            import pandas as pd
            df = pd.read_parquet(path)
            return df.to_dict("records")
        except ImportError:
            raise ImportError("Install 'pandas' and 'pyarrow': pip install pandas pyarrow")

    def _build_example(self, messages: List[Dict[str, str]]) -> Tuple[List[int], List[int]]:
        """Convert ChatML messages to (tokens, labels) with loss masking.

        Labels use the standard next-token prediction shift:
        - tokens[i] is the input at position i
        - labels[i] is the target (= tokens[i+1]) at position i
        - Non-assistant positions in labels are set to IGNORE_INDEX (-100)

        The sequence is stored at length block_size+1 so that __getitem__ can
        return (seq[:-1], seq[1:]) matching BabyLMDataset's interface.
        """
        all_tokens = []
        all_is_assistant = []  # True for tokens the model should predict

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            # Header: <|im_start|>role\n
            header_ids = [self.tokenizer.im_start_id]
            header_ids.extend(self.tokenizer._base.encode_ordinary(role + "\n"))

            # Content — use full encoder for assistant (handles <tool_call> tokens)
            if role == "assistant":
                content_ids = self.tokenizer.encode(content)
            else:
                content_ids = self.tokenizer._base.encode_ordinary(content)

            # Footer: <|im_end|>\n
            footer_ids = [self.tokenizer.im_end_id]
            footer_ids.extend(self.tokenizer._base.encode_ordinary("\n"))

            turn_ids = header_ids + content_ids + footer_ids
            is_assistant = role == "assistant"

            # Mark which tokens are "assistant" (learnable)
            # Header is NOT learnable (even for assistant turns — it's structural)
            # Content + footer ARE learnable for assistant turns
            turn_mask = (
                [False] * len(header_ids) +
                [is_assistant] * len(content_ids) +
                [is_assistant] * len(footer_ids)  # includes <|im_end|> — model must learn to emit it
            )

            all_tokens.extend(turn_ids)
            all_is_assistant.extend(turn_mask)

        # Build labels with next-token shift:
        # labels[i] = tokens[i+1] if tokens[i+1] is assistant-generated, else IGNORE_INDEX
        labels = []
        for i in range(len(all_tokens)):
            if i + 1 < len(all_tokens) and all_is_assistant[i + 1]:
                labels.append(all_tokens[i + 1])
            else:
                labels.append(IGNORE_INDEX)

        return all_tokens, labels

    def _pack_conversations(
        self, examples: List[Tuple[List[int], List[int]]]
    ) -> List[torch.Tensor]:
        """Pack short conversations into block_size+1 sequences."""
        target_len = self.block_size + 1
        packed = []
        cur_tokens = []
        cur_labels = []

        for tokens, labels in examples:
            needed = len(tokens)
            if cur_tokens and len(cur_tokens) + needed > target_len:
                # Pad remainder and finalize
                remaining = target_len - len(cur_tokens)
                cur_tokens.extend([self.tokenizer.pad_id] * remaining)
                cur_labels.extend([IGNORE_INDEX] * remaining)
                packed.append(self._to_tensor(cur_tokens, cur_labels))
                cur_tokens = []
                cur_labels = []

            # If single example is too long, truncate
            if needed > target_len:
                tokens = tokens[:target_len]
                labels = labels[:target_len]
                needed = target_len

            cur_tokens.extend(tokens)
            cur_labels.extend(labels)

        # Finalize last partial sequence
        if cur_tokens:
            remaining = target_len - len(cur_tokens)
            cur_tokens.extend([self.tokenizer.pad_id] * remaining)
            cur_labels.extend([IGNORE_INDEX] * remaining)
            packed.append(self._to_tensor(cur_tokens, cur_labels))

        return packed

    def _pad_conversations(
        self, examples: List[Tuple[List[int], List[int]]]
    ) -> List[torch.Tensor]:
        """Right-pad each conversation individually to block_size+1."""
        target_len = self.block_size + 1
        padded = []
        for tokens, labels in examples:
            if len(tokens) > target_len:
                tokens = tokens[:target_len]
                labels = labels[:target_len]
            remaining = target_len - len(tokens)
            tokens.extend([self.tokenizer.pad_id] * remaining)
            labels.extend([IGNORE_INDEX] * remaining)
            padded.append(self._to_tensor(tokens, labels))
        return padded

    def _to_tensor(self, tokens: List[int], labels: List[int]) -> torch.Tensor:
        """Stack tokens and labels into a single (2, block_size+1) tensor for storage."""
        return torch.stack([
            torch.tensor(tokens, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        ])

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (input_ids, labels) matching BabyLMDataset interface.

        input_ids: seq[:-1]  (block_size tokens)
        labels:    seq[1:]   but with -100 for masked positions
        """
        seq = self.sequences[idx]  # shape: (2, block_size+1)
        tokens = seq[0]  # full token sequence
        labels = seq[1]  # pre-computed labels with masking

        # Standard LM shift: input = all but last, target = all but first
        input_ids = tokens[:-1]
        targets = labels[:-1]  # labels already encode the next-token shift
        return input_ids, targets
