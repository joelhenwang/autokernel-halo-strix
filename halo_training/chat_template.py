"""ChatML template, special token registration, and embedding resize utilities.

Wraps tiktoken (which doesn't support custom tokens) with a ChatMLTokenizer
that handles <|im_start|>, <|im_end|>, <|pad|> and optional tool call tokens.
"""

import re
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import tiktoken

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
PAD = "<|pad|>"
TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"

IGNORE_INDEX = -100

CHATML_TOKENS: Dict[str, int] = {
    IM_START: 50257,
    IM_END: 50258,
    PAD: 50259,
}

STAGE_B_TOKENS: Dict[str, int] = {
    TOOL_CALL_START: 50260,
    TOOL_CALL_END: 50261,
}


# ---------------------------------------------------------------------------
# ChatML Tokenizer (tiktoken wrapper)
# ---------------------------------------------------------------------------

class ChatMLTokenizer:
    """Wraps tiktoken with support for ChatML special tokens."""

    def __init__(self, base: str = "gpt2", extra_tokens: Optional[Dict[str, int]] = None):
        self._base = tiktoken.get_encoding(base)
        self._extra = dict(extra_tokens) if extra_tokens else dict(CHATML_TOKENS)
        self._id_to_token = {v: k for k, v in self._extra.items()}
        # Build regex pattern that matches any special token string
        escaped = [re.escape(tok) for tok in sorted(self._extra, key=len, reverse=True)]
        self._split_pattern = re.compile("(" + "|".join(escaped) + ")")

    @property
    def vocab_size(self) -> int:
        return self._base.n_vocab + len(self._extra)

    @property
    def eos_token_id(self) -> int:
        return self._base.n_vocab - 1  # 50256 for GPT-2

    @property
    def im_start_id(self) -> int:
        return self._extra[IM_START]

    @property
    def im_end_id(self) -> int:
        return self._extra[IM_END]

    @property
    def pad_id(self) -> int:
        return self._extra[PAD]

    def encode(self, text: str) -> List[int]:
        """Encode text, handling special tokens by splitting around them."""
        parts = self._split_pattern.split(text)
        ids = []
        for part in parts:
            if not part:
                continue
            if part in self._extra:
                ids.append(self._extra[part])
            else:
                ids.extend(self._base.encode_ordinary(part))
        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to text, reinserting special token strings."""
        result = []
        buffer = []
        for tid in ids:
            if tid in self._id_to_token:
                if buffer:
                    result.append(self._base.decode(buffer))
                    buffer = []
                result.append(self._id_to_token[tid])
            else:
                buffer.append(tid)
        if buffer:
            result.append(self._base.decode(buffer))
        return "".join(result)

    def encode_chatml(self, messages: List[Dict[str, str]]) -> List[int]:
        """Encode a list of {role, content} messages into ChatML token sequence.

        Format per turn:
            <|im_start|>role\ncontent<|im_end|>\n
        """
        ids = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            # <|im_start|>role\n
            ids.append(self.im_start_id)
            ids.extend(self._base.encode_ordinary(role + "\n"))
            # content
            ids.extend(self._base.encode_ordinary(content))
            # <|im_end|>\n
            ids.append(self.im_end_id)
            ids.extend(self._base.encode_ordinary("\n"))
        return ids

    def start_assistant_turn(self) -> List[int]:
        """Return token IDs for the start of an assistant turn (for generation)."""
        ids = [self.im_start_id]
        ids.extend(self._base.encode_ordinary("assistant\n"))
        return ids


# ---------------------------------------------------------------------------
# Tokenizer Factory
# ---------------------------------------------------------------------------

def build_tokenizer(phase: str = "sft") -> ChatMLTokenizer:
    """Create the appropriate tokenizer for a training phase.

    Args:
        phase: "sft" for Stages C/A, "domain-sft" for Stage B (adds tool tokens).
    """
    if phase == "domain-sft":
        tokens = {**CHATML_TOKENS, **STAGE_B_TOKENS}
    else:
        tokens = CHATML_TOKENS
    return ChatMLTokenizer(base="gpt2", extra_tokens=tokens)


# ---------------------------------------------------------------------------
# Embedding Resize
# ---------------------------------------------------------------------------

def resize_embeddings(model: nn.Module, new_vocab_size: int) -> nn.Module:
    """Safely resize token embeddings and LM head, preserving weight tying.

    Must be called AFTER checkpoint load (so shapes match during load)
    and BEFORE autokernel.optimize() (so patterns see final shapes).

    Args:
        model: Model with .tok_embeddings (nn.Embedding) and .output (nn.Linear).
        new_vocab_size: Target vocabulary size (e.g., 50260 for ChatML).

    Returns:
        The model with resized embeddings (modified in-place).
    """
    old_embed = model.tok_embeddings
    old_vocab = old_embed.num_embeddings
    d_model = old_embed.embedding_dim

    if new_vocab_size == old_vocab:
        return model

    # Create new embedding — initialize new rows from mean of existing embeddings
    # so they produce reasonable hidden states from step 1 (zero-init causes
    # RMSNorm/attention corruption and training divergence)
    new_embed = nn.Embedding(new_vocab_size, d_model, device=old_embed.weight.device,
                             dtype=old_embed.weight.dtype)
    with torch.no_grad():
        new_embed.weight[:old_vocab] = old_embed.weight
        embed_mean = old_embed.weight.mean(dim=0)
        embed_std = old_embed.weight.std()
        for i in range(old_vocab, new_vocab_size):
            new_embed.weight[i] = embed_mean + torch.randn_like(embed_mean) * (embed_std * 0.01)

    model.tok_embeddings = new_embed

    # Recreate output linear and retie weights
    model.output = nn.Linear(d_model, new_vocab_size, bias=False,
                             device=new_embed.weight.device, dtype=new_embed.weight.dtype)
    model.output.weight = model.tok_embeddings.weight

    return model
