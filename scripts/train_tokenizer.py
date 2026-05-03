"""Train a custom BPE tokenizer on a dataset.

Usage:
    python scripts/train_tokenizer.py \
        --input datasets/dolma-10b \
        --output tokenizers/vidar-32k \
        --vocab-size 32000

Requires: pip install tokenizers
"""

import argparse
import io
import json
import os
import sys
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, trainers


def iter_texts(root: Path, max_texts: int = 0):
    """Yield text strings from .jsonl.zst, .parquet, or .txt files."""
    count = 0

    zst_files = sorted(root.rglob("*.jsonl.zst"))
    if zst_files:
        import zstandard
        for f in zst_files:
            dctx = zstandard.ZstdDecompressor()
            with open(f, "rb") as fh:
                reader = dctx.stream_reader(fh)
                text_reader = io.TextIOWrapper(reader, encoding="utf-8")
                for line in text_reader:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = row.get("text", "")
                    if text.strip():
                        yield text
                        count += 1
                        if max_texts and count >= max_texts:
                            return

    parquet_files = sorted(root.rglob("*.parquet"))
    if parquet_files and (not max_texts or count < max_texts):
        import pyarrow.parquet as pq
        for f in parquet_files:
            table = pq.read_table(f)
            if "text" in table.column_names:
                for text in table["text"].to_pylist():
                    if text and text.strip():
                        yield text
                        count += 1
                        if max_texts and count >= max_texts:
                            return

    txt_files = sorted(root.glob("*.txt")) + sorted(root.glob("*.train"))
    if txt_files and (not max_texts or count < max_texts):
        for f in txt_files:
            text = f.read_text(encoding="utf-8")
            if text.strip():
                yield text
                count += 1
                if max_texts and count >= max_texts:
                    return


def main():
    parser = argparse.ArgumentParser(description="Train BPE tokenizer")
    parser.add_argument("--input", required=True, help="Input dataset directory")
    parser.add_argument("--output", required=True, help="Output directory for tokenizer")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--max-texts", type=int, default=0,
                        help="Max documents to train on (0 = all)")
    args = parser.parse_args()

    root = Path(args.input)
    if not root.exists():
        print(f"Error: {root} does not exist")
        sys.exit(1)

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    special_tokens = ["<|endoftext|>", "<|pad|>"]
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    print(f"Training {args.vocab_size}-token BPE on {root}...")
    tokenizer.train_from_iterator(iter_texts(root, args.max_texts), trainer=trainer)

    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, "tokenizer.json")
    tokenizer.save(out_path)

    eos_id = tokenizer.token_to_id("<|endoftext|>")
    pad_id = tokenizer.token_to_id("<|pad|>")
    vocab_size = tokenizer.get_vocab_size()

    print(f"Saved: {out_path}")
    print(f"Vocab size: {vocab_size}")
    print(f"EOS id: {eos_id}, PAD id: {pad_id}")

    test = "The quick brown fox jumps over the lazy dog."
    encoded = tokenizer.encode(test)
    print(f"Test: \"{test}\" -> {len(encoded.ids)} tokens")


if __name__ == "__main__":
    main()
