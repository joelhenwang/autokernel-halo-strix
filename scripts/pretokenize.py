"""Pre-tokenize a dataset of .jsonl.zst files into a single binary token file.

Reads all .jsonl.zst files recursively, tokenizes with tiktoken GPT-2,
and saves as a flat numpy array of uint16 token IDs. The output can be
memory-mapped for fast training without re-tokenizing.

Usage:
    python scripts/pretokenize.py \
        --input datasets/common_crawl_sample \
        --output datasets/common_crawl_sample.bin

Output format: raw uint16 array, loadable with np.memmap or np.fromfile.
"""

import argparse
import io
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
import tiktoken


def iter_texts_jsonl_zst(root: Path):
    """Yield text strings from all .jsonl.zst files under root."""
    import zstandard

    zst_files = sorted(root.rglob("*.jsonl.zst"))
    print(f"Found {len(zst_files)} .jsonl.zst files")

    for i, f in enumerate(zst_files):
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

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(zst_files)} files...")


def iter_texts_parquet(root: Path):
    """Yield text strings from all .parquet files under root."""
    import pyarrow.parquet as pq

    parquet_files = sorted(root.rglob("*.parquet"))
    print(f"Found {len(parquet_files)} .parquet files")

    for f in parquet_files:
        table = pq.read_table(f)
        if "text" in table.column_names:
            for text in table["text"].to_pylist():
                if text and text.strip():
                    yield text


def iter_texts_plain(root: Path):
    """Yield text strings from .txt and .train files."""
    for ext in ("*.train", "*.txt"):
        for f in sorted(root.glob(ext)):
            text = f.read_text(encoding="utf-8")
            if text.strip():
                yield text


def main():
    parser = argparse.ArgumentParser(description="Pre-tokenize dataset to binary")
    parser.add_argument("--input", required=True, help="Input dataset directory")
    parser.add_argument("--output", required=True, help="Output .bin file path")
    parser.add_argument("--tokenizer", default="gpt2", help="Tiktoken encoding name")
    parser.add_argument("--tokenizer-path", default=None,
                        help="HuggingFace tokenizer .json path (overrides --tokenizer)")
    args = parser.parse_args()

    root = Path(args.input)
    if not root.exists():
        print(f"Error: {root} does not exist")
        sys.exit(1)

    if args.tokenizer_path:
        from tokenizers import Tokenizer as HFTokenizer
        hf_tok = HFTokenizer.from_file(args.tokenizer_path)
        eos = hf_tok.token_to_id("<|endoftext|>")
        vocab_size = hf_tok.get_vocab_size()
        enc = None
    else:
        hf_tok = None
        enc = tiktoken.get_encoding(args.tokenizer)
        eos = enc.n_vocab - 1
        vocab_size = enc.n_vocab

    # Detect format
    zst_files = list(root.rglob("*.jsonl.zst"))
    parquet_files = list(root.rglob("*.parquet"))
    txt_files = list(root.glob("*.txt")) + list(root.glob("*.train"))

    if zst_files:
        text_iter = iter_texts_jsonl_zst(root)
        fmt = "jsonl.zst"
    elif parquet_files:
        text_iter = iter_texts_parquet(root)
        fmt = "parquet"
    elif txt_files:
        text_iter = iter_texts_plain(root)
        fmt = "txt"
    else:
        print(f"No supported files found in {root}")
        sys.exit(1)

    print(f"Format: {fmt}")
    tok_name = args.tokenizer_path or args.tokenizer
    print(f"Tokenizing with {tok_name} (vocab={vocab_size})...")

    # Tokenize in chunks to manage memory
    CHUNK_SIZE = 100_000  # docs per chunk
    all_tokens = []
    total_docs = 0
    total_tokens = 0

    buf = []
    for text in text_iter:
        tokens = hf_tok.encode(text).ids if hf_tok else enc.encode_ordinary(text)
        buf.extend(tokens)
        buf.append(eos)
        total_docs += 1

        if total_docs % CHUNK_SIZE == 0:
            chunk = np.array(buf, dtype=np.uint16)
            all_tokens.append(chunk)
            total_tokens += len(buf)
            print(f"  {total_docs:,} docs, {total_tokens:,} tokens so far...")
            buf = []

    # Final leftover
    if buf:
        chunk = np.array(buf, dtype=np.uint16)
        all_tokens.append(chunk)
        total_tokens += len(buf)

    print(f"\nTotal: {total_docs:,} docs, {total_tokens:,} tokens")
    print(f"Concatenating...")

    tokens = np.concatenate(all_tokens)
    assert len(tokens) == total_tokens

    # Save as raw uint16
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    tokens.tofile(args.output)
    size_gb = os.path.getsize(args.output) / 1e9
    print(f"Saved: {args.output} ({size_gb:.2f} GB, {total_tokens:,} tokens)")


if __name__ == "__main__":
    main()
