"""Pre-tokenize a dataset into a binary token file.

Supports multiprocessing for parallel tokenization and sharding for
multi-machine pipelines.

Usage:
    # Single machine, all cores:
    python scripts/pretokenize.py --input datasets/dolma --output datasets/dolma.bin \
        --tokenizer-path tokenizers/vidar-32k/tokenizer.json --workers 16

    # Multi-machine: each machine processes a shard, then cat:
    python scripts/pretokenize.py --input datasets/dolma --output /tmp/shard0.bin \
        --shard-id 0 --num-shards 2 --workers 16
    python scripts/pretokenize.py --input datasets/dolma --output /tmp/shard1.bin \
        --shard-id 1 --num-shards 2 --workers 16
    cat /tmp/shard0.bin /tmp/shard1.bin > datasets/dolma.bin

Output format: raw uint16 array, loadable with np.memmap or np.fromfile.
"""

import argparse
import io
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np


def _tokenize_file_tiktoken(args):
    """Worker: tokenize one file with tiktoken. Returns (tokens_array, n_docs)."""
    fpath, tokenizer_name, eos = args
    import tiktoken
    enc = tiktoken.get_encoding(tokenizer_name)
    tokens = []
    n_docs = 0
    fpath = Path(fpath)

    if fpath.suffix == ".zst":
        import zstandard
        dctx = zstandard.ZstdDecompressor()
        with open(fpath, "rb") as fh:
            reader = dctx.stream_reader(fh)
            text_reader = io.TextIOWrapper(reader, encoding="utf-8")
            for line in text_reader:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = row.get("text", "")
                if text.strip():
                    tokens.extend(enc.encode_ordinary(text))
                    tokens.append(eos)
                    n_docs += 1
    elif fpath.suffix == ".parquet":
        import pyarrow.parquet as pq
        table = pq.read_table(fpath)
        if "text" in table.column_names:
            for text in table["text"].to_pylist():
                if text and text.strip():
                    tokens.extend(enc.encode_ordinary(text))
                    tokens.append(eos)
                    n_docs += 1
    else:
        text = fpath.read_text(encoding="utf-8")
        if text.strip():
            tokens.extend(enc.encode_ordinary(text))
            tokens.append(eos)
            n_docs += 1

    return np.array(tokens, dtype=np.uint16) if tokens else np.array([], dtype=np.uint16), n_docs


def _tokenize_file_hf(args):
    """Worker: tokenize one file with HuggingFace tokenizer. Returns (tokens_array, n_docs)."""
    fpath, tokenizer_path, eos = args
    from tokenizers import Tokenizer as HFTokenizer
    tok = HFTokenizer.from_file(tokenizer_path)
    tokens = []
    n_docs = 0
    fpath = Path(fpath)

    if fpath.suffix == ".zst":
        import zstandard
        dctx = zstandard.ZstdDecompressor()
        with open(fpath, "rb") as fh:
            reader = dctx.stream_reader(fh)
            text_reader = io.TextIOWrapper(reader, encoding="utf-8")
            for line in text_reader:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = row.get("text", "")
                if text.strip():
                    tokens.extend(tok.encode(text).ids)
                    tokens.append(eos)
                    n_docs += 1
    elif fpath.suffix == ".parquet":
        import pyarrow.parquet as pq
        table = pq.read_table(fpath)
        if "text" in table.column_names:
            for text in table["text"].to_pylist():
                if text and text.strip():
                    tokens.extend(tok.encode(text).ids)
                    tokens.append(eos)
                    n_docs += 1
    else:
        text = fpath.read_text(encoding="utf-8")
        if text.strip():
            tokens.extend(tok.encode(text).ids)
            tokens.append(eos)
            n_docs += 1

    return np.array(tokens, dtype=np.uint16) if tokens else np.array([], dtype=np.uint16), n_docs


def main():
    parser = argparse.ArgumentParser(description="Pre-tokenize dataset to binary")
    parser.add_argument("--input", required=True, help="Input dataset directory")
    parser.add_argument("--output", required=True, help="Output .bin file path")
    parser.add_argument("--tokenizer", default="gpt2", help="Tiktoken encoding name")
    parser.add_argument("--tokenizer-path", default=None,
                        help="HuggingFace tokenizer .json path (overrides --tokenizer)")
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of parallel workers (0 = all cores)")
    parser.add_argument("--shard-id", type=int, default=0,
                        help="Shard index for multi-machine (0-based)")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total number of shards (1 = no sharding)")
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
    else:
        import tiktoken
        enc = tiktoken.get_encoding(args.tokenizer)
        eos = enc.n_vocab - 1
        vocab_size = enc.n_vocab

    all_files = sorted(root.rglob("*.jsonl.zst"))
    fmt = "jsonl.zst"
    if not all_files:
        all_files = sorted(root.rglob("*.parquet"))
        fmt = "parquet"
    if not all_files:
        all_files = sorted(root.glob("*.txt")) + sorted(root.glob("*.train"))
        fmt = "txt"
    if not all_files:
        print(f"No supported files found in {root}")
        sys.exit(1)

    if args.num_shards > 1:
        all_files = [f for i, f in enumerate(all_files) if i % args.num_shards == args.shard_id]
        print(f"Shard {args.shard_id}/{args.num_shards}: {len(all_files)} files")

    n_workers = args.workers or os.cpu_count()
    tok_name = args.tokenizer_path or args.tokenizer
    print(f"Format: {fmt}, {len(all_files)} files, {n_workers} workers")
    print(f"Tokenizer: {tok_name} (vocab={vocab_size})")

    if args.tokenizer_path:
        worker_fn = _tokenize_file_hf
        worker_args = [(str(f), args.tokenizer_path, eos) for f in all_files]
    else:
        worker_fn = _tokenize_file_tiktoken
        worker_args = [(str(f), args.tokenizer, eos) for f in all_files]

    total_tokens = 0
    total_docs = 0
    all_chunks = []

    with Pool(n_workers) as pool:
        for i, (chunk, n_docs) in enumerate(pool.imap_unordered(worker_fn, worker_args)):
            if len(chunk) > 0:
                all_chunks.append(chunk)
                total_tokens += len(chunk)
                total_docs += n_docs
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(all_files)} files, {total_docs:,} docs, {total_tokens:,} tokens")

    print(f"\nTotal: {total_docs:,} docs, {total_tokens:,} tokens")
    print("Concatenating...")

    tokens = np.concatenate(all_chunks)
    assert len(tokens) == total_tokens

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    tokens.tofile(args.output)
    size_gb = os.path.getsize(args.output) / 1e9
    print(f"Saved: {args.output} ({size_gb:.2f} GB, {total_tokens:,} tokens)")


if __name__ == "__main__":
    main()
