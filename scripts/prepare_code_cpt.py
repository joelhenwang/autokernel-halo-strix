#!/usr/bin/env python3
"""Download Python code from codeparrot/github-code-clean and prepare for CPT.

Streams parquet files, concatenates Python source code with <|endoftext|>
separators, and writes raw text files compatible with halo_training's
StreamingTextDataset (same format as Dolma/GPT-training-small).

Usage:
    python scripts/prepare_code_cpt.py --output-dir datasets/python-code --max-tokens 1000000000
    python scripts/prepare_code_cpt.py --output-dir datasets/python-code --max-files 3  # ~500M tokens
"""

import argparse
import json
import sys
from pathlib import Path

EOS = "<|endoftext|>"


def main():
    parser = argparse.ArgumentParser(description="Prepare Python code for continued pretraining")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--max-tokens", type=int, default=1_500_000_000, help="Stop after N estimated tokens")
    parser.add_argument("--max-files", type=int, default=None, help="Download at most N parquet files")
    parser.add_argument("--min-length", type=int, default=100, help="Skip files shorter than N chars")
    parser.add_argument("--max-length", type=int, default=50000, help="Truncate files longer than N chars")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get parquet URLs
    print("Fetching parquet file list...")
    try:
        import subprocess
        result = subprocess.run(
            ["hf", "datasets", "parquet", "codeparrot/github-code-clean",
             "--subset", "Python-all", "--format", "json"],
            capture_output=True, text=True, timeout=30
        )
        import json as _json
        parquet_info = _json.loads(result.stdout)
    except Exception as e:
        print(f"Error getting parquet URLs: {e}")
        sys.exit(1)

    urls = [p["url"] for p in parquet_info]
    if args.max_files:
        urls = urls[:args.max_files]

    print(f"Will process {len(urls)} parquet files")

    total_chars = 0
    total_docs = 0
    total_skipped = 0
    file_idx = 0
    current_chunk = []
    chunk_chars = 0
    CHUNK_SIZE = 100_000_000  # ~100MB per output file

    def flush_chunk():
        nonlocal file_idx, current_chunk, chunk_chars
        if not current_chunk:
            return
        out_path = output_dir / f"python_code_{file_idx:04d}.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(EOS.join(current_chunk))
            f.write(EOS)
        size_mb = out_path.stat().st_size / 1e6
        print(f"  Wrote {out_path.name}: {len(current_chunk)} docs, {size_mb:.1f} MB")
        file_idx += 1
        current_chunk = []
        chunk_chars = 0

    # Stream and process
    try:
        import pyarrow.parquet as pq
        import io
        import urllib.request
    except ImportError:
        print("ERROR: Install pyarrow: pip install pyarrow")
        sys.exit(1)

    for i, url in enumerate(urls):
        est_tokens = total_chars // 4
        if est_tokens >= args.max_tokens:
            print(f"Reached token limit ({est_tokens:,} >= {args.max_tokens:,})")
            break

        print(f"Processing parquet {i+1}/{len(urls)}: {url.split('/')[-1]} "
              f"(total so far: {total_docs:,} docs, ~{est_tokens:,} tokens)")

        try:
            # Download parquet to memory
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()

            # Read with pyarrow
            table = pq.read_table(io.BytesIO(data))
            codes = table.column("code").to_pylist()

            for code in codes:
                if not code or len(code) < args.min_length:
                    total_skipped += 1
                    continue

                # Truncate very long files
                if len(code) > args.max_length:
                    code = code[:args.max_length]

                current_chunk.append(code)
                chunk_chars += len(code)
                total_chars += len(code)
                total_docs += 1

                if chunk_chars >= CHUNK_SIZE:
                    flush_chunk()

                # Check token limit
                if total_chars // 4 >= args.max_tokens:
                    break

        except Exception as e:
            print(f"  Error processing {url}: {e}")
            continue

    flush_chunk()

    est_tokens = total_chars // 4
    print(f"\n{'='*60}")
    print(f"Python Code CPT Data Summary")
    print(f"{'='*60}")
    print(f"Total documents: {total_docs:,}")
    print(f"Total skipped (too short): {total_skipped:,}")
    print(f"Total chars: {total_chars:,}")
    print(f"Estimated tokens: {est_tokens:,}")
    print(f"Output files: {file_idx}")
    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()
