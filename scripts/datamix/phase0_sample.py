"""Phase 0: Stream & reservoir-sample from HuggingFace dataset.

Streams a large dataset (e.g., Dolma 100B) without downloading it fully.
Outputs JSONL shards of sampled documents for downstream embedding/clustering.

Usage:
    python scripts/datamix/phase0_sample.py --dataset allenai/dolma --max-docs 2000000
    python scripts/datamix/phase0_sample.py --dataset HuggingFaceFW/fineweb-edu --max-docs 1000000
"""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.datamix import load_pipeline_state, save_pipeline_state, phase_dir


def doc_id(text: str) -> str:
    return hashlib.md5(text[:2000].encode("utf-8", errors="replace")).hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Phase 0: Sample documents from HF dataset")
    parser.add_argument("--dataset", required=True, help="HuggingFace dataset name")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="text", help="Field name containing text")
    parser.add_argument("--max-docs", type=int, default=2_000_000)
    parser.add_argument("--shard-size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--state-dir", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    base = Path(args.state_dir) if args.state_dir else None
    state = load_pipeline_state(base)
    out_dir = phase_dir("phase0", base)

    if args.resume and state.get("phase0", {}).get("status") == "complete":
        print(f"Phase 0 already complete ({state['phase0']['docs_sampled']} docs)")
        return

    random.seed(args.seed)

    from datasets import load_dataset
    print(f"Streaming {args.dataset} (split={args.split})...")
    ds = load_dataset(args.dataset, split=args.split, streaming=True)

    docs = []
    shard_idx = 0
    total_sampled = 0
    seen = 0

    if args.resume:
        existing_shards = sorted(out_dir.glob("shard_*.jsonl"))
        if existing_shards:
            for s in existing_shards:
                count = sum(1 for _ in open(s))
                total_sampled += count
            shard_idx = len(existing_shards)
            print(f"Resuming from shard {shard_idx}, {total_sampled} docs already sampled")

    for example in ds:
        text = example.get(args.text_field, "")
        if not text or len(text) < 50:
            continue
        seen += 1

        if total_sampled < args.max_docs:
            docs.append({"doc_id": doc_id(text), "text": text})
            total_sampled += 1

            if len(docs) >= args.shard_size:
                shard_path = out_dir / f"shard_{shard_idx:04d}.jsonl"
                with open(shard_path, "w", encoding="utf-8") as f:
                    for d in docs:
                        f.write(json.dumps(d, ensure_ascii=False) + "\n")
                print(f"  Wrote {shard_path.name} ({len(docs)} docs, {total_sampled}/{args.max_docs} total)")
                docs = []
                shard_idx += 1

        if total_sampled >= args.max_docs:
            break

        if seen % 500_000 == 0:
            print(f"  Scanned {seen:,} docs, sampled {total_sampled:,}")

    if docs:
        shard_path = out_dir / f"shard_{shard_idx:04d}.jsonl"
        with open(shard_path, "w", encoding="utf-8") as f:
            for d in docs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"  Wrote {shard_path.name} ({len(docs)} docs)")

    state["phase0"] = {
        "status": "complete",
        "docs_sampled": total_sampled,
        "dataset": args.dataset,
        "shards": shard_idx + (1 if docs else 0),
    }
    save_pipeline_state(state, base)
    print(f"Phase 0 complete: {total_sampled:,} docs sampled from {args.dataset}")


if __name__ == "__main__":
    main()
