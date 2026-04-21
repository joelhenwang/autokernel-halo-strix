"""Phase 1: Compute sentence embeddings for sampled documents.

Runs on CPU — leaves GPU free for other work.
Outputs float16 embeddings for FAISS clustering in Phase 2.

Usage:
    python scripts/datamix/phase1_embed.py
    python scripts/datamix/phase1_embed.py --model sentence-transformers/all-MiniLM-L6-v2
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.datamix import load_pipeline_state, save_pipeline_state, phase_dir, check_dependency


def load_docs_from_shards(shard_dir: Path):
    """Load all documents from JSONL shards, return (doc_ids, texts)."""
    doc_ids = []
    texts = []
    for shard_path in sorted(shard_dir.glob("shard_*.jsonl")):
        with open(shard_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                doc_ids.append(obj["doc_id"])
                texts.append(obj["text"][:512])
    return doc_ids, texts


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Embed documents with sentence transformer")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--state-dir", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    base = Path(args.state_dir) if args.state_dir else None
    state = load_pipeline_state(base)
    check_dependency(state, "phase1")

    out_dir = phase_dir("phase1", base)

    if args.resume and state.get("phase1", {}).get("status") == "complete":
        print(f"Phase 1 already complete")
        return

    sample_dir = phase_dir("phase0", base)
    print(f"Loading docs from {sample_dir}...")
    doc_ids, texts = load_docs_from_shards(sample_dir)
    print(f"Loaded {len(texts):,} documents")

    from sentence_transformers import SentenceTransformer

    print(f"Loading model: {args.model}")
    model = SentenceTransformer(args.model, device="cpu")

    print(f"Embedding {len(texts):,} docs (batch_size={args.batch_size})...")
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings_f16 = embeddings.astype(np.float16)
    emb_path = out_dir / "embeddings.npy"
    np.save(emb_path, embeddings_f16)

    idx_path = out_dir / "doc_index.json"
    with open(idx_path, "w") as f:
        json.dump(doc_ids, f)

    state["phase1"] = {
        "status": "complete",
        "total_embedded": len(texts),
        "embedding_dim": embeddings.shape[1],
        "model": args.model,
    }
    save_pipeline_state(state, base)
    print(f"Phase 1 complete: {len(texts):,} embeddings ({embeddings.shape[1]}d) saved to {emb_path}")


if __name__ == "__main__":
    main()
