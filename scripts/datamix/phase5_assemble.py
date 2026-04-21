"""Phase 5: Assemble final pre-mixed, quality-filtered .bin dataset.

Combines Phase 3 mixture weights + Phase 4 quality scores to produce
a tokenized .bin file ready for BabyLMDataset.

Two modes:
  - Sample mode: uses Phase 0 cached docs (fast, for validation)
  - Stream mode: re-streams full HF dataset (slow, for production)

Usage:
    python scripts/datamix/phase5_assemble.py --target-tokens 500000000
    python scripts/datamix/phase5_assemble.py --source-dataset allenai/dolma --stream-full --target-tokens 10000000000
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.datamix import load_pipeline_state, save_pipeline_state, phase_dir, check_dependency


def assemble_from_sample(base: Path, target_tokens: int, quality_threshold: float):
    """Assemble from Phase 0 cached documents (fast)."""
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    eos = enc.n_vocab - 1

    mixture_config = json.loads(
        (phase_dir("phase3", base) / "mixture_config.json").read_text()
    )
    weights = {int(k): v for k, v in mixture_config["weights"].items()}

    scores = {}
    scores_path = phase_dir("phase4", base) / "quality_scores.jsonl"
    if scores_path.exists():
        with open(scores_path) as f:
            for line in f:
                obj = json.loads(line)
                scores[obj["doc_id"]] = obj["quality_prob"]

    sample_dir = phase_dir("phase0", base)
    assignments = np.load(phase_dir("phase2", base) / "assignments.npy")
    doc_index = json.loads((phase_dir("phase1", base) / "doc_index.json").read_text())

    all_texts = {}
    for shard_path in sorted(sample_dir.glob("shard_*.jsonl")):
        with open(shard_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                all_texts[obj["doc_id"]] = obj["text"]

    cluster_docs = {}
    for idx, (did, cluster) in enumerate(zip(doc_index, assignments)):
        quality = scores.get(did, 0.5)
        if quality < quality_threshold:
            continue
        k = int(cluster)
        if k not in cluster_docs:
            cluster_docs[k] = []
        cluster_docs[k].append(did)

    n_clusters = max(weights.keys()) + 1
    all_tokens = []
    for k in range(n_clusters):
        w = weights.get(k, 0.0)
        if w < 1e-6:
            continue
        docs = cluster_docs.get(k, [])
        if not docs:
            continue
        n_want = int(target_tokens * w)
        np.random.shuffle(docs)

        tokens = []
        for did in docs:
            text = all_texts.get(did, "")
            if text:
                tokens.extend(enc.encode_ordinary(text))
                tokens.append(eos)
            if len(tokens) >= n_want:
                break
        tokens = tokens[:n_want]
        all_tokens.extend(tokens)
        print(f"  Cluster {k} (w={w:.3f}): {len(tokens):,} tokens from {min(len(docs), len(tokens))} docs")

    combined = np.array(all_tokens, dtype=np.uint16)
    np.random.shuffle(combined.reshape(-1, 1024)[:len(combined) // 1024])
    return combined


def assemble_from_stream(base: Path, source_dataset: str, split: str,
                         text_field: str, target_tokens: int,
                         quality_threshold: float):
    """Re-stream full dataset, assign clusters on-the-fly, filter by quality."""
    import tiktoken
    import pickle

    enc = tiktoken.get_encoding("gpt2")
    eos = enc.n_vocab - 1

    mixture_config = json.loads(
        (phase_dir("phase3", base) / "mixture_config.json").read_text()
    )
    weights = {int(k): v for k, v in mixture_config["weights"].items()}
    n_clusters = mixture_config["n_clusters"]

    centroids = np.load(phase_dir("phase2", base) / "centroids.npy")

    clf_path = phase_dir("phase4", base) / "quality_classifier.pkl"
    clf = None
    if clf_path.exists():
        with open(clf_path, "rb") as f:
            clf = pickle.load(f)

    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

    per_cluster_budget = {k: int(target_tokens * weights.get(k, 0)) for k in range(n_clusters)}
    per_cluster_collected = {k: 0 for k in range(n_clusters)}

    from datasets import load_dataset
    print(f"Streaming {source_dataset}...")
    ds = load_dataset(source_dataset, split=split, streaming=True)

    all_tokens = []
    batch_texts = []
    batch_raw = []
    batch_size = 64

    for example in ds:
        text = example.get(text_field, "")
        if not text or len(text) < 50:
            continue
        batch_texts.append(text[:512])
        batch_raw.append(text)

        if len(batch_texts) >= batch_size:
            embs = embedder.encode(batch_texts, convert_to_numpy=True, normalize_embeddings=True)
            dists = np.linalg.norm(embs[:, None, :] - centroids[None, :, :], axis=2)
            cluster_assigns = dists.argmin(axis=1)

            if clf is not None:
                quality_probs = clf.predict_proba(embs.astype(np.float32))[:, 1]
            else:
                quality_probs = np.ones(len(embs))

            for i, (raw, k, qp) in enumerate(zip(batch_raw, cluster_assigns, quality_probs)):
                k = int(k)
                if qp < quality_threshold:
                    continue
                if per_cluster_collected[k] >= per_cluster_budget[k]:
                    continue
                tokens = enc.encode_ordinary(raw)
                tokens.append(eos)
                all_tokens.extend(tokens)
                per_cluster_collected[k] += len(tokens)

            batch_texts = []
            batch_raw = []

            total_collected = sum(per_cluster_collected.values())
            if total_collected % 10_000_000 < batch_size * 500:
                print(f"  {total_collected:,}/{target_tokens:,} tokens collected")
            if total_collected >= target_tokens:
                break

    return np.array(all_tokens, dtype=np.uint16)


def main():
    parser = argparse.ArgumentParser(description="Phase 5: Assemble final dataset")
    parser.add_argument("--target-tokens", type=int, default=500_000_000)
    parser.add_argument("--quality-threshold", type=float, default=0.5)
    parser.add_argument("--source-dataset", type=str, default=None,
                        help="HF dataset for stream mode (omit for sample mode)")
    parser.add_argument("--stream-full", action="store_true")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--state-dir", type=str, default=None)
    args = parser.parse_args()

    base = Path(args.state_dir) if args.state_dir else None
    state = load_pipeline_state(base)
    check_dependency(state, "phase5")

    out_dir = phase_dir("phase5", base)

    if args.stream_full and args.source_dataset:
        print("Assembling from full stream...")
        tokens = assemble_from_stream(
            base, args.source_dataset, args.split, args.text_field,
            args.target_tokens, args.quality_threshold,
        )
    else:
        print("Assembling from sampled documents...")
        tokens = assemble_from_sample(base, args.target_tokens, args.quality_threshold)

    out_path = out_dir / "train.bin"
    tokens.tofile(str(out_path))

    state["phase5"] = {
        "status": "complete",
        "n_tokens": len(tokens),
        "output": str(out_path),
        "quality_threshold": args.quality_threshold,
        "stream_mode": args.stream_full,
    }
    save_pipeline_state(state, base)
    print(f"\nPhase 5 complete: {len(tokens):,} tokens written to {out_path}")
    print(f"  File size: {out_path.stat().st_size / 1e9:.2f} GB")
    print(f"\nUse with: python -m halo_training --dataset {out_path}")


if __name__ == "__main__":
    main()
