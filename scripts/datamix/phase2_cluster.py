"""Phase 2: K-means clustering of document embeddings.

Each cluster becomes a "domain" for mixture optimization in Phase 3.
Outputs cluster assignments, centroids, and per-cluster statistics.

Usage:
    python scripts/datamix/phase2_cluster.py
    python scripts/datamix/phase2_cluster.py --n-clusters 16
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.datamix import load_pipeline_state, save_pipeline_state, phase_dir, check_dependency


def main():
    parser = argparse.ArgumentParser(description="Phase 2: K-means clustering")
    parser.add_argument("--n-clusters", type=int, default=16)
    parser.add_argument("--n-iter", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--state-dir", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    base = Path(args.state_dir) if args.state_dir else None
    state = load_pipeline_state(base)
    check_dependency(state, "phase2")

    out_dir = phase_dir("phase2", base)

    if args.resume and state.get("phase2", {}).get("status") == "complete":
        print("Phase 2 already complete")
        return

    emb_dir = phase_dir("phase1", base)
    emb_path = emb_dir / "embeddings.npy"
    print(f"Loading embeddings from {emb_path}...")
    embeddings = np.load(emb_path).astype(np.float32)
    n_docs, dim = embeddings.shape
    print(f"Loaded {n_docs:,} embeddings ({dim}d)")

    import faiss

    print(f"Running K-means (K={args.n_clusters}, niter={args.n_iter})...")
    kmeans = faiss.Kmeans(dim, args.n_clusters, niter=args.n_iter, seed=args.seed, gpu=False)
    kmeans.train(embeddings)

    _, assignments = kmeans.index.search(embeddings, 1)
    assignments = assignments.flatten().astype(np.int32)

    assign_path = out_dir / "assignments.npy"
    np.save(assign_path, assignments)

    centroid_path = out_dir / "centroids.npy"
    np.save(centroid_path, kmeans.centroids)

    idx_path = phase_dir("phase1", base) / "doc_index.json"
    doc_ids = json.loads(idx_path.read_text()) if idx_path.exists() else None

    sample_dir = phase_dir("phase0", base)
    all_texts = {}
    for shard_path in sorted(sample_dir.glob("shard_*.jsonl")):
        with open(shard_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                all_texts[obj["doc_id"]] = obj["text"][:200]

    cluster_stats = {}
    for k in range(args.n_clusters):
        mask = assignments == k
        count = int(mask.sum())
        cluster_embs = embeddings[mask]
        centroid = kmeans.centroids[k]
        dists = np.linalg.norm(cluster_embs - centroid, axis=1)
        nearest_idx = np.where(mask)[0][np.argsort(dists)[:5]]

        representatives = []
        if doc_ids:
            for idx in nearest_idx:
                did = doc_ids[idx]
                representatives.append(all_texts.get(did, "")[:150])

        cluster_stats[str(k)] = {
            "count": count,
            "fraction": round(count / n_docs, 4),
            "representatives": representatives,
        }

    stats_path = out_dir / "cluster_stats.json"
    with open(stats_path, "w") as f:
        json.dump(cluster_stats, f, indent=2, ensure_ascii=False)

    print(f"\nCluster distribution:")
    for k in range(args.n_clusters):
        info = cluster_stats[str(k)]
        rep = info["representatives"][0][:80] if info["representatives"] else ""
        print(f"  Cluster {k:2d}: {info['count']:>8,} docs ({info['fraction']*100:5.1f}%)  {rep}...")

    state["phase2"] = {
        "status": "complete",
        "n_clusters": args.n_clusters,
        "n_docs": n_docs,
    }
    save_pipeline_state(state, base)
    print(f"\nPhase 2 complete: {args.n_clusters} clusters assigned")


if __name__ == "__main__":
    main()
