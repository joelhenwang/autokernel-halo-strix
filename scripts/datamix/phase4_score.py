"""Phase 4: Self-Improving quality scoring via LLM API + local classifier.

Scores a small sample per cluster via API, trains a logistic regression
classifier on (embedding → score) pairs, then predicts scores for all docs.

Usage:
    python scripts/datamix/phase4_score.py --api-provider anthropic
    python scripts/datamix/phase4_score.py --api-provider openai --api-model gpt-4o-mini
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.datamix import load_pipeline_state, save_pipeline_state, phase_dir, check_dependency

SCORING_PROMPT = """Rate the quality of this text for language model pretraining.

Criteria:
- 5: Well-written, informative, factually dense, educational
- 4: Good quality, mostly coherent, some noise
- 3: Average web text, acceptable
- 2: Low quality, repetitive, boilerplate, or poorly written
- 1: Spam, gibberish, harmful, or non-content

Text:
{text}

Respond with ONLY a single number (1-5):"""


async def score_batch_anthropic(texts: list, model: str, api_key: str) -> list:
    """Score a batch of texts using Anthropic API."""
    import anthropic
    client = anthropic.AsyncAnthropic(api_key=api_key)

    async def score_one(text):
        try:
            resp = await client.messages.create(
                model=model,
                max_tokens=4,
                messages=[{"role": "user", "content": SCORING_PROMPT.format(text=text[:1000])}],
            )
            content = resp.content[0].text.strip()
            score = int(content[0]) if content and content[0].isdigit() else 3
            return min(max(score, 1), 5)
        except Exception:
            return 3

    tasks = [score_one(t) for t in texts]
    return await asyncio.gather(*tasks)


async def score_batch_openai(texts: list, model: str, api_key: str) -> list:
    """Score a batch of texts using OpenAI API."""
    import openai
    client = openai.AsyncOpenAI(api_key=api_key)

    async def score_one(text):
        try:
            resp = await client.chat.completions.create(
                model=model,
                max_tokens=4,
                messages=[{"role": "user", "content": SCORING_PROMPT.format(text=text[:1000])}],
            )
            content = resp.choices[0].message.content.strip()
            score = int(content[0]) if content and content[0].isdigit() else 3
            return min(max(score, 1), 5)
        except Exception:
            return 3

    tasks = [score_one(t) for t in texts]
    return await asyncio.gather(*tasks)


def main():
    parser = argparse.ArgumentParser(description="Phase 4: Quality scoring via API + classifier")
    parser.add_argument("--api-provider", choices=["anthropic", "openai"], default="anthropic")
    parser.add_argument("--api-model", default=None,
                        help="Default: claude-3-5-haiku-20241022 (anthropic) or gpt-4o-mini (openai)")
    parser.add_argument("--api-key", default=None, help="API key (or set env var)")
    parser.add_argument("--samples-per-cluster", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=50, help="Concurrent API requests")
    parser.add_argument("--state-dir", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.api_model is None:
        args.api_model = {
            "anthropic": "claude-3-5-haiku-20241022",
            "openai": "gpt-4o-mini",
        }[args.api_provider]

    import os
    api_key = args.api_key or os.environ.get(
        "ANTHROPIC_API_KEY" if args.api_provider == "anthropic" else "OPENAI_API_KEY"
    )
    if not api_key:
        env_var = "ANTHROPIC_API_KEY" if args.api_provider == "anthropic" else "OPENAI_API_KEY"
        print(f"Error: Set --api-key or {env_var} environment variable")
        sys.exit(1)

    base = Path(args.state_dir) if args.state_dir else None
    state = load_pipeline_state(base)
    check_dependency(state, "phase4")

    out_dir = phase_dir("phase4", base)

    if args.resume and state.get("phase4", {}).get("status") == "complete":
        print("Phase 4 already complete")
        return

    assignments = np.load(phase_dir("phase2", base) / "assignments.npy")
    embeddings = np.load(phase_dir("phase1", base) / "embeddings.npy").astype(np.float32)
    doc_ids = json.loads((phase_dir("phase1", base) / "doc_index.json").read_text())
    n_clusters = state["phase2"]["n_clusters"]

    sample_dir = phase_dir("phase0", base)
    all_texts = {}
    for shard_path in sorted(sample_dir.glob("shard_*.jsonl")):
        with open(shard_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                all_texts[obj["doc_id"]] = obj["text"]

    scored_path = out_dir / "api_scores.jsonl"
    scored_samples = []
    if args.resume and scored_path.exists():
        with open(scored_path) as f:
            scored_samples = [json.loads(l) for l in f]
        print(f"Resuming: {len(scored_samples)} docs already scored")

    scored_ids = {s["doc_id"] for s in scored_samples}

    print(f"Scoring {args.samples_per_cluster} docs per cluster ({n_clusters} clusters)...")
    score_fn = score_batch_anthropic if args.api_provider == "anthropic" else score_batch_openai

    for k in range(n_clusters):
        mask = assignments == k
        indices = np.where(mask)[0]
        np.random.shuffle(indices)

        already = sum(1 for s in scored_samples if s.get("cluster") == k)
        n_need = max(0, args.samples_per_cluster - already)
        if n_need == 0:
            continue

        batch_texts = []
        batch_indices = []
        for idx in indices:
            did = doc_ids[idx]
            if did in scored_ids:
                continue
            text = all_texts.get(did, "")
            if not text:
                continue
            batch_texts.append(text)
            batch_indices.append(idx)
            if len(batch_texts) >= n_need:
                break

        print(f"  Cluster {k}: scoring {len(batch_texts)} docs...")
        for i in range(0, len(batch_texts), args.batch_size):
            chunk_texts = batch_texts[i:i + args.batch_size]
            chunk_indices = batch_indices[i:i + args.batch_size]
            scores = asyncio.run(score_fn(chunk_texts, args.api_model, api_key))

            with open(scored_path, "a") as f:
                for idx, score in zip(chunk_indices, scores):
                    entry = {
                        "doc_id": doc_ids[idx],
                        "cluster": k,
                        "embedding_idx": int(idx),
                        "api_score": score,
                    }
                    scored_samples.append(entry)
                    f.write(json.dumps(entry) + "\n")

            time.sleep(0.1)

    print(f"\nAPI scoring complete: {len(scored_samples)} total")

    print("Training quality classifier...")
    from sklearn.linear_model import LogisticRegression

    X_train = np.array([embeddings[s["embedding_idx"]] for s in scored_samples])
    y_train = np.array([1 if s["api_score"] >= 3 else 0 for s in scored_samples])

    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train, y_train)

    from sklearn.metrics import accuracy_score
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    print(f"  Classifier training accuracy: {train_acc:.3f}")

    print("Predicting quality for all documents...")
    all_probs = clf.predict_proba(embeddings)[:, 1]

    scores_path = out_dir / "quality_scores.jsonl"
    with open(scores_path, "w") as f:
        for idx, (did, prob) in enumerate(zip(doc_ids, all_probs)):
            entry = {
                "doc_id": did,
                "cluster": int(assignments[idx]),
                "quality_prob": round(float(prob), 4),
            }
            f.write(json.dumps(entry) + "\n")

    import pickle
    clf_path = out_dir / "quality_classifier.pkl"
    with open(clf_path, "wb") as f:
        pickle.dump(clf, f)

    mean_quality = {}
    for k in range(n_clusters):
        mask = assignments == k
        mean_quality[k] = float(all_probs[mask].mean())
        print(f"  Cluster {k}: mean quality={mean_quality[k]:.3f}")

    state["phase4"] = {
        "status": "complete",
        "docs_scored_api": len(scored_samples),
        "classifier_accuracy": round(train_acc, 3),
        "mean_quality_per_cluster": {str(k): v for k, v in mean_quality.items()},
    }
    save_pipeline_state(state, base)
    print(f"\nPhase 4 complete. Classifier saved to {clf_path}")


if __name__ == "__main__":
    main()
