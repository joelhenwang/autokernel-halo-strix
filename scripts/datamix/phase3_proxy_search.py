"""Phase 3: CLIMB proxy model search for optimal mixture weights.

Trains a small proxy model on different mixture weights, measures validation
loss, and uses LightGBM surrogate to find the optimal mixture. Model-agnostic:
any (model_path, class_name) pair works.

Requires GPU. Run on remote machine.

Usage:
    python scripts/datamix/phase3_proxy_search.py \
        --model models/chimera_halo.py --class-name ChimeraHaloMini \
        --val-dataset datasets/babylm-strict-small
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.datamix import load_pipeline_state, save_pipeline_state, phase_dir, check_dependency


def tokenize_cluster_docs(sample_dir: Path, assignments: np.ndarray,
                          doc_index: list, n_clusters: int,
                          out_dir: Path) -> dict:
    """Pre-tokenize documents per cluster into .bin files."""
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    eos = enc.n_vocab - 1

    all_texts = {}
    for shard_path in sorted(sample_dir.glob("shard_*.jsonl")):
        with open(shard_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                all_texts[obj["doc_id"]] = obj["text"]

    cluster_info = {}
    for k in range(n_clusters):
        mask = assignments == k
        indices = np.where(mask)[0]
        tokens = []
        for idx in indices:
            did = doc_index[idx]
            text = all_texts.get(did, "")
            if text:
                tokens.extend(enc.encode_ordinary(text))
                tokens.append(eos)

        if tokens:
            arr = np.array(tokens, dtype=np.uint16)
            bin_path = out_dir / f"cluster_{k}.bin"
            arr.tofile(str(bin_path))
            cluster_info[k] = {"path": str(bin_path), "n_tokens": len(tokens)}
            print(f"  Cluster {k}: {len(tokens):,} tokens")

    return cluster_info


def build_mixed_dataset(cluster_info: dict, weights: np.ndarray,
                        target_tokens: int, block_size: int):
    """Sample tokens from clusters proportional to weights, return Dataset."""
    from halo_training.data import BabyLMDataset
    import tempfile

    all_tokens = []
    for k, w in enumerate(weights):
        if w < 1e-6 or k not in cluster_info:
            continue
        n_want = int(target_tokens * w)
        arr = np.fromfile(cluster_info[k]["path"], dtype=np.uint16).astype(np.int64)
        if len(arr) == 0:
            continue
        if len(arr) < n_want:
            repeats = (n_want // len(arr)) + 1
            arr = np.tile(arr, repeats)[:n_want]
        else:
            start = np.random.randint(0, max(len(arr) - n_want, 1))
            arr = arr[start:start + n_want]
        all_tokens.append(arr)

    if not all_tokens:
        return None

    combined = np.concatenate(all_tokens).astype(np.uint16)
    np.random.shuffle(combined.reshape(-1, min(block_size, len(combined) // 10 or 1))[:-1])

    tmp = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
    combined.tofile(tmp.name)
    tmp.close()
    return BabyLMDataset(root=tmp.name, block_size=block_size)


def train_proxy_once(model_path: str, class_name: str, dataset,
                     val_dataset, n_steps: int, lr: float,
                     block_size: int, device: str) -> float:
    """Train proxy model for n_steps, return validation CE loss."""
    from halo_training.cli import load_model_from_file
    from torch.utils.data import DataLoader

    model = load_model_from_file(model_path, class_name)
    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)
    loader_iter = iter(loader)

    for step in range(n_steps):
        try:
            x, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            x, y = next(loader_iter)
        x, y = x.to(device), y.to(device)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1)
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

    model.eval()
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, drop_last=True)
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), y.reshape(-1)
                )
            total_loss += loss.item()
            n_batches += 1
            if n_batches >= 50:
                break

    del model, optimizer
    torch.cuda.empty_cache()
    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="Phase 3: CLIMB proxy model search")
    parser.add_argument("--model", required=True, help="Path to model .py file")
    parser.add_argument("--class-name", required=True, help="Model class name")
    parser.add_argument("--val-dataset", required=True, help="Validation dataset path")
    parser.add_argument("--n-rounds", type=int, default=3)
    parser.add_argument("--trials-per-round", default="20,10,5", help="Comma-separated trials per round")
    parser.add_argument("--proxy-steps", type=int, default=500)
    parser.add_argument("--proxy-tokens", type=int, default=500_000)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--state-dir", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    base = Path(args.state_dir) if args.state_dir else None
    state = load_pipeline_state(base)
    check_dependency(state, "phase3")

    out_dir = phase_dir("phase3", base)
    trials_per_round = [int(x) for x in args.trials_per_round.split(",")]

    sample_dir = phase_dir("phase0", base)
    cluster_dir = phase_dir("phase2", base)

    assignments = np.load(cluster_dir / "assignments.npy")
    doc_index = json.loads((phase_dir("phase1", base) / "doc_index.json").read_text())
    n_clusters = state["phase2"]["n_clusters"]

    cluster_bins_dir = out_dir / "cluster_bins"
    cluster_bins_dir.mkdir(exist_ok=True)
    info_path = cluster_bins_dir / "cluster_info.json"
    if info_path.exists():
        cluster_info = {int(k): v for k, v in json.loads(info_path.read_text()).items()}
        print(f"Using existing per-cluster .bin files")
    else:
        print("Pre-tokenizing per-cluster data...")
        cluster_info = tokenize_cluster_docs(
            sample_dir, assignments, doc_index, n_clusters, cluster_bins_dir
        )
        info_path.write_text(json.dumps({str(k): v for k, v in cluster_info.items()}, indent=2))

    from halo_training.data import BabyLMDataset
    val_ds = BabyLMDataset(root=args.val_dataset, block_size=args.block_size)

    results_path = out_dir / "trial_results.jsonl"
    existing_results = []
    if args.resume and results_path.exists():
        with open(results_path) as f:
            existing_results = [json.loads(l) for l in f]
        print(f"Resuming: {len(existing_results)} trials already completed")

    all_results = list(existing_results)

    for round_idx, n_trials in enumerate(trials_per_round):
        print(f"\n=== Round {round_idx + 1}/{len(trials_per_round)}: {n_trials} trials ===")

        if round_idx == 0:
            candidates = [np.random.dirichlet(np.ones(n_clusters)) for _ in range(n_trials)]
        else:
            try:
                import lightgbm as lgb
                X = np.array([r["weights"] for r in all_results])
                y = np.array([r["val_loss"] for r in all_results])
                gbm = lgb.LGBMRegressor(n_estimators=50, verbose=-1)
                gbm.fit(X, y)

                best_pred = float("inf")
                candidates = []
                for _ in range(n_trials * 50):
                    w = np.random.dirichlet(np.ones(n_clusters))
                    pred = gbm.predict(w.reshape(1, -1))[0]
                    candidates.append((pred, w))
                candidates.sort(key=lambda x: x[0])
                candidates = [w for _, w in candidates[:n_trials]]
                print(f"  Surrogate predicted best: {candidates[0].min():.4f}-{candidates[0].max():.4f}")
            except ImportError:
                print("  LightGBM not installed, using Dirichlet sampling")
                candidates = [np.random.dirichlet(np.ones(n_clusters)) for _ in range(n_trials)]

        for trial_idx, weights in enumerate(candidates):
            trial_id = len(all_results)
            print(f"  Trial {trial_id}: weights=[{', '.join(f'{w:.3f}' for w in weights)}]")

            ds = build_mixed_dataset(cluster_info, weights, args.proxy_tokens, args.block_size)
            if ds is None:
                print(f"    Skipped (no data)")
                continue

            t0 = time.time()
            val_loss = train_proxy_once(
                args.model, args.class_name, ds, val_ds,
                args.proxy_steps, args.lr, args.block_size, args.device,
            )
            elapsed = time.time() - t0
            print(f"    val_loss={val_loss:.4f} ({elapsed:.1f}s)")

            result = {
                "trial_id": trial_id,
                "round": round_idx,
                "weights": weights.tolist(),
                "val_loss": val_loss,
                "elapsed_s": elapsed,
            }
            all_results.append(result)
            with open(results_path, "a") as f:
                f.write(json.dumps(result) + "\n")

    best = min(all_results, key=lambda r: r["val_loss"])
    mixture_config = {
        "weights": {str(k): w for k, w in enumerate(best["weights"])},
        "val_loss": best["val_loss"],
        "trial_id": best["trial_id"],
        "n_clusters": n_clusters,
        "proxy_model": args.model,
        "proxy_class": args.class_name,
    }

    config_path = out_dir / "mixture_config.json"
    config_path.write_text(json.dumps(mixture_config, indent=2))

    state["phase3"] = {"status": "complete", "best_val_loss": best["val_loss"]}
    save_pipeline_state(state, base)

    print(f"\nPhase 3 complete. Best mixture (val_loss={best['val_loss']:.4f}):")
    for k, w in enumerate(best["weights"]):
        if w > 0.01:
            print(f"  Cluster {k}: {w:.3f}")


if __name__ == "__main__":
    main()
