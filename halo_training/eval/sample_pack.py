"""Sample-pack regression evaluator (Sprint 2 Phase 4).

Runs deterministic generation against a frozen 20-prompt set (v1) and
produces a stable hash + optionally diff against a prior checkpoint.

Design decisions
----------------
- Frozen prompt file: ``evals/sample_pack_v1.txt`` (locked; bump to v2 if we
  need new prompts — never mutate v1).
- Deterministic generation: one fixed seed, modified per-prompt so samples
  are independent but reproducible.
- Fixed sampling config: ``temperature=0.7, top_p=0.95, top_k=40, rep_pen=1.1``
  — a middle-of-the-road setting safe across undertrained / SFT / aligned
  checkpoints. Not the ablation-winning config (which varies per checkpoint);
  we want a STABLE reference frame so diffs reflect the model, not the
  sampling parameters.
- Hash: sha256 of the concatenated outputs (prompt + continuation per row,
  joined by a sentinel). 16 hex-char prefix emitted.
- Prior checkpoint auto-detection: most recent sibling ``step_*.pt`` with
  a smaller step number in the same directory.
- Diff format: for the first 3 prompts whose outputs differ, emit a short
  preview (first 80 chars of current vs prior) in the scorecard JSON.

Output shape::

    {
        "prompts_file": "evals/sample_pack_v1.txt",
        "num_prompts": 20,
        "sampling_config": {...},
        "output_hash": "sha256:a3b2c4d5e6f70123",
        "diff_vs_prior": None | {
            "prior_checkpoint": "...",
            "prior_hash": "sha256:...",
            "diff_count": 7,
            "preview": [
                {"prompt_idx": 0, "current": "...", "prior": "..."},
                ...
            ]
        }
    }

Sampling config
---------------
Using a fixed config per scorecard (not per-checkpoint winner) trades some
quality-signal fidelity for regression stability: if the model gets better,
we see it; if the sampling config happens to shift the winning temperature,
we don't want that swamping the model diff. The ablation-based winner
config is captured separately in the `sampling` evaluator.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

#: Fixed sampling config for sample-pack regression. DO NOT MUTATE without
#: bumping the prompts file version (v1 -> v2) so diffs remain meaningful.
SAMPLE_PACK_SAMPLING_CONFIG = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "repetition_penalty": 1.1,
    "max_tokens": 100,
    "base_seed": 777,
}


def run(model, tokenizer, validation_splits, args=None) -> Optional[Dict[str, Any]]:
    """Generate 20 deterministic samples; emit hash + optional diff."""
    prompts_file = (
        getattr(args, "prompts_file", None)
        if args is not None
        else None
    ) or "evals/sample_pack_v1.txt"

    if not os.path.exists(prompts_file):
        print(f"  [sample_pack] prompts file not found: {prompts_file}; skipping")
        return None

    prompts = _load_prompts(prompts_file)
    if not prompts:
        print(f"  [sample_pack] {prompts_file} contains no prompts; skipping")
        return None

    # Generate samples under the frozen config
    cfg = SAMPLE_PACK_SAMPLING_CONFIG
    samples = _generate_pack(model, tokenizer, prompts, cfg)

    # Deterministic hash over the prompt + continuation rows
    combined = _canonicalize(samples)
    output_hash = "sha256:" + hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]

    # Diff against prior checkpoint if available
    diff_info: Optional[Dict[str, Any]] = None
    prior_ckpt = _resolve_prior_checkpoint(args)
    if prior_ckpt is not None:
        try:
            prior_hash, prior_samples = _load_or_generate_prior(
                prior_ckpt, prompts, cfg, tokenizer,
                current_model_file=(args.model if args else None),
                current_class_name=(args.class_name if args else None),
            )
            if prior_hash and prior_hash != output_hash:
                diff_info = _build_diff(samples, prior_samples, prior_ckpt, prior_hash)
            elif prior_hash == output_hash:
                diff_info = {
                    "prior_checkpoint": prior_ckpt,
                    "prior_hash": prior_hash,
                    "diff_count": 0,
                    "preview": [],
                }
        except Exception as exc:  # noqa: BLE001
            print(f"  [sample_pack] prior diff failed: {type(exc).__name__}: {exc}")
            diff_info = {
                "prior_checkpoint": prior_ckpt,
                "error": f"{type(exc).__name__}: {exc}",
            }

    return {
        "prompts_file": prompts_file,
        "num_prompts": len(prompts),
        "sampling_config": {k: v for k, v in cfg.items()},
        "output_hash": output_hash,
        "samples_preview": [s[:160] for s in samples[:3]],  # first 3 for sanity
        "diff_vs_prior": diff_info,
    }


# ---------------------------------------------------------------------------
# Prompts + generation
# ---------------------------------------------------------------------------


def _load_prompts(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _canonicalize(samples: List[str]) -> str:
    """Stable serialisation for hashing: prompt-sentinel-continuation rows."""
    return "\n<<<END>>>\n".join(samples)


@torch.no_grad()
def _generate_pack(model, tokenizer, prompts, cfg) -> List[str]:
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder

    bl_decoder = ByteLevelDecoder()
    vocab_size = tokenizer.get_vocab_size()
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    if eos_id is None:
        eos_id = 0

    device = next(model.parameters()).device
    samples: List[str] = []
    for i, prompt in enumerate(prompts):
        torch.manual_seed(cfg["base_seed"] + i)
        enc = tokenizer.encode(prompt)
        prompt_ids = torch.tensor([enc.ids], dtype=torch.long, device=device)
        out = _generate_tokens(
            model, prompt_ids,
            max_tokens=cfg["max_tokens"],
            temperature=cfg["temperature"],
            top_k=cfg["top_k"],
            top_p=cfg["top_p"],
            repetition_penalty=cfg["repetition_penalty"],
            eos_token=eos_id,
            vocab_size=vocab_size,
        )
        toks = [tokenizer.id_to_token(tid) for tid in out[0].tolist()]
        samples.append(bl_decoder.decode(toks))
    return samples


@torch.no_grad()
def _generate_tokens(model, prompt_ids, *, max_tokens, temperature, top_k, top_p,
                     repetition_penalty, eos_token, vocab_size, max_seq_len=2048):
    """Same autoregressive decode logic as ablate_odin_flat_sampling.generate.

    Duplicated locally so the sample-pack evaluator has no import-order
    dependency on the ablation harness.
    """
    ids = prompt_ids
    for _ in range(max_tokens):
        context = ids[:, -max_seq_len:]
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(context)
        if isinstance(logits, tuple):
            logits = logits[0]
        logits = logits[:, -1, :].float()

        if repetition_penalty != 1.0:
            seen = set(ids[0].tolist())
            for tid in seen:
                if 0 <= tid < vocab_size:
                    if logits[0, tid] > 0:
                        logits[0, tid] /= repetition_penalty
                    else:
                        logits[0, tid] *= repetition_penalty

        if temperature > 0:
            logits = logits / temperature

        if top_k and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = torch.where(logits < v[:, [-1]],
                                 torch.full_like(logits, float("-inf")), logits)

        if top_p and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_mask = cumulative > top_p
            sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
            sorted_mask[:, 0] = False
            mask = sorted_mask.scatter(1, sorted_indices, sorted_mask)
            logits = logits.masked_fill(mask, float("-inf"))

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_token], dim=1)
        if next_token.item() == eos_token:
            break
    return ids


# ---------------------------------------------------------------------------
# Prior-checkpoint resolution + diffing
# ---------------------------------------------------------------------------


_STEP_RE = re.compile(r"step_(\d+)(?:_[a-zA-Z]+)?\.pt$")


def _resolve_prior_checkpoint(args) -> Optional[str]:
    """If --prior-checkpoint given, use it. Otherwise auto-detect sibling."""
    if args is None:
        return None
    explicit = getattr(args, "prior_checkpoint", None)
    if explicit:
        return explicit
    current = getattr(args, "checkpoint", None)
    if not current:
        return None
    return _auto_detect_prior(current)


def _auto_detect_prior(checkpoint_path: str) -> Optional[str]:
    """Find the most recent sibling step_*.pt with a smaller step number."""
    p = Path(checkpoint_path)
    m = _STEP_RE.search(p.name)
    if not m:
        return None
    current_step = int(m.group(1))
    best: Optional[Tuple[int, Path]] = None
    for sibling in p.parent.glob("step_*.pt"):
        sm = _STEP_RE.search(sibling.name)
        if not sm:
            continue
        step = int(sm.group(1))
        if step < current_step and (best is None or step > best[0]):
            best = (step, sibling)
    return str(best[1]) if best else None


def _load_or_generate_prior(prior_ckpt, prompts, cfg, tokenizer,
                            *, current_model_file, current_class_name):
    """Load prior samples from cache if available; otherwise regenerate.

    Cache path: same directory as the prior ``.pt``, named
    ``<prior>.sample_pack_v1.txt`` with one sample per line joined by the
    ``<<<END>>>`` sentinel to preserve newlines inside samples.
    """
    cache_path = Path(prior_ckpt).with_suffix(".sample_pack_v1.cache")
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            prior_samples = f.read().split("\n<<<END>>>\n")
        if len(prior_samples) == len(prompts):
            prior_hash = (
                "sha256:" + hashlib.sha256(_canonicalize(prior_samples).encode("utf-8"))
                .hexdigest()[:16]
            )
            return prior_hash, prior_samples

    # Regenerate from prior checkpoint. Expensive (second model load), but
    # happens at most once per prior checkpoint; cached thereafter.
    if not current_model_file or not current_class_name:
        print(f"  [sample_pack] cannot regenerate prior without model info")
        return None, None

    from halo_training.eval.common import load_checkpoint, load_model

    print(f"  [sample_pack] regenerating prior samples from {prior_ckpt} ...")
    prior_model = load_model(current_model_file, current_class_name).to("cuda").eval()
    prior_model, _step = load_checkpoint(prior_model, prior_ckpt)
    prior_model.half()

    prior_samples = _generate_pack(prior_model, tokenizer, prompts, cfg)

    # Free memory: delete the prior model before returning
    del prior_model
    torch.cuda.empty_cache()

    # Cache for next time
    try:
        cache_path.write_text(
            "\n<<<END>>>\n".join(prior_samples), encoding="utf-8"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  [sample_pack] warning: could not write cache {cache_path}: {exc}")

    prior_hash = (
        "sha256:" + hashlib.sha256(_canonicalize(prior_samples).encode("utf-8"))
        .hexdigest()[:16]
    )
    return prior_hash, prior_samples


def _build_diff(current: List[str], prior: List[str],
                prior_ckpt: str, prior_hash: str) -> Dict[str, Any]:
    diffs: List[Dict[str, Any]] = []
    for i, (c, p) in enumerate(zip(current, prior)):
        if c != p:
            diffs.append({
                "prompt_idx": i,
                "current": c[:160],
                "prior": p[:160],
            })
    return {
        "prior_checkpoint": prior_ckpt,
        "prior_hash": prior_hash,
        "diff_count": len(diffs),
        "total_prompts": len(current),
        "preview": diffs[:3],
    }
