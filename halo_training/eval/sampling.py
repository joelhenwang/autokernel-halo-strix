"""Sampling-quality evaluator (Sprint 2 Phase 3).

Wraps ``scripts/ablate_odin_flat_sampling.run_ablation()`` and extracts the
metrics at the stage-3 winning configuration.

Budget: ~1-2 min per checkpoint (15 configs x 2 prompts x 3 samples x 120
tokens = ~10,800 generated tokens). Small OdinFlat throughput at seq ~300
is 100-200 tok/s on Strix Halo, so total time is 50-100 s.

Output shape::

    {
        "winning_config": {"temp": 0.6, "rep_pen": 1.0, "top_p": 1.0, "top_k": 0},
        "distinct_2": 0.765,
        "self_ppl": 9.84,
        "avg_length": 118.5,
        "sample_preview": "The history of the Roman Empire began ...",
        "prompts": ["The history of the Roman Empire", "In the field of physics,"],
        "_ablation_summary": {
            "stage1_winner": {...},
            "stage2_winner": {...},
            "stage3_winner": {...}
        }
    }
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Make ``scripts.*`` importable
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def run(model, tokenizer, validation_splits, args=None) -> Optional[Dict[str, Any]]:
    """Run the 3-stage sampling ablation and report metrics at the winner.

    Returns ``None`` if the ablation harness cannot be imported (e.g. during
    Phase 1 scaffolding verification).
    """
    try:
        from scripts.ablate_odin_flat_sampling import (
            run_ablation,
            select_winning_config,
        )
    except ImportError as exc:
        print(f"  [sampling] could not import ablation harness: {exc}")
        return None

    # The ablation harness needs: tok (HF), bl_decoder (ByteLevel), eos_id, vocab_size
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder

    bl_decoder = ByteLevelDecoder()
    vocab_size = tokenizer.get_vocab_size()
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    if eos_id is None:
        eos_id = 0  # matches AGENTS.md fallback

    prompts = [
        "The history of the Roman Empire",
        "In the field of physics,",
    ]

    results = run_ablation(
        model, tokenizer, bl_decoder, eos_id, vocab_size,
        prompts=prompts, verbose=False,
    )

    winner = select_winning_config(results["stage3"])
    wcfg = winner["cfg"]
    wm = winner["metrics"]

    # Extract the first-prompt preview if present
    previews = wm.get("previews") or {}
    sample_preview = previews.get(0) or previews.get(1) or ""

    return {
        "winning_config": {
            "temperature": wcfg["temp"],
            "repetition_penalty": wcfg["rep_pen"],
            "top_p": wcfg["top_p"],
            "top_k": wcfg["top_k"],
        },
        "distinct_2": round(float(wm["dist2"]), 4),
        "self_ppl": round(float(wm["ppl"]), 3),
        "avg_length": round(float(wm["len"]), 1),
        "sample_preview": sample_preview[:200],
        "prompts": prompts,
        "_ablation_summary": {
            "stage1_winner": _summarise_entry(results["winners"]["stage1"]),
            "stage2_winner": _summarise_entry(results["winners"]["stage2"]),
            "stage3_winner": _summarise_entry(results["winners"]["stage3"]),
        },
    }


def _summarise_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    cfg = entry["cfg"]
    m = entry["metrics"]
    return {
        "temperature": cfg["temp"],
        "repetition_penalty": cfg["rep_pen"],
        "top_p": cfg["top_p"],
        "top_k": cfg["top_k"],
        "distinct_2": round(float(m["dist2"]), 4),
        "self_ppl": round(float(m["ppl"]), 3),
    }
