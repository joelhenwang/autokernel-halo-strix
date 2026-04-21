"""CLIMB + Self-Improving data mixture pipeline.

Shared constants, paths, and resumable state utilities.
"""

import json
from pathlib import Path
from typing import Optional

DEFAULT_STATE_DIR = Path("datasets/datamix_state")

PHASE_DIRS = {
    "phase0": "phase0_sample",
    "phase1": "phase1_embeddings",
    "phase2": "phase2_clusters",
    "phase3": "phase3_proxy_results",
    "phase4": "phase4_scores",
    "phase5": "phase5_final",
}


def get_state_dir(base: Optional[Path] = None) -> Path:
    d = base or DEFAULT_STATE_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_pipeline_state(base: Optional[Path] = None) -> dict:
    f = get_state_dir(base) / "pipeline_state.json"
    if f.exists():
        return json.loads(f.read_text())
    return {}


def save_pipeline_state(state: dict, base: Optional[Path] = None):
    f = get_state_dir(base) / "pipeline_state.json"
    f.write_text(json.dumps(state, indent=2))


def phase_dir(phase: str, base: Optional[Path] = None) -> Path:
    d = get_state_dir(base) / PHASE_DIRS[phase]
    d.mkdir(parents=True, exist_ok=True)
    return d


def check_dependency(state: dict, phase: str):
    """Raise if required predecessor phase is not complete."""
    deps = {
        "phase1": "phase0",
        "phase2": "phase1",
        "phase3": "phase2",
        "phase4": ["phase1", "phase2"],
        "phase5": ["phase2", "phase3"],
    }
    required = deps.get(phase, [])
    if isinstance(required, str):
        required = [required]
    for dep in required:
        info = state.get(dep, {})
        if info.get("status") != "complete":
            raise RuntimeError(f"Phase {phase} requires {dep} to be complete (current: {info.get('status', 'not started')})")
