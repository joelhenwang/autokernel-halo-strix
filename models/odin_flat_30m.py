"""OdinFlat 30M proportional probe model for μP transfer study.

Width: d=512 (vs OdinFlat production 768), layers=8 (vs 14), head_dim=64
preserved. Target param count: ~30M (for fast LR-probe ablation runs).

Used by ``scripts/probe_mup_lr.py`` (Sprint 1.5 Phase B.3) to establish a
μP-transferable LR before the full 122M production run.

See design spec §5.4:
``docs/superpowers/specs/2026-05-06-sprint1.5-spectra-mup-design.md``
"""

from __future__ import annotations

from typing import Tuple

from models.odin_flat import OdinFlatBase


class OdinFlat30M(OdinFlatBase):
    """30M proportional probe: d=512, layers=8, head_dim=64 preserved.

    Matches OdinFlat production architecture under μP's proportional
    scaling rules. Param count ≈ 30M enables a 700-step LR sweep in
    ~45 min DDP wall time on gfx1151, vs ~65 min for the 122M
    full-config probe.

    Construction defaults are the μP "base" width (d_base=512 corresponds
    to d_ratio=2 for the 122M d=768 production target). Callers may
    override any default.
    """

    def __init__(self, **kw):
        kw.setdefault("d_model", 512)
        kw.setdefault("embed_rank", 128)
        kw.setdefault("n_layers", 8)
        kw.setdefault("gqa_positions", (3, 7))
        kw.setdefault("n_heads", 8)
        kw.setdefault("n_kv_heads", 2)
        kw.setdefault("ffn_inner", 1878)
        kw.setdefault("d_conv", 384)
        super().__init__(**kw)


# For symmetry with OdinFlat's classes, expose a smoke / mini variant.
class OdinFlat30MMini(OdinFlatBase):
    """Tiny smoke test for the 30M probe (100× smaller).

    d_model=64 → ~800k params. Used by unit tests.
    """

    def __init__(self, **kw):
        kw.setdefault("vocab_size", 1000)
        kw.setdefault("d_model", 64)
        kw.setdefault("embed_rank", 32)
        kw.setdefault("n_layers", 4)
        kw.setdefault("gqa_positions", (1, 3))
        kw.setdefault("n_heads", 4)
        kw.setdefault("n_kv_heads", 2)
        kw.setdefault("ffn_inner", 128)
        kw.setdefault("d_conv", 64)
        super().__init__(**kw)
