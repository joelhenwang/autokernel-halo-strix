"""Optimizer factory with DeepSpeed CPUAdam and COOKBOOK.md param groups."""

import math
from typing import List, Optional, Tuple, Type

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Sprint 1 (2026-05-06) — IMU-1 parameter grouping
# ---------------------------------------------------------------------------
#
# Every parameter lands in exactly one of two groups:
#
#   Group A (2D weights):
#     - Linear.weight (ndim >= 2) except embed/lm_head tied tensors
#     - FactorizedEmbedding.projection.weight (post-embedding up-proj)
#     -> NorMuon optimizer, lr_2d (default 0.0235), WD 0.1
#
#   Group B (1D / embedding / output head):
#     - biases, LayerNorm gammas, scalar gates (*_scale, head_gate, v_res_scale)
#     - FactorizedEmbedding.embed.weight (raw embedding table)
#     - lm_head.* (tied to embed in FactorizedLMHead; counted once via id())
#     -> AdamW, lr_1d (default 0.007), WD 0.0
#
# The IMU-1 paper validates this split for 430M models. We adopt it at 122M
# on the hypothesis that scale-attenuation of the ~3.85% loss gain is partial,
# not total. Sprint 1's Run 2 validates empirically.


_1D_NAME_MARKERS = (
    "_gate",        # head_gate, any scalar gate
    "_scale",       # v_res_scale, q_scale, k_scale, z_scale, etc.
    ".bias",        # biases (catches .bias even when sub-module)
)


def split_params_2d_vs_1d(
    model: nn.Module,
) -> Tuple[List[Tuple[str, torch.nn.Parameter]], List[Tuple[str, torch.nn.Parameter]]]:
    """Partition model parameters into IMU-1's (2D-weights, 1D-and-embed) groups.

    Returns a pair of ``[(name, param), ...]`` lists. Tied weights (e.g.
    FactorizedLMHead tied to FactorizedEmbedding) appear exactly once:
    the first name encountered in ``named_parameters()`` iteration order,
    typically the embedding's.

    Classification rules (applied in order):
      1. Skip parameters with ``requires_grad=False``
      2. Skip parameters whose id() we've already seen (tied-weight de-dup)
      3. Group B (1D) if ANY of:
         - ``p.ndim < 2`` (scalars, vectors — biases, LN gammas)
         - name starts with ``"tok_embeddings.embed."`` (raw embed table)
         - name starts with ``"lm_head."`` (only reached for untied heads;
           FactorizedLMHead ties to embed so lm_head.weight shares id())
         - name contains any of ``_1D_NAME_MARKERS``
      4. Otherwise Group A (2D weights -> NorMuon)
    """
    group_2d: List[Tuple[str, torch.nn.Parameter]] = []
    group_1d: List[Tuple[str, torch.nn.Parameter]] = []
    seen_ids: set = set()

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in seen_ids:
            # Tied weight: already counted under its first name
            continue
        seen_ids.add(id(p))

        is_1d = (
            p.ndim < 2
            or name.startswith("tok_embeddings.embed.")
            or name.startswith("lm_head.")
            or any(marker in name for marker in _1D_NAME_MARKERS)
        )
        (group_1d if is_1d else group_2d).append((name, p))

    return group_2d, group_1d


def build_imu1_optimizer(
    model: nn.Module,
    lr_2d: float = 0.0235,
    lr_1d: float = 0.007,
    weight_decay_2d: float = 0.1,
    betas: Tuple[float, float] = (0.9, 0.95),
    use_normuon: bool = False,
    # Sprint 1.1 Phase B: throughput-optimization knobs
    ns_dtype: Optional[torch.dtype] = None,
    neuron_norm_min_dim: int = 0,
    cautious_wd: bool = True,
    # Sprint 1.5 Phase A: μP 3-way LR split (disabled by default)
    use_mup: bool = False,
    mup_base_width: int = 256,
    # Sprint 1.5 Phase A: SPECTRA post-clip (disabled by default)
    spectra_post: bool = False,
    spectra_clip_norm: float = 1.0,
    spectra_ns_iter: int = 5,
    # v3 40k campaign additions (all default OFF for baseline parity)
    telemetry_enabled: bool = False,
    telemetry_path: Optional[str] = None,
    trust_cap: float = 0.0,
    trust_cap_scope: str = "none",
    w_gate_up_scale: float = 1.0,
    w_gate_up_ramp_steps: int = 0,
    spectra_branchless: bool = False,
) -> torch.optim.Optimizer:
    """Sprint 1 entry point: build a two-group optimizer per IMU-1 recipe.

    With ``use_normuon=False`` (default until Phase 2 ships NorMuon), both
    groups go to fused AdamW with their respective LRs and WDs. This is the
    "free wins" configuration — no-WD on embeddings + per-group LRs — that
    Phase 1 validates independently of NorMuon.

    With ``use_normuon=True`` (Phase 2+), the 2D group is routed to NorMuon
    and the 1D group stays on AdamW no-WD.

    Sprint 1.5 Phase A additions
    ----------------------------
    ``use_mup=True`` replaces the flat 2D group with a μP 3-way split
    (embedding / hidden / readout) with per-group LR scaling:

        embedding LR = lr_2d
        hidden   LR = lr_2d / d_ratio     where d_ratio = d_model / mup_base_width
        readout  LR = lr_2d / d_ratio**2

    ``spectra_post=True`` threads SPECTRA post-clipping into NorMuon when
    ``use_normuon`` is also True. Combines with μP transparently.

    Sprint 1.1 Phase B throughput knobs (only meaningful with use_normuon=True):

    Parameters
    ----------
    ns_dtype : torch.dtype or None
        Cast dtype for Newton-Schulz inner matmuls. ``None`` or
        ``torch.float32`` uses fp32 (safe default, matches Phase 2 behavior).
        ``torch.float16`` routes NS through rocBLAS HHS_BH_ fp16 kernels,
        which Phase A measured as 8-13x faster on SwiGLU shapes.
    neuron_norm_min_dim : int
        Sprint 1.1 Phase B2. If >0, neuron-wise normalization is skipped on
        2D params whose smaller dimension is below this threshold. Set to
        e.g. 512 to skip embedding-adjacent small projections. 0 = always
        apply (Phase 2 behavior).
    cautious_wd : bool
        Toggle cautious weight decay (per IMU-1 paper). False = standard
        decoupled WD. True is IMU-1 default and matches Phase 2 behavior.
    use_mup : bool
        Sprint 1.5 Phase A. Replaces the 2-way 2D/1D split with a 3-way μP
        split on 2D params (embedding / hidden / readout) with per-group LR
        scaling. 1D params continue through AdamW at ``lr_1d``.
    mup_base_width : int
        μP base width ``d_base`` used for LR scaling. Default 256 (matches
        the 30M probe's d_model).
    spectra_post : bool
        Sprint 1.5 Phase A. Enable SPECTRA post-clipping on the optimizer
        update inside NorMuon. No effect when use_normuon=False.
    spectra_clip_norm : float
        Spectral-norm ceiling for the weight delta per step. Default 1.0.
    spectra_ns_iter : int
        Kept for API forward-compat with future Newton-Schulz spectral
        estimators; unused by the current power-iteration implementation.
    """
    group_2d, group_1d = split_params_2d_vs_1d(model)
    n_2d = sum(p.numel() for _, p in group_2d)
    n_1d = sum(p.numel() for _, p in group_1d)

    # --- Sprint 1.5 Phase A: resolve μP groups if enabled ----------------
    mup_groups = None
    if use_mup:
        from halo_training.mup import build_mup_param_groups
        # build_mup_param_groups returns ONLY 2D params split 3-way. We use
        # weight_decay=0 here and let NorMuon/AdamW apply weight_decay_2d
        # per-group below (so the Sprint 1 knob semantics are preserved).
        mup_raw = build_mup_param_groups(
            model, base_lr=lr_2d, d_base=mup_base_width,
            weight_decay=weight_decay_2d,
        )
        mup_groups = mup_raw
        # Collect ids to resolve the 1D set under μP, which differs from
        # Sprint 1's 1D set because the μP classifier routes tok_embeddings.*
        # and lm_head.* (all of them, not just .embed./.proj.) into the 2D
        # μP groups. We reconstruct the 1D set as "any param with ndim<2 or
        # matching _1D_NAME_MARKERS, that is NOT already in a μP group".
        mup_2d_ids = set()
        for grp in mup_groups:
            for p in grp["params"]:
                mup_2d_ids.add(id(p))

        mup_1d_named: List[Tuple[str, torch.nn.Parameter]] = []
        seen_ids: set = set()
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if id(p) in mup_2d_ids:
                continue  # already routed to a μP 2D group
            if id(p) in seen_ids:
                continue
            seen_ids.add(id(p))
            if p.ndim < 2 or any(marker in name for marker in _1D_NAME_MARKERS):
                mup_1d_named.append((name, p))
            else:
                # 2D param that didn't fall into μP's classification AND
                # isn't 1D — this is unexpected under OdinFlat/OdinHalo.
                # Fall back to treating as hidden so nothing is silently dropped.
                print(f"[μP] NOTE: unclassified 2D param {name} routed to hidden group")
                # Append to hidden group
                for grp in mup_groups:
                    if grp["_mup_group"] == "hidden":
                        grp["params"].append(p)
                        break

        # Override group_1d with the μP-consistent 1D set.
        group_1d = mup_1d_named
        n_1d = sum(p.numel() for _, p in group_1d)

        for grp in mup_groups:
            print(f"[μP] group={grp['_mup_group']:10s} "
                  f"n_params={sum(p.numel() for p in grp['params']):>10,} "
                  f"lr={grp['lr']:.6f}")
        print(f"[μP] group={'1D':10s} "
              f"n_params={n_1d:>10,} lr={lr_1d:.6f} (AdamW)")

    if use_normuon:
        # Phase 2+: NorMuon for 2D, AdamW for 1D.
        # Import is lazy so Phase 1 code doesn't require the module to exist.
        try:
            from halo_training.normuon import NorMuon
        except ImportError as exc:
            raise RuntimeError(
                "build_imu1_optimizer(use_normuon=True) requires "
                "halo_training/normuon.py; this ships in Sprint 1 Phase 2."
            ) from exc
        if use_mup and mup_groups is not None:
            # When μP is active, feed the 3-way split to NorMuon as separate
            # muon_params groups so each retains its per-group LR.
            # v3 T-0.2: attach _telem_param_names for telemetry readability
            # (underscore-prefixed to avoid collision with PyTorch's reserved
            # 'param_names' optimizer key which requires all-or-none groups).
            muon_groups_for_normuon = [
                {"params": grp["params"],
                 "_telem_param_names": grp.get("_mup_param_names",
                                         [f"mup_{grp['_mup_group']}_{i}"
                                          for i in range(len(grp["params"]))]),
                 "lr": grp["lr"],
                 "weight_decay": weight_decay_2d,
                 "_mup_group": grp["_mup_group"]}
                for grp in mup_groups if grp["params"]
            ]
            opt = NorMuon(
                muon_params=muon_groups_for_normuon,
                adamw_params=[
                    {"params": [p for _, p in group_1d], "lr": lr_1d, "weight_decay": 0.0},
                ],
                lr=lr_2d,  # default LR; per-group lrs above override
                weight_decay=weight_decay_2d,
                betas=betas,
                ns_dtype=ns_dtype,
                neuron_norm_min_dim=neuron_norm_min_dim,
                cautious_wd=cautious_wd,
                spectra_post=spectra_post,
                spectra_clip_norm=spectra_clip_norm,
                spectra_ns_iter=spectra_ns_iter,
                telemetry_enabled=telemetry_enabled,
                telemetry_path=telemetry_path,
                trust_cap=trust_cap,
                trust_cap_scope=trust_cap_scope,
                w_gate_up_scale=w_gate_up_scale,
                w_gate_up_ramp_steps=w_gate_up_ramp_steps,
                spectra_branchless=spectra_branchless,
            )
        else:
            opt = NorMuon(
                muon_params=[{"params": [p for _, p in group_2d],
                              "_telem_param_names": [n for n, _ in group_2d]}],
                adamw_params=[
                    {"params": [p for _, p in group_1d], "lr": lr_1d, "weight_decay": 0.0},
                ],
                lr=lr_2d,
                weight_decay=weight_decay_2d,
                betas=betas,
                ns_dtype=ns_dtype,
                neuron_norm_min_dim=neuron_norm_min_dim,
                cautious_wd=cautious_wd,
                spectra_post=spectra_post,
                spectra_clip_norm=spectra_clip_norm,
                spectra_ns_iter=spectra_ns_iter,
                telemetry_enabled=telemetry_enabled,
                telemetry_path=telemetry_path,
                trust_cap=trust_cap,
                trust_cap_scope=trust_cap_scope,
                w_gate_up_scale=w_gate_up_scale,
                w_gate_up_ramp_steps=w_gate_up_ramp_steps,
                spectra_branchless=spectra_branchless,
            )
        # Phase 2 logging + Phase B throughput-knob trail so scorecard runs
        # record exactly which config was used.
        ns_tag = (str(ns_dtype).split(".")[-1] if ns_dtype is not None
                  else "fp32")
        mup_tag = "mup3x" if use_mup else "flat"
        spectra_tag = (f"spectra({spectra_clip_norm})" if spectra_post else "no-spectra")
        print(f"IMU-1 optimizer: NorMuon(2D, n={n_2d:,}, lr={lr_2d}, "
              f"ns_dtype={ns_tag}, neuron_norm_min_dim={neuron_norm_min_dim}, "
              f"cautious_wd={cautious_wd}, {mup_tag}, {spectra_tag}) "
              f"+ AdamW(1D, n={n_1d:,}, lr={lr_1d}, wd=0)")
        return opt

    # Phase 1 default: AdamW for both groups, per-group LR/WD
    if use_mup and mup_groups is not None:
        # AdamW path with μP: feed 3-way groups + 1D group separately.
        adamw_groups = [
            {"params": grp["params"], "lr": grp["lr"],
             "weight_decay": weight_decay_2d,
             "_mup_group": grp["_mup_group"]}
            for grp in mup_groups if grp["params"]
        ] + [
            {"params": [p for _, p in group_1d], "lr": lr_1d,
             "weight_decay": 0.0},
        ]
        opt = torch.optim.AdamW(
            adamw_groups, lr=lr_2d, betas=betas, fused=True)
    else:
        opt = torch.optim.AdamW(
            [
                {
                    "params": [p for _, p in group_2d],
                    "lr": lr_2d,
                    "weight_decay": weight_decay_2d,
                },
                {
                    "params": [p for _, p in group_1d],
                    "lr": lr_1d,
                    "weight_decay": 0.0,
                },
            ],
            lr=lr_2d,
            betas=betas,
            fused=True,
        )
    print(f"IMU-1 optimizer (Phase 1): AdamW {'μP-3way' if use_mup else 'two-group'} — "
          f"2D(n={n_2d:,}, lr={lr_2d}, wd={weight_decay_2d}) + "
          f"1D(n={n_1d:,}, lr={lr_1d}, wd=0)")
    return opt


def _patch_deepspeed():
    """Fix DeepSpeed 0.17.5 bug: cxx_args() returns None on ROCm."""
    try:
        import deepspeed.ops.op_builder.builder as builder
        _orig = builder.OpBuilder.strip_empty_entries

        def _patched(self, args):
            return [x for x in args if x is not None and len(x) > 0]

        builder.OpBuilder.strip_empty_entries = _patched
    except ImportError:
        pass


def _get_cpu_adam():
    """Try to import DeepSpeed CPUAdam with ROCm fixes."""
    _patch_deepspeed()
    try:
        from deepspeed.ops.adam import DeepSpeedCPUAdam
        return DeepSpeedCPUAdam
    except (ImportError, RuntimeError):
        return None


def build_optimizer(
    model: nn.Module,
    base_lr: float = 8e-4,
    weight_decay: float = 0.1,
    optimizer_cls: Optional[Type] = None,
    use_cpu_adam: bool = False,
    use_muon: bool = False,
    muon_lr: float = 0.005,
    polar_ns: bool = False,
    use_lion: bool = False,
    lion_lr_ratio: float = 0.3,
    use_clion: bool = False,
    clion_nu: float = 1e-3,
    clion_gate_mode: str = "per_coord",
) -> torch.optim.Optimizer:
    """Build optimizer with COOKBOOK.md param group rules.

    Param group assignment by name pattern:
    - engram tables: 5x LR, 0 WD
    - decay_bias: 0.1x LR, 0 WD
    - omega/gamma params: 0.125x LR, 0 WD
    - meta_token: 1x LR, 0.01 WD
    - norms/biases: 1x LR, 0 WD
    - everything else: 1x LR, default WD

    Args:
        use_cpu_adam: Force DeepSpeed CPUAdam. Requires params on CPU.
                     Useful for Mode B (7B+) where optimizer states dominate memory.
                     For Mode A (<2B), fused AdamW on GPU is simpler and fast enough.
        use_muon: Use Muon optimizer for 2D weight matrices, AdamW for rest.
                  ~2x token-efficiency, ~50% less optimizer memory.
        muon_lr: Learning rate for Muon params (default 0.02, different scale from AdamW).
        use_lion: Use Lion optimizer (sign-based momentum; 1 state buffer vs AdamW's 2).
                  LR is auto-scaled to base_lr * lion_lr_ratio (default 0.3x).
                  Mutually exclusive with use_muon.
        lion_lr_ratio: Lion LR = base_lr * this_ratio (paper recommends ~0.1-0.3 vs AdamW).
        use_clion: Use CLion (Cautious Lion; arXiv:2604.14587). Same shape as Lion
                   but gates sign() behind a threshold ν per tensor. Lower
                   generalization error than Lion. Uses lion_lr_ratio for LR scaling.
        clion_nu: ν threshold for cautious sign gate (default 1.0 from paper's
                  generalization theorem; try 1/√d for very large models).
    """
    n_opt_flags = int(use_lion) + int(use_muon) + int(use_clion)
    if n_opt_flags > 1:
        raise ValueError("use_lion, use_muon, use_clion are mutually exclusive")

    if use_muon:
        return _build_muon_optimizer(model, base_lr, muon_lr, weight_decay, polar_ns=polar_ns)

    groups = _build_param_groups(model, base_lr, weight_decay)

    if use_lion:
        from halo_training.lion import Lion
        lion_lr = base_lr * lion_lr_ratio
        for g in groups:
            g["lr"] = g["lr"] * lion_lr_ratio
        opt = Lion(groups, lr=lion_lr, betas=(0.9, 0.99), weight_decay=weight_decay)
        print(f"Using Lion optimizer (lr={lion_lr}, scaled from base_lr={base_lr} x {lion_lr_ratio})")
        return opt

    if use_clion:
        from halo_training.clion import CLion
        clion_lr = base_lr * lion_lr_ratio
        for g in groups:
            g["lr"] = g["lr"] * lion_lr_ratio
        opt = CLion(groups, lr=clion_lr, betas=(0.9, 0.99),
                    weight_decay=weight_decay, nu=clion_nu,
                    gate_mode=clion_gate_mode)
        print(f"Using CLion optimizer (lr={clion_lr}, nu={clion_nu}, "
              f"gate_mode={clion_gate_mode}, "
              f"scaled from base_lr={base_lr} x {lion_lr_ratio})")
        return opt

    if optimizer_cls is not None:
        return optimizer_cls(groups, lr=base_lr)

    if use_cpu_adam:
        CPUAdam = _get_cpu_adam()
        if CPUAdam is not None:
            try:
                opt = CPUAdam(groups, lr=base_lr, betas=(0.9, 0.95))
                print("Using DeepSpeed CPUAdam (AVX-512 + OpenMP)")
                return opt
            except Exception as e:
                print(f"DeepSpeed CPUAdam failed ({e}), falling back to AdamW")

    print("Using torch.optim.AdamW(fused=True)")
    return torch.optim.AdamW(groups, lr=base_lr, betas=(0.9, 0.95), fused=True)


def _build_muon_optimizer(
    model: nn.Module,
    adamw_lr: float,
    muon_lr: float,
    weight_decay: float,
    polar_ns: bool = False,
) -> torch.optim.Optimizer:
    """Build Muon optimizer: 2D weights get Muon, rest gets AdamW."""
    from halo_training.muon import Muon, split_params_for_muon

    muon_params, adamw_named = split_params_for_muon(model)

    # Build AdamW groups with COOKBOOK.md rules
    adamw_groups = []
    for name, param in adamw_named:
        if "engram" in name and "table" in name:
            adamw_groups.append({"params": [param], "lr": adamw_lr * 5.0, "weight_decay": 0})
        elif "decay_bias" in name:
            adamw_groups.append({"params": [param], "lr": adamw_lr * 0.1, "weight_decay": 0})
        elif "omega_param" in name or "gamma_param" in name:
            adamw_groups.append({"params": [param], "lr": adamw_lr * 0.125, "weight_decay": 0})
        elif "meta_token" in name:
            adamw_groups.append({"params": [param], "lr": adamw_lr, "weight_decay": 0.01})
        elif "norm" in name or (name.endswith(".bias") and "decay" not in name):
            adamw_groups.append({"params": [param], "lr": adamw_lr, "weight_decay": 0})
        else:
            adamw_groups.append({"params": [param], "lr": adamw_lr, "weight_decay": weight_decay})

    n_muon = len(muon_params)
    n_adamw = sum(len(g["params"]) for g in adamw_groups)
    print(f"Using Muon optimizer: {n_muon} params via Muon (lr={muon_lr}), "
          f"{n_adamw} params via AdamW (lr={adamw_lr})")

    return Muon(
        muon_params=[{"params": muon_params}],
        lr=muon_lr,
        weight_decay=0.01,
        adamw_params=adamw_groups,
        adamw_lr=adamw_lr,
        polar_ns=polar_ns,
    )


def _build_param_groups(
    model: nn.Module,
    base_lr: float,
    weight_decay: float,
) -> List[dict]:
    """Adapted from COOKBOOK.md lines 313-331."""
    groups = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "engram" in name and "table" in name:
            groups.append({
                "params": [param], "lr": base_lr * 5.0,
                "weight_decay": 0, "name": name,
            })
        elif "decay_bias" in name:
            groups.append({
                "params": [param], "lr": base_lr * 0.1,
                "weight_decay": 0, "name": name,
            })
        elif "omega_param" in name or "gamma_param" in name:
            groups.append({
                "params": [param], "lr": base_lr * 0.125,
                "weight_decay": 0, "name": name,
            })
        elif "meta_token" in name:
            groups.append({
                "params": [param], "lr": base_lr,
                "weight_decay": 0.01, "name": name,
            })
        elif "norm" in name or (name.endswith(".bias") and "decay" not in name):
            groups.append({
                "params": [param], "lr": base_lr,
                "weight_decay": 0, "name": name,
            })
        else:
            groups.append({
                "params": [param], "lr": base_lr,
                "weight_decay": weight_decay, "name": name,
            })
    return groups


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int = 100,
    min_lr_ratio: float = 0.1,
    warm_restarts: bool = False,
    restart_period: Optional[int] = None,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Cosine schedule with linear warmup. Base LR 8e-4 -> 8e-5.

    Args:
        warm_restarts: Use CosineAnnealingWarmRestarts instead (for ternary/quantized models).
        restart_period: Steps between restarts (default: total_steps // 4).
    """
    if warm_restarts:
        T_0 = restart_period or max(1, total_steps // 4)
        min_lr = optimizer.defaults["lr"] * min_lr_ratio
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=1, eta_min=min_lr,
        )

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_wsd_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int = 300,
    stable_fraction: float = 0.8,
    min_lr_ratio: float = 0.0,
    wd_start: float = 0.1,
    wd_end: float = 0.01,
) -> "WSDScheduler":
    """Warmup-Stable-Decay schedule with weight decay annealing.

    Warmup: linear 0→1 over warmup_steps.
    Stable: constant 1.0 until total_steps * stable_fraction.
    Decay: linear 1.0→min_lr_ratio over remaining steps.
    Weight decay: anneals wd_start→wd_end during decay phase.
    """
    return WSDScheduler(optimizer, total_steps, warmup_steps,
                        stable_fraction, min_lr_ratio, wd_start, wd_end)


class WSDScheduler(torch.optim.lr_scheduler.LambdaLR):
    """Warmup-Stable-Decay LR schedule with weight decay annealing."""

    def __init__(self, optimizer, total_steps, warmup_steps=300,
                 stable_fraction=0.8, min_lr_ratio=0.0,
                 wd_start=0.1, wd_end=0.01):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.decay_start = int(total_steps * stable_fraction)
        self.min_lr_ratio = min_lr_ratio
        self.wd_start = wd_start
        self.wd_end = wd_end

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            if step < self.decay_start:
                return 1.0
            decay_progress = (step - self.decay_start) / max(1, total_steps - self.decay_start)
            return max(min_lr_ratio, 1.0 - (1.0 - min_lr_ratio) * decay_progress)

        super().__init__(optimizer, lr_lambda)

    def step(self, epoch=None):
        super().step(epoch)
        current_step = self.last_epoch
        if current_step >= self.decay_start:
            decay_progress = (current_step - self.decay_start) / max(
                1, self.total_steps - self.decay_start)
            decay_progress = min(1.0, decay_progress)
            wd = self.wd_start + (self.wd_end - self.wd_start) * decay_progress
            for group in self.optimizer.param_groups:
                if group.get("weight_decay", 0) > 0:
                    group["weight_decay"] = wd
