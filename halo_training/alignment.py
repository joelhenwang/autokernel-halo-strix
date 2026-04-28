"""Unified alignment losses: ORPO, SimPO, KTO + drop-in modifiers (D2PO, ConfPO, AlphaPO).

All losses operate on per-token log probabilities and assistant masks.
No TRL dependency — pure PyTorch.

Usage:
    from halo_training.alignment import orpo_loss, simpo_loss, kto_loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Shared: compute per-token and sequence-level log probs
# ---------------------------------------------------------------------------

def get_per_token_logps(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-token log P(label_t | prefix).

    Args:
        logits: (B, T, V) model output
        labels: (B, T) target token IDs

    Returns:
        (B, T-1) per-token log probs (shifted: position i predicts token i+1)
    """
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    return log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)


def get_sequence_logps(
    per_token_logps: torch.Tensor,
    mask: torch.Tensor,
    average: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sum or average log probs over masked (assistant) positions.

    Args:
        per_token_logps: (B, T-1) from get_per_token_logps
        mask: (B, T) assistant mask (1 = score this token)
        average: if True, return mean instead of sum

    Returns:
        logps: (B,) sequence-level log probs
        lengths: (B,) number of masked tokens per sequence
    """
    shift_mask = mask[:, 1:].float()
    masked = per_token_logps * shift_mask
    lengths = shift_mask.sum(dim=-1).clamp(min=1)
    if average:
        return masked.sum(dim=-1) / lengths, lengths
    return masked.sum(dim=-1), lengths


def forward_logps(
    model: nn.Module,
    input_ids: torch.Tensor,
    mask: torch.Tensor,
    average: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward pass → per-token logps, sequence logps, lengths.

    Returns:
        per_token: (B, T-1)
        seq_logps: (B,)
        lengths: (B,)
    """
    with torch.amp.autocast("cuda", dtype=torch.float16):
        logits = model(input_ids)
    per_token = get_per_token_logps(logits, input_ids)
    seq_logps, lengths = get_sequence_logps(per_token, mask, average=average)
    return per_token, seq_logps, lengths


# ---------------------------------------------------------------------------
# Token-level modifiers (D2PO, ConfPO — applied before aggregation)
# ---------------------------------------------------------------------------

def apply_d2po_decay(
    per_token_logps: torch.Tensor,
    mask: torch.Tensor,
    gamma: float = 0.99,
) -> torch.Tensor:
    """D2PO: exponential temporal decay weighting earlier tokens more.

    Weights: w_t = gamma^(T-1-t), so first token gets gamma^(T-1), last gets 1.0.
    Reversed: earlier tokens weighted MORE (they establish context).

    Args:
        per_token_logps: (B, T-1)
        mask: (B, T) assistant mask
        gamma: decay rate (0.95-0.99)

    Returns:
        (B, T-1) reweighted log probs
    """
    shift_mask = mask[:, 1:]
    T = per_token_logps.shape[1]
    positions = torch.arange(T, device=per_token_logps.device).float()
    weights = gamma ** (T - 1 - positions)
    weights = weights.unsqueeze(0) * shift_mask.float()
    total_w = weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    normalized = weights / total_w * shift_mask.float().sum(dim=-1, keepdim=True).clamp(min=1)
    return per_token_logps * normalized


def apply_confpo_selection(
    per_token_logps: torch.Tensor,
    mask: torch.Tensor,
    _unused: float = 0.0,
) -> torch.Tensor:
    """ConfPO: select preference-critical tokens via policy confidence.

    Selects tokens where model probability <= per-sequence mean probability.
    Low-confidence tokens are most informative for preference learning.
    Zero additional hyperparameters.

    Args:
        per_token_logps: (B, T-1)
        mask: (B, T) assistant mask

    Returns:
        (B, T-1) masked log probs (high-confidence tokens zeroed)
    """
    shift_mask = mask[:, 1:].float()
    probs = per_token_logps.exp()
    masked_probs = probs * shift_mask
    lengths = shift_mask.sum(dim=-1, keepdim=True).clamp(min=1)
    mean_prob = masked_probs.sum(dim=-1, keepdim=True) / lengths
    selected = ((probs <= mean_prob) & (shift_mask > 0)).float()
    selected_count = selected.sum(dim=-1, keepdim=True).clamp(min=1)
    scale = lengths / selected_count
    return per_token_logps * selected * scale


# ---------------------------------------------------------------------------
# ORPO: Monolithic SFT + Odds Ratio Preference (no reference model)
# ---------------------------------------------------------------------------

def orpo_loss(
    model: nn.Module,
    chosen_ids: torch.Tensor,
    chosen_mask: torch.Tensor,
    rejected_ids: torch.Tensor,
    rejected_mask: torch.Tensor,
    lambda_weight: float = 0.1,
    d2po_gamma: float = 0.0,
    confpo_frac: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """ORPO loss = NLL(chosen) + lambda * -log_sigmoid(log_odds_ratio).

    No reference model needed. Single-stage alignment.

    Args:
        lambda_weight: preference loss weight (0.1-0.25)
        d2po_gamma: if >0, apply D2PO temporal decay
        confpo_frac: if >0, apply ConfPO token selection
    """
    with torch.amp.autocast("cuda", dtype=torch.float16):
        chosen_logits = model(chosen_ids)
        rejected_logits = model(rejected_ids)

    # SFT loss on chosen (standard NLL with masking)
    shift_logits = chosen_logits[:, :-1, :].float()
    shift_labels = chosen_ids[:, 1:]
    shift_mask = chosen_mask[:, 1:].float()

    per_token_nll = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction='none',
    ).reshape(shift_logits.shape[0], -1)

    nll_loss = (per_token_nll * shift_mask).sum() / shift_mask.sum().clamp(min=1)

    # Average log probs for odds ratio
    chosen_per_token = get_per_token_logps(chosen_logits, chosen_ids)
    rejected_per_token = get_per_token_logps(rejected_logits, rejected_ids)

    if d2po_gamma > 0:
        chosen_per_token = apply_d2po_decay(chosen_per_token, chosen_mask, d2po_gamma)
        rejected_per_token = apply_d2po_decay(rejected_per_token, rejected_mask, d2po_gamma)

    if confpo_frac > 0:
        chosen_per_token = apply_confpo_selection(chosen_per_token, chosen_mask, confpo_frac)
        rejected_per_token = apply_confpo_selection(rejected_per_token, rejected_mask, confpo_frac)

    chosen_avg_logps, _ = get_sequence_logps(chosen_per_token, chosen_mask, average=True)
    rejected_avg_logps, _ = get_sequence_logps(rejected_per_token, rejected_mask, average=True)

    # Log odds: log(p / (1-p)) = logp - log(1 - exp(logp))
    chosen_log_odds = chosen_avg_logps - torch.log1p(-torch.exp(chosen_avg_logps.clamp(max=-1e-4)))
    rejected_log_odds = rejected_avg_logps - torch.log1p(-torch.exp(rejected_avg_logps.clamp(max=-1e-4)))

    log_odds_ratio = chosen_log_odds - rejected_log_odds
    or_loss = -F.logsigmoid(log_odds_ratio).mean()

    loss = nll_loss + lambda_weight * or_loss

    with torch.no_grad():
        accuracy = (chosen_avg_logps > rejected_avg_logps).float().mean()

    return loss, {
        "loss": loss.item(),
        "nll_loss": nll_loss.item(),
        "or_loss": or_loss.item(),
        "accuracy": accuracy.item(),
        "chosen_logps": chosen_avg_logps.mean().item(),
        "rejected_logps": rejected_avg_logps.mean().item(),
    }


# ---------------------------------------------------------------------------
# SimPO: Reference-Free via Length-Normalized Reward (+ AlphaPO variant)
# ---------------------------------------------------------------------------

def simpo_loss(
    model: nn.Module,
    chosen_ids: torch.Tensor,
    chosen_mask: torch.Tensor,
    rejected_ids: torch.Tensor,
    rejected_mask: torch.Tensor,
    beta: float = 2.0,
    gamma_beta_ratio: float = 0.5,
    alpha: float = 1.0,
    d2po_gamma: float = 0.0,
    confpo_frac: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """SimPO loss with optional AlphaPO reward shaping.

    reward(x,y) = (beta/|y|) * sum log pi(y_t|x,y_<t)
    loss = -log_sigmoid(reward_chosen - reward_rejected - gamma)

    AlphaPO (alpha != 1.0): shapes the reward difference through x^alpha,
    amplifying or compressing the preference signal.

    Args:
        beta: reward scaling (2.0-10.0)
        gamma_beta_ratio: margin as fraction of beta (0.5)
        alpha: AlphaPO shaping exponent (1.0 = standard SimPO)
        d2po_gamma: if >0, apply D2PO temporal decay
        confpo_frac: if >0, apply ConfPO token selection
    """
    with torch.amp.autocast("cuda", dtype=torch.float16):
        chosen_logits = model(chosen_ids)
        rejected_logits = model(rejected_ids)

    chosen_per_token = get_per_token_logps(chosen_logits, chosen_ids)
    rejected_per_token = get_per_token_logps(rejected_logits, rejected_ids)

    if d2po_gamma > 0:
        chosen_per_token = apply_d2po_decay(chosen_per_token, chosen_mask, d2po_gamma)
        rejected_per_token = apply_d2po_decay(rejected_per_token, rejected_mask, d2po_gamma)

    if confpo_frac > 0:
        chosen_per_token = apply_confpo_selection(chosen_per_token, chosen_mask, confpo_frac)
        rejected_per_token = apply_confpo_selection(rejected_per_token, rejected_mask, confpo_frac)

    chosen_sum_logps, chosen_lengths = get_sequence_logps(chosen_per_token, chosen_mask)
    rejected_sum_logps, rejected_lengths = get_sequence_logps(rejected_per_token, rejected_mask)

    gamma = gamma_beta_ratio * beta

    chosen_avg = chosen_sum_logps / chosen_lengths
    rejected_avg = rejected_sum_logps / rejected_lengths

    if alpha == 0.0 or alpha == 1.0:
        # Standard SimPO: reward = beta * avg_logp
        chosen_rewards = beta * chosen_avg
        rejected_rewards = beta * rejected_avg
    else:
        # AlphaPO: r(y;x) = beta * (1 - exp(-alpha * avg_logp)) / alpha
        chosen_rewards = beta * (1.0 - torch.exp(-alpha * chosen_avg)) / alpha
        rejected_rewards = beta * (1.0 - torch.exp(-alpha * rejected_avg)) / alpha

    loss = -F.logsigmoid(chosen_rewards - rejected_rewards - gamma).mean()

    with torch.no_grad():
        accuracy = (chosen_rewards > rejected_rewards).float().mean()

    return loss, {
        "loss": loss.item(),
        "accuracy": accuracy.item(),
        "chosen_reward": chosen_rewards.mean().item(),
        "rejected_reward": rejected_rewards.mean().item(),
        "reward_margin": (chosen_rewards - rejected_rewards).mean().item(),
    }


# ---------------------------------------------------------------------------
# KTO: Kahneman-Tversky Optimization (unpaired, needs reference model)
# ---------------------------------------------------------------------------

def kto_loss(
    model: nn.Module,
    ref_model: nn.Module,
    input_ids: torch.Tensor,
    mask: torch.Tensor,
    labels: torch.Tensor,
    beta: float = 0.1,
    desirable_weight: float = 1.0,
    undesirable_weight: float = 1.0,
    d2po_gamma: float = 0.0,
    confpo_frac: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """KTO loss for unpaired desirable/undesirable examples.

    Args:
        input_ids: (B, T) token IDs
        mask: (B, T) assistant mask
        labels: (B,) 1 for desirable, 0 for undesirable
        beta: KL scaling (0.1)
        desirable_weight: loss weight for good examples
        undesirable_weight: loss weight for bad examples
        d2po_gamma: if >0, apply D2PO temporal decay
        confpo_frac: if >0, apply ConfPO token selection
    """
    # Policy log probs
    with torch.amp.autocast("cuda", dtype=torch.float16):
        policy_logits = model(input_ids)

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        ref_logits = ref_model(input_ids)

    policy_per_token = get_per_token_logps(policy_logits, input_ids)
    ref_per_token = get_per_token_logps(ref_logits, input_ids)

    if d2po_gamma > 0:
        policy_per_token = apply_d2po_decay(policy_per_token, mask, d2po_gamma)
        ref_per_token = apply_d2po_decay(ref_per_token, mask, d2po_gamma)

    if confpo_frac > 0:
        policy_per_token = apply_confpo_selection(policy_per_token, mask, confpo_frac)
        ref_per_token = apply_confpo_selection(ref_per_token, mask, confpo_frac)

    policy_logps, _ = get_sequence_logps(policy_per_token, mask)
    ref_logps, _ = get_sequence_logps(ref_per_token, mask)

    # Per-example rewards
    rewards = policy_logps - ref_logps

    # KL reference point (average reward across batch, detached)
    kl = rewards.detach().mean().clamp(min=0)

    desirable_mask = labels.bool()
    undesirable_mask = ~desirable_mask

    loss = torch.tensor(0.0, device=input_ids.device)
    n_des = desirable_mask.sum()
    n_und = undesirable_mask.sum()

    if n_des > 0:
        des_loss = desirable_weight * (1 - torch.sigmoid(beta * (rewards[desirable_mask] - kl)))
        loss = loss + des_loss.sum() / n_des

    if n_und > 0:
        und_loss = undesirable_weight * (1 - torch.sigmoid(beta * (kl - rewards[undesirable_mask])))
        loss = loss + und_loss.sum() / n_und

    with torch.no_grad():
        accuracy = torch.tensor(0.0, device=input_ids.device)
        if n_des > 0 and n_und > 0:
            accuracy = ((rewards[desirable_mask].mean() > rewards[undesirable_mask].mean()).float())

    return loss, {
        "loss": loss.item(),
        "accuracy": accuracy.item(),
        "kl_ref": kl.item(),
        "desirable_reward": rewards[desirable_mask].mean().item() if n_des > 0 else 0,
        "undesirable_reward": rewards[undesirable_mask].mean().item() if n_und > 0 else 0,
    }
