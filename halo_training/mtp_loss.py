"""Multi-Token Prediction loss for models with MTP auxiliary heads.

When the model returns a dict with "logits" and "mtp1" keys, this loss
computes the standard CE on main logits + weighted CE on auxiliary predictions.
Falls back to standard CE if model returns a plain tensor.
"""

import torch
import torch.nn.functional as F


def build_mtp_loss_fn(mtp_weight: float = 0.3):
    """Build MTP loss function with configurable auxiliary weight.

    Args:
        mtp_weight: Weight for MTP auxiliary loss (DeepSeek V4 uses 0.3).
    """

    def mtp_loss_fn(output, batch):
        _, targets = batch

        if isinstance(output, dict):
            logits = output["logits"]
            V = logits.shape[-1]
            targets = targets.to(logits.device)
            loss_main = F.cross_entropy(logits.view(-1, V), targets.view(-1))

            total = loss_main
            if "mtp1" in output:
                mtp1 = output["mtp1"]
                mtp_targets = targets[:, 2:].reshape(-1).to(mtp1.device)
                loss_mtp = F.cross_entropy(mtp1.reshape(-1, V), mtp_targets)
                total = total + mtp_weight * loss_mtp
            return total
        else:
            V = output.shape[-1]
            targets = targets.to(output.device)
            return F.cross_entropy(output.view(-1, V), targets.view(-1))

    return mtp_loss_fn
