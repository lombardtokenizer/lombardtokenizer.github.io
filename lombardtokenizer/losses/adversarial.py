"""Adversarial and discriminator losses used by codec training."""

from typing import Sequence

import torch
import torch.nn.functional as F


def generator_adversarial_loss(logits: Sequence[torch.Tensor]) -> torch.Tensor:
    """Least-squares generator loss over all discriminator outputs."""

    if not logits:
        raise ValueError("at least one discriminator logit tensor is required")
    return sum(F.mse_loss(logit, torch.ones_like(logit)) for logit in logits)


def discriminator_loss(
    real_logits: Sequence[torch.Tensor], fake_logits: Sequence[torch.Tensor]
) -> torch.Tensor:
    """Least-squares discriminator loss for real and detached fake audio."""

    if len(real_logits) != len(fake_logits) or not real_logits:
        raise ValueError("real and fake discriminator outputs must be non-empty and aligned")
    return sum(
        0.5
        * (
            F.mse_loss(real, torch.ones_like(real))
            + F.mse_loss(fake, torch.zeros_like(fake))
        )
        for real, fake in zip(real_logits, fake_logits)
    )


def feature_matching_loss(
    real_features: Sequence[Sequence[torch.Tensor]],
    fake_features: Sequence[Sequence[torch.Tensor]],
) -> torch.Tensor:
    """L1 distance between discriminator feature maps, with real maps detached."""

    if len(real_features) != len(fake_features) or not real_features:
        raise ValueError("real and fake feature maps must be non-empty and aligned")
    values = []
    for real_layers, fake_layers in zip(real_features, fake_features):
        if len(real_layers) != len(fake_layers) or not real_layers:
            raise ValueError("real and fake discriminator layers must be aligned")
        values.extend(
            F.l1_loss(fake, real.detach())
            for real, fake in zip(real_layers, fake_layers)
        )
    return torch.stack(values).mean()


__all__ = [
    "discriminator_loss",
    "feature_matching_loss",
    "generator_adversarial_loss",
]
