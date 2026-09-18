"""Losses used by the LombardTokenizer trainer."""

import torch
import torch.nn.functional as F

from .adversarial import (
    discriminator_loss,
    feature_matching_loss,
    generator_adversarial_loss,
)
from .distillation import semantic_distill_loss, vocal_effort_distill_loss
from .spectral import MelResolution, mel_loss, mel_spectrogram, multi_resolution_mel_loss


def reconstruction_loss(target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    length = min(target.shape[-1], prediction.shape[-1])
    return F.l1_loss(target[..., :length], prediction[..., :length])


def commitment_loss(value: torch.Tensor) -> torch.Tensor:
    return value.mean()


__all__ = [
    "commitment_loss",
    "discriminator_loss",
    "feature_matching_loss",
    "generator_adversarial_loss",
    "mel_loss",
    "mel_spectrogram",
    "multi_resolution_mel_loss",
    "MelResolution",
    "reconstruction_loss",
    "semantic_distill_loss",
    "vocal_effort_distill_loss",
]
