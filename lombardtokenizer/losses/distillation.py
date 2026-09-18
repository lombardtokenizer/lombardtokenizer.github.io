"""Semantic and vocal-effort distillation losses.

Distillation features use the common ``(batch, time, dimension)`` layout.
For ``d_axis`` distillation, cosine similarity is computed over time for each
feature dimension, matching the SpeechTokenizer/LombardTokenizer objective.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def _align_time(
    feature: torch.Tensor,
    target: torch.Tensor,
    *,
    name: str,
    expected_dimension: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Align only time; feature dimensions must match exactly."""

    if feature.ndim != 3 or target.ndim != 3:
        raise ValueError(
            f"{name} distillation expects student and teacher features with "
            "shape (batch, time, dimension)"
        )
    if feature.shape[0] != target.shape[0]:
        raise ValueError(
            f"{name} batch dimensions must match exactly: "
            f"student={feature.shape[0]}, teacher={target.shape[0]}"
        )
    if feature.shape[-1] != target.shape[-1]:
        expected = (
            f"; expected {expected_dimension}" if expected_dimension is not None else ""
        )
        raise ValueError(
            f"{name} feature dimensions must match exactly: "
            f"student={feature.shape[-1]}, teacher={target.shape[-1]}{expected}"
        )
    if expected_dimension is not None and feature.shape[-1] != expected_dimension:
        raise ValueError(
            f"{name}: expected {expected_dimension} feature dimensions, "
            f"got {feature.shape[-1]}"
        )
    length = min(feature.shape[1], target.shape[1])
    if length == 0:
        raise ValueError(f"{name} distillation cannot align empty time dimensions")
    return feature[:, :length, :], target[:, :length, :]


def _cosine_distillation(
    feature: torch.Tensor,
    target_feature: torch.Tensor,
    *,
    cosine_dimension: int,
    name: str,
    expected_dimension: Optional[int] = None,
) -> torch.Tensor:
    feature, target_feature = _align_time(
        feature,
        target_feature,
        name=name,
        expected_dimension=expected_dimension,
    )
    similarity = F.cosine_similarity(feature, target_feature, dim=cosine_dimension)
    tiny = torch.finfo(similarity.dtype).tiny
    return -torch.log(torch.sigmoid(similarity).clamp_min(tiny)).mean()


def semantic_distill_loss(
    feature: torch.Tensor,
    target_feature: torch.Tensor,
    distill_type: str = "d_axis",
    lambda_sim: float = 1.0,
    expected_dimension: Optional[int] = None,
) -> torch.Tensor:
    """Distill mHuBERT features into VQ1.

    ``d_axis`` computes cosine similarity over time (``dim=1``), producing one
    similarity value per semantic dimension before averaging the loss.
    """

    if distill_type == "d_axis":
        return _cosine_distillation(
            feature,
            target_feature,
            cosine_dimension=1,
            name="semantic",
            expected_dimension=expected_dimension,
        )
    if distill_type == "t_axis":
        aligned_feature, aligned_target = _align_time(
            feature,
            target_feature,
            name="semantic",
            expected_dimension=expected_dimension,
        )
        return F.l1_loss(aligned_feature, aligned_target) + lambda_sim * _cosine_distillation(
            aligned_feature,
            aligned_target,
            cosine_dimension=-1,
            name="semantic",
            expected_dimension=expected_dimension,
        )
    raise ValueError("distill_type must be 'd_axis' or 't_axis'")


def vocal_effort_distill_loss(
    feature: torch.Tensor,
    target_feature: torch.Tensor,
    distill_type: str = "d_axis",
    lambda_sim: float = 1.0,
    expected_dimension: Optional[int] = None,
) -> torch.Tensor:
    """Distill the 32-D LT2 embedding into VQ2."""

    if distill_type == "d_axis":
        return _cosine_distillation(
            feature,
            target_feature,
            cosine_dimension=1,
            name="LT2",
            expected_dimension=expected_dimension,
        )
    if distill_type == "t_axis":
        aligned_feature, aligned_target = _align_time(
            feature,
            target_feature,
            name="LT2",
            expected_dimension=expected_dimension,
        )
        return F.l1_loss(aligned_feature, aligned_target) + lambda_sim * _cosine_distillation(
            aligned_feature,
            aligned_target,
            cosine_dimension=-1,
            name="LT2",
            expected_dimension=expected_dimension,
        )
    raise ValueError("distill_type must be 'd_axis' or 't_axis'")


__all__ = ["semantic_distill_loss", "vocal_effort_distill_loss"]
