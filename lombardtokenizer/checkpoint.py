"""Versioned checkpoint helpers shared by training and inference."""

from pathlib import Path
from typing import Any, Mapping, Optional, Union

import torch


CHECKPOINT_VERSION = 2


def make_checkpoint(
    model: Mapping[str, Any],
    optimizer: Optional[Mapping[str, Any]] = None,
    scheduler: Optional[Mapping[str, Any]] = None,
    epoch: int = 0,
    step: int = 0,
    config: Optional[Mapping[str, Any]] = None,
    discriminator: Optional[Mapping[str, Any]] = None,
    discriminator_optimizer: Optional[Mapping[str, Any]] = None,
    discriminator_scheduler: Optional[Mapping[str, Any]] = None,
    speaker_statistics: Optional[Mapping[str, Any]] = None,
    best_validation_loss: Optional[float] = None,
) -> dict:
    """Create the complete checkpoint envelope used by all workflows."""

    payload = {
        "version": CHECKPOINT_VERSION,
        "checkpoint_version": CHECKPOINT_VERSION,
        "model": dict(model),
        "optimizer": dict(optimizer or {}),
        "scheduler": dict(scheduler or {}),
        "epoch": int(epoch),
        "step": int(step),
        "config": dict(config or {}),
        "best_validation_loss": (
            None if best_validation_loss is None else float(best_validation_loss)
        ),
    }
    if discriminator is not None:
        payload["discriminator"] = dict(discriminator)
    if discriminator_optimizer is not None:
        payload["discriminator_optimizer"] = dict(discriminator_optimizer)
    if discriminator_scheduler is not None:
        payload["discriminator_scheduler"] = dict(discriminator_scheduler)
    if speaker_statistics is not None:
        payload["speaker_statistics"] = dict(speaker_statistics)
    return payload


def save_checkpoint(path: Union[str, Path], payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(payload), destination)


def load_checkpoint(
    path: Union[str, Path], map_location: Union[str, torch.device] = "cpu"
) -> dict:
    payload = torch.load(path, map_location=map_location)
    required = {
        "version",
        "checkpoint_version",
        "model",
        "optimizer",
        "scheduler",
        "epoch",
        "step",
        "config",
    }
    if not isinstance(payload, Mapping) or not required.issubset(payload):
        raise ValueError("checkpoint does not use the LombardTokenizer format")
    if (
        int(payload["version"]) != CHECKPOINT_VERSION
        or int(payload["checkpoint_version"]) != CHECKPOINT_VERSION
    ):
        raise ValueError(
            "unsupported LombardTokenizer checkpoint version: "
            f"{payload['checkpoint_version']} (expected {CHECKPOINT_VERSION})"
        )
    return dict(payload)


def model_state_from_checkpoint(
    path: Union[str, Path], map_location: Union[str, torch.device] = "cpu"
) -> Mapping[str, Any]:
    return load_checkpoint(path, map_location)["model"]


__all__ = [
    "CHECKPOINT_VERSION",
    "load_checkpoint",
    "make_checkpoint",
    "model_state_from_checkpoint",
    "save_checkpoint",
]
