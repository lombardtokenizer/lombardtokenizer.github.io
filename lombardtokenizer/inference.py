"""Inference helpers and audio I/O for user-facing workflows."""

from pathlib import Path
from typing import Optional, Union

import torch

from . import LombardTokenizer


def load_tokenizer(
    checkpoint: str,
    config: Optional[str] = None,
    device: Union[str, torch.device] = "cpu",
) -> LombardTokenizer:
    """Load a tokenizer from an embedded or optional external config."""

    if config is None:
        model = LombardTokenizer.load_from_checkpoint(
            checkpoint,
            map_location=device,
        )
    else:
        model = LombardTokenizer.load_from_checkpoint(
            config_path=config,
            ckpt_path=checkpoint,
            map_location=device,
        )
    return model.to(device).eval()


def load_audio(
    path: Union[str, Path],
    sample_rate: int,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """Load WAV/FLAC audio as mono ``(1, 1, time)`` at ``sample_rate``."""

    import torchaudio

    waveform, source_rate = torchaudio.load(str(path))
    if source_rate != sample_rate:
        waveform = torchaudio.functional.resample(
            waveform, source_rate, sample_rate
        )
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.unsqueeze(0)
    return waveform if device is None else waveform.to(device)


def save_audio(
    path: Union[str, Path],
    waveform: torch.Tensor,
    sample_rate: int,
) -> None:
    """Save a waveform as mono audio, accepting ``(B,C,T)`` or ``(C,T)``."""

    import torchaudio

    if waveform.ndim == 3:
        if waveform.shape[0] == 0:
            raise ValueError("cannot save an empty audio batch")
        waveform = waveform[0]
    elif waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 2:
        raise ValueError("waveform must have shape (T), (C,T), or (B,C,T)")
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(destination), waveform.detach().cpu(), int(sample_rate))


__all__ = ["load_audio", "load_tokenizer", "save_audio"]
