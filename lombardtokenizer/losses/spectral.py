"""Differentiable mel-spectrogram losses for codec reconstruction."""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MelResolution:
    """One resolution used by the multi-resolution mel loss."""

    n_fft: int
    hop_length: int
    win_length: int
    n_mels: int = 80
    f_min: float = 0.0
    f_max: Optional[float] = None

    def __post_init__(self) -> None:
        if self.n_fft <= 0:
            raise ValueError("mel n_fft must be positive")
        if self.hop_length <= 0:
            raise ValueError("mel hop_length must be positive")
        if self.win_length <= 0 or self.win_length > self.n_fft:
            raise ValueError("mel win_length must be in (0, n_fft]")
        if self.n_mels <= 0:
            raise ValueError("mel n_mels must be positive")
        if self.f_min < 0:
            raise ValueError("mel f_min must be non-negative")
        if self.f_max is not None and self.f_max <= self.f_min:
            raise ValueError("mel f_max must be greater than f_min")


_FILTERBANK_CACHE: Dict[Tuple[object, ...], torch.Tensor] = {}


def _hz_to_mel(value: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + value / 700.0)


def _mel_filterbank(
    resolution: MelResolution,
    sample_rate: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    f_max = float(sample_rate / 2 if resolution.f_max is None else resolution.f_max)
    if f_max > sample_rate / 2:
        raise ValueError("mel f_max cannot exceed the Nyquist frequency")
    key = (
        resolution.n_fft,
        resolution.n_mels,
        resolution.f_min,
        f_max,
        sample_rate,
        str(device),
        str(dtype),
    )
    cached = _FILTERBANK_CACHE.get(key)
    if cached is not None:
        return cached

    if resolution.f_min < 0 or resolution.f_min >= f_max:
        raise ValueError("mel f_min must be non-negative and smaller than f_max")
    mel_min = _hz_to_mel(torch.tensor(resolution.f_min, dtype=torch.float32))
    mel_max = _hz_to_mel(torch.tensor(f_max, dtype=torch.float32))
    mel_points = torch.linspace(
        float(mel_min), float(mel_max), resolution.n_mels + 2
    )
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    bins = torch.floor((resolution.n_fft + 1) * hz_points / sample_rate).long()
    filterbank = torch.zeros(
        resolution.n_mels,
        resolution.n_fft // 2 + 1,
        dtype=dtype,
        device=device,
    )
    for index in range(resolution.n_mels):
        left = int(bins[index].item())
        center = int(bins[index + 1].item())
        right = int(bins[index + 2].item())
        left = max(0, min(left, filterbank.shape[1] - 1))
        center = max(0, min(center, filterbank.shape[1] - 1))
        right = max(0, min(right, filterbank.shape[1] - 1))
        if center > left:
            filterbank[index, left:center] = torch.arange(
                center - left, device=device, dtype=dtype
            ) / float(center - left)
        if right > center:
            filterbank[index, center:right] = torch.arange(
                right - center, 0, -1, device=device, dtype=dtype
            ) / float(right - center)
    empty_rows = filterbank.abs().sum(dim=1).eq(0).sum().item()
    if empty_rows:
        raise ValueError(
            "mel filterbank contains empty bands; reduce n_mels or increase n_fft"
        )
    _FILTERBANK_CACHE[key] = filterbank
    return filterbank


def mel_spectrogram(
    waveform: torch.Tensor,
    *,
    sample_rate: int,
    resolution: MelResolution,
) -> torch.Tensor:
    """Return log mel magnitudes with shape ``(batch, n_mels, time)``."""

    if waveform.ndim == 3:
        if waveform.shape[1] != 1:
            raise ValueError("mel_spectrogram expects mono waveforms")
        waveform = waveform[:, 0, :]
    elif waveform.ndim != 2:
        raise ValueError("waveform must have shape (batch, time) or (batch, 1, time)")
    if waveform.shape[-1] < resolution.n_fft:
        waveform = F.pad(waveform, (0, resolution.n_fft - waveform.shape[-1]))
    window = torch.hann_window(
        resolution.win_length,
        device=waveform.device,
        dtype=waveform.dtype,
    )
    spectrum = torch.stft(
        waveform,
        n_fft=resolution.n_fft,
        hop_length=resolution.hop_length,
        win_length=resolution.win_length,
        window=window,
        center=False,
        return_complex=True,
    ).abs()
    filterbank = _mel_filterbank(
        resolution,
        sample_rate,
        device=waveform.device,
        dtype=waveform.dtype,
    )
    mel = torch.einsum("mf,bft->bmt", filterbank, spectrum)
    return torch.log(mel.clamp_min(1.0e-5))


def mel_loss(
    target: torch.Tensor,
    prediction: torch.Tensor,
    *,
    sample_rate: int,
    resolution: MelResolution,
) -> torch.Tensor:
    """L1 distance between log mel-spectrograms at one resolution."""

    target_mel = mel_spectrogram(
        target, sample_rate=sample_rate, resolution=resolution
    )
    prediction_mel = mel_spectrogram(
        prediction, sample_rate=sample_rate, resolution=resolution
    )
    length = min(target_mel.shape[-1], prediction_mel.shape[-1])
    return F.l1_loss(target_mel[..., :length], prediction_mel[..., :length])


def multi_resolution_mel_loss(
    target: torch.Tensor,
    prediction: torch.Tensor,
    *,
    sample_rate: int,
    resolutions: Sequence[MelResolution],
    weights: Sequence[float],
) -> torch.Tensor:
    """Weighted sum of explicit mel resolutions."""

    if len(resolutions) != len(weights):
        raise ValueError("mel resolutions and mel weights must have equal lengths")
    if not resolutions:
        return target.new_zeros(())
    values = [
        mel_loss(
            target,
            prediction,
            sample_rate=sample_rate,
            resolution=resolution,
        )
        for resolution in resolutions
    ]
    return torch.stack(values).mul(target.new_tensor(list(weights))).sum()


__all__ = [
    "MelResolution",
    "mel_loss",
    "mel_spectrogram",
    "multi_resolution_mel_loss",
]
