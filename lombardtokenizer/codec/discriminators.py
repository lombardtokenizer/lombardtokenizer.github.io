"""Small waveform and spectral discriminators for codec training."""

from typing import List, Mapping, NamedTuple, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class DiscriminatorOutput(NamedTuple):
    """Logits and intermediate feature maps for one discriminator family."""

    logits: List[torch.Tensor]
    feature_maps: List[List[torch.Tensor]]


def _validate_waveform(waveform: torch.Tensor) -> None:
    if waveform.ndim != 3 or waveform.shape[1] != 1:
        raise ValueError("discriminators expect mono waveforms with shape (batch, 1, time)")


class _PeriodDiscriminator(nn.Module):
    def __init__(self, period: int, channels: int):
        super().__init__()
        self.period = period
        self.layers = nn.ModuleList(
            [
                nn.Conv2d(1, channels, (5, 1), stride=(3, 1), padding=(2, 0)),
                nn.Conv2d(channels, channels * 2, (5, 1), stride=(3, 1), padding=(2, 0)),
                nn.Conv2d(channels * 2, channels * 4, (5, 1), stride=(3, 1), padding=(2, 0)),
            ]
        )
        self.output = nn.Conv2d(channels * 4, 1, (3, 1), padding=(1, 0))

    def forward(self, waveform: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        length = waveform.shape[-1]
        padding = (self.period - length % self.period) % self.period
        if padding:
            mode = "reflect" if padding < length else "constant"
            waveform = F.pad(waveform, (0, padding), mode=mode)
        batch, channels, length = waveform.shape
        features = waveform.view(batch, channels, length // self.period, self.period)
        feature_maps = []
        for layer in self.layers:
            features = F.leaky_relu(layer(features), negative_slope=0.1)
            feature_maps.append(features)
        return self.output(features).flatten(1), feature_maps


class MultiPeriodDiscriminator(nn.Module):
    """Multi-period waveform discriminator used by the reference trainer."""

    def __init__(
        self,
        periods: Sequence[int] = (2, 3, 5, 7, 11),
        channels: int = 16,
    ):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [_PeriodDiscriminator(period, channels) for period in periods]
        )

    def forward(self, waveform: torch.Tensor) -> DiscriminatorOutput:
        _validate_waveform(waveform)
        outputs = [discriminator(waveform) for discriminator in self.discriminators]
        return DiscriminatorOutput(
            logits=[output[0] for output in outputs],
            feature_maps=[output[1] for output in outputs],
        )


class _ScaleDiscriminator(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Conv1d(1, channels, 15, padding=7),
                nn.Conv1d(channels, channels * 2, 41, stride=4, padding=20),
                nn.Conv1d(channels * 2, channels * 4, 41, stride=4, padding=20),
            ]
        )
        self.output = nn.Conv1d(channels * 4, 1, 3, padding=1)

    def forward(self, waveform: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        features = waveform
        feature_maps = []
        for layer in self.layers:
            features = F.leaky_relu(layer(features), negative_slope=0.1)
            feature_maps.append(features)
        return self.output(features).flatten(1), feature_maps


class MultiScaleDiscriminator(nn.Module):
    """Multi-scale waveform discriminator."""

    def __init__(self, scales: int = 3, channels: int = 16):
        super().__init__()
        if scales <= 0:
            raise ValueError("scales must be positive")
        self.discriminators = nn.ModuleList(
            [_ScaleDiscriminator(channels) for _ in range(scales)]
        )
        self.pools = nn.ModuleList(
            [nn.AvgPool1d(4, stride=2, padding=2) for _ in range(scales - 1)]
        )

    def forward(self, waveform: torch.Tensor) -> DiscriminatorOutput:
        _validate_waveform(waveform)
        current = waveform
        outputs = []
        for index, discriminator in enumerate(self.discriminators):
            outputs.append(discriminator(current))
            if index < len(self.pools):
                current = self.pools[index](current)
        return DiscriminatorOutput(
            logits=[output[0] for output in outputs],
            feature_maps=[output[1] for output in outputs],
        )


class _STFTDiscriminator(nn.Module):
    def __init__(self, n_fft: int, hop_length: int, channels: int):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.layers = nn.ModuleList(
            [
                nn.Conv2d(1, channels, 3, padding=1),
                nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1),
                nn.Conv2d(channels * 2, channels * 4, 3, stride=2, padding=1),
            ]
        )
        self.output = nn.Conv2d(channels * 4, 1, 3, padding=1)

    def forward(self, waveform: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        values = waveform[:, 0, :]
        if values.shape[-1] < self.n_fft:
            values = F.pad(values, (0, self.n_fft - values.shape[-1]))
        window = torch.hann_window(
            self.n_fft, device=values.device, dtype=values.dtype
        )
        spectrum = torch.stft(
            values,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            center=False,
            return_complex=True,
        ).abs().clamp_min(1.0e-5).log().unsqueeze(1)
        feature_maps = []
        for layer in self.layers:
            spectrum = F.leaky_relu(layer(spectrum), negative_slope=0.1)
            feature_maps.append(spectrum)
        return self.output(spectrum).flatten(1), feature_maps


class MultiScaleSTFTDiscriminator(nn.Module):
    """Multi-resolution STFT discriminator."""

    def __init__(
        self,
        n_ffts: Sequence[int] = (1024, 512, 256, 128),
        hop_lengths: Optional[Sequence[int]] = None,
        channels: int = 16,
    ):
        super().__init__()
        if not n_ffts or any(value <= 0 for value in n_ffts):
            raise ValueError("n_ffts must contain positive values")
        if hop_lengths is None:
            hop_lengths = tuple(max(1, value // 4) for value in n_ffts)
        if len(n_ffts) != len(hop_lengths):
            raise ValueError("n_ffts and hop_lengths must have equal lengths")
        self.discriminators = nn.ModuleList(
            [
                _STFTDiscriminator(n_fft, hop, channels)
                for n_fft, hop in zip(n_ffts, hop_lengths)
            ]
        )

    def forward(self, waveform: torch.Tensor) -> DiscriminatorOutput:
        _validate_waveform(waveform)
        outputs = [discriminator(waveform) for discriminator in self.discriminators]
        return DiscriminatorOutput(
            logits=[output[0] for output in outputs],
            feature_maps=[output[1] for output in outputs],
        )


class CodecDiscriminator(nn.Module):
    """The three discriminator families used by the reference codec trainer."""

    def __init__(self, channels: int = 16):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                MultiPeriodDiscriminator(channels=channels),
                MultiScaleDiscriminator(channels=channels),
                MultiScaleSTFTDiscriminator(channels=channels),
            ]
        )

    def forward(self, waveform: torch.Tensor) -> DiscriminatorOutput:
        outputs = [discriminator(waveform) for discriminator in self.discriminators]
        return DiscriminatorOutput(
            logits=[logit for output in outputs for logit in output.logits],
            feature_maps=[maps for output in outputs for maps in output.feature_maps],
        )


def build_discriminator(config: Mapping[str, object]) -> nn.Module:
    """Build the configured discriminator family for codec training."""

    value = config.get("discriminator", {})
    specification = value if isinstance(value, Mapping) else {}
    discriminator_type = str(specification.get("type", "codec"))
    channels = int(specification.get("channels", 16))
    if discriminator_type == "codec":
        return CodecDiscriminator(channels=channels)
    if discriminator_type == "multi_period":
        return MultiPeriodDiscriminator(channels=channels)
    if discriminator_type == "multi_scale":
        return MultiScaleDiscriminator(channels=channels)
    if discriminator_type == "multi_scale_stft":
        n_ffts_value = specification.get("n_ffts", [1024, 512, 256, 128])
        hop_value = specification.get("hop_lengths")
        n_ffts = (
            tuple(int(value) for value in n_ffts_value)
            if isinstance(n_ffts_value, (list, tuple))
            else (1024, 512, 256, 128)
        )
        hop_lengths = (
            tuple(int(value) for value in hop_value)
            if isinstance(hop_value, (list, tuple))
            else None
        )
        return MultiScaleSTFTDiscriminator(
            n_ffts=n_ffts,
            hop_lengths=hop_lengths,
            channels=channels,
        )
    raise ValueError(
        "discriminator.type must be one of: codec, multi_period, "
        "multi_scale, multi_scale_stft"
    )


__all__ = [
    "CodecDiscriminator",
    "DiscriminatorOutput",
    "MultiPeriodDiscriminator",
    "MultiScaleDiscriminator",
    "MultiScaleSTFTDiscriminator",
    "build_discriminator",
]
