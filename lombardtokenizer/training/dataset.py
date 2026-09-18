"""Datasets, aligned crops, and target preparation for training."""

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from ..semantic.mhubert import load_precomputed_embeddings
from ..vocal_effort.normalization import SpeakerIntensityNormalizer, normalize_rms


@dataclass
class LombardTokenizerBatch:
    """One training batch and its two mandatory teacher targets."""

    audio: torch.Tensor
    semantic_features: Optional[torch.Tensor] = None
    vocal_effort_features: Optional[torch.Tensor] = None
    speaker_ids: Optional[List[str]] = None


def _segment_start(
    num_samples: int,
    segment_size: Optional[int],
    *,
    training: bool,
    frame_hop: int,
) -> int:
    if segment_size is None:
        return 0
    if segment_size <= 0:
        raise ValueError("segment_size must be positive")
    if frame_hop <= 0:
        raise ValueError("frame_hop must be positive")
    max_start = max(0, num_samples - segment_size)
    if not training or max_start == 0:
        return 0
    max_frame_start = max_start // frame_hop
    return random.randint(0, max_frame_start) * frame_hop


def crop_or_pad_waveform(
    waveform: torch.Tensor,
    segment_size: Optional[int],
    *,
    training: bool,
    frame_hop: int = 320,
) -> Tuple[torch.Tensor, int]:
    """Crop a waveform and right-pad short inputs to ``segment_size``."""

    if waveform.ndim < 1:
        raise ValueError("waveform must have at least one dimension")
    if segment_size is None:
        return waveform, 0
    start = _segment_start(
        waveform.shape[-1],
        segment_size,
        training=training,
        frame_hop=frame_hop,
    )
    segment = waveform[..., start : start + segment_size]
    padding = segment_size - segment.shape[-1]
    if padding > 0:
        segment = torch.nn.functional.pad(segment, (0, padding))
    return segment, start


def crop_or_pad_semantic_features(
    features: torch.Tensor,
    audio_start: int,
    segment_size: Optional[int],
    *,
    frame_hop: int = 320,
) -> torch.Tensor:
    """Crop semantic features using the audio crop's aligned frame offset."""

    if features.ndim != 2:
        raise ValueError(
            "semantic features must have shape (time, dimension), "
            f"got {tuple(features.shape)}"
        )
    if audio_start < 0:
        raise ValueError("audio_start must be non-negative")
    if segment_size is None:
        return features
    if frame_hop <= 0:
        raise ValueError("frame_hop must be positive")
    semantic_start = audio_start // frame_hop
    num_frames = math.ceil(segment_size / frame_hop)
    aligned = features.new_zeros((num_frames, features.shape[-1]))
    source = features[semantic_start : semantic_start + num_frames]
    aligned[: source.shape[0]] = source
    return aligned


def _vocal_effort_input(waveform: torch.Tensor) -> torch.Tensor:
    """Build batched normalized spectrogram inputs expected by LT2."""

    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 3 or waveform.shape[1] != 1:
        raise ValueError("waveform must have shape (batch, 1, time) or (1, time)")
    waveform = normalize_rms(waveform)
    n_fft = min(1024, max(16, waveform.shape[-1]))
    if waveform.shape[-1] < n_fft:
        # ``torch.stft(center=True)`` uses reflect padding, which is invalid
        # when the input is shorter than the requested FFT size.  Pad after
        # RMS normalization so short signals and silence remain deterministic.
        waveform = F.pad(waveform, (0, n_fft - waveform.shape[-1]))
    features = torch.stft(
        waveform[:, 0, :],
        n_fft=n_fft,
        hop_length=max(1, n_fft // 4),
        window=torch.hann_window(n_fft, device=waveform.device, dtype=waveform.dtype),
        return_complex=True,
    ).abs()
    return features.unsqueeze(1)


def freeze_vocal_effort_encoder(encoder: nn.Module) -> nn.Module:
    """Freeze and switch E2 to evaluation mode before target extraction."""

    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder


class ManifestAudioDataset(Dataset):
    """TSV dataset with aligned audio and semantic features.

    Each row contains ``audio_path<TAB>semantic_feature_path``. The semantic
    feature file must contain ``(time, 768)`` features. Audio crops are aligned
    to semantic frames using the nominal 320-sample downsampling factor. E2
    inference is intentionally owned by the trainer, not DataLoader workers.
    """

    def __init__(
        self,
        manifest: str,
        sample_rate: int = 16000,
        semantic_dimension: int = 768,
        segment_size: Optional[int] = None,
        training: bool = True,
        frame_hop: int = 320,
    ):
        self.manifest = Path(manifest)
        self.sample_rate = sample_rate
        self.semantic_dimension = int(semantic_dimension)
        self.segment_size = segment_size
        self.training = training
        self.frame_hop = int(frame_hop)
        self.records: List[Tuple[Path, Path]] = []
        with self.manifest.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                fields = line.strip().split("\t")
                if not fields or not fields[0]:
                    continue
                if len(fields) < 2 or not fields[1]:
                    raise ValueError(
                        f"manifest row {line_number} requires "
                        "audio_path and semantic_feature_path"
                    )
                self.records.append(
                    (self._resolve(fields[0]), self._resolve(fields[1]))
                )

    def _resolve(self, value: str) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else self.manifest.parent / path

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> LombardTokenizerBatch:
        import torchaudio

        audio_path, semantic_path = self.records[index]
        audio, sample_rate = torchaudio.load(str(audio_path))
        if sample_rate != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sample_rate, self.sample_rate)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        audio, audio_start = crop_or_pad_waveform(
            audio,
            self.segment_size,
            training=self.training,
            frame_hop=self.frame_hop,
        )

        semantic_features = load_precomputed_embeddings(semantic_path).float()
        if semantic_features.ndim == 3 and semantic_features.shape[0] == 1:
            semantic_features = semantic_features.squeeze(0)
        if semantic_features.ndim != 2:
            raise ValueError(
                "semantic target must have shape (time, dimension), "
                f"got {tuple(semantic_features.shape)}"
            )
        if semantic_features.shape[-1] != self.semantic_dimension:
            raise ValueError(
                f"semantic: expected {self.semantic_dimension} dimensions, "
                f"got {semantic_features.shape[-1]}"
            )
        semantic_features = crop_or_pad_semantic_features(
            semantic_features,
            audio_start,
            self.segment_size,
            frame_hop=self.frame_hop,
        )

        return LombardTokenizerBatch(
            audio=audio,
            semantic_features=semantic_features,
            vocal_effort_features=None,
        )


class VocalEffortManifestDataset(Dataset):
    """LT2 dataset with shared speaker-wise target statistics."""

    def __init__(
        self,
        manifest: str,
        sample_rate: int = 16000,
        segment_size: Optional[int] = None,
        training: bool = True,
        frame_hop: int = 320,
        normalizer: Optional[SpeakerIntensityNormalizer] = None,
    ):
        self.manifest = Path(manifest)
        self.sample_rate = sample_rate
        self.segment_size = segment_size
        self.training = training
        self.frame_hop = int(frame_hop)
        self.records: List[Tuple[Path, str, float]] = []
        with self.manifest.open(encoding="utf-8") as stream:
            for line in stream:
                fields = line.strip().split("\t")
                if len(fields) < 3:
                    raise ValueError("LT2 manifest rows require audio, speaker_id, and intensity")
                path = Path(fields[0]).expanduser()
                if not path.is_absolute():
                    path = self.manifest.parent / path
                self.records.append((path, fields[1], float(fields[2])))
        if normalizer is None:
            normalizer = SpeakerIntensityNormalizer().fit(
                (record[1] for record in self.records),
                (record[2] for record in self.records),
            )
        self.normalizer = normalizer

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        import torchaudio

        path, speaker_id, intensity = self.records[index]
        waveform, rate = torchaudio.load(str(path))
        if rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, rate, self.sample_rate)
        waveform = waveform.mean(dim=0, keepdim=True)
        waveform, _ = crop_or_pad_waveform(
            waveform,
            self.segment_size,
            training=self.training,
            frame_hop=self.frame_hop,
        )
        features = _vocal_effort_input(waveform)
        target = torch.tensor(
            [self.normalizer.normalize(speaker_id, intensity)],
            dtype=features.dtype,
        )
        return features, target


__all__ = [
    "LombardTokenizerBatch",
    "ManifestAudioDataset",
    "VocalEffortManifestDataset",
    "crop_or_pad_semantic_features",
    "crop_or_pad_waveform",
    "freeze_vocal_effort_encoder",
]
