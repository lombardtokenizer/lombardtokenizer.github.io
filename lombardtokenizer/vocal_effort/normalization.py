"""Audio and target normalization for the vocal-effort pipeline."""

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple, Union

import torch


def normalize_rms(waveform: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize every waveform to RMS one, leaving silence unchanged."""

    if waveform.ndim == 0:
        raise ValueError("waveform must have at least one dimension")
    dimensions = tuple(range(1, waveform.ndim)) if waveform.ndim > 1 else (0,)
    rms = waveform.square().mean(dim=dimensions, keepdim=True).sqrt()
    return torch.where(rms > eps, waveform / rms.clamp_min(eps), waveform)


class SpeakerIntensityNormalizer:
    """Speaker-wise Min-Max normalization for LT2 targets."""

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self._statistics: Dict[str, Tuple[float, float]] = {}

    @property
    def statistics(self) -> Mapping[str, Tuple[float, float]]:
        return dict(self._statistics)

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        """Return JSON-serializable speaker-wise Min-Max statistics."""

        return {
            speaker: {"minimum": minimum, "maximum": maximum}
            for speaker, (minimum, maximum) in self._statistics.items()
        }

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Mapping[str, float]], eps: float = 1e-8
    ) -> "SpeakerIntensityNormalizer":
        normalizer = cls(eps=eps)
        normalizer._statistics = {
            str(speaker): (
                float(values["minimum"]),
                float(values["maximum"]),
            )
            for speaker, values in value.items()
        }
        if not normalizer._statistics:
            raise ValueError("speaker statistics cannot be empty")
        return normalizer

    def save(self, path: Union[str, Path]) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "SpeakerIntensityNormalizer":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError("speaker statistics must contain a JSON object")
        return cls.from_dict(value)

    def fit(
        self, speaker_ids: Iterable[str], intensities: Iterable[float]
    ) -> "SpeakerIntensityNormalizer":
        grouped: Dict[str, List[float]] = {}
        for speaker_id, intensity in zip(speaker_ids, intensities):
            grouped.setdefault(str(speaker_id), []).append(float(intensity))
        if not grouped:
            raise ValueError("at least one speaker intensity is required")
        self._statistics = {
            speaker: (min(values), max(values))
            for speaker, values in grouped.items()
        }
        return self

    def fit_records(
        self, records: Iterable[Mapping[str, object]]
    ) -> "SpeakerIntensityNormalizer":
        return self.fit(
            (str(record["speaker_id"]) for record in records),
            (float(record["intensity"]) for record in records),
        )

    def normalize(self, speaker_id: str, intensity: float) -> float:
        if str(speaker_id) not in self._statistics:
            raise KeyError(f"no intensity statistics fitted for speaker {speaker_id!r}")
        minimum, maximum = self._statistics[str(speaker_id)]
        return (float(intensity) - minimum) / max(maximum - minimum, self.eps)

    def normalize_tensor(self, speaker_id: str, intensity: torch.Tensor) -> torch.Tensor:
        if str(speaker_id) not in self._statistics:
            raise KeyError(f"no intensity statistics fitted for speaker {speaker_id!r}")
        minimum, maximum = self._statistics[str(speaker_id)]
        return (intensity - minimum) / max(maximum - minimum, self.eps)


__all__ = ["SpeakerIntensityNormalizer", "normalize_rms"]
