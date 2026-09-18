"""Vocal-effort teacher utilities."""

from .intensity_encoder import (
    IntensityEncoder,
    expand_global_vocal_embedding,
)
from .normalization import SpeakerIntensityNormalizer, normalize_rms

__all__ = [
    "IntensityEncoder",
    "SpeakerIntensityNormalizer",
    "expand_global_vocal_embedding",
    "normalize_rms",
]
