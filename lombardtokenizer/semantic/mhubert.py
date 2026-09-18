"""mHuBERT/HuBERT semantic teacher and offline embedding helpers.

The extractor uses one explicitly selected mHuBERT hidden state or an average
of hidden states. Hugging Face mHuBERT checkpoints are loaded through
``AutoProcessor``/``AutoModel``.
"""

from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import torch
from torch import nn


MHuBERTLayer = Union[int, str]


def _as_batch_mono(waveform: torch.Tensor) -> torch.Tensor:
    """Convert ``(T)``, ``(B,T)`` or ``(B,C,T)`` audio to ``(B,T)``."""

    if waveform.ndim == 1:
        return waveform.unsqueeze(0)
    if waveform.ndim == 2:
        return waveform
    if waveform.ndim == 3:
        return waveform.mean(dim=1)
    raise ValueError("waveform must have shape (T), (B,T), or (B,C,T)")


def select_hidden_state(
    hidden_states: Sequence[torch.Tensor], layer: MHuBERTLayer
) -> torch.Tensor:
    """Select one hidden state or average all hidden states."""

    if isinstance(layer, str):
        if layer != "avg":
            raise ValueError("layer must be an integer or the string 'avg'")
        return torch.stack(tuple(hidden_states), dim=0).mean(dim=0)
    try:
        return hidden_states[layer]
    except IndexError as exc:
        raise ValueError(f"hidden-state layer {layer} is out of range") from exc


def load_precomputed_embeddings(
    path: Union[str, Path], map_location: Union[str, torch.device] = "cpu"
) -> torch.Tensor:
    """Load precomputed ``.npy`` or PyTorch embeddings."""

    path = Path(path)
    if path.suffix.lower() == ".npy":
        value = np.load(path, allow_pickle=False)
        return torch.from_numpy(np.asarray(value))

    value: Any = torch.load(path, map_location=map_location)
    if isinstance(value, dict):
        for key in ("embedding", "embeddings", "features", "hidden_states"):
            if key in value:
                value = value[key]
                break
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    return value


class MHuBERTTeacher(nn.Module):
    """Offline-capable HuBERT teacher with explicit layer selection.

    ``layer='avg'`` averages the returned hidden states. An integer is passed
    directly to ``hidden_states``. The
    layer is required at construction time so an absent Lombard configuration
    cannot silently acquire a scientific default.
    """

    def __init__(
        self,
        model_path: str,
        feature_extractor_path: Optional[str] = None,
        layer: Optional[MHuBERTLayer] = None,
        sample_rate: int = 16000,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        extractor_path = feature_extractor_path or model_path
        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError:
            try:
                from transformers import HubertModel, Wav2Vec2FeatureExtractor
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "MHuBERTTeacher requires the 'transformers' package"
                ) from exc
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                extractor_path
            )
            self.model = HubertModel.from_pretrained(model_path)
        else:
            try:
                self.feature_extractor = AutoProcessor.from_pretrained(
                    extractor_path
                )
                self.model = AutoModel.from_pretrained(model_path)
            except AttributeError:
                from transformers import HubertModel, Wav2Vec2FeatureExtractor

                self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                    extractor_path
                )
                self.model = HubertModel.from_pretrained(model_path)
        self.layer = layer
        self.sample_rate = sample_rate
        self.model.eval()
        if device is not None:
            self.to(device)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def preprocess(
        self, waveform: torch.Tensor, sample_rate: Optional[int] = None
    ) -> torch.Tensor:
        """Prepare waveform batches for the feature extractor."""

        waveform = _as_batch_mono(waveform)
        input_rate = self.sample_rate if sample_rate is None else sample_rate
        if input_rate != self.sample_rate:
            try:
                import torchaudio
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "resampling requires the 'torchaudio' package"
                ) from exc
            waveform = torchaudio.functional.resample(
                waveform, input_rate, self.sample_rate
            )

        encoded = self.feature_extractor(
            waveform.detach().cpu().numpy(),
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        if hasattr(encoded, "input_values"):
            return encoded.input_values
        return encoded["input_values"]

    @torch.inference_mode()
    def extract_hidden_states(
        self, waveform: torch.Tensor, sample_rate: Optional[int] = None
    ) -> Sequence[torch.Tensor]:
        """Return all teacher hidden states without gradient tracking."""

        input_values = self.preprocess(waveform, sample_rate).to(self.device)
        output = self.model(input_values, output_hidden_states=True)
        if output.hidden_states is None:
            raise RuntimeError("HuBERT did not return hidden states")
        return output.hidden_states

    @torch.inference_mode()
    def forward(
        self, waveform: torch.Tensor, sample_rate: Optional[int] = None
    ) -> torch.Tensor:
        """Return the configured semantic teacher representation."""

        hidden_states = self.extract_hidden_states(waveform, sample_rate)
        if self.layer is None:
            raise ValueError(
                "semantic teacher layer is not configured; use an integer or 'avg'"
            )
        return select_hidden_state(hidden_states, self.layer)


def extract_offline_embeddings(
    teacher: MHuBERTTeacher,
    waveform: torch.Tensor,
    output_path: Union[str, Path],
    sample_rate: Optional[int] = None,
) -> torch.Tensor:
    """Extract and save one embedding file in ``.npy`` or ``.pt`` format."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    embedding = teacher(waveform, sample_rate).detach().cpu()
    if output_path.suffix.lower() == ".npy":
        np.save(output_path, embedding.numpy())
    else:
        torch.save(embedding, output_path)
    return embedding
