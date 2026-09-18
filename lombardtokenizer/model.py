"""The public LombardTokenizer model.

Only the codec primitives in :mod:`lombardtokenizer.codec` are used here.  The
semantic and vocal-effort roles are deliberately represented by named RVQ
constants and projections so that conversion cannot accidentally swap another
layer.
"""

import json
from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence, Union

import numpy as np
import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from .checkpoint import load_checkpoint
from .codec import ResidualVectorQuantizer, SEANetDecoder, SEANetEncoder


SEMANTIC_RVQ_LAYER = 0
VOCAL_EFFORT_RVQ_LAYER = 1
VQ1 = SEMANTIC_RVQ_LAYER
VQ2 = VOCAL_EFFORT_RVQ_LAYER


@dataclass
class LombardTokenizerOutput:
    """Named outputs of one LombardTokenizer forward pass."""

    reconstructed_audio: torch.Tensor
    commitment_loss: torch.Tensor
    semantic_features: torch.Tensor
    vocal_effort_features: torch.Tensor
    codes: torch.Tensor
    quantized_layers: List[torch.Tensor]

def _model_config(config: Mapping[str, object]) -> Mapping[str, object]:
    nested = config.get("model")
    if isinstance(nested, Mapping):
        return nested
    return config


def _required(config: Mapping[str, object], key: str) -> object:
    if key not in config or config[key] is None:
        raise KeyError(f"model configuration is missing '{key}'")
    return config[key]


class LombardTokenizer(nn.Module):
    """16-kHz LombardTokenizer codec with eight residual quantizers."""

    VQ1 = VQ1
    VQ2 = VQ2
    SEMANTIC_RVQ_LAYER = SEMANTIC_RVQ_LAYER
    VOCAL_EFFORT_RVQ_LAYER = VOCAL_EFFORT_RVQ_LAYER

    def __init__(self, config: Mapping[str, object]):
        super().__init__()
        self.config = dict(config)
        model = _model_config(config)
        semantic = config.get("semantic", {})
        vocal_effort = config.get("vocal_effort", {})
        semantic = semantic if isinstance(semantic, Mapping) else {}
        vocal_effort = vocal_effort if isinstance(vocal_effort, Mapping) else {}

        self.codec_dimension = int(_required(model, "dimension"))
        self.semantic_dimension = int(
            config.get("semantic_dimension", semantic.get("dimension", 768))
        )
        self.vocal_effort_dimension = int(
            config.get(
                "vocal_effort_dimension",
                vocal_effort.get("dimension", 32),
            )
        )
        self.sample_rate = int(_required(model, "sample_rate"))
        self.n_q = int(_required(model, "num_quantizers"))
        self.codebook_size = int(_required(model, "codebook_size"))
        self.semantic_rvq_layer = int(
            model.get("semantic_rvq_layer", SEMANTIC_RVQ_LAYER)
        )
        self.vocal_effort_rvq_layer = int(
            model.get("vocal_effort_rvq_layer", VOCAL_EFFORT_RVQ_LAYER)
        )
        if (self.semantic_rvq_layer, self.vocal_effort_rvq_layer) != (VQ1, VQ2):
            raise ValueError("LombardTokenizer requires semantic RVQ layer 0 and vocal-effort RVQ layer 1")

        strides = [int(stride) for stride in model.get("strides", [8, 5, 4, 2])]
        if not strides or any(stride <= 0 for stride in strides):
            raise ValueError("model.strides must contain positive integers")
        self.downsample_rate = int(np.prod(strides))
        self.frame_hop = self.downsample_rate
        common = {
            "n_filters": int(model.get("n_filters", 64)),
            "dimension": self.codec_dimension,
            "ratios": strides,
            "lstm": int(model.get("lstm_layers", 2)),
            "bidirectional": bool(model.get("bidirectional", True)),
            "dilation_base": int(model.get("dilation_base", 2)),
            "residual_kernel_size": int(model.get("residual_kernel_size", 3)),
            "n_residual_layers": int(model.get("n_residual_layers", 1)),
            "activation": str(model.get("activation", "ELU")),
        }
        self.encoder = SEANetEncoder(**common)
        self.quantizer = ResidualVectorQuantizer(
            dimension=self.codec_dimension,
            n_q=self.n_q,
            bins=self.codebook_size,
        )
        decoder_config = dict(common)
        decoder_config["bidirectional"] = False
        self.decoder = SEANetDecoder(**decoder_config)
        self.semantic_projection = self._make_projection(
            self.codec_dimension, self.semantic_dimension
        )
        self.vocal_effort_projection = self._make_projection(
            self.codec_dimension, self.vocal_effort_dimension
        )

    @staticmethod
    def _make_projection(input_dimension: int, output_dimension: int) -> nn.Module:
        if input_dimension == output_dimension:
            return nn.Identity()
        return nn.Linear(input_dimension, output_dimension)

    @classmethod
    def load_from_checkpoint(
        cls,
        *args: str,
        config_path: Optional[str] = None,
        ckpt_path: Optional[str] = None,
        map_location: Union[str, torch.device] = "cpu",
    ) -> "LombardTokenizer":
        """Load a model from its checkpoint configuration."""

        if len(args) > 1:
            raise TypeError("load_from_checkpoint accepts one checkpoint path")
        if args:
            if ckpt_path is not None:
                raise TypeError("checkpoint path was provided more than once")
            ckpt_path = args[0]
        if ckpt_path is None:
            raise TypeError("checkpoint path is required")

        payload = load_checkpoint(ckpt_path, map_location=map_location)
        if config_path is not None:
            if str(config_path).lower().endswith((".yaml", ".yml")):
                from .config import load_yaml_config

                yaml_config, _ = load_yaml_config(config_path)
                config = yaml_config
            else:
                with open(config_path, encoding="utf-8") as stream:
                    config = json.load(stream)
        elif isinstance(payload, Mapping) and isinstance(payload.get("config"), Mapping):
            config = payload["config"]
        else:
            raise ValueError("config_path is required when the checkpoint has no embedded config")
        model = cls(config)
        model.load_state_dict(payload["model"])
        return model

    def forward(
        self,
        x: torch.Tensor,
        n_q: Optional[int] = None,
        layers: Optional[Sequence[int]] = None,
    ) -> LombardTokenizerOutput:
        number_of_quantizers = self.n_q if n_q is None else int(n_q)
        selected_layers = list(range(number_of_quantizers)) if layers is None else sorted(set(layers))
        if VQ1 not in selected_layers or VQ2 not in selected_layers:
            raise ValueError("forward requires both semantic RVQ layer 0 and vocal-effort RVQ layer 1")
        if number_of_quantizers <= VQ2 or number_of_quantizers > self.n_q:
            raise ValueError(f"forward requires between 2 and {self.n_q} RVQ layers")

        encoded = self.encoder(x)
        quantized, codes, commitment_loss, quantized_list, _ = self.quantizer(
            encoded, n_q=number_of_quantizers, layers=selected_layers
        )
        if len(quantized_list) < 2:
            raise RuntimeError("RVQ did not return both semantic and vocal-effort layers")
        semantic_features = self.semantic_projection(
            rearrange(quantized_list[0], "b d t -> b t d")
        )
        vocal_effort_features = self.vocal_effort_projection(
            rearrange(quantized_list[1], "b d t -> b t d")
        )
        return LombardTokenizerOutput(
            reconstructed_audio=self._match_audio_length(
                self.decoder(quantized), x.shape[-1]
            ),
            commitment_loss=commitment_loss,
            semantic_features=semantic_features,
            vocal_effort_features=vocal_effort_features,
            codes=codes,
            quantized_layers=list(quantized_list),
        )

    def encode(self, audio: torch.Tensor, n_q: Optional[int] = None) -> torch.Tensor:
        """Encode audio into ``(quantizers, batch, frames)`` code indices."""

        return self.quantizer.encode(self.encoder(audio), n_q=n_q)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode RVQ code indices into waveform audio."""

        return self.decoder(self.quantizer.decode(codes))

    @staticmethod
    def _match_audio_length(audio: torch.Tensor, length: int) -> torch.Tensor:
        if audio.shape[-1] >= length:
            return audio[..., :length]
        return F.pad(audio, (0, length - audio.shape[-1]))

    def reconstruct(self, audio: torch.Tensor) -> torch.Tensor:
        """Reconstruct audio using all RVQ layers at the input length."""

        return self._match_audio_length(self.decode(self.encode(audio)), audio.shape[-1])

    @staticmethod
    def swap_rvq_layer(
        source_codes: torch.Tensor, reference_codes: torch.Tensor, layer: int
    ) -> torch.Tensor:
        if source_codes.ndim != 3 or reference_codes.ndim != 3:
            raise ValueError("RVQ codes must have shape (n_q, batch, frames)")
        if source_codes.shape != reference_codes.shape:
            raise ValueError(
                "source and reference RVQ codes must have identical shapes; "
                f"got {tuple(source_codes.shape)} and {tuple(reference_codes.shape)}"
            )
        if layer < 0 or layer >= source_codes.shape[0]:
            raise ValueError(f"RVQ layer {layer} is outside [0, {source_codes.shape[0] - 1}]")
        converted_codes = source_codes.clone()
        converted_codes[layer] = reference_codes[layer]
        return converted_codes

    @staticmethod
    def swap_vocal_effort(
        source_codes: torch.Tensor, reference_codes: torch.Tensor
    ) -> torch.Tensor:
        """Replace only VQ2; VQ1 and VQ3--VQ8 remain from the source."""

        return LombardTokenizer.swap_rvq_layer(source_codes, reference_codes, VQ2)

    @torch.no_grad()
    def convert_vocal_effort(
        self, source: torch.Tensor, reference: torch.Tensor
    ) -> torch.Tensor:
        source_codes = self.encode(source)
        reference_codes = self.encode(reference)
        converted = self.decode(self.swap_vocal_effort(source_codes, reference_codes))
        return self._match_audio_length(converted, source.shape[-1])


__all__ = [
    "LombardTokenizer",
    "LombardTokenizerOutput",
    "SEMANTIC_RVQ_LAYER",
    "VOCAL_EFFORT_RVQ_LAYER",
    "VQ1",
    "VQ2",
]
