"""Self-contained LT2 vocal-effort encoder.

LT2 is the paper's three-CNN, LSTM, FC1, FC2 intensity predictor. FC1 is the
32-dimensional embedding distilled into VQ2; FC2 predicts the normalized
speaker-wise target.
"""

import json
from pathlib import Path
from typing import Mapping, Optional, Union

import torch
from torch import nn

from ..checkpoint import load_checkpoint


class IntensityEncoder(nn.Module):
    """Three-CNN + LSTM + FC1/FC2 LT2 predictor."""

    def __init__(self, config: Mapping[str, object]):
        super().__init__()
        cfg = dict(config)
        configured_dimension = cfg.get("dimension")
        model = cfg.get("model")
        if isinstance(model, Mapping):
            cfg = dict(model)
        if configured_dimension is None:
            configured_dimension = cfg.get("embedding_dim", 32)

        channels = [int(value) for value in cfg.get("channels", [16, 32, 64])]
        kernels = cfg.get("kernels", [[3, 3]] * 3)
        strides = cfg.get("strides", [[1, 1]] * 3)
        paddings = cfg.get("paddings", [[1, 1]] * 3)
        if len(channels) != 3:
            raise ValueError("LT2 requires exactly three CNN layers")
        pools = cfg.get("pools", [2, 2])
        if isinstance(pools[0], (list, tuple)):
            pool_kernel = tuple(int(value) for value in pools[0])
        else:
            pool_kernel = tuple(int(value) for value in pools)
        activation_name = str(cfg.get("activation", "ReLU"))
        activation = getattr(nn, activation_name)
        if cfg.get("lstm_hidden_size") is None:
            raise ValueError("LT2 requires lstm_hidden_size")
        self.lstm_hidden_size = int(cfg["lstm_hidden_size"])
        if self.lstm_hidden_size <= 0:
            raise ValueError("lstm_hidden_size must be positive")
        self.embedding_dimension = int(configured_dimension)
        if "embedding_dim" in cfg and int(cfg["embedding_dim"]) != self.embedding_dimension:
            raise ValueError(
                "vocal_effort.dimension and vocal_effort.model.embedding_dim must match"
            )
        if self.embedding_dimension <= 0:
            raise ValueError("embedding_dim must be positive")
        self.output_dimension = int(cfg.get("output_dimension", 1))
        self.bidirectional = bool(cfg.get("bidirectional", False))
        dropout = float(cfg.get("dropout", 0.0))

        layers = []
        in_channels = 1
        for out_channels, kernel, stride, padding in zip(
            channels, kernels, strides, paddings
        ):
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=tuple(kernel),
                        stride=tuple(stride),
                        padding=tuple(padding),
                    ),
                    activation(),
                    nn.MaxPool2d(pool_kernel),
                ]
            )
            in_channels = out_channels
        self.cnn = nn.Sequential(*layers)
        self.lstm = nn.LSTM(
            input_size=channels[-1],
            hidden_size=self.lstm_hidden_size,
            batch_first=True,
            bidirectional=self.bidirectional,
        )
        lstm_dimension = self.lstm_hidden_size * (2 if self.bidirectional else 1)
        self.fc1 = nn.Linear(lstm_dimension, self.embedding_dimension)
        self.fc2 = nn.Linear(self.embedding_dimension, self.output_dimension)
        self.activation = activation()
        output_name = str(cfg.get("output_activation", "Sigmoid"))
        self.output_activation = getattr(nn, output_name)()
        self.dropout = nn.Dropout(dropout)

    def forward_cnn(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError("LT2 features must have shape (batch, channels, height, width)")
        encoded = self.cnn(features)
        return encoded.mean(dim=2).transpose(1, 2)

    def forward_lstm(self, features: torch.Tensor) -> torch.Tensor:
        sequence = self.forward_cnn(features)
        sequence, _ = self.lstm(sequence)
        return sequence[:, -1, :]

    def extract_embedding(self, features: torch.Tensor) -> torch.Tensor:
        """Return FC1 output with shape ``(batch, 32)``."""

        hidden = self.forward_lstm(features)
        return self.activation(self.fc1(hidden))

    def predict_intensity(self, features: torch.Tensor) -> torch.Tensor:
        embedding = self.extract_embedding(features)
        return self.output_activation(self.fc2(self.dropout(embedding)))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.predict_intensity(features)

    @classmethod
    def load_from_checkpoint(
        cls,
        *args: str,
        config_path: Optional[str] = None,
        ckpt_path: Optional[str] = None,
        map_location: Union[str, torch.device] = "cpu",
    ) -> "IntensityEncoder":
        """Load E2 from its embedded model configuration."""

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
            path = Path(config_path)
            if path.suffix.lower() in {".yaml", ".yml"}:
                try:
                    import yaml
                except ImportError as exc:
                    raise ImportError("YAML configuration requires PyYAML") from exc
                with path.open(encoding="utf-8") as stream:
                    config = yaml.safe_load(stream) or {}
            else:
                with path.open(encoding="utf-8") as stream:
                    config = json.load(stream)
        else:
            config = payload.get("config", {})
        if isinstance(config, Mapping) and isinstance(config.get("vocal_effort"), Mapping):
            config = config["vocal_effort"]
        if not isinstance(config, Mapping) or not config:
            raise ValueError("checkpoint does not contain an E2 model configuration")
        model = cls(config)
        model.load_state_dict(payload["model"])
        return model


def expand_global_vocal_embedding(
    embedding: torch.Tensor, num_frames: int
) -> torch.Tensor:
    """Expand ``(B,D)`` global embeddings to ``(B,T,D)`` without copying."""

    if num_frames < 0:
        raise ValueError("num_frames must be non-negative")
    if embedding.ndim == 1:
        embedding = embedding.unsqueeze(0)
    if embedding.ndim != 2:
        raise ValueError("embedding must have shape (D) or (B,D)")
    return embedding.unsqueeze(1).expand(-1, num_frames, -1)


__all__ = ["IntensityEncoder", "expand_global_vocal_embedding"]
