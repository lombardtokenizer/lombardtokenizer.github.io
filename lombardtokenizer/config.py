"""Configuration loading and validation for the LombardTokenizer recipes."""

import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple


def load_yaml_config(path: str) -> Tuple[Dict[str, Any], Path]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("YAML configuration requires the 'pyyaml' package") from exc
    config_path = Path(path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise ValueError("top-level YAML configuration must be a mapping")
    return config, config_path


def _get_path(config: Mapping[str, Any], path: str) -> Any:
    value: Any = config
    for key in path.split("."):
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def require_values(config: Mapping[str, Any], paths: Iterable[str]) -> None:
    missing = [path for path in paths if _get_path(config, path) is None]
    if missing:
        raise ValueError(
            "configuration values are required before training: " + ", ".join(missing)
        )


def validate_config(config: Mapping[str, Any], training: bool = False) -> None:
    """Validate paper-critical fields without inventing mHuBERT metadata."""

    model = config.get("model", {})
    semantic = config.get("semantic", {})
    vocal = config.get("vocal_effort", {})
    if not isinstance(model, Mapping) or not isinstance(semantic, Mapping) or not isinstance(vocal, Mapping):
        raise ValueError("model, semantic, and vocal_effort must be mappings")
    require_values(
        config,
        (
            "model.sample_rate",
            "model.dimension",
            "model.codebook_size",
            "model.num_quantizers",
            "model.semantic_rvq_layer",
            "model.vocal_effort_rvq_layer",
            "semantic.model_name_or_path",
            "semantic.dimension",
            "vocal_effort.dimension",
            "vocal_effort.normalization",
        ),
    )
    if model["sample_rate"] != 16000 or model["dimension"] != 1024:
        raise ValueError("paper codec requires sample_rate=16000 and dimension=1024")
    if model["codebook_size"] != 1024 or model["num_quantizers"] != 8:
        raise ValueError("paper codec requires 8 RVQ layers with codebook size 1024")
    if model["semantic_rvq_layer"] != 0 or model["vocal_effort_rvq_layer"] != 1:
        raise ValueError("semantic RVQ must be layer 0 and vocal effort RVQ must be layer 1")
    strides = model.get("strides", [8, 5, 4, 2])
    if not isinstance(strides, (list, tuple)) or not strides or any(
        isinstance(stride, bool) or int(stride) <= 0 for stride in strides
    ):
        raise ValueError("model.strides must contain positive integers")
    codec_frame_hop = math.prod(int(stride) for stride in strides)
    training_config = config.get("training", {})
    if not isinstance(training_config, Mapping):
        raise ValueError("training must be a mapping")
    if "frame_hop" in training_config and int(training_config["frame_hop"]) != codec_frame_hop:
        raise ValueError(
            "training.frame_hop must match the codec stride product "
            f"({codec_frame_hop})"
        )
    if semantic["dimension"] != 768 or vocal["dimension"] != 32:
        raise ValueError("paper teachers require semantic=768-D and LT2=32-D")
    vocal_model = vocal.get("model")
    if isinstance(vocal_model, Mapping) and "embedding_dim" in vocal_model:
        if int(vocal_model["embedding_dim"]) != int(vocal["dimension"]):
            raise ValueError(
                "vocal_effort.dimension and vocal_effort.model.embedding_dim must match"
            )
    if vocal["normalization"] != "rms":
        raise ValueError("official LT2 normalization is RMS")
    if vocal.get("type") != "intensity":
        raise ValueError("LT2 vocal effort must use the intensity encoder")
    if vocal.get("target_normalization") != "speaker_min_max":
        raise ValueError("LT2 targets must use speaker-wise Min-Max normalization")
    if training:
        if semantic.get("layer") is None:
            raise ValueError("semantic.layer must be set before training; no mHuBERT layer is assumed")
        require_values(config, ("training.epochs", "training.batch_size", "training.learning_rate"))
        epochs = training_config["epochs"]
        batch_size = training_config["batch_size"]
        learning_rate = training_config["learning_rate"]
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("training.epochs must be a positive integer")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("training.batch_size must be a positive integer")
        try:
            learning_rate = float(learning_rate)
        except (TypeError, ValueError) as exc:
            raise ValueError("training.learning_rate must be a positive number") from exc
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("training.learning_rate must be a positive number")
        if vocal.get("speaker_statistics_required") is not True:
            raise ValueError("speaker-wise LT2 target statistics must be required")


def resolve_config_path(value: str, config_path: Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return str(path.resolve())


def build_codec_config(config: Mapping[str, Any], config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Flatten the public YAML model section for ``LombardTokenizer``."""

    del config_path
    model = dict(config.get("model", config))
    semantic = config.get("semantic", {})
    vocal = config.get("vocal_effort", {})
    semantic = semantic if isinstance(semantic, Mapping) else {}
    vocal = vocal if isinstance(vocal, Mapping) else {}
    model["semantic_dimension"] = int(semantic.get("dimension", 768))
    model["vocal_effort_dimension"] = int(vocal.get("dimension", 32))
    return model


__all__ = [
    "build_codec_config",
    "load_yaml_config",
    "require_values",
    "resolve_config_path",
    "validate_config",
]
