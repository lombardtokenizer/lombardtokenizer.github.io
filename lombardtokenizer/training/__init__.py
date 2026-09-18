"""Public LombardTokenizer training workflows."""

from pathlib import Path
from typing import Any, List, Mapping, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from ..config import build_codec_config, load_yaml_config, validate_config
from ..model import LombardTokenizer
from ..vocal_effort.intensity_encoder import IntensityEncoder
from ..vocal_effort.normalization import SpeakerIntensityNormalizer
from .dataset import (
    LombardTokenizerBatch,
    ManifestAudioDataset,
    VocalEffortManifestDataset,
    freeze_vocal_effort_encoder,
)
from .trainer import LombardTokenizerTrainer, VocalEffortEncoderTrainer, seed_everything, seed_worker


def _resolve_config_reference(value: Any, config_path: Path) -> str:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return str(path.resolve())


def load_frozen_vocal_effort_encoder(
    config: Mapping[str, Any], config_path: Path
) -> IntensityEncoder:
    """Load the self-contained E2 checkpoint used to create LT2 targets."""

    vocal = config.get("vocal_effort", {})
    if not isinstance(vocal, Mapping):
        raise ValueError("vocal_effort configuration must be a mapping")
    checkpoint = vocal.get("checkpoint")
    if checkpoint is None:
        raise ValueError(
            "vocal_effort.checkpoint is required to train LombardTokenizer; "
            "train E2 first"
        )
    encoder = IntensityEncoder.load_from_checkpoint(
        _resolve_config_reference(checkpoint, config_path)
    )
    expected_dimension = int(vocal.get("dimension", 32))
    if encoder.embedding_dimension != expected_dimension:
        raise ValueError(
            "E2 checkpoint embedding dimension does not match "
            f"vocal_effort.dimension ({expected_dimension})"
        )
    freeze_vocal_effort_encoder(encoder)
    return encoder


def collate_lombardtokenizer(
    samples: List[LombardTokenizerBatch],
) -> LombardTokenizerBatch:
    """Collate fixed-size audio and semantic features into a training batch."""

    if not samples:
        raise ValueError("cannot collate an empty LombardTokenizer batch")
    if any(sample.semantic_features is None for sample in samples):
        raise ValueError("every sample must contain semantic features")
    vocal_features = [
        sample.vocal_effort_features
        for sample in samples
        if sample.vocal_effort_features is not None
    ]
    if vocal_features and len(vocal_features) != len(samples):
        raise ValueError("vocal-effort targets must be present for every sample or none")
    return LombardTokenizerBatch(
        audio=torch.stack([sample.audio for sample in samples]),
        semantic_features=torch.stack(
            [sample.semantic_features for sample in samples if sample.semantic_features is not None]
        ),
        vocal_effort_features=(
            None if not vocal_features else torch.stack(vocal_features)
        ),
        speaker_ids=[
            speaker_id
            for sample in samples
            if sample.speaker_ids is not None
            for speaker_id in sample.speaker_ids
        ]
        or None,
    )


def _data_loader_options(training: Mapping[str, Any], *, validation: bool) -> dict:
    num_workers = int(training.get("num_workers", 0))
    return {
        "batch_size": int(training.get("batch_size", 1)),
        "shuffle": bool(training.get("shuffle", True)) if not validation else False,
        "num_workers": num_workers,
        "drop_last": bool(training.get("drop_last", False)) if not validation else False,
        "worker_init_fn": seed_worker if num_workers > 0 else None,
    }


def build_tokenizer_dataloaders(
    config: Mapping[str, Any],
    config_path: Path,
    model: LombardTokenizer,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Build distinct train/validation loaders with aligned dataset options."""

    training = config.get("training", {})
    if not isinstance(training, Mapping):
        raise ValueError("training configuration must be a mapping")
    data = config.get("data", {})
    if not isinstance(data, Mapping):
        raise ValueError("data configuration must be a mapping")
    manifest = data.get("train_manifest")
    if manifest is None:
        raise ValueError("data.train_manifest is required")
    segment_size = training.get("segment_size")
    frame_hop = int(getattr(model, "frame_hop", model.downsample_rate))
    train_dataset = ManifestAudioDataset(
        str((config_path.parent / str(manifest)).resolve()),
        model.sample_rate,
        semantic_dimension=model.semantic_dimension,
        segment_size=None if segment_size is None else int(segment_size),
        training=True,
        frame_hop=frame_hop,
    )
    train_loader = DataLoader(
        train_dataset,
        collate_fn=collate_lombardtokenizer,
        **_data_loader_options(training, validation=False),
    )

    valid_manifest = data.get("valid_manifest")
    if valid_manifest is None:
        return train_loader, None
    valid_dataset = ManifestAudioDataset(
        str((config_path.parent / str(valid_manifest)).resolve()),
        model.sample_rate,
        semantic_dimension=model.semantic_dimension,
        segment_size=None if segment_size is None else int(segment_size),
        training=False,
        frame_hop=frame_hop,
    )
    valid_loader = DataLoader(
        valid_dataset,
        collate_fn=collate_lombardtokenizer,
        **_data_loader_options(training, validation=True),
    )
    return train_loader, valid_loader


def run_tokenizer_training(
    config_path: str,
    mode: str = "pretrain",
    resume: Optional[str] = None,
    fine_tune: Optional[str] = None,
) -> None:
    config, resolved_path = load_yaml_config(config_path)
    if config.get("stage") != mode:
        raise ValueError(f"config stage is {config.get('stage')!r}, expected {mode!r}")
    validate_config(config, training=True)
    seed_everything(int(config.get("training", {}).get("seed", 1234)))
    model = LombardTokenizer(build_codec_config(config, resolved_path))
    vocal_effort_encoder = load_frozen_vocal_effort_encoder(config, resolved_path)
    trainer = LombardTokenizerTrainer(
        model,
        config,
        vocal_effort_encoder=vocal_effort_encoder,
    )
    if fine_tune is not None and resume is not None:
        raise ValueError("fine_tune and resume cannot be used together")
    if fine_tune is not None:
        trainer.load_model_weights(fine_tune)
    if resume is not None:
        trainer.resume_training(resume)
    train_loader, valid_loader = build_tokenizer_dataloaders(
        config, resolved_path, model
    )
    output_dir = _resolve_config_reference(
        config.get("checkpoint", {}).get("output_dir", "checkpoints"),
        resolved_path,
    )
    last_checkpoint = str(Path(output_dir) / "last.pt")
    best_checkpoint = (
        str(Path(output_dir) / "best.pt") if valid_loader is not None else None
    )
    trainer.fit(
        train_loader,
        int(config["training"]["epochs"]),
        validation_batches=valid_loader,
        checkpoint_path=last_checkpoint,
        best_checkpoint_path=best_checkpoint,
    )
    trainer.save(last_checkpoint)


def run_vocal_effort_training(config_path: str, resume: Optional[str] = None) -> None:
    config, resolved_path = load_yaml_config(config_path)
    if config.get("stage") != "vocal_effort":
        raise ValueError("vocal-effort config must have stage=vocal_effort")
    seed_everything(int(config.get("training", {}).get("seed", 1234)))
    validate_config({
        "model": {
            "sample_rate": 16000,
            "dimension": 1024,
            "codebook_size": 1024,
            "num_quantizers": 8,
            "semantic_rvq_layer": 0,
            "vocal_effort_rvq_layer": 1,
        },
        "semantic": {"model_name_or_path": "offline", "dimension": 768},
        "vocal_effort": config["vocal_effort"],
    })
    vocal = config["vocal_effort"]
    if vocal.get("speaker_statistics_required") is not True:
        raise ValueError("speaker-wise target statistics are mandatory for LT2")
    model = IntensityEncoder(vocal)
    data = config.get("data", {})
    training = config.get("training", {})
    manifest = data.get("train_manifest")
    if manifest is None:
        raise ValueError("data.train_manifest is required")
    checkpoint_config = config.get("checkpoint", {})
    if not isinstance(checkpoint_config, Mapping):
        raise ValueError("checkpoint configuration must be a mapping")
    output_dir = Path(
        _resolve_config_reference(
            checkpoint_config.get("output_dir", "checkpoints"), resolved_path
        )
    )
    explicit_stats_path = data.get("speaker_stats_path")
    normalizer = None
    if explicit_stats_path is not None:
        stats_path = Path(_resolve_config_reference(explicit_stats_path, resolved_path))
        if resume is not None:
            raise ValueError(
                "data.speaker_stats_path cannot be combined with --resume; "
                "resume restores statistics from the checkpoint"
            )
        normalizer = SpeakerIntensityNormalizer.load(stats_path)
    trainer = VocalEffortEncoderTrainer(
        model,
        config,
        speaker_normalizer=normalizer,
    )
    if resume is not None:
        trainer.resume_training(resume)
        if trainer.speaker_normalizer is None:
            raise ValueError("E2 resume checkpoint does not contain speaker statistics")
        normalizer = trainer.speaker_normalizer
    dataset = VocalEffortManifestDataset(
        str((resolved_path.parent / str(manifest)).resolve()),
        int(training.get("sample_rate", 16000)),
        segment_size=training.get("segment_size"),
        training=True,
        frame_hop=int(training.get("frame_hop", 320)),
        normalizer=normalizer,
    )
    if explicit_stats_path is None:
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset.normalizer.save(output_dir / "speaker_stats.json")
    trainer.speaker_normalizer = dataset.normalizer
    loader_options = _data_loader_options(training, validation=False)
    train_loader = DataLoader(dataset, **loader_options)
    valid_manifest = data.get("valid_manifest")
    valid_loader = None
    if valid_manifest is not None:
        valid_dataset = VocalEffortManifestDataset(
            str((resolved_path.parent / str(valid_manifest)).resolve()),
            int(training.get("sample_rate", 16000)),
            segment_size=training.get("segment_size"),
            training=False,
            frame_hop=320,
            normalizer=dataset.normalizer,
        )
        valid_loader = DataLoader(
            valid_dataset,
            **_data_loader_options(training, validation=True),
        )
    last_checkpoint = str(output_dir / "last.pt")
    best_checkpoint = (
        str(Path(output_dir) / "best.pt") if valid_loader is not None else None
    )
    history = trainer.fit(
        train_loader,
        validation_batches=valid_loader,
        checkpoint_path=last_checkpoint,
        best_checkpoint_path=best_checkpoint,
    )
    if history:
        latest = history[-1]
        print(
            "epoch={epoch} train_loss={train:.6f}".format(
                epoch=trainer.epoch,
                train=latest["train_loss"],
            )
            + (
                " valid_loss={:.6f}".format(latest["valid_loss"])
                if "valid_loss" in latest
                else ""
            )
        )
    trainer.save(last_checkpoint)


__all__ = [
    "IntensityEncoder",
    "LombardTokenizerBatch",
    "LombardTokenizerTrainer",
    "VocalEffortEncoderTrainer",
    "ManifestAudioDataset",
    "VocalEffortManifestDataset",
    "build_tokenizer_dataloaders",
    "collate_lombardtokenizer",
    "run_tokenizer_training",
    "run_vocal_effort_training",
    "seed_everything",
    "seed_worker",
    "load_frozen_vocal_effort_encoder",
]
