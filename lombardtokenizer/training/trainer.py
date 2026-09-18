"""Training loops for LombardTokenizer and the LT2 teacher."""

import random
from copy import deepcopy
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from ..vocal_effort.intensity_encoder import expand_global_vocal_embedding
from ..vocal_effort.normalization import SpeakerIntensityNormalizer
from ..codec.discriminators import DiscriminatorOutput, build_discriminator
from ..checkpoint import load_checkpoint, make_checkpoint, save_checkpoint
from .dataset import (
    LombardTokenizerBatch,
    _vocal_effort_input,
    freeze_vocal_effort_encoder,
)
from .losses import (
    MelResolution,
    commitment_loss,
    discriminator_loss,
    feature_matching_loss,
    generator_adversarial_loss,
    multi_resolution_mel_loss,
    reconstruction_loss,
    semantic_distill_loss,
    vocal_effort_distill_loss,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """Seed a DataLoader worker from PyTorch's worker seed."""

    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _resolve_device(value: Optional[str]) -> torch.device:
    if value is None or value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = torch.device(value)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")
    return requested


def _fresh_optimizer(
    optimizer: torch.optim.Optimizer,
    parameters: Iterable[torch.nn.Parameter],
    config: Optional[Mapping[str, Any]] = None,
) -> torch.optim.Optimizer:
    """Create a fresh supported optimizer from explicit trainer settings."""

    config = {} if config is None else config
    optimizer_config = config.get("optimizer", {})
    if not isinstance(optimizer_config, Mapping):
        raise ValueError("optimizer configuration must be a mapping")

    def setting(name: str, default: Any) -> Any:
        return optimizer_config.get(name, _config_value(config, name, default))

    defaults = optimizer.defaults
    optimizer_type = type(optimizer)
    if optimizer_type not in (torch.optim.AdamW, torch.optim.Adam):
        raise TypeError(
            "fine-tuning supports only torch.optim.Adam and torch.optim.AdamW"
        )
    kwargs = {
        "lr": float(setting("learning_rate", defaults.get("lr", 1.0e-4))),
        "betas": tuple(setting("betas", defaults.get("betas", (0.9, 0.999)))),
        "eps": float(setting("eps", defaults.get("eps", 1.0e-8))),
        "weight_decay": float(
            setting("weight_decay", defaults.get("weight_decay", 0.0))
        ),
        "amsgrad": bool(setting("amsgrad", defaults.get("amsgrad", False))),
    }
    return optimizer_type(parameters, **kwargs)


def _build_optimizer(
    parameters: Iterable[torch.nn.Parameter], config: Mapping[str, Any]
) -> torch.optim.Optimizer:
    """Build the configured optimizer without copying internal PyTorch defaults."""

    optimizer_config = config.get("optimizer", {})
    if optimizer_config is None:
        optimizer_config = {}
    if not isinstance(optimizer_config, Mapping):
        raise ValueError("optimizer configuration must be a mapping")
    optimizer_type = str(optimizer_config.get("type", "adamw")).lower()
    if optimizer_type == "adamw":
        optimizer = torch.optim.AdamW
    elif optimizer_type == "adam":
        optimizer = torch.optim.Adam
    else:
        raise ValueError("optimizer.type must be 'adam' or 'adamw'")
    defaults = {
        "learning_rate": 1.0e-4,
        "lr": 1.0e-4,
        "betas": (0.9, 0.999),
        "eps": 1.0e-8,
        "weight_decay": 0.0,
        "amsgrad": False,
    }

    def setting(name: str) -> Any:
        return optimizer_config.get(name, _config_value(config, name, defaults[name]))

    return optimizer(
        parameters,
        lr=float(setting("learning_rate")),
        betas=tuple(setting("betas")),
        eps=float(setting("eps")),
        weight_decay=float(setting("weight_decay")),
        amsgrad=bool(setting("amsgrad")),
    )

def _fresh_scheduler(
    scheduler: Any,
    optimizer: torch.optim.Optimizer,
    initial_state: Mapping[str, Any],
    template: Any,
) -> Any:
    """Create a scheduler with a fresh optimizer and initial state."""

    fresh = deepcopy(template if template is not None else scheduler)
    fresh.optimizer = optimizer
    fresh.load_state_dict(deepcopy(dict(initial_state)))
    return fresh


def _config_value(config: Mapping[str, Any], key: str, default: Any) -> Any:
    training = config.get("training")
    if isinstance(training, Mapping) and key in training:
        return training[key]
    return config.get(key, default)


def _loss_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    loss_config = config.get("loss", config)
    if not isinstance(loss_config, Mapping):
        raise ValueError("loss configuration must be a mapping")
    return loss_config


def _loss_weights(
    config: Mapping[str, Any], *, mel_active: bool
) -> Dict[str, float]:
    loss_config = _loss_config(config)
    weights = {
        "reconstruction_loss": float(loss_config.get("reconstruction_weight", 1.0)),
        "commitment_loss": float(loss_config.get("commitment_weight", 1.0)),
        "semantic_distillation_loss": float(
            loss_config.get("semantic_distill_weight", 1.0)
        ),
        "vocal_effort_distillation_loss": float(
            loss_config.get("vocal_effort_distill_weight", 1.0)
        ),
        "mel_loss": 1.0 if mel_active else 0.0,
        "adversarial_loss": float(loss_config.get("adversarial_weight", 0.0)),
        "feature_matching_loss": float(
            loss_config.get("feature_matching_weight", 0.0)
        ),
    }
    if any(not np.isfinite(value) or value < 0 for value in weights.values()):
        raise ValueError("loss weights must be finite and non-negative")
    return weights


def _mel_weights(config: Mapping[str, Any]) -> List[float]:
    value = _loss_config(config).get("mel_weights", [])
    if not isinstance(value, (list, tuple)):
        raise ValueError("loss.mel_weights must be a list or tuple")
    if not value:
        return []
    weights = [float(item) for item in value]
    if any(not np.isfinite(item) or item < 0 for item in weights):
        raise ValueError("mel weights must be finite and non-negative")
    if not any(weights):
        return []
    return weights


def _mel_resolutions(config: Mapping[str, Any], count: int) -> List[MelResolution]:
    if count == 0:
        return []
    value = _loss_config(config).get("mel_resolutions")
    if value is None:
        defaults = [(1024, 80), (512, 64), (256, 40), (128, 20)]
        if count > len(defaults):
            raise ValueError("loss.mel_resolutions is required for more than four scales")
        return [
            MelResolution(
                n_fft=n_fft,
                hop_length=max(1, n_fft // 4),
                win_length=n_fft,
                n_mels=n_mels,
            )
            for n_fft, n_mels in defaults[:count]
        ]
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise ValueError("loss.mel_resolutions and loss.mel_weights must have equal lengths")
    resolutions = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("each loss.mel_resolutions entry must be a mapping")
        n_fft = int(item["n_fft"])
        hop_length = int(item.get("hop_length", n_fft // 4))
        win_length = int(item.get("win_length", n_fft))
        resolutions.append(
            MelResolution(
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                n_mels=int(item.get("n_mels", 80)),
                f_min=float(item.get("f_min", 0.0)),
                f_max=(
                    None
                    if item.get("f_max") is None
                    else float(item["f_max"])
                ),
            )
        )
    return resolutions


class LombardTokenizerTrainer:
    """Explicit single-device trainer for LombardTokenizer."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: Optional[Mapping[str, Any]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        vocal_effort_encoder: Optional[torch.nn.Module] = None,
        discriminator: Optional[torch.nn.Module] = None,
        discriminator_optimizer: Optional[torch.optim.Optimizer] = None,
        discriminator_scheduler: Optional[Any] = None,
    ):
        self.model = model
        self.config = dict(config or {})
        self.device = _resolve_device(_config_value(self.config, "device", None))
        self.model.to(self.device)
        self.vocal_effort_encoder = vocal_effort_encoder
        if self.vocal_effort_encoder is not None:
            if not hasattr(self.vocal_effort_encoder, "extract_embedding"):
                raise TypeError("LT2 encoder must expose extract_embedding")
            freeze_vocal_effort_encoder(self.vocal_effort_encoder)
            self.vocal_effort_encoder.to(self.device)
        self.mel_weights = _mel_weights(self.config)
        self.mel_resolutions = _mel_resolutions(
            self.config, len(self.mel_weights)
        )
        self.weights = _loss_weights(
            self.config, mel_active=bool(self.mel_weights)
        )
        self.optimizer = optimizer or _build_optimizer(self.model.parameters(), self.config)
        self.scheduler = scheduler
        self._scheduler_template = deepcopy(scheduler) if scheduler is not None else None
        self._scheduler_initial_state = (
            deepcopy(scheduler.state_dict()) if scheduler is not None else {}
        )
        self.discriminator = discriminator
        self.discriminator_optimizer = discriminator_optimizer
        self.discriminator_scheduler = discriminator_scheduler
        self.discriminator_active = bool(
            self.weights["adversarial_loss"]
            or self.weights["feature_matching_loss"]
        )
        if self.discriminator_active and self.discriminator is None:
            self.discriminator = build_discriminator(self.config)
        if self.discriminator is not None:
            self.discriminator.to(self.device)
        if self.discriminator_active and self.discriminator is not None:
            self.discriminator_optimizer = self.discriminator_optimizer or _build_optimizer(
                self.discriminator.parameters(), self.config
            )
        self._discriminator_scheduler_template = (
            deepcopy(self.discriminator_scheduler)
            if self.discriminator_scheduler is not None
            else None
        )
        self._discriminator_scheduler_initial_state = (
            deepcopy(self.discriminator_scheduler.state_dict())
            if self.discriminator_scheduler is not None
            else {}
        )
        self.epoch = 0
        self.step = 0
        self.best_validation_loss = float("inf")

    def _prepare_batch(
        self,
        batch: LombardTokenizerBatch,
        *,
        include_vocal_effort_target: bool = True,
    ) -> LombardTokenizerBatch:
        audio = batch.audio.to(self.device)
        vocal_effort_features = (
            None
            if batch.vocal_effort_features is None
            else batch.vocal_effort_features.to(self.device)
        )
        if (
            include_vocal_effort_target
            and vocal_effort_features is None
            and self.vocal_effort_encoder is not None
        ):
            self.vocal_effort_encoder.eval()
            effort_input = _vocal_effort_input(audio)
            with torch.no_grad():
                vocal_effort_features = self.vocal_effort_encoder.extract_embedding(
                    effort_input
                ).detach().float()
            expected_dimension = int(
                getattr(self.model, "vocal_effort_dimension", 32)
            )
            if (
                vocal_effort_features.ndim != 2
                or vocal_effort_features.shape[-1] != expected_dimension
            ):
                raise ValueError(
                    "LT2 encoder must return (batch, dimension) features with "
                    f"dimension {expected_dimension}; got "
                    f"{tuple(vocal_effort_features.shape)}"
                )
        return LombardTokenizerBatch(
            audio=audio,
            semantic_features=(
                None
                if batch.semantic_features is None
                else batch.semantic_features.to(self.device)
            ),
            vocal_effort_features=vocal_effort_features,
            speaker_ids=batch.speaker_ids,
        )

    @staticmethod
    def _validate_targets(batch: LombardTokenizerBatch) -> None:
        if batch.semantic_features is None:
            raise ValueError(
                "semantic target is required for LombardTokenizer training"
            )
        if batch.vocal_effort_features is None:
            raise ValueError(
                "vocal-effort target is required for LombardTokenizer training"
            )

    def _losses_from_output(
        self,
        batch: LombardTokenizerBatch,
        output: Any,
    ) -> Dict[str, torch.Tensor]:
        self._validate_targets(batch)
        if batch.semantic_features is None or batch.vocal_effort_features is None:
            raise RuntimeError("training targets unexpectedly missing")
        vocal_target = batch.vocal_effort_features
        if vocal_target.ndim == 2:
            vocal_target = expand_global_vocal_embedding(
                vocal_target, output.vocal_effort_features.shape[1]
            )
        elif vocal_target.ndim != 3:
            raise ValueError(
                "vocal-effort target must have shape (batch, dimension) or "
                "(batch, time, dimension)"
            )
        losses = {
            "reconstruction_loss": reconstruction_loss(
                batch.audio, output.reconstructed_audio
            ),
            "commitment_loss": commitment_loss(output.commitment_loss),
            "semantic_distillation_loss": semantic_distill_loss(
                output.semantic_features,
                batch.semantic_features,
                expected_dimension=768,
            ),
            "vocal_effort_distillation_loss": vocal_effort_distill_loss(
                output.vocal_effort_features,
                vocal_target,
                expected_dimension=32,
            ),
            "mel_loss": multi_resolution_mel_loss(
                batch.audio,
                output.reconstructed_audio,
                sample_rate=int(_config_value(self.config, "sample_rate", 16000)),
                resolutions=self.mel_resolutions,
                weights=self.mel_weights,
            ),
        }
        zero = batch.audio.new_zeros(())
        losses["adversarial_loss"] = zero
        losses["feature_matching_loss"] = zero
        return losses

    def _discriminator_output(self, audio: torch.Tensor) -> DiscriminatorOutput:
        if self.discriminator is None:
            raise RuntimeError("a discriminator is required for adversarial training")
        result = self.discriminator(audio)
        if isinstance(result, DiscriminatorOutput):
            return result
        if isinstance(result, tuple) and len(result) == 2:
            logits, feature_maps = result
            return DiscriminatorOutput(
                list(logits), [list(item) for item in feature_maps]
            )
        if isinstance(result, torch.Tensor):
            return DiscriminatorOutput([result], [])
        raise TypeError(
            "discriminator must return DiscriminatorOutput, (logits, feature_maps), "
            "or a tensor"
        )

    @contextmanager
    def _freeze_discriminator(self):
        if self.discriminator is None:
            yield
            return
        requires_grad = [
            parameter.requires_grad for parameter in self.discriminator.parameters()
        ]
        for parameter in self.discriminator.parameters():
            parameter.requires_grad_(False)
        try:
            yield
        finally:
            for parameter, enabled in zip(
                self.discriminator.parameters(), requires_grad
            ):
                parameter.requires_grad_(enabled)

    def _generator_losses(
        self, batch: LombardTokenizerBatch
    ) -> Dict[str, torch.Tensor]:
        self._validate_targets(batch)
        output = self.model(batch.audio)
        losses = self._losses_from_output(batch, output)
        if not self.discriminator_active:
            return losses
        with self._freeze_discriminator():
            with torch.no_grad():
                real_output = self._discriminator_output(batch.audio)
            fake_output = self._discriminator_output(output.reconstructed_audio)
        if self.weights["adversarial_loss"]:
            losses["adversarial_loss"] = generator_adversarial_loss(
                fake_output.logits
            )
        if self.weights["feature_matching_loss"]:
            losses["feature_matching_loss"] = feature_matching_loss(
                real_output.feature_maps, fake_output.feature_maps
            )
        return losses

    def compute_losses(self, batch: LombardTokenizerBatch) -> Dict[str, torch.Tensor]:
        """Compute every configured generator loss for one batch."""

        return self._generator_losses(self._prepare_batch(batch))

    def total_loss(self, losses: Mapping[str, torch.Tensor]) -> torch.Tensor:
        missing = set(self.weights) - set(losses)
        if missing:
            raise ValueError("missing required losses: " + ", ".join(sorted(missing)))
        return sum(self.weights[name] * losses[name] for name in self.weights)

    @staticmethod
    def _metrics(
        losses: Mapping[str, torch.Tensor], total: torch.Tensor
    ) -> Dict[str, float]:
        result = {name: float(value.detach()) for name, value in losses.items()}
        result["loss"] = float(total.detach())
        return result

    def train_step(self, batch: LombardTokenizerBatch) -> Dict[str, float]:
        """Run a discriminator step when enabled, then one generator step."""

        metrics: Dict[str, float] = {}
        if self.discriminator_active:
            metrics.update(self.train_discriminator_step(batch))
        metrics.update(self.train_generator_step(batch))
        return metrics

    def train_generator_step(self, batch: LombardTokenizerBatch) -> Dict[str, float]:
        """Run one generator optimizer step."""

        self.model.train()
        if self.discriminator is not None:
            self.discriminator.train()
        batch = self._prepare_batch(batch)
        self.optimizer.zero_grad(set_to_none=True)
        losses = self._generator_losses(batch)
        total = self.total_loss(losses)
        if not torch.isfinite(total):
            raise FloatingPointError("non-finite LombardTokenizer training loss")
        total.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.step += 1
        return self._metrics(losses, total)

    def train_discriminator_step(
        self, batch: LombardTokenizerBatch
    ) -> Dict[str, float]:
        """Run one discriminator optimizer step using detached reconstructions."""

        if not self.discriminator_active or self.discriminator is None:
            raise RuntimeError("discriminator training is not enabled")
        if self.discriminator_optimizer is None:
            raise RuntimeError("discriminator optimizer is not configured")
        self.model.train()
        self.discriminator.train()
        batch = self._prepare_batch(batch, include_vocal_effort_target=False)
        with torch.no_grad():
            generated = self.model(batch.audio).reconstructed_audio
        real_output = self._discriminator_output(batch.audio)
        fake_output = self._discriminator_output(generated.detach())
        loss = discriminator_loss(real_output.logits, fake_output.logits)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite discriminator training loss")
        self.discriminator_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.discriminator_optimizer.step()
        if self.discriminator_scheduler is not None:
            self.discriminator_scheduler.step()
        return {"discriminator_loss": float(loss.detach())}

    def train_batch(self, batch: LombardTokenizerBatch) -> Dict[str, float]:
        """Run one complete training step."""

        return self.train_step(batch)

    @torch.no_grad()
    def validation_step(self, batch: LombardTokenizerBatch) -> Dict[str, float]:
        """Evaluate one batch without changing model or optimizer state."""

        self.model.eval()
        batch = self._prepare_batch(batch)
        losses = self._generator_losses(batch)
        total = self.total_loss(losses)
        if not torch.isfinite(total):
            raise FloatingPointError("non-finite LombardTokenizer validation loss")
        return self._metrics(losses, total)

    @staticmethod
    def _mean_metrics(metrics: Iterable[Mapping[str, float]]) -> Dict[str, float]:
        rows = list(metrics)
        if not rows:
            return {}
        keys = rows[0].keys()
        return {key: sum(row[key] for row in rows) / len(rows) for key in keys}

    def fit(
        self,
        batches: Iterable[LombardTokenizerBatch],
        epochs: int,
        validation_batches: Optional[Iterable[LombardTokenizerBatch]] = None,
        checkpoint_path: Optional[str] = None,
        best_checkpoint_path: Optional[str] = None,
    ) -> List[Dict[str, float]]:
        """Train until the target epoch and optionally save checkpoints."""

        history = []
        target_epochs = int(epochs)
        if target_epochs < 0:
            raise ValueError("epochs must be non-negative")
        best_validation_loss = self.best_validation_loss
        for _ in range(self.epoch, target_epochs):
            train_metrics = [self.train_step(batch) for batch in batches]
            if not train_metrics:
                raise ValueError("training dataset cannot be empty")
            self.epoch += 1
            metrics = {f"train_{key}": value for key, value in self._mean_metrics(train_metrics).items()}
            if validation_batches is not None:
                valid_metrics = [
                    self.validation_step(batch) for batch in validation_batches
                ]
                if not valid_metrics:
                    raise ValueError("validation dataset cannot be empty")
                metrics.update(
                    {
                        f"valid_{key}": value
                        for key, value in self._mean_metrics(valid_metrics).items()
                    }
                )
            history.append(metrics)
            is_best = (
                "valid_loss" in metrics and metrics["valid_loss"] < best_validation_loss
            )
            if is_best:
                best_validation_loss = metrics["valid_loss"]
                self.best_validation_loss = best_validation_loss
            if checkpoint_path is not None:
                self.save(checkpoint_path)
            if is_best and best_checkpoint_path is not None:
                self.save(best_checkpoint_path)
        return history

    def save(self, path: str) -> None:
        payload = make_checkpoint(
            model=self.model.state_dict(),
            optimizer=self.optimizer.state_dict(),
            scheduler=self.scheduler.state_dict() if self.scheduler is not None else {},
            epoch=self.epoch,
            step=self.step,
            config=self.config,
            discriminator=(
                self.discriminator.state_dict()
                if self.discriminator is not None
                else None
            ),
            discriminator_optimizer=(
                self.discriminator_optimizer.state_dict()
                if self.discriminator_optimizer is not None
                else None
            ),
            discriminator_scheduler=(
                self.discriminator_scheduler.state_dict()
                if self.discriminator_scheduler is not None
                else None
            ),
            best_validation_loss=self.best_validation_loss,
        )
        save_checkpoint(path, payload)

    def resume_training(self, path: str) -> None:
        """Restore model, optimizer, scheduler, epoch and step."""

        payload = load_checkpoint(path, map_location=self.device)
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(payload["scheduler"])
        if "discriminator" in payload:
            if self.discriminator is None:
                self.discriminator = build_discriminator(payload["config"])
                self.discriminator.to(self.device)
                self.discriminator_active = True
            self.discriminator.load_state_dict(payload["discriminator"])
            if self.discriminator_optimizer is None:
                self.discriminator_optimizer = torch.optim.AdamW(
                    self.discriminator.parameters(),
                    lr=float(_config_value(self.config, "learning_rate", 1.0e-4)),
                )
            if "discriminator_optimizer" in payload:
                self.discriminator_optimizer.load_state_dict(
                    payload["discriminator_optimizer"]
                )
        if self.discriminator_scheduler is not None and "discriminator_scheduler" in payload:
            self.discriminator_scheduler.load_state_dict(
                payload["discriminator_scheduler"]
            )
        self.epoch = int(payload["epoch"])
        self.step = int(payload["step"])
        self.best_validation_loss = float(
            payload.get("best_validation_loss")
            if payload.get("best_validation_loss") is not None
            else float("inf")
        )

    def load_model_weights(self, path: str) -> None:
        """Load only model weights for a fresh fine-tuning run."""

        payload = load_checkpoint(path, map_location=self.device)
        self.model.load_state_dict(payload["model"])
        self.optimizer = _fresh_optimizer(
            self.optimizer, self.model.parameters(), self.config
        )
        if self.scheduler is not None:
            self.scheduler = _fresh_scheduler(
                self.scheduler,
                self.optimizer,
                self._scheduler_initial_state,
                self._scheduler_template,
            )
        if self.discriminator_optimizer is not None and self.discriminator is not None:
            self.discriminator_optimizer = _fresh_optimizer(
                self.discriminator_optimizer,
                self.discriminator.parameters(),
                self.config,
            )
            if self.discriminator_scheduler is not None:
                self.discriminator_scheduler = _fresh_scheduler(
                    self.discriminator_scheduler,
                    self.discriminator_optimizer,
                    self._discriminator_scheduler_initial_state,
                    self._discriminator_scheduler_template,
                )
        self.epoch = 0
        self.step = 0

class VocalEffortEncoderTrainer:
    """Supervised LT2 trainer with explicit normalized targets."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: Optional[Mapping[str, Any]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        speaker_normalizer: Optional[SpeakerIntensityNormalizer] = None,
    ):
        self.model = model
        self.config = dict(config or {})
        self.device = _resolve_device(_config_value(self.config, "device", None))
        self.model.to(self.device)
        self.optimizer = optimizer or _build_optimizer(model.parameters(), self.config)
        self.scheduler = scheduler
        self._scheduler_template = deepcopy(scheduler) if scheduler is not None else None
        self._scheduler_initial_state: Mapping[str, Any] = (
            deepcopy(scheduler.state_dict()) if scheduler is not None else {}
        )
        self.speaker_normalizer = speaker_normalizer
        self.epoch = 0
        self.step = 0
        self.best_validation_loss = float("inf")

    def train_step(self, features: torch.Tensor, target: torch.Tensor) -> float:
        self.model.train()
        features = features.to(self.device)
        target = target.to(self.device)
        self.optimizer.zero_grad(set_to_none=True)
        prediction = self.model.predict_intensity(features)
        loss = torch.nn.functional.mse_loss(prediction, target)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite LT2 training loss")
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.step += 1
        return float(loss.detach())

    def train_batch(self, features: torch.Tensor, target: torch.Tensor) -> float:
        return self.train_step(features, target)

    @torch.no_grad()
    def validation_step(self, features: torch.Tensor, target: torch.Tensor) -> float:
        self.model.eval()
        prediction = self.model.predict_intensity(
            features.to(self.device)
        )
        loss = torch.nn.functional.mse_loss(prediction, target.to(self.device))
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite LT2 validation loss")
        return float(loss.detach())

    def fit(
        self,
        batches: Iterable,
        epochs: Optional[int] = None,
        validation_batches: Optional[Iterable] = None,
        checkpoint_path: Optional[str] = None,
        best_checkpoint_path: Optional[str] = None,
    ) -> List[Dict[str, float]]:
        """Train E2 until the target epoch and report train/validation MSE."""

        total_epochs = int(
            _config_value(self.config, "epochs", 1) if epochs is None else epochs
        )
        if total_epochs < 0:
            raise ValueError("epochs must be non-negative")
        history: List[Dict[str, float]] = []
        best_validation_loss = self.best_validation_loss
        for _ in range(self.epoch, total_epochs):
            train_losses = []
            for features, target in batches:
                train_losses.append(self.train_step(features, target))
            if not train_losses:
                raise ValueError("training dataset cannot be empty")
            self.epoch += 1
            metrics = {
                "train_loss": sum(train_losses) / len(train_losses)
            }
            if validation_batches is not None:
                validation_losses = [
                    self.validation_step(features, target)
                    for features, target in validation_batches
                ]
                if not validation_losses:
                    raise ValueError("validation dataset cannot be empty")
                metrics["valid_loss"] = sum(validation_losses) / len(validation_losses)
            history.append(metrics)
            is_best = (
                "valid_loss" in metrics and metrics["valid_loss"] < best_validation_loss
            )
            if is_best:
                best_validation_loss = metrics["valid_loss"]
                self.best_validation_loss = best_validation_loss
            if checkpoint_path is not None:
                self.save(checkpoint_path)
            if is_best and best_checkpoint_path is not None:
                self.save(best_checkpoint_path)
        return history

    def save(self, path: str) -> None:
        save_checkpoint(
            path,
            make_checkpoint(
                model=self.model.state_dict(),
                optimizer=self.optimizer.state_dict(),
                scheduler=self.scheduler.state_dict() if self.scheduler is not None else {},
                epoch=self.epoch,
                step=self.step,
                config=self.config,
                speaker_statistics=(
                    self.speaker_normalizer.to_dict()
                    if self.speaker_normalizer is not None
                    else None
                ),
                best_validation_loss=self.best_validation_loss,
            ),
        )

    def resume_training(self, path: str) -> None:
        payload = load_checkpoint(path, map_location=self.device)
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(payload["scheduler"])
        if "speaker_statistics" in payload:
            self.speaker_normalizer = SpeakerIntensityNormalizer.from_dict(
                payload["speaker_statistics"]
            )
        self.epoch = int(payload["epoch"])
        self.step = int(payload["step"])
        self.best_validation_loss = float(
            payload.get("best_validation_loss")
            if payload.get("best_validation_loss") is not None
            else float("inf")
        )

    def load_model_weights(self, path: str) -> None:
        payload = load_checkpoint(path, map_location=self.device)
        self.model.load_state_dict(payload["model"])
        self.optimizer = _fresh_optimizer(
            self.optimizer, self.model.parameters(), self.config
        )
        if self.scheduler is not None:
            self.scheduler = _fresh_scheduler(
                self.scheduler,
                self.optimizer,
                self._scheduler_initial_state,
                self._scheduler_template,
            )
        self.epoch = 0
        self.step = 0

__all__ = [
    "LombardTokenizerTrainer",
    "VocalEffortEncoderTrainer",
    "seed_everything",
    "seed_worker",
]
