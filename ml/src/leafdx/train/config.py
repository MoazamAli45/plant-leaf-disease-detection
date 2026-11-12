"""Experiment configuration: one YAML file per run, overridable from the command line.

    python -m leafdx.train.run --config configs/paper/cnn.yaml --set train.epochs=2 seed=1

Every field is written back to the run directory (``config.yaml``) so a result can always be
traced to the exact settings that produced it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataCfg:
    train_manifest: str = "manifests/paper_split/train.csv"
    val_manifest: str = "manifests/paper_split/val.csv"
    test_manifest: str | None = None
    data_root: str = "data"  # manifest paths are relative to this
    # Optional: where raw/npd/{train,valid} actually live (e.g. a Kaggle input mount).
    npd_root: str | None = None
    image_size: int = 224
    # Keras' old flow_from_directory resized with "nearest" by default.
    resize_method: str = "nearest"
    # "rescale": x / 255 (ImageDataGenerator(rescale=1/255)); "mobilenet": [-1, 1].
    preprocessing: str = "rescale"


@dataclass
class AugmentCfg:
    """ImageDataGenerator-style parameters (same names and units)."""

    enabled: bool = True
    rotation_range: float = 30.0  # degrees
    width_shift_range: float = 0.2  # fraction of width
    height_shift_range: float = 0.2
    shear_range: float = 0.2  # degrees, as in ImageDataGenerator
    zoom_range: float = 0.2  # zoom in [1 - z, 1 + z]
    brightness_range: tuple[float, float] = (0.8, 1.2)  # multiplicative
    horizontal_flip: bool = True
    vertical_flip: bool = True
    fill_mode: str = "nearest"


@dataclass
class ModelCfg:
    name: str = "paper_cnn"  # paper_cnn | paper_mobilenet
    num_classes: int = 38
    output_activation: str = "sigmoid"
    dropout: float = 0.2
    # MobileNet only
    alpha: float = 1.0
    weights: str | None = "imagenet"
    freeze_backbone: bool = True


@dataclass
class TrainCfg:
    epochs: int = 20
    batch_size: int = 32
    optimizer: str = "adam"
    learning_rate: float = 1e-3
    loss: str = "categorical_crossentropy"
    # null = one full pass over the data per epoch
    steps_per_epoch: int | None = 100
    validation_steps: int | None = 150
    shuffle_buffer: int | None = None  # None = full shuffle (manifests are class-sorted)
    mixed_precision: bool = False


@dataclass
class ExperimentCfg:
    name: str = "unnamed"
    description: str = ""
    seed: int = 42
    output_dir: str = "outputs"
    data: DataCfg = field(default_factory=DataCfg)
    augment: AugmentCfg = field(default_factory=AugmentCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    train: TrainCfg = field(default_factory=TrainCfg)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build(cls: type, values: dict[str, Any]) -> Any:
    known = {f.name: f for f in fields(cls)}
    unknown = set(values) - set(known)
    if unknown:
        raise KeyError(f"unknown {cls.__name__} fields: {sorted(unknown)}")
    kwargs = {}
    for name, val in values.items():
        default = getattr(cls(), name) if name in known else None
        if is_dataclass(default) and isinstance(val, dict):
            kwargs[name] = _build(type(default), val)
        elif isinstance(default, tuple) and isinstance(val, list):
            kwargs[name] = tuple(val)
        else:
            kwargs[name] = val
    return cls(**kwargs)


def _set_dotted(d: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def load_config(path: Path, overrides: list[str] | None = None) -> ExperimentCfg:
    """Load YAML, apply ``key.sub=value`` overrides (values parsed as YAML)."""
    raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
    for item in overrides or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"override must be key=value, got {item!r}")
        _set_dotted(raw, key.strip(), yaml.safe_load(value))
    return _build(ExperimentCfg, raw)
