from __future__ import annotations

import datetime as _dt
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .utils import load_json, save_json

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _model_tag(name: str) -> str:
    return name.rsplit("/", 1)[-1].lower()


@dataclass
class ExperimentConfig:
    dataset_variant: str
    experiment_name: str = ""
    output_dir: str = ""

    model_type: str = "embedding_logreg"
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL
    classifier_type: str = "logistic_regression"
    classifier_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"max_iter": 1000, "C": 1.0, "random_state": 42}
    )
    embedding_batch_size: int = 64
    normalize_embeddings: bool = True

    train_frac: float = 0.8
    val_frac: float = 0.1
    test_frac: float = 0.1
    random_seed: int = 42
    split_seed: int = 42
    splits_dir: str = "outputs/splits"

    # Runtime-populated (filled by train/evaluate; persisted into config.json)
    pool_sizes: dict[str, dict[str, int]] | None = None
    global_splits_file: str | None = None
    realized_refuse_fraction: dict[str, float] | None = None
    timestamp: str | None = None

    def __post_init__(self) -> None:
        if not self.experiment_name:
            self.experiment_name = (
                f"{self.dataset_variant}__{_model_tag(self.embedding_model_name)}"
                f"__seed{self.random_seed}"
            )
        if not self.output_dir:
            self.output_dir = f"outputs/{self.experiment_name}"

    def stamp_timestamp(self) -> None:
        self.timestamp = _dt.datetime.now().isoformat(timespec="seconds")

    def to_json(self, path: str | Path) -> None:
        save_json(asdict(self), path)

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        data = load_json(path)
        return cls(**data)
