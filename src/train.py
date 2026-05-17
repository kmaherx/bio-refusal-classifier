"""Train flow: load splits, compose variant, fit classifier."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import ExperimentConfig
from .data import build_all_splits, get_global_splits
from .models import Classifier, build_classifier
from .utils import set_seed


def run(cfg: ExperimentConfig) -> tuple[Classifier, dict[str, pd.DataFrame]]:
    set_seed(cfg.random_seed)

    fracs = (cfg.train_frac, cfg.val_frac, cfg.test_frac)
    global_splits = get_global_splits(cfg.split_seed, fracs, cfg.splits_dir)

    cfg.pool_sizes = global_splits["pool_sizes"]
    cfg.global_splits_file = str(
        Path(cfg.splits_dir)
        / f"global_splits__seed{cfg.split_seed}__fracs{fracs[0]}-{fracs[1]}-{fracs[2]}.json"
    )

    splits_data, realized = build_all_splits(cfg.dataset_variant, cfg.split_seed, global_splits)
    cfg.realized_refuse_fraction = realized

    model = build_classifier(cfg)
    train_df = splits_data["train"]
    model.fit(train_df["text"].tolist(), train_df["label"].to_numpy())

    return model, splits_data
