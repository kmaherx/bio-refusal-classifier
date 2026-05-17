#!/usr/bin/env python3
"""CLI entry point for running one or all variants."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow `python scripts/run_experiment.py` to import the src/ package without install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ExperimentConfig  # noqa: E402
from src.data import VARIANT_NAMES  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run a bio-refusal-classifier experiment.")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--dataset_variant",
        choices=VARIANT_NAMES,
        help="Variant to run (one of the four registered variants).",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Run all four variants and print a cross-variant comparison.",
    )
    p.add_argument("--embedding_model", default=None, help="Override embedding model name.")
    p.add_argument(
        "--classifier_kwargs",
        default=None,
        help="JSON dict merged into classifier_kwargs.",
    )
    p.add_argument("--seed", type=int, default=None, help="Set random_seed (and split_seed if not overridden).")
    p.add_argument("--split_seed", type=int, default=None, help="Override split_seed only.")
    p.add_argument("--train_frac", type=float, default=None)
    p.add_argument("--val_frac", type=float, default=None)
    p.add_argument("--test_frac", type=float, default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--experiment_name", default=None)
    p.add_argument("--config", default=None, help="Load base config from JSON path; CLI flags override.")
    return p


def _overrides_from_args(args: argparse.Namespace) -> dict:
    o: dict = {}
    if args.embedding_model is not None:
        o["embedding_model_name"] = args.embedding_model
    if args.seed is not None:
        o["random_seed"] = args.seed
        if args.split_seed is None:
            o["split_seed"] = args.seed
    if args.split_seed is not None:
        o["split_seed"] = args.split_seed
    if args.train_frac is not None:
        o["train_frac"] = args.train_frac
    if args.val_frac is not None:
        o["val_frac"] = args.val_frac
    if args.test_frac is not None:
        o["test_frac"] = args.test_frac
    if args.output_dir is not None:
        o["output_dir"] = args.output_dir
    if args.experiment_name is not None:
        o["experiment_name"] = args.experiment_name
    if args.classifier_kwargs is not None:
        kwargs = json.loads(args.classifier_kwargs)
        # merge over defaults
        base = ExperimentConfig(dataset_variant="balanced_easy").classifier_kwargs
        merged = {**base, **kwargs}
        o["classifier_kwargs"] = merged
    return o


def main() -> int:
    args = build_parser().parse_args()
    overrides = _overrides_from_args(args)

    from src import evaluate, train
    from src.experiments import make_config, run_all

    if args.all:
        run_all(overrides)
        return 0

    if args.config:
        cfg = ExperimentConfig.from_json(args.config)
        for k, v in overrides.items():
            setattr(cfg, k, v)
        cfg.dataset_variant = args.dataset_variant
    else:
        cfg = make_config(args.dataset_variant, overrides)

    model, splits = train.run(cfg)
    metrics = evaluate.run(model, splits, cfg)
    evaluate.print_summary(metrics, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
