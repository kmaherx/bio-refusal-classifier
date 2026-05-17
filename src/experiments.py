"""Variant registry + multi-variant sweep."""

from __future__ import annotations

from typing import Any

from .config import ExperimentConfig
from .data import VARIANT_NAMES

__all__ = ["VARIANT_NAMES", "make_config", "run_all"]


def make_config(variant: str, overrides: dict[str, Any] | None = None) -> ExperimentConfig:
    overrides = dict(overrides or {})
    overrides["dataset_variant"] = variant
    return ExperimentConfig(**overrides)


def run_all(base_overrides: dict[str, Any] | None = None) -> dict[str, dict]:
    """Run all four variants in order and print a comparison table.

    Returns ``{variant: metrics}``.
    """
    from . import evaluate, train

    base_overrides = dict(base_overrides or {})
    base_overrides.pop("dataset_variant", None)  # don't let caller pin this

    results: dict[str, dict] = {}
    for variant in VARIANT_NAMES:
        cfg = make_config(variant, base_overrides)
        model, splits = train.run(cfg)
        metrics = evaluate.run(model, splits, cfg)
        evaluate.print_summary(metrics, cfg)
        results[variant] = metrics

    _print_comparison(results)
    return results


def _print_comparison(results: dict[str, dict]) -> None:
    print()
    print("=" * 86)
    print("  CROSS-VARIANT COMPARISON  (test set)")
    print("=" * 86)
    print(
        f"  {'variant':<20s}  {'accuracy':>9s}  {'refuse_recall':>13s}  "
        f"{'dont_refuse_recall':>18s}  {'realized_r_test':>16s}"
    )
    print("-" * 86)
    for variant, m in results.items():
        rr = (m.get("realized_refuse_fraction") or {}).get("test", float("nan"))
        print(
            f"  {variant:<20s}  {m['test_accuracy']:>9.4f}  "
            f"{m['test_refuse_recall']:>13.4f}  "
            f"{m['test_dont_refuse_recall']:>18.4f}  "
            f"{rr:>16.4f}"
        )
    print("=" * 86)
    print()
