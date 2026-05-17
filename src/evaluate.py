"""Evaluation flow: metrics, plots, CSVs, terminal summary."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from .config import ExperimentConfig  # noqa: E402
from .models import Classifier  # noqa: E402
from .utils import ensure_dir, save_json  # noqa: E402

LABEL_NAMES = ["dont_refuse", "refuse"]


def _predict_on_split(model: Classifier, df: pd.DataFrame) -> pd.DataFrame:
    texts = df["text"].tolist()
    proba = model.predict_proba(texts)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return df.assign(pred_label=pred, pred_score=proba)


def _per_split_metrics(df: pd.DataFrame) -> dict:
    y_true = df["label"].to_numpy()
    y_pred = df["pred_label"].to_numpy()
    rep = classification_report(
        y_true, y_pred, labels=[0, 1], target_names=LABEL_NAMES, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "dont_refuse_precision": float(rep["dont_refuse"]["precision"]),
        "dont_refuse_recall": float(rep["dont_refuse"]["recall"]),
        "dont_refuse_f1": float(rep["dont_refuse"]["f1-score"]),
        "refuse_precision": float(rep["refuse"]["precision"]),
        "refuse_recall": float(rep["refuse"]["recall"]),
        "refuse_f1": float(rep["refuse"]["f1-score"]),
        "confusion_matrix": cm,
        "n": int(len(df)),
        "n_refuse": int(y_true.sum()),
    }


def run(model: Classifier, splits: dict[str, pd.DataFrame], cfg: ExperimentConfig) -> dict:
    out_dir = ensure_dir(cfg.output_dir)

    val_pred = _predict_on_split(model, splits["val"])
    test_pred = _predict_on_split(model, splits["test"])

    val_m = _per_split_metrics(val_pred)
    test_m = _per_split_metrics(test_pred)

    metrics = {
        "test_accuracy": test_m["accuracy"],
        "test_refuse_precision": test_m["refuse_precision"],
        "test_refuse_recall": test_m["refuse_recall"],
        "test_refuse_f1": test_m["refuse_f1"],
        "test_dont_refuse_precision": test_m["dont_refuse_precision"],
        "test_dont_refuse_recall": test_m["dont_refuse_recall"],
        "test_dont_refuse_f1": test_m["dont_refuse_f1"],
        "test_confusion_matrix": test_m["confusion_matrix"],
        "val_accuracy": val_m["accuracy"],
        "val_refuse_precision": val_m["refuse_precision"],
        "val_refuse_recall": val_m["refuse_recall"],
        "val_refuse_f1": val_m["refuse_f1"],
        "val_dont_refuse_precision": val_m["dont_refuse_precision"],
        "val_dont_refuse_recall": val_m["dont_refuse_recall"],
        "val_dont_refuse_f1": val_m["dont_refuse_f1"],
        "val_confusion_matrix": val_m["confusion_matrix"],
        "n_train": int(len(splits["train"])),
        "n_val": int(len(splits["val"])),
        "n_test": int(len(splits["test"])),
        "n_train_refuse": int(splits["train"]["label"].sum()),
        "n_val_refuse": val_m["n_refuse"],
        "n_test_refuse": test_m["n_refuse"],
        "realized_refuse_fraction": cfg.realized_refuse_fraction,
    }

    # Sanity guard: realized fraction should match what build_variant returned.
    if cfg.realized_refuse_fraction is not None:
        test_r = test_m["n_refuse"] / max(test_m["n"], 1)
        diff = abs(test_r - cfg.realized_refuse_fraction.get("test", test_r))
        assert diff < 0.005, f"refuse fraction mismatch: {test_r:.4f} vs cfg {cfg.realized_refuse_fraction}"

    cfg.stamp_timestamp()
    save_json(asdict(cfg), Path(out_dir, "config.json"))
    save_json(metrics, Path(out_dir, "metrics.json"))

    _write_classification_report(test_pred, val_pred, out_dir, cfg)
    _write_confusion_matrix_png(test_m["confusion_matrix"], out_dir)
    _write_predictions_csv(val_pred, test_pred, out_dir)
    _write_failure_csvs(test_pred, out_dir)

    return metrics


def _write_classification_report(
    test_pred: pd.DataFrame, val_pred: pd.DataFrame, out_dir: Path, cfg: ExperimentConfig
) -> None:
    lines = []
    lines.append(f"experiment: {cfg.experiment_name}")
    lines.append(f"variant:    {cfg.dataset_variant}")
    lines.append(f"model:      {cfg.model_type} / {cfg.embedding_model_name}")
    lines.append(f"seeds:      random={cfg.random_seed} split={cfg.split_seed}")
    lines.append("")
    lines.append("=== TEST ===")
    lines.append(
        classification_report(
            test_pred["label"], test_pred["pred_label"], labels=[0, 1], target_names=LABEL_NAMES,
            zero_division=0,
        )
    )
    cm = confusion_matrix(test_pred["label"], test_pred["pred_label"], labels=[0, 1])
    lines.append("confusion matrix (rows=true, cols=pred):")
    lines.append("                 pred=dont_refuse  pred=refuse")
    lines.append(f"true=dont_refuse {cm[0, 0]:>15d} {cm[0, 1]:>12d}")
    lines.append(f"true=refuse      {cm[1, 0]:>15d} {cm[1, 1]:>12d}")
    lines.append("")
    lines.append("=== VAL ===")
    lines.append(
        classification_report(
            val_pred["label"], val_pred["pred_label"], labels=[0, 1], target_names=LABEL_NAMES,
            zero_division=0,
        )
    )
    (out_dir / "classification_report.txt").write_text("\n".join(lines))


def _write_confusion_matrix_png(cm: list[list[int]], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=np.array(cm), display_labels=LABEL_NAMES)
    disp.plot(ax=ax, colorbar=False, cmap="Blues", values_format="d")
    ax.set_title("Test confusion matrix")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def _write_predictions_csv(
    val_pred: pd.DataFrame, test_pred: pd.DataFrame, out_dir: Path
) -> None:
    df = pd.concat([val_pred, test_pred], ignore_index=True)
    df = df.rename(columns={"label": "true_label", "source": "source_dataset"})
    df[
        [
            "id",
            "text",
            "source_dataset",
            "split",
            "true_label",
            "pred_label",
            "pred_score",
        ]
    ].to_csv(out_dir / "predictions.csv", index=False)


def _write_failure_csvs(test_pred: pd.DataFrame, out_dir: Path) -> None:
    fp = test_pred[(test_pred.label == 0) & (test_pred.pred_label == 1)].sort_values(
        "pred_score", ascending=False
    )
    fn = test_pred[(test_pred.label == 1) & (test_pred.pred_label == 0)].sort_values(
        "pred_score", ascending=True
    )
    cols = ["id", "text", "source", "split", "label", "pred_label", "pred_score"]
    fp[cols].to_csv(out_dir / "false_positives.csv", index=False)
    fn[cols].to_csv(out_dir / "false_negatives.csv", index=False)


def print_summary(metrics: dict, cfg: ExperimentConfig) -> None:
    cm = metrics["test_confusion_matrix"]
    print()
    print("=" * 72)
    print(f"  experiment: {cfg.experiment_name}")
    print(f"  variant:    {cfg.dataset_variant}")
    print(f"  model:      {cfg.model_type} / {cfg.embedding_model_name}")
    print(f"  seeds:      random={cfg.random_seed} split={cfg.split_seed}")
    rr = cfg.realized_refuse_fraction or {}
    print(
        f"  sizes:      train={metrics['n_train']} (refuse={metrics['n_train_refuse']}) "
        f"val={metrics['n_val']} test={metrics['n_test']}"
    )
    if rr:
        print(
            f"  realized r: train={rr.get('train', 0):.3f} val={rr.get('val', 0):.3f} "
            f"test={rr.get('test', 0):.3f}"
        )
    print()
    print(f"  TEST accuracy:                 {metrics['test_accuracy']:.4f}")
    print(
        f"  TEST refuse        P/R/F1:     {metrics['test_refuse_precision']:.3f} / "
        f"{metrics['test_refuse_recall']:.3f} / {metrics['test_refuse_f1']:.3f}"
    )
    print(
        f"  TEST dont_refuse   P/R/F1:     {metrics['test_dont_refuse_precision']:.3f} / "
        f"{metrics['test_dont_refuse_recall']:.3f} / {metrics['test_dont_refuse_f1']:.3f}"
    )
    print()
    print("  confusion matrix [rows=true]:")
    print("                pred=dont_refuse  pred=refuse")
    print(f"  dont_refuse  {cm[0][0]:>14d} {cm[0][1]:>12d}")
    print(f"  refuse       {cm[1][0]:>14d} {cm[1][1]:>12d}")
    print()
    print(f"  artifacts:  {cfg.output_dir}/")
    print("=" * 72)
    print("> remember to update PROGRESS.md")
    print()
