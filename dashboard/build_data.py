#!/usr/bin/env python3
"""Generate `dashboard/data.js` and `outputs/question_summaries.json`.

Reads the four experiment dirs under `outputs/`, extracts test-set predictions
and confusion matrices, builds heuristic question summaries, and emits a single
`window.DASHBOARD_DATA = {...}` JS file that the dashboard reads directly.

Idempotent: existing summaries in `outputs/question_summaries.json` are reused.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
DASHBOARD_DIR = ROOT / "dashboard"
SUMMARIES_PATH = OUTPUTS / "question_summaries.json"
DATA_JS_PATH = DASHBOARD_DIR / "data.js"

# Mapping from raw variant key to a friendly display name shown in the UI.
DISPLAY_NAMES = {
    "balanced_easy": "Balanced × Easy",
    "imbalanced_easy": "Imbalanced × Easy",
    "balanced_hard": "Balanced × Hard",
    "imbalanced_hard": "Imbalanced × Hard",
}
EXPERIMENT_ORDER = ["balanced_easy", "imbalanced_easy", "balanced_hard", "imbalanced_hard"]

# MMLU IDs look like `mmlu_bio_<subject>_<split>_<idx>` where <subject> can
# itself contain underscores (e.g. high_school_biology). Pattern: strip the
# leading prefix, then peel off the last two tokens (split + idx).
_MMLU_PREFIX = "mmlu_bio_"
_MMLU_TAIL = re.compile(r"_(?:validation|test)_\d+$")


def parse_mmlu_subject(qid: str) -> str | None:
    if not qid.startswith(_MMLU_PREFIX):
        return None
    rest = qid[len(_MMLU_PREFIX) :]
    m = _MMLU_TAIL.search(rest)
    if not m:
        return rest
    return rest[: m.start()]


def make_summary(source: str, subject: str | None, text: str, max_words: int = 10) -> str:
    """First ~10 words of the question, ellipsised, prefixed with source tag."""
    prefix = f"[{source}/{subject}]" if subject else f"[{source}]"
    words = text.split()
    if len(words) <= max_words:
        body = " ".join(words)
    else:
        body = " ".join(words[:max_words]) + "…"
    return f"{prefix} {body}"


def load_existing_summaries() -> dict[str, dict[str, Any]]:
    if SUMMARIES_PATH.exists():
        return json.loads(SUMMARIES_PATH.read_text())
    return {}


def discover_experiment_dirs() -> dict[str, Path]:
    """Return ``{variant_key: dir}`` for the experiment dirs we recognize.

    Picks the most recently modified dir per variant if multiple match.
    """
    out: dict[str, Path] = {}
    for variant in EXPERIMENT_ORDER:
        candidates = sorted(
            OUTPUTS.glob(f"{variant}__*__seed*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            out[variant] = candidates[0]
    return out


def build_experiment(
    variant: str,
    exp_dir: Path,
    questions: dict[str, dict[str, Any]],
    summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    metrics = json.loads((exp_dir / "metrics.json").read_text())
    config = json.loads((exp_dir / "config.json").read_text())

    df = pd.read_csv(exp_dir / "predictions.csv")
    df = df[df["split"] == "test"].copy()
    df["true_label"] = df["true_label"].astype(int)
    df["pred_label"] = df["pred_label"].astype(int)

    cells: dict[str, list[dict[str, Any]]] = {f"{t}_{p}": [] for t in (0, 1) for p in (0, 1)}
    for _, row in df.sort_values("pred_score", ascending=False).iterrows():
        qid = row["id"]
        if qid not in questions:
            source = row["source_dataset"]
            subject = parse_mmlu_subject(qid) if source == "mmlu_bio" else None
            text = row["text"]
            cached = summaries.get(qid)
            if cached and cached.get("text") == text:
                summary = cached["summary"]
            else:
                summary = make_summary(source, subject, text)
            entry = {
                "source": source,
                "subject": subject,
                "summary": summary,
                "text": text,
            }
            questions[qid] = entry
            summaries[qid] = entry

        cells[f"{row['true_label']}_{row['pred_label']}"].append(
            {"id": qid, "score": float(row["pred_score"])}
        )

    realized_r = (config.get("realized_refuse_fraction") or {}).get("test")

    return {
        "key": variant,
        "display": DISPLAY_NAMES.get(variant, variant),
        "n_test": metrics["n_test"],
        "realized_r": realized_r,
        "matrix": metrics["test_confusion_matrix"],
        "headline": {
            "accuracy": metrics["test_accuracy"],
            "refuse_recall": metrics["test_refuse_recall"],
            "dont_refuse_recall": metrics["test_dont_refuse_recall"],
        },
        "cells": cells,
    }


def main() -> int:
    exp_dirs = discover_experiment_dirs()
    missing = [v for v in EXPERIMENT_ORDER if v not in exp_dirs]
    if missing:
        print(
            f"warning: no experiment dir found for variants {missing}. "
            f"Run `uv run python scripts/run_experiment.py --all` first.",
            file=sys.stderr,
        )

    summaries = load_existing_summaries()
    questions: dict[str, dict[str, Any]] = {}
    experiments: list[dict[str, Any]] = []
    for variant in EXPERIMENT_ORDER:
        if variant not in exp_dirs:
            continue
        experiments.append(build_experiment(variant, exp_dirs[variant], questions, summaries))

    # Sanity guard: confusion matrix counts must equal the cell list lengths.
    for exp in experiments:
        m = exp["matrix"]
        for t in (0, 1):
            for p in (0, 1):
                expected = m[t][p]
                actual = len(exp["cells"][f"{t}_{p}"])
                if expected != actual:
                    raise AssertionError(
                        f"{exp['key']} cell ({t},{p}) length mismatch: "
                        f"matrix={expected} cells={actual}"
                    )

    payload = {"experiments": experiments, "questions": questions}

    DASHBOARD_DIR.mkdir(exist_ok=True)
    DATA_JS_PATH.write_text(
        "// Generated by dashboard/build_data.py — do not hand-edit.\n"
        "window.DASHBOARD_DATA = "
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + ";\n"
    )
    SUMMARIES_PATH.write_text(json.dumps(summaries, ensure_ascii=False, indent=2, sort_keys=True))

    n_q = len(questions)
    n_exp = len(experiments)
    data_kb = DATA_JS_PATH.stat().st_size / 1024
    print(
        f"wrote {DATA_JS_PATH.relative_to(ROOT)} ({data_kb:.1f} KB) — "
        f"{n_exp} experiments, {n_q} unique questions"
    )
    print(f"wrote {SUMMARIES_PATH.relative_to(ROOT)} ({len(summaries)} summaries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
