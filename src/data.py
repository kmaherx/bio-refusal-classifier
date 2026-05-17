"""Data loading, global splits, and variant composition.

Two design principles drive this module:

1. **Global-pool splits** — each source dataset is split train/val/test exactly
   once (per seed), and variants compose from those frozen pools. A given
   example's split is invariant across variants so cross-variant comparison
   and hyperparameter sweeps are leak-free.

2. **WMDP is the limit** — WMDP-bio (~1273 rows) is the rarest data, so every
   variant uses ALL WMDP available in its split pool. The benign class is
   sampled at ``n_refuse * (1 - r) / r`` where ``r`` is the target refuse
   fraction. A variant may declare ``r=None`` to mean "use all available
   benign data" — appropriate when the design intent is maximum coverage of
   a finite, high-quality negative source (e.g. MMLU bio) and the realized
   ratio is dictated by data availability. The realized refuse fraction is
   always recorded.

Negative sources for the **primary** variants are both drawn from MMLU so the
easy/hard contrast is purely topical (non-bio vs. bio); refuse and benign
share the same stripped-MCQ format. This is intentional methodological
control — see ``plans/002_swap_dolly_for_mmlu_non_bio.md`` for the trade-off
discussion.

Dolly is **maintained but secondary**: the loader and a pair of Dolly-easy
variants are present so anyone can compare against open-ended instruction
negatives, but they are excluded from the default ``--all`` sweep.

Future seams: ``load_source(name)`` and ``VARIANTS`` are the natural extension
points — add new sources (synthetic trajectories, activation tensors) and new
variant entries here.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import pandas as pd
from datasets import get_dataset_config_names, load_dataset

from .utils import ensure_dir, load_json, save_json

logger = logging.getLogger(__name__)


# --- source dataset names ----------------------------------------------------

SOURCE_WMDP = "wmdp"
SOURCE_MMLU_BIO = "mmlu_bio"
SOURCE_MMLU_OTHER = "mmlu_other"
SOURCE_DOLLY = "dolly"  # secondary; kept for legacy comparison

MMLU_BIO_SUBJECTS: list[str] = [
    "high_school_biology",
    "college_biology",
    "college_medicine",
    "medical_genetics",
    "virology",
    "anatomy",
    "clinical_knowledge",
    "professional_medicine",
    "nutrition",
]

# Aggregate configs in cais/mmlu we never want as subject pools.
_MMLU_AGGREGATE_CONFIGS = {"all", "auxiliary_train"}

_mmlu_other_subjects_cache: list[str] | None = None


def mmlu_other_subjects() -> list[str]:
    """All MMLU subjects except the 9 we use as hard negatives.

    Computed once per process from ``cais/mmlu``'s config list, with the
    aggregate ``all`` / ``auxiliary_train`` entries removed.
    """
    global _mmlu_other_subjects_cache
    if _mmlu_other_subjects_cache is None:
        all_configs = set(get_dataset_config_names("cais/mmlu"))
        _mmlu_other_subjects_cache = sorted(
            all_configs - set(MMLU_BIO_SUBJECTS) - _MMLU_AGGREGATE_CONFIGS
        )
    return _mmlu_other_subjects_cache


# --- loaders -----------------------------------------------------------------


def _clean(text: str) -> str:
    return " ".join((text or "").split())


def load_wmdp_bio() -> pd.DataFrame:
    """Load WMDP-bio questions only (answer choices dropped).

    The dataset ships as a single ``test`` split (it's eval-only); we treat each
    question as a refuse-class example.
    """
    d = load_dataset("cais/wmdp", "wmdp-bio")["test"]
    rows = []
    for i, ex in enumerate(d):
        text = _clean(ex["question"])
        if text:
            rows.append({"id": f"{SOURCE_WMDP}_{i:06d}", "text": text, "source": SOURCE_WMDP})
    return pd.DataFrame(rows)


def _load_mmlu_subjects(subjects: list[str], source_name: str) -> pd.DataFrame:
    """Shared loader for any subset of MMLU subjects.

    Combines each subject's ``validation`` + ``test`` splits into a single
    pool. ``dev`` (5 rows per subject) is skipped — it's used elsewhere for
    in-context prompting and adds noise here.

    IDs encode subject and origin split so future stratified splits can stay
    subject-balanced and so the dashboard can show subject in the badge.
    """
    rows = []
    for subj in subjects:
        d = load_dataset("cais/mmlu", subj)
        for split_name in ("validation", "test"):
            if split_name not in d:
                continue
            for i, ex in enumerate(d[split_name]):
                text = _clean(ex["question"])
                if not text:
                    continue
                rows.append(
                    {
                        "id": f"{source_name}_{subj}_{split_name}_{i:04d}",
                        "text": text,
                        "source": source_name,
                    }
                )
    return pd.DataFrame(rows)


def load_mmlu_bio() -> pd.DataFrame:
    """Hard-negative source: 9 bio/medical MMLU subjects, questions only."""
    return _load_mmlu_subjects(MMLU_BIO_SUBJECTS, SOURCE_MMLU_BIO)


def load_mmlu_other() -> pd.DataFrame:
    """Easy-negative source: every MMLU subject except the 9 bio/medical ones.

    Same stripped-MCQ format as WMDP and MMLU bio — easy vs. hard becomes a
    pure topical contrast, with format held constant. See module docstring.
    """
    return _load_mmlu_subjects(mmlu_other_subjects(), SOURCE_MMLU_OTHER)


def load_dolly() -> pd.DataFrame:
    """Secondary easy-negative source: Dolly-15k instructions.

    Open-ended user-style instructions (different format from WMDP/MMLU). Kept
    available for comparison against the primary MMLU-non-bio negatives but
    not part of the default ``--all`` sweep — see the ``LEGACY_DOLLY_VARIANTS``
    note below.
    """
    d = load_dataset("databricks/databricks-dolly-15k")["train"]
    rows = []
    for i, ex in enumerate(d):
        text = _clean(ex["instruction"])
        if text:
            rows.append({"id": f"{SOURCE_DOLLY}_{i:06d}", "text": text, "source": SOURCE_DOLLY})
    return pd.DataFrame(rows)


_LOADERS = {
    SOURCE_WMDP: load_wmdp_bio,
    SOURCE_MMLU_BIO: load_mmlu_bio,
    SOURCE_MMLU_OTHER: load_mmlu_other,
    SOURCE_DOLLY: load_dolly,
}


def load_source(name: str) -> pd.DataFrame:
    if name not in _LOADERS:
        raise KeyError(f"unknown source {name!r}; known: {sorted(_LOADERS)}")
    return _LOADERS[name]()


_SOURCE_CACHE: dict[str, pd.DataFrame] = {}


def load_all_sources() -> dict[str, pd.DataFrame]:
    """In-process cache so repeated calls don't re-tokenize."""
    for name in _LOADERS:
        if name not in _SOURCE_CACHE:
            _SOURCE_CACHE[name] = load_source(name)
    return _SOURCE_CACHE


# --- global splits -----------------------------------------------------------


def _splits_filename(seed: int, fracs: tuple[float, float, float]) -> str:
    a, b, c = fracs
    return f"global_splits__seed{seed}__fracs{a}-{b}-{c}.json"


def get_global_splits(
    seed: int,
    fracs: tuple[float, float, float],
    cache_dir: str | Path,
) -> dict:
    """Return ``{source: {split: [ids]}}``, computing or reading from cache.

    Per-source: stable shuffle with ``random.Random(seed)``, slice by fracs.
    Within a single source the class label is constant, so per-source
    stratification is moot; class stratification happens at variant
    composition time via the refuse fraction.
    """
    cache_dir = ensure_dir(cache_dir)
    path = cache_dir / _splits_filename(seed, fracs)
    if path.exists():
        return load_json(path)

    if abs(sum(fracs) - 1.0) > 1e-6:
        raise ValueError(f"fracs must sum to 1.0; got {fracs}")

    sources = load_all_sources()
    splits: dict[str, dict[str, list[str]]] = {}
    pool_sizes: dict[str, dict[str, int]] = {}
    for name, df in sources.items():
        ids = df["id"].tolist()
        rng_local = random.Random(f"{seed}:{name}")
        rng_local.shuffle(ids)
        n = len(ids)
        n_train = int(round(n * fracs[0]))
        n_val = int(round(n * fracs[1]))
        # remainder is test so totals always equal n
        train = ids[:n_train]
        val = ids[n_train : n_train + n_val]
        test = ids[n_train + n_val :]
        splits[name] = {"train": train, "val": val, "test": test}
        pool_sizes[name] = {"train": len(train), "val": len(val), "test": len(test)}

    payload = {
        "seed": seed,
        "fracs": list(fracs),
        "sources": splits,
        "pool_sizes": pool_sizes,
    }
    save_json(payload, path)
    return payload


# --- variant composition -----------------------------------------------------


# (refuse_source, benign_source, refuse_fraction)
#
# refuse_fraction = None  →  use ALL available benign data; realized ratio is
# dictated by source pool sizes. Used when a finite, high-quality negative
# source (e.g. MMLU bio) is the limit and we'd rather take the full pool than
# downsample.
VARIANTS: dict[str, tuple[str, str, float | None]] = {
    # --- primary (MMLU-only) ---
    "balanced_easy": (SOURCE_WMDP, SOURCE_MMLU_OTHER, 0.5),
    "imbalanced_easy": (SOURCE_WMDP, SOURCE_MMLU_OTHER, 0.1),
    "balanced_hard": (SOURCE_WMDP, SOURCE_MMLU_BIO, 0.5),
    # Use all MMLU bio; realized ratio is ~0.38 against WMDP (data quality > target ratio).
    "imbalanced_hard": (SOURCE_WMDP, SOURCE_MMLU_BIO, None),
    # --- secondary (Dolly easy negatives) ---
    # Available via `--dataset_variant <name>` but excluded from `--all`.
    "balanced_easy_dolly": (SOURCE_WMDP, SOURCE_DOLLY, 0.5),
    "imbalanced_easy_dolly": (SOURCE_WMDP, SOURCE_DOLLY, 0.1),
}

VARIANT_NAMES = list(VARIANTS)

# `--all` only sweeps these. Dolly variants are runnable individually but the
# default story is MMLU-only — see plans/002_swap_dolly_for_mmlu_non_bio.md.
PRIMARY_VARIANTS = [
    "balanced_easy",
    "imbalanced_easy",
    "balanced_hard",
    "imbalanced_hard",
]


def _derived_rng(seed: int, variant: str, split: str) -> random.Random:
    # Stable, distinct per-(variant,split) RNG for reproducible sampling.
    # Use a string key so determinism is independent of Python's hash randomization.
    return random.Random(f"{seed}:{variant}:{split}")


def build_variant(
    name: str,
    split: str,
    seed: int,
    global_splits: dict,
    sources: dict[str, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, float]:
    """Return ``(rows, realized_refuse_fraction)`` for a variant/split.

    Uses ALL refuse IDs in the split's pool. For the benign class:

    - ``r is None``: use the entire benign pool. Realized ratio reflects the
      relative sizes of the refuse and benign pools.
    - ``r`` is a float: sample ``n_refuse * (1 - r) / r`` benign rows to hit
      the target. If the benign pool is smaller than the target, take the full
      pool and log; the realized ratio will be reported regardless.
    """
    if name not in VARIANTS:
        raise KeyError(f"unknown variant {name!r}; known: {VARIANT_NAMES}")
    refuse_src, benign_src, r = VARIANTS[name]

    if sources is None:
        sources = load_all_sources()

    src_splits = global_splits["sources"]
    refuse_ids = src_splits[refuse_src][split]
    benign_pool = src_splits[benign_src][split]

    n_refuse = len(refuse_ids)
    rng = _derived_rng(seed, name, split)

    if r is None:
        # Use all benign data; ratio is whatever pool sizes dictate.
        benign_ids = list(benign_pool)
        rng.shuffle(benign_ids)
    else:
        target_n_benign = int(round(n_refuse * (1 - r) / r))
        if target_n_benign > len(benign_pool):
            logger.warning(
                "variant=%s split=%s: benign pool (%d) < target for r=%.3f (%d); "
                "using full pool — realized ratio will differ from target",
                name,
                split,
                len(benign_pool),
                r,
                target_n_benign,
            )
            benign_ids = list(benign_pool)
            rng.shuffle(benign_ids)
        else:
            benign_ids = rng.sample(list(benign_pool), target_n_benign)

    realized_r = n_refuse / (n_refuse + len(benign_ids)) if (n_refuse + len(benign_ids)) else 0.0

    refuse_df = sources[refuse_src].set_index("id").loc[refuse_ids].reset_index()
    benign_df = sources[benign_src].set_index("id").loc[benign_ids].reset_index()

    refuse_df = refuse_df.assign(label=1, split=split)
    benign_df = benign_df.assign(label=0, split=split)

    combined = pd.concat([refuse_df, benign_df], ignore_index=True)
    # Deterministic shuffle so positives and negatives are interleaved.
    combined = combined.sample(frac=1.0, random_state=rng.randint(0, 2**31 - 1)).reset_index(
        drop=True
    )
    return combined[["id", "text", "source", "label", "split"]], realized_r


def build_all_splits(
    variant: str, seed: int, global_splits: dict
) -> tuple[dict[str, pd.DataFrame], dict[str, float]]:
    sources = load_all_sources()
    out_data: dict[str, pd.DataFrame] = {}
    realized: dict[str, float] = {}
    for split in ("train", "val", "test"):
        df, r = build_variant(variant, split, seed, global_splits, sources=sources)
        out_data[split] = df
        realized[split] = r
    return out_data, realized
