# CLAUDE.md

Project conventions for Claude Code sessions on this repo. Read this before making changes.

## Project intent

Binary biology-refusal classifier. WMDP-bio questions (refuse) vs. benign prompts (Dolly easy / MMLU bio hard). v1 is a sentence-transformer + sklearn LR baseline. The infrastructure is deliberately modular — future seams (TF-IDF, fine-tuned encoders, activation probes, trajectory classifiers) plug into the existing `Classifier` protocol and `VARIANTS` registry without changes to `train.py` / `evaluate.py`.

## Two load-bearing design principles

1. **Global-pool splits across variants.** Each source dataset is split once with a fixed seed; variants compose from those frozen pools. A given example's split is invariant across variants. Do not break this — silent cross-variant leakage would invalidate sweeps and comparisons. The invariant test: WMDP-test IDs are identical across all four primary variants for any given seed.
2. **WMDP is the limit.** WMDP-bio is the rarest data (~1,273 rows). Every variant uses *all* WMDP available in each split's pool; the benign side is sampled to hit the target ratio. If you add a new variant or change ratios, follow the same rule.

## Primary vs. secondary variants

`src/data.py:VARIANTS` registers six variants. `src/data.py:PRIMARY_VARIANTS` lists only the four MMLU-only ones; `experiments.py:run_all` and the dashboard iterate over PRIMARY only. The two `*_dolly` variants are runnable individually (`--dataset_variant balanced_easy_dolly`) but excluded from the default sweep. v2 made this distinction because the project's headline story is MMLU-only (format-consistent negatives); Dolly is kept around as a deployment-realism comparison option. If you add a new "primary" variant, register it in both `VARIANTS` and `PRIMARY_VARIANTS`.

## Package management: uv

- **Use uv for everything** (`uv add`, `uv run`, `uv sync`). Do **not** introduce `pip install` or a `requirements.txt`.
- The venv inherits **system torch** via `uv venv --system-site-packages`. The GPU is RTX 5090 (sm_120 / Blackwell); generic PyPI torch wheels are not built for this capability and will fail at kernel launch.
- `pyproject.toml` contains `override-dependencies = ["torch ; sys_platform == 'never'"]` to stop uv from resolving torch transitively. Leave this alone.
- If `sentence-transformers` (or any other dep) ever tries to clobber torch on `uv add`, fix the override; do not delete it.

## File system

- `/workspace` is persistent across pod resets. `/root` is transient.
- All artifacts go under `/workspace/bio-refusal-classifier/`. HF cache is at `$HF_HOME=/workspace/.cache/huggingface/`.
- `outputs/*` is gitignored except `outputs/splits/`, which is committed so split partitions are reproducible.

## Plan archiving convention

After any Claude Code plan-mode session, copy the plan from `/root/.claude/plans/<slug>.md` into `plans/NNN_<descriptive_name>.md`:

- **Prefix:** zero-padded 3-digit number. The next prefix is `max(existing) + 1`; do not reuse numbers even if a plan is superseded.
- **Name:** short, descriptive, snake_case, reflecting the actual contents of the plan.
- Example: this v1 build's plan is `plans/001_v1_baseline_scaffold.md`.

`/root` does not survive pod restarts; `/workspace` does. The archive is what future sessions will read.

## PROGRESS.md convention

`PROGRESS.md` is a living document tracking project state and experiment results. **Update it after every experiment run.** Each row in the experiment log should record: timestamp, variant, model, test accuracy, refuse recall, dont_refuse recall, artifact path, and one-line notes on anything notable (failure modes, surprises, sanity-check failures).

`evaluate.print_summary` prints `> remember to update PROGRESS.md` at the end of every run — heed it.

Treat PROGRESS.md as load-bearing context: a future agent (human or AI) should be able to scan it and know what's been tried, what worked, what failed, and what's next.

## Plan experiments before running them

Any new experimental direction (new model type, new dataset variant, new hyperparameter sweep, new probing methodology) gets a plan in `plans/NNN_<descriptive_name>.md` **before execution**. The plan documents motivation, hypothesis, datasets, metrics, expected failure modes, and what success looks like — same shape as `plans/001_v1_baseline_scaffold.md`, which discussed the four v1 variants before they were run.

This is not "one plan per experiment, one experiment per plan" — a single plan can cover a coordinated batch (the four v1 variants were one plan). The point is that brainstorming and documentation happen *before* the run, not afterward in PROGRESS.md.

Reserved for follow-ups, not for trivial one-offs: fixing a bug, reframing a definition, regenerating artifacts after a code change, or updating docs do not need their own plan. Use judgment; when in doubt, write the plan.

## Extension seams

- **New classifier type:** add a class implementing the `Classifier` protocol in `src/models.py`, then extend `build_classifier(cfg)` in the same file.
- **New variant or source:** add a `load_*` function and `_LOADERS` entry in `src/data.py`, plus an entry in `VARIANTS`. The global-pool split routine and `build_variant` will pick it up automatically.
- **New evaluation slice:** extend `_per_split_metrics` and the artifact writers in `src/evaluate.py`. Keep the existing artifact filenames stable — downstream consumers (PROGRESS.md updates, comparison tables, the dashboard) depend on them.
- **Dashboard:** `dashboard/` is a static HTML+JS visualisation of the confusion matrices, fed by `dashboard/data.js` (generated from `outputs/<variant>__*/predictions.csv` + `metrics.json` by `dashboard/build_data.py`). Rebuild after every `--all` run: `uv run python dashboard/build_data.py`. Per-question heuristic summaries persist in `outputs/question_summaries.json` so reruns are deterministic.

## Output artifact contract (do not break)

Per-run, under `outputs/<experiment_name>/`:

- `config.json`, `metrics.json`, `classification_report.txt`, `confusion_matrix.png`, `predictions.csv`, `false_positives.csv`, `false_negatives.csv`.
- `metrics.json` always includes both `test_*` and `val_*` versions of accuracy / per-class P/R/F1 / confusion matrix, plus `n_train`, `n_val`, `n_test`, `n_*_refuse`, and `realized_refuse_fraction`.

## Things to verify, not assume

- **Environmental claims from agent reports.** Disk quotas, package versions, GPU availability, dataset row counts — verify by direct inspection (`df`, `nvidia-smi`, `uv pip list`, `len(dataset)`) rather than trusting a summary. (Lesson from v1 build: an exploration agent reported 351 TB free; the real local quota was 50 GB.)
- **Realized refuse fractions are the ground truth.** Some variants declare `r=None` (use all benign), in which case the realized refuse fraction is dictated by pool sizes — e.g., `imbalanced_hard` over MMLU bio realizes ~0.38. Always trust `realized_refuse_fraction` in `config.json` / `metrics.json` rather than inferring from the variant name.
