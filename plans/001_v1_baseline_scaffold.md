# Plan: Build `bio-refusal-classifier`

## Context

Greenfield build of a take-home repo at `/workspace/bio-refusal-classifier/`. The spec lives in `/workspace/notes/initial_plan.md` (with supporting notes in `notes.txt` and `task.pdf`); the spec already captures two non-trivial design principles agreed earlier this session: (1) **global-pool splits** — each source dataset is split once with a fixed seed and variants compose from frozen pools so a given example's split is invariant across variants; (2) **WMDP is the limit** — always use all WMDP at each split level and sample benign at `n_refuse * (1-r) / r`.

V1 deliverable: an embedding (`sentence-transformers/all-MiniLM-L6-v2`) + scikit-learn logistic regression baseline run across four dataset variants (balanced/imbalanced × easy/hard negatives), with rich per-experiment artifacts. Infrastructure is modular so future seams (autoregressive sidecars, activation probes, trajectory classifiers) slot in without surgery.

Environment confirmed: Python 3.12.3, `uv 0.9.0` at `/usr/bin/uv` (use uv per user request), system torch 2.8.0+cu128 built for RTX 5090 (Blackwell sm_120), `HF_HOME=/workspace/.cache/huggingface/`, network to HF works, **~50 GB local quota under `/workspace`** (the earlier 351 TB figure was a clustered-filesystem total, not a local budget). `/workspace/` is empty except `/workspace/notes/`. The system torch is critical — generic PyPI torch wheels will not run on sm_120; the venv must inherit system site-packages so it can see torch without uv trying to manage its own.

Disk budget consequence: HF caches must stay lean. MiniLM (~80 MB), small BERT/DistilBERT (~250 MB), WMDP-bio (<5 MB), Dolly-15k (~13 MB), and the 9 MMLU bio subjects (<20 MB total) sum to well under 1 GB. Avoid pulling PubMedQA-artificial (~few GB) — see hard-negative source choice below.

This session's plan file (`/root/.claude/plans/great-ok-well-you-buzzing-coral.md`) will also be copied to `/workspace/bio-refusal-classifier/plans/` for persistence — `/root` is transient, `/workspace` survives pod resets.

---

## Repo layout

```
/workspace/bio-refusal-classifier/
  pyproject.toml          # uv-managed; pins everything except torch
  uv.lock
  README.md               # task, install, run, output layout, interpretation
  CLAUDE.md               # project conventions for future Claude Code sessions
  PROGRESS.md             # living TODO list + experiment log; updated after every run
  .gitignore              # .venv, __pycache__, outputs/* except outputs/splits
  plans/                  # archived copies of plan files from /root/.claude/plans/
  src/
    __init__.py
    config.py             # ExperimentConfig dataclass + JSON (de)serialization
    data.py               # load_source / get_global_splits / build_variant + VARIANTS
    models.py             # Classifier protocol + EmbeddingLogReg + build_classifier
    train.py              # run(cfg) -> (model, splits)
    evaluate.py           # run(model, splits, cfg) -> metrics; writes artifacts
    experiments.py        # VARIANT_NAMES, make_config, run_all
    utils.py              # set_seed, ensure_dir, save_json/load_json, get_device
  scripts/
    run_experiment.py     # CLI: --dataset_variant / --all / overrides
  outputs/
    splits/               # global_splits__seed42__fracs0.8-0.1-0.1.json
    <experiment_name>/    # per-run artifacts (see Evaluate flow)
```

---

## Build steps (in order)

### Step 0 — De-risk HF dataset shapes before writing code

Spend a few minutes in a REPL via `uv run python` to confirm:

- `cais/wmdp`, config `wmdp-bio`, split `test`, field `question` (drop `choices`, `answer`). Expected ~1273 rows. WMDP follows the MMLU multiple-choice format by design.
- `databricks/databricks-dolly-15k`, split `train`, field `instruction`. Expected ~15011 rows.
- `cais/mmlu`, configs (one per subject): `high_school_biology`, `college_biology`, `college_medicine`, `medical_genetics`, `virology`, `anatomy`, `clinical_knowledge`, `professional_medicine`, `nutrition`. Field `question` (drop `choices`, `answer`). Splits per subject include `dev` (~5 rows), `validation` (~10–30 rows), `test` (most). **Combine all 9 subjects across `validation` + `test` (skip `dev` because it's tiny and used for in-context prompting) into a single hard-negative pool.** Each row's `id` should encode subject so we can stratify by subject in the global split. Expected combined size: ~1800–2100 rows — verify exact counts because this drives the realized imbalanced_hard ratio.

If any of these shapes differ, fix loaders before moving on. This is the riskiest unknown.

**Hard-negative source rationale:** MMLU bio/medical subjects are a much better methodological match for WMDP than PubMedQA — WMDP is explicitly built in MMLU's 4-option-multiple-choice format with similar academic phrasing, so MMLU questions stripped of choices look structurally identical to WMDP questions stripped of choices. This gives a fairer "is the model picking up on bio-risk semantics or just on question-format style?" test. Trade-off: MMLU has less volume than PubMedQA — see imbalanced_hard handling below.

### Step 1 — Project init with uv

```
cd /workspace
uv init --python 3.12 --no-readme bio-refusal-classifier
cd /workspace/bio-refusal-classifier
uv venv --python 3.12 --system-site-packages    # inherit system torch
uv add datasets transformers sentence-transformers scikit-learn pandas matplotlib numpy
uv add --group dev pytest ruff
```

Add to `pyproject.toml`:

```toml
[tool.uv]
override-dependencies = ["torch ; sys_platform == 'never'"]   # rely on system torch
```

Smoke test:

```
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expect `2.8.0+cu128 True NVIDIA GeForce RTX 5090`. If `sentence-transformers` re-pulls torch despite the override, fall back to `--no-deps` install for that one package.

`.gitignore`: `.venv/`, `__pycache__/`, `*.egg-info/`, `.ruff_cache/`, `outputs/*`, `!outputs/splits/`.

### Step 2 — `utils.py` and `config.py`

`ExperimentConfig` dataclass with fields: `experiment_name`, `dataset_variant`, `output_dir`, `model_type="embedding_logreg"`, `embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"`, `classifier_type="logistic_regression"`, `classifier_kwargs={"max_iter":1000,"C":1.0}`, `embedding_batch_size=64`, `normalize_embeddings=True`, `train_frac=0.8`, `val_frac=0.1`, `test_frac=0.1`, `random_seed=42`, `split_seed=42`, `splits_dir="outputs/splits"`, plus runtime-populated `pool_sizes`, `global_splits_file`, `timestamp`.

`utils.py` exposes `set_seed`, `ensure_dir`, `save_json`, `load_json`, `get_device`, `timer`.

Smoke: round-trip `ExperimentConfig` through `save_json` / `load_json` and check equality.

### Step 3 — `data.py` loaders

`load_wmdp_bio()`, `load_dolly()`, `load_pubmedqa()` each return `DataFrame[id: str, text: str, source: str]` where `id = f"{source}_{row_idx}"`, `text` is whitespace-trimmed, empty texts dropped.

Smoke: in REPL, call each, print row count, head, text-length describe; verify WMDP `text` has no answer choices.

### Step 4 — `data.py` global splits

`get_global_splits(seed, fracs, cache_dir) -> dict[source, dict[split, list[id]]]`. Cache to `{cache_dir}/global_splits__seed{seed}__fracs{a}-{b}-{c}.json` (atomic write via `.tmp`+rename). Shuffle per-source with `random.Random(seed)`, slice by fracs.

Smoke: call twice, confirm cache reuse and bit-identical output; eyeball WMDP pool sizes (~1018/127/128).

### Step 5 — `data.py` variant composition

```
VARIANTS = {
    "balanced_easy":   ("wmdp", "dolly", 0.5),
    "imbalanced_easy": ("wmdp", "dolly", 0.1),
    "balanced_hard":   ("wmdp", "mmlu_bio", 0.5),
    "imbalanced_hard": ("wmdp", "mmlu_bio", 0.25),   # target 1:3; see fallback note
}
```

`build_variant(name, split, seed)`: use **all** refuse IDs in the split's WMDP pool; sample `n_benign = round(n_refuse * (1-r) / r)` from the benign pool via `random.Random((seed, name, split))`; lookup texts, attach `label` (1 refuse, 0 benign), `source`, `split`; shuffle; return.

**Imbalanced_hard fallback:** because MMLU bio coverage is finite, the target `r=0.25` may be infeasible in some splits. The composition routine should:
1. Compute the requested `n_benign`.
2. If `len(benign_pool) < n_benign`, take the entire benign_pool and **log a warning naming the realized refuse fraction**. Do not raise.
3. Record `realized_refuse_fraction` per split into `config.json` / `metrics.json` and into the `print_summary` output.
4. If the realized fraction in train differs from the target by more than ~5 percentage points, surface this prominently in PROGRESS.md notes.

This keeps the variant in v1 (cleaner than dropping it) and turns the limitation into a documented observation rather than a silent fudge. For easy-variant builds (`dolly` benign), the fallback should never trigger; if it does, that's a bug.

Smoke: build all four variants × three splits; check realized refuse fractions; for `imbalanced_easy` expect ~0.10; for `imbalanced_hard` expect 0.25 if MMLU pool suffices else the documented fallback value; assert the WMDP-test ID set is identical across all four variants.

### Step 6 — `models.py`

`Classifier` Protocol with `fit(X, y)`, `predict(X)`, `predict_proba(X)`. `EmbeddingLogReg(embedding_model_name, classifier_kwargs, batch_size, device, normalize_embeddings)`: lazily loads `SentenceTransformer`, encodes via `model.encode(..., batch_size, normalize_embeddings, convert_to_numpy=True)`, then `sklearn.linear_model.LogisticRegression`. `build_classifier(cfg)` dispatches on `cfg.model_type`. Docstrings mention future seams: `TFIDFLogReg`, `FineTunedEncoder`, `ActivationProbe`.

Smoke: 20 toy strings, verify `predict_proba` shape `(20,2)` and rows sum to 1; encode full `balanced_easy` train split and confirm GPU is used.

### Step 7 — `train.py` + `evaluate.py`

`train.run(cfg)`: seed → `get_global_splits` (record pool sizes + file path back into `cfg`) → `build_variant` for train/val/test → `build_classifier` → fit on train → return `(model, splits)`.

`evaluate.run(model, splits, cfg)`: writes to `cfg.output_dir`:
- `config.json` (with `timestamp`, `pool_sizes`, `global_splits_file`)
- `metrics.json` (test_* and val_* keys: accuracy, per-class P/R/F1, 2×2 confusion matrix; plus n_train/val/test and refuse counts)
- `classification_report.txt` (sklearn report + ASCII confusion matrix)
- `confusion_matrix.png` (matplotlib `ConfusionMatrixDisplay`, labels `dont_refuse` / `refuse`)
- `predictions.csv` (`id, text, source_dataset, split, true_label, pred_label, pred_score` for val + test combined)
- `false_positives.csv` (test set, sorted by `pred_score` desc)
- `false_negatives.csv` (test set, sorted by `pred_score` asc)

`print_summary(metrics, cfg)`: concise stdout block — sizes, test accuracy, per-class P/R/F1, ASCII confusion matrix, artifact path.

End-to-end smoke on `balanced_easy`; expect run under 2 min on a 5090; eye-check all artifacts.

### Step 8 — `experiments.py` + CLI

`VARIANT_NAMES`, `make_config(variant, overrides)`, `run_all(base_cfg)` that loops the four variants and prints a comparison table (rows = variant; cols = test accuracy, refuse-recall, dont_refuse-recall).

`scripts/run_experiment.py` (argparse): `--dataset_variant` (mutually exclusive with `--all`), `--all`, `--embedding_model`, `--classifier_kwargs JSON`, `--seed`, `--split_seed`, `--train_frac/--val_frac/--test_frac`, `--output_dir`, `--experiment_name`, `--config PATH`. Flow: build `cfg` → `train.run` → `evaluate.run` → `print_summary`.

### Step 9 — README, CLAUDE.md, PROGRESS.md

**README** covers: task objective; dataset table (name, HF path, role, row count) — listing WMDP-bio (refuse), Dolly-15k (easy benign), and the 9 MMLU bio/medical subjects under `cais/mmlu` (hard benign); install (`uv sync` with `--system-site-packages` note); single-variant and `--all` run commands; output directory layout; metrics and interpretation (`refuse` recall = safety, `dont_refuse` recall = over-refusal); include the verbatim plan note about balanced-vs-deployment trade-offs; reproducibility note re. `outputs/splits/`; future-work seams.

**Compute / reproducibility section in README** (separate subsection):
- Python version (capture via `python3 --version` at build time).
- torch version + build (`torch.__version__` — should be `2.8.0+cu128`).
- GPU model + capability (`torch.cuda.get_device_name(0)`, `torch.cuda.get_device_capability(0)` — expect RTX 5090 / sm_120).
- CUDA driver + runtime from `nvidia-smi` (one line, e.g., `Driver Version: ... CUDA Version: ...`).
- Embedding model + seed (already in `config.json`, but worth surfacing).
- Note that the system torch is sm_120-specific and the venv inherits it via `--system-site-packages`; running on different hardware will need a torch reinstall.

Capture these values **at build time** into a small `repro.txt` written next to README (or embed them into README directly) so the doc isn't drifting from environment claims.

**CLAUDE.md** covers conventions for future Claude Code sessions on this repo:
- Project intent and the two design principles (global-pool splits; WMDP is the limit).
- Package management is **uv**; venv inherits system torch via `--system-site-packages`; do not let uv install torch.
- `/workspace` is persistent, `/root` is transient — write artifacts under `/workspace`.
- **Plan-archiving convention:** after any planning session (Claude Code plan mode), copy the plan file from `/root/.claude/plans/<slug>.md` into `/workspace/bio-refusal-classifier/plans/NNN_<descriptive_name>.md` so the plan survives pod resets and is reviewable from the repo. **Naming rule:** zero-padded 3-digit prefix (`001_`, `002_`, ...) followed by a short descriptive snake_case name reflecting the plan's actual contents. Existing prefixes set the next number; do not reuse. Example: this plan archives as `001_v1_baseline_scaffold.md`.
- **PROGRESS.md convention:** PROGRESS.md is a living document tracking project/experiment TODOs and progress. Update it after **every** experiment run with: which variant was run, key metrics (test accuracy, refuse-recall, dont_refuse-recall), notable failure modes, and any TODOs surfaced. Treat it as load-bearing context for future sessions — both human and AI readers should be able to scan it and know what's been tried, what worked, and what's next.
- Where the main extension seams are (Classifier protocol in `models.py`, VARIANTS registry in `data.py`).
- Output artifact contract (so future agents know not to break the file set downstream tools depend on).

**PROGRESS.md** initial sections:
- **Status** — one-line summary of where the project is (e.g., "v1 baseline built; balanced_easy and imbalanced_easy completed").
- **Open TODOs** — checkbox list. Seeded with: run all four variants; write the ½–1 page take-home write-up; (future) TF-IDF baseline; (future) fine-tuned encoder; (future) activation probe.
- **Experiment log** — reverse-chronological table with columns: timestamp, variant, model, test_accuracy, refuse_recall, dont_refuse_recall, artifacts dir, one-line notes. New row appended (at top) after each run.
- **Open questions / decisions** — running log; seed with the defaults from this plan (PubMedQA config choice, class_weight, etc.) so future readers see why those were chosen.

To keep the convention honest, `evaluate.print_summary` ends with the line: `> remember to update PROGRESS.md` — small nudge to make the rule self-enforcing.

### Step 10 — Archive this plan into the repo

```
mkdir -p /workspace/bio-refusal-classifier/plans
cp /root/.claude/plans/great-ok-well-you-buzzing-coral.md \
   /workspace/bio-refusal-classifier/plans/001_v1_baseline_scaffold.md
```

---

## Defaults chosen (under-specified in the spec; documented in code/README)

- **Hard-negative source:** 9 MMLU bio/medical subjects under `cais/mmlu` (high_school_biology, college_biology, college_medicine, medical_genetics, virology, anatomy, clinical_knowledge, professional_medicine, nutrition), questions only, answer choices stripped — same handling as WMDP. PubMedQA dropped from v1 (poorer format match + ~5–10 GB disk cost).
- **Imbalanced_hard ratio:** target refuse_fraction = 0.25 (1:3). Falls back to max feasible given MMLU pool size if insufficient; realized ratio logged.
- **Dolly category filter:** none — use all 15k.
- **Validation set role:** reported in `metrics.json` for visibility; not used for selection in v1.
- **LR `class_weight`:** `None` so refuse-recall is comparable across variants.
- **Decision threshold:** 0.5; threshold sweeps are future work.
- **HF revision pinning:** not pinned in v1; row counts logged in `metrics.json` for drift detection.
- **Embedding cache:** not in v1; MiniLM is fast.
- **`outputs/`:** gitignored except `outputs/splits/`.

---

## Critical files to be created

- `/workspace/bio-refusal-classifier/pyproject.toml`
- `/workspace/bio-refusal-classifier/src/config.py`
- `/workspace/bio-refusal-classifier/src/data.py`
- `/workspace/bio-refusal-classifier/src/models.py`
- `/workspace/bio-refusal-classifier/src/train.py`
- `/workspace/bio-refusal-classifier/src/evaluate.py`
- `/workspace/bio-refusal-classifier/src/experiments.py`
- `/workspace/bio-refusal-classifier/src/utils.py`
- `/workspace/bio-refusal-classifier/scripts/run_experiment.py`
- `/workspace/bio-refusal-classifier/README.md`
- `/workspace/bio-refusal-classifier/CLAUDE.md`
- `/workspace/bio-refusal-classifier/PROGRESS.md`
- `/workspace/bio-refusal-classifier/plans/001_v1_baseline_scaffold.md`

---

## Verification

After build, one command proves end-to-end correctness:

```
uv run python scripts/run_experiment.py --all
```

Expected wall-clock under ~5 min on a 5090 (dominated by first-run HF downloads).

**Artifacts to eye-check:**
1. `outputs/splits/global_splits__seed42__fracs0.8-0.1-0.1.json` — WMDP / Dolly / PubMedQA pool sizes sum to source totals.
2. `outputs/balanced_easy__*/config.json` — `pool_sizes`, `global_splits_file`, `timestamp` populated.
3. `outputs/balanced_easy__*/metrics.json` — test accuracy in ~0.85–0.98 range (easy negatives).
4. `outputs/balanced_hard__*/metrics.json` — noticeably lower than balanced_easy; if not, suspect leakage.
5. `outputs/imbalanced_easy__*/metrics.json` — accuracy near 0.9 is trivial at 10/90; check refuse-recall.
   `outputs/imbalanced_hard__*/metrics.json` — verify `realized_refuse_fraction` is logged (target 0.25; may be higher if MMLU pool was the limit).
6. `false_positives.csv` and `false_negatives.csv` for `balanced_easy` — 5 rows each, plausible borderline cases (catches id↔text misalignment).
7. `confusion_matrix.png` — opens cleanly, labels `dont_refuse` / `refuse`.
8. **Cross-variant WMDP-test invariant:** the WMDP id-set from `predictions.csv` filtered to `split=="test" and source=="wmdp"` is identical across all four runs.

Build-time guard inside `evaluate.run`: assert the realized refuse fraction is within 1% of the configured `r`.
