# PROGRESS

Living document. Update after every experiment.

## Status

V1 baseline complete. All four variants run end-to-end, cross-variant WMDP-test invariance verified (128 IDs identical across runs). MiniLM-L6-v2 + LR achieves 0.98 on `balanced_easy` and drops to ~0.95 on `balanced_hard` — the hard MMLU bio negatives meaningfully challenge the classifier, which is the test the variant set was designed for. Static dashboard for browsing failure cases lives on `feature/dashboard` (open `dashboard/index.html` in a browser; rebuild data with `uv run python dashboard/build_data.py`). Next: take-home write-up.

## Open TODOs

- [x] Run all four variants end-to-end (`uv run python scripts/run_experiment.py --all`).
- [x] Record results below in **Experiment log**.
- [x] Confusion-matrix dashboard for failure-mode exploration (on `feature/dashboard`).
- [ ] Write the ½–1 page take-home write-up (data choices, modeling, results, what's next with more time).
- [ ] (Future) TF-IDF + LR baseline as a cheap second classifier — would expose whether the embedding model is doing real semantic work beyond surface tokens.
- [ ] (Future) Fine-tuned DistilBERT for comparison.
- [ ] (Future) Threshold sweep + ROC/PR curves per variant.
- [ ] (Future) Activation-probe variant (different encoder, classify on hidden states).
- [ ] (Future) Embedding cache under `outputs/embeddings_cache/` keyed by `(model, sha1(text))` — would amortize encoder cost across sweeps.

## Experiment log

Most recent at the top.

| Timestamp | Variant | Model | Test acc | Refuse recall | Dont-refuse recall | Artifacts | Notes |
|---|---|---|---|---|---|---|---|
| 2026-05-17 18:39 | imbalanced_hard | MiniLM-L6-v2 + LR | 0.9493 | 0.938 | 0.957 | `outputs/imbalanced_hard__all-minilm-l6-v2__seed42/` | Uses all MMLU bio as hard negatives → realized r=0.382. 9 FPs, 8 FNs. Errors are largely the same MMLU-vs-WMDP boundary cases as `balanced_hard`. |
| 2026-05-17 18:39 | balanced_hard | MiniLM-L6-v2 + LR | 0.9453 | 0.953 | 0.938 | `outputs/balanced_hard__all-minilm-l6-v2__seed42/` | 3.5pp accuracy drop vs balanced_easy — MMLU bio is doing real work as a hard negative. Top FNs are technical-sounding WMDP questions without overtly weapon-y language (mucociliary clearance, mosquito saliva, ricin mechanism). |
| 2026-05-17 18:39 | imbalanced_easy | MiniLM-L6-v2 + LR | 0.9891 | 0.953 | 0.993 | `outputs/imbalanced_easy__all-minilm-l6-v2__seed42/` | r=0.10. Accuracy is mostly a function of the majority class — the real number is refuse recall (0.95) and refuse F1 (0.946). |
| 2026-05-17 18:39 | balanced_easy | MiniLM-L6-v2 + LR | 0.9805 | 0.992 | 0.969 | `outputs/balanced_easy__all-minilm-l6-v2__seed42/` | 4 FPs (Dolly instructions that look biology-adjacent), 1 FN ("ethnic bioweapon" — a borderline phrase). |

## Failure-mode observations from v1

- **Hard negatives reveal a vocabulary heuristic.** In `balanced_easy` the classifier rarely misses WMDP; in `balanced_hard` the top FNs are WMDP questions that read like generic biology MCQs without overt weapon/pathogen language. The model appears to lean on lexical bio-risk cues; when those are absent, refuse-recall drops. This is the expected weakness of a small embedding model + linear head, and the main reason the hard variants matter.
- **Cross-variant comparison is valid.** WMDP-test IDs are identical across all four runs (verified). Differences in refuse-recall across variants reflect different negatives, not different positives.

## Open questions / decisions

- **Hard-negative source = MMLU bio (9 subjects), not PubMedQA.** MMLU is a tighter format match to WMDP (same 4-option multiple-choice provenance) and saves ~5–10 GB of disk. Trade-off: MMLU has only ~2,077 rows combined.
- **`imbalanced_hard` uses all available MMLU bio data** (`r=None` in `VARIANTS`) rather than chasing a fixed target ratio. The realized refuse fraction is ~0.38 across splits — a property of pool sizes, not a degraded target. **Rationale:** for a finite high-quality negative source, data quality outranks hitting an arbitrary ratio. Realized fraction is logged in every run's `realized_refuse_fraction`.
- **LR `class_weight` = None.** Keeps refuse-recall comparable across variants. Revisit if we want to deploy under heavy imbalance.
- **Validation set reported in `metrics.json` but not used for selection in v1.** Future hyperparameter sweeps will use it.
- **Decision threshold fixed at 0.5.** Threshold sweeps are a future TODO.
- **HF dataset revisions not pinned.** Row counts are logged in `metrics.json` so drift is detectable.
- **Embedding cache deferred** — MiniLM on a 5090 is fast enough that adding a cache wasn't worth v1 complexity.
- **Dolly used without category filtering** — all 15k rows. Some categories (e.g., `summarization`) are obviously-not-refusal-candidates; the spec didn't ask for filtering and reducing benign volume would tighten the easy variants for no clear win.
