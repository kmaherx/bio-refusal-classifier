# PROGRESS

Living document. Update after every experiment.

## Status

V2 complete: easy-negative source swapped from Dolly to MMLU non-bio (48 subjects), holding question format constant between refuse and benign. Dolly preserved as a secondary option (two `*_dolly` variants in `VARIANTS`) but excluded from the default `--all` sweep. Surprise finding: the format swap did **not** materially change easy-variant accuracy (~0.98 either way) — topic alone is doing the discriminative work. FPs collapsed to zero on `balanced_easy`; FNs grew slightly. Hard variants unchanged. Dashboard regenerated with v2 numbers. Next: take-home write-up.

## Open TODOs

- [x] Run all four variants end-to-end (`uv run python scripts/run_experiment.py --all`).
- [x] Record results below in **Experiment log**.
- [x] Confusion-matrix dashboard for failure-mode exploration (merged from `feature/dashboard`).
- [x] Swap Dolly for MMLU non-bio as primary easy negative; keep Dolly secondary (merged from `feature/mmlu-easy`).
- [x] Write the ½–1 page take-home write-up (at [`writeup.txt`](writeup.txt) — data choices, modeling, results, what's next).
- [ ] (Future) Re-run the two `*_dolly` variants and add them as optional toggles in the dashboard.
- [ ] (Future) TF-IDF + LR baseline as a cheap second classifier — would expose whether the embedding model is doing real semantic work beyond surface tokens.
- [ ] (Future) Fine-tuned DistilBERT for comparison.
- [ ] (Future) Threshold sweep + ROC/PR curves per variant.
- [ ] (Future) Activation-probe variant (different encoder, classify on hidden states).
- [ ] (Future) Embedding cache under `outputs/embeddings_cache/` keyed by `(model, sha1(text))` — would amortize encoder cost across sweeps.

## Experiment log

Most recent at the top.

| Timestamp | Variant | Model | Test acc | Refuse recall | Dont-refuse recall | Artifacts | Notes |
|---|---|---|---|---|---|---|---|
| 2026-05-17 19:42 | imbalanced_hard (v2) | MiniLM-L6-v2 + LR | 0.9493 | 0.938 | 0.957 | `outputs/imbalanced_hard__all-minilm-l6-v2__seed42/` | Identical to v1 — hard variants don't reference the easy-negative source. WMDP-test invariance holds. |
| 2026-05-17 19:42 | balanced_hard (v2) | MiniLM-L6-v2 + LR | 0.9453 | 0.953 | 0.938 | `outputs/balanced_hard__all-minilm-l6-v2__seed42/` | Identical to v1 (same reason). |
| 2026-05-17 19:42 | imbalanced_easy (v2) | MiniLM-L6-v2 + LR | 0.9930 | 0.938 | 0.999 | `outputs/imbalanced_easy__all-minilm-l6-v2__seed42/` | MMLU non-bio easy negs. 1 FP, 8 FNs. Refuse precision jumps to 0.99 (vs 0.94 in v1); refuse recall drops 0.953 → 0.938. |
| 2026-05-17 19:42 | balanced_easy (v2) | MiniLM-L6-v2 + LR | 0.9844 | 0.969 | 1.000 | `outputs/balanced_easy__all-minilm-l6-v2__seed42/` | MMLU non-bio easy negs. **Zero FPs** (vs 4 in v1); 4 FNs (vs 1 in v1). Refuse precision = 1.00. |
| 2026-05-17 18:39 | imbalanced_hard (v1) | MiniLM-L6-v2 + LR | 0.9493 | 0.938 | 0.957 | (replaced) | v1 baseline with Dolly easy negatives. |
| 2026-05-17 18:39 | balanced_hard (v1) | MiniLM-L6-v2 + LR | 0.9453 | 0.953 | 0.938 | (replaced) | v1 baseline. |
| 2026-05-17 18:39 | imbalanced_easy (v1) | MiniLM-L6-v2 + LR | 0.9891 | 0.953 | 0.993 | (replaced) | v1 baseline with Dolly easy negatives. |
| 2026-05-17 18:39 | balanced_easy (v1) | MiniLM-L6-v2 + LR | 0.9805 | 0.992 | 0.969 | (replaced) | v1 baseline with Dolly easy negatives: 4 FPs (biology-adjacent Dolly instructions), 1 FN. |

## Failure-mode observations

- **Hard negatives reveal a vocabulary heuristic** (unchanged from v1 → v2). In `balanced_easy` the classifier rarely misses WMDP; in `balanced_hard` the top FNs are WMDP questions that read like generic biology MCQs without overt weapon/pathogen language. The model leans on lexical bio-risk cues; when those are absent, refuse-recall drops. This is the expected weakness of a small embedding model + linear head, and the main reason the hard variants matter.
- **Format wasn't doing much work after all.** v2 hypothesis was that swapping Dolly's open-ended-instruction format for MMLU non-bio's stripped-MCQ format would change easy-variant accuracy noticeably. It did not — accuracy moved from 0.9805 to 0.9844 (essentially flat), and the failure structure shifted from "occasional biology-adjacent Dolly FPs" to "no FPs at all, a few more WMDP FNs." Topic-based discrimination is doing the work; format was a small confound at most. Useful negative result.
- **Cross-variant comparison is valid.** WMDP-test IDs are identical across all four primary variants (verified). Hard-variant numbers are byte-identical between v1 and v2 by design.

## Open questions / decisions

- **Easy-negative source = MMLU non-bio (48 subjects, primary), Dolly-15k (secondary).** v2 swaps the default easy negative from Dolly to MMLU non-bio so refuse and benign share the same stripped-MCQ format and the easy/hard contrast is purely topical. Dolly variants are still runnable via `--dataset_variant balanced_easy_dolly` / `imbalanced_easy_dolly` but are not part of `--all`. **Trade-off:** the classifier never sees open-ended instructions during training; less deployment-realistic. Worth accepting for v2's cleaner experimental control.
- **"Non-bio" definition.** Anything in MMLU's 57 subjects except the 9 we use as hard negatives. Borderline-health subjects (`human_aging`, `human_sexuality`, `professional_psychology`, `high_school_psychology`) are included as non-bio. Curating them out would change the pool by ~600 rows out of ~13,500 (<5%); flagged as a known caveat.
- **Hard-negative source = MMLU bio (9 subjects), not PubMedQA.** MMLU is a tighter format match to WMDP (same 4-option multiple-choice provenance) and saves ~5–10 GB of disk. Trade-off: MMLU has only ~2,077 rows combined.
- **`imbalanced_hard` uses all available MMLU bio data** (`r=None` in `VARIANTS`) rather than chasing a fixed target ratio. The realized refuse fraction is ~0.38 across splits — a property of pool sizes, not a degraded target. **Rationale:** for a finite high-quality negative source, data quality outranks hitting an arbitrary ratio. Realized fraction is logged in every run's `realized_refuse_fraction`.
- **LR `class_weight` = None.** Keeps refuse-recall comparable across variants. Revisit if we want to deploy under heavy imbalance.
- **Validation set reported in `metrics.json` but not used for selection in v1.** Future hyperparameter sweeps will use it.
- **Decision threshold fixed at 0.5.** Threshold sweeps are a future TODO.
- **HF dataset revisions not pinned.** Row counts are logged in `metrics.json` so drift is detectable.
- **Embedding cache deferred** — MiniLM on a 5090 is fast enough that adding a cache wasn't worth v1 complexity.
- **Dolly used without category filtering** — all 15k rows. Some categories (e.g., `summarization`) are obviously-not-refusal-candidates; the spec didn't ask for filtering and reducing benign volume would tighten the easy variants for no clear win.
