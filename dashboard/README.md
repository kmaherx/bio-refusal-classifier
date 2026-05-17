# dashboard

Single static page that visualises the v1 confusion matrices and lets you drill into individual misclassified (or correctly classified) questions across the four dataset variants.

## View it

Open `dashboard/index.html` directly in a browser. No server required (works from `file://`).

## Rebuild the data

```bash
uv run python dashboard/build_data.py
```

Reads `outputs/<variant>__*/predictions.csv` + `metrics.json` for each of the four variants, writes:

- `dashboard/data.js` — `window.DASHBOARD_DATA = {...}` blob loaded by `index.html`.
- `outputs/question_summaries.json` — per-question heuristic summaries, persisted so the script is idempotent.

Both files are checked in so reviewers can open `index.html` without running anything.

## Files

```
dashboard/
  index.html        # hand-written; layout scaffolding
  styles.css        # responsive layout (3-col desktop ≥768px, 1-col mobile)
  app.js            # state + render + click handlers; reads window.DASHBOARD_DATA
  build_data.py     # generator (Python; reads outputs/, writes data.js + summaries)
  data.js           # generated; do not hand-edit
  README.md         # this file
```

## UI

- **Toggles** (top of left panel on desktop, top section on mobile) — pick one of the four experiments. Matrix and list reset.
- **Matrix** — 2×2 of true vs. predicted label. Each cell is a button; click to load its questions.
- **List** — middle panel/section. Substring filter at the top. Each item shows the question summary and its pred_score; click to view full text.
- **Detail** — right panel on desktop, bottom section on mobile. Source/subject, true/pred labels as badges, pred_score, full question text, and the question's id.

## Conventions

- Test set only. Val rows in `predictions.csv` are dropped.
- Cell colour hints: green = correct, yellow = false positive, red = false negative.
- Summaries: first ~10 words of the question, ellipsised, prefixed with `[source]` or `[source/subject]` for MMLU.
- `pred_score` is `P(refuse)`, so high scores mean "refuse" and low scores mean "don't refuse" regardless of the true label.
