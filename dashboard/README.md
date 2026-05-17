# dashboard

Single static page that visualises the v1 confusion matrices and lets you drill into individual misclassified (or correctly classified) questions across the four dataset variants.

## View it

Open `dashboard/index.html` directly in a browser. No server required (works from `file://`).

## Rebuild

```bash
uv run python dashboard/build_data.py
```

Reads `outputs/<variant>__*/predictions.csv` + `metrics.json` for each of the four variants, then assembles a fully self-contained `dashboard/index.html` by inlining `template.html`, `styles.css`, and `app.js` together with the data blob.

Outputs:

- `dashboard/index.html` — self-contained (HTML + inlined CSS + inlined JS + inlined data). The only file a reviewer needs to open. Generated; do not hand-edit.
- `outputs/question_summaries.json` — per-question heuristic summaries, persisted so the script is idempotent.

Both are checked in so reviewers don't need to re-run the build.

## Files

```
dashboard/
  index.html        # GENERATED — open this in a browser
  template.html     # source: HTML scaffold with __STYLES__/__DATA__/__APP__ placeholders
  styles.css        # source: responsive layout (3-col desktop ≥768px, 1-col mobile)
  app.js            # source: state + render + click handlers
  build_data.py     # generator (Python; reads outputs/, writes index.html + summaries)
  README.md         # this file
```

To make changes to the dashboard:

1. Edit `template.html`, `styles.css`, and/or `app.js`.
2. Run `uv run python dashboard/build_data.py` to regenerate `index.html`.
3. Reload the page in your browser.

The self-contained `index.html` is the file you ship to reviewers; the three source files exist for maintainability.

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
