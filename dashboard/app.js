// Confusion-matrix dashboard. Reads window.DASHBOARD_DATA (loaded by data.js).

(() => {
  const data = window.DASHBOARD_DATA;
  if (!data) {
    document.body.innerHTML =
      "<p style='padding:2rem;font-family:sans-serif'>No data. Run <code>uv run python dashboard/build_data.py</code> first.</p>";
    return;
  }

  const LABELS = ["dont_refuse", "refuse"];
  const CELL_CLASSES = {
    "0_0": "tn",
    "0_1": "fp",
    "1_0": "fn",
    "1_1": "tp",
  };
  const CELL_NAMES = {
    "0_0": "true negative",
    "0_1": "false positive",
    "1_0": "false negative",
    "1_1": "true positive",
  };

  const experiments = Object.fromEntries(data.experiments.map((e) => [e.key, e]));
  const initial = data.experiments[0]?.key ?? null;

  const state = {
    activeExperiment: initial,
    activeCell: null,
    selectedQuestionId: null,
    filter: "",
  };

  const els = {
    toggles: document.querySelector(".toggles"),
    headline: document.querySelector(".headline"),
    matrix: document.querySelector(".matrix"),
    filter: document.querySelector(".filter"),
    listMeta: document.querySelector(".list-meta"),
    qlist: document.querySelector(".qlist"),
    detail: document.querySelector(".detail"),
  };

  // --- toggles ---
  function renderToggles() {
    els.toggles.innerHTML = "";
    for (const exp of data.experiments) {
      const b = document.createElement("button");
      b.className = "toggle" + (exp.key === state.activeExperiment ? " active" : "");
      b.textContent = exp.display;
      b.setAttribute("role", "tab");
      b.setAttribute("aria-selected", exp.key === state.activeExperiment ? "true" : "false");
      b.addEventListener("click", () => {
        if (state.activeExperiment === exp.key) return;
        state.activeExperiment = exp.key;
        state.activeCell = null;
        state.selectedQuestionId = null;
        state.filter = "";
        els.filter.value = "";
        renderAll();
      });
      els.toggles.appendChild(b);
    }
  }

  // --- matrix + headline ---
  function renderHeadline() {
    const exp = experiments[state.activeExperiment];
    if (!exp) {
      els.headline.textContent = "";
      return;
    }
    const h = exp.headline;
    const r = exp.realized_r;
    els.headline.innerHTML =
      `accuracy <span class="num">${h.accuracy.toFixed(4)}</span> · ` +
      `refuse recall <span class="num">${h.refuse_recall.toFixed(3)}</span> · ` +
      `dont_refuse recall <span class="num">${h.dont_refuse_recall.toFixed(3)}</span> · ` +
      `n_test <span class="num">${exp.n_test}</span> · ` +
      `realized r <span class="num">${r != null ? r.toFixed(3) : "—"}</span>`;
  }

  function renderMatrix() {
    els.matrix.innerHTML = "";
    const exp = experiments[state.activeExperiment];
    if (!exp) return;

    const corner = el("div", "corner", "");
    corner.style.gridColumn = "1";
    corner.style.gridRow = "1";

    const colLabel0 = el("div", "col-label", "pred=dont_refuse");
    colLabel0.style.gridColumn = "2";
    colLabel0.style.gridRow = "1";
    const colLabel1 = el("div", "col-label", "pred=refuse");
    colLabel1.style.gridColumn = "3";
    colLabel1.style.gridRow = "1";

    const rowLabel0 = el("div", "row-label", "true=dont_refuse");
    rowLabel0.style.gridColumn = "1";
    rowLabel0.style.gridRow = "2";
    const rowLabel1 = el("div", "row-label", "true=refuse");
    rowLabel1.style.gridColumn = "1";
    rowLabel1.style.gridRow = "3";

    els.matrix.append(corner, colLabel0, colLabel1, rowLabel0, rowLabel1);

    for (let t = 0; t < 2; t++) {
      for (let p = 0; p < 2; p++) {
        const key = `${t}_${p}`;
        const count = exp.matrix[t][p];
        const cell = document.createElement("button");
        cell.className = `cell ${CELL_CLASSES[key]}` + (key === state.activeCell ? " selected" : "");
        cell.style.gridColumn = String(p + 2);
        cell.style.gridRow = String(t + 2);
        cell.innerHTML =
          `<span class="count">${count}</span><span class="cell-name">${CELL_NAMES[key]}</span>`;
        cell.addEventListener("click", () => {
          state.activeCell = key;
          state.selectedQuestionId = null;
          state.filter = "";
          els.filter.value = "";
          renderMatrix();
          renderList();
          renderDetail();
        });
        els.matrix.appendChild(cell);
      }
    }
  }

  // --- list ---
  function getActiveCellItems() {
    const exp = experiments[state.activeExperiment];
    if (!exp || !state.activeCell) return [];
    return exp.cells[state.activeCell] ?? [];
  }

  function renderList() {
    els.qlist.innerHTML = "";
    if (!state.activeCell) {
      els.listMeta.textContent = "Click a matrix cell to load questions.";
      return;
    }

    const items = getActiveCellItems();
    const filter = state.filter.trim().toLowerCase();
    const filtered = filter
      ? items.filter((it) => (data.questions[it.id]?.summary || "").toLowerCase().includes(filter))
      : items;

    if (items.length === 0) {
      els.listMeta.textContent = "No items in this cell.";
      return;
    }
    els.listMeta.textContent = filter
      ? `${filtered.length} shown / ${items.length} total`
      : `${items.length} item${items.length === 1 ? "" : "s"}`;

    const frag = document.createDocumentFragment();
    for (const it of filtered) {
      const q = data.questions[it.id];
      if (!q) continue;
      const li = document.createElement("li");
      li.className = "qitem";
      const btn = document.createElement("button");
      btn.type = "button";
      if (it.id === state.selectedQuestionId) btn.classList.add("selected");
      const summary = document.createElement("span");
      summary.textContent = q.summary;
      const score = document.createElement("span");
      score.className = "qscore";
      score.textContent = it.score.toFixed(3);
      btn.append(summary, score);
      btn.addEventListener("click", () => {
        state.selectedQuestionId = it.id;
        renderList();
        renderDetail();
      });
      li.appendChild(btn);
      frag.appendChild(li);
    }
    els.qlist.appendChild(frag);

    if (filtered.length === 0) {
      const empty = document.createElement("li");
      empty.className = "qlist-empty";
      empty.textContent = "No matches.";
      els.qlist.appendChild(empty);
    }
  }

  // --- detail ---
  function renderDetail() {
    els.detail.innerHTML = "";
    const id = state.selectedQuestionId;
    if (!id) {
      const p = document.createElement("p");
      p.className = "detail-empty";
      p.textContent = "Click a question to view its full text.";
      els.detail.appendChild(p);
      return;
    }
    const q = data.questions[id];
    if (!q) return;

    const items = getActiveCellItems();
    const found = items.find((it) => it.id === id);
    const score = found ? found.score : null;

    const [tStr, pStr] = state.activeCell.split("_");
    const trueLabel = LABELS[Number(tStr)];
    const predLabel = LABELS[Number(pStr)];

    const meta = document.createElement("div");
    meta.className = "meta";
    const sourceBadge = document.createElement("span");
    sourceBadge.className = "badge";
    sourceBadge.textContent = q.subject ? `${q.source}/${q.subject}` : q.source;
    const trueBadge = document.createElement("span");
    trueBadge.className = `badge ${trueLabel}`;
    trueBadge.textContent = `true=${trueLabel}`;
    const predBadge = document.createElement("span");
    predBadge.className = `badge ${predLabel}`;
    predBadge.textContent = `pred=${predLabel}`;
    meta.append(sourceBadge, trueBadge, predBadge);
    if (score != null) {
      const scoreBadge = document.createElement("span");
      scoreBadge.className = "badge";
      scoreBadge.textContent = `score ${score.toFixed(3)}`;
      meta.append(scoreBadge);
    }
    els.detail.appendChild(meta);

    const qid = document.createElement("div");
    qid.className = "qid";
    qid.textContent = `id: ${id}`;
    els.detail.appendChild(qid);

    const text = document.createElement("div");
    text.className = "qtext";
    text.textContent = q.text;
    els.detail.appendChild(text);
  }

  function renderAll() {
    renderToggles();
    renderHeadline();
    renderMatrix();
    renderList();
    renderDetail();
  }

  function el(tag, className, text) {
    const e = document.createElement(tag);
    e.className = className;
    if (text != null) e.textContent = text;
    return e;
  }

  els.filter.addEventListener("input", (e) => {
    state.filter = e.target.value;
    renderList();
  });

  renderAll();
})();
