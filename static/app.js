/* Review UI: vanilla JS, no build step. */
(() => {
  "use strict";

  const TYPE_LABEL = {
    NAME: "Names", EMAIL: "Emails", PHONE: "Phones", ADDRESS: "Addresses",
    ORG_ADDRESS: "Org addresses", ID: "IDs",
  };
  const TYPE_TAG = {
    NAME: "Name", EMAIL: "Email", PHONE: "Phone", ADDRESS: "Address",
    ORG_ADDRESS: "Org addr", ID: "ID",
  };
  const TYPE_ORDER = ["NAME", "EMAIL", "PHONE", "ADDRESS", "ID", "ORG_ADDRESS"];

  const $ = (sel) => document.querySelector(sel);
  const el = (tag, attrs = {}, ...kids) => {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") n.className = v;
      else if (k === "text") n.textContent = v;
      else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
      else if (v !== undefined && v !== null && v !== false) n.setAttribute(k, v);
    }
    for (const kid of kids) if (kid != null) n.append(kid);
    return n;
  };

  const state = {
    doc: null,          // {filename, text, spans, entities}
    status: new Map(),  // "start:end" -> approved | rejected | pending
    focus: -1,          // index into visible spans
    filter: "ALL",
    selection: null,    // {start, end, text}
    mode: "pseudonym",  // export replacement style
    allowPending: false,
    lastExport: null,   // {receipt, file, redacted_text}
    generalizations: [],// [{find, replace}] applied to the redacted document on export
    summary: null,      // {summary, method, audit, qi, ...}
    summaryKey: null,
    manual: [],         // manual redactions in this document (kept across re-analysis)
    feedback: [],       // queued learning events
    feedbackSeq: 0,
    learning: null,     // /api/learn summary
  };

  const key = (sp) => `${sp.start}:${sp.end}`;
  const tier = (c) => (c >= 0.9 ? "hi" : c >= 0.6 ? "md" : "lo");
  const statusOf = (sp) => state.status.get(key(sp)) || "pending";
  const entityOf = (sp) => state.doc.entities.find((e) => e.id === sp.entity_id);
  const visibleSpans = () =>
    state.doc.spans.filter((s) => state.filter === "ALL" || s.type === state.filter);

  function shape(sp) {
    if (sp.type === "PHONE") return "PHONE:" + sp.text.replace(/\D/g, "").length;
    if (sp.type !== "ID") return sp.type;
    return "ID:" + sp.text.replace(/\s+/g, " ").replace(/[A-Z]/g, "A").replace(/[a-z]/g, "a").replace(/\d/g, "9");
  }

  // ---------------------------------------------------------------- status bar
  async function loadStatus() {
    try {
      const st = await (await fetch("/api/status")).json();
      const det = $("#detector-pill");
      const loaded = st.detector.status === "loaded";
      const down = st.detector.status === "unavailable";
      det.textContent = loaded ? "Detector · rules + GLiNER" : down ? "Detector · rules only" : "Detector · rules (NER loads on first run)";
      det.classList.toggle("dashed", down);
      if (down) det.title = st.detector.error || "";
      const mp = $("#model-pill");
      const o = st.ollama;
      mp.textContent = `Summary · ${o.summary_model}` + (o.reachable ? (o.model_installed ? "" : " (not pulled)") : " (Ollama offline)");
      mp.classList.toggle("dashed", !o.reachable || !o.model_installed);
      renderLocal(st);
    } catch (_) {
      $("#detector-pill").textContent = "Detector · unknown";
    }
  }

  // ---------------------------------------------------------------- loading docs
  async function loadSamples() {
    try {
      const list = await (await fetch("/api/samples")).json();
      const box = $("#sample-list");
      box.replaceChildren(...list.map((s) =>
        el("button", { class: "pill btn chip", type: "button", onclick: () => openSample(s.name) },
          prettyName(s.name), el("span", { class: "count", text: s.kind.toUpperCase() }))));
    } catch (_) { /* samples are optional */ }
  }

  function prettyName(fn) {
    const base = fn.replace(/\.[^.]+$/, "").replace(/[_-]+/g, " ").trim();
    const nice = base.charAt(0).toUpperCase() + base.slice(1);
    return nice.replace(/\b(hr|kyc|pdf|msa|nda|id)\b/gi, (w) => w.toUpperCase());
  }

  function show(view) {
    for (const id of ["landing", "loading", "review"]) $("#" + id).hidden = id !== view;
  }

  async function run(promise, label) {
    $("#landing-error").hidden = true;
    $("#loading-text").textContent = label || "Reading the document…";
    show("loading");
    try {
      const res = await promise;
      const body = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(typeof body.detail === "string" ? body.detail : `Request failed (${res.status}).`);
      setDoc(body);
      loadStatus();
    } catch (err) {
      show("landing");
      const box = $("#landing-error");
      box.replaceChildren(el("strong", { text: "That document could not be opened." }), err.message);
      box.hidden = false;
    }
  }

  const openSample = (name) => run(fetch(`/api/samples/${encodeURIComponent(name)}`), `Opening ${prettyName(name)}…`);
  function openFile(file) {
    const fd = new FormData();
    fd.append("file", file);
    run(fetch("/api/analyze", { method: "POST", body: fd }), `Reading ${file.name}…`);
  }
  function openText(text) {
    const fd = new FormData();
    fd.append("text", text);
    run(fetch("/api/analyze", { method: "POST", body: fd }));
  }

  function setDoc(doc, keepStatus = false) {
    const old = keepStatus ? state.status : new Map();
    state.doc = doc;
    state.status = new Map();
    for (const sp of doc.spans) state.status.set(key(sp), old.get(key(sp)) || (sp.source === "manual" ? "approved" : "pending"));
    if (!keepStatus) {
      state.filter = "ALL"; state.focus = -1; state.allowPending = false;
      state.generalizations = []; state.summary = null; state.summaryKey = null; state.manual = [];
      $("#summary-panel").hidden = true;
    }
    $("#doc-file").textContent = doc.filename;
    $("#doc-title").textContent = prettyName(doc.filename);
    show("review");
    renderDoc();
    renderFilter();
    renderCandidates();
    renderProgress();
    if (!keepStatus && visibleSpans().length) setFocus(0, false);
  }

  // ---------------------------------------------------------------- rendering
  function renderDoc() {
    const { text, spans } = state.doc;
    const root = $("#doc");
    const frag = document.createDocumentFragment();
    let pos = 0;
    for (const sp of spans) {
      if (sp.start < pos) continue; // safety: spans are non-overlapping
      if (sp.start > pos) frag.append(el("span", { class: "seg", "data-start": pos }, text.slice(pos, sp.start)));
      const mark = el("mark", {
        class: `m ${tier(sp.confidence)}`, "data-id": sp.id,
        title: `${sp.type} · ${sp.confidence.toFixed(2)} · ${sp.source}`,
      },
        el("span", { class: "txt", "data-start": sp.start }, text.slice(sp.start, sp.end)),
        el("span", { class: "tag", "aria-hidden": "true" }));
      mark.addEventListener("click", () => {
        if (window.getSelection().toString()) return;
        const i = visibleSpans().findIndex((s) => s.id === sp.id);
        if (i < 0) { state.filter = "ALL"; renderFilter(); renderCandidates(); }
        setFocus(visibleSpans().findIndex((s) => s.id === sp.id));
      });
      frag.append(mark);
      pos = sp.end;
    }
    if (pos < text.length) frag.append(el("span", { class: "seg", "data-start": pos }, text.slice(pos)));
    root.replaceChildren(frag);
    updateMarks();
  }

  function updateMarks() {
    const focused = currentSpan();
    for (const sp of state.doc.spans) {
      const m = document.querySelector(`mark[data-id="${sp.id}"]`);
      if (!m) continue;
      const st = statusOf(sp);
      const ent = entityOf(sp);
      m.classList.toggle("approved", st === "approved");
      m.classList.toggle("rejected", st === "rejected");
      m.classList.toggle("focus", focused && focused.id === sp.id);
      m.classList.toggle("dim", state.filter !== "ALL" && sp.type !== state.filter);
      m.querySelector(".tag").textContent = st === "approved" && ent ? ent.pseudonym : TYPE_TAG[sp.type] || sp.type;
    }
    for (const b of document.querySelectorAll(".mention")) {
      const sp = state.doc.spans.find((s) => s.id === b.dataset.id);
      if (!sp) continue;
      const st = statusOf(sp);
      b.classList.toggle("approved", st === "approved");
      b.classList.toggle("rejected", st === "rejected");
      b.classList.toggle("focus", focused && focused.id === sp.id);
    }
    for (const card of document.querySelectorAll(".entity")) {
      card.classList.toggle("active", !!focused && focused.entity_id === card.dataset.id);
      const ent = state.doc.entities.find((e) => e.id === card.dataset.id);
      if (!ent) continue;
      const sts = ent.span_ids.map((id) => statusOf(state.doc.spans.find((s) => s.id === id)));
      const a = sts.filter((s) => s === "approved").length;
      const r = sts.filter((s) => s === "rejected").length;
      card.querySelector(".entity-meta").textContent =
        `${ent.span_ids.length} mention${ent.span_ids.length > 1 ? "s" : ""} · max ${ent.confidence.toFixed(2)} · ${a} approved · ${r} rejected`;
    }
  }

  function renderFilter() {
    const counts = {};
    for (const sp of state.doc.spans) counts[sp.type] = (counts[sp.type] || 0) + 1;
    const chip = (t, label, n) => el("button", {
      class: "pill btn chip", type: "button", "aria-pressed": state.filter === t ? "true" : "false",
      onclick: () => { state.filter = t; state.focus = -1; renderFilter(); renderCandidates(); setFocus(0, false); },
    }, label, el("span", { class: "count", text: String(n) }));
    const chips = [chip("ALL", "All", state.doc.spans.length)];
    for (const t of TYPE_ORDER) if (counts[t]) chips.push(chip(t, TYPE_LABEL[t], counts[t]));
    $("#type-filter").replaceChildren(...chips);
  }

  function renderCandidates() {
    const box = $("#candidates");
    const ents = state.doc.entities.filter((e) => state.filter === "ALL" || e.type === state.filter);
    if (!ents.length) {
      box.replaceChildren(el("p", { class: "empty", text: "Nothing detected here." }));
      return;
    }
    box.replaceChildren(...ents.map((e) => {
      const spans = e.span_ids.map((id) => state.doc.spans.find((s) => s.id === id)).filter(Boolean);
      return el("section", { class: "entity", "data-id": e.id },
        el("div", { class: "entity-head" },
          el("span", { class: "pill pill-label strong", text: e.pseudonym }),
          el("span", { class: "pill pill-label", text: TYPE_TAG[e.type] || e.type })),
        el("p", { class: "entity-name", text: e.canonical }),
        el("p", { class: "entity-meta" }),
        el("div", { class: "mentions" }, ...spans.map((sp) =>
          el("button", {
            class: `mention ${tier(sp.confidence)}`, type: "button", "data-id": sp.id,
            title: `${sp.confidence.toFixed(2)} · ${sp.source}`,
            onclick: () => setFocus(visibleSpans().findIndex((s) => s.id === sp.id)),
          }, sp.text.replace(/\s+/g, " ")))),
        el("div", { class: "entity-actions" },
          el("button", { class: "pill btn small", type: "button", onclick: () => setMany(spans, "approved") }, "Approve all"),
          el("button", { class: "pill btn small", type: "button", onclick: () => setMany(spans, "rejected") }, "Reject all")));
    }));
    updateMarks();
  }

  function renderProgress() {
    const spans = state.doc.spans;
    const n = spans.length || 1;
    let a = 0, r = 0;
    for (const sp of spans) { const s = statusOf(sp); if (s === "approved") a++; else if (s === "rejected") r++; }
    const p = spans.length - a - r;
    $("#prog-approved").style.width = `${(a / n) * 100}%`;
    $("#prog-rejected").style.width = `${(r / n) * 100}%`;
    $("#progress").setAttribute("aria-valuenow", String(Math.round(((a + r) / n) * 100)));
    const lbl = $("#progress-label");
    lbl.replaceChildren(el("b", { text: String(a) }), " approved · ", el("b", { text: String(r) }), " rejected · ",
      el("b", { text: String(p) }), " pending");
    renderExport(p);
  }

  // ---------------------------------------------------------------- export
  function pendingCount() {
    return state.doc ? state.doc.spans.filter((sp) => statusOf(sp) === "pending").length : 0;
  }

  function renderExport(pending = pendingCount()) {
    const ok = pending === 0 || state.allowPending;
    $("#export-txt").disabled = !ok;
    $("#export-pdf").disabled = !ok;
    for (const b of document.querySelectorAll(".mode-btn")) b.setAttribute("aria-pressed", String(b.dataset.mode === state.mode));
    const note = $("#export-note");
    if (pending === 0) {
      note.textContent = "Everything reviewed. Approved spans and all their linked variants will be removed.";
    } else if (state.allowPending) {
      note.textContent = `Redact anyway is on: ${pending} pending span${pending > 1 ? "s" : ""} will stay in the file.`;
    } else {
      note.replaceChildren(`${pending} span${pending > 1 ? "s" : ""} still pending.`,
        el("button", { class: "pill btn small", type: "button", onclick: askRedactAnyway }, "Redact anyway…"));
    }
  }

  function askRedactAnyway() {
    const n = pendingCount();
    $("#confirm-body").textContent =
      `${n} span${n > 1 ? "s are" : " is"} still pending. Pending spans are NOT redacted: they stay readable in the exported file. ` +
      "Only approved spans and their linked variants are removed.";
    $("#confirm").hidden = false;
    $("#confirm-yes").focus();
  }

  async function exportDoc(fmt) {
    if (pendingCount() && !state.allowPending) { askRedactAnyway(); return; }
    const spans = state.doc.spans.map((sp) => ({ ...sp, status: statusOf(sp) }));
    toast(`Building ${fmt.toUpperCase()} and verifying…`);
    try {
      const res = await fetch("/api/export", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: state.doc.text, spans, entities: state.doc.entities,
          mode: state.mode, format: fmt, allow_pending: state.allowPending,
          generalizations: state.generalizations }),
      });
      const body = await res.json();
      if (!res.ok) throw new Error(body.detail || `Export failed (${res.status}).`);
      state.lastExport = body;
      renderReceipt(body.receipt);
      startSummary(spans);
    } catch (err) {
      toast(err.message);
    }
  }

  function download(name, blob) {
    const url = URL.createObjectURL(blob);
    const a = el("a", { href: url, download: name });
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function downloadFile() {
    const f = state.lastExport.file;
    const bytes = Uint8Array.from(atob(f.base64), (c) => c.charCodeAt(0));
    download(f.name, new Blob([bytes], { type: f.mime }));
  }

  function downloadReceipt() {
    const r = state.lastExport.receipt;
    download(r.file.name.replace(/\.[^.]+$/, "") + ".receipt.json",
      new Blob([JSON.stringify(r, null, 2)], { type: "application/json" }));
  }

  // ---------------------------------------------------------------- summary + QI
  const escapeHtml = (s) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const withPlaceholders = (s) => escapeHtml(s).replace(/\[[A-Z]+(?:_[A-Z]+)*(?:_\d+)?\]/g, (m) => `<span class="ph">${m}</span>`);

  async function startSummary(spans) {
    const key = JSON.stringify([state.mode, spans.map((s) => [s.start, s.end, s.status])]);
    const panel = $("#summary-panel");
    panel.hidden = false;
    if (state.summaryKey === key && state.summary) { renderSummary(); return; }
    state.summaryKey = key;
    $("#sp-badges").replaceChildren(el("span", { class: "pill pill-label dashed", text: "Summarising…" }));
    $("#sp-note").hidden = true;
    const t = $("#sp-text");
    t.className = "sp-text loading";
    t.textContent = "Summarising the redacted text only, then auditing the summary for leaks…";
    $("#sp-qi-list").replaceChildren();
    $("#risk-label").textContent = "";
    try {
      const res = await fetch("/api/summarize", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: state.doc.text, spans, entities: state.doc.entities, mode: state.mode }),
      });
      const body = await res.json();
      if (!res.ok) throw new Error(body.detail || `Summary failed (${res.status}).`);
      state.summary = body;
      renderSummary();
    } catch (err) {
      t.className = "sp-text";
      t.textContent = err.message;
      $("#sp-badges").replaceChildren(el("span", { class: "pill pill-label dashed", text: "No summary" }));
    }
  }

  function renderSummary() {
    const s = state.summary;
    const a = s.audit;
    const badges = [];
    badges.push(s.method === "llm"
      ? el("span", { class: "pill pill-label", text: `${s.model} · ${s.seconds.toFixed(1)} s` })
      : el("span", { class: "pill pill-label dashed", text: `Extractive summary · ${s.fallback_reason}` }));
    badges.push(el("span", { class: `pill pill-label ${a.passed ? "strong" : "dashed"}`,
      text: a.passed ? `Leak audit: passed · ${a.checks} checks` : `Leak audit: ${a.remaining_hits.length} unresolved` }));
    if (a.regenerated) badges.push(el("span", { class: "pill pill-label", text: "Regenerated once (strict prompt)" }));
    if (a.replaced.length) badges.push(el("span", { class: "pill pill-label", text: `${a.replaced.length} leak${a.replaced.length > 1 ? "s" : ""} replaced with placeholders` }));
    $("#sp-badges").replaceChildren(...badges);

    const note = $("#sp-note");
    note.hidden = s.method === "llm";
    if (s.method !== "llm") note.textContent = `Extractive fallback: the top-scored sentences from the redacted text, shown because ${s.fallback_reason}.`;

    // summary text with QI highlights
    const text = s.summary;
    const qi = s.qi;
    let html = "", pos = 0;
    for (const q of qi.sentences) {
      html += withPlaceholders(text.slice(pos, q.start));
      html += `<span class="qi ${q.level}" title="${escapeHtml(q.explanation)}">${withPlaceholders(text.slice(q.start, q.end))}</span>`;
      pos = q.end;
    }
    html += withPlaceholders(text.slice(pos));
    const t = $("#sp-text");
    t.className = "sp-text";
    t.innerHTML = html;

    renderMeter(qi);
    $("#sp-qi-list").replaceChildren(...qi.sentences.filter((q) => q.level !== "low").map((q) =>
      qiItem(q, "Generalise", () => generaliseSummary(q))));
  }

  function renderMeter(qi) {
    const n = { none: 0, low: 1, medium: 2, high: 3 }[qi.max_level];
    document.querySelectorAll("#risk-meter .rm-seg").forEach((seg, i) => seg.classList.toggle("on", i < n));
    const risky = qi.counts.high + qi.counts.medium;
    $("#risk-label").replaceChildren(
      qi.max_level === "none" ? "No quasi-identifiers found." :
        el("span", {}, "Highest risk: ", el("b", { text: qi.max_level }),
          ` · ${risky} risky of ${qi.total_sentences} sentence${qi.total_sentences === 1 ? "" : "s"}` +
          (qi.counts.low ? ` · ${qi.counts.low} low` : "")));
  }

  function qiItem(q, label, onClick, done = false) {
    const quote = el("q");
    let pos = 0;
    for (const t of q.terms) {
      const s = t.start - q.start, e = t.end - q.start;
      quote.append(q.text.slice(pos, s), el("span", { class: "term", title: t.label, text: q.text.slice(s, e) }));
      pos = e;
    }
    quote.append(q.text.slice(pos));
    const swaps = q.terms.filter((t) => t.replacement).map((t) => `${t.text} → ${t.replacement}`).join(", ");
    return el("div", { class: `qi-item ${q.level}` },
      el("span", { class: "pill pill-label strong", text: q.level }),
      quote,
      el("p", { class: "why", text: q.explanation }),
      done ? el("p", { class: "done", text: `Generalised: ${swaps}` }) :
        swaps ? el("button", { class: "pill btn small", type: "button", title: swaps, onclick: onClick }, label) : null);
  }

  async function generaliseSummary(q) {
    const s = state.summary;
    let text = s.summary;
    for (const t of [...q.terms].sort((a, b) => b.start - a.start)) {
      if (t.replacement) text = text.slice(0, t.start) + t.replacement + text.slice(t.end);
    }
    s.summary = text;
    try {
      s.qi = await (await fetch("/api/qi", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }) })).json();
    } catch (_) { /* keep the old report */ }
    renderSummary();
    toast("Sentence generalised.");
  }

  async function checkDocumentQI() {
    if (!state.lastExport) return;
    const list = $("#doc-qi-list");
    list.replaceChildren(el("p", { class: "risk-label", text: "Scanning the redacted document…" }));
    const qi = await (await fetch("/api/qi", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: state.lastExport.redacted_text }) })).json();
    const risky = qi.sentences.filter((q) => q.level !== "low");
    const applied = new Set(state.generalizations.map((g) => g.find));
    list.replaceChildren(
      el("p", { class: "risk-label", text: risky.length
        ? `${risky.length} risky sentence${risky.length > 1 ? "s" : ""} in the redacted document (highest: ${qi.max_level}).`
        : "No medium or high risk sentences in the redacted document." }),
      ...risky.map((q) => {
        const terms = q.terms.filter((t) => t.replacement);
        const done = terms.length > 0 && terms.every((t) => applied.has(t.text));
        const item = qiItem(q, "Generalise in export", () => {
          for (const t of terms) {
            if (!state.generalizations.some((g) => g.find === t.text)) state.generalizations.push({ find: t.text, replace: t.replacement });
          }
          item.replaceWith(qiItem(q, "", null, true));
          $("#doc-qi-export").hidden = false;
          toast(`${state.generalizations.length} term${state.generalizations.length > 1 ? "s" : ""} will be generalised in the next export.`);
        }, done);
        return item;
      }));
    $("#doc-qi-export").hidden = state.generalizations.length === 0;
  }

  function renderReceipt(r) {
    const v = r.verification;
    const row = (k, ...val) => [el("dt", { text: k }), el("dd", {}, ...val)];
    const types = Object.entries(r.removed.by_type).map(([t, n]) =>
      el("span", { class: "pill pill-label" }, `${TYPE_TAG[t] || t} `, el("b", { text: String(n) })));
    const ts = new Date(r.created_at);
    const rows = [
      ...row("File", r.file.name, ` · ${(r.file.bytes / 1024).toFixed(1)} KB`),
      ...row("Style", `${r.format.toUpperCase()} · ${r.mode === "mask" ? "mask [NAME]" : "pseudonym [PERSON_1]"}`),
      ...row("Removed", `${r.removed.total} item${r.removed.total === 1 ? "" : "s"} across ${r.removed.entities} entit${r.removed.entities === 1 ? "y" : "ies"}`),
      ...row("By type", el("div", { class: "types" }, ...types)),
      ...(r.generalizations ? row("Generalised", `${r.generalizations} quasi-identifier term${r.generalizations > 1 ? "s" : ""}`) : []),
      ...row("Checks", `${v.checks} performed · ${v.recoverable} recoverable` + (v.skipped.length ? ` · ${v.skipped.length} skipped` : "")),
    ];
    if (r.pdf_metadata) rows.push(...row("Metadata", r.pdf_metadata.info_entries === 0 && !r.pdf_metadata.xmp ? "None: no author, title or producer; no XMP" : `${r.pdf_metadata.info_entries} entries · XMP ${r.pdf_metadata.xmp ? "present" : "absent"}`));
    if (r.review.pending_left_unredacted || r.review.overridden_rejections) {
      const notes = [];
      if (r.review.pending_left_unredacted) notes.push(`${r.review.pending_left_unredacted} pending left unredacted`);
      if (r.review.overridden_rejections) notes.push(`${r.review.overridden_rejections} rejected mention${r.review.overridden_rejections > 1 ? "s" : ""} redacted because the same entity was approved`);
      rows.push(...row("Review", notes.join(" · ")));
    }
    rows.push(...row("SHA-256", el("span", { class: "hash", text: r.file.sha256 })));
    rows.push(...row("Issued", `${ts.toLocaleString()} (${r.created_at})`));

    const card = $("#receipt-card");
    card.replaceChildren(
      el("p", { class: "eyebrow", id: "receipt-title", text: "Redaction receipt" }),
      el("h3", { class: "display mid", text: `${v.recoverable} recoverable` }),
      el("p", { class: "verdict" }, el("span", {
        class: `pill pill-label ${v.passed ? "strong" : "dashed"}`,
        text: v.passed ? "Verified: nothing removed can be found in the exported file" : "Check failed: some removed values are still present",
      })),
      ...(v.failures.length ? [el("ul", { class: "fail-list" }, ...v.failures.map((f) =>
        el("li", { text: `${f.kind.replace("_", " ")} · ${TYPE_TAG[f.type] || f.type} · ${f.hint}` })))] : []),
      el("hr", { class: "tear" }),
      el("dl", {}, ...rows),
      el("hr", { class: "tear" }),
      el("div", { class: "chips actions no-print" },
        el("button", { class: "pill btn solid", type: "button", onclick: downloadFile }, `Download ${r.format.toUpperCase()}`),
        el("button", { class: "pill btn", type: "button", onclick: downloadReceipt }, "Receipt JSON"),
        el("button", { class: "pill btn", type: "button", onclick: () => window.print() }, "Print"),
        el("button", { class: "pill btn", type: "button", onclick: () => { $("#receipt").hidden = true; } }, "Close")),
    );
    $("#receipt").hidden = false;
  }

  // ---------------------------------------------------------------- focus & actions
  function currentSpan() {
    const v = visibleSpans();
    return state.focus >= 0 && state.focus < v.length ? v[state.focus] : null;
  }

  function setFocus(i, scroll = true) {
    const v = visibleSpans();
    if (!v.length) { state.focus = -1; updateMarks(); return; }
    state.focus = Math.max(0, Math.min(v.length - 1, i));
    updateMarks();
    if (!scroll) return;
    const sp = v[state.focus];
    document.querySelector(`mark[data-id="${sp.id}"]`)?.scrollIntoView({ block: "center", behavior: "smooth" });
    document.querySelector(`.mention[data-id="${sp.id}"]`)?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }

  function setMany(spans, st, learnFrom = true) {
    for (const sp of spans) state.status.set(key(sp), st);
    updateMarks();
    renderProgress();
    if (learnFrom && (st === "approved" || st === "rejected")) {
      const seen = new Set();
      const events = [];
      for (const sp of spans) {
        if (sp.source === "manual") continue;
        const k = `${sp.type}|${sp.text}`;
        if (seen.has(k)) continue;
        seen.add(k);
        events.push({ action: st === "approved" ? "approve" : "reject", text: sp.text, type: sp.type, source: sp.source });
      }
      queueFeedback(events);
    }
  }

  // ---------------------------------------------------------------- learning
  let feedbackTimer;
  function queueFeedback(events, now = false) {
    if (!events.length) return;
    state.feedback.push(...events);
    clearTimeout(feedbackTimer);
    feedbackTimer = setTimeout(flushFeedback, now ? 0 : 350);
  }

  function docPayload() {
    return { text: state.doc.text, filename: state.doc.filename,
      manual_spans: state.manual.map(({ start, end, text, type }) => ({ start, end, text, type })) };
  }

  async function flushFeedback() {
    if (!state.feedback.length || !state.doc) return;
    const events = state.feedback.splice(0);
    const seq = ++state.feedbackSeq;
    try {
      const res = await fetch("/api/feedback", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ events, ...docPayload() }),
      });
      if (!res.ok) throw new Error();
      const body = await res.json();
      setLearning(body.learning);
      if (seq === state.feedbackSeq && !state.feedback.length) applyAnalysis(body.analysis);
    } catch (_) {
      toast("Could not save that feedback.");
    }
  }

  function applyAnalysis(a) {
    if (!a || !state.doc || a.text !== state.doc.text) return;
    const focusedKey = currentSpan() ? key(currentSpan()) : null;
    const oldIndex = state.focus;
    const before = state.doc.spans.length;
    setDoc({ ...state.doc, spans: a.spans, entities: a.entities }, true);
    const v = visibleSpans();
    const i = focusedKey ? v.findIndex((s) => key(s) === focusedKey) : -1;
    state.focus = -1;
    setFocus(i >= 0 ? i : Math.min(oldIndex, v.length - 1), false);
    const diff = a.spans.length - before;
    if (diff < 0) toast(`Learned: ${-diff} matching span${diff < -1 ? "s" : ""} cleared.`);
    else if (diff > 0) toast(`Learned: ${diff} more span${diff > 1 ? "s" : ""} flagged.`);
  }

  function setLearning(l) {
    state.learning = l;
    $("#learn-count").textContent = String(l.corrections);
    $("#learn-n").textContent = String(l.corrections);
    $("#learn-path").textContent = l.path;
    const p = l.precision || {};
    const fmt = (x) => (x == null ? "—" : `${(x * 100).toFixed(1)}%`);
    $("#precision-value").textContent = `${fmt(p.before)} → ${fmt(p.after)}`;
    $("#precision-note").textContent = p.before == null ? "Run eval/run_eval.py to measure." :
      `Active-learning demo: ${p.note}. Recall ${fmt(p.recall_before)} → ${fmt(p.recall_after)}.`;
    if (!$("#learn-drawer").hidden) renderLearning();
  }

  async function loadLearning() {
    try { setLearning(await (await fetch("/api/learn")).json()); } catch (_) { /* optional */ }
  }

  function renderLearning() {
    const l = state.learning;
    if (!l) return;
    const groups = [
      ["allowlist", "Allowlist", "Rejected strings, never flagged again."],
      ["denylist", "Denylist", "Approved or manually added strings, always flagged."],
      ["shape", "Non-PII shapes", "Learned after 2 rejections of the same shape."],
      ["candidate", "Candidate rules", "Learned after 2 manual adds of the same shape."],
    ];
    const undoBtn = (id) => el("button", { class: "pill btn small", type: "button", onclick: () => undoRule(id) }, "Undo");
    const sections = groups.map(([kind, title, hint]) => {
      const rules = l.rules.filter((r) => r.kind === kind);
      return el("section", { class: "learn-section" },
        el("h4", { text: `${title} · ${rules.length}` }), el("p", { class: "fine", text: hint }),
        ...(rules.length ? rules.map((r) => el("div", { class: "rule" },
          el("span", { class: "what" }, el("span", { class: "pill pill-label", text: TYPE_TAG[r.type] || r.type }),
            el("b", { text: r.text }), el("span", { class: "detail", text: r.detail })),
          undoBtn(r.id))) : [el("p", { class: "empty-rules", text: "Nothing yet." })]));
    });
    const progress = (l.shape_progress || []).map((p) =>
      el("p", { class: "fine", text: `${TYPE_TAG[p.type] || p.type} shape ${p.shape}: ${p.seen} of ${p.needed} rejections` }));
    sections.splice(3, 0, ...progress);
    sections.push(el("section", { class: "learn-section" },
      el("h4", { text: "GLiNER thresholds" }),
      el("p", { class: "fine", text: "Calibrated per type from approvals and rejections, kept within 0.25–0.70." }),
      ...(l.thresholds.length ? l.thresholds.map((t) => el("div", { class: "rule" },
        el("span", { class: "what" }, el("span", { class: "pill pill-label", text: TYPE_TAG[t.type] || t.type }),
          el("b", { text: `${t.base.toFixed(2)} → ${t.value.toFixed(2)}` }),
          el("span", { class: "detail", text: `${t.approve} approved · ${t.manual} manual · ${t.reject} rejected` })),
        undoBtn(t.id))) : [el("p", { class: "empty-rules", text: "All types at the default 0.45." })])));
    $("#learn-sections").replaceChildren(...sections);
  }

  async function undoRule(id) {
    const body = { rule_id: id, ...(state.doc ? docPayload() : {}) };
    try {
      const res = await fetch("/api/learn/undo", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      if (!res.ok) throw new Error();
      const j = await res.json();
      setLearning(j.learning);
      renderLearning();
      if (j.analysis) applyAnalysis(j.analysis);
      toast("Rule undone.");
    } catch (_) { toast("Could not undo that rule."); }
  }

  let resetArmed = false;
  async function resetLearning() {
    const btn = $("#learn-reset");
    if (!resetArmed) {
      resetArmed = true;
      btn.textContent = "Click again to forget everything";
      btn.classList.add("solid");
      setTimeout(() => { resetArmed = false; btn.textContent = "Reset learning"; btn.classList.remove("solid"); }, 4000);
      return;
    }
    resetArmed = false;
    btn.textContent = "Reset learning";
    btn.classList.remove("solid");
    const j = await (await fetch("/api/learn/reset", { method: "POST" })).json();
    setLearning(j.learning);
    renderLearning();
    if (state.doc) {
      const res = await fetch("/api/feedback", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ events: [], ...docPayload() }) });
      if (res.ok) applyAnalysis((await res.json()).analysis);
    }
    toast("Learning reset.");
  }

  const toggleDrawer = (force) => {
    const d = $("#learn-drawer");
    d.hidden = force === undefined ? !d.hidden : !force;
    if (!d.hidden) renderLearning();
  };

  function renderLocal(st) {
    const n = st.network || {};
    const pill = $("#local-pill");
    pill.textContent = `Local only · ${n.outbound_calls ?? "?"} outbound`;
    pill.classList.toggle("dashed", !n.guard_installed);
    const d = st.detector, o = st.ollama;
    const rows = [
      ["GLiNER", d.status === "loaded" ? "loaded" : d.status === "unavailable" ? "unavailable (rules only)" : "loads on first document"],
      ["Ollama", o.reachable ? `running · ${o.summary_model}${o.model_installed ? "" : " not pulled"}` : "offline (extractive fallback)"],
      ["Outbound network calls", String(n.outbound_calls ?? "?")],
      ["Blocked attempts", String(n.blocked_attempts ?? "?")],
      ["Local connections", String(n.local_connections ?? "?")],
    ];
    $("#local-rows").replaceChildren(...rows.flatMap(([k, v]) => [el("dt", { text: k }), el("dd", { text: v })]));
  }

  function decide(st) {
    const sp = currentSpan();
    if (!sp) return;
    setMany([sp], st);
    // advance to the next pending span
    const v = visibleSpans();
    for (let k = 1; k <= v.length; k++) {
      const j = (state.focus + k) % v.length;
      if (statusOf(v[j]) === "pending") { setFocus(j); return; }
    }
    toast("Everything here has been reviewed.");
  }

  function redactSimilar() {
    const sp = currentSpan();
    if (!sp) return;
    let group, why;
    const ent = entityOf(sp);
    if (sp.type === "NAME" && ent) {
      group = state.doc.spans.filter((s) => s.entity_id === ent.id);
      why = `${ent.pseudonym} (${ent.canonical})`;
    } else {
      const sh = shape(sp);
      group = state.doc.spans.filter((s) => s.entity_id === sp.entity_id || (s.type === sp.type && shape(s) === sh));
      why = `${TYPE_LABEL[sp.type].toLowerCase()} shaped like “${sp.text.replace(/\s+/g, " ")}”`;
    }
    setMany(group, "approved");
    toast(`Approved ${group.length} similar: ${why}`);
  }

  // ---------------------------------------------------------------- manual redaction
  function selectionOffsets() {
    const sel = window.getSelection();
    if (!sel || sel.isCollapsed || !sel.rangeCount) return null;
    const range = sel.getRangeAt(0);
    const docEl = $("#doc");
    if (!docEl.contains(range.commonAncestorContainer)) return null;
    const off = (node, o) => {
      let n = node.nodeType === 3 ? node.parentElement : node;
      if (n.closest(".tag")) return null;
      const host = n.closest("[data-start]");
      if (!host) return null;
      if (node.nodeType !== 3) { // element boundary: count chars of preceding children
        let t = 0;
        for (let i = 0; i < o && i < node.childNodes.length; i++) t += node.childNodes[i].textContent.length;
        o = t;
      }
      return Number(host.dataset.start) + o;
    };
    let s = off(range.startContainer, range.startOffset);
    let e = off(range.endContainer, range.endOffset);
    if (s == null || e == null || e <= s) return null;
    const text = state.doc.text;
    while (s < e && /\s/.test(text[s])) s++;
    while (e > s && /\s/.test(text[e - 1])) e--;
    if (e <= s) return null;
    return { start: s, end: e, text: text.slice(s, e), rect: range.getBoundingClientRect() };
  }

  function showSelButton() {
    const s = selectionOffsets();
    const btn = $("#sel-btn");
    state.selection = s;
    if (!s) { btn.hidden = true; return; }
    btn.style.left = `${window.scrollX + s.rect.left}px`;
    btn.style.top = `${window.scrollY + s.rect.bottom + 8}px`;
    btn.hidden = false;
  }

  function openTypePop() {
    const s = state.selection || selectionOffsets();
    if (!s) { toast("Select some text in the document first, then press M."); return; }
    state.selection = s;
    const overlap = state.doc.spans.find((sp) => s.start < sp.end && s.end > sp.start);
    if (overlap) { toast("That selection overlaps an existing span. Approve or reject that one instead."); return; }
    const pop = $("#type-pop");
    $("#type-pop-chips").replaceChildren(...TYPE_ORDER.map((t, i) =>
      el("button", { class: "pill btn chip", type: "button", onclick: () => addManual(t) },
        TYPE_TAG[t], el("kbd", { text: String(i + 1) }))));
    pop.style.left = `${window.scrollX + s.rect.left}px`;
    pop.style.top = `${window.scrollY + s.rect.bottom + 8}px`;
    pop.hidden = false;
    $("#sel-btn").hidden = true;
  }

  async function addManual(type) {
    const s = state.selection;
    $("#type-pop").hidden = true;
    if (!s) return;
    state.manual.push({ start: s.start, end: s.end, text: s.text, type });
    state.status.set(`${s.start}:${s.end}`, "approved");
    window.getSelection().removeAllRanges();
    state.selection = null;
    clearTimeout(feedbackTimer);
    state.feedback.push({ action: "manual", text: s.text, type });
    const before = state.doc.spans.length + 1;
    const seq = ++state.feedbackSeq;
    try {
      const res = await fetch("/api/feedback", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ events: state.feedback.splice(0), ...docPayload() }),
      });
      if (!res.ok) throw new Error();
      const body = await res.json();
      setLearning(body.learning);
      if (seq !== state.feedbackSeq) return;
      setDoc({ ...state.doc, spans: body.analysis.spans, entities: body.analysis.entities }, true);
      const added = body.analysis.spans.length - before;
      const v = visibleSpans();
      setFocus(Math.max(0, v.findIndex((x) => x.start === s.start && x.end === s.end)));
      toast(`Added ${TYPE_TAG[type]} “${s.text}”` + (added > 0 ? ` and ${added} more occurrence${added > 1 ? "s" : ""}.` : "."));
    } catch (_) {
      toast("Could not add that redaction.");
    }
  }

  // ---------------------------------------------------------------- misc UI
  let toastTimer;
  function toast(msg) {
    const t = $("#toast");
    t.textContent = msg;
    t.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { t.hidden = true; }, 2600);
  }

  const toggleHelp = (force) => { const h = $("#help"); h.hidden = force === undefined ? !h.hidden : !force; };

  function onKey(e) {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const tag = (e.target.tagName || "").toLowerCase();
    if (tag === "textarea" || tag === "input") return;
    if (e.key === "Escape") {
      toggleDrawer(false); $("#local-pop").hidden = true;
      toggleHelp(false); $("#type-pop").hidden = true; $("#confirm").hidden = true; $("#receipt").hidden = true;
      return;
    }
    if (!$("#receipt").hidden || !$("#confirm").hidden) return;
    if (e.key.toLowerCase() === "l") { e.preventDefault(); toggleDrawer(); return; }
    if (e.key === "?") { e.preventDefault(); toggleHelp(); return; }
    if (!$("#help").hidden) return;
    const pop = $("#type-pop");
    if (!pop.hidden && /^[1-6]$/.test(e.key)) { e.preventDefault(); addManual(TYPE_ORDER[Number(e.key) - 1]); return; }
    if (!state.doc || $("#review").hidden) return;
    const k = e.key.toLowerCase();
    const actions = {
      a: () => decide("approved"),
      r: () => decide("rejected"),
      j: () => setFocus(state.focus + 1),
      k: () => setFocus(state.focus - 1),
      s: redactSimilar,
      m: openTypePop,
      e: () => exportDoc("pdf"),
    };
    if (actions[k]) { e.preventDefault(); actions[k](); }
  }

  function wire() {
    const input = $("#file-input");
    const pick = () => input.click();
    $("#open-btn").addEventListener("click", pick);
    $("#choose-btn").addEventListener("click", pick);
    input.addEventListener("change", () => { if (input.files[0]) openFile(input.files[0]); input.value = ""; });

    const dz = $("#dropzone");
    dz.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); } });
    for (const ev of ["dragenter", "dragover"]) dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("over"); });
    for (const ev of ["dragleave", "drop"]) dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("over"); });
    dz.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) openFile(f); });

    $("#paste-btn").addEventListener("click", () => {
      const t = $("#paste-text").value;
      if (t.trim()) openText(t);
    });
    $("#help-btn").addEventListener("click", () => toggleHelp());
    $("#help-close").addEventListener("click", () => toggleHelp(false));
    $("#help").addEventListener("click", (e) => { if (e.target.id === "help") toggleHelp(false); });
    $("#sel-btn").addEventListener("mousedown", (e) => e.preventDefault());
    $("#sel-btn").addEventListener("click", openTypePop);
    $("#doc").addEventListener("mouseup", () => setTimeout(showSelButton, 0));
    document.addEventListener("mousedown", (e) => {
      if (!e.target.closest("#type-pop, #sel-btn, #doc")) { $("#type-pop").hidden = true; $("#sel-btn").hidden = true; }
    });
    document.addEventListener("keydown", onKey);

    for (const b of document.querySelectorAll(".mode-btn")) {
      b.addEventListener("click", () => { state.mode = b.dataset.mode; renderExport(); });
    }
    $("#export-txt").addEventListener("click", () => exportDoc("txt"));
    $("#export-pdf").addEventListener("click", () => exportDoc("pdf"));
    $("#confirm-no").addEventListener("click", () => { $("#confirm").hidden = true; });
    $("#confirm-yes").addEventListener("click", () => {
      state.allowPending = true;
      $("#confirm").hidden = true;
      renderExport();
    });
    $("#receipt").addEventListener("click", (e) => { if (e.target.id === "receipt") $("#receipt").hidden = true; });
    $("#learn-btn").addEventListener("click", () => toggleDrawer());
    $("#learn-close").addEventListener("click", () => toggleDrawer(false));
    $("#learn-reset").addEventListener("click", resetLearning);
    $("#local-pill").addEventListener("click", () => {
      const p = $("#local-pop");
      p.hidden = !p.hidden;
      $("#local-pill").setAttribute("aria-expanded", String(!p.hidden));
      if (!p.hidden) loadStatus();
    });
    document.addEventListener("mousedown", (e) => {
      if (!e.target.closest("#local-pop, #local-pill")) $("#local-pop").hidden = true;
    });
    $("#sp-close").addEventListener("click", () => { $("#summary-panel").hidden = true; });
    $("#doc-qi-btn").addEventListener("click", checkDocumentQI);
    $("#gen-pdf").addEventListener("click", () => exportDoc("pdf"));
    $("#gen-txt").addEventListener("click", () => exportDoc("txt"));
  }

  wire();
  loadStatus();
  loadSamples();
  loadLearning();
  setInterval(loadStatus, 15000);
  // deep link for demos: /#sample=hr_offer_letter.txt
  const m = location.hash.match(/^#sample=(.+)$/);
  if (m) openSample(decodeURIComponent(m[1]));
})();
