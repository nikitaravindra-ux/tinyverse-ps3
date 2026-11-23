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
    lastExport: null,   // {receipt, file}
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
    if (!keepStatus) { state.filter = "ALL"; state.focus = -1; state.allowPending = false; }
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
          mode: state.mode, format: fmt, allow_pending: state.allowPending }),
      });
      const body = await res.json();
      if (!res.ok) throw new Error(body.detail || `Export failed (${res.status}).`);
      state.lastExport = body;
      renderReceipt(body.receipt);
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

  function setMany(spans, st) {
    for (const sp of spans) state.status.set(key(sp), st);
    updateMarks();
    renderProgress();
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
    const strip = ({ id, entity_id, ...rest }) => rest;
    const spans = state.doc.spans.map(strip);
    spans.push({ start: s.start, end: s.end, text: s.text, type, confidence: 1.0, source: "manual" });
    spans.sort((a, b) => a.start - b.start);
    state.status.set(`${s.start}:${s.end}`, "approved");
    window.getSelection().removeAllRanges();
    state.selection = null;
    try {
      const res = await fetch("/api/link", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: state.doc.text, spans }),
      });
      if (!res.ok) throw new Error();
      const body = await res.json();
      const before = state.doc.spans.length + 1;
      setDoc({ ...state.doc, spans: body.spans, entities: body.entities }, true);
      const added = body.spans.length - before;
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
      toggleHelp(false); $("#type-pop").hidden = true; $("#confirm").hidden = true; $("#receipt").hidden = true;
      return;
    }
    if (!$("#receipt").hidden || !$("#confirm").hidden) return;
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
  }

  wire();
  loadStatus();
  loadSamples();
  // deep link for demos: /#sample=hr_offer_letter.txt
  const m = location.hash.match(/^#sample=(.+)$/);
  if (m) openSample(decodeURIComponent(m[1]));
})();
