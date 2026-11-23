"""Local learning from reviewer feedback, stored in data/learned.json (git-ignored).

  reject          -> exact string goes on the allowlist (never flagged again);
                     2 rejected strings of the same shape (e.g. AAA-99999-A) -> that shape
                     becomes a non-PII pattern for the type
  approve         -> exact string goes on the denylist (always flagged in future docs)
  manual add      -> denylist too; 2 manual adds of the same new shape -> candidate regex
                     rule for that type
  every event     -> per-type GLiNER threshold recalibrated from the approve/reject ratio,
                     clamped to [0.25, 0.70]

Shapes are only learned when they contain a digit (structured codes). A letters-only
shape like "Aaaaa" would match every capitalised five-letter word, so names are learned
through the allow/deny lists only.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.detect import GLINER_THRESHOLD, GLINER_TO_TYPE

ROOT = Path(__file__).resolve().parent.parent
STORE_PATH = Path(os.environ.get("PS3_LEARN_PATH", ROOT / "data" / "learned.json"))

THRESHOLD_MIN, THRESHOLD_MAX = 0.25, 0.70
THRESHOLD_SPAN = 0.25          # max movement from the base threshold at extreme ratios
THRESHOLD_PRIOR = 4            # smoothing: a few events only nudge slightly
SHAPE_MIN_EXAMPLES = 2
GLINER_TYPES = sorted(set(GLINER_TO_TYPE.values()))

_lock = threading.RLock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _rid(prefix: str, *parts: str) -> str:
    return prefix + ":" + hashlib.sha1("\x1f".join(parts).encode()).hexdigest()[:10]


def shape_of(text: str) -> str:
    """'SKU-88213-B' -> 'AAA-99999-A'; whitespace collapsed."""
    t = re.sub(r"\s+", " ", text.strip())
    return re.sub(r"[A-Z]", "A", re.sub(r"[a-z]", "a", re.sub(r"\d", "9", t)))


def shape_regex(shape: str) -> str:
    out = []
    for m in re.finditer(r"A+|a+|9+| |.", shape):
        run = m.group(0)
        cls = {"A": "[A-Z]", "a": "[a-z]", "9": r"\d", " ": r"\s+"}.get(run[0])
        if cls is None:
            out.append(re.escape(run))
        elif run[0] == " ":
            out.append(cls)
        else:
            out.append(cls + (f"{{{len(run)}}}" if len(run) > 1 else ""))
    return r"(?<![\w-])" + "".join(out) + r"(?![\w-])"


def shape_learnable(shape: str) -> bool:
    return "9" in shape and len(shape) >= 4


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------
def _empty() -> dict:
    return {"version": 1, "allowlist": [], "denylist": [], "reject_shapes": {}, "manual_shapes": {},
            "shapes": [], "candidates": [], "type_stats": {}, "events": 0, "updated": None}


def load() -> dict:
    with _lock:
        try:
            data = json.loads(STORE_PATH.read_text(encoding="utf-8"))
            base = _empty()
            base.update(data)
            return base
        except (FileNotFoundError, json.JSONDecodeError):
            return _empty()


def save(state: dict) -> None:
    with _lock:
        state["updated"] = _now()
        STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = STORE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(STORE_PATH)


def reset() -> None:
    with _lock:
        save(_empty())


# ---------------------------------------------------------------------------
# Recording feedback
# ---------------------------------------------------------------------------
def _remove_entry(lst: list, text: str) -> None:
    n = _norm(text)
    lst[:] = [e for e in lst if _norm(e["text"]) != n]


def _bump_stats(state: dict, typ: str, key: str) -> None:
    st = state["type_stats"].setdefault(typ, {"approve": 0, "reject": 0, "manual": 0})
    st[key] += 1


def _learn_shape(state: dict, bucket: str, target: str, typ: str, text: str, rid_prefix: str) -> None:
    shp = shape_of(text)
    if not shape_learnable(shp):
        return
    key = f"{typ}|{shp}"
    examples = state[bucket].setdefault(key, [])
    if _norm(text) not in [_norm(x) for x in examples]:
        examples.append(text)
    if len(examples) >= SHAPE_MIN_EXAMPLES and not any(r["type"] == typ and r["shape"] == shp for r in state[target]):
        state[target].append({"id": _rid(rid_prefix, typ, shp), "type": typ, "shape": shp,
                              "regex": shape_regex(shp), "examples": examples[:5], "created": _now()})


def _gliner_evidence(ev: dict) -> bool:
    """Approving a regex-validated Aadhaar says nothing about GLiNER's threshold: only
    spans the model contributed to (or events without a source, e.g. API callers) count."""
    src = ev.get("source")
    return src is None or "gliner" in src


def record(events: list[dict]) -> dict:
    """events: [{action: approve|reject|manual, text, type}]. Returns the new summary."""
    with _lock:
        state = load()
        for ev in events:
            action, text, typ = ev.get("action"), (ev.get("text") or "").strip(), ev.get("type")
            if not text or not typ or action not in ("approve", "reject", "manual"):
                continue
            state["events"] += 1
            if action == "reject":
                _remove_entry(state["denylist"], text)
                if not any(_norm(e["text"]) == _norm(text) for e in state["allowlist"]):
                    state["allowlist"].append({"id": _rid("allow", _norm(text)), "text": text, "type": typ,
                                               "created": _now()})
                if _gliner_evidence(ev):
                    _bump_stats(state, typ, "reject")
                _learn_shape(state, "reject_shapes", "shapes", typ, text, "shape")
            else:
                _remove_entry(state["allowlist"], text)
                _remove_entry(state["denylist"], text)
                state["denylist"].append({"id": _rid("deny", _norm(text)), "text": text, "type": typ,
                                          "source": action, "created": _now()})
                if action == "manual" or _gliner_evidence(ev):
                    _bump_stats(state, typ, action)
                if action == "manual":
                    _learn_shape(state, "manual_shapes", "candidates", typ, text, "cand")
                # a confirmed value must not be hidden by a learned non-PII shape
                state["shapes"] = [s for s in state["shapes"]
                                   if not (s["type"] == typ and s["shape"] == shape_of(text))]
        save(state)
        return summary(state)


def undo(rule_id: str) -> bool:
    with _lock:
        state = load()
        found = False
        for lst in ("allowlist", "denylist"):
            for e in list(state[lst]):
                if e["id"] == rule_id:
                    state[lst].remove(e)
                    found = True
                    if lst == "allowlist":  # forget it as a shape example too
                        key = f"{e['type']}|{shape_of(e['text'])}"
                        ex = state["reject_shapes"].get(key, [])
                        state["reject_shapes"][key] = [x for x in ex if _norm(x) != _norm(e["text"])]
        for lst, bucket in (("shapes", "reject_shapes"), ("candidates", "manual_shapes")):
            for r in list(state[lst]):
                if r["id"] == rule_id:
                    state[lst].remove(r)
                    state[bucket].pop(f"{r['type']}|{r['shape']}", None)  # start counting afresh
                    found = True
        if rule_id.startswith("thr:"):
            typ = rule_id.split(":", 1)[1]
            if typ in state["type_stats"]:
                del state["type_stats"][typ]
                found = True
        if found:
            save(state)
        return found


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def thresholds(state: Optional[dict] = None) -> dict[str, float]:
    state = state or load()
    out = {}
    for typ in GLINER_TYPES:
        st = state["type_stats"].get(typ, {})
        a = st.get("approve", 0) + st.get("manual", 0)
        r = st.get("reject", 0)
        thr = GLINER_THRESHOLD + THRESHOLD_SPAN * (r - a) / (r + a + THRESHOLD_PRIOR)
        out[typ] = round(min(THRESHOLD_MAX, max(THRESHOLD_MIN, thr)), 3)
    return out


# ---------------------------------------------------------------------------
# Applying what was learned
# ---------------------------------------------------------------------------
def _word_re(text: str) -> re.Pattern:
    body = r"\s+".join(re.escape(t) for t in text.split())
    return re.compile(rf"(?<![\w@]){body}(?![\w@])", re.I)


def apply(text: str, spans: list[dict], state: Optional[dict] = None) -> list[dict]:
    """Filter detector spans with the allowlist / learned shapes, then add denylist hits
    and candidate-rule hits that don't overlap anything already flagged."""
    state = state or load()
    allow = {_norm(e["text"]) for e in state["allowlist"]}
    shapes = {(s["type"], s["shape"]) for s in state["shapes"]}

    def suppressed(sp: dict) -> bool:
        return _norm(sp["text"]) in allow or (sp["type"], shape_of(sp["text"])) in shapes

    deny_norm = {_norm(e["text"]) for e in state["denylist"]}
    kept = [sp for sp in spans if _norm(sp["text"]) in deny_norm or not suppressed(sp)]

    def free(s: int, e: int) -> bool:
        return not any(s < k["end"] and e > k["start"] for k in kept)

    for entry in sorted(state["denylist"], key=lambda e: -len(e["text"])):
        for m in _word_re(entry["text"]).finditer(text):
            if free(m.start(), m.end()):
                kept.append({"start": m.start(), "end": m.end(), "text": text[m.start():m.end()],
                             "type": entry["type"], "confidence": 0.95, "source": "learned"})
    for rule in state["candidates"]:
        for m in re.finditer(rule["regex"], text):
            cand = {"start": m.start(), "end": m.end(), "text": m.group(0), "type": rule["type"],
                    "confidence": 0.8, "source": "learned-rule"}
            if free(m.start(), m.end()) and not suppressed(cand):
                kept.append(cand)
    kept.sort(key=lambda s: s["start"])
    return kept


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
EVAL_RESULTS = ROOT / "eval" / "results.json"


def eval_precision() -> dict:
    """Best honest active-learning result (largest precision change) from eval/results.json."""
    try:
        al = json.loads(EVAL_RESULTS.read_text(encoding="utf-8"))["active_learning"]
        run = al["runs"][al["best"]] if "runs" in al else al
        title = run.get("title", "Indian synthetic")
        return {"before": run["precision_before"], "after": run["precision_after"],
                "recall_before": run["recall_before"], "recall_after": run["recall_after"],
                "dataset": title, "train_docs": run["train_docs"], "test_docs": run["test_docs"],
                "note": f"{title}: simulated reviewer on {run['train_docs']} docs, "
                        f"measured on {run['test_docs']} held-out docs"}
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return {"before": None, "after": None, "note": "run eval/run_eval.py to measure"}


def summary(state: Optional[dict] = None) -> dict:
    state = state or load()
    thr = thresholds(state)
    rules = []
    for e in state["allowlist"]:
        rules.append({"id": e["id"], "kind": "allowlist", "type": e["type"], "text": e["text"],
                      "detail": "never flagged"})
    for e in state["denylist"]:
        rules.append({"id": e["id"], "kind": "denylist", "type": e["type"], "text": e["text"],
                      "detail": "always flagged" + (" (manual add)" if e.get("source") == "manual" else "")})
    for s in state["shapes"]:
        rules.append({"id": s["id"], "kind": "shape", "type": s["type"], "text": s["shape"],
                      "detail": f"non-PII pattern, learned from {len(s['examples'])} rejections"})
    for c in state["candidates"]:
        rules.append({"id": c["id"], "kind": "candidate", "type": c["type"], "text": c["shape"],
                      "detail": f"candidate rule /{c['regex']}/ from {len(c['examples'])} manual adds"})
    threshold_rows = []
    for typ, value in thr.items():
        st = state["type_stats"].get(typ)
        if st:
            threshold_rows.append({"id": f"thr:{typ}", "kind": "threshold", "type": typ,
                                   "base": GLINER_THRESHOLD, "value": value, **st})
    progress = {}
    for key, ex in state["reject_shapes"].items():
        typ, shp = key.split("|", 1)
        if 0 < len(ex) < SHAPE_MIN_EXAMPLES:
            progress[key] = {"type": typ, "shape": shp, "seen": len(ex), "needed": SHAPE_MIN_EXAMPLES}
    return {
        "corrections": len(rules),
        "events": state["events"],
        "rules": rules,
        "thresholds": threshold_rows,
        "all_thresholds": thr,
        "shape_progress": list(progress.values()),
        "precision": eval_precision(),
        "path": str(STORE_PATH.relative_to(ROOT)) if STORE_PATH.is_relative_to(ROOT) else str(STORE_PATH),
        "updated": state.get("updated"),
    }
