"""End-to-end evaluation.

    .venv/bin/python eval/run_eval.py               # clean detector (learning disabled)
    .venv/bin/python eval/run_eval.py --with-learning   # apply data/learned.json
    .venv/bin/python eval/run_eval.py --limit 50    # quick run

Reports span-level precision / recall / F1 per type (overlap match) for regex-only,
GLiNER-only and hybrid on every dataset, plus hard-negative false-positive rate,
name-linking accuracy and summary leaks on the hand-made set, quasi-identifier traps,
seconds per document, peak RAM, and the active-learning before/after precision.

Writes eval/results.json, eval/results.md and eval/results.png.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import resource
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Evaluation runs local-only: the cached GLiNER weights, no Hub calls.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from app import audit, learn, redact, summarize  # noqa: E402
from app import detect as det  # noqa: E402
from app.config import SUMMARY_MODEL  # noqa: E402
from app.link import link  # noqa: E402

TYPES = ["NAME", "EMAIL", "PHONE", "ADDRESS", "ID"]
CONFIGS = {
    "regex": {"use_ner": False},
    "gliner": {"use_ner": True, "use_rules": False},
    "hybrid": {"use_ner": True},
}
DATASETS = [
    ("indian", "Indian synthetic (Faker en_IN)", HERE / "data" / "indian_synthetic.jsonl"),
    ("ai4privacy", "ai4privacy pii-masking-300k", HERE / "cache" / "ai4privacy.jsonl"),
    ("gretel", "Gretel synthetic PII finance", HERE / "cache" / "gretel.jsonl"),
    ("nemotron", "NVIDIA Nemotron-PII", HERE / "cache" / "nemotron.jsonl"),
]
PURPLE, ALMOND = "#320b35", "#F2ECC0"


def log(msg: str) -> None:
    print(msg, flush=True)


def load_jsonl(path: Path, limit: int | None) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit else rows


# ---------------------------------------------------------------------------
# Prediction & scoring
# ---------------------------------------------------------------------------
def predict(text: str, cfg: str, learned: dict | None = None) -> list[dict]:
    det._gliner_cache.clear()  # time every config honestly: no cached model output
    thr = learn.thresholds(learned) if learned is not None else None
    spans = det.detect(text, type_thresholds=thr, **CONFIGS[cfg])
    if learned is not None:
        spans = learn.apply(text, spans, learned)
    return spans


def _type(t: str) -> str:
    return "ADDRESS" if t == "ORG_ADDRESS" else t


def _ov(a, b) -> bool:
    return a["start"] < b["end"] and b["start"] < a["end"]


class Score:
    def __init__(self):
        self.c = defaultdict(lambda: {"tp_pred": 0, "n_pred": 0, "tp_gold": 0, "n_gold": 0})

    def add(self, preds: list[dict], gold: list[dict]) -> None:
        preds = [dict(p, type=_type(p["type"])) for p in preds if _type(p["type"]) in TYPES]
        for p in preds:
            c = self.c[p["type"]]
            c["n_pred"] += 1
            c["tp_pred"] += any(g["type"] == p["type"] and _ov(g, p) for g in gold)
        for g in gold:
            c = self.c[g["type"]]
            c["n_gold"] += 1
            c["tp_gold"] += any(p["type"] == g["type"] and _ov(g, p) for p in preds)

    @staticmethod
    def prf(c) -> dict:
        p = c["tp_pred"] / c["n_pred"] if c["n_pred"] else None
        r = c["tp_gold"] / c["n_gold"] if c["n_gold"] else None
        f = 2 * p * r / (p + r) if p is not None and r is not None and (p + r) else (0.0 if p is not None and r is not None else None)
        return {"precision": p, "recall": r, "f1": f, **c}

    def report(self) -> dict:
        per = {t: self.prf(self.c[t]) for t in TYPES if self.c[t]["n_pred"] or self.c[t]["n_gold"]}
        tot = {k: sum(self.c[t][k] for t in TYPES) for k in ("tp_pred", "n_pred", "tp_gold", "n_gold")}
        return {"per_type": per, "micro": self.prf(tot)}


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
def eval_datasets(limit, learned) -> tuple[dict, dict]:
    out, hardneg = {}, {}
    for key, title, path in DATASETS:
        if not path.exists():
            log(f"- {title}: missing ({path.relative_to(ROOT)}), skipped")
            out[key] = {"title": title, "skipped": f"{path.relative_to(ROOT)} not found"}
            continue
        docs = load_jsonl(path, limit)
        out[key] = {"title": title, "docs": len(docs), "gold_spans": sum(len(d["spans"]) for d in docs), "configs": {}}
        for cfg in CONFIGS:
            sc = Score()
            neg_hit, neg_total, neg_kind = 0, 0, defaultdict(lambda: [0, 0])
            t0 = time.perf_counter()
            for d in docs:
                preds = predict(d["text"], cfg, learned)
                sc.add(preds, d["spans"])
                for n in d.get("negatives", []):
                    hit = any(_ov(n, p) for p in preds)
                    neg_total += 1
                    neg_hit += hit
                    neg_kind[n["kind"]][0] += hit
                    neg_kind[n["kind"]][1] += 1
            secs = (time.perf_counter() - t0) / max(1, len(docs))
            rep = sc.report()
            rep["sec_per_doc"] = secs
            out[key]["configs"][cfg] = rep
            m = rep["micro"]
            log(f"- {title} [{cfg}] P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} ({secs:.3f}s/doc)")
            if neg_total:
                hardneg[cfg] = {"false_positive_rate": neg_hit / neg_total, "flagged": neg_hit, "total": neg_total,
                                "by_kind": {k: {"flagged": v[0], "total": v[1]} for k, v in sorted(neg_kind.items())}}
    return out, hardneg


def load_handmade() -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted((HERE / "handmade").glob("*.json"))]


def gold_mentions(doc: dict) -> list[dict]:
    items = [(m, i) for i, p in enumerate(doc["people"]) for m in p["mentions"]]
    items.sort(key=lambda x: -len(x[0]))
    taken, out = [], []
    for m, pid in items:
        for mm in re.finditer(rf"(?<!\w){re.escape(m)}(?!\w)", doc["text"]):
            s, e = mm.span()
            if any(s < te and ts < e for ts, te in taken):
                continue
            taken.append((s, e))
            out.append({"start": s, "end": e, "person": pid, "text": m})
    return sorted(out, key=lambda x: x["start"])


def eval_linking(docs: list[dict]) -> dict:
    total = correct = mentions = detected = 0
    per_doc = []
    for d in docs:
        if not d["people"]:
            continue
        spans, _ = link(d["text"], predict(d["text"], "hybrid"))
        names = [s for s in spans if s["type"] == "NAME"]
        gm = gold_mentions(d)
        for g in gm:
            hit = next((s for s in names if _ov(s, g)), None)
            g["pred"] = hit["entity_id"] if hit else None
        dc = dt = 0
        for i in range(len(gm)):
            for j in range(i + 1, len(gm)):
                a, b = gm[i], gm[j]
                same_gold = a["person"] == b["person"]
                ok = a["pred"] is not None and b["pred"] is not None and (a["pred"] == b["pred"]) == same_gold
                dt += 1
                dc += ok
        mentions += len(gm)
        detected += sum(1 for g in gm if g["pred"])
        total += dt
        correct += dc
        per_doc.append({"doc": d["id"], "pairs": dt, "correct": dc, "mentions": len(gm),
                        "missed": [g["text"] for g in gm if not g["pred"]]})
    return {"pairwise_accuracy": correct / total if total else None, "pairs": total, "correct_pairs": correct,
            "mention_recall": detected / mentions if mentions else None, "mentions": mentions, "per_doc": per_doc}


def _leaked(value: str, text: str) -> bool:
    if re.search(rf"(?<!\w){re.escape(value)}(?!\w)", text, re.I):
        return True
    if value.isdigit():
        return value in re.sub(r"(?<=\d)[\s\-./]+(?=\d)", "", text)
    return False


def eval_summaries(docs: list[dict]) -> dict:
    rows, leaks, interventions, secs = [], 0, 0, []
    for d in docs:
        spans, ents = link(d["text"], predict(d["text"], "hybrid"))
        for s in spans:
            s["status"] = "approved"
        view = redact.redacted_view(d["text"], spans, ents, "pseudonym")
        t0 = time.perf_counter()
        res = audit.audited_summary(view["redacted_text"], view["removed"], ents, "pseudonym",
                                    summarizer=summarize.summarize)
        secs.append(time.perf_counter() - t0)
        leaked = [v for v in d["must_not_leak"] if _leaked(v, res["summary"])]
        survived = [v for v in d["must_not_leak"] if _leaked(v, view["redacted_text"])]
        a = res["audit"]
        acted = a["initial_hits"] > 0
        interventions += acted
        leaks += len(leaked)
        qi = audit.qi_report(view["redacted_text"])
        traps = []
        for trap in d["qi_traps"]:
            pos = view["redacted_text"].find(trap)
            sent = next((s for s in qi["sentences"] if pos >= 0 and s["start"] <= pos < s["end"]), None)
            traps.append({"trap": trap, "level": sent["level"] if sent else "none",
                          "flagged": bool(sent and sent["level"] in ("medium", "high")),
                          "explanation": sent["explanation"] if sent else None})
        rows.append({"doc": d["id"], "method": res["method"], "leaks": len(leaked), "leaked_kinds": len(leaked),
                     "survived_redaction": len(survived), "audit_initial_hits": a["initial_hits"],
                     "regenerated": a["regenerated"], "replaced": len(a["replaced"]), "checks": a["checks"],
                     "seconds": round(secs[-1], 2), "qi_traps": traps})
        log(f"  {d['id']}: method={res['method']} leaks={len(leaked)} audit_hits={a['initial_hits']} "
            f"regen={a['regenerated']} replaced={len(a['replaced'])} {secs[-1]:.1f}s")
    traps = [t for r in rows for t in r["qi_traps"]]
    return {"summary_leaks": leaks, "docs": len(rows), "auditor_interventions": interventions,
            "avg_summary_seconds": sum(secs) / len(secs) if secs else None,
            "methods": sorted({r["method"] for r in rows}),
            "qi_traps_flagged": sum(t["flagged"] for t in traps), "qi_traps_total": len(traps), "per_doc": rows}


AL_RUNS = [  # (dataset key, train docs, held-out test docs)
    ("indian", 20, 30),
    ("gretel", 20, 100),
    ("ai4privacy", 20, 100),
]


def _rule_examples(summary: dict, kind: str, n: int = 4) -> list[str]:
    return [f"{r['type']}: {r['text']}" for r in summary["rules"] if r["kind"] == kind][:n]


def eval_active_learning_on(key: str, n_train: int, n_test: int) -> dict:
    """Simulated reviewer = domain adaptation. On the first `n_train` docs of a dataset
    every hybrid candidate is approved or rejected using the gold labels (as a careful
    human would); then precision/recall/F1 are measured on the next `n_test` held-out
    docs of the same dataset, before vs after learning."""
    title, path = next((t, p) for k, t, p in DATASETS if k == key)
    docs = load_jsonl(path, n_train + n_test)
    train, test = docs[:n_train], docs[n_train:n_train + n_test]
    original = learn.STORE_PATH
    learn.STORE_PATH = HERE / "cache" / f"active_learning_{key}.json"   # never touches data/learned.json
    try:
        learn.reset()
        approvals = rejections = 0
        rejected_types: dict[str, int] = defaultdict(int)
        for d in train:
            state = learn.load()
            events = []
            for p in predict(d["text"], "hybrid", state):
                # A human approves anything that is real PII, even if the type label is off
                # (it must be redacted either way), and rejects only candidates that are not PII.
                good = any(_ov(g, p) for g in d["spans"])
                events.append({"action": "approve" if good else "reject", "text": p["text"], "type": p["type"],
                               "source": p["source"]})
                approvals += good
                rejections += not good
                if not good:
                    rejected_types[_type(p["type"])] += 1
            learn.record(events)
        learned_state = learn.load()
        summary = learn.summary(learned_state)
        before, after = Score(), Score()
        for d in test:
            before.add(predict(d["text"], "hybrid", None), d["spans"])
            after.add(predict(d["text"], "hybrid", learned_state), d["spans"])
        # ablation: which learned component produced the change?
        lists_only = dict(learned_state, type_stats={})
        thr_only = dict(learn._empty(), type_stats=learned_state["type_stats"])
        abl = {}
        for name, st in (("lists_only", lists_only), ("thresholds_only", thr_only)):
            sc = Score()
            for d in test:
                sc.add(predict(d["text"], "hybrid", st), d["spans"])
            mm = sc.report()["micro"]
            abl[name] = {"precision": mm["precision"], "recall": mm["recall"], "f1": mm["f1"],
                         "fp": mm["n_pred"] - mm["tp_pred"]}
        b, a = before.report(), after.report()
        bm, am = b["micro"], a["micro"]
        res = {"dataset": key, "title": title, "train_docs": len(train), "test_docs": len(test),
               "approvals": approvals, "rejections": rejections, "rejected_by_type": dict(rejected_types),
               "rules_learned": summary["corrections"],
               "rule_kinds": {k: sum(1 for r in summary["rules"] if r["kind"] == k)
                              for k in ("allowlist", "denylist", "shape", "candidate")},
               "shape_examples": _rule_examples(summary, "shape"),
               "allowlist_examples": [r["text"].replace("\n", " ") for r in summary["rules"] if r["kind"] == "allowlist"][:8],
               "ablation": abl,
               "thresholds": summary["all_thresholds"],
               "precision_before": bm["precision"], "precision_after": am["precision"],
               "recall_before": bm["recall"], "recall_after": am["recall"],
               "f1_before": bm["f1"], "f1_after": am["f1"],
               "fp_before": bm["n_pred"] - bm["tp_pred"], "fp_after": am["n_pred"] - am["tp_pred"],
               "per_type": {t: {"before": b["per_type"].get(t), "after": a["per_type"].get(t)} for t in TYPES}}
        log(f"- {title}: {approvals} approvals / {rejections} rejections; "
            f"P {bm['precision']:.3f}->{am['precision']:.3f}  R {bm['recall']:.3f}->{am['recall']:.3f}  "
            f"F1 {bm['f1']:.3f}->{am['f1']:.3f}")
        return res
    finally:
        learn.STORE_PATH = original


def eval_active_learning() -> dict:
    runs = {}
    for key, n_train, n_test in AL_RUNS:
        path = next(p for k, _, p in DATASETS if k == key)
        if path.exists():
            runs[key] = eval_active_learning_on(key, n_train, n_test)
    # "best" = the largest honest precision gain, ties broken by F1 gain
    best = max(runs, key=lambda k: (round(runs[k]["precision_after"] - runs[k]["precision_before"], 4),
                                    runs[k]["f1_after"] - runs[k]["f1_before"])) if runs else None
    return {"runs": runs, "best": best,
            "protocol": "simulated reviewer approves/rejects every hybrid candidate on the first 20 docs of a "
                        "dataset using gold labels; metrics on held-out docs from the same dataset"}


def peak_ram_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def pct(x) -> str:
    return "–" if x is None else f"{100 * x:.1f}"


def write_markdown(res: dict) -> None:
    L = ["# Evaluation results", "",
         f"Generated {res['generated']} on {res['machine']} (Python {res['python']}). "
         f"Learning: **{'applied (data/learned.json)' if res['with_learning'] else 'disabled (clean detector)'}**. "
         f"Summary model: `{res['summary_model']}`.", "",
         "Span-level scores use overlap matching: a predicted span is correct if it overlaps a gold span of the "
         "same type; a gold span is found if any same-type prediction overlaps it. `ORG_ADDRESS` predictions "
         "count as `ADDRESS`. Label mapping: [label_mapping.md](label_mapping.md).", "",
         "![F1 by dataset and configuration](results.png)", "", "## Overall (micro-averaged)", "",
         "| Dataset | Docs | Config | Precision | Recall | F1 | s/doc |", "|---|---:|---|---:|---:|---:|---:|"]
    for key, d in res["datasets"].items():
        if d.get("skipped"):
            L.append(f"| {d['title']} | – | skipped: {d['skipped']} | | | | |")
            continue
        for cfg, r in d["configs"].items():
            m = r["micro"]
            L.append(f"| {d['title']} | {d['docs']} | {cfg} | {pct(m['precision'])} | {pct(m['recall'])} | "
                     f"{pct(m['f1'])} | {r['sec_per_doc']:.3f} |")
    for key, d in res["datasets"].items():
        if d.get("skipped"):
            continue
        L += ["", f"## {d['title']}: per type (P / R / F1, %)", "", "| Type | Gold | " +
              " | ".join(CONFIGS) + " |", "|---|---:|" + "---|" * len(CONFIGS)]
        for t in TYPES:
            gold = d["configs"]["hybrid"]["per_type"].get(t, {}).get("n_gold", 0)
            cells = []
            for cfg in CONFIGS:
                r = d["configs"][cfg]["per_type"].get(t)
                cells.append("–" if not r else f"{pct(r['precision'])} / {pct(r['recall'])} / {pct(r['f1'])}")
            L.append(f"| {t} | {gold} | " + " | ".join(cells) + " |")
    hn = res["hard_negatives"]
    if hn:
        L += ["", "## Hard negatives (Indian set)", "",
              "Invalid-checksum 12-digit numbers, SKUs, dates, invoice numbers, ISBNs, amounts and version "
              "numbers. False-positive rate = share of these that any predicted span touches (lower is better).", "",
              "| Config | FP rate | Flagged / total | " + " | ".join(next(iter(hn.values()))["by_kind"]) + " |",
              "|---|---:|---:|" + "---:|" * len(next(iter(hn.values()))["by_kind"])]
        for cfg, h in hn.items():
            L.append(f"| {cfg} | {pct(h['false_positive_rate'])}% | {h['flagged']} / {h['total']} | " +
                     " | ".join(f"{v['flagged']}/{v['total']}" for v in h["by_kind"].values()) + " |")
    lk = res["linking"]
    L += ["", "## Name-variant linking (hand-made set, hybrid)", "",
          f"Pairwise accuracy: **{pct(lk['pairwise_accuracy'])}%** ({lk['correct_pairs']}/{lk['pairs']} mention pairs "
          f"correctly placed in the same / different person). Mention recall: {pct(lk['mention_recall'])}% "
          f"({lk['mentions']} gold mentions).", "", "| Doc | Correct pairs | Missed mentions |", "|---|---:|---|"]
    for r in lk["per_doc"]:
        L.append(f"| {r['doc']} | {r['correct']}/{r['pairs']} | {', '.join(r['missed']) or '–'} |")
    sm = res["summaries"]
    L += ["", "## Summary leak traps & quasi-identifier traps (hand-made set)", "",
          f"Summary leaks found: **{sm['summary_leaks']}** (target 0) across {sm['docs']} documents "
          f"(method: {', '.join(sm['methods'])}). The leak auditor intervened on {sm['auditor_interventions']} "
          f"document(s). Average audited summary time: {sm['avg_summary_seconds']:.1f}s.", "",
          f"Quasi-identifier traps flagged (medium/high): **{sm['qi_traps_flagged']}/{sm['qi_traps_total']}**.", "",
          "| Doc | Method | Leaks | Audit hits (1st draft) | Regenerated | Replaced | QI traps |",
          "|---|---|---:|---:|---|---:|---|"]
    for r in sm["per_doc"]:
        traps = "; ".join(f"{t['level']}" for t in r["qi_traps"]) or "–"
        L.append(f"| {r['doc']} | {r['method']} | {r['leaks']} | {r['audit_initial_hits']} | "
                 f"{'yes' if r['regenerated'] else 'no'} | {r['replaced']} | {traps} |")
    al = res["active_learning"]
    L += ["", "## Active learning as domain adaptation (simulated reviewer)", "",
          "Protocol: on the first 20 documents of a dataset, a simulated reviewer approves or rejects every "
          "hybrid candidate using the gold labels, as a careful human would in the UI: a candidate that overlaps "
          "any gold PII span is approved (even if its type label is off, since it must be redacted either way), "
          "and anything else is rejected. "
          "The learned rules (allowlist, denylist, non-PII shapes, candidate rules, per-type GLiNER thresholds) "
          "are then applied to **held-out documents from the same dataset** and compared with the clean detector. "
          "No decoys were added to any dataset.", "",
          "| Dataset | Train / test docs | Reviewer feedback | Rules learned | Precision | Recall | F1 | False positives |",
          "|---|---:|---|---:|---|---|---|---|"]
    for k, r in al["runs"].items():
        L.append(f"| {r['title']} | {r['train_docs']} / {r['test_docs']} | {r['approvals']} approved, "
                 f"{r['rejections']} rejected | {r['rules_learned']} | {pct(r['precision_before'])} → "
                 f"{pct(r['precision_after'])} | {pct(r['recall_before'])} → {pct(r['recall_after'])} | "
                 f"{pct(r['f1_before'])} → {pct(r['f1_after'])} | {r['fp_before']} → {r['fp_after']} |")
    L += ["", "What was learned and why the results differ:", ""]
    for k, r in al["runs"].items():
        rej = ", ".join(f"{t} {n}" for t, n in sorted(r["rejected_by_type"].items(), key=lambda x: -x[1])) or "none"
        shapes = "; ".join(r["shape_examples"]) or "none"
        dp = (r["precision_after"] or 0) - (r["precision_before"] or 0)
        dr = (r["recall_after"] or 0) - (r["recall_before"] or 0)
        if r["rejections"] <= 2:
            why = (f"There was little to learn: the clean detector made only {r['rejections']} false-positive "
                   f"candidate(s) in {r['train_docs']} documents, so the reviewer had almost nothing to reject. "
                   f"Precision was already {pct(r['precision_before'])}%; the small recall change comes from "
                   f"approved strings (denylist) and lower GLiNER thresholds.")
        else:
            ab = r["ablation"]
            lp = 100 * ((ab["lists_only"]["precision"] or 0) - (r["precision_before"] or 0))
            tp = 100 * ((ab["thresholds_only"]["precision"] or 0) - (r["precision_before"] or 0))
            tr = 100 * ((ab["thresholds_only"]["recall"] or 0) - (r["recall_before"] or 0))
            driver = "the allow/deny lists" if abs(lp) >= abs(tp) else "the per-type GLiNER thresholds"
            ex = ", ".join(f"“{x}”" for x in r["allowlist_examples"][:6]) or "none"
            why = (f"The reviewer rejected {r['rejections']} false positives (by type: {rej}); learned non-PII "
                   f"shapes: {shapes} (rejected values were mostly words, and only digit-bearing codes become "
                   f"shapes). On {r['test_docs']} unseen documents precision moved {100 * dp:+.1f} points and "
                   f"recall {100 * dr:+.1f}. Ablation: lists alone {lp:+.1f} precision points, thresholds alone "
                   f"{tp:+.1f} precision / {tr:+.1f} recall points, so the change is driven mainly by {driver}. "
                   f"Rejected strings include {ex}: mostly role and domain words the model mistakes for "
                   f"names, which then stop being flagged in new documents of the same kind.")
        L.append(f"- **{r['title']}**: {why} Thresholds after learning: {r['thresholds']}.")
    if al.get("best"):
        b = al["runs"][al["best"]]
        L += ["", f"The review UI's “What I learned” drawer shows the largest honest precision change: "
                  f"**{b['title']}, {pct(b['precision_before'])}% → {pct(b['precision_after'])}%**.", ""]
    L += [
          "## Resources", "", f"- Peak RAM (whole run, includes the GLiNER model): **{res['peak_ram_mb']:.0f} MB**",
          f"- Total runtime: {res['runtime_seconds'] / 60:.1f} min", ""]
    (HERE / "results.md").write_text("\n".join(L), encoding="utf-8")


def write_chart(res: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def mix(t):  # tint of purple toward almond
        p, a = (0x32, 0x0B, 0x35), (0xF2, 0xEC, 0xC0)
        return "#" + "".join(f"{round(p[i] * t + a[i] * (1 - t)):02x}" for i in range(3))

    keys = [k for k, d in res["datasets"].items() if not d.get("skipped")]
    titles = [res["datasets"][k]["title"].replace(" (Faker en_IN)", "\n(Faker en_IN)").replace(" pii-masking-300k", "\npii-masking-300k")
              .replace(" synthetic PII finance", "\nPII finance") for k in keys]
    styles = {"regex": dict(color=ALMOND, edgecolor=PURPLE, linewidth=1.2),
              "gliner": dict(color=mix(0.45), edgecolor=PURPLE, linewidth=1.2),
              "hybrid": dict(color=PURPLE, edgecolor=PURPLE, linewidth=1.2)}
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=160)
    bg = mix(0.0)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    w = 0.26
    for i, cfg in enumerate(CONFIGS):
        vals = [res["datasets"][k]["configs"][cfg]["micro"]["f1"] or 0 for k in keys]
        xs = [x + (i - 1) * w for x in range(len(keys))]
        bars = ax.bar(xs, vals, w * 0.92, label={"regex": "Regex only", "gliner": "GLiNER only", "hybrid": "Hybrid"}[cfg],
                      **styles[cfg])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=PURPLE)
    ax.set_xticks(range(len(keys)), titles, color=PURPLE, fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Span F1 (micro, overlap match)", color=PURPLE)
    ax.set_title("PII detection F1 by dataset", color=PURPLE, fontsize=13, fontstyle="italic", loc="left")
    ax.tick_params(colors=PURPLE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(PURPLE)
    ax.yaxis.grid(True, color=mix(0.15), linewidth=0.8)
    ax.set_axisbelow(True)
    leg = ax.legend(frameon=False, loc="upper right", ncol=3)
    for t in leg.get_texts():
        t.set_color(PURPLE)
    fig.tight_layout()
    fig.savefig(HERE / "results.png", facecolor=bg)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="max docs per dataset")
    ap.add_argument("--with-learning", action="store_true", help="apply data/learned.json (default: clean detector)")
    ap.add_argument("--skip-summaries", action="store_true")
    ap.add_argument("--only", choices=["active-learning"], help="re-run one section, reuse eval/results.json for the rest")
    args = ap.parse_args()
    t_start = time.perf_counter()
    learned = learn.load() if args.with_learning else None
    if args.only == "active-learning":
        res = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
        det.load_model()
        log("# Active learning (domain adaptation)")
        res["active_learning"] = eval_active_learning()
        res["peak_ram_mb"] = max(res.get("peak_ram_mb", 0), peak_ram_mb())
        res["active_learning_generated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        (HERE / "results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        write_markdown(res)
        write_chart(res)
        log("updated eval/results.json, eval/results.md")
        return

    log("Loading GLiNER…")
    det.load_model()
    log(f"detector: {det.detector_status()['mode']}")

    log("\n# Datasets")
    datasets, hardneg = eval_datasets(args.limit, learned)
    log("\n# Linking")
    linking = eval_linking(load_handmade())
    log(f"pairwise accuracy {linking['pairwise_accuracy']:.3f}, mention recall {linking['mention_recall']:.3f}")
    summaries = {"summary_leaks": None, "docs": 0, "auditor_interventions": 0, "avg_summary_seconds": 0.0,
                 "methods": [], "qi_traps_flagged": 0, "qi_traps_total": 0, "per_doc": []}
    if not args.skip_summaries:
        log("\n# Summaries (leak + QI traps)")
        summaries = eval_summaries(load_handmade())
    log("\n# Active learning (domain adaptation)")
    al = eval_active_learning()

    res = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "machine": f"{platform.machine()} {platform.system()} {platform.release()}",
           "python": platform.python_version(), "with_learning": args.with_learning, "limit": args.limit,
           "summary_model": SUMMARY_MODEL, "datasets": datasets, "hard_negatives": hardneg, "linking": linking,
           "summaries": summaries, "active_learning": al, "peak_ram_mb": peak_ram_mb(),
           "runtime_seconds": time.perf_counter() - t_start}
    (HERE / "results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    write_markdown(res)
    write_chart(res)
    log(f"\nwrote eval/results.json, eval/results.md, eval/results.png  (peak RAM {res['peak_ram_mb']:.0f} MB, "
        f"{res['runtime_seconds'] / 60:.1f} min)")


if __name__ == "__main__":
    main()
