"""Download + sample the public PII datasets into eval/cache/ (git-ignored).

This is the ONLY part of the project that needs the internet, so it deliberately does
not import `app` (no network guard, no Hugging Face offline mode).

    .venv/bin/python eval/load_datasets.py

Writes eval/cache/<name>.jsonl rows {id, text, spans:[{start, end, type}]} and
eval/label_mapping.md describing exactly which source labels map to which type.
"""
from __future__ import annotations

import ast
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(CACHE / "hf"))

from datasets import load_dataset  # noqa: E402  (after HF_HOME)

SEED = 42
TYPES = ("NAME", "EMAIL", "PHONE", "ADDRESS", "ID")
PART = "ADDRESS_PART"   # city/state/country: kept only when merged into an adjacent address

# ---------------------------------------------------------------------------
# Label mappings (anything not listed is dropped)
# ---------------------------------------------------------------------------
AI4PRIVACY = {
    "NAME": ["GIVENNAME*", "LASTNAME*", "SURNAME*", "FIRSTNAME*", "MIDDLENAME*"],
    "EMAIL": ["EMAIL"],
    "PHONE": ["TEL", "TELEPHONENUM", "PHONENUMBER"],
    "ADDRESS": ["STREET", "BUILDING", "BUILDINGNUM", "SECADDRESS", "POSTCODE", "ZIPCODE"],
    PART: ["CITY", "STATE", "COUNTY"],
    "ID": ["IDCARD", "IDCARDNUM", "SOCIALNUMBER", "SOCIALNUM", "PASSPORT", "PASSPORTNUM", "DRIVERLICENSE",
           "DRIVERLICENSENUM", "TAXNUM", "CREDITCARDNUMBER", "ACCOUNTNUM"],
}
GRETEL = {
    "NAME": ["name", "first_name", "last_name", "customer_name", "employee_name"],
    "EMAIL": ["email"],
    "PHONE": ["phone_number", "fax_number"],
    "ADDRESS": ["street_address", "address"],
    PART: ["city", "state", "postcode", "zipcode"],
    "ID": ["account_number", "bban", "iban", "credit_card_number", "credit_card_security_code_excluded",
           "ssn", "customer_id", "employee_id", "passport_number", "driver_license_number", "tax_id",
           "national_id", "swift_bic_code", "routing_number", "account_pin_excluded"],
}
NEMOTRON = {
    "NAME": ["first_name", "last_name", "name", "middle_name"],
    "EMAIL": ["email"],
    "PHONE": ["phone_number", "fax_number"],
    "ADDRESS": ["street_address"],
    PART: ["city", "state", "postcode", "county"],
    "ID": ["ssn", "account_number", "credit_debit_card", "customer_id", "employee_id", "national_id",
           "passport_number", "tax_id", "bank_routing_number", "swift_bic", "certificate_license_number",
           "medical_record_number", "health_plan_beneficiary_number", "unique_id", "license_plate",
           "vehicle_identifier", "driver_license_number"],
}


def _mapper(table: dict):
    exact, prefix = {}, []
    for typ, labels in table.items():
        for lbl in labels:
            if lbl.endswith("*"):
                prefix.append((lbl[:-1].lower(), typ))
            elif not lbl.endswith("_excluded"):
                exact[lbl.lower()] = typ

    def m(label: str):
        lab = label.lower()
        if lab in exact:
            return exact[lab]
        for p, typ in prefix:
            if re.fullmatch(re.escape(p) + r"\d*", lab):
                return typ
        return None
    return m


def merge_spans(text: str, spans: list[dict]) -> list[dict]:
    """Join adjacent name parts / address parts (gap of punctuation+space only), drop
    city/state parts that are not attached to an address."""
    spans = sorted(spans, key=lambda s: s["start"])
    out: list[dict] = []
    for sp in spans:
        if out:
            prev = out[-1]
            gap = text[prev["end"]:sp["start"]]
            joinable = len(gap) <= 3 and re.fullmatch(r"[\s,.\-]*", gap or "")
            same = (prev["type"] == sp["type"] == "NAME") or \
                   ({prev["type"], sp["type"]} <= {"ADDRESS", PART} and "ADDRESS" in {prev["type"], sp["type"]}
                    or prev["type"] == sp["type"] == "ADDRESS")
            if joinable and same:
                prev["end"] = max(prev["end"], sp["end"])
                prev["type"] = "ADDRESS" if "ADDRESS" in (prev["type"], sp["type"]) else prev["type"]
                continue
            if joinable and prev["type"] == PART and sp["type"] == PART:
                prev["end"] = sp["end"]
                continue
        out.append(dict(sp))
    return [s for s in out if s["type"] in TYPES and s["end"] > s["start"]]


def convert(text: str, raw_spans, mapper, counts: Counter, dropped: Counter) -> list[dict]:
    spans = []
    for s in raw_spans:
        label = s["label"]
        typ = mapper(label)
        if typ is None:
            dropped[label] += 1
            continue
        start, end = int(s["start"]), int(s["end"])
        if not (0 <= start < end <= len(text)) or not text[start:end].strip():
            dropped[f"{label} (bad offsets)"] += 1
            continue
        counts[f"{label} -> {typ}"] += 1
        spans.append({"start": start, "end": end, "type": typ})
    return merge_spans(text, spans)


def sample(ds_iter, n: int, keep) -> list[dict]:
    rows = []
    for row in ds_iter:
        if keep(row):
            rows.append(row)
            if len(rows) >= n:
                break
    return rows


def write(name: str, rows: list[dict]) -> None:
    path = CACHE / f"{name}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {len(rows)} docs -> {path.relative_to(HERE.parent)}")


def main() -> None:
    report = {}

    # ---- ai4privacy ------------------------------------------------------
    print("ai4privacy/pii-masking-300k (validation, English, 500)")
    counts, dropped = Counter(), Counter()
    ds = load_dataset("ai4privacy/pii-masking-300k", split="validation", streaming=True)
    ds = ds.shuffle(seed=SEED, buffer_size=10_000)
    rows = sample(ds, 500, lambda r: r.get("language") == "English" and r.get("set", "validation") == "validation")
    m = _mapper(AI4PRIVACY)
    out = [{"id": r["id"], "text": r["source_text"],
            "spans": convert(r["source_text"], r["privacy_mask"], m, counts, dropped)} for r in rows]
    write("ai4privacy", out)
    report["ai4privacy/pii-masking-300k"] = (AI4PRIVACY, counts, dropped, len(out), "validation split, language == English, shuffled (seed 42), first 500")

    # ---- gretel ----------------------------------------------------------
    print("gretelai/synthetic_pii_finance_multilingual (test, English, 300)")
    counts, dropped = Counter(), Counter()
    ds = load_dataset("gretelai/synthetic_pii_finance_multilingual", split="test", streaming=True)
    ds = ds.shuffle(seed=SEED, buffer_size=10_000)
    rows = sample(ds, 300, lambda r: r.get("language") == "English")
    m = _mapper(GRETEL)
    out = []
    for r in rows:
        raw = json.loads(r["pii_spans"]) if isinstance(r["pii_spans"], str) else r["pii_spans"]
        out.append({"id": str(r.get("index")), "text": r["generated_text"],
                    "spans": convert(r["generated_text"], raw, m, counts, dropped)})
    write("gretel", out)
    report["gretelai/synthetic_pii_finance_multilingual"] = (GRETEL, counts, dropped, len(out), "test split, language == English, shuffled (seed 42), first 300")

    # ---- nemotron --------------------------------------------------------
    print("nvidia/nemotron-pii (300)")
    counts, dropped = Counter(), Counter()
    try:
        try:
            ds = load_dataset("nvidia/nemotron-pii", split="test", streaming=True)
            split = "test"
        except Exception:
            ds = load_dataset("nvidia/nemotron-pii", split="train", streaming=True)
            split = "train"
        ds = ds.shuffle(seed=SEED, buffer_size=10_000)
        rows = sample(ds, 300, lambda r: True)
        m = _mapper(NEMOTRON)
        out = []
        for r in rows:
            raw = r["spans"]
            if isinstance(raw, str):
                raw = ast.literal_eval(raw)
            out.append({"id": r["uid"], "text": r["text"], "spans": convert(r["text"], raw, m, counts, dropped)})
        write("nemotron", out)
        report["nvidia/nemotron-pii"] = (NEMOTRON, counts, dropped, len(out), f"{split} split, shuffled (seed 42), first 300")
    except Exception as exc:
        msg = f"skipped: {type(exc).__name__}: {exc}"
        print("  " + msg)
        report["nvidia/nemotron-pii"] = (NEMOTRON, counts, dropped, 0, msg)

    write_mapping(report)


def write_mapping(report: dict) -> None:
    lines = ["# Label mapping used by the evaluation", "",
             "Generated by `eval/load_datasets.py`. Source labels are mapped to the tool's five types; "
             "every other label is dropped (not scored).", "",
             "- Adjacent name parts (first + last name) are merged into one `NAME` span.",
             "- Address components are merged into one `ADDRESS` span. City/state/postcode labels "
             "only count when they are attached (gap of `, ` or similar) to a street/building component. "
             "A bare city (\"users from London\") is not personal-address data for this tool, so it is dropped.",
             "- `ORG_ADDRESS` predictions are scored as `ADDRESS`.", ""]
    for name, (table, counts, dropped, n, how) in report.items():
        lines += [f"## {name}", "", f"Sample: {how}. Documents written: **{n}**.", "", "| Type | Source labels |", "|---|---|"]
        for typ in (*TYPES, PART):
            if table.get(typ):
                lbls = ", ".join(f"`{x}`" for x in table[typ] if not x.endswith("_excluded"))
                lines.append(f"| {typ if typ != PART else 'ADDRESS (only when attached)'} | {lbls} |")
        lines += ["", "Observed mapped labels (counts in the sample):", ""]
        lines += [f"- `{k}`: {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])] or ["- none"]
        lines += ["", "Dropped labels (counts in the sample):", ""]
        lines += [f"- `{k}`: {v}" for k, v in sorted(dropped.items(), key=lambda kv: -kv[1])] or ["- none"]
        lines.append("")
    (HERE / "label_mapping.md").write_text("\n".join(lines), encoding="utf-8")
    print("  wrote eval/label_mapping.md")


if __name__ == "__main__":
    sys.exit(main())
