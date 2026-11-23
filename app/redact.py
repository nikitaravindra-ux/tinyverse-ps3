"""Redaction, export (TXT / rebuilt PDF) and the integrity verifier.

The PDF is rebuilt from the redacted *text* with reportlab: nothing from the source
PDF is copied, so there is no hidden text layer under a black box to recover. The
document info dictionary is removed and no XMP stream is written.
"""
from __future__ import annotations

import hashlib
import io
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.detect import normalize_email

FONT_PATH = Path(__file__).resolve().parent / "fonts" / "DejaVuSans.ttf"
FONT_NAME = "DejaVuSans"
MODES = ("pseudonym", "mask")
FORMATS = ("txt", "pdf")

# Role mailboxes are too generic to verify as tokens ("hr" also appears in "HR/OFR/…").
GENERIC_USERNAMES = {"hr", "info", "admin", "support", "contact", "sales", "accounts", "office",
                     "mail", "help", "billing", "careers", "jobs", "noreply", "no-reply", "team"}
MIN_USERNAME_LEN = 3


# ---------------------------------------------------------------------------
# Building the redacted text
# ---------------------------------------------------------------------------
def _variant_pattern(variant: str) -> re.Pattern:
    toks = variant.split()
    body = r"\s+".join(re.escape(t) for t in toks)
    return re.compile(rf"(?<!\w){body}(?!\w)", re.I)


def _replacement(span: dict, entity: Optional[dict], mode: str) -> str:
    if mode == "pseudonym" and entity:
        return f"[{entity['pseudonym']}]"
    return f"[{span['type']}]"


def plan_redaction(text: str, spans: list[dict], entities: list[dict]) -> dict:
    """Decide which character ranges are removed.

    Approved spans are removed, and so is every occurrence (case-insensitive, any
    whitespace) of every variant of an entity that has at least one approved mention.
    Linked mentions the reviewer rejected are overridden for consistency and counted.
    """
    ents = {e["id"]: e for e in entities}
    by_id = {s["id"]: s for s in spans if "id" in s}
    approved = [s for s in spans if s.get("status") == "approved"]
    approved_entities = {s.get("entity_id") for s in approved if s.get("entity_id") in ents}

    regions: list[dict] = []
    for s in approved:
        regions.append({"start": s["start"], "end": s["end"], "type": s["type"],
                        "entity_id": s.get("entity_id"), "origin": "approved"})
    for eid in approved_entities:
        e = ents[eid]
        variants = set(e.get("variants") or [])
        variants |= {by_id[i]["text"] for i in e.get("span_ids", []) if i in by_id}
        for v in sorted(variants, key=len, reverse=True):
            if not v.strip():
                continue
            for m in _variant_pattern(v).finditer(text):
                regions.append({"start": m.start(), "end": m.end(), "type": e["type"],
                                "entity_id": eid, "origin": "variant"})

    # resolve overlaps: longest first, then earliest
    regions.sort(key=lambda r: (-(r["end"] - r["start"]), r["start"]))
    chosen: list[dict] = []
    for r in regions:
        if any(r["start"] < c["end"] and r["end"] > c["start"] for c in chosen):
            continue
        chosen.append(r)
    chosen.sort(key=lambda r: r["start"])

    overridden = [s for s in spans if s.get("status") == "rejected"
                  and any(s["start"] < c["end"] and s["end"] > c["start"] for c in chosen)]
    pending = [s for s in spans if s.get("status", "pending") == "pending"
               and not any(s["start"] < c["end"] and s["end"] > c["start"] for c in chosen)]
    return {"regions": chosen, "overridden_rejections": len(overridden), "pending_left": len(pending)}


def apply_redaction(text: str, regions: list[dict], entities: list[dict], mode: str) -> str:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    ents = {e["id"]: e for e in entities}
    out, pos = [], 0
    for r in regions:
        out.append(text[pos:r["start"]])
        out.append(_replacement(r, ents.get(r.get("entity_id")), mode))
        pos = r["end"]
    out.append(text[pos:])
    return "".join(out)


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
def build_txt(redacted: str) -> bytes:
    return redacted.encode("utf-8")


def _register_font() -> str:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    if FONT_NAME not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(FONT_NAME, str(FONT_PATH)))
    return FONT_NAME


def _wrap(line: str, font: str, size: float, width: float) -> list[str]:
    from reportlab.lib.utils import simpleSplit
    from reportlab.pdfbase.pdfmetrics import stringWidth

    if stringWidth(line, font, size) <= width:
        return [line]
    indent = len(line) - len(line.lstrip(" "))
    parts = simpleSplit(line.strip(), font, size, width - stringWidth(" " * indent, font, size))
    return [" " * indent + p for p in parts] or [""]


def build_pdf(redacted: str) -> bytes:
    """Draw the redacted text onto fresh pages, then strip the info dict (no XMP)."""
    from pypdf import PdfReader, PdfWriter
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas

    font = _register_font()
    size, leading = 10, 14
    width, height = A4
    left, right, top, bottom = 20 * mm, 20 * mm, 20 * mm, 22 * mm
    usable = width - left - right

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4, invariant=1, pageCompression=1)
    page = 1

    def footer():
        c.setFont(font, 7.5)
        c.drawRightString(width - right, 12 * mm, f"Redacted copy · page {page}")

    y = height - top
    for raw in redacted.replace("\t", "    ").split("\n"):
        for line in _wrap(raw, font, size, usable):
            if y < bottom:
                footer()
                c.showPage()
                page += 1
                y = height - top
            c.setFont(font, size)
            c.drawString(left, y, line)
            y -= leading
    footer()
    c.save()

    # Re-write without the /Info dictionary (reportlab always writes Producer etc.)
    reader = PdfReader(io.BytesIO(buf.getvalue()))
    writer = PdfWriter()
    for p in reader.pages:
        writer.add_page(p)
    writer.metadata = None
    if "/Metadata" in writer._root_object:
        del writer._root_object["/Metadata"]
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def pdf_metadata(data: bytes) -> dict:
    """Metadata left in a PDF: the info dict entries and whether an XMP stream exists."""
    from pypdf import PdfReader

    r = PdfReader(io.BytesIO(data))
    info = {k: str(v) for k, v in (r.metadata or {}).items()}
    return {"info": info, "xmp": "/Metadata" in r.trailer["/Root"]}


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------
def extract_exported_text(data: bytes, fmt: str) -> str:
    if fmt == "pdf":
        from pypdf import PdfReader

        return "\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(data)).pages)
    return data.decode("utf-8")


def _hint(value: str) -> str:
    """Never echo PII into the receipt: keep the first char and the length."""
    v = value.strip()
    return f"{v[:1]}{'•' * max(0, min(len(v), 12) - 1)} ({len(v)} chars)"


def build_checks(text: str, regions: list[dict], entities: list[dict]) -> tuple[list[dict], list[dict]]:
    checks: dict[tuple[str, str], dict] = {}
    skipped: list[dict] = []
    ents = {e["id"]: e for e in entities}

    def add(kind, typ, value):
        value = value.strip()
        if value:
            checks.setdefault((kind, value.lower()), {"kind": kind, "type": typ, "value": value})

    removed_entities = set()
    for r in regions:
        s = text[r["start"]:r["end"]]
        add("removed_string", r["type"], s)
        if r.get("entity_id"):
            removed_entities.add(r["entity_id"])
        if r["type"] in ("PHONE", "ID"):
            for run in re.findall(r"\d{4,}", s):
                add("digit_run", r["type"], run)
            joined = re.sub(r"\D", "", s)
            if len(joined) >= 4:
                add("digit_run", r["type"], joined)
        if r["type"] == "EMAIL":
            email = normalize_email(s) or s
            user = email.split("@", 1)[0]
            if len(user) < MIN_USERNAME_LEN or user.lower() in GENERIC_USERNAMES:
                skipped.append({"kind": "email_username", "type": "EMAIL", "hint": _hint(user),
                                "reason": "generic or too short to verify as a token"})
            else:
                add("email_username", "EMAIL", user)
    for eid in removed_entities:
        e = ents.get(eid)
        if e and e["type"] == "NAME":
            for v in e.get("variants", []):
                add("name_variant", "NAME", v)
    return list(checks.values()), skipped


def _found(check: dict, out_text: str, out_joined_digits: str) -> bool:
    v = check["value"]
    if check["kind"] == "digit_run":
        pat = rf"(?<!\d){re.escape(v)}(?!\d)"
        return bool(re.search(pat, out_text) or re.search(pat, out_joined_digits))
    toks = v.split()
    body = r"\s+".join(re.escape(t) for t in toks)
    return bool(re.search(rf"(?<!\w){body}(?!\w)", out_text, re.I))


def verify(data: bytes, fmt: str, checks: list[dict]) -> list[dict]:
    out = extract_exported_text(data, fmt)
    joined = re.sub(r"(?<=\d)[\s\-./]+(?=\d)", "", out)
    return [c for c in checks if _found(c, out, joined)]


# ---------------------------------------------------------------------------
# One-shot export
# ---------------------------------------------------------------------------
def export(text: str, spans: list[dict], entities: list[dict], mode: str = "pseudonym",
           fmt: str = "pdf", source_name: str = "document") -> dict:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if fmt not in FORMATS:
        raise ValueError(f"format must be one of {FORMATS}")
    plan = plan_redaction(text, spans, entities)
    regions = plan["regions"]
    redacted = apply_redaction(text, regions, entities, mode)
    data = build_pdf(redacted) if fmt == "pdf" else build_txt(redacted)
    sha = hashlib.sha256(data).hexdigest()

    checks, skipped = build_checks(text, regions, entities)
    failures = verify(data, fmt, checks)

    per_type: dict[str, int] = {}
    for r in regions:
        per_type[r["type"]] = per_type.get(r["type"], 0) + 1
    meta = pdf_metadata(data) if fmt == "pdf" else None

    receipt = {
        "title": "Redaction Receipt",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": mode,
        "format": fmt,
        "file": {"name": f"redacted-{sha[:10]}.{fmt}", "sha256": sha, "bytes": len(data)},
        "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "removed": {"total": len(regions), "by_type": per_type,
                    "entities": len({r.get('entity_id') for r in regions if r.get('entity_id')})},
        "verification": {
            "method": "re-opened the exported file, extracted its text"
                      + (" with pypdf" if fmt == "pdf" else "")
                      + ", searched case-insensitively for removed strings, name variants, "
                        "4+ digit runs of phones/IDs and email usernames",
            "checks": len(checks),
            "recoverable": len(failures),
            "failures": [{"kind": f["kind"], "type": f["type"], "hint": _hint(f["value"])} for f in failures],
            "skipped": skipped,
            "passed": not failures,
        },
        "review": {"overridden_rejections": plan["overridden_rejections"],
                   "pending_left_unredacted": plan["pending_left"]},
    }
    if meta is not None:
        receipt["pdf_metadata"] = {"info_entries": len(meta["info"]), "xmp": meta["xmp"]}
    return {"receipt": receipt, "data": data, "redacted_text": redacted,
            "mime": "application/pdf" if fmt == "pdf" else "text/plain; charset=utf-8"}
