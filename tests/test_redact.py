"""Redaction, rebuilt-PDF export and the integrity verifier."""
import base64
import io
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfReader

from app import detect as det
from app import redact
from app.detect import detect
from app.link import link
from app.main import analyze_text, app, extract_text

SAMPLES = Path(__file__).resolve().parent.parent / "samples"
SAMPLE_FILES = ["hr_offer_letter.txt", "bank_kyc_form.txt", "incident_report.txt",
                "vendor_contract.txt", "medical_reimbursement_claim.pdf"]


@pytest.fixture(autouse=True)
def rules_only(monkeypatch):
    monkeypatch.setattr(det, "_model", None)
    monkeypatch.setitem(det._model_state, "status", "not_loaded")
    monkeypatch.setitem(sys.modules, "gliner", None)


def analysed(name, status="approved"):
    path = SAMPLES / name
    doc = analyze_text(extract_text(name, path.read_bytes()), name)
    for s in doc["spans"]:
        s["status"] = status
    return doc


def pdf_text(data):
    return "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(data)).pages)


# ------------------------------------------------------------------ every sample
@pytest.mark.parametrize("name", SAMPLE_FILES)
@pytest.mark.parametrize("fmt", ["pdf", "txt"])
@pytest.mark.parametrize("mode", ["pseudonym", "mask"])
def test_export_sample_zero_recoverable(name, fmt, mode):
    doc = analysed(name)
    res = redact.export(doc["text"], doc["spans"], doc["entities"], mode, fmt)
    r = res["receipt"]
    assert r["verification"]["recoverable"] == 0, r["verification"]["failures"]
    assert r["verification"]["passed"] is True
    assert r["removed"]["total"] >= len(doc["spans"])
    assert r["verification"]["checks"] > 0
    out = pdf_text(res["data"]) if fmt == "pdf" else res["data"].decode()
    # belt and braces: no detected value survives
    for s in doc["spans"]:
        assert s["text"] not in out, s["text"]
    token = "[PERSON_1]" if mode == "pseudonym" else "[NAME]"
    assert token in out


@pytest.mark.parametrize("name", SAMPLE_FILES)
def test_pdf_metadata_is_empty(name):
    doc = analysed(name)
    data = redact.export(doc["text"], doc["spans"], doc["entities"], "pseudonym", "pdf")["data"]
    reader = PdfReader(io.BytesIO(data))
    assert not reader.metadata  # no /Info: no author, title, producer, dates
    assert "/Info" not in reader.trailer
    assert "/Metadata" not in reader.trailer["/Root"]  # no XMP stream
    for word in (b"ReportLab", b"/Producer", b"/Author", b"/Title", b"/Creator", b"x:xmpmeta"):
        assert word not in data


def test_pdf_is_rebuilt_not_overlaid():
    """The source PDF's text layer must not survive under the output."""
    name = "medical_reimbursement_claim.pdf"
    doc = analysed(name)
    data = redact.export(doc["text"], doc["spans"], doc["entities"], "mask", "pdf")["data"]
    out = pdf_text(data)
    assert "Richa" not in out and "BQRPS4821K" not in out
    assert b"DejaVuSans" in data  # text drawn fresh with our embedded font


# ------------------------------------------------------------------ semantics
def _doc(text, statuses):
    spans, entities = link(text, detect(text, use_ner=False))
    for s in spans:
        s["status"] = statuses.get(s["text"], "approved")
    return text, spans, entities


def test_linked_variants_rescanned_case_insensitively():
    text = "Name: Richa Sharma\nRICHA SHARMA signed. Later R. Sharma and Ms. Sharma replied."
    text, spans, entities = _doc(text, {})
    # pretend the uppercase copy was never detected
    spans = [s for s in spans if s["text"] != "RICHA SHARMA"]
    res = redact.export(text, spans, entities, "pseudonym", "txt")
    out = res["redacted_text"]
    assert "sharma" not in out.lower()
    assert out.count("[PERSON_1]") == 4


def test_pending_spans_left_and_reported():
    text = "Mail riya@gmail.com. PAN BQRPS4821K."
    text, spans, entities = _doc(text, {"BQRPS4821K": "pending"})
    res = redact.export(text, spans, entities, "mask", "txt")
    assert res["redacted_text"] == "Mail [EMAIL]. PAN BQRPS4821K."
    assert res["receipt"]["review"]["pending_left_unredacted"] == 1
    assert res["receipt"]["verification"]["recoverable"] == 0


def test_rejected_mention_of_approved_entity_is_overridden():
    text = "Name: Richa Sharma. Ms. Sharma agreed."
    text, spans, entities = _doc(text, {"Ms. Sharma": "rejected"})
    res = redact.export(text, spans, entities, "pseudonym", "txt")
    assert "Sharma" not in res["redacted_text"]
    assert res["receipt"]["review"]["overridden_rejections"] == 1


def test_mask_vs_pseudonym_tokens():
    text = "Name: Richa Sharma, riya@gmail.com; Name: Arjun Menon"
    text, spans, entities = _doc(text, {})
    assert redact.export(text, spans, entities, "mask", "txt")["redacted_text"] == \
        "Name: [NAME], [EMAIL]; Name: [NAME]"
    assert redact.export(text, spans, entities, "pseudonym", "txt")["redacted_text"] == \
        "Name: [PERSON_1], [EMAIL_1]; Name: [PERSON_2]"


# ------------------------------------------------------------------ verifier catches leaks
def test_verifier_flags_leaks():
    text = "Name: Richa Sharma, mobile +91 98450 12345, mail richa.s92@gmail.com, Aadhaar 2345 6789 0124"
    text, spans, entities = _doc(text, {})
    plan = redact.plan_redaction(text, spans, entities)
    checks, _ = redact.build_checks(text, plan["regions"], entities)
    leaky = redact.build_pdf("Phone 98450 and RICHA SHARMA, user RICHA.S92, id 234567890124")
    kinds = {(f["kind"], f["value"].lower()) for f in redact.verify(leaky, "pdf", checks)}
    assert ("name_variant", "richa sharma") in kinds
    assert ("digit_run", "98450") in kinds
    assert ("email_username", "richa.s92") in kinds
    assert ("digit_run", "234567890124") in kinds
    clean = redact.build_pdf(redact.apply_redaction(text, plan["regions"], entities, "mask"))
    assert redact.verify(clean, "pdf", checks) == []


def test_receipt_never_contains_raw_pii():
    text = "Name: Richa Sharma, PAN BQRPS4821K"
    text, spans, entities = _doc(text, {})
    r = redact.export(text, spans, entities, "pseudonym", "pdf")["receipt"]
    assert "Richa" not in str(r) and "BQRPS4821K" not in str(r)
    assert len(r["file"]["sha256"]) == 64 and r["created_at"]


def test_generic_usernames_skipped_not_failed():
    text = "Write to hr@acme.in about HR policy."
    text, spans, entities = _doc(text, {})
    r = redact.export(text, spans, entities, "mask", "txt")["receipt"]
    assert r["verification"]["recoverable"] == 0
    assert r["verification"]["skipped"][0]["kind"] == "email_username"


# ------------------------------------------------------------------ API
def test_api_export_roundtrip_and_hash():
    import hashlib

    client = TestClient(app)
    doc = analysed("hr_offer_letter.txt")
    payload = {"text": doc["text"], "spans": doc["spans"], "entities": doc["entities"],
               "mode": "pseudonym", "format": "pdf"}
    j = client.post("/api/export", json=payload).json()
    data = base64.b64decode(j["file"]["base64"])
    assert hashlib.sha256(data).hexdigest() == j["receipt"]["file"]["sha256"]
    assert j["receipt"]["verification"]["recoverable"] == 0
    assert data.startswith(b"%PDF")


def test_api_export_blocks_pending_without_confirmation():
    client = TestClient(app)
    doc = analysed("hr_offer_letter.txt", status="pending")
    payload = {"text": doc["text"], "spans": doc["spans"], "entities": doc["entities"], "format": "txt"}
    r = client.post("/api/export", json=payload)
    assert r.status_code == 409 and "pending" in r.json()["detail"]
    r = client.post("/api/export", json={**payload, "allow_pending": True})
    assert r.status_code == 200
    assert r.json()["receipt"]["review"]["pending_left_unredacted"] == len(doc["spans"])


def test_api_export_rejects_bad_mode():
    client = TestClient(app)
    r = client.post("/api/export", json={"text": "x", "spans": [], "entities": [], "mode": "blur"})
    assert r.status_code == 422
