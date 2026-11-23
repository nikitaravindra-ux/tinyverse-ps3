"""API tests run rules-only so they don't need the GLiNER weights."""
import io
import sys

import pytest
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas

from app import detect as det
from app.main import app


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(det, "_model", None)
    monkeypatch.setitem(det._model_state, "status", "not_loaded")
    monkeypatch.setitem(sys.modules, "gliner", None)
    return TestClient(app)


def _pdf(text=None):
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    if text:
        c.drawString(72, 720, text)
    else:
        c.rect(72, 600, 200, 100, fill=1)  # an "image-only" page: shapes, no text
    c.save()
    return buf.getvalue()


def test_analyze_txt(client):
    body = b"Dear Ms. Sharma,\nRicha Sharma (PAN BQRPS4821K) joined. Richa starts Monday."
    r = client.post("/api/analyze", files={"file": ("letter.txt", body, "text/plain")})
    assert r.status_code == 200
    j = r.json()
    assert set(j) >= {"text", "spans", "entities", "filename"}
    assert {s["text"] for s in j["spans"]} >= {"Ms. Sharma", "BQRPS4821K", "Richa"}
    person = [e for e in j["entities"] if e["type"] == "NAME"]
    assert len(person) == 1 and person[0]["pseudonym"] == "PERSON_1"


def test_analyze_text_pdf(client):
    pdf = _pdf("Employee EMP-10293, mobile 9876543210, email riya@gmail.com, signed by Ms. Riya Sen.")
    r = client.post("/api/analyze", files={"file": ("x.pdf", pdf, "application/pdf")})
    assert r.status_code == 200
    assert {"EMP-10293", "9876543210", "riya@gmail.com"} <= {s["text"] for s in r.json()["spans"]}


def test_scanned_pdf_rejected(client):
    r = client.post("/api/analyze", files={"file": ("scan.pdf", _pdf(), "application/pdf")})
    assert r.status_code == 422
    assert "scanned or image-only" in r.json()["detail"]


def test_unsupported_type_rejected(client):
    r = client.post("/api/analyze", files={"file": ("x.docx", b"PK\x03\x04", "application/octet-stream")})
    assert r.status_code == 415


def test_analyze_pasted_text_and_relink(client):
    j = client.post("/api/analyze", data={"text": "Contact Anil Kumar or Anil."}).json()
    t = j["text"]
    manual = {"start": t.index("Anil Kumar"), "end": t.index("Anil Kumar") + 10, "text": "Anil Kumar",
              "type": "NAME", "confidence": 1.0, "source": "manual"}
    r = client.post("/api/link", json={"text": t, "spans": j["spans"] + [manual]}).json()
    texts = [s["text"] for s in r["spans"]]
    assert "Anil Kumar" in texts and "Anil" in texts  # bare first name found by the scan
    assert len([e for e in r["entities"] if e["type"] == "NAME"]) == 1


def test_samples_endpoint(client):
    names = [s["name"] for s in client.get("/api/samples").json()]
    assert "medical_reimbursement_claim.pdf" in names
    r = client.get("/api/samples/hr_offer_letter.txt")
    assert r.status_code == 200 and r.json()["entities"]
    assert client.get("/api/samples/..%2Fapp%2Fmain.py").status_code == 404


def test_status_has_summary_model(client):
    from app.config import SUMMARY_MODEL

    assert client.get("/api/status").json()["ollama"]["summary_model"] == SUMMARY_MODEL == "llama3.2:3b"
