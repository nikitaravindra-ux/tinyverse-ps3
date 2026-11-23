"""/api/status must report the rules-only fallback when GLiNER cannot load."""
import sys

from fastapi.testclient import TestClient

from app import detect as det
from app.main import app


def test_status_reports_rules_fallback(monkeypatch):
    monkeypatch.setattr(det, "_model", None)
    monkeypatch.setitem(det._model_state, "status", "not_loaded")
    monkeypatch.setitem(sys.modules, "gliner", None)  # makes `from gliner import GLiNER` fail
    client = TestClient(app)

    r = client.post("/api/detect", json={"text": "Mail riya@gmail.com, PAN BQRPS4821K"})
    assert r.status_code == 200
    assert {s["text"] for s in r.json()["spans"]} == {"riya@gmail.com", "BQRPS4821K"}

    st = client.get("/api/status").json()["detector"]
    assert st["status"] == "unavailable"
    assert "rules-only" in st["mode"]
    assert st["error"]
