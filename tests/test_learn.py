"""Learning from feedback + the local-only network guard."""
import socket
import sys

import pytest
from fastapi.testclient import TestClient

from app import detect as det
from app import learn, netguard
from app.main import app


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(learn, "STORE_PATH", tmp_path / "learned.json")
    monkeypatch.setattr(det, "_model", None)
    monkeypatch.setitem(det._model_state, "status", "not_loaded")
    monkeypatch.setitem(sys.modules, "gliner", None)


client = TestClient(app)


def flagged(text):
    return {s["text"] for s in client.post("/api/analyze", data={"text": text}).json()["spans"]}


def test_rejecting_same_shape_twice_stops_flagging_in_new_doc():
    doc = "Codes EMP-11111 and EMP-22222 were issued."
    assert {"EMP-11111", "EMP-22222"} <= flagged(doc)
    r = client.post("/api/feedback", json={"events": [{"action": "reject", "text": "EMP-11111", "type": "ID"}],
                                           "text": doc}).json()
    assert "EMP-22222" in {s["text"] for s in r["analysis"]["spans"]}  # one rejection: shape not yet learned
    r = client.post("/api/feedback", json={"events": [{"action": "reject", "text": "EMP-22222", "type": "ID"}],
                                           "text": doc}).json()
    assert r["analysis"]["spans"] == []
    assert any(x["kind"] == "shape" and x["text"] == "AAA-99999" for x in r["learning"]["rules"])
    assert "EMP-33333" not in flagged("A new document mentions EMP-33333.")


def test_approved_string_detected_in_new_doc():
    assert "Zorawar" not in flagged("Zorawar arrived late.")
    client.post("/api/feedback", json={"events": [{"action": "approve", "text": "Zorawar", "type": "NAME"}]})
    spans = client.post("/api/analyze", data={"text": "Later, Zorawar signed."}).json()["spans"]
    hit = next(s for s in spans if s["text"] == "Zorawar")
    assert hit["type"] == "NAME" and hit["source"] == "learned"


def test_undo_restores_previous_behaviour():
    client.post("/api/feedback", json={"events": [{"action": "approve", "text": "Zorawar", "type": "NAME"}]})
    rule = next(r for r in client.get("/api/learn").json()["rules"] if r["kind"] == "denylist")
    client.post("/api/learn/undo", json={"rule_id": rule["id"]})
    assert "Zorawar" not in flagged("Zorawar arrived.")

    for code in ("EMP-11111", "EMP-22222"):
        client.post("/api/feedback", json={"events": [{"action": "reject", "text": code, "type": "ID"}]})
    assert "EMP-44444" not in flagged("See EMP-44444.")
    shape = next(r for r in client.get("/api/learn").json()["rules"] if r["kind"] == "shape")
    client.post("/api/learn/undo", json={"rule_id": shape["id"]})
    assert "EMP-44444" in flagged("See EMP-44444.")


def test_manual_adds_become_candidate_rule_and_thresholds_calibrate():
    for code in ("ZX-4471-Q", "ZX-9902-K"):
        client.post("/api/feedback", json={"events": [{"action": "manual", "text": code, "type": "ID"}]})
    assert "ZX-1234-P" in flagged("Ticketless ref ZX-1234-P noted.")
    thr = learn.thresholds()
    assert thr["ID"] < det.GLINER_THRESHOLD
    client.post("/api/feedback", json={"events": [{"action": "reject", "text": f"x{i}", "type": "NAME"}
                                                  for i in range(200)]})
    assert learn.thresholds()["NAME"] <= 0.70
    client.post("/api/feedback", json={"events": [{"action": "approve", "text": f"Y{i}", "type": "PHONE"}
                                                  for i in range(200)]})
    assert learn.thresholds()["PHONE"] >= 0.25


def test_names_never_become_shape_rules():
    for n in ("Rohit", "Mohan"):
        client.post("/api/feedback", json={"events": [{"action": "reject", "text": n, "type": "NAME"}]})
    assert not [r for r in client.get("/api/learn").json()["rules"] if r["kind"] == "shape"]


def test_reset_learning():
    client.post("/api/feedback", json={"events": [{"action": "approve", "text": "Zorawar", "type": "NAME"}]})
    assert client.get("/api/learn").json()["corrections"] == 1
    client.post("/api/learn/reset")
    assert client.get("/api/learn").json()["corrections"] == 0


def test_outbound_connection_blocked_and_counted():
    before = netguard.status()["blocked_attempts"]
    with pytest.raises(ConnectionRefusedError):
        socket.create_connection(("93.184.216.34", 80), timeout=2)
    st = client.get("/api/status").json()["network"]
    assert st["blocked_attempts"] == before + 1 and st["outbound_calls"] == 0
    assert st["recent_blocked"][-1]["host"] == "93.184.216.34"


def test_localhost_still_allowed():
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    c = socket.create_connection(("127.0.0.1", port), timeout=2)
    c.close()
    srv.close()
    assert netguard.is_local(("::1", 11434)) and netguard.is_local(("localhost", 11434))
