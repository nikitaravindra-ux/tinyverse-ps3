"""Summarizer guard, leak auditor (planted leaks), regenerate/replace flow, QI risk."""
import sys

import pytest
import requests
from fastapi.testclient import TestClient

from app import audit, redact
from app import detect as det
from app import summarize as summ
from app.detect import detect
from app.link import link
from app.main import app

DOC = (
    "OFFER LETTER\n"
    "Name: Richa Sharma\n"
    "Dear Ms. Sharma, your Employee ID is EMP-10293 and PAN BQRPS4821K.\n"
    "Call +91 98450 12345 or mail richa.sharma92@gmail.com. Aadhaar: 2345 6789 0124.\n"
    "R. Sharma joins on 1st April 2025 in Bengaluru. Fee Rs. 2025 paid.\n"
)


@pytest.fixture(autouse=True)
def rules_only(monkeypatch):
    monkeypatch.setattr(det, "_model", None)
    monkeypatch.setitem(det._model_state, "status", "not_loaded")
    monkeypatch.setitem(sys.modules, "gliner", None)


@pytest.fixture()
def doc():
    spans, entities = link(DOC, detect(DOC, use_ner=False))
    for s in spans:
        s["status"] = "approved"
    view = redact.redacted_view(DOC, spans, entities, "pseudonym")
    return {"spans": spans, "entities": entities, **view}


def kinds(summary, d):
    res = audit.audit_summary(summary, d["removed"], d["entities"], d["redacted_text"], use_ner=False)
    return {h["kind"] for h in res["hits"]}, res


# ------------------------------------------------------------------ guard
def test_guard_refuses_unredacted_input(doc):
    values = [r["value"] for r in doc["removed"]]
    with pytest.raises(summ.RedactionGuardError):
        summ.summarize(DOC, values, use_llm=False)
    with pytest.raises(AssertionError):  # it is an AssertionError subclass
        summ.assert_redacted("note: richa SHARMA called", values)
    summ.assert_redacted(doc["redacted_text"], values)  # redacted text passes


# ------------------------------------------------------------------ planted leaks
def test_clean_summary_passes(doc):
    found, res = kinds("[PERSON_1] received an offer and joins on 1st April 2025 in Bengaluru.", doc)
    assert found == set()
    assert res["checks"] > 10


@pytest.mark.parametrize("planted,kind", [
    ("Richa Sharma joins in April.", "removed_value"),
    ("The candidate, whose mobile ends 12345, joins soon.", "partial_digits"),
    ("Her Aadhaar ends in 0124.", "partial_digits"),
    ("Employee 10293 was onboarded.", "partial_digits"),
    ("Reach her as richa.sharma92 on mail.", "email_username"),
    ("Sharma accepted the offer.", "surname"),
    ("R.S. accepted the offer.", "initials"),
    ("Richa S. accepted the offer.", "initials"),
    ("Rihca accepted the offer.", "fuzzy_name"),
    ("Sharmaa accepted the offer.", "fuzzy_name"),
    ("Contact her at riya.k@yahoo.com for details.", "detector"),
    ("Her PAN BQRPS4821K was verified.", "detector"),
])
def test_auditor_catches_planted_leak(doc, planted, kind):
    found, _ = kinds(planted, doc)
    assert kind in found, found


def test_removed_first_name_is_a_leak_even_if_detection_missed_it():
    # detection missed the bare "Sunita", so it is still in the redacted document
    entities = [{"id": "e1", "type": "NAME", "pseudonym": "PERSON_1", "canonical": "Sunita Patil",
                 "variants": ["Sunita Patil"], "span_ids": ["s1"], "confidence": 0.99}]
    removed = [{"value": "Sunita Patil", "type": "NAME", "entity_id": "e1"}]
    source = "Name: [PERSON_1]\nContact Sunita today."
    res = audit.audit_summary("Sunita runs the firm.", removed, entities, source, use_ner=False)
    assert [h["kind"] for h in res["hits"]] == ["fuzzy_name"]


def test_years_and_amounts_from_document_are_not_leaks(doc):
    # "2025" appears in the redacted doc (date and fee), so it is not a partial-digit leak
    found, _ = kinds("[PERSON_1] joins on 1st April 2025 and paid Rs. 2025.", doc)
    assert found == set()


# ------------------------------------------------------------------ regenerate / replace flow
def _fake(outputs):
    calls = []

    def fake(text, values, strict=False):
        summ.assert_redacted(text, values)
        calls.append(strict)
        return {"summary": outputs[len(calls) - 1], "method": "llm", "model": "fake",
                "seconds": 0.1, "fallback_reason": None}
    return fake, calls


def test_leak_triggers_one_strict_regeneration(doc):
    fake, calls = _fake(["Richa Sharma joins.", "[PERSON_1] joins."])
    r = audit.audited_summary(doc["redacted_text"], doc["removed"], doc["entities"], "pseudonym", fake,
                              use_ner=False)
    assert calls == [False, True]
    assert r["summary"] == "[PERSON_1] joins."
    assert r["audit"]["passed"] and r["audit"]["regenerated"] and r["audit"]["replaced"] == []


def test_persistent_leak_is_replaced_and_logged(doc):
    fake, calls = _fake(["Sharma joins; call 98450.", "Ms. Sharma (mobile 98450) joins."])
    r = audit.audited_summary(doc["redacted_text"], doc["removed"], doc["entities"], "pseudonym", fake,
                              use_ner=False)
    assert calls == [False, True]
    assert "Sharma" not in r["summary"] and "98450" not in r["summary"]
    assert "[PERSON_1]" in r["summary"] and "[PHONE_1]" in r["summary"]
    assert r["audit"]["passed"] is True
    assert {h["kind"] for h in r["audit"]["replaced"]} >= {"partial_digits"}
    assert [e["event"] for e in r["audit"]["log"]] == ["regenerated", "replaced"]
    assert "Sharma" not in str(r["audit"])  # the log never echoes the leaked value


def test_mask_mode_replacement(doc):
    fake, _ = _fake(["Sharma joins.", "Sharma joins."])
    view = redact.redacted_view(DOC, doc["spans"], doc["entities"], "mask")
    r = audit.audited_summary(view["redacted_text"], view["removed"], doc["entities"], "mask", fake,
                              use_ner=False)
    assert r["summary"] == "[NAME] joins."


# ------------------------------------------------------------------ fallback
def test_extractive_fallback_when_ollama_down(doc, monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError("refused")
    monkeypatch.setattr(summ.requests, "post", boom)
    values = [r["value"] for r in doc["removed"]]
    r = summ.summarize(doc["redacted_text"], values)
    assert r["method"] == "extractive" and r["fallback_reason"] == "Ollama is not running"
    assert r["summary"].startswith("- ")
    summ.assert_redacted(r["summary"], values)


def test_extractive_picks_content_sentences():
    text = ("Subject: Offer\nName: [PERSON_1]\n"
            "We are pleased to offer [PERSON_1] the role of analyst at the company with a joining date in April.\n"
            "The annual compensation for the analyst role at the company is Rs. 18,50,000 per year.\n"
            "Signature: ____\n")
    out = summ.extractive_summary(text, k=2)
    assert "pleased to offer" in out and "compensation" in out and "Signature" not in out


# ------------------------------------------------------------------ QI risk
def test_qi_flags_cfo_of_pune_based_firm():
    r = audit.qi_report("The report was approved by the CFO of a Pune-based firm last week.")
    s = r["sentences"][0]
    assert s["level"] == "high" and r["max_level"] == "high"
    assert {"role", "location", "org"} <= set(s["categories"])
    assert "Role + city" in s["explanation"] and "single out" in s["explanation"]
    reps = {t["text"]: t["replacement"] for t in s["terms"]}
    assert reps["CFO"] == "senior executive"
    out = audit.generalize(s["text"], [dict(t, start=t["start"] - s["start"], end=t["end"] - s["start"])
                                       for t in s["terms"]])
    assert "CFO" not in out and "Pune" not in out
    assert audit.qi_report(out)["max_level"] in ("low", "none")


@pytest.mark.parametrize("text,level", [
    ("She is the only female engineer in the plant.", "high"),
    ("The 34-year-old principal of a Jayanagar school complained.", "high"),
    ("The director of Dhruva Logistics Private Limited signed.", "medium"),
    ("The invoice was paid on 12/03/2025.", "low"),
    ("The weather was pleasant.", "none"),
])
def test_qi_levels(text, level):
    assert audit.qi_report(text)["max_level"] == level


def test_qi_city_generalisation_and_birth_dates():
    r = audit.qi_report("Born on 14/08/1986 in Pune, the CTO leads the team.")
    reps = {t["text"]: t["replacement"] for t in r["sentences"][0]["terms"]}
    assert reps["Pune"] == "a city in India"
    assert reps["14/08/1986"] == "the 1980s"
    assert reps["CTO"] == "senior executive"


def test_qi_prose_is_not_a_company_name():
    r = audit.qi_report("This Master Services Agreement is entered into at Pune.")
    assert r["max_level"] == "low"
    assert [t["label"] for t in r["sentences"][0]["terms"]] == ["city"]


def test_qi_ignores_placeholders():
    assert audit.qi_report("[PERSON_1] lives at [ADDRESS_1].")["max_level"] == "none"


# ------------------------------------------------------------------ API
def test_api_summarize_with_fallback(monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError("refused")
    monkeypatch.setattr(summ.requests, "post", boom)
    client = TestClient(app)
    j = client.post("/api/analyze", data={"text": DOC}).json()
    spans = [dict(s, status="approved") for s in j["spans"]]
    r = client.post("/api/summarize", json={"text": j["text"], "spans": spans, "entities": j["entities"]}).json()
    assert r["method"] == "extractive" and r["audit"]["passed"]
    assert "Sharma" not in r["summary"] and set(r["qi"]) >= {"sentences", "max_level"}


def test_api_export_with_generalizations():
    client = TestClient(app)
    text = "Name: Richa Sharma\nShe is the CFO of a Pune-based firm."
    j = client.post("/api/analyze", data={"text": text}).json()
    spans = [dict(s, status="approved") for s in j["spans"]]
    r = client.post("/api/export", json={
        "text": j["text"], "spans": spans, "entities": j["entities"], "format": "txt",
        "generalizations": [{"find": "CFO", "replace": "senior executive"},
                            {"find": "Pune-based", "replace": "regionally based"}]}).json()
    assert r["redacted_text"] == "Name: [PERSON_1]\nShe is the senior executive of a regionally based firm."
    assert r["receipt"]["generalizations"] == 2 and r["receipt"]["verification"]["passed"]
