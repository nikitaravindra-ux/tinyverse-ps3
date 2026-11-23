"""Entity linking rules (rules-only detection, no model)."""
from pathlib import Path

from app.detect import detect
from app.link import link, parse_name

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


def ents(text, spans=None):
    spans = detect(text, use_ner=False) if spans is None else spans
    return link(text, spans)


def name_span(text, sub, conf=0.9):
    s = text.index(sub)
    return {"start": s, "end": s + len(sub), "text": sub, "type": "NAME", "confidence": conf, "source": "test"}


def test_parse_name_kinds():
    assert parse_name("Richa Sharma")["kind"] == "full"
    assert parse_name("R. Sharma")["kind"] == "initial_surname"
    assert parse_name("Ms. Sharma")["kind"] == "hon_surname"
    assert parse_name("Richa")["kind"] == "single"
    p = parse_name("Ms. Richa Sharma")
    assert (p["kind"], p["first"], p["surname"], p["gender"]) == ("full", "Richa", "Sharma", "f")


def test_initial_and_honorific_and_first_name_link():
    t = "Richa Sharma joined. R. Sharma signed. Ms. Sharma agreed. Richa will start Monday."
    spans = [name_span(t, x) for x in ["Richa Sharma", "R. Sharma", "Ms. Sharma"]]
    spans, entities = ents(t, spans)
    people = [e for e in entities if e["type"] == "NAME"]
    assert len(people) == 1
    e = people[0]
    assert e["pseudonym"] == "PERSON_1"
    assert e["canonical"] == "Richa Sharma"
    # the bare "Richa" was never detected; the exact-occurrence scan finds it
    assert "Richa" in e["variants"]
    assert len(e["span_ids"]) == 4


def test_incompatible_initial_not_linked():
    t = "Richa Sharma met K. Sharma."
    spans = [name_span(t, "Richa Sharma"), name_span(t, "K. Sharma")]
    _, entities = ents(t, spans)
    assert len([e for e in entities if e["type"] == "NAME"]) == 2


def test_honorific_surname_ambiguous_when_surname_not_unique():
    t = "Richa Sharma and Anil Sharma met. Ms. Sharma left."
    spans = [name_span(t, x) for x in ["Richa Sharma", "Anil Sharma", "Ms. Sharma"]]
    _, entities = ents(t, spans)
    people = [e for e in entities if e["type"] == "NAME"]
    assert len(people) == 3  # Ms. Sharma can't be resolved, stays separate


def test_honorific_gender_conflict_not_linked():
    t = "Ms. Richa Sharma met. Mr. Sharma left."
    spans = [name_span(t, "Ms. Richa Sharma"), name_span(t, "Mr. Sharma")]
    _, entities = ents(t, spans)
    assert len([e for e in entities if e["type"] == "NAME"]) == 2


def test_first_name_ambiguous_not_linked():
    t = "Richa Sharma and Richa Verma met. Richa left."
    spans = [name_span(t, "Richa Sharma"), name_span(t, "Richa Verma"), name_span(t, "Richa", 0.5)]
    spans[2]["start"] = t.rindex("Richa"); spans[2]["end"] = spans[2]["start"] + 5
    _, entities = ents(t, spans)
    assert len([e for e in entities if e["type"] == "NAME"]) == 3


def test_case_and_punctuation_normalised():
    t = "RICHA SHARMA wrote. R Sharma replied."
    spans = [name_span(t, "RICHA SHARMA"), name_span(t, "R Sharma")]
    _, entities = ents(t, spans)
    assert len([e for e in entities if e["type"] == "NAME"]) == 1


def test_exact_occurrence_scan_catches_missed_repeats():
    t = "Contact Priya Nair today.\n\nPriya Nair\nManager"
    spans = [name_span(t, "Priya Nair")]
    spans, entities = ents(t, spans)
    assert [s["text"] for s in spans] == ["Priya Nair", "Priya Nair"]
    assert spans[1]["source"] == "link"


def test_non_name_grouping_by_value():
    t = "PAN BQRPS4821K. Again: PAN BQRPS4821K. Mail riya@gmail.com or riya [at] gmail [dot] com."
    spans, entities = ents(t)
    ids = [e for e in entities if e["type"] == "ID"]
    emails = [e for e in entities if e["type"] == "EMAIL"]
    assert len(ids) == 1 and len(ids[0]["span_ids"]) == 2
    assert len(emails) == 1 and len(emails[0]["span_ids"]) == 2
    assert all(s.get("entity_id") for s in spans)


def test_span_ids_in_document_order():
    t = (SAMPLES / "hr_offer_letter.txt").read_text()
    spans, entities = ents(t)
    assert [s["id"] for s in spans] == [f"s{i + 1}" for i in range(len(spans))]
    assert [s["start"] for s in spans] == sorted(s["start"] for s in spans)
    known = {s["id"] for s in spans}
    assert all(set(e["span_ids"]) <= known for e in entities)


def test_hr_letter_richa_cluster():
    t = (SAMPLES / "hr_offer_letter.txt").read_text()
    spans, entities = ents(t)
    richa = next(e for e in entities if e["canonical"] == "Richa Sharma")
    for v in ["Ms. Sharma", "R. Sharma", "Richa"]:
        assert v in richa["variants"], v
