"""Evaluation tooling: scorer arithmetic, Indian-set validity, learning off by default."""
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

from app import detect as det
from app import learn
from app.detect import is_valid_aadhaar

EVAL = Path(__file__).resolve().parent.parent / "eval"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, EVAL / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def run_eval():
    return _load("run_eval")


@pytest.fixture(autouse=True)
def rules_only(monkeypatch):
    monkeypatch.setattr(det, "_model", None)
    monkeypatch.setitem(det._model_state, "status", "not_loaded")
    monkeypatch.setitem(sys.modules, "gliner", None)


def test_score_overlap_matching(run_eval):
    sc = run_eval.Score()
    gold = [{"start": 0, "end": 10, "type": "NAME"}, {"start": 20, "end": 30, "type": "ID"}]
    preds = [{"start": 2, "end": 8, "type": "NAME"},            # overlap, same type -> TP
             {"start": 20, "end": 30, "type": "PHONE"},          # wrong type -> FP, ID missed
             {"start": 40, "end": 45, "type": "ORG_ADDRESS"}]    # counts as ADDRESS -> FP
    sc.add(preds, gold)
    rep = sc.report()
    assert rep["per_type"]["NAME"]["precision"] == 1.0 and rep["per_type"]["NAME"]["recall"] == 1.0
    assert rep["per_type"]["ID"]["recall"] == 0.0
    assert rep["micro"]["precision"] == pytest.approx(1 / 3)
    assert rep["micro"]["recall"] == pytest.approx(1 / 2)


def test_indian_set_gold_and_negatives_are_valid():
    path = EVAL / "data" / "indian_synthetic.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 200
    aadhaars = bad = 0
    for r in rows:
        for s in r["spans"]:
            t = r["text"][s["start"]:s["end"]]
            if s["type"] == "ID" and "Aadhaar" in r["text"][max(0, s["start"] - 18):s["start"]]:
                assert is_valid_aadhaar(t), t
                aadhaars += 1
        for n in r["negatives"]:
            if n["kind"] == "invalid_aadhaar":
                assert not is_valid_aadhaar(r["text"][n["start"]:n["end"]])
                bad += 1
    assert aadhaars >= 100 and bad >= 100


def test_indian_set_is_reproducible(tmp_path, monkeypatch):
    gen = _load("make_indian_set")
    monkeypatch.setattr(gen, "OUT", tmp_path / "x.jsonl")
    gen.main()
    assert (tmp_path / "x.jsonl").read_bytes() == (EVAL / "data" / "indian_synthetic.jsonl").read_bytes()


def test_eval_ignores_learning_unless_flagged(run_eval, monkeypatch, tmp_path):
    monkeypatch.setattr(learn, "STORE_PATH", tmp_path / "learned.json")
    for code in ("EMP-11111", "EMP-22222"):   # teach the (wrong) rule that EMP ids are not PII
        learn.record([{"action": "reject", "text": code, "type": "ID"}])
    text = "Employee EMP-33333 joined."
    clean = run_eval.predict(text, "regex", None)
    learned = run_eval.predict(text, "regex", learn.load())
    assert [s["text"] for s in clean] == ["EMP-33333"]
    assert learned == []
