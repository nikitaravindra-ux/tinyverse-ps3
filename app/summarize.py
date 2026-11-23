"""Summaries of the REDACTED text only: Ollama (SUMMARY_MODEL) with an extractive fallback.

`summarize()` refuses any input that still contains a removed value. That check is
a real exception, not an `assert` statement, so it can't be optimised away with -O.
"""
from __future__ import annotations

import logging
import math
import re
import time
from collections import Counter
from typing import Optional

import requests

from app.config import OLLAMA_URL, SUMMARY_MODEL

log = logging.getLogger(__name__)

OLLAMA_TIMEOUT = 120
PLACEHOLDER_RE = re.compile(r"\[[A-Z]+(?:_[A-Z]+)*(?:_\d+)?\]")

SYSTEM_PROMPT = """You summarise documents that have already been redacted for privacy.

Rules:
- Placeholders in square brackets such as [PERSON_1], [EMAIL_2], [ADDRESS_1], [NAME], [ID] stand for removed personal data. Copy them exactly as written, brackets included.
- Never guess, expand, rename or describe who or what a placeholder might be.
- Never invent names, phone numbers, email addresses, street addresses or ID numbers.
- Do not copy long numbers (account, card, ID or phone numbers). Dates and money amounts are fine.

Write a short summary in plain text, no markdown bold or headings: one sentence saying what the document is, then 3 to 5 lines starting with "- " covering the key facts, decisions, dates and amounts. Stay under 130 words."""

STRICT_ADDENDUM = """

STRICT MODE: a previous summary of this document leaked personal data and was rejected.
- Refer to every person ONLY by their placeholder, e.g. [PERSON_1]. Never write a personal name, first name, surname, initials, or a nickname.
- Never write an email address or username, and never write any digits from a phone, account, card or ID number, not even partially.
- If you are unsure whether a detail identifies someone, leave it out."""


class RedactionGuardError(AssertionError):
    """Raised when text handed to the summarizer still contains a removed value."""


def _value_pattern(value: str) -> re.Pattern:
    body = r"\s+".join(re.escape(t) for t in value.split())
    return re.compile(rf"(?<!\w){body}(?!\w)", re.I)


def assert_redacted(text: str, removed_values: list[str]) -> None:
    """Hard guard: the summarizer only ever sees redacted text."""
    for v in removed_values:
        if v.strip() and _value_pattern(v).search(text):
            raise RedactionGuardError("Summarizer input still contains a removed value; refusing to summarise.")


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------
def _clean(out: str) -> str:
    out = out.replace("**", "").replace("__", "")
    out = re.sub(r"^\s*#+\s*", "", out, flags=re.M)
    out = re.sub(r"^\s*[*•]\s+", "- ", out, flags=re.M)
    out = re.sub(r"^(here is|here's)[^\n]*summary[^\n]*:\s*\n", "", out, flags=re.I)
    return out.strip()


def ollama_generate(text: str, strict: bool = False, model: str = SUMMARY_MODEL) -> str:
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": model,
            "system": SYSTEM_PROMPT + (STRICT_ADDENDUM if strict else ""),
            "prompt": f"Redacted document:\n<<<\n{text}\n>>>\n\nSummary:",
            "stream": False,
            "keep_alive": "10m",
            "options": {"temperature": 0.1 if strict else 0.2, "num_predict": 320, "num_ctx": 4096},
        },
        timeout=OLLAMA_TIMEOUT,
    )
    r.raise_for_status()
    return _clean(r.json().get("response", ""))


# ---------------------------------------------------------------------------
# Extractive fallback
# ---------------------------------------------------------------------------
_STOP = set("""a an the and or of to in on at for by with from as is are was were be been this that
these those it its into will shall may can not no any all each per our your their his her we you
they he she i me my us them than then there here which who whom whose what when where how also
such under within upon above below between""".split())
_ABBREV = re.compile(r"(?:\b(?:Mr|Ms|Mrs|Dr|No|Nos|Rs|Pvt|Ltd|St|Co|Inc|vs|etc|e\.g|i\.e|Ref|approx)|\b[A-Z])\.$")


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Sentence spans (start, end). Lines are also boundaries (forms, letters)."""
    spans = []
    for lm in re.finditer(r"[^\n]+", text):
        line, base = lm.group(0), lm.start()
        start = 0
        for m in re.finditer(r"[.!?](?=\s+[A-Z\[\"(])", line):
            if _ABBREV.search(line[:m.end()]):
                continue
            spans.append((base + start, base + m.end()))
            start = m.end()
        spans.append((base + start, base + len(line)))
    out = []
    for s, e in spans:
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e > s:
            out.append((s, e))
    return out


def extractive_summary(text: str, k: int = 5) -> str:
    sents = [(s, e, text[s:e]) for s, e in split_sentences(text)]
    words = lambda t: [w.lower() for w in re.findall(r"[A-Za-z]{3,}", PLACEHOLDER_RE.sub(" ", t))]
    freq = Counter(w for _, _, t in sents for w in words(t) if w not in _STOP)
    if not freq:
        return text[:400]
    top = max(freq.values())
    scored = []
    n = len(sents)
    for i, (s, e, t) in enumerate(sents):
        ws = [w for w in words(t) if w not in _STOP]
        if len(ws) < 5 or len(t) < 40 or t.count(":") > 1:
            continue  # skip headers, form fields, signature lines
        score = sum(freq[w] / top for w in ws) / math.sqrt(len(ws))
        score *= 1.15 if i < n * 0.3 else 1.0  # documents state their purpose early
        scored.append((score, i, t))
    best = sorted(sorted(scored, reverse=True)[:k], key=lambda x: x[1])
    if not best:  # form-like document with no prose: fall back to its first content lines
        best = [(0, i, t) for i, (_, _, t) in enumerate(sents) if len(t.split()) >= 3][:k]
    lines = ["- " + re.sub(r"\s+", " ", t).strip() for _, _, t in best]
    return "\n".join(lines) or "- " + re.sub(r"\s+", " ", text[:300]).strip()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def summarize(redacted_text: str, removed_values: list[str], strict: bool = False,
              use_llm: bool = True) -> dict:
    """Return {summary, method: 'llm'|'extractive', model, seconds, fallback_reason}."""
    assert_redacted(redacted_text, removed_values)
    t0 = time.perf_counter()
    reason: Optional[str] = None
    if use_llm:
        try:
            out = ollama_generate(redacted_text, strict=strict)
            if out:
                return {"summary": out, "method": "llm", "model": SUMMARY_MODEL,
                        "seconds": round(time.perf_counter() - t0, 2), "fallback_reason": None}
            reason = "the model returned an empty response"
        except requests.ConnectionError:
            reason = "Ollama is not running"
        except requests.Timeout:
            reason = "Ollama timed out"
        except Exception as exc:  # model missing, HTTP error, ...
            reason = f"Ollama error ({type(exc).__name__})"
        log.warning("LLM summary unavailable (%s); using extractive fallback", reason)
    else:
        reason = "LLM disabled"
    return {"summary": extractive_summary(redacted_text), "method": "extractive", "model": None,
            "seconds": round(time.perf_counter() - t0, 2), "fallback_reason": reason}
