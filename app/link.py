"""Entity linking: cluster PII spans into entities and give each a pseudonym.

NAME spans are linked with name-variant rules:
  * same surname + compatible first initial  -> same entity  ("R. Sharma" ~ "Richa Sharma")
  * honorific + surname ("Ms. Sharma")        -> links when that surname is unique in the doc
  * first-name-only mention ("Richa")         -> links when exactly one entity has that first name
  * case / punctuation are normalised before comparing

Other types are grouped by their normalised value (same PAN twice = one entity).
Finally the document is scanned for every exact occurrence of every linked variant
so repeated mentions the detector missed are still covered.
"""
from __future__ import annotations

import re
from typing import Optional

HONORIFICS = {"mr", "ms", "mrs", "miss", "mx", "dr", "prof", "shri", "smt", "sri", "kum", "kumari"}
_MALE = {"mr", "shri", "sri"}
_FEMALE = {"ms", "mrs", "miss", "smt", "kum", "kumari"}

PSEUDONYM_PREFIX = {
    "NAME": "PERSON",
    "EMAIL": "EMAIL",
    "PHONE": "PHONE",
    "ADDRESS": "ADDRESS",
    "ORG_ADDRESS": "ORG_ADDRESS",
    "ID": "ID",
}

# Capitalised words that must never be scanned as a bare first name / surname.
_SCAN_STOPWORDS = {
    "The", "This", "That", "Dear", "Name", "Date", "Sir", "Madam", "Team", "Manager", "Director",
    "Head", "Officer", "Lead", "Bank", "Road", "Nagar", "Main", "Cross", "India", "Limited",
}


# ---------------------------------------------------------------------------
# Name parsing
# ---------------------------------------------------------------------------
def _norm(tok: str) -> str:
    return re.sub(r"[^a-z]", "", tok.lower())


def parse_name(text: str) -> Optional[dict]:
    """Split a NAME mention into honorific / first / middle / surname / kind."""
    toks = re.findall(r"[A-Za-z][A-Za-z'\-]*\.?", text)
    hon = None
    parts: list[str] = []
    for t in toks:
        w = _norm(t)
        if not w:
            continue
        if not parts and hon is None and w in HONORIFICS:
            hon = w
            continue
        parts.append(re.sub(r"'s$", "", t.rstrip(".")))
    if not parts:
        return None
    gender = "m" if hon in _MALE else "f" if hon in _FEMALE else None
    base = {"honorific": hon, "gender": gender, "first": None, "initial": None,
            "middle": [], "surname": None}
    if len(parts) == 1:
        p = parts[0]
        if hon:
            return dict(base, kind="hon_surname", surname=p)
        if len(p) == 1:
            return None
        return dict(base, kind="single", first=p)
    first, *middle, surname = parts
    if len(surname) == 1:  # "Sharma R." style: treat the long token as the surname
        first, surname = surname, first
    if len(first) == 1:
        return dict(base, kind="initial_surname", initial=first.upper(), middle=middle, surname=surname)
    return dict(base, kind="full", first=first, initial=first[0].upper(), middle=middle, surname=surname)


class _Entity:
    def __init__(self, typ: str):
        self.type = typ
        self.spans: list[dict] = []
        self.first: Optional[str] = None   # normalised
        self.initial: Optional[str] = None
        self.middle: list[str] = []
        self.surname: Optional[str] = None  # normalised
        self.gender: Optional[str] = None
        self.key: Optional[str] = None      # non-NAME grouping key

    def add(self, span: dict, p: Optional[dict] = None):
        self.spans.append(span)
        if not p:
            return
        if p.get("first") and not self.first:
            self.first = _norm(p["first"])
        if p.get("initial") and not self.initial:
            self.initial = p["initial"]
        if p.get("middle") and not self.middle:
            self.middle = [_norm(m) for m in p["middle"]]
        if p.get("surname") and not self.surname:
            self.surname = _norm(p["surname"])
        if p.get("gender") and not self.gender:
            self.gender = p["gender"]


def _gender_ok(e: _Entity, p: dict) -> bool:
    return not (e.gender and p.get("gender") and e.gender != p["gender"])


def _middle_ok(a: list[str], b: list[str]) -> bool:
    for x, y in zip(a, b):
        if x and y and x[0] != y[0]:
            return False
    return True


def _link_names(name_spans: list[dict]) -> list[_Entity]:
    parsed = [(sp, parse_name(sp["text"])) for sp in name_spans]
    entities: list[_Entity] = []
    order = {"full": 0, "initial_surname": 1, "hon_surname": 2, "single": 3}
    # Resolve the most informative mentions first so the shorter ones have something to link to.
    parsed.sort(key=lambda x: (order.get(x[1]["kind"], 9) if x[1] else 9, x[0]["start"]))

    def new(sp, p):
        e = _Entity("NAME")
        e.add(sp, p)
        entities.append(e)

    for sp, p in parsed:
        if p is None:
            new(sp, None)
            continue
        kind = p["kind"]
        sur = _norm(p["surname"]) if p.get("surname") else None
        if kind == "full":
            fn = _norm(p["first"])
            mid = [_norm(m) for m in p["middle"]]
            match = [e for e in entities if e.surname == sur and e.first == fn
                     and _middle_ok(e.middle, mid) and _gender_ok(e, p)]
            if match:
                match[0].add(sp, p)
            else:
                new(sp, p)
        elif kind == "initial_surname":
            match = [e for e in entities if e.surname == sur and e.initial == p["initial"] and _gender_ok(e, p)]
            if len(match) == 1:
                match[0].add(sp, p)
            else:
                new(sp, p)  # unknown or ambiguous initial: its own entity
        elif kind == "hon_surname":
            match = [e for e in entities if e.surname == sur]
            if len(match) == 1 and _gender_ok(match[0], p):
                match[0].add(sp, p)
            else:
                new(sp, p)
        else:  # single token: a first name, or possibly a bare surname
            fn = _norm(p["first"])
            by_first = [e for e in entities if e.first == fn]
            by_sur = [e for e in entities if e.surname == fn]
            if len(by_first) == 1:
                by_first[0].add(sp, p)
            elif not by_first and len(by_sur) == 1:
                by_sur[0].add(sp, None)
            else:
                new(sp, p)
    return entities


# ---------------------------------------------------------------------------
# Non-name grouping
# ---------------------------------------------------------------------------
def value_key(span: dict) -> str:
    t, typ = span["text"], span["type"]
    if typ == "EMAIL":
        from app.detect import normalize_email

        return (normalize_email(t) or t).lower()
    if typ == "PHONE":
        d = re.sub(r"\D", "", t)
        return d[-10:] if len(d) >= 10 else d
    if typ == "ID":
        return re.sub(r"[\s\-]", "", t).upper()
    return re.sub(r"[\W_]+", " ", t).strip().lower()


# ---------------------------------------------------------------------------
# Exact-occurrence scan
# ---------------------------------------------------------------------------
def _variant_regex(variant: str) -> re.Pattern:
    toks = variant.split()
    body = r"\s+".join(re.escape(t) for t in toks)
    return re.compile(rf"(?<![\w.@]){body}(?![\w@]|\.\w)")


def _overlaps(s: int, e: int, taken: list[tuple[int, int]]) -> bool:
    return any(s < te and e > ts for ts, te in taken)


_NEIGHBOUR_BEFORE = re.compile(r"(?:\b[A-Z][a-z]+|\b[A-Z]\.) $")
_NEIGHBOUR_AFTER = re.compile(r" [A-Z][a-z]+\b")


def _grow_name(text: str, s: int, e: int, taken: list[tuple[int, int]]) -> tuple[int, int]:
    """A bare first name / surname hit next to another capitalised word is part of an
    undetected full name ("Richa Sharma"): include the neighbour so no fragment is left."""
    m = _NEIGHBOUR_BEFORE.search(text, max(0, s - 30), s)
    if m and m.end() == s:
        word = m.group(0).strip()
        if word.rstrip(".") not in _SCAN_STOPWORDS and _norm(word) not in HONORIFICS \
                and not _overlaps(m.start(), s, taken):
            s = m.start()
    m = _NEIGHBOUR_AFTER.match(text, e)
    if m and m.group(0).strip() not in _SCAN_STOPWORDS and not _overlaps(e, m.end(), taken):
        e = m.end()
    return s, e


def _scan(text: str, entities: list[_Entity], all_spans: list[dict]) -> None:
    taken = [(sp["start"], sp["end"]) for sp in all_spans]
    first_counts: dict[str, int] = {}
    sur_counts: dict[str, int] = {}
    for e in entities:
        if e.type != "NAME":
            continue
        if e.first:
            first_counts[e.first] = first_counts.get(e.first, 0) + 1
        if e.surname:
            sur_counts[e.surname] = sur_counts.get(e.surname, 0) + 1

    jobs: list[tuple[str, _Entity, float, str]] = []  # (variant, entity, confidence, kind)
    for e in entities:
        best = max(sp["confidence"] for sp in e.spans)
        for v in {sp["text"] for sp in e.spans}:
            conf = max(sp["confidence"] for sp in e.spans if sp["text"] == v)
            jobs.append((v, e, conf, "variant"))
        if e.type != "NAME":
            continue
        # implied bare mentions: first name / surname, when unambiguous across the doc
        for norm_tok, counts in ((e.first, first_counts), (e.surname, sur_counts)):
            if not norm_tok or counts.get(norm_tok) != 1 or len(norm_tok) < 3:
                continue
            if norm_tok == e.first and sur_counts.get(norm_tok):
                continue  # this first name is also someone's surname
            if norm_tok == e.surname and first_counts.get(norm_tok):
                continue
            surface = next((w for sp in e.spans for w in re.findall(r"[A-Za-z]+", sp["text"])
                            if _norm(w) == norm_tok and w[0].isupper()), None)
            if surface and surface not in _SCAN_STOPWORDS:
                jobs.append((surface, e, round(min(0.85, best), 3), "bare"))

    # longest variants first so "Ms. Richa Sharma" wins over "Richa"
    jobs.sort(key=lambda j: -len(j[0]))
    for variant, e, conf, kind in jobs:
        for m in _variant_regex(variant).finditer(text):
            s, en = m.span()
            if _overlaps(s, en, taken):
                continue
            if kind == "bare":
                s, en = _grow_name(text, s, en, taken)
            sp = {"start": s, "end": en, "text": text[s:en], "type": e.type,
                  "confidence": conf, "source": "link"}
            # a grown hit ("Richa Sharma") can teach the entity its first name
            e.add(sp, parse_name(sp["text"]) if kind == "bare" else None)
            all_spans.append(sp)
            taken.append((s, en))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _canonical(e: _Entity) -> str:
    if e.type != "NAME":
        return max((sp["text"] for sp in e.spans), key=len)

    def score(t: str):
        p = parse_name(t)
        rank = {"full": 3, "initial_surname": 2, "hon_surname": 1, "single": 0}.get(p["kind"], 0) if p else -1
        return (rank, len(t) - (len(p["honorific"]) + 2 if p and p["honorific"] else 0))

    best = max((sp["text"] for sp in e.spans), key=score)
    p = parse_name(best)
    if p and p["honorific"]:  # strip the honorific for display: "Ms. Richa Sharma" -> "Richa Sharma"
        best = re.sub(rf"^\s*{re.escape(p['honorific'])}\.?\s+", "", best, flags=re.I)
    return best


def link(text: str, spans: list[dict], scan: bool = True) -> tuple[list[dict], list[dict]]:
    """Return (spans, entities). Spans gain ``entity_id``; new spans found by the
    exact-occurrence scan are added with ``source='link'``. Span ids are reassigned
    in document order."""
    spans = [dict(sp) for sp in spans]
    names = [sp for sp in spans if sp["type"] == "NAME"]
    entities = _link_names(names)

    groups: dict[tuple[str, str], _Entity] = {}
    for sp in spans:
        if sp["type"] == "NAME":
            continue
        k = (sp["type"], value_key(sp))
        if k not in groups:
            groups[k] = _Entity(sp["type"])
            groups[k].key = k[1]
        groups[k].add(sp)
    entities.extend(groups.values())

    if scan:
        for _ in range(2):  # second pass uses names learned from grown hits
            before = len(spans)
            _scan(text, entities, spans)
            if len(spans) == before:
                break

    spans.sort(key=lambda sp: sp["start"])
    for i, sp in enumerate(spans):
        sp["id"] = f"s{i + 1}"

    entities.sort(key=lambda e: min(sp["start"] for sp in e.spans))
    counters: dict[str, int] = {}
    out = []
    for i, e in enumerate(entities):
        prefix = PSEUDONYM_PREFIX.get(e.type, e.type)
        counters[prefix] = counters.get(prefix, 0) + 1
        eid = f"e{i + 1}"
        members = sorted(e.spans, key=lambda sp: sp["start"])
        for sp in members:
            sp["entity_id"] = eid
        variants = []
        for sp in members:
            if sp["text"] not in variants:
                variants.append(sp["text"])
        out.append({
            "id": eid,
            "type": e.type,
            "pseudonym": f"{prefix}_{counters[prefix]}",
            "canonical": _canonical(e),
            "variants": variants,
            "span_ids": [sp["id"] for sp in members],
            "confidence": max(sp["confidence"] for sp in members),
        })
    return spans, out
