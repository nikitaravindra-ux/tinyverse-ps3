"""Leak auditor for summaries + quasi-identifier (QI) risk scoring.

Leak audit: the summary is checked with the full detector and fuzzy matches against
everything that was removed: partial digits (4+), email usernames, surname-only
mentions, initials, and names within Levenshtein distance 2.

QI risk: a sentence that combines a role, a location, an organisation, an age/date
and/or a rare attribute can single out one person even with no name in it.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from app.detect import detect, normalize_email
from app.link import HONORIFICS, parse_name
from app.redact import GENERIC_USERNAMES, MIN_USERNAME_LEN

log = logging.getLogger(__name__)

PLACEHOLDER_RE = re.compile(r"\[[A-Z]+(?:_[A-Z]+)*(?:_\d+)?\]")
MASK_TOKEN = {"NAME": "[NAME]", "EMAIL": "[EMAIL]", "PHONE": "[PHONE]", "ADDRESS": "[ADDRESS]",
              "ORG_ADDRESS": "[ORG_ADDRESS]", "ID": "[ID]"}
# Capitalised words that are never treated as fuzzy name matches.
_COMMON_CAPS = {
    "The", "This", "That", "They", "Their", "There", "These", "With", "From", "Date", "Dear", "Name",
    "Over", "Under", "After", "Before", "Offer", "Letter", "Bank", "Form", "Report", "Order", "Mail",
    "Email", "Phone", "Mobile", "Address", "Amount", "Total", "Vendor", "Client", "Agreement",
    "Contract", "Summary", "Document", "Employee", "Incident", "Section", "Account", "Policy",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def levenshtein(a: str, b: str, cap: int = 3) -> int:
    a, b = a.lower(), b.lower()
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def _word_re(value: str, flags=re.I) -> re.Pattern:
    body = r"\s+".join(re.escape(t) for t in value.split())
    return re.compile(rf"(?<!\w){body}(?!\w)", flags)


def _mask(v: str) -> str:
    v = v.strip()
    return f"{v[:1]}{'•' * max(0, min(len(v), 10) - 1)} ({len(v)} chars)"


def _in_placeholder(s: int, e: int, summary: str) -> bool:
    return any(m.start() <= s and e <= m.end() for m in PLACEHOLDER_RE.finditer(summary))


def placeholder_for(item: dict, entities: dict, mode: str) -> str:
    ent = entities.get(item.get("entity_id"))
    if mode == "pseudonym" and ent:
        return f"[{ent['pseudonym']}]"
    return MASK_TOKEN.get(item["type"], f"[{item['type']}]")


# ---------------------------------------------------------------------------
# Leak audit
# ---------------------------------------------------------------------------
def audit_summary(summary: str, removed: list[dict], entities: list[dict], redacted_source: str,
                  use_ner: bool = True) -> dict:
    """Return {checks, hits:[{kind,type,start,end,entity_id,hint}]}.

    ``removed``: [{value, type, entity_id}] for every region removed from the document.
    ``redacted_source``: the redacted document; values that legitimately remain in it
    (kept by the reviewer, dates, amounts) are not leaks.
    """
    ents = {e["id"]: e for e in entities}
    hits: list[dict] = []
    checks = 0

    def hit(kind, typ, s, e, entity_id=None):
        if _in_placeholder(s, e, summary):
            return
        hits.append({"kind": kind, "type": typ, "start": s, "end": e, "entity_id": entity_id,
                     "hint": _mask(summary[s:e])})

    src_lower = redacted_source.lower()

    # 1) full detector on the summary
    checks += 1
    for sp in detect(summary, use_ner=use_ner):
        if sp["type"] == "ORG_ADDRESS" or _in_placeholder(sp["start"], sp["end"], summary):
            continue
        if PLACEHOLDER_RE.fullmatch(sp["text"].strip("[]() ")):
            continue
        if _word_re(sp["text"]).search(redacted_source):
            continue  # also present in the redacted document: kept by the reviewer, not a leak
        match = next((r for r in removed if r["type"] == sp["type"]
                      and re.sub(r"\W", "", r["value"]).lower() == re.sub(r"\W", "", sp["text"]).lower()), None)
        hit("detector", sp["type"], sp["start"], sp["end"], match and match.get("entity_id"))

    # 2) exact removed values
    for r in removed:
        checks += 1
        for m in _word_re(r["value"]).finditer(summary):
            hit("removed_value", r["type"], m.start(), m.end(), r.get("entity_id"))

    # 3) partial digits (4+) from removed phones / IDs
    digit_sources = [(re.sub(r"\D", "", r["value"]), r) for r in removed if r["type"] in ("PHONE", "ID")]
    digit_sources = [(d, r) for d, r in digit_sources if len(d) >= 4]
    for m in re.finditer(r"\d[\d \-]*\d|\d", summary):
        run = re.sub(r"\D", "", m.group(0))
        if len(run) < 4:
            continue
        checks += 1
        if re.search(rf"(?<!\d){re.escape(m.group(0))}(?!\d)", redacted_source):
            continue  # e.g. a year or amount that is in the redacted document anyway
        for d, r in digit_sources:
            if run in d:
                hit("partial_digits", r["type"], m.start(), m.end(), r.get("entity_id"))
                break

    # 4) email usernames
    for r in removed:
        if r["type"] != "EMAIL":
            continue
        user = (normalize_email(r["value"]) or r["value"]).split("@", 1)[0]
        if len(user) < MIN_USERNAME_LEN or user.lower() in GENERIC_USERNAMES:
            continue
        checks += 1
        for m in _word_re(user).finditer(summary):
            hit("email_username", "EMAIL", m.start(), m.end(), r.get("entity_id"))

    # 5-7) names: surname-only, initials, Levenshtein <= 2
    people: list[tuple[Optional[str], dict]] = []
    for e in entities:
        if e["type"] != "NAME" or not any(r.get("entity_id") == e["id"] for r in removed):
            continue
        for v in e["variants"]:
            p = parse_name(v)
            if p:
                people.append((e["id"], p))
    name_tokens: dict[str, Optional[str]] = {}
    for eid, p in people:
        for tok in (p.get("first"), p.get("surname"), *p.get("middle", [])):
            if tok and len(tok) >= 3 and tok.lower() not in HONORIFICS:
                name_tokens.setdefault(tok, eid)

    for eid, p in people:
        sur = p.get("surname")
        if sur and len(sur) >= 3:  # a removed person's surname is a leak wherever it appears
            checks += 1
            for m in _word_re(sur).finditer(summary):
                hit("surname", "NAME", m.start(), m.end(), eid)
        first = p.get("first") or p.get("initial")
        if first and sur:
            fi, si = first[0], sur[0]
            checks += 1
            for pat in (rf"\b{fi}\.\s?{si}\.", rf"\b{fi}\.?\s+{re.escape(sur)}\b",
                        rf"\b{re.escape(first)}\s+{si}\.(?!\w)" if len(first) > 1 else None):
                if pat:
                    for m in re.finditer(pat, summary):
                        hit("initials", "NAME", m.start(), m.end(), eid)

    for m in re.finditer(r"\b[A-Z][a-z]{3,}\b", summary):
        word = m.group(0)
        exact_name = any(word.lower() == t.lower() for t in name_tokens)
        if word in _COMMON_CAPS or (word.lower() in src_lower.split() and not exact_name):
            continue
        checks += 1
        for tok, eid in name_tokens.items():
            limit = 2 if len(tok) >= 5 else 1
            if levenshtein(word, tok) <= limit:
                hit("fuzzy_name", "NAME", m.start(), m.end(), eid)
                break

    # de-duplicate overlapping hits (keep the longest)
    hits.sort(key=lambda h: (-(h["end"] - h["start"]), h["start"]))
    kept: list[dict] = []
    for h in hits:
        if not any(h["start"] < k["end"] and h["end"] > k["start"] for k in kept):
            kept.append(h)
    kept.sort(key=lambda h: h["start"])
    return {"checks": checks, "hits": kept}


def replace_hits(summary: str, hits: list[dict], entities: list[dict], mode: str) -> str:
    ents = {e["id"]: e for e in entities}
    out = summary
    for h in sorted(hits, key=lambda h: -h["start"]):
        out = out[:h["start"]] + placeholder_for(h, ents, mode) + out[h["end"]:]
    return out


def audited_summary(redacted_text: str, removed: list[dict], entities: list[dict], mode: str,
                    summarizer, use_ner: bool = True) -> dict:
    """Summarise -> audit -> (regenerate once, strict) -> (replace leaks + log)."""
    values = [r["value"] for r in removed]
    first = summarizer(redacted_text, values, strict=False)
    result = dict(first)
    audit1 = audit_summary(first["summary"], removed, entities, redacted_text, use_ner)
    total_checks = audit1["checks"]
    log_entries: list[dict] = []
    regenerated = False
    current, audit = first["summary"], audit1

    if audit1["hits"] and first["method"] == "llm":
        regenerated = True
        log_entries.append({"event": "regenerated", "reason": f"{len(audit1['hits'])} leak(s) in first draft",
                            "hits": [{k: h[k] for k in ("kind", "type", "hint")} for h in audit1["hits"]]})
        second = summarizer(redacted_text, values, strict=True)
        result.update({k: second[k] for k in ("method", "model", "fallback_reason")})
        result["seconds"] = round(first["seconds"] + second["seconds"], 2)
        current = second["summary"]
        audit = audit_summary(current, removed, entities, redacted_text, use_ner)
        total_checks += audit["checks"]

    replaced = []
    if audit["hits"]:
        for h in audit["hits"]:
            replaced.append({k: h[k] for k in ("kind", "type", "hint")})
            log.warning("summary leak replaced: kind=%s type=%s", h["kind"], h["type"])
        current = replace_hits(current, audit["hits"], entities, mode)
        log_entries.append({"event": "replaced", "count": len(replaced), "hits": replaced})
        final = audit_summary(current, removed, entities, redacted_text, use_ner)
        total_checks += final["checks"]
        audit = final

    result["summary"] = current
    result["audit"] = {
        "passed": not audit["hits"],
        "checks": total_checks,
        "initial_hits": len(audit1["hits"]),
        "regenerated": regenerated,
        "replaced": replaced,
        "remaining_hits": [{k: h[k] for k in ("kind", "type", "hint")} for h in audit["hits"]],
        "log": log_entries,
    }
    return result


# ---------------------------------------------------------------------------
# Quasi-identifier risk
# ---------------------------------------------------------------------------
_CITIES = ("Bengaluru|Bangalore|Mumbai|Bombay|Delhi|New Delhi|Pune|Chennai|Madras|Hyderabad|Kolkata|"
           "Calcutta|Ahmedabad|Jaipur|Lucknow|Kanpur|Nagpur|Indore|Bhopal|Patna|Surat|Vadodara|Nashik|"
           "Noida|Gurugram|Gurgaon|Ghaziabad|Faridabad|Kochi|Cochin|Thiruvananthapuram|Coimbatore|"
           "Madurai|Mysuru|Mysore|Mangaluru|Mangalore|Hubballi|Belagavi|Tumakuru|Visakhapatnam|"
           "Vijayawada|Bhubaneswar|Guwahati|Chandigarh|Ludhiana|Amritsar|Dehradun|Raipur|Ranchi|"
           "Thane|Navi Mumbai|Aurangabad|Goa|Panaji|Shimla|Srinagar|Jammu|Udaipur|Jodhpur|Varanasi|Agra")
_AREAS = ("Koramangala|HSR Layout|Indiranagar|Jayanagar|Whitefield|Marathahalli|Bellandur|Electronic City|"
          "Malleshwaram|Basavanagudi|JP Nagar|BTM Layout|Hebbal|Yelahanka|Banashankari|Rajajinagar|"
          "Andheri|Bandra|Powai|Juhu|Dadar|Colaba|Worli|Borivali|Hinjewadi|Kothrud|Baner|Wakad|"
          "Bhosari|Hadapsar|Viman Nagar|Gachibowli|Hitech City|Madhapur|Banjara Hills|Jubilee Hills|"
          "Adyar|T Nagar|Velachery|Anna Nagar|Salt Lake|Connaught Place|Saket|Dwarka|Karol Bagh|"
          "Shivaji Nagar|Gandhi Nagar|MIDC")
_STATES = ("Karnataka|Maharashtra|Tamil Nadu|Kerala|Telangana|Andhra Pradesh|Gujarat|Rajasthan|"
           "Uttar Pradesh|Madhya Pradesh|West Bengal|Odisha|Punjab|Haryana|Bihar|Assam")
_ORG_SUFFIX = (r"(?:Pvt\.?\s+Ltd\.?|Private\s+Limited|Limited|Ltd\.?|LLP|Inc\.?|Bank|Services|Logistics|"
               r"Softlabs|Systems|Technologies|Solutions|Hospital|Institute|University|College|School|"
               r"Industries|Enterprises|Foundation|Trust)")

# (category, label, weight, pattern, flags, replacement or callable)
QI_RULES: list[tuple] = [
    ("role", "Role", 2, r"\b(?:CFO|CEO|CTO|COO|CIO|CISO|CMO|CHRO|CXO|MD)\b", 0, "senior executive"),
    ("role", "Role", 2, r"\bchief\s+\w+\s+officer\b", re.I, "senior executive"),
    ("role", "Role", 2, r"\b(?:managing\s+director|vice[- ]president|chairman|chairperson|chairwoman|"
                        r"co-?founder|founder|president)\b", re.I, "senior executive"),
    ("role", "Role", 2, r"\b(?:director|proprietor|owner|partner)\b", re.I, "senior leader"),
    ("role", "Role", 2, r"\b(?:principal|headmaster|headmistress|dean|vice[- ]chancellor|registrar|warden)\b",
     re.I, "senior administrator"),
    ("role", "Role", 2, r"\bhead\s+of\s+[A-Za-z]+\b", re.I, "department head"),
    ("role", "Role", 2, r"\b(?:SOC|team|tech|project)\s+lead\b", re.I, "team lead"),
    ("location", "neighbourhood", 2, rf"\b(?:{_AREAS})\b", 0, "a neighbourhood"),
    ("location", "city", 1, rf"\b(?:{_CITIES})-based\b", 0, "regionally based"),
    ("location", "city", 1, rf"\b(?:{_CITIES})\b", 0, "a city in India"),
    ("location", "state", 1, rf"\b(?:{_STATES})\b", 0, "a state in India"),
    ("org", "company name", 2, rf"\b(?:[A-Z][A-Za-z&]+\s+){{1,4}}{_ORG_SUFFIX}", 0, "an organisation"),
    ("org", "company name", 2, rf"\b(?:[A-Z]{{3,}}\s+){{1,4}}(?:PRIVATE\s+LIMITED|LIMITED|LTD|BANK|SERVICES)\b",
     0, "AN ORGANISATION"),
    ("org", "company size", 2, r"\b\d+[- ](?:person|people|member|employee|staff|seat)s?\b", re.I, "small"),
    ("org", "company size", 2, r"\b(?:family-run|family-owned|one-person|single-branch)\b", re.I, "small"),
    ("org", "organisation type", 1, r"\b(?:firm|company|startup|start-up|organisation|organization|bank|"
                                    r"hospital|school|college|NGO|agency|proprietorship)\b", re.I, None),
    ("age", "age", 2, r"\b\d{1,3}[- ]year[- ]old\b", re.I, "adult"),
    ("age", "age", 2, r"\baged?\s*:?\s*\d{1,3}\b", re.I, "an adult"),
    ("rare", "rare attribute", 3, r"\b(?:only|sole|lone|first|youngest|oldest)\s+(?:[a-z]+\s+){0,2}"
                                  r"(?:woman|women|female|male|man|engineer|employee|doctor|member|person|"
                                  r"candidate|student|director|officer|executive|analyst|manager)\b",
     re.I, lambda t: t.split()[-1]),
    ("rare", "health detail", 3, r"\b(?:pregnan\w*|wheelchair|HIV|cancer|diabetic|disabilit\w+|"
                                 r"visually impaired|hearing impaired|maternity leave)\b", re.I, "a personal circumstance"),
    ("date", "date", 1, r"\b\d{1,2}[/.-]\d{1,2}[/.-](?:\d{4}|\d{2})\b", 0, None),
    ("date", "date", 1, r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?,?\s+\d{4}\b",
     0, None),
]
_COMPILED = [(c, lbl, w, re.compile(p, f), rep) for c, lbl, w, p, f, rep in QI_RULES]
_LEADING_STOP = {"This", "The", "A", "An", "Our", "Your", "Their", "Its", "Each", "Any", "Such", "That",
                 "These", "THIS", "THE", "AND", "OF", "Master", "MASTER", "Dear"}
_BIRTH = re.compile(r"\b(?:born|birth|DOB|d\.o\.b)\b", re.I)
_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
           "October", "November", "December"]


def _generalize_date(t: str, birth: bool) -> str:
    year = re.search(r"(\d{4})\s*$", t)
    if birth and year:
        return f"the {year.group(1)[:3]}0s"
    m = re.match(r"(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})", t)
    if m and 1 <= int(m.group(2)) <= 12:
        return f"{_MONTHS[int(m.group(2)) - 1]} {m.group(3)}"
    m = re.match(r"\d{1,2}(?:st|nd|rd|th)?\s+([A-Za-z]+)\.?,?\s+(\d{4})", t)
    if m:
        return f"{m.group(1)} {m.group(2)}"
    return year.group(1) if year else t


def _explain(labels: list[str], level: str) -> str:
    if level == "low":
        return f"Only one indirect detail ({labels[0]}): low risk on its own."
    joined = " + ".join(labels)
    return joined[:1].upper() + joined[1:] + " could single out one person."


def qi_report(text: str) -> dict:
    from app.summarize import split_sentences

    sentences = []
    for s, e in split_sentences(text):
        sent = text[s:e]
        found = []
        for cat, label, weight, pat, rep in _COMPILED:
            for m in pat.finditer(sent):
                if PLACEHOLDER_RE.search(m.group(0)):
                    continue
                ms, me, mtext = m.start(), m.end(), m.group(0)
                if label == "company name":
                    # "This Master Services ..." is prose, not a company: drop leading stopwords
                    words = mtext.split()
                    while len(words) > 1 and words[0] in _LEADING_STOP:
                        ms = sent.index(words[1], ms + len(words[0]))
                        words = words[1:]
                    if len(words) < 2:
                        continue
                    mtext = sent[ms:me]
                if cat == "date":
                    if _BIRTH.search(sent):
                        cat, label, weight = "age", "date of birth", 2
                    rep_text = _generalize_date(mtext, cat == "age")
                elif callable(rep):
                    rep_text = rep(mtext)
                else:
                    rep_text = rep
                found.append({"start": s + ms, "end": s + me, "text": mtext,
                              "category": cat, "label": label, "weight": weight, "replacement": rep_text})
        # overlapping matches: keep the longest / heaviest
        found.sort(key=lambda f: (-(f["end"] - f["start"]), -f["weight"]))
        terms: list[dict] = []
        for f in found:
            if not any(f["start"] < t["end"] and f["end"] > t["start"] for t in terms):
                terms.append(f)
        terms.sort(key=lambda f: f["start"])
        cats: dict[str, dict] = {}
        for t in terms:
            if t["category"] not in cats or t["weight"] > cats[t["category"]]["weight"]:
                cats[t["category"]] = t
        score = sum(t["weight"] for t in cats.values())
        n = len(cats)
        if n == 0:
            continue
        unique = "rare" in cats and cats["rare"]["label"] == "rare attribute"  # "only/sole/first ..."
        if n >= 3 or score >= 5 or ("rare" in cats and n >= 2) or unique:
            level = "high"
        elif n == 2 or score >= 3:
            level = "medium"
        else:
            level = "low"
        labels = [cats[c]["label"] for c in ("role", "location", "org", "age", "rare", "date") if c in cats]
        sentences.append({
            "start": s, "end": e, "text": sent, "score": score, "level": level,
            "categories": list(cats), "explanation": _explain(labels, level),
            "terms": [{k: t[k] for k in ("start", "end", "text", "category", "label", "replacement")}
                      for t in terms],
        })
    order = {"none": 0, "low": 1, "medium": 2, "high": 3}
    max_level = max((x["level"] for x in sentences), key=order.get, default="none")
    counts = {lvl: sum(1 for x in sentences if x["level"] == lvl) for lvl in ("high", "medium", "low")}
    return {"sentences": sentences, "max_level": max_level, "counts": counts,
            "total_sentences": len(split_sentences(text))}


def generalize(text: str, terms: list[dict]) -> str:
    """Apply replacements (terms carry start/end/replacement; those without one are kept)."""
    out = text
    for t in sorted(terms, key=lambda t: -t["start"]):
        if t.get("replacement"):
            out = out[:t["start"]] + t["replacement"] + out[t["end"]:]
    return out


def apply_generalizations(text: str, pairs: list[dict]) -> str:
    """Global word-boundary replacement of {find, replace} pairs (used on the redacted document)."""
    for p in sorted(pairs, key=lambda p: -len(p.get("find", ""))):
        f, r = p.get("find", ""), p.get("replace")
        if f.strip() and r:
            text = _word_re(f, 0).sub(lambda m: r, text)
    return text
