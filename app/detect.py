"""Hybrid PII detector: regex + validators, optionally fused with GLiNER NER.

Every detector returns spans shaped as::

    {id, start, end, text, type, confidence, source}

``type`` is one of NAME, EMAIL, PHONE, ADDRESS, ID. ``source`` is one of
``regex`` (pattern only), ``regex+validator`` (pattern plus a checksum,
structure or keyword-context check), ``gliner`` or ``gliner+rules``.
"""
from __future__ import annotations

import logging
import re
import threading
from typing import Callable, Iterable, Optional

from app.config import GLINER_MODEL, ORG_ADDRESS_CONFIDENCE

log = logging.getLogger(__name__)

TYPES = ("NAME", "EMAIL", "PHONE", "ADDRESS", "ORG_ADDRESS", "ID")
CONF_VALIDATED = 0.97
CONF_REGEX = 0.80

GLINER_LABELS = ["person", "email", "phone number", "address", "id number"]
GLINER_TO_TYPE = {
    "person": "NAME",
    "email": "EMAIL",
    "phone number": "PHONE",
    "address": "ADDRESS",
    "id number": "ID",
}
GLINER_THRESHOLD = 0.45
CHUNK_WORDS = 300
CHUNK_OVERLAP = 60

# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------
_VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]


def digits_of(s: str) -> str:
    return re.sub(r"\D", "", s)


def verhoeff_valid(num: str) -> bool:
    d = digits_of(num)
    if not d:
        return False
    c = 0
    for i, ch in enumerate(reversed(d)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(ch)]]
    return c == 0


def is_valid_aadhaar(s: str) -> bool:
    d = digits_of(s)
    return len(d) == 12 and d[0] not in "01" and len(set(d)) > 1 and verhoeff_valid(d)


def luhn_valid(num: str) -> bool:
    d = digits_of(num)
    if not d:
        return False
    total = 0
    for i, ch in enumerate(reversed(d)):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def is_valid_card(s: str) -> bool:
    d = digits_of(s)
    # Visa 4, Mastercard 2/5, Amex 3, Discover/RuPay 6, RuPay 8
    return 13 <= len(d) <= 19 and d[0] in "234568" and len(set(d)) > 1 and luhn_valid(d)


# 4th char of a PAN encodes holder type: Person, Company, HUF, Firm, AOP,
# Trust, BOI, Local authority, Artificial juridical person, Government.
_PAN_RE = re.compile(r"[A-Z]{3}[PCHFATBLJG][A-Z]\d{4}[A-Z]")


def is_valid_pan(s: str) -> bool:
    return bool(_PAN_RE.fullmatch(s.strip()))


def is_valid_ifsc(s: str) -> bool:
    return bool(re.fullmatch(r"[A-Z]{4}0[A-Z0-9]{6}", s.strip()))


_EMAIL_STRICT = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9._%+-]{0,62}[A-Za-z0-9])?@"
    r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,24}"
)


def is_valid_email(s: str) -> bool:
    s = s.strip()
    return bool(_EMAIL_STRICT.fullmatch(s)) and ".." not in s


def normalize_email(raw: str) -> Optional[str]:
    """Turn 'riya [at] gmail [dot] com' / 'riya(at)gmail.com' into 'riya@gmail.com'.

    Returns None when the text is not a plausible (possibly obfuscated) email.
    A bare ' at ' with only real dots ('work at google.com') is rejected.
    """
    bracketed = re.search(r"[\[\(\{<]\s*(?:at|dot)\s*[\]\)\}>]", raw, re.I)
    spelled = re.search(r"\s+at\s+", raw, re.I) and re.search(r"\s+dot\s+", raw, re.I)
    if not ("@" in raw or bracketed or spelled):
        return None
    s = re.sub(r"\s*[\[\(\{<]\s*at\s*[\]\)\}>]\s*|\s+at\s+|\s*@\s*", "@", raw, flags=re.I)
    s = re.sub(r"\s*[\[\(\{<]\s*dot\s*[\]\)\}>]\s*|\s+dot\s+", ".", s, flags=re.I)
    s = s.strip()
    return s if is_valid_email(s) else None


def is_valid_phone(s: str) -> bool:
    raw = s.strip()
    d = digits_of(raw)
    if raw.startswith("+"):
        if d.startswith("91"):
            rest = d[2:]
            if rest.startswith("0"):
                rest = rest[1:]
            return len(rest) == 10 and rest[0] != "0"
        return 8 <= len(d) <= 15
    if d.startswith("0"):
        # STD code + subscriber number (11 digits) or 0 + mobile
        return len(d) == 11 and d[1] != "0"
    return len(d) == 10 and d[0] in "6789"


def is_valid_pin(s: str) -> bool:
    d = digits_of(s)
    return len(d) == 6 and d[0] != "0"


# ---------------------------------------------------------------------------
# False-positive guards: text ranges that must never be flagged as PII
# ---------------------------------------------------------------------------
_MONTH = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?"
_GUARD_PATTERNS: list[tuple[str, re.Pattern]] = [
    # dates
    ("date", re.compile(r"(?<![\d/.-])\d{1,2}[/.-]\d{1,2}[/.-](?:\d{4}|\d{2})(?![\d/.-])")),
    ("date", re.compile(r"(?<![\d/.-])\d{4}[/.-]\d{1,2}[/.-]\d{1,2}(?![\d/.-])")),
    ("date", re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH},?\s+\d{{4}}\b")),
    ("date", re.compile(rf"\b{_MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}\b")),
    # times
    ("time", re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AaPp]\.?[Mm]\.?)?(?:\s*(?:hrs|IST))?")),
    ("time", re.compile(r"\b\d{1,2}\s*[AaPp]\.?[Mm]\b\.?")),
    # money
    ("amount", re.compile(r"(?:₹|\bRs\.?|\bINR|\bUSD|\$)\s*\d[\d,]*(?:\.\d+)?(?:\s*/-)?", re.I)),
    ("amount", re.compile(r"\b\d[\d,]*(?:\.\d+)?\s*(?:/-|rupees\b|lakhs?\b|crores?\b)", re.I)),
    # invoice / order / SKU / product and similar business codes
    ("code", re.compile(
        r"\b(?:invoice|inv|order|ord|sku|product|prod|item|po|purchase\s+order|txn|transaction|"
        r"bill|receipt|tracking|awb|shipment|batch|lot|model|part|serial|ticket|incident|"
        r"case|claim|policy|contract|agreement|tender|quotation|quote)\b\.?\s*"
        r"(?:no\.?|number|num|id|code|#)?\s*[:#.\-]?\s*#?\s*"
        r"(?=[A-Z0-9/-]*\d)[A-Z0-9][A-Z0-9/-]*",
        re.I,
    )),
    ("code", re.compile(
        r"\b(?:SKU|INV|ORD|PO|PRD|ITM|TXN|INC|TKT|CLM|BATCH|LOT|CTR|VND|QTN|PH)[-/#]?\d[\w/-]*"
    )),
    ("code", re.compile(
        r"\bref(?:erence)?\.?\s*(?:no\.?|number|#)\s*[:.\-]?\s*(?=[A-Z0-9/-]*\d)[A-Z0-9][A-Z0-9/-]*",
        re.I,
    )),
    # ISBNs
    ("isbn", re.compile(r"\bISBN(?:-1[03])?:?\s*[\dX][\d\s-]{8,16}[\dX]\b", re.I)),
    ("isbn", re.compile(r"\b97[89]-\d{1,5}-\d{1,7}-\d{1,7}-\d\b")),
    # version numbers
    ("version", re.compile(r"\bv(?:ersion)?\.?\s*\d+(?:\.\d+){1,3}\b", re.I)),
    ("version", re.compile(r"\b\d+\.\d+\.\d+(?:\.\d+)?\b")),
]


def protected_ranges(text: str) -> list[tuple[int, int, str]]:
    out = []
    for kind, pat in _GUARD_PATTERNS:
        for m in pat.finditer(text):
            out.append((m.start(), m.end(), kind))
    return sorted(out)


def _protected_fraction(start: int, end: int, ranges: list[tuple[int, int, str]]) -> float:
    if end <= start:
        return 0.0
    covered = [False] * (end - start)
    for s, e, _ in ranges:
        if e <= start or s >= end:
            continue
        for i in range(max(s, start), min(e, end)):
            covered[i - start] = True
    return sum(covered) / (end - start)


def _is_failed_aadhaar_like(span: dict) -> bool:
    """A bare 12-digit number that fails Verhoeff is not an Aadhaar -> not PII."""
    t = span["text"].strip()
    return bool(re.fullmatch(r"\d{4}[ -]?\d{4}[ -]?\d{4}", t)) and not is_valid_aadhaar(t)


# ---------------------------------------------------------------------------
# Rule-based detectors
# ---------------------------------------------------------------------------
def _mk(start: int, end: int, text: str, typ: str, conf: float, source: str, **extra) -> dict:
    d = {"start": start, "end": end, "text": text[start:end], "type": typ,
         "confidence": round(conf, 3), "source": source}
    d.update(extra)
    return d


_NAME_STOPWORDS = {
    "The", "This", "That", "These", "Those", "In", "On", "At", "For", "And", "Of", "To", "By",
    "With", "From", "As", "An", "It", "Is", "We", "Our", "Your", "All", "Any", "Each", "No",
    "Note", "Section", "Annexure", "Schedule", "Clause", "Part", "Table", "Figure", "Box",
    "Ltd", "Pvt", "Inc", "Co", "Road", "Nagar", "Street", "Block", "Main", "Cross",
    "Date", "Place", "Signature", "Name", "Dear", "Regards", "Sincerely", "Director", "Manager",
    # sentence openers that precede an initial ("Later R. Sharma replied")
    "Later", "Also", "Then", "However", "Further", "Furthermore", "Kindly", "Please", "Thus",
    "Hence", "Here", "There", "When", "While", "After", "Before", "Since", "Only", "Both",
    "Contact", "Meanwhile", "Additionally", "Finally", "Today", "Yesterday", "Signed", "Sent",
    "Copy", "Cc", "From", "Attention", "Verified", "Approved", "Reviewed", "Prepared",
}


def _all_stopwords(s: str) -> bool:
    words = [w.strip(".,") for w in s.split()]
    return bool(words) and all(w in _NAME_STOPWORDS for w in words)


def _has_stopword(s: str) -> bool:
    return any(w.strip(".,") in _NAME_STOPWORDS for w in s.split())


class Rule:
    """A regex that yields one span type, optionally gated or graded by a validator.

    mode="require": drop matches that fail the validator.
    mode="grade":   validator-pass -> 0.97, fail -> 0.80.
    mode="context": the pattern itself carries validating context -> 0.97.
    mode="require_plain": drop matches that fail a sanity filter, survivors -> 0.80.
    mode="plain":   pattern only -> 0.80.
    """

    def __init__(self, name: str, typ: str, pattern: str, flags: int = 0,
                 validator: Optional[Callable[[str], bool]] = None,
                 mode: str = "plain", group: int = 0):
        self.name, self.typ = name, typ
        self.pat = re.compile(pattern, flags)
        self.validator, self.mode, self.group = validator, mode, group

    def find(self, text: str) -> Iterable[dict]:
        for m in self.pat.finditer(text):
            s, e = m.span(self.group)
            if s < 0 or e <= s:
                continue
            val = text[s:e]
            # trim trailing punctuation/space
            while e > s and text[e - 1] in " .,;:-\n\t":
                e -= 1
            val = text[s:e]
            if self.mode == "context":
                yield _mk(s, e, text, self.typ, CONF_VALIDATED, "regex+validator", rule=self.name)
                continue
            ok = self.validator(val) if self.validator else False
            if self.mode == "require":
                if ok:
                    yield _mk(s, e, text, self.typ, CONF_VALIDATED, "regex+validator", rule=self.name)
            elif self.mode == "require_plain":
                if ok:
                    yield _mk(s, e, text, self.typ, CONF_REGEX, "regex", rule=self.name)
            elif self.mode == "grade":
                yield _mk(s, e, text, self.typ, CONF_VALIDATED if ok else CONF_REGEX,
                          "regex+validator" if ok else "regex", rule=self.name)
            else:
                yield _mk(s, e, text, self.typ, CONF_REGEX, "regex", rule=self.name)


_UPI_HANDLES = (
    r"ok(?:axis|sbi|hdfcbank|icici)|ybl|ibl|axl|paytm|upi|apl|yapl|rapl|icici|sbi|"
    r"hdfcbank|axisbank|kotak|kmbl|barodampay|aubank|idfcbank|indus|federal|fbl|"
    r"jupiteraxis|pthdfc|ptsbi|ptyes|ptaxis|waicici|wahdfcbank|freecharge|airtel|jio|"
    r"axisb|yesbank|unionbank|cnrb|pnb|boi|citi|hsbc|dbs"
)
_EMAIL_AT = r"(?:\s*[\[\(\{<]\s*at\s*[\]\)\}>]\s*|\s+at\s+|\s*@\s*)"
_EMAIL_DOT = r"(?:\s*[\[\(\{<]\s*dot\s*[\]\)\}>]\s*|\s+dot\s+|\.)"
_EMAIL_TLD = r"(?:com|in|org|net|co|edu|gov|io|info|biz|ac|me|us|uk|ai)"

_NAME_WORD = r"[A-Z][a-z]+(?:-[A-Z][a-z]+)?"
_HONORIFIC = r"(?:Mr|Ms|Mrs|Miss|Mx|Dr|Prof|Shri|Smt|Sri|Kum|Kumari)"
_NAME_LABELS = (
    r"(?:Name|Full\s+Name|Employee\s+Name|Customer\s+Name|Applicant(?:\s+Name)?|"
    r"Account\s+Holder|Nominee|Father'?s\s+Name|Mother'?s\s+Name|Spouse|Signed\s+by|"
    r"Signature|Prepared\s+by|Reported\s+by|Reviewed\s+by|Approved\s+by|Witness|"
    r"Contact\s+Person|Attn|Authori[sz]ed\s+Signatory|Complainant|Employee)"
)

RULES: list[Rule] = [
    # --- EMAIL -----------------------------------------------------------
    Rule("email", "EMAIL", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,24}\b",
         validator=is_valid_email, mode="grade"),
    Rule("email_obfuscated", "EMAIL",
         rf"\b[A-Za-z0-9._%+-]+{_EMAIL_AT}[A-Za-z0-9-]+(?:{_EMAIL_DOT}[A-Za-z0-9-]+)*{_EMAIL_DOT}{_EMAIL_TLD}\b",
         flags=re.I, validator=lambda s: normalize_email(s) is not None, mode="require"),
    # --- ID: UPI (checked before generic handles) ------------------------
    Rule("upi", "ID", rf"\b[A-Za-z0-9._-]{{2,64}}@(?:{_UPI_HANDLES})\b(?!\.?\w)",
         validator=lambda s: True, mode="require"),
    Rule("upi_generic", "ID", r"\b[A-Za-z0-9._-]{2,64}@[a-z]{2,20}\b(?!\.?\w|@)"),
    # --- PHONE -----------------------------------------------------------
    Rule("phone_in_intl", "PHONE",
         r"(?<![\w+])\+91[\s.-]?(?:\(0\)[\s.-]?)?(?:[6-9]\d{4}[\s.-]?\d{5}|[6-9]\d{2}[\s.-]?\d{3}[\s.-]?\d{4}|"
         r"\d{2,4}[\s.-]?\d{3,4}[\s.-]?\d{4})(?!\d)",
         validator=is_valid_phone, mode="grade"),
    Rule("phone_intl", "PHONE",
         r"(?<![\w+])\+(?!91)\d{1,3}[\s.-]?(?:\(\d{1,4}\)[\s.-]?)?\d{2,4}(?:[\s.-]?\d{2,4}){1,3}(?!\d)",
         validator=is_valid_phone, mode="require"),
    Rule("phone_landline", "PHONE",
         r"(?<![\w(+/-])(?:\(0\d{2,4}\)|0\d{2,4})[\s-]?\d{3,4}[\s-]?\d{4}(?![\d/-])",
         validator=is_valid_phone, mode="grade"),
    Rule("phone_mobile", "PHONE",
         r"(?<![\w+/-])(?:[6-9]\d{4}[\s-]?\d{5}|[6-9]\d{2}[\s-]\d{3}[\s-]\d{4})(?![\w/-])",
         validator=is_valid_phone, mode="grade"),
    # --- ID: national / financial ----------------------------------------
    Rule("aadhaar", "ID", r"(?<![\w-])[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}(?![\w-])",
         validator=is_valid_aadhaar, mode="require"),
    Rule("pan", "ID", r"\b[A-Z]{5}\d{4}[A-Z]\b", validator=is_valid_pan, mode="grade"),
    Rule("ifsc", "ID", r"\b[A-Z]{4}0[A-Z0-9]{6}\b", validator=is_valid_ifsc, mode="require"),
    Rule("credit_card", "ID", r"(?<![\d-])\d(?:[ -]?\d){12,18}(?![\d-])",
         validator=is_valid_card, mode="require"),
    Rule("account_number", "ID",
         r"(?i)\b(?:a/c|acct|account)\.?\s*(?:no|number|num)?\.?\s*[:#-]?\s*(\d(?:[ -]?\d){8,17})(?![\d-])",
         mode="context", group=1),
    # --- ID: employee / student / other keyed IDs ------------------------
    Rule("employee_id", "ID", r"\b(?:EMP|EMPL|STF|STU|STD)[-/]?\d{3,8}\b"),
    Rule("usn", "ID", r"\b[1-4][A-Z]{2}\d{2}[A-Z]{2,3}\d{3}\b"),
    Rule("keyed_id", "ID",
         r"(?i)\b(?:employee|emp|staff|student|roll|usn|customer|cust|member|patient|passport|"
         r"voter|epic|driving\s+licen[cs]e|dl|uan|pf|aadhaar|pan|ckyc|kyc)\s*"
         r"(?:id|no|number|code|#)?\.?\s*[:#]\s*((?=[A-Z0-9/-]*\d)[A-Z0-9][A-Z0-9/-]{3,19})",
         mode="context", group=1),
    # --- NAME (fallback-quality; GLiNER does the heavy lifting) -----------
    Rule("name_honorific", "NAME",
         rf"\b{_HONORIFIC}\.?[ ](?:[A-Z]\.[ ]?){{0,2}}{_NAME_WORD}(?:[ ]{_NAME_WORD}){{0,2}}",
         validator=lambda s: not _has_stopword(s.split(None, 1)[-1]), mode="require_plain"),
    Rule("name_initial", "NAME", rf"\b(?:{_NAME_WORD}[ ])?(?:[A-Z]\.[ ]?){{1,2}}[A-Z][a-z]{{2,}}\b",
         validator=lambda s: not _has_stopword(s), mode="require_plain"),
    Rule("name_label", "NAME",
         rf"\b{_NAME_LABELS}[ \t]*[:\-][ \t]*((?:{_HONORIFIC}\.?[ ])?{_NAME_WORD}(?:[ ](?:[A-Z]\.|{_NAME_WORD})){{0,3}})",
         mode="context", group=1),
]

# --- ADDRESS ---------------------------------------------------------------
_STREET_CUE = re.compile(
    r"\b(?:"
    r"\d+(?:st|nd|rd|th)\s+(?:Main|Cross|Block|Stage|Phase|Floor)\b(?:\s+Road)?"
    r"|(?:Sector|Phase|Block|Stage)[\s-]?\d+[A-Z]?\b"
    r"|(?:Flat|Apt\.?|Apartment|House|Plot|Door|Villa|Unit)\s*(?:No\.?\s*)?[#:]?\s*[A-Z]?-?\d+[A-Z]?\b"
    r"|[A-Z][A-Za-z]*\.?\s+(?:Road|Rd\.?|Nagar|Layout|Colony|Marg|Street|St\.|Lane|Enclave|Extension|"
    r"Puram|Halli|Pet|Chowk|Bazaar|Apartments|Residency|Towers|Heights|Society|Complex)\b"
    r")"
)
_PIN = re.compile(r"(?<![\w-])[1-9]\d{2}\s?\d{3}(?![\w-])")
_ADDR_LABEL = re.compile(
    r"(?im)^[ \t]*(?:(?:residential|permanent|current|correspondence|registered|office|"
    r"communication|mailing|postal|home)\s+)?address(?:\s*\(.*?\))?[ \t]*[:\-][ \t]*"
)
_PIN_LABEL = re.compile(r"(?i)\b(?:pin(?:\s*code)?|pincode|postal\s+code)\s*[:\-]?\s*([1-9]\d{2}\s?\d{3})\b")
_BANK_LINE = re.compile(r"(?i)^[ \t]*(?:bank|branch|bank\s+name|branch\s+name)\s*:")
_FIELD_LINE = re.compile(r"^[ \t]*[A-Za-z][A-Za-z ./()'-]{1,40}:\s")
# A run of capitalised/numeric comma-separated tokens following a street cue
_ADDR_TAIL = re.compile(
    r"(?:[ \t]?[,\-][ \t]?|[ \t]{1,2})(?:[A-Z0-9#][\w./#()-]*)(?![\w./#()-]*:)(?:[ \t]{1,2}(?:[A-Z0-9#][\w./#()-]*(?![\w./#()-]*:)|of|near|opp\.?))*"
)
_ADDR_HEAD = re.compile(
    r"(?:(?:[A-Z][a-z]+\s+)?(?:No\.?|#)\s*\d+[A-Z]?(?:/\d+)?,?\s*|\d+[A-Z]?(?:/\d+)?,\s*)?"
    r"(?:[A-Z][\w.'-]*,?[ \t]+){0,3}$"
)


def _address_spans(text: str, guards: list[tuple[int, int, str]]) -> list[dict]:
    spans: list[dict] = []

    def line_bounds(pos: int) -> tuple[int, int]:
        ls = text.rfind("\n", 0, pos) + 1
        le = text.find("\n", pos)
        return ls, (len(text) if le == -1 else le)

    def block_end(pos: int) -> int:
        """End of the address block starting at pos: stops at blank line or a new field."""
        end = pos
        cur = pos
        lines = 0
        while cur < len(text) and lines < 4:
            ls, le = line_bounds(cur)
            line = text[ls:le]
            if lines > 0 and (not line.strip() or _FIELD_LINE.match(line)):
                break
            end = le
            lines += 1
            if _PIN.search(line):
                break
            cur = le + 1
        return end

    # 1) explicit "Address:" fields
    for m in _ADDR_LABEL.finditer(text):
        s = m.end()
        e = block_end(s)
        val = text[s:e].rstrip(" .,;\n")
        if len(val.strip()) < 6:
            continue
        e = s + len(val)
        has_pin = bool(_PIN.search(val))
        spans.append(_mk(s, e, text, "ADDRESS", CONF_VALIDATED if has_pin else CONF_REGEX,
                         "regex+validator" if has_pin else "regex", rule="address_label"))

    # 2) street-keyword cues, extended back over the house number and forward to the PIN
    for m in _STREET_CUE.finditer(text):
        ls, le = line_bounds(m.start())
        if _BANK_LINE.match(text[ls:le]):
            continue  # bank branch names are not personal addresses
        head = _ADDR_HEAD.search(text[ls:m.start()])
        s = ls + head.start() if head and head.group(0).strip() else m.start()
        # strip leading prose words that the head may have swallowed
        while s < m.start() and text[s] in " ,\t":
            s += 1
        # forward: PIN within the same block?
        window_end = block_end(m.start())
        pin = _PIN.search(text, m.end(), window_end)
        if pin and _protected_fraction(pin.start(), pin.end(), guards) < 0.5:
            e, conf, src = pin.end(), CONF_VALIDATED, "regex+validator"
        else:
            tail = _ADDR_TAIL.match(text, m.end())
            e = tail.end() if tail else m.end()
            # allow the tail to continue across commas
            while True:
                nxt = _ADDR_TAIL.match(text, e)
                if not nxt or nxt.end() == e or "\n" in text[e:nxt.end()]:
                    break
                e = nxt.end()
            conf, src = CONF_REGEX, "regex"
        while e > s and text[e - 1] in " .,;:-\n\t":
            e -= 1
        spans.append(_mk(s, e, text, "ADDRESS", conf, src, rule="address_street"))

    # 3) PIN with an explicit "PIN:" label
    for m in _PIN_LABEL.finditer(text):
        s, e = m.span(1)
        spans.append(_mk(s, e, text, "ADDRESS", CONF_VALIDATED, "regex+validator", rule="pin_label"))

    return spans


def rule_spans(text: str, guards: Optional[list] = None) -> list[dict]:
    guards = protected_ranges(text) if guards is None else guards
    out: list[dict] = []
    for r in RULES:
        out.extend(r.find(text))
    out.extend(_address_spans(text, guards))
    return out


# ---------------------------------------------------------------------------
# GLiNER (lazy, optional)
# ---------------------------------------------------------------------------
_model = None
_model_lock = threading.Lock()
_model_state = {"status": "not_loaded", "error": None, "model": GLINER_MODEL}


def load_model(force: bool = False):
    """Load GLiNER once. Returns the model or None (rules-only fallback)."""
    global _model
    if _model is not None:
        return _model
    if _model_state["status"] == "unavailable" and not force:
        return None
    with _model_lock:
        if _model is not None:
            return _model
        _model_state["status"] = "loading"
        try:
            from gliner import GLiNER  # heavy import, keep lazy

            _model = GLiNER.from_pretrained(GLINER_MODEL)
            _model_state.update(status="loaded", error=None)
            log.info("GLiNER model %s loaded", GLINER_MODEL)
        except Exception as exc:  # missing package, no cached weights, offline, ...
            _model = None
            _model_state.update(status="unavailable", error=f"{type(exc).__name__}: {exc}")
            log.warning("GLiNER unavailable, falling back to rules: %s", exc)
    return _model


def detector_status() -> dict:
    st = dict(_model_state)
    st["mode"] = "hybrid (rules + GLiNER)" if st["status"] == "loaded" else (
        "rules-only (GLiNER model unavailable)" if st["status"] == "unavailable" else "rules (GLiNER loads on first use)")
    return st


def _chunks(text: str, size: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP):
    words = [m.span() for m in re.finditer(r"\S+", text)]
    if not words:
        return
    step = max(1, size - overlap)
    i = 0
    while True:
        window = words[i:i + size]
        yield window[0][0], window[-1][1]
        if i + size >= len(words):
            break
        i += step


def _gliner_sane(span: dict) -> bool:
    t, typ = span["text"].strip(), span["type"]
    if not t:
        return False
    if typ == "NAME":
        return t[0].isupper() and not re.search(r"\d|@", t) and len(t) >= 2 and not _all_stopwords(t)
    if typ == "EMAIL":
        return "@" in t or normalize_email(t) is not None
    if typ == "PHONE":
        return len(digits_of(t)) >= 8
    if typ == "ID":
        return bool(re.search(r"\d", t)) and len(re.sub(r"\W", "", t)) >= 4
    return len(t) >= 6  # ADDRESS


def gliner_spans(text: str, threshold: float = GLINER_THRESHOLD) -> list[dict]:
    model = load_model()
    if model is None:
        return []
    out = []
    for cs, ce in _chunks(text):
        chunk = text[cs:ce]
        try:
            ents = model.predict_entities(chunk, GLINER_LABELS, threshold=threshold)
        except Exception as exc:
            log.warning("GLiNER prediction failed on chunk: %s", exc)
            continue
        for ent in ents:
            typ = GLINER_TO_TYPE.get(ent["label"])
            if not typ:
                continue
            s, e = cs + ent["start"], cs + ent["end"]
            sp = _mk(s, e, text, typ, float(ent["score"]), "gliner")
            if _gliner_sane(sp):
                out.append(sp)
    return out


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------
_TYPE_PRIORITY = {"EMAIL": 5, "ID": 4, "PHONE": 3, "ADDRESS": 2, "ORG_ADDRESS": 2, "NAME": 1}
_STRUCTURED = {"EMAIL", "PHONE", "ID"}


def _overlap(a: dict, b: dict) -> int:
    return max(0, min(a["end"], b["end"]) - max(a["start"], b["start"]))


def _boost(gl: list[dict], rules: list[dict]) -> list[dict]:
    """GLiNER spans agreeing with a same-type rule span get a combined confidence."""
    fused = []
    used_rules = set()
    for g in gl:
        best = None
        for i, r in enumerate(rules):
            if r["type"] != g["type"]:
                continue
            ov = _overlap(g, r)
            if ov and ov >= 0.5 * min(g["end"] - g["start"], r["end"] - r["start"]):
                if best is None or r["confidence"] > rules[best]["confidence"]:
                    best = i
        if best is None:
            fused.append(g)
            continue
        r = rules[best]
        used_rules.add(best)
        conf = min(0.99, 1 - (1 - g["confidence"]) * (1 - r["confidence"]))
        if g["type"] in _STRUCTURED:
            s, e = r["start"], r["end"]
        else:
            s, e = min(g["start"], r["start"]), max(g["end"], r["end"])
        fused.append({"start": s, "end": e, "type": g["type"], "confidence": round(conf, 3),
                      "source": "gliner+rules", "rule": r.get("rule")})
    fused.extend(r for i, r in enumerate(rules) if i not in used_rules)
    return fused


def merge_spans(spans: list[dict], text: str) -> list[dict]:
    """Resolve overlaps: highest confidence wins; same-type overlaps are unioned;
    lower-confidence spans of another type are trimmed to their non-overlapping part."""
    order = sorted(spans, key=lambda s: (-s["confidence"], -_TYPE_PRIORITY[s["type"]],
                                         -(s["end"] - s["start"]), s["start"]))
    accepted: list[dict] = []
    for sp in order:
        pieces = [dict(sp)]
        for acc in accepted:
            nxt = []
            for p in pieces:
                if not _overlap(p, acc):
                    nxt.append(p)
                    continue
                if p["type"] == acc["type"]:
                    acc["start"] = min(acc["start"], p["start"])
                    acc["end"] = max(acc["end"], p["end"])
                    continue
                # different type: keep remainders for free-text types only
                if p["type"] in ("NAME", "ADDRESS"):
                    for s, e in ((p["start"], acc["start"]), (acc["end"], p["end"])):
                        seg = text[s:e] if e > s else ""
                        if len(re.sub(r"[\W_]", "", seg)) >= 4:
                            q = dict(p)
                            q["start"], q["end"] = s, e
                            nxt.append(q)
            pieces = nxt
        for p in pieces:
            # absorb into an existing same-type span the piece now touches
            accepted.append(p)
    # unions may now overlap each other; fold same-type overlaps
    accepted.sort(key=lambda s: s["start"])
    folded: list[dict] = []
    for sp in accepted:
        if folded and folded[-1]["type"] == sp["type"] and sp["start"] <= folded[-1]["end"]:
            last = folded[-1]
            last["end"] = max(last["end"], sp["end"])
            last["confidence"] = max(last["confidence"], sp["confidence"])
            continue
        if folded and sp["start"] < folded[-1]["end"]:
            # residual cross-type overlap: keep the stronger, trim the weaker
            last = folded[-1]
            if sp["confidence"] > last["confidence"]:
                last["end"] = sp["start"]
            else:
                sp["start"] = last["end"]
            if sp["end"] <= sp["start"]:
                continue
        folded.append(sp)
    return folded


def _tidy(span: dict, text: str) -> Optional[dict]:
    s, e = span["start"], span["end"]
    while s < e and text[s] in " \t\n,;:.-":
        s += 1
    while e > s and text[e - 1] in " \t\n,;:-(":
        e -= 1
    # keep a trailing '.' only for initials like "R."
    if e > s and text[e - 1] == "." and not re.search(r"\b[A-Z]\.$", text[s:e]):
        e -= 1
    if e <= s:
        return None
    span = dict(span, start=s, end=e, text=text[s:e])
    return span


# ---------------------------------------------------------------------------
# Organisation vs personal addresses
# ---------------------------------------------------------------------------
_ORG_CONTEXT = re.compile(
    r"(?i)\b(?:registered|regd\.?|corporate|head|branch|site|works|factory)\s+office\b"
    r"|\b(?:its|our|the\s+company'?s|company)\s+(?:registered\s+)?(?:office|premises|address)\b"
    r"|\boffice\s+(?:at|address)\b|\bbranch\b|\bwarehouse\b|\bfactory\b|\bcampus\b|\bheadquarter"
)
_ORG_NAME = re.compile(
    r"\b(?:Pvt|Private|Limited|Ltd|LLP|Inc|Corp|Corporation|Company|Bank|Technologies|Solutions|"
    r"Services|Softlabs|Systems|Logistics|Industries|Enterprises|Hospital|University|Institute|TPA)\b",
    re.I,
)
_PERSONAL_CONTEXT = re.compile(
    r"(?i)\b(?:residential|residence|resides?|residing|home|permanent|current|personal|lives|living|"
    r"staying|correspondence|his|her|their)\b"
)
_ORG_LABEL = re.compile(r"(?i)\b(?:registered|regd\.?|office|business|company|branch|corporate|billing)\s+address\s*[:\-]")


def _is_org_address(span: dict, text: str) -> bool:
    """Letterhead / registered-office / company addresses (not personal data)."""
    ls = text.rfind("\n", 0, span["start"]) + 1
    before = text[ls:span["start"]]
    if not before.strip() and ls > 0:  # address starts the line: look at the previous line
        pls = text.rfind("\n", 0, ls - 1) + 1
        prev = text[pls:ls - 1]
        if _ORG_LABEL.search(prev):
            return True
        # letterhead: an ALL-CAPS company line directly above
        letters = re.sub(r"[^A-Za-z]", "", prev)
        if letters and letters.isupper() and len(prev.split()) >= 2 and _ORG_NAME.search(prev):
            return True
        if _PERSONAL_CONTEXT.search(prev):
            return False
        before = prev
    if _ORG_LABEL.search(before):
        return True
    org = [m.end() for m in _ORG_CONTEXT.finditer(before)]
    per = [m.end() for m in _PERSONAL_CONTEXT.finditer(before)]
    if org and (not per or org[-1] > per[-1]):
        return True
    # address within the first lines of the document right under a company header
    if span["start"] < 250 and _ORG_NAME.search(text[:span["start"]]) and not per:
        head = text[:span["start"]]
        return head.count("\n") <= 3
    return False


def classify_addresses(spans: list[dict], text: str) -> list[dict]:
    for sp in spans:
        if sp["type"] == "ADDRESS" and _is_org_address(sp, text):
            sp["type"] = "ORG_ADDRESS"
            sp["confidence"] = ORG_ADDRESS_CONFIDENCE
    return spans


def _split_names(spans: list[dict], text: str) -> list[dict]:
    """A NAME never spans a line break or a column gap (e.g. two signature blocks side by side)."""
    out = []
    for sp in spans:
        if sp["type"] != "NAME" or not re.search(r"\n|\t| {3,}", sp["text"]):
            out.append(sp)
            continue
        pos = sp["start"]
        for part in re.split(r"(\n|\t| {3,})", text[sp["start"]:sp["end"]]):
            if part.strip() and not re.fullmatch(r"\n|\t| {3,}", part) and not _all_stopwords(part):
                q = dict(sp, start=pos, end=pos + len(part))
                q["text"] = text[q["start"]:q["end"]]
                out.append(q)
            pos += len(part)
    return out


def detect(text: str, use_ner: bool = True, threshold: float = GLINER_THRESHOLD) -> list[dict]:
    """Detect PII spans in ``text``. Falls back to rules if GLiNER is unavailable."""
    if not text or not text.strip():
        return []
    guards = protected_ranges(text)
    rules = rule_spans(text, guards)
    gl = gliner_spans(text, threshold) if use_ner else []
    candidates = _boost(gl, rules) if gl else rules

    kept = []
    for sp in candidates:
        sp = _tidy(sp, text)
        if sp is None:
            continue
        if _protected_fraction(sp["start"], sp["end"], guards) >= 0.5:
            continue
        if sp.get("rule") != "account_number" and _is_failed_aadhaar_like(sp):
            continue
        kept.append(sp)

    merged = classify_addresses(merge_spans(_split_names(kept, text), text), text)
    result = []
    for i, sp in enumerate(m for m in (_tidy(s, text) for s in merged) if m):
        result.append({
            "id": f"s{i + 1}",
            "start": sp["start"],
            "end": sp["end"],
            "text": sp["text"],
            "type": sp["type"],
            "confidence": round(float(sp["confidence"]), 3),
            "source": sp["source"],
        })
    return result
