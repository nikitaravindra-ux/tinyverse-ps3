"""Generate 200 synthetic Indian documents with gold spans and hard negatives.

    .venv/bin/python eval/make_indian_set.py

Fixed seed, so the output (eval/data/indian_synthetic.jsonl) is reproducible.
Each row: {id, template, text, spans:[{start,end,type,person?}], negatives:[{start,end,kind}]}
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from faker import Faker

SEED = 20251123
N_DOCS = 200
OUT = Path(__file__).resolve().parent / "data" / "indian_synthetic.jsonl"

rng = random.Random(SEED)
Faker.seed(SEED)
fake = Faker("en_IN")

# ---------------------------------------------------------------------------
# Checksums (self-contained so the generator does not depend on the app)
# ---------------------------------------------------------------------------
_D = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5], [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
      [3, 4, 0, 1, 2, 8, 9, 5, 6, 7], [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
      [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3], [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
      [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
_P = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4], [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
      [8, 9, 1, 6, 0, 4, 3, 5, 2, 7], [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
      [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]]
_INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def verhoeff_ok(num: str) -> bool:
    c = 0
    for i, ch in enumerate(reversed(num)):
        c = _D[c][_P[i % 8][int(ch)]]
    return c == 0


def verhoeff_digit(num: str) -> str:
    c = 0
    for i, ch in enumerate(reversed(num)):
        c = _D[c][_P[(i + 1) % 8][int(ch)]]
    return str(_INV[c])


def aadhaar() -> str:
    body = str(rng.randint(2, 9)) + "".join(str(rng.randint(0, 9)) for _ in range(10))
    n = body + verhoeff_digit(body)
    assert verhoeff_ok(n)
    return f"{n[:4]} {n[4:8]} {n[8:]}" if rng.random() < 0.7 else n


def bad_twelve() -> str:
    while True:
        n = str(rng.randint(2, 9)) + "".join(str(rng.randint(0, 9)) for _ in range(11))
        if not verhoeff_ok(n):
            return n


def pan(surname: str) -> str:
    L = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return "".join(rng.choice(L) for _ in range(3)) + "P" + surname[0].upper() + \
        f"{rng.randint(0, 9999):04d}" + rng.choice(L)


def ifsc() -> str:
    return rng.choice(["HDFC", "ICIC", "SBIN", "UTIB", "KKBK", "PUNB", "BARB"]) + "0" + f"{rng.randint(0, 999999):06d}"


def upi(first: str) -> str:
    return f"{first.lower()}{rng.randint(1, 99)}@{rng.choice(['okaxis', 'oksbi', 'okhdfcbank', 'ybl', 'paytm', 'okicici'])}"


def phone() -> str:
    d = str(rng.randint(6, 9)) + "".join(str(rng.randint(0, 9)) for _ in range(9))
    return rng.choice([f"+91 {d[:5]} {d[5:]}", f"+91-{d}", f"{d[:5]} {d[5:]}", d])


def address() -> str:
    return fake.address().replace("\n", ", ")


# hard negatives
def sku() -> str:
    return f"SKU-{rng.randint(10000, 99999)}-{rng.choice('ABCDEFGH')}"


def invoice() -> str:
    return f"INV-{rng.choice([2024, 2025])}-{rng.randint(1000, 9999)}"


def date() -> str:
    d, m, y = rng.randint(1, 28), rng.randint(1, 12), rng.choice([2024, 2025])
    return rng.choice([f"{d:02d}/{m:02d}/{y}", f"{d:02d}-{m:02d}-{y}", f"{d} March {y}"])


def isbn() -> str:
    return f"978-81-{rng.randint(100, 999)}-{rng.randint(1000, 9999)}-{rng.randint(0, 9)}"


def amount() -> str:
    v = rng.randint(1, 99) * 1000 + rng.choice([0, 500])
    s = f"{v:,}"
    return rng.choice([f"Rs. {s}", f"₹ {s}", f"Rs {s}/-", f"INR {s}"])


# ---------------------------------------------------------------------------
class Doc:
    def __init__(self):
        self.parts: list[str] = []
        self.len = 0
        self.spans: list[dict] = []
        self.negs: list[dict] = []

    def t(self, s: str):
        self.parts.append(s)
        self.len += len(s)
        return self

    def pii(self, s: str, typ: str, person: int | None = None):
        sp = {"start": self.len, "end": self.len + len(s), "type": typ}
        if person is not None:
            sp["person"] = person
        self.spans.append(sp)
        return self.t(s)

    def neg(self, s: str, kind: str):
        self.negs.append({"start": self.len, "end": self.len + len(s), "kind": kind})
        return self.t(s)

    @property
    def text(self) -> str:
        return "".join(self.parts)


def person():
    female = rng.random() < 0.5
    first = fake.first_name_female() if female else fake.first_name_male()
    last = fake.last_name()
    return {"first": first, "last": last, "hon": "Ms." if female else "Mr.", "full": f"{first} {last}"}


def hr_letter(d: Doc):
    p, mgr = person(), person()
    co = fake.company()
    d.t(f"{co.upper()}\nRef No: HR/{rng.randint(100, 999)}/2025        Date: ").neg(date(), "date").t("\n\nTo,\n")
    d.pii(f"{p['hon']} {p['full']}", "NAME", 0).t("\n").pii(address(), "ADDRESS").t("\n\nDear ")
    d.pii(f"{p['hon']} {p['last']}", "NAME", 0).t(",\n\nWe are pleased to offer you a role reporting to ")
    d.pii(mgr["full"], "NAME", 1).t(". Your annual CTC will be ").neg(amount(), "amount")
    d.t(", credited to A/c No. ").pii("".join(str(rng.randint(0, 9)) for _ in range(rng.randint(11, 14))), "ID")
    d.t(", IFSC ").pii(ifsc(), "ID").t(".\nPlease bring your PAN ").pii(pan(p["last"]), "ID")
    d.t(" and Aadhaar ").pii(aadhaar(), "ID").t(" for verification.\n\n")
    d.pii(p["first"], "NAME", 0).t(", your laptop (").neg(sku(), "sku").t(") will be issued on joining. ")
    d.pii(f"{p['first'][0]}. {p['last']}", "NAME", 0).t("'s references were verified. For queries call ")
    d.pii(phone(), "PHONE").t(" or write to ").pii(fake.email(), "EMAIL").t(".\n\nRegards,\n").pii(mgr["full"], "NAME", 1)


def kyc_form(d: Doc):
    p = person()
    d.t("KYC UPDATE FORM            Form version v").neg(f"{rng.randint(1, 4)}.{rng.randint(0, 9)}.{rng.randint(0, 9)}", "version")
    d.t("\nFull Name: ").pii(p["full"], "NAME", 0).t("\nMobile: ").pii(phone(), "PHONE")
    d.t("\nEmail: ").pii(fake.email(), "EMAIL").t("\nResidential Address: ").pii(address(), "ADDRESS")
    d.t("\nAadhaar Number: ").pii(aadhaar(), "ID").t("\nPAN: ").pii(pan(p["last"]), "ID")
    d.t("\nUPI ID: ").pii(upi(p["first"]), "ID").t("\nIFSC: ").pii(ifsc(), "ID")
    d.t("\n\nFOR OFFICE USE\nApplication received on ").neg(date(), "date").t(", fee ").neg(amount(), "amount")
    d.t(" (receipt no. ").neg(bad_twelve(), "invalid_aadhaar").t(").\nLegacy customer key ").neg(bad_twelve(), "invalid_aadhaar")
    d.t(" migrated.\nSignature of applicant: ").pii(f"{p['first'][0]}. {p['last']}", "NAME", 0)


def incident(d: Doc):
    p, lead = person(), person()
    d.t("INCIDENT NOTE\nLogged on ").neg(date(), "date").t(" against order #").neg(bad_twelve(), "invalid_aadhaar")
    d.t(".\n\nA spreadsheet listing ").neg(sku(), "sku").t(" and invoice ").neg(invoice(), "invoice")
    d.t(" for ").neg(amount(), "amount").t(" also exposed the details of a contractor, ")
    d.pii(f"{p['hon']} {p['full']}", "NAME", 0).t(": PAN ").pii(pan(p["last"]), "ID").t(", mobile ")
    d.pii(phone(), "PHONE").t(" and address ").pii(address(), "ADDRESS").t(".\n")
    d.pii(f"{p['hon']} {p['last']}", "NAME", 0).t(" was informed by ").pii(lead["full"], "NAME", 1)
    d.t(", who can be reached at ").pii(fake.email(), "EMAIL").t(". Reference manual ISBN ").neg(isbn(), "isbn").t(".")


def invoice_mail(d: Doc):
    p = person()
    d.t("Subject: Payment for ").neg(invoice(), "invoice").t("\n\nHi ").pii(p["first"], "NAME", 0)
    d.t(",\n\nThe payment of ").neg(amount(), "amount").t(" for ").neg(sku(), "sku").t(" was received on ")
    d.neg(date(), "date").t(". The refund for order #").neg(bad_twelve(), "invalid_aadhaar")
    d.t(" goes to your UPI ").pii(upi(p["first"]), "ID").t(".\nCatalogue reference: ISBN ").neg(isbn(), "isbn")
    d.t(".\n\nThanks,\n").pii(p["full"], "NAME", 0).t(" (accounts)\n").pii(phone(), "PHONE")


TEMPLATES = [("hr_letter", hr_letter), ("kyc_form", kyc_form), ("incident", incident), ("invoice_mail", invoice_mail)]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for i in range(N_DOCS):
            name, fn = TEMPLATES[i % len(TEMPLATES)]
            d = Doc()
            fn(d)
            text = d.text
            for sp in d.spans + d.negs:
                assert text[sp["start"]:sp["end"]].strip(), sp
            fh.write(json.dumps({"id": f"in-{i:03d}", "template": name, "text": text, "spans": d.spans,
                                 "negatives": d.negs}, ensure_ascii=False) + "\n")
    print(f"wrote {N_DOCS} docs -> {OUT}")


if __name__ == "__main__":
    main()
