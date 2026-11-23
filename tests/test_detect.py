"""Tests for validators, rule detection and false-positive guards (rules only, no model)."""
from pathlib import Path

import pytest

from app.detect import (
    detect,
    is_valid_aadhaar,
    is_valid_card,
    is_valid_email,
    is_valid_ifsc,
    is_valid_pan,
    is_valid_phone,
    luhn_valid,
    normalize_email,
    verhoeff_valid,
)

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


def found(text):
    return {(s["text"], s["type"]) for s in detect(text, use_ner=False)}


def texts(text):
    return [s["text"] for s in detect(text, use_ner=False)]


# ---------------------------------------------------------------- validators
class TestVerhoeff:
    @pytest.mark.parametrize("n", ["499118665246", "234567890124", "491837264503", "2345 6789 0124"])
    def test_valid(self, n):
        assert verhoeff_valid(n)
        assert is_valid_aadhaar(n)

    @pytest.mark.parametrize("n", ["448812339021", "234567890123", "987654321098", "499118665247"])
    def test_invalid(self, n):
        assert not verhoeff_valid(n)
        assert not is_valid_aadhaar(n)

    def test_aadhaar_cannot_start_with_0_or_1(self):
        assert not is_valid_aadhaar("123456789012")
        assert not is_valid_aadhaar("012345678901")

    def test_wrong_length(self):
        assert not is_valid_aadhaar("23456789012")


class TestLuhn:
    @pytest.mark.parametrize("n", ["4111 1111 1111 1111", "4532015112830366", "5520-0000-0000-0000",
                                   "6081001234567896", "378282246310005"])
    def test_valid_cards(self, n):
        assert luhn_valid(n)
        assert is_valid_card(n)

    @pytest.mark.parametrize("n", ["4111 1111 1111 1112", "4532015112830367", "1234567890123456"])
    def test_invalid_cards(self, n):
        assert not is_valid_card(n)

    def test_too_short(self):
        assert not is_valid_card("4242")


class TestStructuredIds:
    @pytest.mark.parametrize("p", ["BQRPS4821K", "AAACB1234F", "AJKPI7765M"])
    def test_pan_valid(self, p):
        assert is_valid_pan(p)

    @pytest.mark.parametrize("p", ["BQRXS4821K", "BQRP4821K", "bqrps4821k", "BQRPS48211"])
    def test_pan_invalid(self, p):
        assert not is_valid_pan(p)

    @pytest.mark.parametrize("c", ["HDFC0001234", "SBIN0003456", "KCBL0000457"])
    def test_ifsc_valid(self, c):
        assert is_valid_ifsc(c)

    @pytest.mark.parametrize("c", ["HDFC1001234", "HDF00001234", "HDFC000123"])
    def test_ifsc_invalid(self, c):
        assert not is_valid_ifsc(c)


class TestEmail:
    @pytest.mark.parametrize("e", ["riya@gmail.com", "a.b+c@sub.example.co.in"])
    def test_valid(self, e):
        assert is_valid_email(e)

    @pytest.mark.parametrize("e", ["riya@gmail", "riya..x@gmail.com", "@gmail.com"])
    def test_invalid(self, e):
        assert not is_valid_email(e)

    @pytest.mark.parametrize("raw", ["riya [at] gmail [dot] com", "riya(at)gmail.com",
                                     "riya {at} gmail {dot} com", "riya at gmail dot com"])
    def test_obfuscated_normalises(self, raw):
        assert normalize_email(raw) == "riya@gmail.com"

    def test_bare_at_with_real_dot_rejected(self):
        assert normalize_email("work at google.com") is None


class TestPhoneValidator:
    @pytest.mark.parametrize("p", ["+91 98450 12345", "+91-9845012345", "9876543210", "080-4123 5678",
                                   "(022) 2345 6789", "+44 20 7946 0958", "+91 80 2345 6789"])
    def test_valid(self, p):
        assert is_valid_phone(p)

    @pytest.mark.parametrize("p", ["5876543210", "+91 12345", "0000000000", "080-123"])
    def test_invalid(self, p):
        assert not is_valid_phone(p)


# ---------------------------------------------------------------- detection
class TestDetection:
    def test_emails(self):
        t = "Mail riya@gmail.com or riya [at] gmail [dot] com or riya(at)gmail.com today."
        f = found(t)
        assert ("riya@gmail.com", "EMAIL") in f
        assert ("riya [at] gmail [dot] com", "EMAIL") in f
        assert ("riya(at)gmail.com", "EMAIL") in f

    @pytest.mark.parametrize("phone", ["+91 98450 12345", "080-2345 6789", "(022) 2345 6789",
                                       "9876543210", "+44 20 7946 0958"])
    def test_phones(self, phone):
        assert (phone, "PHONE") in found(f"Call me on {phone} after lunch.")

    def test_mobile_must_start_6_to_9(self):
        assert not any(t == "5876543210" for t in texts("Ref 5876543210 here"))

    def test_valid_aadhaar_flagged(self):
        assert ("2345 6789 0124", "ID") in found("Aadhaar 2345 6789 0124 on file")

    def test_pan_ifsc_upi_card(self):
        t = ("PAN BQRPS4821K, IFSC HDFC0001234, UPI riya@okaxis, card 4111 1111 1111 1111.")
        f = found(t)
        for v in ["BQRPS4821K", "HDFC0001234", "riya@okaxis", "4111 1111 1111 1111"]:
            assert (v, "ID") in f, v

    def test_upi_is_not_email(self):
        types = {s["text"]: s["type"] for s in detect("Pay farhanq@ybl now", use_ner=False)}
        assert types.get("farhanq@ybl") == "ID"

    def test_employee_and_student_ids(self):
        f = found("Employee EMP-10293 and student 1MS24CI079 attended.")
        assert ("EMP-10293", "ID") in f
        assert ("1MS24CI079", "ID") in f

    def test_account_number_by_keyword(self):
        assert ("50100234567812", "ID") in found("Credit to A/c No. 50100234567812 by Friday.")
        assert ("001234500067891", "ID") in found("Account Number: 001234500067891")

    def test_address_with_pin(self):
        t = "She lives at Flat 302, Lakeview Residency, 5th Cross, HSR Layout, Bengaluru 560102."
        addr = [s for s in detect(t, use_ner=False) if s["type"] == "ADDRESS"]
        assert addr and addr[0]["text"].startswith("Flat 302") and addr[0]["text"].endswith("560102")
        assert addr[0]["confidence"] == 0.97

    def test_address_label_field(self):
        t = "Residential Address: No. 17, 2nd Main Road, Jayanagar, Bengaluru 560011\nPAN: BQRPS4821K"
        addr = [s["text"] for s in detect(t, use_ner=False) if s["type"] == "ADDRESS"]
        assert addr == ["No. 17, 2nd Main Road, Jayanagar, Bengaluru 560011"]

    def test_names_by_rules(self):
        f = found("Dear Ms. Sharma, R. Sharma signed.\nName: Richa Sharma")
        assert ("Ms. Sharma", "NAME") in f
        assert ("R. Sharma", "NAME") in f
        assert ("Richa Sharma", "NAME") in f

    def test_span_shape(self):
        spans = detect("Email riya@gmail.com", use_ner=False)
        assert set(spans[0]) == {"id", "start", "end", "text", "type", "confidence", "source"}
        s = spans[0]
        assert "Email riya@gmail.com"[s["start"]:s["end"]] == s["text"]

    def test_validated_vs_plain_confidence(self):
        by_text = {s["text"]: s for s in detect("PAN BQRPS4821K; emp EMP-10293", use_ner=False)}
        assert by_text["BQRPS4821K"]["confidence"] == 0.97
        assert by_text["BQRPS4821K"]["source"] == "regex+validator"
        assert by_text["EMP-10293"]["confidence"] == 0.8
        assert by_text["EMP-10293"]["source"] == "regex"


# ---------------------------------------------------------------- false-positive guards
class TestFalsePositiveGuards:
    @pytest.mark.parametrize("text", [
        "Date: 12/03/2025",
        "Signed on 12-03-2025 and 2025-03-12",
        "Meeting on 12 March 2025 at 10:30 AM",
        "Starts at 14:20 IST",
        "Amount payable: ₹ 1,25,000",
        "CTC Rs. 18,50,000/- per annum",
        "Fee Rs 250 paid",
        "Laptop SKU-88213-B issued",
        "Please quote order #448812339021",
        "Invoice INV-2025-7781 attached",
        "Product code: PRD-99812",
        "ISBN 978-81-203-4567-8",
        "ISBN-13: 9788120345678",
        "Upgraded to v2.4.1 and version 3.1.4",
        "Legacy ledger key 234567890123 retained",
        "Old reference 987654321098 migrated",
        "Receipt no. 448812339021",
    ])
    def test_not_flagged(self, text):
        assert detect(text, use_ner=False) == [], detect(text, use_ner=False)

    def test_12_digit_failing_verhoeff_never_pii(self):
        assert texts("number 448812339021 here") == []

    def test_guard_does_not_hide_real_pii_nearby(self):
        t = "On 12/03/2025 Richa (riya@gmail.com, +91 98450 12345) paid ₹ 5,000 for SKU-88213-B."
        f = found(t)
        assert ("riya@gmail.com", "EMAIL") in f
        assert ("+91 98450 12345", "PHONE") in f
        assert not any("12/03/2025" in x or "SKU" in x or "5,000" in x for x, _ in f)


# ---------------------------------------------------------------- samples
@pytest.mark.parametrize("path", sorted(SAMPLES.glob("*.txt")), ids=lambda p: p.name)
def test_samples_have_no_known_false_positives(path):
    text = path.read_text(encoding="utf-8")
    flagged = " | ".join(texts(text))
    for bad in ["448812339021", "SKU-88213-B", "12/03/2025", "234567890123", "987654321098",
                "978-81-203-4567-8", "INV-2025", "v2.4.1", "18,50,000"]:
        assert bad not in flagged, f"{bad} flagged in {path.name}"


def test_samples_detect_core_pii():
    hr = (SAMPLES / "hr_offer_letter.txt").read_text(encoding="utf-8")
    flagged = {s["text"] for s in detect(hr, use_ner=False)}
    for want in ["EMP-10293", "BQRPS4821K", "2345 6789 0124", "1MS24CI079", "+91 98450 12345",
                 "richa.sharma92 [at] gmail [dot] com", "50100234567812", "Richa Sharma"]:
        assert want in flagged, want
