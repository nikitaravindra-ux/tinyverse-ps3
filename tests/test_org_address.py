from pathlib import Path

from app.detect import detect

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


def addrs(text):
    return [(s["type"], s["confidence"], s["text"]) for s in detect(text, use_ner=False)
            if s["type"] in ("ADDRESS", "ORG_ADDRESS")]


def test_letterhead_is_org_address():
    t = "ACME SOFTWARE PRIVATE LIMITED\n4th Floor, Tech Park, Outer Ring Road, Bengaluru 560103\n\nDear Sir,"
    assert addrs(t) == [("ORG_ADDRESS", 0.5, "4th Floor, Tech Park, Outer Ring Road, Bengaluru 560103")]


def test_registered_office_is_org_address():
    t = "a company having its registered office at Plot 47, Sector 18, MIDC Bhosari, Pune 411026."
    assert addrs(t)[0][:2] == ("ORG_ADDRESS", 0.5)


def test_residential_stays_personal():
    t = "Residential Address: No. 17, 2nd Main Road, Jayanagar, Bengaluru 560011"
    assert addrs(t)[0][:2] == ("ADDRESS", 0.97)
    t2 = "his residential address at House No. 221, 14th Cross, Indiranagar, Bengaluru 560038."
    assert addrs(t2)[0][:2] == ("ADDRESS", 0.97)


def test_samples_split_org_and_personal():
    hr = addrs((SAMPLES / "hr_offer_letter.txt").read_text())
    assert [a[0] for a in hr] == ["ORG_ADDRESS", "ADDRESS"]
    contract = addrs((SAMPLES / "vendor_contract.txt").read_text())
    assert {a[0] for a in contract} == {"ORG_ADDRESS"}
