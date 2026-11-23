# Tinyverse PS-3: Privacy-Aware Document Redaction & Summary Tool

Finds personal data in Indian business documents (HR letters, KYC forms, incident
reports, contracts), then redacts it and summarises the result. **It all runs on your
own machine.** No cloud APIs are used, and no document leaves the laptop.

| Layer | Tech |
|---|---|
| Backend | Python 3.11, FastAPI |
| Frontend | Plain HTML/CSS/vanilla JS served by FastAPI (no build step) |
| PDF in / out | `pypdf` (text PDFs) / `reportlab` |
| NER | [GLiNER](https://github.com/urchade/GLiNER) `urchade/gliner_multi_pii-v1` |
| Summaries | Ollama on `localhost:11434` |

## Quick start

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --reload      # http://127.0.0.1:8000
.venv/bin/python -m pytest                   # tests (rules only, no model needed)
```

The GLiNER weights (~1 GB) download from Hugging Face on first use and are cached
locally after that. If the model can't be loaded, the detector falls back to rules
only, and `GET /api/status` reports it:

```json
{"detector": {"status": "unavailable", "mode": "rules-only (GLiNER model unavailable)", "error": "..."},
 "ollama": {"reachable": true, "models": ["llama3.2:3b"]}}
```

## Detector (`app/detect.py`)

`detect(text)` returns spans `{id, start, end, text, type, confidence, source}` with
`type` ∈ `NAME, EMAIL, PHONE, ADDRESS, ID`.

**Rules + validators**
- **EMAIL**: standard addresses and obfuscated ones (`riya [at] gmail [dot] com`, `riya(at)gmail.com`).
- **PHONE**: `+91 98xxx xxxxx`, `080-2xxx xxxx`, `(022) xxxx xxxx`, 10-digit mobiles starting 6–9, and international `+CC` numbers.
- **ID**:
  - Aadhaar (12 digits, **Verhoeff checksum**)
  - PAN (`AAAAA9999A` with a valid holder-type letter)
  - IFSC
  - UPI (`name@okaxis`, …)
  - credit/debit cards (**Luhn**)
  - `EMP-10293`, VTU USNs like `1MS24CI079`
  - account numbers next to `A/c` / `Account No`
  - keyed IDs (`Customer ID:`, `Passport No:` …)
- **ADDRESS**: street cues (Road, Nagar, Layout, Cross, Main, Sector, Flat, Apt …) extended to a 6-digit PIN code, plus `Address:` fields.
- **NAME** (fallback quality): honorifics (`Ms. Sharma`), initials (`R. Sharma`), labelled fields (`Name:`, `Approved by:`).

**GLiNER** labels: `person, email, phone number, address, id number`. The model loads
lazily. Text is chunked into 300-word windows with a 60-word overlap, and offsets are
mapped back to the full document.

**Confidence**

| Source | Confidence |
|---|---|
| Regex that passed a validator (checksum, structure or keyword context) | 0.97 |
| Plain regex | 0.80 |
| GLiNER | the model's score |
| GLiNER agreeing with a same-type rule | `1-(1-g)(1-r)`, capped at 0.99 |

Overlaps are merged: the highest confidence wins, spans of the same type are unioned,
and a weaker NAME/ADDRESS span is trimmed around a stronger span of another type.

**False-positive guards**: these are never flagged as PII:
- dates and times
- amounts with `₹`/`Rs`/`INR`
- invoice/order/SKU/product/PO/ticket codes (`SKU-88213-B`, `order #448812339021`)
- ISBNs and version numbers
- any bare 12-digit number that fails Verhoeff

## Samples (`samples/`)

All data is synthetic. Every Aadhaar/PAN/card number is made up, though some are
generated to pass their checksums.

- `hr_offer_letter.txt`: name variants ("Richa Sharma", "R. Sharma", "Ms. Sharma", "Richa"), a signature block and a letterhead
- `bank_kyc_form.txt`: KYC fields, Aadhaar, PAN, passport, UPI, card, IFSC
- `incident_report.txt`: a DLP incident with PII embedded in prose, and decoy numbers
- `vendor_contract.txt`: two-party contract, bank details, ISBN and version decoys
- `medical_reimbursement_claim.pdf`: text PDF (regenerate it with `python samples/make_pdf.py`)

## Layout

```
app/        main.py (API), detect.py, plus link/redact/summarize/audit/learn (upcoming)
static/     frontend
eval/       evaluation scripts (cache in eval/cache/, git-ignored)
samples/    synthetic test documents
tests/      pytest suite
```
