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
| Summaries | Ollama on `localhost:11434`, model `llama3.2:3b` (set once in `app/config.py` → `SUMMARY_MODEL`) |

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

**Organisation addresses**: letterhead, registered-office and company addresses are
not personal data. They are still detected, but tagged `ORG_ADDRESS` with confidence
0.5, so they show up as low-confidence review candidates instead of being treated
like home addresses.

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

## Entity linking (`app/link.py`)

NAME spans are clustered into people, and each person gets a pseudonym
(`PERSON_1`, `PERSON_2`, …):

- **Initial and surname:** "R. Sharma" joins "Richa Sharma" when the surname matches and the first initial fits.
- **Honorific and surname:** "Ms. Sharma" links only when Sharma is unique in the document and the honorific's gender doesn't conflict.
- **First name only:** "Richa" links only when exactly one person has that first name.
- **Normalisation:** case and punctuation are normalised before any comparison.
- **Exact-occurrence scan:** after clustering, the whole document is searched for every exact occurrence of every linked variant, plus each unambiguous bare first name and surname. Repeats the detector missed are added with `source: "link"`.

Other types are grouped by normalised value, so the same PAN twice, or
`riya@gmail.com` and `riya [at] gmail [dot] com`, become one entity.

## API

| Endpoint | Purpose |
|---|---|
| `POST /api/analyze` | multipart `file` (.txt / text PDF) or form `text` → `{text, spans, entities}`. Scanned/image-only PDFs get a 422 with a clear message |
| `POST /api/link` | `{text, spans}` → re-clustered `{spans, entities}` (used after manual additions) |
| `GET /api/samples`, `/api/samples/{name}` | list / analyse bundled samples |
| `GET /api/status` | detector mode (hybrid vs rules-only) and Ollama / summary-model availability |

## Review UI (`static/`)

The UI is a single page with no build step. The document is on the left, with each
highlighted span labelled by a type tag. The candidates are on the right, grouped by
entity.

**Confidence tones**

| Confidence | Style |
|---|---|
| ≥ 0.90 | solid purple fill |
| 0.60–0.89 | purple outline |
| < 0.60 | dashed outline |

**Keyboard**

| Key | Action |
|---|---|
| `A` | approve |
| `R` | reject |
| `J` / `K` | next / previous span |
| `S` | redact all similar (the same person, or the same type and shape) |
| `M` | add a manual redaction from selected text |
| `?` | shortcut sheet |

**Deep link:** `/#sample=hr_offer_letter.txt` opens a sample directly.

**Fonts are bundled locally**, so the UI works offline:

| Role | Font |
|---|---|
| Headings | *The Seasons* from `static/fonts/TheSeasons.otf` (drop the file in; not committed), falling back to Gloock, then serif |
| Body | Montserrat (OFL, licence in `static/fonts/`) |

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
app/        main.py (API), config.py, detect.py, link.py; redact/summarize/audit/learn upcoming
static/     index.html, styles.css, app.js, fonts/
eval/       evaluation scripts (cache in eval/cache/, git-ignored)
samples/    synthetic test documents
tests/      pytest suite
```
