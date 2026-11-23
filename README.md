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
.venv/bin/python -m app.fetch_model          # one-time GLiNER download (the only online step)
.venv/bin/uvicorn app.main:app --reload      # http://127.0.0.1:8000
.venv/bin/python -m pytest                   # tests (rules only, no model needed)
```

The GLiNER weights (~1 GB) are downloaded once with `app.fetch_model`. The app
itself runs with Hugging Face in offline mode, so it loads the cached weights and
never contacts the Hub. If the model can't be loaded, the detector falls back to rules
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
| `POST /api/summarize` | audited summary of the redacted text + QI report |
| `POST /api/qi` | quasi-identifier report for any text |
| `POST /api/export` | redacted TXT/PDF (base64) + Redaction Receipt |
| `POST /api/feedback` | record approve/reject/manual events and re-analyse the open document |
| `GET /api/learn`, `POST /api/learn/undo`, `POST /api/learn/reset` | learned rules, undo one, forget all |
| `GET /api/status` | detector mode (hybrid vs rules-only) and Ollama / summary-model availability |

## Redaction & export (`app/redact.py`)

`POST /api/export` takes `{text, spans (with status), entities, mode, format, allow_pending}`.

**Modes**

| `mode` | Replacement |
|---|---|
| `pseudonym` | `[PERSON_1]`, `[EMAIL_1]`, … |
| `mask` | `[NAME]`, `[EMAIL]`, … |

**What gets removed**
- Every approved span.
- For each entity with an approved mention, every occurrence of every one of its variants. The text is rescanned case-insensitively, so an undetected "RICHA SHARMA" in a header still goes.
- Rejected mentions of an approved entity are redacted too, and the receipt counts them as overridden.
- Pending spans are only left in when `allow_pending` is set ("redact anyway"). Otherwise the API returns 409.

**Output formats**
- **TXT:** UTF-8 text.
- **PDF:** rebuilt from scratch from the redacted text with reportlab, using embedded DejaVu Sans so ₹ survives.
- Nothing is ever drawn over the original PDF.
- The `/Info` dictionary is removed and no XMP stream is written: no author, title, producer or dates.

**Integrity verifier**: the exported file is reopened (with pypdf for PDF) and searched case-insensitively for:
- every removed string
- every name variant
- every 4+ digit run from removed phones/IDs, and the joined digit string
- every email username

Role mailboxes such as `hr@` are listed as skipped rather than checked, because "hr" also appears in ordinary text.

**Redaction Receipt**: the result is returned as a receipt containing:
- counts per type and entity count
- `recoverable` (0, or masked failure hints; the receipt never echoes PII)
- the SHA-256 of the exported file
- PDF metadata status and a UTC timestamp

In the UI it is a printable card, with downloads for the file and the receipt JSON.

## Summary, leak audit & quasi-identifiers (`app/summarize.py`, `app/audit.py`)

`POST /api/summarize` recomputes the redaction from the reviewed spans on the server,
so the removed values never come from the client. It then:

1. **Guards the input.** It refuses to summarise if any removed value is still in the text (`RedactionGuardError`, a real exception, not a strippable `assert`). The summarizer only ever sees redacted text.
2. **Summarises with Ollama.** It uses `SUMMARY_MODEL` (`llama3.2:3b`), with a prompt that copies placeholders verbatim and never guesses identities. If Ollama is down, it falls back to an **extractive summary** (top-scored sentences), and the UI labels it as such.
3. **Leak-audits the summary.** It runs the full detector on the summary, plus fuzzy checks against the removed values:
   - partial digits (4+)
   - email usernames
   - surname-only mentions
   - initials (`R.S.`, `R. Sharma`, `Richa S.`)
   - Levenshtein ≤ 2 on name tokens

   Any hit triggers **one regeneration with a stricter prompt**. If that still leaks, the leaked text is **replaced with its placeholder and logged**; the log contains masked hints only.
4. **Scores quasi-identifier (QI) risk** per sentence. The categories are role/title, location (city, neighbourhood, state, "X-based"), organisation (name, type, size), age/date of birth and rare attributes ("only female engineer", health details).
   - **high:** role + city + firm
   - **medium:** two categories
   - Each scored sentence gets an explanation, e.g. "Role + city + organisation type could single out one person".
   - **Generalise** applies the suggested rewrites (CFO → senior executive, Pune → a city in India, Pune-based → regionally based, birth dates → decade).
   - The QI check runs on the summary and, on request, on the redacted document. Document-level generalisations are applied in the next export (`generalizations` on `/api/export`) and counted in the receipt.

**Timing:** llama3.2:3b on an M-series MacBook Air, audited summary per sample.

| Sample | Seconds |
|---|---|
| HR letter (cold start) | 9.5 |
| KYC form | 5.8 |
| Incident report | 5.0 |
| Vendor contract | 6.9 |
| Medical claim PDF | 4.4 |
| **Average** | **6.3** |

## Learning from feedback (`app/learn.py`)

Feedback is stored in `data/learned.json`, which is local only and git-ignored.

| Feedback | What is learned |
|---|---|
| Reject | The exact string goes on the **allowlist** and is never flagged again. After **2 rejections of the same shape** (e.g. `AAA-99999`), that shape becomes a non-PII pattern for the type. |
| Approve / manual add | The exact string goes on the **denylist** and is always flagged in future documents. After **2 manual adds of a new shape**, the shape becomes a candidate regex rule for that type. |
| Every event | Each type's GLiNER threshold is recalibrated from its approve/reject ratio. It starts at 0.45 and stays within **0.25–0.70**. One approval nudges it down by about 0.05. |

- **Digit-only shapes:** shapes are only learned when they contain a digit. Names are learned through the allow/deny lists, because a shape like `Aaaaa` would match every capitalised word.
- **Instant feedback:** each event re-runs detection on the open document straight away. GLiNER's raw predictions are cached per document, so the re-run is instant. For example, rejecting one `EMP-…` code that matches a learned shape clears the others.
- **"What I learned" drawer (`L`):** lists every rule (allowlist, denylist, shapes, candidate rules, thresholds) with **undo** per rule and **Reset learning**. It shows "N corrections learned", plus a precision before → after placeholder for the eval phase.

## Local-only guarantee (`app/netguard.py`)

- **The guard:** at startup the app patches `socket.socket.connect` / `connect_ex`. Any connection that isn't to loopback (`127.0.0.0/8`, `::1`, `localhost`, Unix sockets) is refused with `OutboundBlockedError` and counted. Ollama on localhost keeps working.
- **The "Local only" pill:** it reads `/api/status` and shows:
  - whether GLiNER is loaded
  - Ollama status
  - outbound network calls (0)
  - blocked attempts
  - local connections

  All of these are real counters from the guard; nothing is hardcoded.

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

**Export panel:** pseudonym/mask toggle and TXT/PDF buttons. The buttons are enabled once nothing is pending, or after a "redact anyway" confirmation. `E` exports PDF.

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
app/        main.py (API), config.py, detect.py, link.py, redact.py, summarize.py, audit.py,
            learn.py, netguard.py, fetch_model.py
app/fonts/  DejaVu Sans for rebuilt PDFs (licence included)
static/     index.html, styles.css, app.js, fonts/
eval/       evaluation scripts (cache in eval/cache/, git-ignored)
samples/    synthetic test documents
tests/      pytest suite
```
