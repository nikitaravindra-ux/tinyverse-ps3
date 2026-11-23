"""FastAPI entry point for the privacy-aware redaction tool (runs fully locally)."""
from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Optional

import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import netguard

netguard.install()  # block + count every non-localhost connection from here on

from app import detect as detector  # noqa: E402
from app import learn  # noqa: E402
from app.config import OLLAMA_URL, SUMMARY_MODEL
from app.link import link
from app import audit, redact
from app import summarize as summarizer

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"
SAMPLES = ROOT / "samples"
TEXT_EXTS = {".txt", ".md", ".text"}
MAX_BYTES = 10 * 1024 * 1024
MIN_CHARS_PER_PAGE = 25  # below this a PDF is treated as scanned / image-only

app = FastAPI(title="Tinyverse PS-3: Privacy-Aware Redaction & Summary")


class DetectRequest(BaseModel):
    text: str
    use_ner: bool = True


class ExportRequest(BaseModel):
    text: str
    spans: list[dict]          # each carries status: approved | rejected | pending
    entities: list[dict]
    mode: str = "pseudonym"    # pseudonym -> [PERSON_1]; mask -> [NAME]
    format: str = "pdf"        # pdf | txt
    allow_pending: bool = False
    generalizations: list[dict] = []   # [{find, replace}] applied to the redacted text


class SummaryRequest(BaseModel):
    text: str
    spans: list[dict]
    entities: list[dict]
    mode: str = "pseudonym"


class QIRequest(BaseModel):
    text: str


class LinkRequest(BaseModel):
    text: str
    spans: list[dict]


def _ollama_status() -> dict:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=1.5)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        return {"reachable": True, "summary_model": SUMMARY_MODEL,
                "model_installed": SUMMARY_MODEL in models, "models": models}
    except Exception as exc:
        return {"reachable": False, "summary_model": SUMMARY_MODEL, "error": type(exc).__name__}


@app.get("/api/status")
def status() -> dict:
    return {"detector": detector.detector_status(), "ollama": _ollama_status(),
            "network": netguard.status(), "learning": {"corrections": learn.summary()["corrections"]}}


@app.post("/api/detect")
def detect(req: DetectRequest) -> dict:
    spans = detector.detect(req.text, use_ner=req.use_ner)
    return {"spans": spans, "detector": detector.detector_status()}




def extract_text(filename: str, data: bytes) -> str:
    """Text from a TXT or text-based PDF. Scanned/image-only PDFs are rejected."""
    suffix = Path(filename.lower()).suffix
    if suffix == ".pdf" or data[:5] == b"%PDF-":
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError

        try:
            reader = PdfReader(io.BytesIO(data))
            pages = [(p.extract_text() or "") for p in reader.pages]
        except (PdfReadError, ValueError) as exc:
            raise HTTPException(422, f"Could not read this PDF: {exc}") from exc
        text = "\n".join(pages)
        n = max(1, len(pages))
        if len(text.strip()) < MIN_CHARS_PER_PAGE * n:
            raise HTTPException(
                422,
                "This PDF looks scanned or image-only: it has no extractable text layer. "
                "Only text-based PDFs are supported (OCR is out of scope). "
                "Export the document as a text PDF or upload a .txt file instead.",
            )
        return text
    if suffix and suffix not in TEXT_EXTS:
        raise HTTPException(415, f"Unsupported file type '{suffix}'. Upload a .txt or text-based .pdf.")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def analyze_text(text: str, filename: str = "pasted.txt", manual_spans: Optional[list] = None) -> dict:
    text = text.replace("\r\n", "\n")
    state = learn.load()
    spans = detector.detect(text, type_thresholds=learn.thresholds(state))
    spans = learn.apply(text, spans, state)
    for m in manual_spans or []:  # keep this session's manual redactions
        spans = [s for s in spans if not (s["start"] < m["end"] and s["end"] > m["start"])]
        spans.append({k: m[k] for k in ("start", "end", "text", "type")} | {"confidence": 1.0, "source": "manual"})
    spans, entities = link(text, sorted(spans, key=lambda s: s["start"]))
    return {"filename": filename, "text": text, "spans": spans, "entities": entities,
            "detector": detector.detector_status()}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> dict:
    data = await file.read()
    return {"filename": file.filename, "text": extract_text(file.filename or "upload.txt", data)}


@app.post("/api/analyze")
async def analyze(file: Optional[UploadFile] = File(None), text: Optional[str] = Form(None)) -> dict:
    """Upload a TXT/PDF (or post `text`) -> text + spans + entities."""
    if file is not None:
        data = await file.read()
        if len(data) > MAX_BYTES:
            raise HTTPException(413, "File too large (10 MB max).")
        name = file.filename or "upload.txt"
        content = extract_text(name, data)
    elif text is not None:
        name, content = "pasted.txt", text
    else:
        raise HTTPException(400, "Send a file or a text field.")
    if not content.strip():
        raise HTTPException(422, "The document is empty.")
    return analyze_text(content, name)


@app.post("/api/link")
def relink(req: LinkRequest) -> dict:
    """Re-cluster after manual edits in the review UI."""
    spans, entities = link(req.text, req.spans)
    return {"spans": spans, "entities": entities}


@app.post("/api/export")
def export(req: ExportRequest) -> dict:
    """Redact approved spans (+ all linked variants), write TXT or a rebuilt PDF,
    re-open it and verify nothing removed is recoverable."""
    if req.mode not in redact.MODES:
        raise HTTPException(422, f"mode must be one of {list(redact.MODES)}")
    if req.format not in redact.FORMATS:
        raise HTTPException(422, f"format must be one of {list(redact.FORMATS)}")
    pending = sum(1 for s in req.spans if s.get("status", "pending") == "pending")
    if pending and not req.allow_pending:
        raise HTTPException(409, f"{pending} span(s) are still pending review. "
                                 "Review them or confirm 'redact anyway'.")
    result = redact.export(req.text, req.spans, req.entities, req.mode, req.format,
                           generalizations=req.generalizations)
    return {
        "receipt": result["receipt"],
        "redacted_text": result["redacted_text"],
        "file": {"name": result["receipt"]["file"]["name"], "mime": result["mime"],
                 "base64": base64.b64encode(result["data"]).decode("ascii")},
    }


@app.post("/api/summarize")
def summarize(req: SummaryRequest) -> dict:
    """Summarise the redacted text only, audit the summary for leaks, score QI risk.

    The redaction is recomputed here from the reviewed spans, so the removed values
    used by the input guard and the auditor never depend on the client."""
    if req.mode not in redact.MODES:
        raise HTTPException(422, f"mode must be one of {list(redact.MODES)}")
    view = redact.redacted_view(req.text, req.spans, req.entities, req.mode)
    result = audit.audited_summary(view["redacted_text"], view["removed"], req.entities, req.mode,
                                   summarizer=summarizer.summarize)
    result["qi"] = audit.qi_report(result["summary"])
    return result


@app.post("/api/qi")
def qi(req: QIRequest) -> dict:
    return audit.qi_report(req.text)


class FeedbackRequest(BaseModel):
    events: list[dict]                 # [{action: approve|reject|manual, text, type}]
    text: Optional[str] = None         # current document: re-analysed after learning
    filename: str = "document.txt"
    manual_spans: list[dict] = []


@app.post("/api/feedback")
def feedback(req: FeedbackRequest) -> dict:
    learned = learn.record(req.events)
    out = {"learning": learned}
    if req.text is not None:
        out["analysis"] = analyze_text(req.text, req.filename, req.manual_spans)
    return out


@app.get("/api/learn")
def learned() -> dict:
    return learn.summary()


class UndoRequest(BaseModel):
    rule_id: str
    text: Optional[str] = None
    filename: str = "document.txt"
    manual_spans: list[dict] = []


@app.post("/api/learn/undo")
def undo(req: UndoRequest) -> dict:
    if not learn.undo(req.rule_id):
        raise HTTPException(404, "No such learned rule.")
    out = {"learning": learn.summary()}
    if req.text is not None:
        out["analysis"] = analyze_text(req.text, req.filename, req.manual_spans)
    return out


@app.post("/api/learn/reset")
def reset_learning() -> dict:
    learn.reset()
    return {"learning": learn.summary()}


@app.get("/api/samples")
def samples() -> list[dict]:
    return [{"name": p.name, "kind": p.suffix.lstrip(".")}
            for p in sorted(SAMPLES.iterdir()) if p.suffix in (".txt", ".pdf")
            and not (p.suffix == ".txt" and p.with_suffix(".pdf").exists())]


@app.get("/api/samples/{name}")
def analyze_sample(name: str) -> dict:
    path = (SAMPLES / name).resolve()
    if path.parent != SAMPLES.resolve() or not path.is_file():
        raise HTTPException(404, "No such sample.")
    return analyze_text(extract_text(path.name, path.read_bytes()), path.name)


if STATIC.exists():
    app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
