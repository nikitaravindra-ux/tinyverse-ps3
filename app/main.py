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

from app import detect as detector
from app.config import OLLAMA_URL, SUMMARY_MODEL
from app.link import link
from app import redact

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
    return {"detector": detector.detector_status(), "ollama": _ollama_status()}


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


def analyze_text(text: str, filename: str = "pasted.txt") -> dict:
    text = text.replace("\r\n", "\n")
    spans = detector.detect(text)
    spans, entities = link(text, spans)
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
    result = redact.export(req.text, req.spans, req.entities, req.mode, req.format)
    return {
        "receipt": result["receipt"],
        "file": {"name": result["receipt"]["file"]["name"], "mime": result["mime"],
                 "base64": base64.b64encode(result["data"]).decode("ascii")},
    }


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
