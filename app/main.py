"""FastAPI entry point for the privacy-aware redaction tool (runs fully locally)."""
from __future__ import annotations

import io
from pathlib import Path

import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import detect as detector

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"
OLLAMA_URL = "http://localhost:11434"

app = FastAPI(title="Tinyverse PS-3: Privacy-Aware Redaction & Summary")


class DetectRequest(BaseModel):
    text: str
    use_ner: bool = True


def _ollama_status() -> dict:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=1.5)
        r.raise_for_status()
        return {"reachable": True, "models": [m["name"] for m in r.json().get("models", [])]}
    except Exception as exc:
        return {"reachable": False, "error": type(exc).__name__}


@app.get("/api/status")
def status() -> dict:
    return {"detector": detector.detector_status(), "ollama": _ollama_status()}


@app.post("/api/detect")
def detect(req: DetectRequest) -> dict:
    spans = detector.detect(req.text, use_ner=req.use_ner)
    return {"spans": spans, "detector": detector.detector_status()}


def extract_text(filename: str, data: bytes) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        text = "\n".join((p.extract_text() or "") for p in reader.pages)
        if not text.strip():
            raise HTTPException(422, "PDF has no extractable text (scanned PDFs are not supported).")
        return text
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> dict:
    data = await file.read()
    text = extract_text(file.filename or "upload.txt", data)
    return {"filename": file.filename, "text": text}


if STATIC.exists():
    app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    page = STATIC / "index.html"
    if page.exists():
        return FileResponse(page)
    return {"message": "Frontend not built yet. See /docs and /api/status."}
