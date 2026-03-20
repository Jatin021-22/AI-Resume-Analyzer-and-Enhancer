from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from loguru import logger
import tempfile
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyze import extract_text, match_resume_with_job
from config import settings

limiter = Limiter(key_func=get_remote_address)
router  = APIRouter(prefix="/resume", tags=["Resume"])

MAX_BYTES = settings.MAX_FILE_SIZE_MB * 1024 * 1024


def _validate_pdf_magic(content: bytes) -> bool:
    """Check PDF magic bytes — don't trust just the extension."""
    return content[:4] == b"%PDF"


@router.post("/analyze")
@limiter.limit("10/minute")
async def analyze_resume(
    request: Request,
    file: UploadFile = File(...),
    job_description: str = Form(...),
):
    """Upload PDF resume and job description → full AI analysis."""

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    if not job_description.strip():
        raise HTTPException(400, "Job description cannot be empty.")
    if len(job_description) > 10_000:
        raise HTTPException(400, "Job description too long (max 10,000 chars).")

    content = await file.read()

    if len(content) > MAX_BYTES:
        raise HTTPException(400, f"File too large. Max {settings.MAX_FILE_SIZE_MB}MB.")
    if not _validate_pdf_magic(content):
        raise HTTPException(400, "File does not appear to be a valid PDF.")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        resume_text = extract_text(tmp_path)

        if not resume_text.strip():
            raise HTTPException(
                422,
                "Could not extract text. Ensure the PDF is text-based, not a scanned image.",
            )

        result = match_resume_with_job(resume_text, job_description)
        logger.info(f"Analysis complete — score={result.get('match_score')} file={file.filename}")
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Analysis failed: {exc}")
        raise HTTPException(500, "Analysis failed. Please try again.")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@router.post("/analyze-text")
@limiter.limit("10/minute")
async def analyze_resume_text(
    request: Request,
    resume_text: str = Form(...),
    job_description: str = Form(...),
):
    """Paste resume text directly (no PDF upload needed)."""

    if not resume_text.strip():
        raise HTTPException(400, "Resume text cannot be empty.")
    if len(resume_text) > 50_000:
        raise HTTPException(400, "Resume text too long (max 50,000 chars).")
    if not job_description.strip():
        raise HTTPException(400, "Job description cannot be empty.")

    try:
        result = match_resume_with_job(resume_text, job_description)
        logger.info(f"Text analysis complete — score={result.get('match_score')}")
        return JSONResponse(content=result)
    except Exception as exc:
        logger.error(f"Text analysis failed: {exc}")
        raise HTTPException(500, "Analysis failed. Please try again.")
