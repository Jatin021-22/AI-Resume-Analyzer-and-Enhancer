from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from slowapi import Limiter
from slowapi.util import get_remote_address
from loguru import logger
import tempfile
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyze import extract_text, match_resume_with_job
from config import settings
from database import get_db
from models import ResumeAnalysis
from models.user import User
from utils.auth import get_optional_user, get_current_user

limiter   = Limiter(key_func=get_remote_address)
router    = APIRouter(prefix="/resume", tags=["Resume"])
MAX_BYTES = settings.MAX_FILE_SIZE_MB * 1024 * 1024


def _validate_pdf_magic(content: bytes) -> bool:
    return content[:4] == b"%PDF"


@router.post("/analyze")
@limiter.limit("10/minute")
async def analyze_resume(
    request: Request,
    file: UploadFile = File(...),
    job_description: str = Form(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_optional_user),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")
    if not job_description.strip():
        raise HTTPException(400, "Job description cannot be empty.")
    if len(job_description) > 10_000:
        raise HTTPException(400, "Job description too long.")

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
            raise HTTPException(422, "Could not extract text from PDF.")

        result = match_resume_with_job(resume_text, job_description)

        # ── Save to database ──────────────────────────
        analysis = ResumeAnalysis(
            user_id         = current_user.id if current_user else None,
            match_score     = result.get("match_score"),
            confidence      = result.get("confidence"),
            insight         = result.get("insight"),
            matched_skills  = result.get("matched_skills"),
            missing_skills  = result.get("missing_skills"),
            resume_skills   = result.get("resume_skills"),
            job_skills      = result.get("job_skills"),
            suggestions     = result.get("suggestions"),
            experience      = result.get("experience"),
            role_detected   = result.get("role_detection", {}).get("primary_role"),
            ats_score       = result.get("ats", {}).get("ats_score"),
            fresher         = result.get("fresher"),
            job_description = job_description[:500],
            resume_filename = file.filename,
        )
        db.add(analysis)
        db.commit()
        db.refresh(analysis)
        # ─────────────────────────────────────────────

        result["analysis_id"] = analysis.id
        logger.info(
            f"Saved — id={analysis.id} "
            f"score={result.get('match_score')} "
            f"user={'logged-in' if current_user else 'guest'}"
        )
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
    db: Session = Depends(get_db),
    current_user: User = Depends(get_optional_user),
):
    if not resume_text.strip():
        raise HTTPException(400, "Resume text cannot be empty.")
    if len(resume_text) > 50_000:
        raise HTTPException(400, "Resume text too long.")
    if not job_description.strip():
        raise HTTPException(400, "Job description cannot be empty.")

    try:
        result = match_resume_with_job(resume_text, job_description)

        # ── Save to database ──────────────────────────
        analysis = ResumeAnalysis(
            user_id         = current_user.id if current_user else None,
            match_score     = result.get("match_score"),
            confidence      = result.get("confidence"),
            insight         = result.get("insight"),
            matched_skills  = result.get("matched_skills"),
            missing_skills  = result.get("missing_skills"),
            resume_skills   = result.get("resume_skills"),
            job_skills      = result.get("job_skills"),
            suggestions     = result.get("suggestions"),
            experience      = result.get("experience"),
            role_detected   = result.get("role_detection", {}).get("primary_role"),
            ats_score       = result.get("ats", {}).get("ats_score"),
            fresher         = result.get("fresher"),
            job_description = job_description[:500],
            resume_filename = "text-input",
        )
        db.add(analysis)
        db.commit()
        db.refresh(analysis)
        # ─────────────────────────────────────────────

        result["analysis_id"] = analysis.id
        logger.info(
            f"Text analysis saved — id={analysis.id} "
            f"user={'logged-in' if current_user else 'guest'}"
        )
        return JSONResponse(content=result)

    except Exception as exc:
        logger.error(f"Text analysis failed: {exc}")
        raise HTTPException(500, "Analysis failed. Please try again.")


@router.get("/history")
async def get_history(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    limit: int = 20,
):
    """Returns only the logged-in user's analyses."""
    analyses = (
        db.query(ResumeAnalysis)
        .filter(ResumeAnalysis.user_id == current_user.id)
        .order_by(ResumeAnalysis.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id":         a.id,
            "score":      a.match_score,
            "confidence": a.confidence,
            "role":       a.role_detected,
            "ats_score":  a.ats_score,
            "filename":   a.resume_filename,
            "fresher":    a.fresher,
            "created_at": str(a.created_at),
        }
        for a in analyses
    ]