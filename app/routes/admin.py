
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func
from database import get_db
from models.feedback import Feedback
from models.user import User
from models.resume import ResumeAnalysis
import os

router = APIRouter(prefix="/api/admin", tags=["admin"])

ADMIN_KEY = os.getenv("ADMIN_KEY", "resumeai_admin_2024")  # Set in .env !

def check_admin(x_admin_key: str = None):
    # For now admin auth is handled client-side via JS
    # For production: use Header(alias="X-Admin-Key") and validate here
    pass

# ── STATS ──
@router.get("/stats")
def admin_stats(db: Session = Depends(get_db)):
    analyses = db.query(func.count(ResumeAnalysis.id)).scalar()
    users    = db.query(func.count(User.id)).scalar()
    feedback = db.query(func.count(Feedback.id)).scalar()
    avg      = db.query(func.avg(ResumeAnalysis.match_score)).scalar()
    return {
        "analyses":  analyses or 0,
        "users":     users or 0,
        "feedback":  feedback or 0,
        "avg_score": round(avg or 0, 1),
    }

# ── ALL FEEDBACK ──
@router.get("/feedback")
def list_feedback(limit: int = 200, db: Session = Depends(get_db)):
    items = db.query(Feedback).order_by(Feedback.created_at.desc()).limit(limit).all()
    return {"items": [
        {
            "id":         f.id,
            "name":       f.name,
            "type":       f.type,
            "rating":     f.rating,
            "message":    f.message,
            "resolved":   f.resolved,
            "created_at": f.created_at.isoformat() if f.created_at else None,
        } for f in items
    ]}

# ── RATING DISTRIBUTION ──
@router.get("/feedback/rating-distribution")
def rating_dist(db: Session = Depends(get_db)):
    rows = db.query(Feedback.rating, func.count(Feedback.id)).group_by(Feedback.rating).all()
    return {str(r): c for r, c in rows}

# ── TYPE DISTRIBUTION ──
@router.get("/feedback/type-distribution")
def type_dist(db: Session = Depends(get_db)):
    rows = db.query(Feedback.type, func.count(Feedback.id)).group_by(Feedback.type).all()
    return {t: c for t, c in rows}

# ── RESOLVE ──
@router.patch("/feedback/{fb_id}/resolve")
def resolve_feedback(fb_id: int, db: Session = Depends(get_db)):
    fb = db.query(Feedback).filter(Feedback.id == fb_id).first()
    if not fb:
        raise HTTPException(404, "Not found")
    fb.resolved = True
    db.commit()
    return {"status": "resolved"}

# ── DELETE ──
@router.delete("/feedback/{fb_id}")
def delete_feedback(fb_id: int, db: Session = Depends(get_db)):
    fb = db.query(Feedback).filter(Feedback.id == fb_id).first()
    if not fb:
        raise HTTPException(404, "Not found")
    db.delete(fb)
    db.commit()
    return {"status": "deleted"}

# ── USERS ──
@router.get("/users")
def list_users(db: Session = Depends(get_db)):
    users = db.query(User).order_by(User.created_at.desc()).all()
    return {"users": [
        {
            "id":         u.id,
            "name":       u.name,
            "email":      u.email,
            "plan":       u.plan,
            "is_active":  u.is_active,
            "created_at": u.created_at.isoformat() if u.created_at else None,
        } for u in users
    ]}

# ── ANALYSES ──
@router.get("/analyses")
def list_analyses(limit: int = 50, db: Session = Depends(get_db)):
    items = db.query(ResumeAnalysis).order_by(ResumeAnalysis.created_at.desc()).limit(limit).all()
    return {"items": [
        {
            "id":            a.id,
            "user_id":       a.user_id,
            "match_score":   a.match_score,
            "confidence":    a.confidence,
            "role_detected": a.role_detected,
            "ats_score":     a.ats_score,
            "fresher":       a.fresher,
            "created_at":    a.created_at.isoformat() if a.created_at else None,
        } for a in items
    ]}