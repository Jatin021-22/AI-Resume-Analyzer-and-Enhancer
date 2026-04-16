from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from pydantic import BaseModel, validator
from typing import Optional
from database import get_db
from models.feedback import Feedback

router = APIRouter(prefix="/api/feedback", tags=["feedback"])

class FeedbackCreate(BaseModel):
    name: Optional[str] = None
    type: str = "general"
    rating: int = 3
    message: str

    @validator("message")
    def message_not_empty(cls, v):
        if not v or len(v.strip()) < 5:
            raise ValueError("Message must be at least 5 characters")
        return v.strip()[:2000]  # cap at 2000 chars

    @validator("rating")
    def rating_range(cls, v):
        if v < 1 or v > 5:
            raise ValueError("Rating must be 1–5")
        return v

    @validator("type")
    def type_valid(cls, v):
        allowed = {"bug", "feature", "praise", "general"}
        if v not in allowed:
            return "general"
        return v

@router.post("/submit", status_code=201)
async def submit_feedback(data: FeedbackCreate, db: Session = Depends(get_db)):
    fb = Feedback(**data.dict())
    db.add(fb)
    db.commit()
    db.refresh(fb)
    return {"status": "ok", "id": fb.id}
