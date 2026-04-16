from sqlalchemy import (
    Column, Integer, String,
    JSON, DateTime, ForeignKey, Text, Boolean
)
from sqlalchemy.orm import relationship
from database import Base
from datetime import datetime


class ResumeAnalysis(Base):
    __tablename__ = "resume_analyses"

    id              = Column(Integer, primary_key=True, index=True)
    user_id         = Column(Integer, ForeignKey("users.id"), nullable=True)
    match_score     = Column(Integer)
    confidence      = Column(String)
    insight         = Column(Text)
    matched_skills  = Column(JSON)
    missing_skills  = Column(JSON)
    resume_skills   = Column(JSON)
    job_skills      = Column(JSON)
    suggestions     = Column(JSON)
    experience      = Column(JSON)
    role_detected   = Column(String)
    ats_score       = Column(Integer)
    fresher         = Column(Boolean)
    job_description = Column(Text)
    resume_filename = Column(String)
    created_at      = Column(DateTime, default=datetime.utcnow)

    # Many analyses belong to one user
    user = relationship("User", back_populates="analyses")