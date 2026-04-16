
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean
from sqlalchemy.sql import func
from database import Base

class Feedback(Base):
    __tablename__ = "feedback"

    id         = Column(Integer, primary_key=True, index=True)
    name       = Column(String, nullable=True)           # Optional
    type       = Column(String, default="general")       # bug/feature/praise/general
    rating     = Column(Integer, default=3)              # 1–5
    message    = Column(Text, nullable=False)
    resolved   = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
