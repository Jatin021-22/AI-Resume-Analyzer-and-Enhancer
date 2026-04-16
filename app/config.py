from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    APP_NAME: str = "ResumeAI"
    DEBUG: bool = False
    SECRET_KEY: str = "dev-secret-key-change-in-production"
    DATABASE_URL: str = ""
    MAX_FILE_SIZE_MB: int = 5
    ALLOWED_ORIGINS: List[str] = ["http://localhost:8000"]

    class Config:
        # Go one level up from app/ to find .env
        env_file = os.path.join(os.path.dirname(__file__), "..", ".env")
        env_file_encoding = "utf-8"

settings = Settings()