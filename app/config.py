from pydantic_settings import BaseSettings
from typing import List
import secrets


class Settings(BaseSettings):
    APP_NAME: str = "ResumeAI"
    DEBUG: bool = False
    SECRET_KEY: str = secrets.token_hex(32)
    MAX_FILE_SIZE_MB: int = 5
    ALLOWED_ORIGINS: List[str] = ["http://localhost:8000"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
