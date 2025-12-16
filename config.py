import os
from dataclasses import dataclass

@dataclass
class Config:
    DEBUG: bool = True
    SECRET_KEY: str = os.getenv("SECRET_KEY", "change-this")
    SQLALCHEMY_DATABASE_URI: str = os.getenv("DATABASE_URL", "sqlite:///dream_app.db")
    SQLALCHEMY_ECHO: bool = False
    MEDIA_DIR: str = os.getenv("MEDIA_DIR", os.path.abspath("./media"))
    CORS_ORIGINS: str = os.getenv("CORS_ORIGINS", "*")
