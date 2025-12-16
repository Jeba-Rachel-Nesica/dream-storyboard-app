from pydantic import BaseModel, Field, EmailStr
from typing import Optional

class Controls(BaseModel):
    soften: bool = True
    helper: bool = False
    exit: bool = True

class RewriteIn(BaseModel):
    nightmare: str = Field(min_length=1)
    controls: Controls

class RewriteOut(BaseModel):
    text: str

class TTSIn(BaseModel):
    text: str = Field(min_length=1)

class JournalIn(BaseModel):
    sleep_quality: int = Field(ge=1, le=5)
    nightmare_intensity: int = Field(ge=1, le=5)
    emotions: Optional[str] = ""
    note: Optional[str] = ""

class RegisterIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6)

class LoginIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6)
