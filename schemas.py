from pydantic import BaseModel, field_validator

from app_config import settings
from app_constants import ALLOWED_MESSAGE_RE


class Query(BaseModel):
    message: str
    stream: bool = True
    session_id: str
    chat_history: list[dict] = []

    @field_validator("message")
    @classmethod
    def sanitize(cls, value: str) -> str:
        cleaned = ALLOWED_MESSAGE_RE.sub("", value.strip())
        if len(cleaned) > settings.max_input_chars:
            raise ValueError(
                f"Input exceeds {settings.max_input_chars} characters")
        if not cleaned:
            raise ValueError("Empty message after sanitization")
        return cleaned
