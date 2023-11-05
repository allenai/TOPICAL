from datetime import datetime
from typing import NewType

from pydantic import BaseModel, field_validator

EntityMention = NewType("EntityMention", tuple[str, list[tuple[str, float]]])


class Date(BaseModel):
    year: str
    month: str | None = None
    day: str | None = None

    @field_validator("year")
    @classmethod
    def year_is_valid(cls, v):
        if not v.isdigit() or len(v) != 4:
            raise ValueError(f"Year must be a four digit number, got {v}")
        return v

    def datetime(self) -> datetime.date:
        """Returns the date as a datetime.date object. If the month or day are missing, they will be set to 1."""
        return datetime(int(self.year), int(self.month or 1), int(self.day or 1))

    def timestamp(self) -> int:
        """Returns the date as a UNIX timestamp."""
        return self.to_datetime().timestamp()


def sanitize_text(text: str, lowercase: bool = False) -> str:
    """Cleans text by removing whitespace, newlines and tabs and (optionally) lowercasing."""
    sanitized_text = " ".join(text.strip().split())
    sanitized_text = sanitized_text.lower() if lowercase else sanitized_text
    return sanitized_text
