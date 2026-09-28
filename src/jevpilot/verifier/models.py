from pydantic import BaseModel


class VerificationResult(BaseModel):
    success: bool
    observed_value: str | None = None
    message: str = ""
