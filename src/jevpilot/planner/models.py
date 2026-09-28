from pydantic import BaseModel


class Plan(BaseModel):
    target_selector: str
    verification_selector: str
    expected_text: str
