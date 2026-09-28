from enum import StrEnum

from pydantic import BaseModel


class ActionType(StrEnum):
    CLICK = "click"


class ActionCandidate(BaseModel):
    id: str
    action_type: ActionType
    selector: str
    text: str
    enabled: bool
