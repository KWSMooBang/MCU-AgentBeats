from typing import Any, Literal
from pydantic import BaseModel, HttpUrl, Field


class EvalRequest(BaseModel):
    """Request format sent by the AgentBeats platform to green agents."""
    participants: dict[str, HttpUrl] # role -> agent URL
    config: dict[str, Any]
    
# Protocols between Green and Purple agents
# From Green Agent to Purple Agent
class InitPayload(BaseModel):
    """Initial task description sent to purple agent."""
    type: Literal["init"] = "init"
    text: str = Field(..., description="Task description")

class ObservationPayload(BaseModel):
    """Observation sent to purple agent at each step."""
    type: Literal["obs"] = "obs"
    step: int = Field(..., ge=0, description="Current step number")
    obs: str = Field(..., description="Base64 encoded image")

# From Purple Agent to Green Agent
class AckPayload(BaseModel):
    """Acknowledgment from purple agent."""
    type: Literal["ack"] = "ack"
    success: bool = False
    message: str = ""

class ActionPayload(BaseModel):
    """Action response from purple agent."""
    type: Literal["action"] = "action"
    buttons: list[int] = Field(..., description="Button states")
    camera: list[float] = Field(..., description="Camera movements")