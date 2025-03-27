from deep_research_anything.models.base import AgentSelectorSchema, BaseResearchState
from pydantic import BaseModel, Field
from typing import Dict, Any
import time


class BaseEvent(BaseModel):
    """Base class for all events"""

    timestamp: float = Field(default_factory=time.time)


class ResearchStateEvent(BaseEvent):
    research_state: BaseResearchState


class ReasoningEvent(BaseEvent):
    reasoning: str
    action: str


class AgentSelectionEvent(BaseEvent):
    agent_selection: AgentSelectorSchema


class ErrorEvent(BaseEvent):
    error: str
    traceback: str
    action: str
