from dataclasses import Field, dataclass
from typing import Any, Callable, Union
from pydantic import BaseModel


class KnowledgeItem(BaseModel):
    content: str
    sources: list[str]  # List of URLs to support the knowledge


class Cognition(BaseModel):
    content: str


class BaseResearchState(BaseModel):
    """Base class for research state that can be extended by users"""

    goal: str
    knowledge: list[KnowledgeItem] = []
    cognition: Cognition = Cognition(content="")
    execution_results: Any = []

    class Config:
        extra = "allow"  # Allow additional fields to be added dynamically


class ParameterSchema(BaseModel):
    code: str
    value: int | str | bool | list[str | int] | dict[str, str | int]


class AgentSelectorSchema(BaseModel):
    selected_agent: str
    parameters: list[ParameterSchema]
    cognition: str


@dataclass
class AgentArgs:
    state: BaseResearchState  # Accept any state that inherits from BaseResearchState
    research_datetime: str
    _notify_progress: Callable[["BaseEvent"], None]
