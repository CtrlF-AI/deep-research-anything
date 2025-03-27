from typing import List
from pydantic import BaseModel

from deep_research_anything.models.event import BaseEvent
from deep_research_anything.models.base import KnowledgeItem


class ConflictingPairSchema(BaseModel):
    index1: int
    index2: int


class ConflictingPairsSchema(BaseModel):
    conflicting_pairs: list[ConflictingPairSchema]


class ResolvedKnowledgeSchema(BaseModel):
    reasoning: str
    resolved_knowledge: str


class GenerateEssayStartEvent(BaseEvent):
    pass


class GenerateEssayCompleteEvent(BaseEvent):
    essay: str


class RefinedKnowledgeEvent(BaseEvent):
    original_items: List[KnowledgeItem]
    refined_items: List[KnowledgeItem]


class EssaySchema(BaseModel):
    outline: str
    essay: str
