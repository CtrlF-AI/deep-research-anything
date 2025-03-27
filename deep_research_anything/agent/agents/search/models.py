from typing import List
from pydantic import BaseModel

from deep_research_anything.models.event import BaseEvent
from deep_research_anything.models.base import KnowledgeItem


class Page(BaseModel):
    url: str
    title: str
    description: str  # short description of the page
    markdown: str  # markdown content of the page
    modified_time: str = ""


class SearchQuery(BaseModel):
    sub_goal: str
    query_string: str
    search_result_pages: List[Page]


class AllowedToReadSchema(BaseModel):
    allowed: bool


class KnowledgeItemsSchema(BaseModel):
    knowledge: list[str]


# Search-related events
class BatchSearchStartEvent(BaseEvent):
    query_strings: List[str]
    sub_goal: str


class BatchSearchCompleteEvent(BaseEvent):
    query_strings: List[str]
    results: List[Page]


class BatchReadStartEvent(BaseEvent):
    pages: List[Page]


class BatchReadCompleteEvent(BaseEvent):
    pages: List[Page]


class PageNotAllowedToReadEvent(BaseEvent):
    page: Page


class NewKnowledgeEvent(BaseEvent):
    new_items: List[KnowledgeItem]
