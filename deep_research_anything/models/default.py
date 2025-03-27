from typing import List, Dict
from deep_research_anything.agent.agents.search.models import Page, SearchQuery
from deep_research_anything.models.base import BaseResearchState


class DefaultResearchState(BaseResearchState):
    """Default implementation of research state with standard fields"""

    searched_queries: List[SearchQuery] = []
    all_url_to_result_page: Dict[str, Page] = {}
