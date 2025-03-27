import asyncio
from datetime import datetime, timedelta
import json
from typing import List, Tuple

from deep_research_anything.agent.parameter import Parameter
from deep_research_anything.agent.agent import ExecutionResult, Agent
from deep_research_anything.models.base import AgentArgs, KnowledgeItem
from deep_research_anything.models.event import ReasoningEvent
from deep_research_anything.agent.agents.search.models import (
    AllowedToReadSchema,
    BatchReadCompleteEvent,
    BatchReadStartEvent,
    BatchSearchCompleteEvent,
    BatchSearchStartEvent,
    KnowledgeItemsSchema,
    Page,
    PageNotAllowedToReadEvent,
    SearchQuery,
)
from deep_research_anything.agent.agents.search.models import NewKnowledgeEvent
from deep_research_anything.llms import O3MiniModel, generate_dict
from deep_research_anything.agent.agents.search._firecrawl import firecrawl_search


class QueryStringsParameter(Parameter):
    def __init__(self):
        name = "Query Strings List"
        code = "query_strings"
        description = (
            "List of internet search queries, can include multiple query strings"
        )
        super().__init__(
            name=name,
            code=code,
            description=description,
            optional=False,
            value=None,
            parameter_type=list[str],
        )


class SubGoalParameter(Parameter):
    def __init__(
        self,
    ):
        name = "Sub-Goal"
        code = "sub_goal"
        description = "Next research sub-goal"
        super().__init__(
            name=name,
            code=code,
            description=description,
            optional=False,
            value=None,
            parameter_type=str,
        )


class SearchAgent(Agent):
    content_max_length = 40000

    def __init__(
        self,
        name="Search and Read Agent",
        code="search",
        description="Based on the research goal and current research status, input your next sub-goal and corresponding internet search query list, and this agent will find and read all relevant pages.",
        documentation="""
When providing parameters, defining next sub-goal and query strings. 
You should derive your sub-goal with DEPTH from the research goal and current research status. Focus on identifying unexplored but important aspects that need investigation. For example, if you've discovered a person attended a specific school and learned their teacher's name, your next sub-goal might be to investigate the relationship between them or learn more about the teacher's background. Always ensure your sub-goals build upon previous findings while remaining focused on the final research goal. Don't be afraid to dig deeper into specific details that could provide valuable insights, but avoid going too far off-topic.

Your search queries should focus on your sub-goal. Each query should be a specific that can be searched on Google. Don't try to directly search for final answers or conclusions. Instead, search for specific facts or different perspectives that help you build analysis like a human would. Don't search for too many keywords in a single query; keep queries short and focused.
For example, if your sub-goal is to research artificial intelligence safety:
- Bad query: "What are all the risks of artificial intelligence and how to solve them" (seeking direct answers)
- Bad query: "AI safety alignment control risk catastrophic scenarios regulation ethics" (too broad, unfocused)
- Good query list: ["AI alignment technical approaches", "large language model safety measures", "AI regulation policies", "AI risk assessment"]

Search for specific facts, data points, and relationships so you can later synthesize them into insights. Collect evidence methodically, not looking for ready-made conclusions.
The query language should be consistent with the language of the research goal.
""",
        read_permission_model=O3MiniModel,
        extract_knowledge_model=O3MiniModel,
    ):
        super().__init__(
            name=name, code=code, description=description, documentation=documentation
        )
        self.read_permission_model = read_permission_model
        self.extract_knowledge_model = extract_knowledge_model
        self._init_parameters()

    def _init_parameters(self):
        """Initialize parameters - can be overridden in subclasses"""
        self.add_parameter(SubGoalParameter())
        self.add_parameter(QueryStringsParameter())

    def get_page_allowed_to_read_prompt(self, page: Page, research_datetime: str) -> str:
        """Get the prompt for checking if a page is allowed to be read"""
        return f"""You are a professional, experienced researcher. You are performing a research task. You need to assume you are at {research_datetime} and determine whether this page is allowed to be read based on the following criteria:
1. The page only contains information published before {research_datetime} (to prevent future information leakage)
2. The page only contains timeless information (general knowledge, concepts, etc.)
If the page is allowed to be read, output true, otherwise output false. If you cannot determine or are suspicious, please return false.

<page>
{page.title}  (Time: {page.modified_time}; URL: {page.url})
</page>
"""

    def get_extract_knowledge_prompt(
        self, final_goal: str, research_datetime: str, sub_goal: str, page: Page
    ) -> str:
        """Get the prompt for extracting knowledge from a source"""
        return f"""You are a professional, experienced researcher. The current time is {research_datetime}. Given a searched page, return a maximum of 2 learnings (called "knowledge items"), but feel free to return less if the contents are clear. Make sure each learning is unique and not similar to each other. The learnings should be concise and to the point, as detailed and information dense as possible. Make sure to include any entities like people, places, companies, products, things, etc in the learnings, as well as any exact metrics, numbers, or dates. The learnings will be used to research the topic further. Return an empty list if the page is irrelevant to the research goal or sub-goal.

<content>
{page.title}  (Time: {page.modified_time}; URL: {page.url})
{page.markdown[:self.content_max_length]}
</content>

<final_goal>
{final_goal}
</final_goal>

<current_sub_goal>
{sub_goal}
</current_sub_goal>
"""

    async def is_page_allowed_to_read(
        self, page: Page, research_datetime: str
    ) -> Tuple[str, bool]:
        """Check if a page is allowed to be read"""
        reasoning, res = await generate_dict(
            model=self.read_permission_model,
            prompt=self.get_page_allowed_to_read_prompt(page, research_datetime),
            schema=AllowedToReadSchema,
            with_reasoning=True,
        )
        return reasoning, res.allowed

    async def extract_knowledge_from_source(
        self, final_goal: str, research_datetime: str, sub_goal: str, page: Page
    ) -> Tuple[str, List[KnowledgeItem]]:
        """Extract new knowledge from the source"""
        reasoning, res = await generate_dict(
            model=self.extract_knowledge_model,
            prompt=self.get_extract_knowledge_prompt(
                final_goal, research_datetime, sub_goal, page
            ),
            schema=KnowledgeItemsSchema,
            with_reasoning=True,
        )
        return reasoning, [
            KnowledgeItem(content=k, sources=[page.url]) for k in res.knowledge
        ]

    async def process_page(
        self, page: Page, agent_args: AgentArgs
    ) -> List[KnowledgeItem]:
        """Process a single page to extract knowledge"""
        # If the page doesn't have a modified_time, use LLM to determine if it's allowed to read
        research_datetime_dt = datetime.strptime(
            agent_args.research_datetime, "%Y-%m-%d %H:%M:%S"
        )
        need_check_page_modified_time = research_datetime_dt < datetime.now() - timedelta(
            hours=1
        )
        if not page.modified_time:
            if need_check_page_modified_time:
                reasoning, allowed = await self.is_page_allowed_to_read(
                    page=page, research_datetime=agent_args.research_datetime
                )
                await agent_args._notify_progress(
                    ReasoningEvent(
                        reasoning=reasoning, action="is_page_allowed_to_read"
                    )
                )
                if not allowed:
                    await agent_args._notify_progress(
                        PageNotAllowedToReadEvent(page=page)
                    )
                    return []

        # Extract new knowledge from the source
        sub_goal = (
            agent_args.state.searched_queries[-1].sub_goal
            if agent_args.state.searched_queries
            else ""
        )
        reasoning, new_knowledge_items = await self.extract_knowledge_from_source(
            final_goal=agent_args.state.goal,
            research_datetime=agent_args.research_datetime,
            sub_goal=sub_goal,
            page=page,
        )
        await agent_args._notify_progress(
            ReasoningEvent(reasoning=reasoning, action="extract_knowledge_from_source")
        )
        return new_knowledge_items

    async def _execute(self, agent_args: AgentArgs) -> ExecutionResult:
        """Execute search and read operations"""
        # First perform the search
        sub_goal, query_strings, all_pages = await self._execute_search(agent_args)
        if not all_pages:
            return ExecutionResult.success(
                agent_code=self.code,
                result="No readable search result pages found",
                message="Search completed but no relevant pages found",
                ext={
                    "sub_goal": sub_goal,
                    "query_strings": query_strings,
                    "knowledge": [],
                },
            )

        # Create parallel tasks for reading all pages
        read_tasks = []
        # Notify that reading has started
        await agent_args._notify_progress(BatchReadStartEvent(pages=all_pages))
        for page in all_pages:
            read_tasks.append(self.process_page(page, agent_args))

        # Execute all reading tasks in parallel
        knowledge_items = await asyncio.gather(*read_tasks)
        knowledge_items = [item for items in knowledge_items for item in items]
        agent_args.state.knowledge.extend(knowledge_items)

        # Notify about new knowledge if any
        if knowledge_items:
            await agent_args._notify_progress(
                NewKnowledgeEvent(new_items=knowledge_items)
            )

        # Notify that reading is complete for all pages
        await agent_args._notify_progress(BatchReadCompleteEvent(pages=all_pages))

        return ExecutionResult.success(
            agent_code=self.code,
            result=f"Successfully completed search and read {len(all_pages)} pages",
            message=f"Search sub-goal: {sub_goal}, Queries: {query_strings}, Obtained {len(knowledge_items)} new knowledge items",
            ext={
                "sub_goal": sub_goal,
                "query_strings": query_strings,
                "knowledge": knowledge_items,
                "pages_read": len(all_pages),
            },
        )

    async def _execute_search(
        self, agent_args: AgentArgs
    ) -> Tuple[str, List[str], List[Page]]:
        """Execute the search operation part"""
        sub_goal = self.parameters["sub_goal"].value
        query_strings = self.parameters["query_strings"].value

        if not isinstance(query_strings, list):
            try:
                query_strings = json.loads(query_strings)
            except json.JSONDecodeError:
                raise ValueError("Invalid query strings format")

        all_pages = []
        all_queries = []

        # Create search query objects
        for query_string in query_strings:
            search_query = SearchQuery(
                sub_goal=sub_goal, query_string=query_string, search_result_pages=[]
            )

            agent_args.state.searched_queries.append(search_query)
            all_queries.append(search_query)

        # Notify that search is starting
        await agent_args._notify_progress(
            BatchSearchStartEvent(query_strings=query_strings, sub_goal=sub_goal)
        )

        # Create parallel tasks
        search_tasks = []
        for search_query in all_queries:
            search_tasks.append(self._execute_single_search(search_query, agent_args))

        # Execute all search tasks in parallel
        search_results = await asyncio.gather(*search_tasks)

        # Process search results
        for search_query, pages in search_results:
            all_pages.extend(pages)

        url_to_page = {page.url: page for page in all_pages}

        await agent_args._notify_progress(
            BatchSearchCompleteEvent(
                query_strings=query_strings,
                results=list(url_to_page.values()),
            )
        )

        # Update agent state
        agent_args.state.all_url_to_result_page.update(url_to_page)

        return sub_goal, query_strings, list(url_to_page.values())

    async def _execute_single_search(
        self, search_query: SearchQuery, agent_args: AgentArgs
    ) -> Tuple[SearchQuery, List[Page]]:
        """Execute a single search query"""
        # Execute search
        result = await asyncio.to_thread(firecrawl_search, search_query.query_string)

        # Process results
        pages = [
            Page(
                url=item["url"],
                title=item["title"],
                description=item["description"],
                markdown=item["markdown"],
                modified_time=item.get("modifiedTime", ""),
            )
            for item in result["data"]
            if item.get("url") and "markdown" in item
        ]

        # remove duplicate comparing to the previous search result pages
        pages = [
            page
            for page in pages
            if page.url not in agent_args.state.all_url_to_result_page
        ]

        # remove pages that are newer than the news datetime
        pages = [
            page
            for page in pages
            if not page.modified_time or page.modified_time <= agent_args.research_datetime
        ]  # keep pages without modified_time for now

        search_query.search_result_pages = pages

        return search_query, pages
