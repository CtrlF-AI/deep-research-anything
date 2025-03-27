from typing import List, Tuple
from deep_research_anything.models.base import KnowledgeItem
from deep_research_anything.models.event import ReasoningEvent
from deep_research_anything.llms import O3MiniModel, generate_dict
import asyncio

from deep_research_anything.agent.agents.end.models import ConflictingPairsSchema, RefinedKnowledgeEvent, ResolvedKnowledgeSchema
from deep_research_anything.agent.agents.search.models import Page


class CrossValidator:
    def __init__(self, model=O3MiniModel):
        self.model = model

    async def cross_validate_all_knowledge(self, agent_args) -> bool:
        """Cross-validate all knowledge after research is completed"""
        #  it must ensure agent_args.state.all_url_to_result_page exists
        if not hasattr(agent_args.state, "all_url_to_result_page"):
            return

        # First, find all conflicting knowledge pairs
        reasoning, conflicting_pairs = await self.find_conflicting_knowledge_pairs(
            knowledge_items=agent_args.state.knowledge,
            research_datetime=agent_args.research_datetime,
            final_goal=agent_args.state.goal,
        )

        await agent_args._notify_progress(
            ReasoningEvent(
                reasoning=reasoning, action="find_conflicting_knowledge_pairs"
            )
        )

        if not conflicting_pairs:
            return False  # No conflicting knowledge pairs, no need for cross-validation

        # Create parallel tasks to process each conflicting knowledge pair
        tasks = []
        for pair in conflicting_pairs:
            tasks.append(
                self.resolve_knowledge_conflict(
                    knowledge_index1=pair[0],
                    knowledge_index2=pair[1],
                    knowledge_item1=agent_args.state.knowledge[pair[0]],
                    knowledge_item2=agent_args.state.knowledge[pair[1]],
                    agent_args=agent_args,
                )
            )

        # Execute all tasks in parallel
        results = await asyncio.gather(*tasks) if tasks else []

        # Process results
        original_items = []
        refined_knowledge_items = []

        # Create a set to track knowledge item indices to be removed
        indices_to_remove = set()

        for knowledge_indices, reasoning_resolve, resolved_item in results:
            if resolved_item:  # Ensure there is an actual resolution
                # Record the reasoning process for resolving conflicts
                await agent_args._notify_progress(
                    ReasoningEvent(
                        reasoning=reasoning_resolve, action="resolve_knowledge_conflict"
                    )
                )

                # Save original knowledge items
                original_items.append(agent_args.state.knowledge[knowledge_indices[0]])
                original_items.append(agent_args.state.knowledge[knowledge_indices[1]])

                # Add resolved knowledge item
                refined_knowledge_items.append(resolved_item)

                # Mark indices to be removed
                indices_to_remove.add(knowledge_indices[0])
                indices_to_remove.add(knowledge_indices[1])

        # Create a new knowledge list
        new_knowledge = []

        # Add all original knowledge items that don't need to be removed (i.e., items that don't need cross-validation)
        for i, item in enumerate(agent_args.state.knowledge):
            if i not in indices_to_remove:
                new_knowledge.append(item)

        # Add all resolved knowledge items
        new_knowledge.extend(refined_knowledge_items)

        # Update knowledge base
        agent_args.state.knowledge = new_knowledge

        # Notify about refined knowledge (if any)
        if refined_knowledge_items:
            await agent_args._notify_progress(
                RefinedKnowledgeEvent(
                    original_items=original_items, refined_items=refined_knowledge_items
                )
            )

        return True

    def get_conflicting_pairs_prompt(
        self, knowledge_list: str, research_datetime: str, final_goal: str
    ) -> str:
        return f"""You are a professional, experienced researcher. The current time is {research_datetime}. You need to identify knowledge pairs with logical or data conflicts from the following knowledge items.

<Final Goal>
{final_goal}
</Final Goal>

<All Knowledge Items>
{knowledge_list}
</All Knowledge Items>

<Task>
1. Carefully analyze all knowledge items
2. Identify knowledge pairs with logical contradictions or data conflicts
3. Return all conflicting knowledge pairs in a list. Each pair should have an index1 and an index2 as the index of the knowledge items in the knowledge list.

If there is no conflict, return an empty list.
</Task>
"""

    async def find_conflicting_knowledge_pairs(
        self, knowledge_items: List[KnowledgeItem], research_datetime: str, final_goal: str
    ) -> Tuple[str, List[Tuple[int, int]]]:
        """Find all knowledge pairs with logical or data conflicts"""
        if len(knowledge_items) < 2:
            return "Less than 2 knowledge items, no need to check for conflicts", []

        # Build a list of all knowledge items with their numbers
        knowledge_list = "\n".join(
            [
                f"Knowledge item #{i}: {item.content}"
                for i, item in enumerate(knowledge_items)
            ]
        )

        prompt = self.get_conflicting_pairs_prompt(
            knowledge_list, research_datetime, final_goal
        )

        reasoning, res = await generate_dict(
            model=self.model,
            prompt=prompt,
            schema=ConflictingPairsSchema,
            with_reasoning=True,
        )

        return reasoning, [
            (pair.index1, pair.index2) for pair in res.conflicting_pairs
        ]
    
    def _put_related_sources(self, source_pages: List[Page], content_max_length: int) -> str:
        # Calculate total length and determine if we need to cut content
        total_chars = sum(len(page.markdown) for page in source_pages)
        # 1 word ≈ 2.5 tokens
        estimated_tokens = sum(len(page.markdown.split(' ')) for page in source_pages) * 2.5
        
        if estimated_tokens > content_max_length and len(source_pages) > 0:
            # Calculate the ratio to keep (e.g., if we need to cut 3/8, we keep 5/8)
            keep_ratio = content_max_length / estimated_tokens
            
            # Create a list of truncated page content based on the ratio
            truncated_sources = []
            for i, page in enumerate(source_pages):
                # Calculate how much of this page to keep
                chars_to_keep = int(len(page.markdown) * keep_ratio)
                truncated_content = page.markdown[:chars_to_keep]
                truncated_sources.append(
                    f'Source {i}: {chr(10)}{page.title}  (Time: {page.modified_time}; URL: {page.url}) '
                    f'{chr(10)}{truncated_content}{chr(10)}'
                )
            
            return "\n".join(truncated_sources)
        else:
            # If no truncation needed, use the original content_max_length
            return "\n".join(
                f'Source {i}: {chr(10)}{page.title}  (Time: {page.modified_time}; URL: {page.url}) '
                f'{chr(10)}{page.markdown[:content_max_length]}{chr(10)}'
                for i, page in enumerate(source_pages)
            )

    def get_resolve_conflict_prompt(
        self,
        knowledge_item1: KnowledgeItem,
        knowledge_item2: KnowledgeItem,
        agent_args,
        source_pages: List[Page],
        content_max_length: int,
    ) -> str:
        related_sources = self._put_related_sources(source_pages, content_max_length)
        return f"""You are a professional, experienced researcher. The current time is {agent_args.research_datetime}. You need to resolve conflicts between two knowledge items.

<Final Goal>
{agent_args.state.goal}
</Final Goal>

<Conflicting Knowledge Items>
Knowledge Item 1: {knowledge_item1.content}
Knowledge Item 2: {knowledge_item2.content}
</Conflicting Knowledge Items>

<Relevant Sources>
{related_sources}
</Relevant Sources>

<Task>
1. Analyze the points of conflict between the two knowledge items
2. Evaluate the reliability and timeliness of each source
3. Synthesize all information to resolve the conflict
4. Provide a new, more accurate knowledge item that integrates correct information from conflicting items

Return a solid reasoning and the text of the resolved knowledge item. Include all numerical data and facts, as well as logical inferences that serve the goal.
</Task>
"""

    async def resolve_knowledge_conflict(
        self,
        knowledge_index1: int,
        knowledge_index2: int,
        knowledge_item1: KnowledgeItem,
        knowledge_item2: KnowledgeItem,
        agent_args,
        content_max_length: int = 40000,
    ) -> Tuple[Tuple[int, int], str, KnowledgeItem]:
        """Resolve conflicts between two knowledge items"""
        # Get all relevant source pages
        all_sources = list(set(knowledge_item1.sources + knowledge_item2.sources))
        source_pages = [agent_args.state.all_url_to_result_page[s] for s in all_sources]

        prompt = self.get_resolve_conflict_prompt(
            knowledge_item1,
            knowledge_item2,
            agent_args,
            source_pages,
            content_max_length,
        )

        reasoning, res = await generate_dict(
            model=self.model,
            prompt=prompt,
            schema=ResolvedKnowledgeSchema,
            with_reasoning=True,
        )

        # Create a new knowledge item that includes all sources
        return (
            (knowledge_index1, knowledge_index2),
            reasoning,
            KnowledgeItem(content=res.resolved_knowledge, sources=all_sources),
        )
