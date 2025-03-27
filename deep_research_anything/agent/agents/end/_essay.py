from typing import List, Tuple
from deep_research_anything.agent.agents.end.models import EssaySchema
from deep_research_anything.models.base import KnowledgeItem, Cognition
from deep_research_anything.llms import O3MiniModelForEssay, generate_dict, generate_text


class EssayGenerator:
    def __init__(self, model=O3MiniModelForEssay):
        self.model = model

    def get_essay_prompt(
        self,
        final_goal: str,
        research_datetime: str,
        knowledge: List[KnowledgeItem],
        cognition: Cognition,
    ) -> str:
        return f"""The current time is {research_datetime}. Given the following prompt from the user, write a final report on the topic using the learnings from research. Make it as as detailed as possible, aim for 3 or more pages, include ALL the learnings from research:


<prompt>
{final_goal}
</prompt>

<learnings>
{chr(10).join(f'- {k.content}' for k in knowledge)}
</learnings>

<additional_insights>
{cognition.content}
</additional_insights>
"""

    async def generate(
        self,
        final_goal: str,
        research_datetime: str,
        knowledge: List[KnowledgeItem],
        cognition: Cognition,
    ) -> Tuple[str, str, str]:
        context = []
        prompt = self.get_essay_prompt(final_goal, research_datetime, knowledge, cognition)

        reasoning, res = await generate_dict(
            model=self.model, prompt=prompt, 
            schema=EssaySchema,
            with_reasoning=True, context=context
        )

        all_urls = set(
            url for knowledge_item in knowledge for url in knowledge_item.sources
        )
        urls_section = "\n\n## Sources\n\n" + "\n".join(f"- {url}" for url in all_urls)
        return context, reasoning, res.essay + urls_section
