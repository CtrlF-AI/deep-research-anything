from typing import Dict, List, Tuple, Any
from pydantic import BaseModel, Field
from deep_research_anything.models.base import AgentSelectorSchema, BaseResearchState
from deep_research_anything.agent.agent import Agent
from deep_research_anything.llms import O3MiniModel, generate_dict


class AgentSelector:
    def __init__(self, model: Dict[str, Any] = O3MiniModel):
        self.model = model

    def prepare_agent_prompt(self, allowed_agents: List[Agent]) -> str:
        splitter = "=" * 20
        agent_prompt = f"\n{splitter}\n".join(
            [agent.get_prompt_with_parameter() for agent in allowed_agents]
        )
        return agent_prompt

    async def select_agent_and_parameters(
        self, state: BaseResearchState, research_datetime: str, allowed_agents: List[Agent]
    ) -> Tuple[str, AgentSelectorSchema]:
        if len(allowed_agents) == 1 and allowed_agents[0].code == "end":
            return (
                "Only end agent is allowed to be selected",
                AgentSelectorSchema(
                    selected_agent=allowed_agents[0].code,
                    parameters=[],
                    cognition=state.cognition.content,
                ),
            )
        final_goal = state.goal
        knowledge = state.knowledge
        last_execution_result = (
            state.execution_results[-1] if state.execution_results else None
        )
        cognition = state.cognition.content
        all_searched_queries = state.searched_queries
        current_search_query = (
            all_searched_queries[-1] if all_searched_queries else None
        )
        sub_goal = current_search_query.sub_goal if current_search_query else ""

        agent_prompt = self.prepare_agent_prompt(allowed_agents)
        reasoning, res = await generate_dict(
            model=self.model,
            prompt=f"""You are a professional, experienced researcher. The current time is {research_datetime}. Based on the research goal, you need to select the next agent to use, specify its parameters, and update your thinking based on new information.

<Final Goal>
{final_goal}
</Final Goal>

<Current Sub-Goal>
{sub_goal}
</Current Sub-Goal>

<Knowledge Accumulated>
{chr(10).join(f'- {item.content}' for item in knowledge) if knowledge else 'No research findings yet'}
</Knowledge Accumulated>

<Previous Searches>
{chr(10).join(f'- {query.query_string}' for query in all_searched_queries)}
</Previous Searches>

{f'''
<Last Agent Execution Result>
{last_execution_result.to_dict()} 
</Last Agent Execution Result>''' if last_execution_result else ''}

<Available Agents and Parameters>
{agent_prompt}
</Available Agents and Parameters>

<Current Thinking>
{cognition}
</Current Thinking>
""",
            schema=AgentSelectorSchema,
            with_reasoning=True,
        )
        return reasoning, res
