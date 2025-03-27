import asyncio
from typing import List, Optional, Callable, Type
from deep_research_anything.llms import O3MiniModel
from deep_research_anything.models.base import AgentArgs, BaseResearchState
from deep_research_anything.models.default import DefaultResearchState
from deep_research_anything.models.event import (
    BaseEvent,
    ReasoningEvent,
    AgentSelectionEvent,
)
from deep_research_anything.agent.call import AgentSelector
from dotenv import load_dotenv
from deep_research_anything.agent.agent import ExecutionResult, Agent
from deep_research_anything.agent.agents.end import EndAgent
from deep_research_anything.agent.agents.search import SearchAgent
from deep_research_anything.agent.registry import AgentRegistry

load_dotenv()


class Research:
    def __init__(
        self,
        *,
        goal: str,
        research_datetime: str,
        research_state_cls: Type[BaseResearchState] = DefaultResearchState,
        agents: List[Agent] | None = None,
        agent_selector: AgentSelector | None = None,
        progress_callback: Optional[Callable[[BaseEvent], None]] = None,
    ):
        self.state = research_state_cls(goal=goal)
        self.progress_callback = progress_callback
        self.research_datetime = research_datetime
        self.agent_selector = agent_selector or AgentSelector(O3MiniModel)
        if agents is None:
            agents = [
                SearchAgent(),
                EndAgent(),
            ]  # in default, only search and end agents are allowed
        for agent in agents:
            AgentRegistry.register(agent)

    async def _notify_progress(self, event: BaseEvent):
        if self.progress_callback:
            # Check if progress_callback is an async function
            if asyncio.iscoroutinefunction(self.progress_callback):
                await self.progress_callback(event)
            else:
                # Run callback in a non-blocking way
                # Using run_in_executor instead of to_thread to avoid tracemalloc warning
                self.progress_callback(event)

    async def _debug_print(self, *args):
        print(*args)

    def get_agent_args(self):
        return AgentArgs(
            state=self.state,
            research_datetime=self.research_datetime,
            _notify_progress=self._notify_progress,
        )

    async def select_and_execute_agent(
        self, allow_end: bool = False, force_end: bool = False
    ) -> ExecutionResult:
        allowed_agents = list(AgentRegistry.get_all_agents())
        if not allow_end:
            allowed_agents = [agent for agent in allowed_agents if agent.code != "end"]

        if force_end:
            # If force_end is True, only allow the end agent
            allowed_agents = [agent for agent in allowed_agents if agent.code == "end"]

        reasoning, agent_selection = (
            await self.agent_selector.select_agent_and_parameters(
                state=self.state,
                research_datetime=self.research_datetime,
                allowed_agents=allowed_agents,
            )
        )

        await self._notify_progress(
            ReasoningEvent(reasoning=reasoning, action="select_agent_and_parameters")
        )
        await self._notify_progress(
            AgentSelectionEvent(agent_selection=agent_selection)
        )

        # Update cognition
        self.state.cognition.content = agent_selection.cognition

        # Get selected agent
        selected_agent_code = agent_selection.selected_agent
        agent = AgentRegistry.get_agent(selected_agent_code)

        if not agent:
            result = ExecutionResult.error(
                agent_code=selected_agent_code,
                message=f"Agent not found: {selected_agent_code}",
            )
            return result

        # Set agent parameters
        for param in agent_selection.parameters:
            if param.code in agent.parameters:
                agent.parameters[param.code].set_value(param.value)

        result = await agent.execute(self.get_agent_args())
        return result

    async def run_research(
        self, max_trajectory_length: int = 3, min_trajectory_length: int = 1
    ) -> dict:
        """Run the research process with agent-based trajectory"""
        trajectory_length = 0

        while True:
            self._debug_print(f"Trajectory: {trajectory_length}")

            # Only allow end agent if this is the last possible step
            allow_end = (
                trajectory_length >= min_trajectory_length
                and trajectory_length >= max_trajectory_length - 1
            )

            result = await self.select_and_execute_agent(allow_end=allow_end)
            self.state.execution_results.append(result)

            # Force end if we've reached max trajectory length
            if (
                trajectory_length >= max_trajectory_length - 1
                and result.agent_code != "end"
            ):
                # Execute end agent as the final step
                result = await self.select_and_execute_agent(
                    allow_end=True, force_end=True
                )
                self.state.execution_results.append(result)
                return result.result

            if result.agent_code == "end":
                return result.result

            trajectory_length += 1


if __name__ == "__main__":
    result = asyncio.run(
        Research(
            goal="The microsoft published Majorana 1, a quant chip. What stock will be influenced in a time span of a few days to a week? （i.e. Which stocks will rise and which stocks will fall?）",
            research_datetime="2025-02-24 12:00:00",
        ).run_research()
    )
