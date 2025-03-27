from deep_research_anything.models.event import ErrorEvent, ReasoningEvent, ResearchStateEvent
from deep_research_anything.agent.agent import ExecutionResult, Agent
from deep_research_anything.agent.agents.end._cross_validation import CrossValidator
from deep_research_anything.agent.agents.end._essay import EssayGenerator
from deep_research_anything.agent.agents.end.models import (
    GenerateEssayCompleteEvent,
    GenerateEssayStartEvent,
)
from deep_research_anything.agent.agent import Agent


class EndAgent(Agent):
    def __init__(
        self,
        name="End",
        code="end",
        description="End the current research process and generate the final report",
        documentation="""You should only choose 'end' when you have completed all of the following conditions:
1. You've fully analyzed the research topic from different angles
2. You've thoroughly examined all important factors and parts
3. You've included numbers and data to support your findings when possible
4. You've confirmed all information with at least 2 different sources
5. You've identified and assessed all major limitations and uncertainties""",
        cross_validator=CrossValidator(),
        essay_generator=EssayGenerator(),
        success_message="Research process ended, preparing to generate final report",
    ):
        super().__init__(
            name=name, code=code, description=description, documentation=documentation
        )
        self.cross_validator = cross_validator
        self.essay_generator = essay_generator
        self.success_message = success_message

    async def _execute(self, agent_args) -> ExecutionResult:
        """Execute the end operation, return the end signal"""
        # cross-validate knowledge for consistency
        max_rounds = 3
        if self.cross_validator:
            try:
                while await self.cross_validator.cross_validate_all_knowledge(agent_args):
                    max_rounds -= 1
                    if max_rounds <= 0:
                        break
            except Exception as e:
                import traceback
                await agent_args._notify_progress(
                    ErrorEvent(error=str(e), action="cross_validate_all_knowledge", traceback=traceback.format_exc())
                )

        await agent_args._notify_progress(
            ResearchStateEvent(research_state=agent_args.state)
        )

        essay = None
        if self.essay_generator:
            # Notify essay generation start
            await agent_args._notify_progress(GenerateEssayStartEvent())
            context, reasoning, essay = await self.essay_generator.generate(
                final_goal=agent_args.state.goal,
                research_datetime=agent_args.research_datetime,
                knowledge=agent_args.state.knowledge,
                cognition=agent_args.state.cognition,
            )
            await agent_args._notify_progress(
                ReasoningEvent(reasoning=reasoning, action="generate_essay")
            )
            # Notify essay generation complete
            await agent_args._notify_progress(GenerateEssayCompleteEvent(essay=essay))

        return ExecutionResult.success(
            agent_code=self.code,
            result={
                "end": True,
                "essay": essay,
                "final_knowledge": [k.model_dump() for k in agent_args.state.knowledge],
            },
            message=self.success_message,
        )
