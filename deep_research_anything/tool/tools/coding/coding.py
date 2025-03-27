import asyncio
from deep_research_anything.tool.tool_parameter import ToolParameter
from deep_research_anything.tool.tool import ToolExecutionResult, Tool
from deep_research_anything.tool.tools.coding.models import CodeExecutionEvent
from deep_research_anything.tool.tools.coding.utils import run_code_manager


class CodeParameter(ToolParameter):
    def __init__(self):
        super().__init__(
            name="Code",
            code="code_str",
            description="Input Python code as a string for execution. Print results or store them in a result variable to view output.",
        )


class Coding(Tool):
    def __init__(self):
        super().__init__(
            name="coding",
            code="coding",
            description="Input Python code, execute it and get the results",
            documentation="""""",
        )
        self.parameters = {"code_str": CodeParameter()}

    async def _execute(self, agent_args) -> ToolExecutionResult:
        code_result = await asyncio.to_thread(
            run_code_manager, self.parameters["code_str"].value, timeout=60
        )
        event = CodeExecutionEvent(code_result=code_result)
        await agent_args._notify_progress(event)
        return ToolExecutionResult.success(
            tool_code=self.code, result=code_result, message="Code execution completed"
        )
