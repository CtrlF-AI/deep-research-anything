from deep_research_anything.tool.tool_parameter import ToolParameter
from deep_research_anything.models.base import AgentArgs
from deep_research_anything.models.event import ErrorEvent


class ToolExecutionResult:
    def __init__(self, tool_code, status="success", result=None, message="", ext=None):
        self.tool_code = tool_code
        self.status = status  # "success", "error", "warning", etc.
        self.result = result  # execution result data
        self.message = message  # descriptive message
        self.ext = ext or {}  # dictionary for additional information

    def to_dict(self):
        return {
            "agent_code": self.tool_code,
            "status": self.status,
            "result": self.result,
            "message": self.message,
            "ext": self.ext,
        }

    @classmethod
    def error(cls, tool_code: str, message: str, ext: object = None) -> object:
        return cls(tool_code=tool_code, status="error", message=message, ext=ext)

    @classmethod
    def success(cls, tool_code, result=None, message="", ext=None):
        return cls(
            tool_code=tool_code,
            status="success",
            result=result,
            message=message,
            ext=ext,
        )


class Tool:
    def __init__(self, name, description, code=None, documentation=""):
        self.name = name
        self.code = name if not code else code
        self.description = description
        self.documentation = documentation
        # TODO: Change to ParameterRegistry format
        self.parameters = {}

    def _debug_print(self, *args):
        print(*args)

    def add_parameter(self, parameter: ToolParameter):
        if parameter.code in self.parameters:
            raise ValueError(
                f"Parameter code '{parameter.code}' already exists, cannot add duplicate."
            )
        self.parameters[parameter.code] = parameter

    async def execute(self, agent_args: AgentArgs):
        # Check if ready
        if not self.is_ready():
            not_ready_parameters = [
                parameter.code
                for para_code, parameter in self.parameters.items()
                if not parameter.is_ready()
            ]
            return ToolExecutionResult.error(
                tool_code=self.code,
                message=f"Parameters not ready: {', '.join(not_ready_parameters)}",
            )
        self.log_parameters()
        try:
            execution_result = await self._execute(agent_args)
        except Exception as e:
            import traceback

            traceback.print_exc()
            execution_result = ToolExecutionResult.error(
                tool_code=self.code,
                message=f"Agent execution error, error message: {e}",
            )
            await agent_args._notify_progress(
                ErrorEvent(error=str(e), action=f"execute: {self.code}", traceback=traceback.format_exc())
            )
        return execution_result

    async def _execute(self, agent_args: AgentArgs):
        """
        Subclasses should override this method to implement specific execution logic
        """
        return ToolExecutionResult.success(
            tool_code=self.code, message="Agent executed successfully"
        )

    def _get_prompt(self):
        return f"""- Agent Name: {self.name}
- Agent Code: {self.code}
- Agent Description: {self.description}
- Agent Detailed Documentation: {self.documentation}"""

    def get_prompt_with_parameter(self):
        prompt = (
            f"[Agent]\n{self._get_prompt()}"
            + "\n[Parameters]\n"
            + "\n".join(
                [
                    f"Parameter {i} information: {para[1].get_prompt()}"
                    for i, para in enumerate(self.parameters.items())
                ]
            )
        )
        return prompt

    def is_ready(self):
        for para_code, parameter in self.parameters.items():
            if not parameter.is_ready():
                return False
        return True

    def reset(self):
        for para_code, parameter in self.parameters.items():
            parameter.reset()

    def get_str_parameters(self):
        return "\n".join(
            [
                f"- {parameter.code}: {parameter.value}"
                for para_code, parameter in self.parameters.items()
            ]
        )

    def log_parameters(self):
        self._debug_print(
            f"Agent {self.code} executing with parameters:\n{self.get_str_parameters()}"
        )
