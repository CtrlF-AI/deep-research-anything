from pydantic import BaseModel
from deep_research_anything.models.event import BaseEvent


class CodeResult(BaseModel):
    code_str: str
    status: str
    output: str
    stdout: str


class CodeExecutionEvent(BaseEvent):
    code_result: CodeResult
