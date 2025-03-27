import threading
import io
import sys

from deep_research_anything.tool.tools.coding.models import CodeResult


def run_code_manager(code_str: str, timeout=60) -> CodeResult:
    code_result = CodeResult(code_str=code_str, status="success", output="", stdout="")
    thread = threading.Thread(target=run_code, args=(code_str, code_result))
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        code_result.status = "error"
        code_result.output = f"Execution exceeded {timeout} seconds timeout."
    return code_result


def run_code(code_str: str, code_result: CodeResult):
    try:
        # Redirect stdout to capture print statements
        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output

        exec_globals = {}
        exec(code_str, exec_globals)

        # Reset stdout
        sys.stdout = old_stdout

        # Capture the output
        code_result.status = "success"
        code_result.stdout = redirected_output.getvalue()
        code_result.output = exec_globals.get("result", "")
    except Exception as e:
        # Reset stdout in case of exception
        sys.stdout = old_stdout
        code_result.stdout = ""
        code_result.status = "error"
        code_result.output = str(e)

    return code_result
