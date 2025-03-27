def prepare_tool_prompt(allowed_tools):
    splitter = "=" * 20
    tool_prompt = f"\n{splitter}\n".join([tool.get_prompt_with_parameter() for tool in allowed_tools])
    return tool_prompt