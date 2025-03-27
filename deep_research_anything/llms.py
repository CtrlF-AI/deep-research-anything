import asyncio
import json
import os
from typing import Type
from litellm import acompletion
from dotenv import load_dotenv
from pydantic import BaseModel
import re

load_dotenv()

# Get API key and endpoint from environment variables
api_key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("OPENAI_ENDPOINT", "https://api.openai.com/v1")


# Configure o3-mini model
O3MiniModel = {
    "model": "o3-mini",
    "api_key": api_key,
    "api_base": api_base,
    "max_tokens": 4000,
    "temperature": 1,
}

O3MiniModelForEssay = {
    "model": "o3-mini",
    "api_key": api_key,
    "api_base": api_base,
    "max_tokens": 32768,
    "temperature": 1,
}


def fix_json_string(json_str):
    """
    Fix unescaped double quotes in JSON string values.
    For example, convert ...ocean decade conference"international cooperation"theme... to ...ocean decade conference\"international cooperation\"theme...
    """
    # First, if there are no enclosing braces, add them and try
    if not json_str.startswith("{"):
        json_str = "{" + json_str
    if not json_str.endswith("}"):
        json_str = json_str + "}"
    # Regular expression:
    # 1. (:\s*") captures the colon followed by a quote (start of value)
    # 2. ((?:.|\n)*?) non-greedy capture of internal content (allowing newlines)
    # 3. ("(?=\s*[,}\]])) captures the closing quote followed by comma, } or ]
    pattern = re.compile(r'(:\s*")((?:.|\n)*?)("(?=\s*[,}\]]))')

    def replacer(match):
        prefix = match.group(1)  # contains colon and opening quote
        content = match.group(2)  # content inside the value string
        suffix = match.group(3)  # closing quote

        # Process internal content: replace unescaped " with \" using negative lookbehind
        fixed_content = re.sub(r'(?<!\\)"', r'\\"', content)
        return prefix + fixed_content + suffix

    return pattern.sub(replacer, json_str)


async def generate_text(
    *,
    model: dict,
    system: str | None = None,
    prompt: str,
    context: list[dict[str, str]] | None = None,
    with_reasoning: bool = False,
):
    """
    Generate a text response using litellm
    """
    """
    Generate a text response using litellm

    Args:
        model: Model configuration dictionary
        system: System prompt
        prompt: User prompt
        context: Context for the prompt
        with_reasoning: Whether to return reasoning along with the response
    """
    if model["model"].startswith("deepseek"):
        messages = [{"role": "user", "content": prompt}]
    else:
        messages = [{"role": "user", "content": prompt}]

    print("\033[32mPrompt: \n", messages[0]["content"], "\033[0m")

    if context is not None:
        if system and not context:
            messages.insert(0, {"role": "system", "content": f"{system}"})
        context.extend(messages)
        messages = context
    else:
        if system:
            messages.insert(0, {"role": "system", "content": f"{system}"})

    response = await acompletion(messages=messages, **model)

    content = response.choices[0].message.content.strip()
    message = response.choices[0].message
    reasoning = (
        (message.get("provider_specific_fields") or {}).get("reasoning_content", "")
        if isinstance(message, dict)
        else (
            (getattr(message, "provider_specific_fields") or {}).get(
                "reasoning_content", ""
            )
            if message
            else ""
        )
    )

    if with_reasoning:
        return reasoning, content
    return content


async def generate_dict(
    *,
    model: dict,
    system: str | None = None,
    prompt: str,
    schema: Type[BaseModel],
    context: list[dict[str, str]] | None = None,
    with_reasoning: bool = False,
    max_retries: int = 3,
) -> BaseModel:
    """
    Generate a structured response using litellm

    Args:
        model: Model configuration dictionary
        system: System prompt
        prompt: User prompt
        schema: JSON schema for response validation
        context: Context for the prompt
        with_reasoning: Whether to return reasoning along with the response
        max_retries: Maximum number of retry attempts if JSON parsing fails
    """
    messages = [{"role": "user", "content": prompt}]
    print("\033[32mPrompt: \n", messages[0]["content"], "\033[0m")
    if context is not None:
        if system and not context:
            messages.insert(0, {"role": "system", "content": f"{system}"})
        context.extend(messages)
        messages = context
    else:
        if system:
            messages.insert(0, {"role": "system", "content": f"{system}"})

    retry_count = 0
    while retry_count < max_retries:
        try:
            response = await acompletion(messages=messages, response_format=schema, **model)
            json_content = response.choices[0].message.content.strip()
            if json_content.startswith("```json"):
                json_content = json_content[len("```json") :].strip()
            if json_content.endswith("```"):
                json_content = json_content[: -len("```")]
            message = response.choices[0].message
            # TODO: support claude litellm
            reasoning = (
                (message.get("provider_specific_fields") or {}).get(
                    "reasoning_content", ""
                )
                if isinstance(message, dict)
                else (
                    (getattr(message, "provider_specific_fields") or {}).get(
                        "reasoning_content", ""
                    )
                    if message
                    else ""
                )
            )

            parsed_json = json.loads(fix_json_string(json_content.strip()))
            # create a pydantic model from the schema
            pydantic_model = schema.model_validate(parsed_json)

            if context:
                context.append(
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            parsed_json, indent=2, ensure_ascii=False
                        ),
                    }
                )

            if with_reasoning:
                return reasoning, pydantic_model
            else:
                return pydantic_model

        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries:
                raise e
            print(
                f"\033[31mRetry {retry_count}/{max_retries}: JSON parsing failed: {e}\033[0m"
            )


if __name__ == "__main__":

    class TestSchema(BaseModel):
        test: str

    print(
        asyncio.run(
            generate_dict(
                model=O3MiniModel,
                prompt="Test prompt",
                schema=TestSchema,
            )
        )
    )
