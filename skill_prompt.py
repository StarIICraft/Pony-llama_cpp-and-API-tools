import json
from typing import Any


_ROOT_PROMPT_KEYS = (
    "system_prompt",
    "instructions",
    "instruction",
)

_SKILL_KEYS = (
    "skill",
    "skills",
)

_NESTED_PROMPT_KEYS = (
    "instructions",
    "instruction",
    "system_prompt",
    "content",
    "prompt",
    "text",
    "body",
    "markdown",
)


def _strip_code_fence(text: str) -> str:
    value = text.strip()

    if not value.startswith("```"):
        return value

    lines = value.splitlines()

    if not lines or not lines[-1].strip().startswith("```"):
        return value

    return "\n".join(lines[1:-1]).strip()


def _json_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    parsed = value

    # Text-preset and API nodes may return a JSON string that itself
    # contains another JSON-encoded string.
    for _ in range(3):
        if not isinstance(parsed, str):
            break

        text = _strip_code_fence(parsed)

        if not text or text[0] not in '"{[':
            break

        try:
            next_value = json.loads(text)
        except (TypeError, ValueError):
            break

        if next_value == parsed:
            break

        parsed = next_value

    return parsed


def _prompt_from_value(value: Any, allow_nested_json=True) -> str:
    value = _json_value(value) if allow_nested_json else value

    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, (int, float, bool)):
        return str(value)

    if isinstance(value, (list, tuple)):
        parts = [
            _prompt_from_value(item)
            for item in value
        ]

        return "\n\n".join(
            part for part in parts if part
        )

    if not isinstance(value, dict):
        return ""

    for key in _NESTED_PROMPT_KEYS:
        if key in value:
            prompt = _prompt_from_value(value[key])

            if prompt:
                return prompt

    return ""


def extract_skill_prompt(value: Any) -> str | None:
    """
    从明确的 skill/system_prompt JSON 结构中提取系统提示词。

    返回 None 表示没有识别到 skill 字段，调用方应保留原输入。
    返回值可能为空字符串，表示识别到了 skill 但内容为空。
    """
    parsed = _json_value(value)

    if not isinstance(parsed, dict):
        return None

    for key in _ROOT_PROMPT_KEYS:
        if key in parsed:
            return _prompt_from_value(parsed[key])

    for key in _SKILL_KEYS:
        if key in parsed:
            return _prompt_from_value(parsed[key])

    # 同时支持 {name, description, instructions} 形式的 skill 对象。
    if (
        "instructions" in parsed
        or "instruction" in parsed
    ):
        return _prompt_from_value(parsed)

    return None


def resolve_system_prompt(system_prompt: Any, skill: Any = None) -> str:
    """
    解析原文本框或独立 skill 输入。

    优先级：
        1. 独立 skill 非空时覆盖原系统提示词；
        2. 原系统提示词中的明确 JSON skill 字段；
        3. 原始系统提示词。
    """
    skill_text = str(skill or "").strip()

    if skill_text:
        extracted = extract_skill_prompt(skill_text)

        if extracted and extracted.strip():
            return extracted

        if extracted is None:
            return skill_text

    system_text = str(system_prompt or "")
    extracted = extract_skill_prompt(system_text)

    if extracted and extracted.strip():
        return extracted

    return system_text
