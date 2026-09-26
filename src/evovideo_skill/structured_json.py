from __future__ import annotations

import json
import re
from typing import Any


class StructuredJSONError(ValueError):
    pass


def parse_json_object(content: Any) -> dict[str, Any]:
    """Extract one complete JSON object from common LLM response wrappers."""
    if isinstance(content, dict):
        return content
    if isinstance(content, list):
        content = "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        )
    text = str(content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = _decode_embedded_object(text)
    if not isinstance(payload, dict):
        raise StructuredJSONError("structured response must be a JSON object")
    return payload


def _decode_embedded_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    if text.count("{") > text.count("}"):
        raise StructuredJSONError("LLM JSON object appears truncated before its closing brace")
    raise StructuredJSONError("LLM response contains no complete JSON object")
