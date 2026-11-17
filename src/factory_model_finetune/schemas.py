"""The JSON schemas the factory uses for each model role, plus a validator for them.

Copied from the factory's pipeline.py, evidence.py and memory.py so examples are checked against
the same shapes the factory enforces.
"""

import json
import re
from typing import Any

from factory_model_finetune.roles import PROFILES

Schema = dict[str, Any]

SCHEMA_FILTER = "schema"  # name under which schema rejections are reported
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S)
SCHEMA_MARKER = "matches this JSON Schema:\n"


def object_schema(props: Schema) -> Schema:
    """Build a JSON Schema object that requires all listed properties and forbids extras.

    Args:
        props: Mapping of property names to their sub-schemas.

    Returns:
        A schema dict enforcing strict structure (no additional properties).
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(props),
        "properties": props,
    }


STR: Schema = {"type": "string"}
STRS: Schema = {"type": "array", "items": STR}
# Reusable schema for simple id/text pairs used in acceptance criteria, edge cases, etc.
ITEM = object_schema({"id": STR, "text": STR})

# `goal` leads the spec because the factory's spec prompt asks for it first and renders it in the
# "What you asked for" section; the copy of pipeline.py this was built from predates that field.
SPEC_SCHEMA = object_schema(
    {
        "goal": STR,
        "title": STR,
        "summary": STR,
        "acceptance_criteria": {"type": "array", "items": ITEM},
        "edge_cases": {"type": "array", "items": ITEM},
        "out_of_scope": STRS,
        "assumptions": STRS,
        "architecture": object_schema(
            {
                "overview": STR,
                "components": {
                    "type": "array",
                    "items": object_schema({"name": STR, "responsibility": STR}),
                },
                "data_model": STR,
                "decisions": {
                    "type": "array",
                    "items": object_schema({"decision": STR, "why": STR}),
                },
            }
        ),
    }
)
PLAN_SCHEMA = object_schema(
    {
        "tasks": {
            "type": "array",
            "items": object_schema(
                {
                    "id": STR,
                    "title": STR,
                    "details": STR,
                    "files": STRS,
                    "kind": {"type": "string", "enum": ["code", "trivial"]},
                    "tests": STRS,
                    "covers": STRS,
                }
            ),
        }
    }
)
EDITS_SCHEMA = object_schema(
    {
        "edits": {
            "type": "array",
            "items": object_schema({"path": STR, "find": STR, "replace": STR}),
        },
        "note": STR,
    }
)
REVIEW_SCHEMA = object_schema(
    {
        "verdict": {"type": "string", "enum": ["pass", "changes"]},
        "summary": STR,
        "findings": {
            "type": "array",
            "items": object_schema(
                {
                    "severity": {"type": "string", "enum": ["blocker", "major", "minor", "nit"]},
                    "file": STR,
                    "line": {"type": "integer"},
                    "issue": STR,
                    "suggestion": STR,
                }
            ),
        },
    }
)
MAPPING_SCHEMA = object_schema(
    {
        "criteria": {
            "type": "array",
            "items": object_schema({"id": STR, "tests": STRS, "notes": STR}),
        },
        "edge_cases": {
            "type": "array",
            "items": object_schema({"id": STR, "handling": STR, "tests": STRS}),
        },
    }
)
MEMORY_SCHEMA = object_schema({"you": STRS, "decisions": STRS})


def detect_profile_schema(options: list[str]) -> Schema:
    """Schema for the profile-detection reply; `profile` is limited to the options shown.

    Args:
        options: List of valid profile names the model may choose from.

    Returns:
        A schema requiring a valid profile selection and a justification string.
    """
    return object_schema({"profile": {"type": "string", "enum": options}, "why": STR})


# Roles whose reply is JSON. The others (summarize, commit_message, pr_body) reply in plain text.
ROLE_SCHEMAS: dict[str, Schema] = {
    "spec": SPEC_SCHEMA,
    "plan": PLAN_SCHEMA,
    "review": REVIEW_SCHEMA,
    "trivial_edit": EDITS_SCHEMA,
    "docs_pass": EDITS_SCHEMA,
    "evidence": MAPPING_SCHEMA,
    "memory": MEMORY_SCHEMA,
    "detect_profile": detect_profile_schema(list(PROFILES)),
}


def schema_problem(value: Any, schema: Schema, path: str = "$") -> str | None:
    """Return the first validation error for `value` against `schema`, or None if valid.

    This implements a subset of JSON Schema validation sufficient for the factory's needs.
    It checks type, required fields, additional properties, and enum constraints recursively.

    Args:
        value: The data to validate.
        schema: The JSON Schema to validate against.
        path: Dot-notation path prefix for error messages (e.g., "$", "$.tasks[0]").

    Returns:
        An error message string describing the first failure, or None if the value is valid.
    """
    kind = schema.get("type")
    if kind == "object":
        if not isinstance(value, dict):
            return f"{path} should be an object"
        for key in schema.get("required", []):
            if key not in value:
                return f"{path}.{key} is missing"
        properties = schema.get("properties") or {}
        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in properties:
                    return f"{path}.{key} is not allowed"
        for key, sub_schema in properties.items():
            if key in value and (
                problem := schema_problem(value[key], sub_schema, f"{path}.{key}")
            ):
                return problem
    elif kind == "array":
        if not isinstance(value, list):
            return f"{path} should be an array"
        for i, item in enumerate(value):
            if problem := schema_problem(item, schema.get("items") or {}, f"{path}[{i}]"):
                return problem
    elif kind == "string" and not isinstance(value, str):
        return f"{path} should be a string"
    elif kind == "integer" and not (isinstance(value, int) and not isinstance(value, bool)):
        return f"{path} should be an integer"
    elif kind == "boolean" and not isinstance(value, bool):
        return f"{path} should be a boolean"
    if "enum" in schema and value not in schema["enum"]:
        return f"{path} should be one of {schema['enum']}"
    return None


def parse_json_reply(text: str) -> Any:
    """Extract and parse a JSON object from a model's reply, stripping markdown fences and prose.

    The model may wrap the JSON in ```json ... ``` or include explanatory text. This function
    removes think blocks, strips surrounding whitespace, and attempts direct parsing first.
    If that fails, it searches for the first '{' and last '}' to isolate the JSON payload.

    Args:
        text: The raw text response from the model.

    Returns:
        The parsed Python object (dict, list, etc.).

    Raises:
        ValueError: If no valid JSON object can be extracted from the text.
    """
    text = THINK_BLOCK.sub("", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def valid_reply(reply: str, schema: Schema | None) -> str | None:
    """Validate and normalise a model reply against an optional JSON schema.

    If `schema` is None, the reply is treated as plain text: think blocks are removed,
    whitespace is stripped, and empty results become None.

    If `schema` is provided, the reply is parsed as JSON, validated against the schema,
    and re-serialised to ensure canonical formatting (removing fences/prose).

    Args:
        reply: The raw text response from the model.
        schema: The expected JSON Schema, or None for plain text replies.

    Returns:
        The normalised reply string if valid, or None if invalid/empty.
    """
    if schema is None:
        return THINK_BLOCK.sub("", reply).strip() or None
    try:
        value = parse_json_reply(reply)
    except ValueError:
        return None
    if schema_problem(value, schema):
        return None
    return json.dumps(value, ensure_ascii=False)


def schema_in_system_prompt(system: str) -> Schema | None:
    """Extract the JSON Schema appended to a system prompt by the factory.

    The factory appends a marker string followed by the JSON schema to the system prompt.
    This function locates that marker and parses the trailing JSON.

    Args:
        system: The full system prompt string.

    Returns:
        The parsed schema dict if found and valid, otherwise None.
    """
    _, marker, rest = system.partition(SCHEMA_MARKER)
    if not marker:
        return None
    try:
        schema = json.loads(rest)
    except json.JSONDecodeError:
        return None
    return schema if isinstance(schema, dict) else None
