"""Text helpers ported from the factory, so synthetic prompts read exactly like production ones."""

import ast
import json
import re
from pathlib import Path
from string import Template
from typing import Any

from factory_model_finetune.errors import ConfigError
from factory_model_finetune.schemas import SCHEMA_MARKER, Schema

# The brackets (U+2039, U+203A) the factory swaps in for `<` and `>` in think tags.
DEFANG_OPEN, DEFANG_CLOSE = chr(0x2039), chr(0x203A)
JS_COMMENTS = re.compile(r"/\*.*?\*/|//[^\n]*", re.S)
# The factory's gate names and the test that ran are fixed here: evidence text only needs to look
# like a green run for the PR-body role.
GATE_NAMES = ("format", "lint", "types", "test")


def render_prompt(prompts_dir: Path, name: str, **values: Any) -> str:
    """Render a Jinja-free prompt template from the prompts directory.

    Replaces `$name` placeholders with values. Non-string values are serialised
    as indented JSON to preserve structure in the prompt text.

    Args:
        prompts_dir: Directory containing `.md` template files.
        name: Base name of the template file (without `.md` extension).
        **values: Placeholder values for the template.

    Returns:
        The rendered prompt string.

    Raises:
        ConfigError: If the template file does not exist.
    """
    path = prompts_dir / f"{name}.md"
    if not path.is_file():
        raise ConfigError(f"prompt template not found at {path}; check --prompts-dir")
    rendered = {k: v if isinstance(v, str) else json.dumps(v, indent=2) for k, v in values.items()}
    return Template(path.read_text()).safe_substitute(rendered)


def defang_think(text: str) -> str:
    """Replace standard XML-style think tags with Unicode brackets.

    The factory swaps `` for U+2039/U+203A brackets to
    prevent the LM Studio inference engine from interpreting them as special
    tokens or control sequences. This ensures the model treats the content as
    plain text reasoning steps.
    """
    return re.sub(r"<(/?)think>", lambda tag: f"{DEFANG_OPEN}{tag[1]}think{DEFANG_CLOSE}", text)


def guided_system(system: str, schema: Schema) -> str:
    """Construct the system prompt for reasoning roles by appending the JSON schema.

    Appends a directive instructing the model to reply with a single JSON object
    conforming to the provided schema. This enforces structured output for
    downstream parsing.

    Args:
        system: The base system instructions.
        schema: The JSON schema definition for the expected output.

    Returns:
        The combined system prompt string.
    """
    return f"{system}\n\nReply with a single JSON object that {SCHEMA_MARKER}{json.dumps(schema)}"


def spec_markdown(spec: dict[str, Any]) -> str:
    """Render a specification dictionary into a structured Markdown document.

    Formats the spec into sections including title, goal, acceptance criteria,
    edge cases, architecture, and decisions. This format is consumed by
    downstream planning and coding stages.

    Args:
        spec: The specification dictionary containing title, goal, criteria,
              edge cases, and architecture details.

    Returns:
        A Markdown string representing the full specification.
    """
    arch = spec["architecture"]
    out = [f"# {spec['title']}", ""]
    if spec.get("goal"):
        out += ["## What you asked for", "", spec["goal"], ""]
    out += [spec["summary"], "", "## Acceptance criteria", ""]
    out += [f"- **{c['id']}** {c['text']}" for c in spec["acceptance_criteria"]]
    out += ["", "## Edge cases", ""] + [f"- **{e['id']}** {e['text']}" for e in spec["edge_cases"]]
    out += ["", "## Architecture", "", arch["overview"], ""]
    out += [f"- **{c['name']}**: {c['responsibility']}" for c in arch["components"]]
    if arch["data_model"]:
        out += ["", "**Data model:** " + arch["data_model"]]
    if arch["decisions"]:
        out += ["", "### Decisions", ""] + [
            f"- {d['decision']}: {d['why']}" for d in arch["decisions"]
        ]
    for title, key in (("Out of scope", "out_of_scope"), ("Assumptions", "assumptions")):
        if spec[key]:
            out += ["", f"## {title}", ""] + [f"- {x}" for x in spec[key]]
    return "\n".join(out) + "\n"


def plan_markdown(plan: dict[str, Any]) -> str:
    """Render a task plan dictionary into a structured Markdown document.

    Iterates over tasks, assigning execution roles ("local model" for trivial
    tasks, "Claude" otherwise) and listing associated files and tests.

    Args:
        plan: The plan dictionary containing a list of task objects.

    Returns:
        A Markdown string representing the execution plan.
    """
    out = ["# Plan", ""]
    for task in plan["tasks"]:
        who = "local model" if task["kind"] == "trivial" else "Claude"
        out += [f"## {task['id']}. {task['title']} ({who})", "", task["details"], ""]
        if task["files"]:
            out.append("Files: " + ", ".join(f"`{f}`" for f in task["files"]))
        if task["tests"]:
            out += ["Tests:"] + [f"- {x}" for x in task["tests"]]
        out.append("")
    return "\n".join(out)


def evidence_markdown(
    title: str, spec: dict[str, Any], mapping: dict[str, Any], tests: list[str]
) -> str:
    """Render an evidence report summarising test coverage against acceptance criteria.

    Maps test results to specific acceptance criteria and edge cases from the
    spec, marking them as verified (✅) or unverified (⚠️). Includes gate
    status and summary statistics.

    Args:
        title: Title for the evidence report.
        spec: The original specification containing acceptance criteria and edge cases.
        mapping: A dictionary mapping criteria/edge-case IDs to their test names and notes.
        tests: A list of test names that passed in the current run.

    Returns:
        A Markdown string summarising the verification status.
    """
    known = set(tests)
    by_id = {c["id"]: c for c in mapping["criteria"]}
    edge_by_id = {e["id"]: e for e in mapping["edge_cases"]}
    criteria = [
        (c, [t for t in by_id.get(c["id"], {}).get("tests", []) if t in known])
        for c in spec["acceptance_criteria"]
    ]
    edges = [
        (e, [t for t in edge_by_id.get(e["id"], {}).get("tests", []) if t in known])
        for e in spec["edge_cases"]
    ]
    verified = sum(bool(found) for _, found in criteria)
    edges_verified = sum(bool(found) for _, found in edges)
    lines = [
        f"# Evidence: {title}",
        "",
        f"**Requirements verified:** {verified}/{len(criteria)} · "
        f"**Edge cases verified:** {edges_verified}/{len(edges)} · "
        f"**Tests:** {len(tests)} passed, 0 failed, 0 skipped",
        "",
        "## Gates",
        "",
        "| Gate | Result | Time | Log |",
        "|---|---|---|---|",
    ]
    lines += [f"| {g} | pass | 1.0s | `gates/01-review/{g}.log` |" for g in GATE_NAMES]
    lines += ["", "## Requirements", ""]
    for criterion, found in criteria:
        lines.append(f"- {'✅' if found else '⚠️'} **{criterion['id']}** {criterion['text']}")
        lines += [f"  - test: `{name}`" for name in found]
        if notes := by_id.get(criterion["id"], {}).get("notes"):
            lines.append(f"  - {notes}")
    lines += ["", "## Edge cases", "", "| | Case | Handling | Tests |", "|---|---|---|---|"]
    for edge, found in edges:
        handling = edge_by_id.get(edge["id"], {}).get("handling", "")
        names = ", ".join(f"`{n}`" for n in found) or "—"
        lines.append(
            f"| {'✅' if found else '⚠️'} | **{edge['id']}** {edge['text']} | {handling} | {names} |"
        )
    return "\n".join(lines) + "\n"


def replace_once(content: str, find: str, replace: str) -> str | None:
    """Perform a replacement only if the target string appears exactly once.

    Ensures deterministic edits by rejecting ambiguous matches (multiple occurrences)
    or missing matches. This prevents accidental modifications to similar strings
    elsewhere in the file.

    Args:
        content: The original string content.
        find: The substring to locate.
        replace: The replacement string.

    Returns:
        The modified string if exactly one match is found, otherwise None.
    """
    if not find or content.count(find) != 1:
        return None
    return content.replace(find, replace, 1)


def is_comment_only_change(path: str, before: str, after: str) -> bool:
    """Determine if two file versions differ only by comments, docstrings, or whitespace.

    For Python files, this parses the AST and removes docstrings before comparing
    the structural nodes. For other files, it strips comments and normalises
    whitespace. This is used to detect trivial changes that don't affect logic.

    Args:
        path: The file path, used to determine parsing strategy (e.g., `.py`).
        before: The original file content.
        after: The modified file content.

    Returns:
        True if the semantic content is identical, False otherwise.
    """
    if path.endswith(".py"):
        try:
            return ast.dump(_without_docstrings(ast.parse(before))) == ast.dump(
                _without_docstrings(ast.parse(after))
            )
        except SyntaxError:
            return False

    def normalised(source: str) -> str:
        return re.sub(r"\s+", "", JS_COMMENTS.sub("", source))

    return normalised(before) == normalised(after)


def _without_docstrings(tree: ast.Module) -> ast.Module:
    """Remove docstring nodes from an AST tree in-place.

    Iterates through all nodes and removes the first expression if it is a
    string constant (docstring) for modules, functions, and classes. Replaces
    the removed node with `ast.Pass()` to maintain valid syntax if the body
    becomes empty.

    Args:
        tree: The AST module to modify.

    Returns:
        The modified AST tree.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            continue
        first = node.body[0] if node.body else None
        is_docstring = (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        )
        if is_docstring:
            node.body = node.body[1:] or [ast.Pass()]
    return tree
