"""Structured XML-tag protocol between the agent loop and the LLM.

Why XML tags instead of JSON?
    Large models tend to follow tag-based formats more reliably than strict
    JSON: tag boundaries are robust, and free text inside a tag needs no
    escaping. This matters most for tool calling, where we must reliably tell
    "a message for the user" apart from "run this tool".

Two response variants are supported:
    <message>...text for the user...</message>
    <tool_call>
      <name>tool_name</name>
      <arguments>{"key": "value"}</arguments>
    </tool_call>

The XML *shell* selects the variant; tool ``<arguments>`` carry a small JSON
object so structured argument *types* (numbers, booleans, arrays) survive.

We do NOT use a strict XML parser -- model output is often not well-formed XML.
A small, lenient tag extractor is used instead (also fits the project's
"implement it yourself" spirit). This module sits ABOVE the provider layer.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


# Base instructions. When tools are available, a catalog + tool-call format is
# appended by ``build_system_prompt``.
PROTOCOL_INSTRUCTIONS = """\
Reply using XML-style tags, and output nothing outside the tags.

To answer the user, use exactly one <message> block:
  <message>your reply to the user</message>

Format the text inside <message> as GitHub-Flavored Markdown so it renders
richly in the terminal:
- Inline code -> wrap in single backticks, e.g. `variable`.
- Code blocks -> use triple-backtick fences with a language, e.g. ```python.
- Quotes / callouts -> prefix lines with "> ".
- You may also use **bold**, *italics*, lists, and headings.
- Do NOT use LaTeX or $...$ math; it is not rendered and shows up as literal
  dollar signs. Write math in plain text or inside a code span instead.

Rules:
- The text inside a tag may span multiple lines and does not need escaping.
- Do not wrap the <message>/<tool_call> tags themselves in a code fence
  (fences are only for code *inside* your Markdown reply).
"""

# Appended only when at least one tool is registered.
TOOL_INSTRUCTIONS_TEMPLATE = """\
{catalog}

To call a tool instead of answering, use exactly one <tool_call> block:
  <tool_call>
    <name>tool_name</name>
    <arguments>{{"arg": "value"}}</arguments>
  </tool_call>

The <arguments> content must be a valid JSON object (use {{}} if no arguments).
After a tool runs you will receive its result and can then either call another
tool or reply with a <message>. Only emit ONE block per turn.
"""


class ResponseType:
    """String constants for response variants (avoids magic strings)."""

    MESSAGE = "message"
    TOOL_CALL = "tool_call"


@dataclass
class AgentResponse:
    """A parsed, structured model response.

    ``type`` selects the variant:
    - ``message``   -> ``content`` holds the user-facing text.
    - ``tool_call`` -> ``tool_name`` + ``tool_arguments`` describe the call.

    ``raw`` keeps extracted fields; ``parse_error`` records why we fell back or
    why a tool call is malformed (useful for debugging and UI surfacing).
    """

    type: str
    content: str = ""
    tool_name: Optional[str] = None
    tool_arguments: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)
    parse_error: Optional[str] = None

    @property
    def is_message(self) -> bool:
        return self.type == ResponseType.MESSAGE

    @property
    def is_tool_call(self) -> bool:
        return self.type == ResponseType.TOOL_CALL


def build_system_prompt(base_prompt: str, tools_catalog: str = "") -> str:
    """Combine the base system prompt, protocol rules, and (optional) tools.

    ``tools_catalog`` is the string produced by ``ToolRegistry.render_for_prompt``.
    When non-empty, the tool-call format and catalog are appended so the model
    knows what it can call and how.
    """
    parts = []
    base = base_prompt.strip()
    if base:
        parts.append(base)
    parts.append(PROTOCOL_INSTRUCTIONS.strip())
    catalog = tools_catalog.strip()
    if catalog:
        parts.append(TOOL_INSTRUCTIONS_TEMPLATE.format(catalog=catalog).strip())
    return "\n\n".join(parts)


def extract_tag(text: str, tag: str) -> Optional[str]:
    """Return the inner text of the first ``<tag>...</tag>`` found in ``text``.

    Lenient on purpose: case-insensitive, ``.`` matches newlines, tolerates text
    before/after the block, and ignores optional attributes on the opening tag.
    Returns the stripped inner text, or ``None`` if the tag is not present.
    """
    pattern = re.compile(
        rf"<{tag}(?:\s[^>]*)?>(.*?)</{tag}>",
        re.DOTALL | re.IGNORECASE,
    )
    match = pattern.search(text)
    if match is None:
        return None
    return match.group(1).strip()


def _find_first_tag(text: str, tags) -> Optional[str]:
    """Return which of ``tags`` appears first (by opening position) in text."""
    first: Optional[str] = None
    first_pos = len(text) + 1
    for tag in tags:
        match = re.search(rf"<{tag}(?:\s[^>]*)?>", text, re.IGNORECASE)
        if match and match.start() < first_pos:
            first_pos = match.start()
            first = tag
    return first


def parse_agent_response(text: str) -> AgentResponse:
    """Parse raw model output into a structured :class:`AgentResponse`.

    Resolution order is by *position*: whichever of ``<tool_call>`` or
    ``<message>`` opens first wins, so a stray mention of one tag inside the
    other's content does not confuse us. If neither is present we degrade
    gracefully to a message carrying the original text.
    """
    which = _find_first_tag(text, (ResponseType.TOOL_CALL, ResponseType.MESSAGE))

    if which == ResponseType.TOOL_CALL:
        return _parse_tool_call(text)

    if which == ResponseType.MESSAGE:
        inner = extract_tag(text, ResponseType.MESSAGE) or ""
        return AgentResponse(
            type=ResponseType.MESSAGE, content=inner, raw={"message": inner}
        )

    return AgentResponse(
        type=ResponseType.MESSAGE,
        content=text.strip(),
        parse_error="no <message> or <tool_call> tag found",
    )


def _parse_tool_call(text: str) -> AgentResponse:
    inner = extract_tag(text, ResponseType.TOOL_CALL) or ""
    name = extract_tag(inner, "name")
    args_text = extract_tag(inner, "arguments")

    if not name:
        return AgentResponse(
            type=ResponseType.TOOL_CALL,
            raw={"tool_call": inner},
            parse_error="tool_call missing <name>",
        )

    arguments: Dict[str, Any] = {}
    parse_error: Optional[str] = None
    if args_text:
        try:
            parsed = json.loads(args_text)
            if isinstance(parsed, dict):
                arguments = parsed
            else:
                parse_error = "<arguments> is not a JSON object"
        except json.JSONDecodeError as exc:
            parse_error = f"invalid JSON in <arguments>: {exc}"

    return AgentResponse(
        type=ResponseType.TOOL_CALL,
        tool_name=name,
        tool_arguments=arguments,
        raw={"tool_call": inner, "arguments_text": args_text or ""},
        parse_error=parse_error,
    )