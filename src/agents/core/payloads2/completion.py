# ============================================================================== #
# completions.py — Astral AI agents SDK unified completion models & helpers       #
# ============================================================================== #
#
# Provider-agnostic Pydantic schemas normalizing completion-style responses.
# Includes enums, content parts, output items, and conversion utilities.
#
# Inspired by Astral's completion.py v8 and tailored to handle OpenAI-like
# response output conversion to Astral OutputItem objects.
# ============================================================================== #

from __future__ import annotations

import json
import logging
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union, Annotated, Type

from pydantic import BaseModel, Field, ConfigDict

# Logger setup
logger = logging.getLogger("astral.agents.completions")

# ============================================================================== #
# Constants
# ============================================================================== #

PROVIDER_PREFIX = "[OpenAI]"

BUILTIN_TOOL_CALL_TYPES = {
    "web_search_call",
    "file_search_call",
    "computer_call",
    # Add other built-in tool call types as needed
}

# ============================================================================== #
# Enums
# ============================================================================== #


class ResponseStatus(str, Enum):
    """Overall status of a provider response."""
    COMPLETED = "completed"
    FAILED = "failed"
    IN_PROGRESS = "in_progress"
    INCOMPLETE = "incomplete"


class StopReason(str, Enum):
    """Reasons why generation stopped."""
    END_TURN = "end_turn"
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"
    TOOL_USE = "tool_use"
    CONTENT_FILTER = "content_filter"


# ============================================================================== #
# Enum mixin for value-based serialization
# ============================================================================== #


class UseEnumValues:
    """Mixin to serialize Enum fields by their value."""
    model_config = ConfigDict(use_enum_values=True)


# ============================================================================== #
# Content parts
# ============================================================================== #


class TextPart(BaseModel):
    """Plain-text content block."""
    type: Literal["text"] = Field("text")
    text: str = Field(..., description="Text content.")
    annotations: Optional[List[Any]] = Field(None, description="Provider-specific annotations for this text block.")


class ToolUsePart(BaseModel, UseEnumValues):
    """Block representing a function or tool invocation."""
    type: Literal["tool_use"] = Field("tool_use")
    id: str = Field(..., description="Unique ID for this tool call.")
    name: str = Field(..., description="Name of the tool or function.")
    input: Dict[str, Any] = Field(..., description="JSON-serializable input payload.")
    status: Optional[ResponseStatus] = Field(None, description="Status of the tool call.")


class ToolReferencePart(BaseModel, UseEnumValues):
    """Block referencing an external tool call without embedding result."""
    type: Literal["tool_reference"] = Field("tool_reference")
    call_id: str = Field(..., description="ID of the referenced tool call.")
    tool_name: Optional[str] = Field(None, description="Name of the referenced tool.")
    status: Optional[ResponseStatus] = Field(None, description="Status of the referenced tool.")
    data: Optional[Dict[str, Any]] = Field(None, description="Additional data from the tool reference.")


ContentPart = Annotated[
    Union[
        TextPart,
        ToolUsePart,
        ToolReferencePart,
    ],
    Field(discriminator="type"),
]


# ============================================================================== #
# Output items
# ============================================================================== #


class MessageOutput(BaseModel, UseEnumValues):
    """A chat message item in the response."""
    type: Literal["message"] = Field("message")
    id: str = Field(..., description="Provider-generated message identifier.")
    role: Literal["assistant", "user"] = Field(..., description="Canonical role of the speaker.")
    provider_role: Optional[str] = Field(None, description="Original provider role if non-canonical.")
    status: ResponseStatus = Field(..., description="Generation status for this message.")
    content: List[ContentPart] = Field(..., description="Ordered list of content blocks.")
    stop_reason: Optional[StopReason] = Field(None, description="Reason generation stopped, if provided.")
    stop_sequence: Optional[str] = Field(None, description="Custom stop sequence encountered, if any.")


OutputItem = Annotated[Union[MessageOutput], Field(discriminator="type")]


# ============================================================================== #
# Base provider output (simplified for completions)
# ============================================================================== #


class CompletionOutput(BaseModel):
    """Completion-style output containing ordered output items."""
    provider_id: str = Field(..., description="Provider response identifier.")
    provider_model_id: str = Field(..., description="Identifier of the model that handled the request.")
    provider_timestamp: int = Field(..., description="Unix timestamp when provider created response.")
    status: ResponseStatus = Field(..., description="Overall generation status.")
    raw_provider_response: Optional[Dict[str, Any]] = Field(None, description="Original unmodified provider payload.")
    items: List[OutputItem] = Field(..., description="Ordered output items returned by the provider.")


# ============================================================================== #
# Utility functions
# ============================================================================== #


def _get_status(obj: Dict[str, Any]) -> ResponseStatus:
    """Extract the ResponseStatus from a provider output dictionary."""
    status_val = obj.get("status", ResponseStatus.COMPLETED)
    try:
        return ResponseStatus(status_val)
    except ValueError:
        logger.warning(f"{PROVIDER_PREFIX} Unknown status '{status_val}', defaulting to COMPLETED")
        return ResponseStatus.COMPLETED


def output_to_items(output: List[BaseModel]) -> List[OutputItem]:
    """
    Convert raw output list (OpenAI style) to Astral OutputItem objects.

    Supported block types:
      - "message" → MessageOutput with TextPart(s)
      - "function_call" → MessageOutput with ToolUsePart
      - "<built-in>_call" → MessageOutput with ToolReferencePart

    Unknown block types are silently skipped for forward compatibility.

    Args:
        output: List of output blocks from the provider response

    Returns:
        List of standardized OutputItem objects
    """
    items: List[OutputItem] = []

    for py_obj in output:
        obj = py_obj.model_dump() if hasattr(py_obj, "model_dump") else dict(py_obj)
        typ: str = obj.get("type", "")
        logger.debug(f"{PROVIDER_PREFIX} Processing output item of type: {typ}, id: {obj.get('id', 'unknown')}")

        status = _get_status(obj)

        # Handle plain message with text parts
        if typ == "message":
            parts = [
                TextPart(text=content.get("text", ""))
                for content in obj.get("content", [])
                if content.get("type") in {"output_text", "text"}
            ]

            items.append(
                MessageOutput(
                    type="message",
                    id=obj["id"],
                    role=obj["role"],
                    provider_role=obj.get("provider_role"),
                    status=status,
                    content=parts,
                    stop_reason=obj.get("stop_reason"),
                    stop_sequence=obj.get("stop_sequence"),
                )
            )
            logger.debug(f"{PROVIDER_PREFIX} Added MessageOutput with id: {obj['id']}")

        # Handle function calls → ToolUsePart inside MessageOutput
        elif typ == "function_call":
            try:
                input_json = json.loads(obj.get("arguments", "{}"))
            except json.JSONDecodeError as e:
                logger.error(f"{PROVIDER_PREFIX} Failed to parse function arguments JSON: {e}")
                input_json = {}

            tool_part = ToolUsePart(
                id=obj["id"],
                name=obj.get("name", "unknown_function"),
                input=input_json,
                status=status,
            )

            items.append(
                MessageOutput(
                    type="message",
                    id=obj["id"],
                    role="assistant",
                    provider_role=None,
                    status=status,
                    content=[tool_part],
                )
            )
            logger.debug(f"{PROVIDER_PREFIX} Added MessageOutput with ToolUsePart for function call: {obj.get('name')}")

        # Handle built-in tool calls → ToolReferencePart inside MessageOutput
        elif typ in BUILTIN_TOOL_CALL_TYPES:
            aux_payload = {k: v for k, v in obj.items() if k not in {"type", "id", "status"}}
            tool_name = typ
            if typ == "computer_call":
                tool_name = "computer_use"

            tool_ref_part = ToolReferencePart(
                type="tool_reference",
                call_id=obj.get("id", ""),
                tool_name=tool_name,
                status=status,
                data=aux_payload,
            )

            items.append(
                MessageOutput(
                    type="message",
                    id=obj.get("id", ""),
                    role="assistant",
                    provider_role=None,
                    status=status,
                    content=[tool_ref_part],
                )
            )
            logger.debug(f"{PROVIDER_PREFIX} Added MessageOutput with ToolReferencePart for built-in tool call: {typ}")

        else:
            logger.debug(f"{PROVIDER_PREFIX} Skipping unknown output item type: {typ}")

    return items
