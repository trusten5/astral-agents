# ==============================================================================
# outputpayloads.py — Output payload models for OpenAI to Astral mapping
# ==============================================================================
# Purpose: Define output payload models for messages and tool outputs
# ==============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Literal

from ._astral_types import BaseUsage, MessageOutput, ReasoningOutput
from ._share import (
    RunItemBase,
    ResponseOutputMessage,
    TResponseInputItem,
    TResponseOutputItem
)

if TYPE_CHECKING:
    from agent import Agent

# ==============================================================================
# Output Payload Models
# ==============================================================================

@dataclass
class MessageOutputItem(RunItemBase[ResponseOutputMessage]):
    """Represents a message output from OpenAI."""
    raw_item: ResponseOutputMessage
    type: Literal["message_output_item"] = "message_output_item"

    def to_astral_message(self) -> MessageOutput:
        """Convert to Astral's MessageOutput format."""
        return self.raw_item.to_astral_message()

@dataclass
class HandoffOutputItem(RunItemBase[TResponseInputItem]):
    """Represents the output of an agent handoff."""
    raw_item: TResponseInputItem
    source_agent: "Agent[Any]"
    target_agent: "Agent[Any]"
    type: Literal["handoff_output_item"] = "handoff_output_item"

    def to_astral_message(self) -> MessageOutput:
        """Convert to Astral's MessageOutput format."""
        if hasattr(self.raw_item, "to_astral_message"):
            return self.raw_item.to_astral_message()
        elif hasattr(self.raw_item, "to_astral_tool_use"):
            tool_use = self.raw_item.to_astral_tool_use()
            return MessageOutput(
                type="message",
                id=tool_use.id,
                role="assistant",
                status=tool_use.status or "completed",
                content=[tool_use]
            )
        else:
            raise ValueError(f"Cannot convert {type(self.raw_item)} to Astral MessageOutput")

@dataclass
class ToolCallOutputItem(RunItemBase[TResponseOutputItem]):
    """Represents the output from a tool call."""
    raw_item: TResponseOutputItem
    type: Literal["tool_call_output_item"] = "tool_call_output_item"

    def to_astral_message(self) -> MessageOutput:
        """Convert to Astral's MessageOutput format."""
        return self.raw_item.to_astral_message()

@dataclass
class ModelResponse:
    """Encapsulates the full response from an OpenAI model run."""
    output: List[TResponseOutputItem]
    usage: BaseUsage
    response_id: str | None

    def to_input_items(self) -> List[TResponseInputItem]:
        """Convert model outputs into input items suitable for subsequent calls."""
        return [item.model_dump(exclude_unset=True) for item in self.output]  # type: ignore

    def to_astral_messages(self) -> List[MessageOutput]:
        """Convert all outputs to Astral's MessageOutput format."""
        messages = []
        for item in self.output:
            if hasattr(item, "to_astral_message"):
                messages.append(item.to_astral_message())
        return messages

# ==============================================================================
# Public exports
# ==============================================================================

__all__ = [
    "MessageOutputItem",
    "HandoffOutputItem",
    "ToolCallOutputItem",
    "ModelResponse"
] 