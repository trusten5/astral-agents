# ==============================================================================
# inputpayloads.py — Input payload models for OpenAI to Astral mapping
# ==============================================================================
# Purpose: Define input payload models for tool calls and reasoning steps
# ==============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from ._share import (
    RunItemBase,
    ResponseFunctionToolCall,
    ResponseReasoningItem,
    ToolCallItemTypes,
    TResponseInputItem
)

if TYPE_CHECKING:
    from agent import Agent

# ==============================================================================
# Input Payload Models
# ==============================================================================

@dataclass
class HandoffCallItem(RunItemBase[ResponseFunctionToolCall]):
    """Represents a tool call for agent handoff."""
    raw_item: ResponseFunctionToolCall
    type: Literal["handoff_call_item"] = "handoff_call_item"

    def to_astral_tool_use(self):
        """Convert to Astral's ToolUsePart format."""
        return self.raw_item.to_astral_tool_use()

@dataclass
class ToolCallItem(RunItemBase[ToolCallItemTypes]):
    """Represents a tool call such as function or computer action."""
    raw_item: ToolCallItemTypes
    type: Literal["tool_call_item"] = "tool_call_item"

    def to_astral_tool_use(self):
        """Convert to Astral's ToolUsePart format."""
        return self.raw_item.to_astral_tool_use()

@dataclass
class ReasoningItem(RunItemBase[ResponseReasoningItem]):
    """Represents a reasoning step in the agent's thought process."""
    raw_item: ResponseReasoningItem
    type: Literal["reasoning_item"] = "reasoning_item"

    def to_astral_reasoning(self):
        """Convert to Astral's ReasoningOutput format."""
        return self.raw_item.to_astral_reasoning()

# ==============================================================================
# Public exports
# ==============================================================================

__all__ = [
    "HandoffCallItem",
    "ToolCallItem",
    "ReasoningItem"
] 