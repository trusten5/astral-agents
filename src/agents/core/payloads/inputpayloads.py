# ==============================================================================
# inputpayloads.py — Agent run input payload models
# ==============================================================================
# Purpose: Define dataclasses for agent run items representing tool calls and reasoning steps.
# Sections: Imports, Dataclasses
# ==============================================================================

# ==============================================================================
# Imports
# ==============================================================================

# Standard Library --------------------------------------------------------------
import abc
import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar, Union

# Third-Party -------------------------------------------------------------------
from pydantic import BaseModel
from typing_extensions import TypeAlias

# Internal ----------------------------------------------------------------------
from exceptions import AgentsException, ModelBehaviorError
from ._share import RunItemBase, ResponseFunctionToolCall, ToolCallItemTypes, ResponseReasoningItem

if TYPE_CHECKING:
    from agent import Agent


# ==============================================================================
# Dataclasses
# ==============================================================================
@dataclass
class HandoffCallItem(RunItemBase[ResponseFunctionToolCall]):
    """Represents a tool call for agent handoff."""

    raw_item: ResponseFunctionToolCall
    type: Literal["handoff_call_item"] = "handoff_call_item"


@dataclass
class ToolCallItem(RunItemBase[ToolCallItemTypes]):
    """Represents a tool call such as function or computer action."""

    raw_item: ToolCallItemTypes
    type: Literal["tool_call_item"] = "tool_call_item"


@dataclass
class ReasoningItem(RunItemBase[ResponseReasoningItem]):
    """Represents a reasoning step in the agent's run."""

    raw_item: ResponseReasoningItem
    type: Literal["reasoning_item"] = "reasoning_item"
