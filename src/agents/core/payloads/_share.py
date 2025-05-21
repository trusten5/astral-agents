# ============================================================================== 
# _share.py — Shared type aliases and imports for agent responses               
# ============================================================================== 
# Purpose: Provide type aliases and import necessary response types              
# Sections: Imports, Type Aliases                                                 
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

# OpenAI SDK - Replace w/ Astral -------------------------------------------------
from openai.types.responses import (
    Response,
    ResponseComputerToolCall,
    ResponseFileSearchToolCall,
    ResponseFunctionToolCall,
    ResponseFunctionWebSearch,
    ResponseInputItemParam,
    ResponseOutputItem,
    ResponseOutputMessage,
    ResponseOutputRefusal,
    ResponseOutputText,
    ResponseStreamEvent,
)
from openai.types.responses.response_input_item_param import (
    ComputerCallOutput,
    FunctionCallOutput,
)
from openai.types.responses.response_reasoning_item import ResponseReasoningItem

# Internal ----------------------------------------------------------------------
from exceptions import AgentsException, ModelBehaviorError

if TYPE_CHECKING:
    from agent import Agent


# ============================================================================== 
# Type Aliases                                                                  
# ============================================================================== 

TResponse: TypeAlias = Response
"""Alias for OpenAI SDK Response type."""

TResponseInputItem: TypeAlias = ResponseInputItemParam
"""Alias for OpenAI SDK ResponseInputItemParam type."""

TResponseOutputItem: TypeAlias = ResponseOutputItem
"""Alias for OpenAI SDK ResponseOutputItem type."""

TResponseStreamEvent: TypeAlias = ResponseStreamEvent
"""Alias for OpenAI SDK ResponseStreamEvent type."""

T = TypeVar("T", bound=Union[TResponseOutputItem, TResponseInputItem])

ToolCallItemTypes: TypeAlias = Union[
    ResponseFunctionToolCall,
    ResponseComputerToolCall,
    ResponseFileSearchToolCall,
    ResponseFunctionWebSearch,
]
"""Union type representing all tool call item variants."""

RunItem: TypeAlias = Union[
    "MessageOutputItem",
    "HandoffCallItem",
    "HandoffOutputItem",
    "ToolCallItem",
    "ToolCallOutputItem",
    "ReasoningItem",
]
"""Union type representing all possible run items."""
