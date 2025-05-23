# ==============================================================================
# _share.py — Shared base classes and types for OpenAI to Astral mapping
# ==============================================================================
# Purpose: Define shared base classes and types used across input and output payloads
# ==============================================================================

from __future__ import annotations

import abc
from typing import Any, Dict, Generic, List, Literal, Optional, TypeVar, Union
from pydantic import BaseModel, Field

from ._astral_types import (
    ResponseStatus,
    StopReason,
    ReasoningEffort,
    ContentPart,
    TextPart,
    ImagePart,
    AudioPart,
    ToolUsePart,
    ToolResultPart,
    ToolReferencePart,
    MessageOutput,
    ReasoningOutput,
    OutputItem
)

# ==============================================================================
# Type Variables
# ==============================================================================

T = TypeVar("T", bound=BaseModel)

# ==============================================================================
# Base Classes
# ==============================================================================

class RunItemBase(Generic[T]):
    """Base class for all run items that wrap raw provider responses."""
    
    def __init__(self, raw_item: T):
        self.raw_item = raw_item
        
    @property
    @abc.abstractmethod
    def type(self) -> str:
        """The type discriminator for this run item."""
        pass

# ==============================================================================
# Response Types
# ==============================================================================

class ResponseFunctionToolCall(BaseModel):
    """OpenAI function tool call mapped to Astral format."""
    type: Literal["function_call"]
    id: str
    name: str
    arguments: str
    status: Optional[ResponseStatus] = None

    def to_astral_tool_use(self) -> ToolUsePart:
        """Convert to Astral's ToolUsePart format."""
        import json
        try:
            input_json = json.loads(self.arguments)
        except json.JSONDecodeError:
            input_json = {}
        
        return ToolUsePart(
            type="tool_use",
            id=self.id,
            name=self.name,
            input=input_json,
            status=self.status
        )

class ResponseOutputMessage(BaseModel):
    """OpenAI message output mapped to Astral format."""
    type: Literal["message"]
    id: str
    role: str
    content: List[Dict[str, Any]]
    status: ResponseStatus
    provider_role: Optional[str] = None
    stop_reason: Optional[StopReason] = None
    stop_sequence: Optional[str] = None

    def to_astral_message(self) -> MessageOutput:
        """Convert to Astral's MessageOutput format."""
        content_parts: List[ContentPart] = []
        
        for part in self.content:
            if part["type"] == "text":
                content_parts.append(TextPart(text=part["text"]))
            elif part["type"] == "image":
                content_parts.append(ImagePart(url=part["url"], alt_text=part.get("alt_text")))
            elif part["type"] == "audio":
                content_parts.append(AudioPart(url=part["url"], transcription=part.get("transcription")))
            elif part["type"] == "tool_use":
                content_parts.append(ToolUsePart(**part))
            elif part["type"] == "tool_result":
                content_parts.append(ToolResultPart(**part))
            elif part["type"] == "tool_reference":
                content_parts.append(ToolReferencePart(**part))
        
        return MessageOutput(
            type="message",
            id=self.id,
            role=self.role,
            provider_role=self.provider_role,
            status=self.status,
            content=content_parts,
            stop_reason=self.stop_reason,
            stop_sequence=self.stop_sequence
        )

class ResponseComputerCall(BaseModel):
    """OpenAI computer call mapped to Astral format."""
    type: Literal["computer_call"]
    id: str
    command: str
    status: Optional[ResponseStatus] = None

    def to_astral_tool_use(self) -> ToolUsePart:
        """Convert to Astral's ToolUsePart format."""
        return ToolUsePart(
            type="tool_use",
            id=self.id,
            name="computer_use",
            input={"command": self.command},
            status=self.status
        )

class ResponseReasoningItem(BaseModel):
    """OpenAI reasoning output mapped to Astral format."""
    type: Literal["reasoning"]
    effort: Optional[ReasoningEffort] = None
    summary: Optional[str] = None

    def to_astral_reasoning(self) -> ReasoningOutput:
        """Convert to Astral's ReasoningOutput format."""
        return ReasoningOutput(
            type="reasoning",
            effort=self.effort,
            summary=self.summary
        )

# ==============================================================================
# Type Aliases
# ==============================================================================

ToolCallItemTypes = Union[ResponseFunctionToolCall, ResponseComputerCall]
TResponseInputItem = Union[ResponseFunctionToolCall, ResponseComputerCall, ResponseReasoningItem]
TResponseOutputItem = Union[ResponseOutputMessage]

# ==============================================================================
# Public exports
# ==============================================================================

__all__ = [
    "RunItemBase",
    "ResponseFunctionToolCall",
    "ResponseOutputMessage",
    "ResponseComputerCall",
    "ResponseReasoningItem",
    "ToolCallItemTypes",
    "TResponseInputItem",
    "TResponseOutputItem"
] 