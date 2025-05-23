# ==============================================================================
# _astral_types.py — Type definitions for Astral's core library
# ==============================================================================
# Purpose: Define type stubs and base classes that mirror Astral's core library
# structure for type checking and compatibility.
# ==============================================================================

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union, TypeVar, Annotated
from pydantic import BaseModel, Field, ConfigDict

# ==============================================================================
# Enums
# ==============================================================================

class ResponseStatus(str, Enum):
    """Overall status of a provider response."""
    COMPLETED = "completed"
    FAILED = "failed"
    IN_PROGRESS = "in_progress"
    INCOMPLETE = "incomplete"

class ReasoningEffort(str, Enum):
    """Self-reported reasoning effort levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class StopReason(str, Enum):
    """Reasons why generation stopped."""
    END_TURN = "end_turn"
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"
    TOOL_USE = "tool_use"
    CONTENT_FILTER = "content_filter"

# ==============================================================================
# Base Models
# ==============================================================================

class UseEnumValues:
    """Mixin to serialize Enum fields by their value."""
    model_config = ConfigDict(use_enum_values=True)

class BaseUsage(BaseModel):
    """Base usage tracking model."""
    pass

class BaseCost(BaseModel):
    """Base cost tracking model."""
    pass

class ChatUsage(BaseUsage):
    """Chat-specific usage tracking."""
    pass

class ChatCost(BaseCost):
    """Chat-specific cost tracking."""
    pass

# ==============================================================================
# Content Parts
# ==============================================================================

class TextPart(BaseModel):
    """Plain-text content block."""
    type: Literal["text"] = Field("text")
    text: str = Field(..., description="Text content.")
    annotations: Optional[List[Any]] = Field(None, description="Provider-specific annotations for this text block.")

class ImagePart(BaseModel):
    """Image content block."""
    type: Literal["image"] = Field("image")
    url: str = Field(..., description="URL of the image.")
    alt_text: Optional[str] = Field(None, description="Alternative text for accessibility.")

class AudioPart(BaseModel):
    """Audio content block."""
    type: Literal["audio"] = Field("audio")
    url: str = Field(..., description="URL of the audio.")
    transcription: Optional[str] = Field(None, description="Transcript of the audio, if provided.")

class ToolUsePart(BaseModel, UseEnumValues):
    """Block representing a function or tool invocation."""
    type: Literal["tool_use"] = Field("tool_use")
    id: str = Field(..., description="Unique ID for this tool call.")
    name: str = Field(..., description="Name of the tool or function.")
    input: Dict[str, Any] = Field(..., description="JSON-serializable input payload.")
    status: Optional[ResponseStatus] = Field(None, description="Status of the tool call.")

class ToolResultPart(BaseModel):
    """Block representing the output of a tool invocation."""
    type: Literal["tool_result"] = Field("tool_result")
    tool_use_id: str = Field(..., description="ID of the corresponding tool invocation.")
    content: Optional[List[Any]] = Field(None, description="Structured content blocks returned by the tool.")

class ToolReferencePart(BaseModel, UseEnumValues):
    """Block referencing an external tool call without embedding result."""
    type: Literal["tool_reference"] = Field("tool_reference")
    call_id: str = Field(..., description="ID of the referenced tool call.")
    tool_name: Optional[str] = Field(None, description="Name of the referenced tool.")
    status: Optional[ResponseStatus] = Field(None, description="Status of the referenced tool.")
    data: Optional[Dict[str, Any]] = Field(None, description="Additional data from the tool reference.")

# Recursive union after definitions
ContentPart = Annotated[
    Union[
        TextPart,
        ImagePart,
        AudioPart,
        ToolUsePart,
        ToolResultPart,
        ToolReferencePart,
    ],
    Field(discriminator="type"),
]

# ==============================================================================
# Output Items
# ==============================================================================

class MessageOutput(BaseModel, UseEnumValues):
    """A chat message item in the response."""
    type: Literal["message"] = Field("message", description="Discriminator for a message item.")
    id: str = Field(..., description="Provider-generated message identifier.")
    role: Literal["assistant", "user"] = Field(..., description="Canonical role of the speaker.")
    provider_role: Optional[str] = Field(None, description="Original provider role if non-canonical.")
    status: ResponseStatus = Field(..., description="Generation status for this message.")
    content: List[ContentPart] = Field(..., description="Ordered list of content blocks.")
    stop_reason: Optional[StopReason] = Field(None, description="Reason generation stopped, if provided.")
    stop_sequence: Optional[str] = Field(None, description="Custom stop sequence encountered, if any.")

class ReasoningOutput(BaseModel, UseEnumValues):
    """A reasoning item in the response."""
    type: Literal["reasoning"] = Field("reasoning", description="Discriminator for a reasoning item.")
    effort: Optional[ReasoningEffort] = Field(None, description="Self-reported reasoning effort level.")
    summary: Optional[str] = Field(None, description="High-level summary of the chain of thought.")

OutputItem = Annotated[Union[MessageOutput, ReasoningOutput], Field(discriminator="type")]

# ==============================================================================
# Type Variables
# ==============================================================================

ResponseT = TypeVar('ResponseT', bound=OutputItem)
UsageT = TypeVar('UsageT', bound=BaseUsage)
CostT = TypeVar('CostT', bound=BaseCost)

# ==============================================================================
# Public exports
# ==============================================================================

__all__ = [
    # Enums
    "ResponseStatus",
    "ReasoningEffort",
    "StopReason",
    # Base Models
    "UseEnumValues",
    "BaseUsage",
    "BaseCost",
    "ChatUsage",
    "ChatCost",
    # Content Parts
    "TextPart",
    "ImagePart",
    "AudioPart",
    "ToolUsePart",
    "ToolResultPart",
    "ToolReferencePart",
    "ContentPart",
    # Output Items
    "MessageOutput",
    "ReasoningOutput",
    "OutputItem",
    # Type Variables
    "ResponseT",
    "UsageT",
    "CostT"
] 