# ==============================================================================
# __init__.py — Run Item Type Definitions for OpenAI to Astral mapping
# ==============================================================================
# Purpose: Define the union type representing all possible run item variants
# ==============================================================================

from typing import Union
from typing_extensions import TypeAlias

# Import all the types we need to expose
from .inputpayloads import (
    HandoffCallItem,
    ToolCallItem,
    ReasoningItem
)
from .outputpayloads import (
    MessageOutputItem,
    HandoffOutputItem,
    ToolCallOutputItem,
    ModelResponse
)

# ==============================================================================
# Type Definitions
# ==============================================================================

# Define RunItem here to avoid circular dependencies
RunItem: TypeAlias = Union[
    HandoffCallItem,
    ToolCallItem,
    ReasoningItem,
    MessageOutputItem,
    HandoffOutputItem,
    ToolCallOutputItem,
]
"""Union type representing all possible run items."""

# ==============================================================================
# Public exports
# ==============================================================================

__all__ = [
    # The RunItem type alias
    "RunItem",
    
    # Input payload types
    "HandoffCallItem",
    "ToolCallItem",
    "ReasoningItem",
    
    # Output payload types
    "MessageOutputItem",
    "HandoffOutputItem",
    "ToolCallOutputItem",
    "ModelResponse",
]
