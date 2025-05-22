# ==============================================================================
# stream.py — Streaming response content block types and state management
# ==============================================================================
# Purpose: Define block types and manage streaming response state and content
# ==============================================================================

# ============================================================================
# Imports
# ============================================================================

from enum import Enum
from typing import Any, Dict

# Import related output types from completion.py (if applicable)
from .completion import (
    MessageOutput as _MessageOutput,
    ReasoningOutput as _ReasoningOutput,
)

# ============================================================================
# Type Aliases
# ============================================================================

StreamingMessageOutput = _MessageOutput
StreamingReasoningOutput = _ReasoningOutput

# ============================================================================
# Public exports
# ============================================================================

__all__ = [
    "BlockType",
    "ResponseStatusEnum",
    "StreamingMessageOutput",
    "StreamingReasoningOutput",
    "_OutputState",
]

# ============================================================================
# Enums
# ============================================================================

class BlockType(str, Enum):
    """Discriminator for in-stream content categories."""
    TEXT = "text"
    TOOL = "tool"
    STRUCTURED = "structured"
    REASONING = "reasoning"

class ResponseStatusEnum(str, Enum):
    """Status of a streaming response."""
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ERROR = "error"

# ============================================================================
# Block state management
# ============================================================================

class _Block:
    """Manages the state of an individual content block."""

    def __init__(self, block_type: BlockType):
        self.type = block_type
        self.content: str = ""
        self.json_content: Any = {}
        self.closed: bool = False
        self.tool_name: str | None = None
        self.tool_id: str | None = None

    def add_text(self, text: str) -> None:
        """Add text content to this block if not closed."""
        if not self.closed:
            self.content += text

    def add_json(self, json_part: Any) -> None:
        """Add or merge JSON content to this block if not closed."""
        if not self.closed:
            if isinstance(json_part, dict) and isinstance(self.json_content, dict):
                self.json_content.update(json_part)
            else:
                self.json_content = json_part

    def close(self) -> None:
        """Mark the block as complete/closed."""
        self.closed = True

class _OutputState:
    """Tracks the state of streaming output and manages multiple content blocks."""

    def __init__(self):
        self.item = None  # Placeholder for the streaming item (e.g., message, reasoning, etc.)
        self.blocks: Dict[int, _Block] = {}

    def block(self, index: int, block_type: BlockType) -> _Block:
        """Get or create a block at the given index."""
        if index not in self.blocks:
            self.blocks[index] = _Block(block_type)
        return self.blocks[index]
