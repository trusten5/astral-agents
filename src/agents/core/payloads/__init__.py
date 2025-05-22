# ==============================================================================
# __init__.py — Run Item Type Definitions                                       
# ==============================================================================
# Purpose: Define the union type representing all possible run item variants.   
# Sections: Imports, Type Definitions, Public Exports                               
# ==============================================================================

# ==============================================================================
# Imports                                                                       
# ==============================================================================

# Standard Library --------------------------------------------------------------
from typing import Union

# Third‑Party -------------------------------------------------------------------
from typing_extensions import TypeAlias

# Internal ----------------------------------------------------------------------
from .inputpayloads import HandoffCallItem, ToolCallItem, ReasoningItem
from .outputpayloads import MessageOutputItem, HandoffOutputItem, ToolCallOutputItem

# ==============================================================================
# Type Definitions                                                              
# ==============================================================================

RunItem: TypeAlias = Union[
    "HandoffCallItem",
    "ToolCallItem",
    "ReasoningItem",
    "MessageOutputItem",
    "HandoffOutputItem",
    "ToolCallOutputItem",
]
"""Union type representing all possible run items."""

# ==============================================================================
# Public exports                                                                
# ==============================================================================

__all__ = [
    "RunItem",
]
