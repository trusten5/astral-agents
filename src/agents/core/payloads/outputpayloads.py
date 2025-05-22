# ==============================================================================
# outputpayloads.py — Data classes and models for agent output payloads
# ==============================================================================
# Purpose: Define output payload representations from model and agent responses
# Sections: Imports, Dataclasses, Model Response, (Commented) Helpers
# ==============================================================================

# ==============================================================================
# Imports
# ==============================================================================

# Standard Library --------------------------------------------------------------
import abc
import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Union, Generic, Literal, TypeVar

# Third-Party -------------------------------------------------------------------
from pydantic import BaseModel
from typing_extensions import TypeAlias

# Internal ----------------------------------------------------------------------
from exceptions import AgentsException, ModelBehaviorError
from ._share import RunItemBase, ResponseOutputMessage, TResponseInputItem, FunctionCallOutput, ComputerCallOutput, TResponseOutputItem
# from .usage import Usage

if TYPE_CHECKING:
    from agent import Agent

# ==============================================================================
# Public exports
# ==============================================================================
__all__ = [
    "MessageOutputItem",
    "HandoffOutputItem",
    "ToolCallOutputItem",
    "ModelResponse",
]

# ==============================================================================
# Dataclasses
# ==============================================================================


@dataclass
class MessageOutputItem(RunItemBase[ResponseOutputMessage]):
    """Represents a message output from the LLM."""

    raw_item: ResponseOutputMessage
    type: Literal["message_output_item"] = "message_output_item"


@dataclass
class HandoffOutputItem(RunItemBase[TResponseInputItem]):
    """Represents the output of an agent handoff."""

    raw_item: TResponseInputItem
    source_agent: "Agent[Any]"
    target_agent: "Agent[Any]"
    type: Literal["handoff_output_item"] = "handoff_output_item"


@dataclass
class ToolCallOutputItem(RunItemBase[Union[FunctionCallOutput, ComputerCallOutput]]):
    """Represents the output of a tool call."""

    raw_item: Union[FunctionCallOutput, ComputerCallOutput]
    output: Any
    type: Literal["tool_call_output_item"] = "tool_call_output_item"


@dataclass
class ModelResponse:
    """Encapsulates the full response from a model run."""

    output: list[TResponseOutputItem]
    usage: Usage
    response_id: str | None

    def to_input_items(self) -> list[TResponseInputItem]:
        """
        Convert model outputs into input items suitable for subsequent calls.

        Returns:
            List of input items derived from outputs.
        """
        return [item.model_dump(exclude_unset=True) for item in self.output]  # type: ignore


# ==============================================================================
# To-Do / Helpers (commented out)
# ==============================================================================

# class ItemHelpers:
#     @classmethod
#     def extract_last_content(cls, message: TResponseOutputItem) -> str:
#         """Extracts the last text content or refusal from a message."""
#         if not isinstance(message, ResponseOutputMessage):
#             return ""
#
#         last_content = message.content[-1]
#         if isinstance(last_content, ResponseOutputText):
#             return last_content.text
#         elif isinstance(last_content, ResponseOutputRefusal):
#             return last_content.refusal
#         else:
#             raise ModelBehaviorError(f"Unexpected content type: {type(last_content)}")
#
#     @classmethod
#     def extract_last_text(cls, message: TResponseOutputItem) -> str | None:
#         """Extracts the last text content from a message, if any. Ignores refusals."""
#         if isinstance(message, ResponseOutputMessage):
#             last_content = message.content[-1]
#             if isinstance(last_content, ResponseOutputText):
#                 return last_content.text
#
#         return None
#
#     @classmethod
#     def input_to_new_input_list(
#         cls, input: str | list[TResponseInputItem]
#     ) -> list[TResponseInputItem]:
#         """Converts a string or list of input items into a list of input items."""
#         if isinstance(input, str):
#             return [
#                 {
#                     "content": input,
#                     "role": "user",
#                 }
#             ]
#         return copy.deepcopy(input)
#
#     @classmethod
#     def text_message_outputs(cls, items: list[RunItem]) -> str:
#         """Concatenates all the text content from a list of message output items."""
#         text = ""
#         for item in items:
#             if isinstance(item, MessageOutputItem):
#                 text += cls.text_message_output(item)
#         return text
#
#     @classmethod
#     def text_message_output(cls, message: MessageOutputItem) -> str:
#         """Extracts all the text content from a single message output item."""
#         text = ""
#         for item in message.raw_item.content:
#             if isinstance(item, ResponseOutputText):
#                 text += item.text
#         return text
#
#     @classmethod
#     def tool_call_output_item(
#         cls, tool_call: ResponseFunctionToolCall, output: str
#     ) -> FunctionCallOutput:
#         """Creates a tool call output item from a tool call and its output."""
#         return {
#             "call_id": tool_call.call_id,
#             "output": output,
#             "type": "function_call_output",
#         }
