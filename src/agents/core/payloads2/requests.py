# ============================================================================== #
# request.py — Astral AI canonical *request* schema                            #
# ============================================================================== #
#
# Public request objects surfaced by the high‑level Astral client:
#
#   • CompletionRequest                 → free‑form text / tool use
#   • StructuredCompletionRequest[T]    → JSON / function calling
#   • EmbeddingRequest                  → vector embeddings
#
# All inherit from **_BaseRequest** so identifiers, astral‑params, and
# shared helpers remain consistent across the SDK.
# ============================================================================== #

from __future__ import annotations

# Standard Library --------------------------------------------------------------
from typing import (
    Any,
    Dict,
    Generic,
    Iterable,
    List,
    Literal,
    Optional,
    Set,
    Type,
    TypeAlias,
    TypeVar,
    Union,
    TypedDict,
    TYPE_CHECKING,
)

# Pydantic -----------------------------------------------------------------------
from pydantic import BaseModel, Field, ConfigDict

# Astral AI ----------------------------------------------------------------------
# from ._base import NotGiven, NOT_GIVEN, _BaseResource, S
# from .astral import AstralParams
# from astral_ai.messages._models import Messages, SystemMessage
# from astral_ai.tools.function_tools import FunctionTool

# HTTPX --------------------------------------------------------------------------
from httpx import Timeout

# OpenAI -------------------------------------------------------------------------
from openai.types.responses import ResponseTextConfigParam
from openai.types.shared.reasoning_effort import ReasoningEffort

# Forward ref: built‑in provider tools
if TYPE_CHECKING:
    from astral_ai.tools.built_in_tools import BuiltInTool
else:
    BuiltInTool = ForwardRef("BuiltInTool")  # run-time only

# ============================================================================== #
# Public exports
# ============================================================================== #

__all__ = [
    "_BaseRequest",
    "_BaseCompletionRequest",
    "CompletionRequest",
    "CompletionRequestJSON",
    "StructuredCompletionRequest",
    "EmbeddingRequest",
]

# ============================================================================== #
# Aliases
# ============================================================================== #

Modality: TypeAlias = Optional[Literal["text", "audio"]]
ResponseMode = Union[Literal["text", "json"], Type[S]]
ResponseFormat: TypeAlias = ResponseTextConfigParam
Tool = Union[FunctionTool, "BuiltInTool"]
ToolChoice: TypeAlias = Union[
    Literal["none", "auto", "required"],
    Dict[str, Any],  # concrete selection dict
]
Metadata: TypeAlias = Dict[str, str]

# ============================================================================== #
# Helper data structures
# ============================================================================== #


class Reasoning(BaseModel):
    """Tunable reasoning knobs for o1/o3‑mini‑style models."""

    effort: Optional[ReasoningEffort] = Field(
        default=None,
        description="Constrain reasoning effort (e.g. 'low', 'medium', 'high').",
        examples=["low", "medium", "high"],
    )
    summary: Optional[Literal["auto", "concise", "detailed"]] = Field(
        default=None,
        description="Ask the model to summarise its chain‑of‑thought. Useful for debugging.",
        examples=["auto", "concise", "detailed"],
    )


class TextContentPart(TypedDict, total=False):
    """Single chunk of plain‑text content."""

    text: str
    type: Literal["text"]


class ResponsePrediction(TypedDict, total=False):
    """Token‑match hint allowing providers to short‑circuit generation early."""

    content: Union[str, Iterable[TextContentPart]]
    """The content that should be matched when generating a model response."""

# ============================================================================== #
# Base request models
# ============================================================================== #


class _BaseRequest(_BaseResource):
    """Common fields shared by *every* request Astral accepts."""

    astral: AstralParams = Field(
        default_factory=AstralParams,
        description="Internal tuning / retry knobs (opaque to most users).",
    )

    @property
    def request_id(self) -> str:
        """Unique identifier for the request."""
        return self.resource_id

    def safe_dump(self, *, exclude: Set[str] | None = None) -> Dict[str, Any]:
        """Return a dict minus volatile fields."""
        exclude = exclude or set()
        astral = self.astral.model_dump()
        request_dict = self.model_dump(
            exclude=exclude,
            exclude_none=True,
            exclude_unset=True,
            exclude_defaults=True,
        )
        request_dict["astral"] = astral
        return request_dict

    def dump_for_response(self, exclude_extra: Set[str] | None = None) -> Dict[str, Any]:
        """Return a dict minus volatile fields, excluding 'model'."""
        exclude = {"model"} | (exclude_extra or set())
        return self.safe_dump(exclude=exclude)

    model_config = ConfigDict(arbitrary_types_allowed=True)


# ============================================================================== #
# Base *completion* request with shared generation knobs
# ============================================================================== #


class _BaseCompletionRequest(_BaseRequest):
    """All common parameters for completion / chat requests."""

    system_message: SystemMessage | NotGiven = Field(
        default=NOT_GIVEN,
        description=(
            "System prompt defining assistant behaviour. Defaults to provider’s standard prompt."
        ),
        examples=[
            "You are a helpful AI assistant that specialises in Python.",
            "You are a financial advisor providing investment guidance.",
        ],
    )

    messages: Messages = Field(
        description="Conversation history (user ↔ assistant).",
        examples=[
            [{"role": "user", "content": "Hello!"}],
            [
                {"role": "user", "content": "Capital of France?"},
                {"role": "assistant", "content": "Paris."},
            ],
        ],
    )

    max_output_tokens: int | NotGiven | None = Field(
        default=NOT_GIVEN,
        description="Upper bound on generated tokens for the *assistant* response.",
        examples=[1024, 2048, 4000],
    )

    metadata: Metadata | NotGiven = Field(
        default=NOT_GIVEN,
        description="Up to 16 key/value pairs for tracing and analytics.",
        examples=[{"user_id": "123", "session": "abc"}],
    )

    parallel_tool_calls: bool | NotGiven | None = Field(
        default=NOT_GIVEN,
        description="Allow the model to call multiple tools concurrently.",
        examples=[True, False],
    )

    reasoning: Reasoning | NotGiven | None = Field(
        default=NOT_GIVEN,
        description="Fine‑tune step‑by‑step reasoning effort & summary style.",
        examples=[{"effort": "high"}, {"summary": "concise"}],
    )

    response_mode: ResponseMode | NotGiven = Field(
        default=NOT_GIVEN,
        description="Desired output style: 'text', 'json', or a Pydantic model class.",
        examples=["text", "json", "MySchema"],
    )

    temperature: float | NotGiven | None = Field(
        default=NOT_GIVEN,
        description="Randomness of the output (0‑2). Lower is deterministic; higher creative.",
        examples=[0.0, 0.7, 1.2],
    )

    tool_choice: ToolChoice | NotGiven = Field(
        default=NOT_GIVEN,
        description="Decide which (if any) tools the model should invoke.",
        examples=[
            "none",
            "auto",
            {"type": "function", "function": {"name": "get_weather"}},
        ],
    )

    function_tools: List[FunctionTool] | NotGiven = Field(
        default=NOT_GIVEN,
        description="Custom functions the model can call.",
    )

    built_in_tools: List[Dict[str, Any]] | NotGiven = Field(
        default=NOT_GIVEN,
        description="Provider‑supplied tools such as web_search or file_search.",
    )

    top_p: float | NotGiven | None = Field(
        default=NOT_GIVEN,
        description="Nucleus sampling cutoff (0‑1). Use instead of temperature to limit diversity.",
        examples=[0.1, 0.95],
    )

    truncation: Literal["auto", "disabled"] | NotGiven | None = Field(
        default=NOT_GIVEN,
        description="Context window overflow behaviour.",
        examples=["auto", "disabled"],
    )

    user: str | NotGiven = Field(
        default=NOT_GIVEN,
        description="External end‑user identifier for abuse monitoring.",
        examples=["user_12345"],
    )

    timeout: float | Timeout | None | NotGiven = Field(
        default=NOT_GIVEN,
        description="Override client default timeout (seconds).",
        examples=[30, 120],
    )

# ============================================================================== #
# Concrete request classes
# ============================================================================== #


class CompletionRequest(_BaseCompletionRequest):
    """Standard chat completion request."""

    pass


class CompletionRequestJSON(_BaseCompletionRequest):
    """Completion that explicitly requires a JSON string response."""

    response_mode: Literal["json"] = Field("json", description="Force JSON output from the model.")


class StructuredCompletionRequest(_BaseCompletionRequest, Generic[S]):
    """Completion expecting output conforming to **StructuredOutputResponseT**."""

    response_mode: Type[S] = Field(..., description="Pydantic schema the model must satisfy.")


# ============================================================================== #
# Embedding requests
# ============================================================================== #


class EmbeddingRequest(_BaseRequest):
    """Generate vector embeddings for text or batch of texts."""

    input: Union[str, List[str]] = Field(
        ..., description="Text(s) to embed.",
    )
