# ==============================================================================
# response.py — canonical response schema for payloads2 module
# ==============================================================================

from __future__ import annotations

from typing import Any, Dict, Generic, Mapping, Optional, TypeVar, Union, Literal

from pydantic import BaseModel, Field, PrivateAttr, model_validator, ConfigDict
from pydantic_core import to_jsonable_python

# Assume these are your internal imports that mirror Astral's internal models
# from ._base import _BaseResource
# from .usage import BaseUsage, BaseCost, ChatUsage, ChatCost
# from .request import _BaseCompletionRequest
# from .resources.completion import CompletionOutput, CompletionContent
# from .streaming_events import LifecycleEvent, ContentEvent  # your streaming event enums

# ==============================================================================
# Public exports
# ==============================================================================

__all__ = [
    "_BaseResponse",
    "CompletionResponse",
    "StructuredCompletionResponse",
    "StreamStartResponse",
    "StreamDeltaResponse",
    "StreamCompleteResponse",
    "StreamStructuredStartResponse",
    "StreamStructuredCompleteResponse",
]

# ==============================================================================
# Type vars & aliases
# ==============================================================================

ResponseT = TypeVar("ResponseT", bound="_BaseResponse")
UsageT = TypeVar("UsageT", bound=BaseUsage)
CostT = TypeVar("CostT", bound=BaseCost)

RequestParamsType = Union[Dict[str, Any], _BaseCompletionRequest]
ResponseMode = Literal["standard", "structured"]

# ==============================================================================
# Private helper mixin for propagating response metadata into nested objects
# ==============================================================================


class _PropagatePrivate(BaseModel, Generic[UsageT, CostT]):
    """Mixin to propagate response_id and provider info into usage & cost objects."""

    @model_validator(mode="after")
    def _after_init(self: ResponseT) -> ResponseT:
        if getattr(self, "usage", None) is not None:
            self.usage._response_id = self.response_id  # type: ignore[attr-defined]
            self.usage._model_provider = self.provider_name
            self.usage._model_name = self.model

        if getattr(self, "cost", None) is not None:
            self.cost._response_id = self.response_id  # type: ignore[attr-defined]
            self.cost._model_provider = self.provider_name
            self.cost._model_name = self.model

        return self


# ==============================================================================
# Rate limit metadata container
# ==============================================================================


class RateLimit(BaseModel):
    tier: str = Field(..., description="Pricing or quota tier, e.g. 'paid'")
    limit: int = Field(..., description="Total requests allowed in the window")
    remaining: int = Field(..., description="Requests left before throttling")
    reset: int = Field(..., description="Epoch seconds until quota resets")


# ==============================================================================
# Base Response with common fields
# ==============================================================================


class _BaseResponse(_BaseResource):
    usage: Optional[BaseUsage] = Field(default=None, description="Token accounting")
    cost: Optional[BaseCost] = Field(default=None, description="Spend estimate")
    latency_ms: Optional[float] = Field(default=None, description="End-to-end latency")

    _rate_limits: Optional[RateLimit] = PrivateAttr(default=None)

    @property
    def response_id(self) -> str:
        return self.resource_id

    @property
    def rate_limits(self) -> Optional[RateLimit]:
        return self._rate_limits

    @rate_limits.setter
    def rate_limits(self, value: Optional[RateLimit]) -> None:
        self._rate_limits = value

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        # Always serialize usage and cost nested models properly
        kwargs.pop("serialize_as_any", None)  # ignore user flag for safety

        result = super().model_dump(serialize_as_any=True, **kwargs)
        ordered = {"response_id": self.response_id, "provider_name": self.provider_name, **result}
        return ordered

    def model_dump_json(self, **kwargs) -> str:
        kwargs.pop("serialize_as_any", None)

        def to_json(obj: Any) -> str:
            return json.dumps(to_jsonable_python(obj))

        return to_json(self.model_dump(**kwargs))


# ==============================================================================
# Base Completion Response with usage/cost and request metadata
# ==============================================================================


class _BaseCompletionResponse(_PropagatePrivate[ChatUsage, ChatCost], _BaseResponse):
    usage: Optional[ChatUsage] = Field(default=None, description="How many tokens were used")
    cost: Optional[ChatCost] = Field(default=None, description="How much did the completion cost")

    request_params: RequestParamsType = Field(default_factory=dict, description="The request that produced this response")
    metadata: Mapping[str, Any] = Field(default_factory=dict, description="Metadata about the request")

    def __init__(self, **data: Any) -> None:
        if "request_params" in data and isinstance(data["request_params"], _BaseCompletionRequest):
            data["request_params"] = data["request_params"].dump_for_response()

        params = dict(data.pop("request_params", {}))
        metadata = data.get("metadata", params.get("metadata", {}))

        super().__init__(**data, request_params=params, metadata=metadata)


# ==============================================================================
# CompletionResponse — unrestricted free-form text / tool use
# ==============================================================================


class CompletionResponse(_BaseCompletionResponse):
    response_mode: ResponseMode = Field(default="standard", description="The type of response", example="standard")
    output: CompletionOutput = Field(..., description="Raw message output and tool use")
    content: Optional[CompletionContent] = Field(default=None, description="Aggregated text/parts for quick access")

    def __init__(self, **data: Any) -> None:
        if "content" not in data and "output" in data:
            data["content"] = CompletionContent.from_output(data["output"])
        super().__init__(**data)


# ==============================================================================
# StructuredCompletionResponse — JSON / function calling
# ==============================================================================


class StructuredCompletionResponse(_BaseCompletionResponse, Generic[ResponseT]):
    response_mode: ResponseMode = Field(default="structured", description="The type of response", example="structured")
    output: CompletionOutput = Field(..., description="Raw messages and tool calls")
    content: Optional[CompletionContent] = Field(default=None, description="Aggregated text/parts")
    parsed: Optional[ResponseT] = Field(default=None, description="User supplied Pydantic model parsed from JSON")

    def __init__(self, **data: Any) -> None:
        if "content" not in data and "output" in data:
            data["content"] = CompletionContent.from_output(data["output"])
        super().__init__(**data)


# ==============================================================================
# Streaming response base class for events
# ==============================================================================


class BaseStreamEvent(BaseModel):
    stream_event: Union[LifecycleEvent, ContentEvent] = Field(..., description="Stream lifecycle or content event")


class StreamStartResponse(_BaseCompletionResponse, BaseStreamEvent):
    stream_event: Literal[LifecycleEvent.STREAM_START] = Field(LifecycleEvent.STREAM_START)
    output: Optional[CompletionOutput] = Field(default=None, description="Raw message output / deltas")
    content: Optional[CompletionContent] = Field(default=None, description="Aggregated text / parts")


class StreamDeltaResponse(BaseStreamEvent, Generic[ResponseT]):
    stream_event: ContentEvent
    payload: ResponseT
    raw_provider_payload: Dict[str, Any]

    model_config = ConfigDict(use_enum_values=True)


class StreamCompleteResponse(_BaseCompletionResponse, BaseStreamEvent):
    stream_event: Literal[LifecycleEvent.STREAM_COMPLETE] = Field(LifecycleEvent.STREAM_COMPLETE)
    output: CompletionOutput = Field(..., description="Raw message output / deltas")
    content: Optional[CompletionContent] = Field(default=None, description="Aggregated text / parts")

    def __init__(self, **data: Any) -> None:
        if "content" not in data and "output" in data:
            data["content"] = CompletionContent.from_output(data["output"])
        super().__init__(**data)


# ==============================================================================
# Structured streaming response variants
# ==============================================================================


class StreamStructuredStartResponse(StreamStartResponse, Generic[ResponseT]):
    response_mode: ResponseMode = Field("structured", frozen=True)
    parsed: Optional[ResponseT] = Field(default=None, description="User-supplied parsed Pydantic model")


class StreamStructuredCompleteResponse(StreamCompleteResponse, Generic[ResponseT]):
    response_mode: ResponseMode = Field("structured", frozen=True)
    parsed: Optional[ResponseT] = Field(default=None, description="User-supplied parsed Pydantic model")


# ==============================================================================
# Type aliases for convenience
# ==============================================================================

StreamResponse = Union[
    StreamStartResponse,
    StreamDeltaResponse[Any],
    StreamCompleteResponse,
]

StructuredStreamResponse = Union[
    StreamStructuredStartResponse[Any],
    StreamDeltaResponse[Any],
    StreamStructuredCompleteResponse[Any],
]

