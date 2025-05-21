# ==============================================================================
# model.py — Conversion utilities for AI agent tools and responses    
# ==============================================================================
# Purpose: Convert internal tool representations to external API formats and     
#          provide generalized model response handling for AI agents.            
# Sections: Imports, Constants, Type Aliases, Data Classes, ToolConverter,       
#           ResponsesModel                                                      
# ==============================================================================

# ==============================================================================
# Imports                                                                        
# ==============================================================================

# Standard Library --------------------------------------------------------------
from __future__ import annotations
import json
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    overload,
    Union,
    List,
    Optional,
    AsyncIterator,
)

# Internal ----------------------------------------------------------------------
from .exceptions import UserError
# from .agent_output import AgentOutputSchemaBase
# from .handoffs import Handoff
# from .items import ItemHelpers, ModelResponse, TResponseInputItem
# from .logger import logger
# from .tool import ComputerTool, FileSearchTool, FunctionTool, Tool, WebSearchTool
# from .tracing import SpanError, response_span
# from .usage import Usage
# from .version import __version__
# from .interface import Model, ModelTracing

if TYPE_CHECKING:
    from .model_settings import ModelSettings

# ==============================================================================
# Constants                                                                     
# ==============================================================================

_USER_AGENT = f"Agents/Python {__version__}"
_HEADERS = {"User-Agent": _USER_AGENT}

# ==============================================================================
# Type Aliases                                                                   
# ==============================================================================

IncludeLiteral = Literal[
    "file_search_call.results",
    "message.input_image.image_url",
    "computer_call_output.output.image_url",
]

# ==============================================================================
# Data Classes                                                                   
# ==============================================================================

@dataclass
class ConvertedTools:
    tools: List[dict]
    includes: List[IncludeLiteral]

# ==============================================================================
# ToolConverter                                                                  
# ==============================================================================

class ToolConverter:
    @staticmethod
    def convert_tool_choice(tool_choice: Optional[str]) -> Union[str, dict, None]:
        if tool_choice in {"auto", "required", "none"}:
            return tool_choice
        if tool_choice:
            return {"type": tool_choice}
        return None

    @staticmethod
    def get_response_format(
        output_schema: Optional[AgentOutputSchemaBase],
    ) -> Optional[dict]:
        if output_schema and not output_schema.is_plain_text():
            return {
                "format": {
                    "type": "json_schema",
                    "name": "final_output",
                    "schema": output_schema.json_schema(),
                    "strict": output_schema.is_strict_json_schema(),
                }
            }
        return None

    @staticmethod
    def convert_tools(tools: List[Tool], handoffs: List[Handoff]) -> ConvertedTools:
        converted_tools: List[dict] = []
        includes: List[IncludeLiteral] = []

        computer_tools = [tool for tool in tools if isinstance(tool, ComputerTool)]
        if len(computer_tools) > 1:
            raise UserError(
                f"Only one computer tool is allowed. Found {len(computer_tools)}."
            )

        for tool in tools:
            converted_tool, include = ToolConverter._convert_tool(tool)
            converted_tools.append(converted_tool)
            if include:
                includes.append(include)

        for handoff in handoffs:
            converted_tools.append(ToolConverter._convert_handoff_tool(handoff))

        return ConvertedTools(tools=converted_tools, includes=includes)

    @staticmethod
    def _convert_tool(tool: Tool) -> tuple[dict, Optional[IncludeLiteral]]:
        if isinstance(tool, FunctionTool):
            return (
                {
                    "name": tool.name,
                    "parameters": tool.params_json_schema,
                    "strict": tool.strict_json_schema,
                    "type": "function",
                    "description": tool.description,
                },
                None,
            )

        if isinstance(tool, WebSearchTool):
            return (
                {
                    "type": "web_search_preview",
                    "user_location": tool.user_location,
                    "search_context_size": tool.search_context_size,
                },
                None,
            )

        if isinstance(tool, FileSearchTool):
            tool_dict = {
                "type": "file_search",
                "vector_store_ids": tool.vector_store_ids,
            }
            if tool.max_num_results:
                tool_dict["max_num_results"] = tool.max_num_results
            if tool.ranking_options:
                tool_dict["ranking_options"] = tool.ranking_options
            if tool.filters:
                tool_dict["filters"] = tool.filters

            include = (
                "file_search_call.results" if tool.include_search_results else None
            )
            return tool_dict, include

        if isinstance(tool, ComputerTool):
            return (
                {
                    "type": "computer_use_preview",
                    "environment": tool.computer.environment,
                    "display_width": tool.computer.dimensions[0],
                    "display_height": tool.computer.dimensions[1],
                },
                None,
            )

        raise UserError(f"Unknown tool type: {type(tool)}")

    @staticmethod
    def _convert_handoff_tool(handoff: Handoff) -> dict:
        return {
            "name": handoff.tool_name,
            "parameters": handoff.input_json_schema,
            "strict": handoff.strict_json_schema,
            "type": "function",
            "description": handoff.tool_description,
        }

# ==============================================================================
# ResponsesModel                                                                 
# ==============================================================================

class ResponsesModel(Model):
    """
    Generalized implementation of `Model` for AI agent responses.

    Args:
        model_name: Name of the AI model.
        client: Client instance to interact with the AI backend.
    """

    def __init__(self, model_name: str, client: Any) -> None:
        self.model_name = model_name
        self._client = client

    def _non_null_or_default(self, value: Any, default: Any = None) -> Any:
        return value if value is not None else default

    async def get_response(
        self,
        system_instructions: Optional[str],
        input_data: Union[str, List[TResponseInputItem]],
        model_settings: ModelSettings,
        tools: List[Tool],
        output_schema: Optional[AgentOutputSchemaBase],
        handoffs: List[Handoff],
        tracing: ModelTracing,
        previous_response_id: Optional[str],
    ) -> ModelResponse:
        """
        Fetch a complete response from the model.

        Returns:
            ModelResponse containing output, usage, and response ID.
        """
        with response_span(disabled=tracing.is_disabled()) as span_response:
            try:
                response = await self._fetch_response(
                    system_instructions,
                    input_data,
                    model_settings,
                    tools,
                    output_schema,
                    handoffs,
                    previous_response_id,
                    stream=False,
                )

                logger.debug("LLM responded with output.")
                usage = Usage(
                    requests=1,
                    input_tokens=response.get("usage", {}).get("input_tokens", 0),
                    output_tokens=response.get("usage", {}).get("output_tokens", 0),
                    total_tokens=response.get("usage", {}).get("total_tokens", 0),
                )

                if tracing.include_data():
                    span_response.span_data.response = response
                    span_response.span_data.input = input_data

            except Exception as e:
                span_response.set_error(
                    SpanError(message="Error getting response", data={"error": str(e)})
                )
                logger.error(f"Error getting response: {e}")
                raise

        return ModelResponse(
            output=response.get("output", []),
            usage=usage,
            response_id=response.get("id"),
        )

    async def stream_response(
        self,
        system_instructions: Optional[str],
        input_data: Union[str, List[TResponseInputItem]],
        model_settings: ModelSettings,
        tools: List[Tool],
        output_schema: Optional[AgentOutputSchemaBase],
        handoffs: List[Handoff],
        tracing: ModelTracing,
        previous_response_id: Optional[str],
    ) -> AsyncIterator[dict]:
        """
        Stream response chunks asynchronously.

        Yields:
            Response chunks as dicts.
        """
        with response_span(disabled=tracing.is_disabled()) as span_response:
            try:
                stream = await self._fetch_response(
                    system_instructions,
                    input_data,
                    model_settings,
                    tools,
                    output_schema,
                    handoffs,
                    previous_response_id,
                    stream=True,
                )

                final_response = None

                async for chunk in stream:
                    if chunk.get("type") == "response_completed":
                        final_response = chunk.get("response")
                    yield chunk

                if final_response and tracing.include_data():
                    span_response.span_data.response = final_response
                    span_response.span_data.input = input_data

            except Exception as e:
                span_response.set_error(
                    SpanError(message="Error streaming response", data={"error": str(e)})
                )
                logger.error(f"Error streaming response: {e}")
                raise

    async def _fetch_response(
        self,
        system_instructions: Optional[str],
        input_data: Union[str, List[TResponseInputItem]],
        model_settings: ModelSettings,
        tools: List[Tool],
        output_schema: Optional[AgentOutputSchemaBase],
        handoffs: List[Handoff],
        previous_response_id: Optional[str],
        stream: bool = False,
    ) -> Union[dict, AsyncIterator[dict]]:
        """
        Internal method to prepare inputs and call the client for a response.

        Returns:
            Either a complete response dict or an async iterator for streaming.
        """
        list_input = ItemHelpers.input_to_new_input_list(input_data)

        parallel_tool_calls = (
            True
            if model_settings.parallel_tool_calls and tools
            else False
            if model_settings.parallel_tool_calls is False
            else None
        )

        tool_choice = ToolConverter.convert_tool_choice(model_settings.tool_choice)
        converted_tools = ToolConverter.convert_tools(tools, handoffs)
        response_format = ToolConverter.get_response_format(output_schema)

        params = {
            "model": self.model_name,
            "inputs": list_input,
            "temperature": model_settings.temperature,
            "max_tokens": model_settings.max_tokens,
            "top_p": model_settings.top_p,
            "frequency_penalty": model_settings.frequency_penalty,
            "presence_penalty": model_settings.presence_penalty,
            "stop_sequences": model_settings.stop_sequences,
            "stream": stream,
            "n": model_settings.n,
            "logit_bias": model_settings.logit_bias,
            "logprobs": model_settings.logprobs,
            "user": model_settings.user,
            "functions": converted_tools.tools,
            "function_call": tool_choice,
            "parallel_tool_calls": parallel_tool_calls,
            "tool_include": converted_tools.includes,
            "output_schema": response_format,
            "system": system_instructions,
            "parent_response_id": previous_response_id,
            "client_trace_id": tracing.trace_id,
            "client_span_id": tracing.span_id,
        }

        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}

        if stream:
            return await self._client.stream(params)
        else:
            return await self._client.call(params)
