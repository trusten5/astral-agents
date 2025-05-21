# ==============================================================================
# agent.py — Definition of the Agent class and agent-to-tool wrapping logic
# ==============================================================================
# Purpose: Define the Agent class used to interact with models, tools, and
#          workflows, including MCP integrations and system prompt logic.
# Sections: Imports, Public Exports, Agent Definition, Public Methods
# ==============================================================================

# ==============================================================================
# Imports
# ==============================================================================

# Standard Library --------------------------------------------------------------
from __future__ import annotations
import dataclasses
import inspect
from collections.abc import Awaitable
from dataclasses import dataclass, field

# Typing -----------------------------------------------------------------------
from typing import TYPE_CHECKING, Any, Callable, Generic, Literal, cast
from typing_extensions import NotRequired, TypeAlias, TypedDict, TypeVar

# Internal ---------------------------------------------------------------------
# from .run_context import RunContextWrapper
# from .models import Model, ModelSettings
# from .tools import Tool, function_tool
# from .mcp import MCPServer, MCPConfig, MCPUtil
from guardrail import InputGuardrail, OutputGuardrail
# from .agent_hooks import AgentHooks
# from handoff import Handoff
# from .run_result import RunResult, ItemHelpers
# from .utils import _transforms, logger, MaybeAwaitable, StopAtTools, \
#     ToolsToFinalOutputFunction, ToolToFinalOutputResult, AgentOutputSchemaBase

# ==============================================================================
# Public exports
# ==============================================================================
__all__ = ["Agent"]

# ==============================================================================
# Type Variables
# ==============================================================================
TContext = TypeVar("TContext", default=Any)

# ==============================================================================
# Agent Definition
# ==============================================================================

@dataclass
class Agent(Generic[TContext]):
    """
    Defines an LLM-powered Agent with tools, models, and optional handoffs.
    """
    name: str
    instructions: str | Callable[[RunContextWrapper[TContext], Agent[TContext]],
                                 MaybeAwaitable[str]] | None = None
    # `instructions` can be a static string or a function returning a string.
    # This allows agents to have either fixed prompts or context-aware dynamic prompts.

    handoff_description: str | None = None
    handoffs: list[Agent[Any] | Handoff[TContext]] = field(default_factory=list)

    model: str | Model | None = None
    model_settings: ModelSettings = field(default_factory=ModelSettings)

    tools: list[Tool] = field(default_factory=list)
    mcp_servers: list[MCPServer] = field(default_factory=list)
    mcp_config: MCPConfig = field(default_factory=lambda: MCPConfig())

    input_guardrails: list[InputGuardrail[TContext]] = field(default_factory=list)
    output_guardrails: list[OutputGuardrail[TContext]] = field(default_factory=list)

    output_type: type[Any] | AgentOutputSchemaBase | None = None
    hooks: AgentHooks[TContext] | None = None

    tool_use_behavior: Literal["run_llm_again", "stop_on_first_tool"] \
        | StopAtTools \
        | ToolsToFinalOutputFunction = "run_llm_again"

    reset_tool_choice: bool = True

    # ==========================================================================
    # Public Methods
    # ==========================================================================

    def clone(self, **kwargs: Any) -> Agent[TContext]:
        """
        Create a copy of the Agent with optional updated fields.

        Args:
            **kwargs: Any field overrides.

        Returns:
            New Agent instance.
        """
        return dataclasses.replace(self, **kwargs)

    def as_tool(
        self,
        tool_name: str | None = None,
        tool_description: str | None = None,
        custom_output_extractor: Callable[[RunResult], Awaitable[str]] | None = None,
    ) -> Tool:
        """
        Wrap the agent as a Tool instance usable by orchestrators.

        Args:
            tool_name: Optional override for the tool's name.
            tool_description: Optional override for the tool's description.
            custom_output_extractor: Optional function to extract tool output.

        Returns:
            A callable Tool instance.
        """

        @function_tool(
            name_override=tool_name or _transforms.transform_string_function_style(self.name),
            description_override=tool_description or "",
        )
        async def run_agent(context: RunContextWrapper, input: str) -> str:
            from .run import Runner
            output = await Runner.run(
                starting_agent=self,
                input=input,
                context=context.context,
            )
            if custom_output_extractor:
                return await custom_output_extractor(output)
            return ItemHelpers.text_message_outputs(output.new_items)

        return run_agent

    async def get_system_prompt(self, run_context: RunContextWrapper[TContext]) -> str | None:
        """
        Get the system prompt from the agent's instructions.

        Args:
            run_context: The run context wrapper.

        Returns:
            System prompt string or None.
        """
        if isinstance(self.instructions, str):
            return self.instructions
        elif callable(self.instructions):
            result = self.instructions(run_context, self)
            return await result if inspect.iscoroutine(result) else result
        elif self.instructions is not None:
            logger.error(
                f"Instructions must be a string or function, got {self.instructions}"
            )
        return None

    async def get_mcp_tools(self) -> list[Tool]:
        """
        Retrieve tools from all configured MCP servers.

        Returns:
            List of Tool objects.
        """
        return await MCPUtil.get_all_function_tools(
            self.mcp_servers,
            self.mcp_config.get("convert_schemas_to_strict", False),
        )

    async def get_all_tools(self) -> list[Tool]:
        """
        Combine MCP and local tools.

        Returns:
            Complete list of tools.
        """
        return [*await self.get_mcp_tools(), *self.tools]
