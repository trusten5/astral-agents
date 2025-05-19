# ==============================================================================
# hooks.py — Hook system for agent/tool lifecycle events                        
# ==============================================================================
# Purpose: Define context, protocols, and registry to manage hook execution     
# Sections: Imports, Public API, Context, Protocols, Registry, Defaults         
# ==============================================================================

# ==============================================================================
# Imports                                                                        
# ==============================================================================

# Standard Library --------------------------------------------------------------
import logging
import traceback
from typing import Any, Callable, Dict, List, Optional, Protocol, Union


# ==============================================================================
# Public API                                                                     
# ==============================================================================

__all__ = [
    "HookContext",
    "AgentHook",
    "ToolHook",
    "ResultHook",
    "ErrorHook",
    "HookRegistry",
    "global_hooks",
]


# ==============================================================================
# Context                                                                       
# ==============================================================================

class HookContext:
    """
    Context object passed to each hook, representing the execution environment.

    Args:
        agent_name: Name of the agent.
        tool_name: Optional name of the tool in use.
        metadata: Optional dictionary for additional hook metadata.

    Returns:
        A context object with agent, tool, and metadata fields.
    """

    def __init__(
        self,
        agent_name: str,
        tool_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.agent_name = agent_name
        self.tool_name = tool_name
        self.metadata = metadata or {}

    def with_tool(self, tool_name: str) -> "HookContext":
        """Create a new context with an updated tool name."""
        return HookContext(
            agent_name=self.agent_name,
            tool_name=tool_name,
            metadata=self.metadata,
        )


# ==============================================================================
# Protocols                                                                      
# ==============================================================================

class AgentHook(Protocol):
    def __call__(self, context: HookContext, input_data: Any) -> None: ...


class ToolHook(Protocol):
    def __call__(self, context: HookContext, input_data: Any) -> None: ...


class ResultHook(Protocol):
    def __call__(self, context: HookContext, result: Any) -> None: ...


class ErrorHook(Protocol):
    def __call__(self, context: HookContext, error: Exception) -> None: ...


# ==============================================================================
# Registry                                                                      
# ==============================================================================

class HookRegistry:
    """
    Registry for agent/tool lifecycle hooks.

    Allows registering multiple hooks per lifecycle event and executing them safely.
    """

    def __init__(self):
        self._agent_start_hooks: List[AgentHook] = []
        self._agent_end_hooks: List[ResultHook] = []
        self._tool_start_hooks: List[ToolHook] = []
        self._tool_end_hooks: List[ResultHook] = []
        self._error_hooks: List[ErrorHook] = []

    def register_agent_start(self, hook: AgentHook) -> None:
        """Register a hook to run when an agent starts."""
        self._agent_start_hooks.append(hook)

    def register_agent_end(self, hook: ResultHook) -> None:
        """Register a hook to run when an agent ends."""
        self._agent_end_hooks.append(hook)

    def register_tool_start(self, hook: ToolHook) -> None:
        """Register a hook to run when a tool starts."""
        self._tool_start_hooks.append(hook)

    def register_tool_end(self, hook: ResultHook) -> None:
        """Register a hook to run when a tool ends."""
        self._tool_end_hooks.append(hook)

    def register_error(self, hook: ErrorHook) -> None:
        """Register a hook to run when an error occurs."""
        self._error_hooks.append(hook)

    def run_agent_start(self, context: HookContext, input_data: Any) -> None:
        self._run_hooks(self._agent_start_hooks, context, input_data)

    def run_agent_end(self, context: HookContext, result: Any) -> None:
        self._run_hooks(self._agent_end_hooks, context, result)

    def run_tool_start(self, context: HookContext, input_data: Any) -> None:
        self._run_hooks(self._tool_start_hooks, context, input_data)

    def run_tool_end(self, context: HookContext, result: Any) -> None:
        self._run_hooks(self._tool_end_hooks, context, result)

    def run_error(self, context: HookContext, error: Exception) -> None:
        self._run_hooks(self._error_hooks, context, error, is_error=True)

    def _run_hooks(
        self,
        hooks: List[Union[AgentHook, ToolHook, ResultHook, ErrorHook]],
        context: HookContext,
        data: Any,
        is_error: bool = False,
    ) -> None:
        for hook in hooks:
            try:
                hook(context, data)
            except Exception as e:
                logger.warning(
                    f"[HOOK ERROR] Exception in hook for agent '{context.agent_name}', "
                    f"tool '{context.tool_name}': {e}\n{traceback.format_exc()}"
                )
                # Prevent infinite recursion if the error occurs in an error hook
                if not is_error:
                    self.run_error(context, e)


# ==============================================================================
# Defaults                                                                       
# ==============================================================================

global_hooks = HookRegistry()


def default_log_agent_start(context: HookContext, input_data: Any) -> None:
    logger.info(f"[AGENT START] {context.agent_name} | input: {input_data}")


def default_log_agent_end(context: HookContext, result: Any) -> None:
    logger.info(f"[AGENT END] {context.agent_name} | result: {result}")


def default_log_tool_start(context: HookContext, input_data: Any) -> None:
    logger.info(
        f"[TOOL START] {context.tool_name} (Agent: {context.agent_name}) | input: {input_data}"
    )


def default_log_tool_end(context: HookContext, result: Any) -> None:
    logger.info(
        f"[TOOL END] {context.tool_name} (Agent: {context.agent_name}) | result: {result}"
    )


def default_log_error(context: HookContext, error: Exception) -> None:
    logger.error(
        f"[ERROR] Agent '{context.agent_name}', Tool '{context.tool_name}': {repr(error)}"
    )


# Register default hooks
global_hooks.register_agent_start(default_log_agent_start)
global_hooks.register_agent_end(default_log_agent_end)
global_hooks.register_tool_start(default_log_tool_start)
global_hooks.register_tool_end(default_log_tool_end)
global_hooks.register_error(default_log_error)
