# ==============================================================================
# guardrails.py — Guardrail functions and decorators                           
# ==============================================================================
# Purpose: Define input/output guardrail classes and decorators for agent checks
# Sections: Imports, Public exports, Data Classes, Guardrail Classes, Type Aliases, Decorators
# ==============================================================================

# ==============================================================================
# Imports                                                                      
# ==============================================================================

# Standard Library --------------------------------------------------------------
from __future__ import annotations
import inspect
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Union,
    Awaitable,
    overload,
)

# Third-Party -------------------------------------------------------------------
from typing_extensions import TypeVar

# Internal ----------------------------------------------------------------------
from .exceptions import UserError
from .items import TResponseInputItem
from .run_context import RunContextWrapper, TContext
from .util._types import MaybeAwaitable

if TYPE_CHECKING:
    from .agent import Agent


# ==============================================================================
# Public exports                                                               
# ==============================================================================
__all__ = [
    'GuardrailFunctionOutput',
    'InputGuardrailResult',
    'OutputGuardrailResult',
    'InputGuardrail',
    'OutputGuardrail',
    'input_guardrail',
    'output_guardrail',
]


# ==============================================================================
# Data Classes                                                                 
# ==============================================================================

@dataclass
class GuardrailFunctionOutput:
    """The output of a guardrail function.

    Attributes:
        output_info: Optional metadata about what the guardrail did.
        tripwire_triggered: If True, agent execution should halt.
    """
    output_info: Any
    tripwire_triggered: bool


@dataclass
class InputGuardrailResult:
    """Container for the result of running an input guardrail."""
    guardrail: InputGuardrail[Any]
    output: GuardrailFunctionOutput


@dataclass
class OutputGuardrailResult:
    """Container for the result of running an output guardrail."""
    guardrail: OutputGuardrail[Any]
    agent_output: Any
    agent: Agent[Any]
    output: GuardrailFunctionOutput


# ==============================================================================
# Guardrail Classes                                                            
# ==============================================================================

@dataclass
class InputGuardrail(Generic[TContext]):
    """Runs checks in parallel to agent execution to validate input or context."""

    guardrail_function: Callable[
        [RunContextWrapper[TContext], Agent[Any], str | list[TResponseInputItem]],
        MaybeAwaitable[GuardrailFunctionOutput],
    ]
    name: str | None = None

    def get_name(self) -> str:
        return self.name or self.guardrail_function.__name__

    async def run(
        self,
        context: RunContextWrapper[TContext],
        agent: Agent[Any],
        input: str | list[TResponseInputItem],
    ) -> InputGuardrailResult:
        if not callable(self.guardrail_function):
            raise UserError(
                f"Input guardrail function must be callable, got: {type(self.guardrail_function)}"
            )

        result = self.guardrail_function(context, agent, input)
        if inspect.isawaitable(result):
            result = await result

        return InputGuardrailResult(guardrail=self, output=result)


@dataclass
class OutputGuardrail(Generic[TContext]):
    """Runs validation on the agent's final output."""

    guardrail_function: Callable[
        [RunContextWrapper[TContext], Agent[Any], Any],
        MaybeAwaitable[GuardrailFunctionOutput],
    ]
    name: str | None = None

    def get_name(self) -> str:
        return self.name or self.guardrail_function.__name__

    async def run(
        self,
        context: RunContextWrapper[TContext],
        agent: Agent[Any],
        agent_output: Any,
    ) -> OutputGuardrailResult:
        if not callable(self.guardrail_function):
            raise UserError(
                f"Output guardrail function must be callable, got: {type(self.guardrail_function)}"
            )

        result = self.guardrail_function(context, agent, agent_output)
        if inspect.isawaitable(result):
            result = await result

        return OutputGuardrailResult(
            guardrail=self,
            agent_output=agent_output,
            agent=agent,
            output=result,
        )


# ==============================================================================
# Type Aliases                                                                 
# ==============================================================================

TContext_co = TypeVar("TContext_co", bound=Any, covariant=True)

_InputGuardrailFunc = Callable[
    [RunContextWrapper[TContext_co], Agent[Any], str | list[TResponseInputItem]],
    MaybeAwaitable[GuardrailFunctionOutput],
]

_OutputGuardrailFunc = Callable[
    [RunContextWrapper[TContext_co], Agent[Any], Any],
    MaybeAwaitable[GuardrailFunctionOutput],
]


# ==============================================================================
# Decorators                                                                   
# ==============================================================================

@overload
def input_guardrail(
    func: _InputGuardrailFunc[TContext_co],
) -> InputGuardrail[TContext_co]: ...


@overload
def input_guardrail(
    *, name: str | None = None,
) -> Callable[[_InputGuardrailFunc[TContext_co]], InputGuardrail[TContext_co]]: ...


def input_guardrail(
    func: _InputGuardrailFunc[TContext_co] | None = None,
    *,
    name: str | None = None,
) -> (
    InputGuardrail[TContext_co]
    | Callable[[_InputGuardrailFunc[TContext_co]], InputGuardrail[TContext_co]]
):
    """Decorator to define an input guardrail from a function."""
    def decorator(f: _InputGuardrailFunc[TContext_co]) -> InputGuardrail[TContext_co]:
        return InputGuardrail(guardrail_function=f, name=name)

    return decorator(func) if func else decorator


@overload
def output_guardrail(
    func: _OutputGuardrailFunc[TContext_co],
) -> OutputGuardrail[TContext_co]: ...


@overload
def output_guardrail(
    *, name: str | None = None,
) -> Callable[[_OutputGuardrailFunc[TContext_co]], OutputGuardrail[TContext_co]]: ...


def output_guardrail(
    func: _OutputGuardrailFunc[TContext_co] | None = None,
    *,
    name: str | None = None,
) -> (
    OutputGuardrail[TContext_co]
    | Callable[[_OutputGuardrailFunc[TContext_co]], OutputGuardrail[TContext_co]]
):
    """Decorator to define an output guardrail from a function."""
    def decorator(f: _OutputGuardrailFunc[TContext_co]) -> OutputGuardrail[TContext_co]:
        return OutputGuardrail(guardrail_function=f, name=name)

    return decorator(func) if func else decorator
