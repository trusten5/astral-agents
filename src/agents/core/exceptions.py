# ==============================================================================
# exceptions.py — Agents SDK Exception Definitions
# ==============================================================================
# Purpose: Define custom exceptions for Agents SDK error handling
# Sections: Imports, Public exports, Exceptions
# ==============================================================================

# Standard Library --------------------------------------------------------------
from typing import TYPE_CHECKING

# Internal ----------------------------------------------------------------------
if TYPE_CHECKING:
    from .guardrail import InputGuardrailResult, OutputGuardrailResult


# ==============================================================================
# Public exports
# ==============================================================================
__all__ = [
    "AgentsException",
    "MaxTurnsExceeded",
    "ModelBehaviorError",
    "UserError",
    "InputGuardrailTripwireTriggered",
    "OutputGuardrailTripwireTriggered",
]


# ==============================================================================
# Exceptions
# ==============================================================================


class AgentsException(Exception):
    """Base class for all exceptions in the Agents SDK."""


class MaxTurnsExceeded(AgentsException):
    """Exception raised when the maximum number of turns is exceeded."""

    message: str

    def __init__(self, message: str):
        self.message = message


class ModelBehaviorError(AgentsException):
    """
    Exception raised when the model does something unexpected,
    e.g., calling a tool that doesn't exist or providing malformed JSON.
    """

    message: str

    def __init__(self, message: str):
        self.message = message


class UserError(AgentsException):
    """Exception raised when the user makes an error using the SDK."""

    message: str

    def __init__(self, message: str):
        self.message = message


class InputGuardrailTripwireTriggered(AgentsException):
    """Exception raised when a guardrail tripwire is triggered."""

    guardrail_result: "InputGuardrailResult"
    """The result data of the guardrail that was triggered."""

    def __init__(self, guardrail_result: "InputGuardrailResult"):
        self.guardrail_result = guardrail_result
        super().__init__(
            f"Guardrail {guardrail_result.guardrail.__class__.__name__} triggered tripwire"
        )


class OutputGuardrailTripwireTriggered(AgentsException):
    """Exception raised when a guardrail tripwire is triggered."""

    guardrail_result: "OutputGuardrailResult"
    """The result data of the guardrail that was triggered."""

    def __init__(self, guardrail_result: "OutputGuardrailResult"):
        self.guardrail_result = guardrail_result
        super().__init__(
            f"Guardrail {guardrail_result.guardrail.__class__.__name__} triggered tripwire"
        )
