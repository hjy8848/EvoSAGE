"""Backend environments, cases and tool result types."""

from .types import ActionResult, BackendEvent, CaseSpec, ToolCall, ToolResult, UserEnvironmentState
from .base import BackendEnvironment
from .ecommerce import EcommerceBackend
from .scenario import ScenarioBackend
from .factory import build_case_spec, create_backend
from .errors import (
    BACKEND_ERROR_POLICIES,
    BackendErrorPolicy,
    is_failed_backend_event,
    is_recoverable_backend_failure,
    is_terminal_backend_failure,
    policy_for_error,
)

__all__ = [
    "ActionResult",
    "BackendEvent",
    "CaseSpec",
    "ToolCall",
    "ToolResult",
    "UserEnvironmentState",
    "BackendEnvironment",
    "EcommerceBackend",
    "ScenarioBackend",
    "build_case_spec",
    "create_backend",
    "BACKEND_ERROR_POLICIES",
    "BackendErrorPolicy",
    "is_failed_backend_event",
    "is_recoverable_backend_failure",
    "is_terminal_backend_failure",
    "policy_for_error",
]
