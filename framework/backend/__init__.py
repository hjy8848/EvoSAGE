"""Backend environments, cases and tool result types."""

from .types import ActionResult, BackendEvent, CaseSpec, ToolCall, ToolResult, UserEnvironmentState
from .base import BackendEnvironment
from .ecommerce import EcommerceBackend
from .scenario import ScenarioBackend
from .factory import build_case_spec, create_backend

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
]
