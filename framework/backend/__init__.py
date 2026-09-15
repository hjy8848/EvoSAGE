"""Backend environments, cases and tool result types."""

from .types import ActionResult, BackendEvent, CaseSpec, ToolCall, ToolResult
from .base import BackendEnvironment
from .ecommerce import EcommerceBackend
from .factory import build_case_spec, create_backend

__all__ = [
    "ActionResult",
    "BackendEvent",
    "CaseSpec",
    "ToolCall",
    "ToolResult",
    "BackendEnvironment",
    "EcommerceBackend",
    "build_case_spec",
    "create_backend",
]
