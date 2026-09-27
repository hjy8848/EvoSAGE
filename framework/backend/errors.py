"""Stable recovery policy for backend error codes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BackendErrorPolicy:
    recoverable: bool
    requires_customer_input: bool = False
    retryable_same_turn: bool = False
    retryable_next_turn: bool = False
    terminal: bool = False


BACKEND_ERROR_POLICIES = {
    "order_not_found": BackendErrorPolicy(
        recoverable=True, requires_customer_input=True, retryable_next_turn=True,
    ),
    "customer_not_found": BackendErrorPolicy(
        recoverable=True, requires_customer_input=True, retryable_next_turn=True,
    ),
    "record_not_found": BackendErrorPolicy(
        recoverable=True, requires_customer_input=True, retryable_next_turn=True,
    ),
    "order_not_verified": BackendErrorPolicy(recoverable=True, retryable_same_turn=True),
    "record_not_verified": BackendErrorPolicy(recoverable=True, retryable_same_turn=True),
    "refund_requires_review": BackendErrorPolicy(recoverable=True, retryable_next_turn=True),
    "refund_not_eligible": BackendErrorPolicy(recoverable=False),
    "interception_unavailable": BackendErrorPolicy(recoverable=False),
    "pickup_unavailable": BackendErrorPolicy(recoverable=False),
    "unknown_tool": BackendErrorPolicy(recoverable=False),
    "unsupported_action": BackendErrorPolicy(recoverable=False),
    "backend_exception": BackendErrorPolicy(recoverable=False, terminal=True),
}

UNKNOWN_BACKEND_ERROR_POLICY = BackendErrorPolicy(recoverable=False)


def policy_for_error(error_code: Any) -> BackendErrorPolicy:
    """Return a fail-closed policy; only explicitly fatal codes terminate."""
    return BACKEND_ERROR_POLICIES.get(str(error_code or ""), UNKNOWN_BACKEND_ERROR_POLICY)


def is_failed_backend_event(event: dict) -> bool:
    result = event.get("result") or {}
    return (
        event.get("event_type") in {"tool_call", "action_execution"}
        and not bool(result.get("success"))
    )


def is_terminal_backend_failure(event: dict) -> bool:
    return is_failed_backend_event(event) and policy_for_error(
        (event.get("result") or {}).get("error_code")
    ).terminal


def is_recoverable_backend_failure(event: dict) -> bool:
    return is_failed_backend_event(event) and policy_for_error(
        (event.get("result") or {}).get("error_code")
    ).recoverable
