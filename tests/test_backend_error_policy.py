from framework.backend.errors import (
    BACKEND_ERROR_POLICIES,
    BackendErrorPolicy,
    policy_for_error,
    is_terminal_backend_failure,
)


def test_known_recoverable_backend_errors_have_explicit_retry_semantics():
    assert BACKEND_ERROR_POLICIES["order_not_found"] == BackendErrorPolicy(
        recoverable=True,
        requires_customer_input=True,
        retryable_next_turn=True,
    )
    assert policy_for_error("order_not_verified").retryable_same_turn
    assert policy_for_error("refund_requires_review").retryable_next_turn
    assert not policy_for_error("refund_not_eligible").recoverable
    assert not policy_for_error("refund_not_eligible").terminal


def test_only_explicit_fatal_backend_errors_terminate_unknown_errors_do_not():
    assert policy_for_error("backend_exception").terminal
    assert policy_for_error("an_unregistered_error") == BackendErrorPolicy(recoverable=False)
    assert is_terminal_backend_failure({
        "event_type": "tool_call",
        "result": {"success": False, "error_code": "backend_exception"},
    })
    assert not is_terminal_backend_failure({
        "event_type": "tool_call",
        "result": {"success": False, "error_code": "future_error_code"},
    })
    assert not is_terminal_backend_failure({
        "event_type": "tool_call",
        "result": {"success": True},
    })
