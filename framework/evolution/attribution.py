"""Deterministic evaluator-only attribution of failures to canonical SOP steps."""

from __future__ import annotations

from typing import Any, Iterable, Optional


def _path(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, (list, tuple)) else []


def _canonical_path(value: Iterable[str], expected: bool = False) -> list[str]:
    path = [item for item in _path(value) if item not in {"start", "end"}]
    # Legacy PathList canonical paths do not contain action_* nodes.  Keep
    # attribution aligned with the existing path scorer in that representation.
    if expected and not any(item.startswith("action_") for item in path):
        return path
    return path


def infer_failure_location(report: Any, simulation: Any = None,
                           path_config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Return a deterministic ``sop_node``/``path_step_index`` attribution.

    Indices are zero-based and refer to the normalized canonical policy path.
    This helper is analysis-only: it never changes prompts, execution, or
    benchmark scores.
    """
    details = getattr(report, "details", {}) or {}
    diagnostics = details.get("diagnostics", {}) if isinstance(details, dict) else {}
    case_spec = getattr(simulation, "case_spec", {}) or {}
    metadata = case_spec.get("metadata", {}) if isinstance(case_spec, dict) else {}
    path_config = path_config or metadata.get("path_config") or metadata.get("legacy_path_config") or {}

    expected = _path(getattr(report, "gold_path", None))
    if not expected:
        expected = _path(path_config.get("expected_path"))
    if not expected:
        expected = _path(metadata.get("expected_path"))
    predicted = _path(getattr(report, "predicted_path", None))
    if not predicted and isinstance(details, dict):
        predicted = _path(details.get("predicted_path"))

    error_types = set(getattr(report, "error_categories", []) or [])
    protocol_failure = (
        "json_parse_failed" in error_types
        or "protocol_failure" in error_types
        or bool(diagnostics.get("protocol_failure", False))
    )
    if protocol_failure and not predicted:
        return {"path_step_index": None, "sop_node": None, "reason": None}

    expected = _canonical_path(expected, expected=True)
    predicted = _canonical_path(predicted, expected=False)
    if expected and not any(item.startswith("action_") for item in expected):
        predicted = [item for item in predicted if not item.startswith("action_")]

    for index, (expected_node, predicted_node) in enumerate(zip(expected, predicted)):
        if expected_node != predicted_node:
            return {
                "path_step_index": index,
                "sop_node": expected_node,
                "reason": "canonical_path_divergence",
            }
    if len(expected) != len(predicted):
        index = min(len(expected), len(predicted))
        node = expected[index] if index < len(expected) else (predicted[index] if index < len(predicted) else None)
        return {
            "path_step_index": index,
            "sop_node": node,
            "reason": "canonical_path_length_divergence",
        }

    errors = error_types
    failure = not bool(getattr(report, "task_success", False))
    failure = failure or float(getattr(report, "required_verification_score", 1.0)) < 1.0
    failure = failure or float(getattr(report, "policy_compliance_score", 1.0)) < 1.0
    failure = failure or float(getattr(report, "action_execution_score", 1.0)) < 1.0
    failure = failure or float(getattr(report, "goal_fulfillment", 1.0)) < 1.0
    failure = failure or bool(errors)
    if expected and failure:
        # With a correct policy path, the final canonical node is the most
        # specific deterministic location available for action/goal failures.
        return {
            "path_step_index": len(expected) - 1,
            "sop_node": expected[-1],
            "reason": "terminal_execution_or_goal_failure",
        }

    # Preserve a deterministic attribution supplied by the evaluator only if
    # it is already numeric and tied to a canonical node; never invent one.
    index = diagnostics.get("first_divergence_step") if isinstance(diagnostics, dict) else None
    node = diagnostics.get("first_divergence_node") if isinstance(diagnostics, dict) else None
    if isinstance(index, int) or isinstance(node, str):
        return {"path_step_index": index if isinstance(index, int) else None, "sop_node": node, "reason": "evaluator_diagnostic"}
    return {"path_step_index": None, "sop_node": None, "reason": None}
