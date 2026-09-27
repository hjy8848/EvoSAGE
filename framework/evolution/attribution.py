"""Deterministic evaluator-only attribution of failures to SOP and lifecycle stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Iterable, Optional


FAILURE_STAGES = frozenset({
    "INPUT_COLLECTION",
    "VERIFICATION",
    "DECISION",
    "ACTION",
    "CONFIRMATION",
    "RECOVERY",
    "TERMINATION",
    "UNKNOWN",
})

ERROR_STAGE_MAP = {
    "canonical_path_divergence": ("DECISION", "SOP_POLICY_PATH_DIVERGED"),
    "missed_backend_verification": ("VERIFICATION", "REQUIRED_BACKEND_FACT_NOT_VERIFIED"),
    "wrong_tool_selection": ("VERIFICATION", "REQUIRED_BACKEND_FACT_NOT_VERIFIED"),
    "wrong_tool_arguments": ("VERIFICATION", "TOOL_ARGUMENT_PRECONDITION_VIOLATED"),
    "tool_call_failure": ("VERIFICATION", "REQUIRED_BACKEND_FACT_NOT_VERIFIED"),
    "user_claim_overtrusted": ("DECISION", "USER_CLAIM_USED_WITHOUT_AUTHORITY_CHECK"),
    "authoritative_conflict": ("DECISION", "USER_CLAIM_USED_WITHOUT_AUTHORITY_CHECK"),
    "wrong_final_action": ("DECISION", "FINAL_ACTION_NOT_ALLOWED"),
    "action_execution_failure": ("ACTION", "EXPECTED_OUTCOME_NOT_REACHED"),
    "action_failure": ("ACTION", "EXPECTED_OUTCOME_NOT_REACHED"),
    "goal_not_fulfilled": ("ACTION", "EXPECTED_OUTCOME_NOT_REACHED"),
    "claimed_action_not_executed": ("CONFIRMATION", "ACTION_CLAIM_WITHOUT_SUCCESSFUL_EXECUTION"),
    "tool_loop_limit": ("RECOVERY", "FAILED_ACTION_NOT_RECOVERED"),
}

_STAGE_PRIORITY = {
    "VERIFICATION": 0,
    "DECISION": 1,
    "ACTION": 2,
    "CONFIRMATION": 3,
    "RECOVERY": 4,
    "TERMINATION": 5,
    "INPUT_COLLECTION": 6,
    "UNKNOWN": 7,
}

_TRIGGER_BY_ERROR = {
    "canonical_path_divergence": "SOP_PATH_DIVERGENCE",
    "wrong_tool_arguments": "INVALID_TOOL_ARGUMENT",
    "tool_call_failure": "TOOL_CALL_FAILURE",
    "wrong_tool_selection": "WRONG_TOOL_SELECTED",
    "missed_backend_verification": "REQUIRED_VERIFICATION_SKIPPED",
    "user_claim_overtrusted": "USER_CLAIM_CONFLICT",
    "authoritative_conflict": "USER_CLAIM_CONFLICT",
    "wrong_final_action": "WRONG_FINAL_ACTION",
    "action_execution_failure": "ACTION_FAILURE",
    "action_failure": "ACTION_FAILURE",
    "goal_not_fulfilled": "GOAL_UNMET",
    "claimed_action_not_executed": "FALSE_COMPLETION_CLAIM",
    "tool_loop_limit": "RECOVERY_LOOP_EXHAUSTED",
}

_DECISION_BY_ERROR = {
    "canonical_path_divergence": "WRONG_POLICY_PATH",
    "wrong_tool_arguments": "INVALID_QUERY_ARGUMENT",
    "tool_call_failure": "FAILED_TOOL_CALL",
    "wrong_tool_selection": "WRONG_TOOL",
    "missed_backend_verification": "ACTION_WITHOUT_REQUIRED_VERIFICATION",
    "user_claim_overtrusted": "TRUSTED_UNVERIFIED_USER_CLAIM",
    "authoritative_conflict": "TRUSTED_UNVERIFIED_USER_CLAIM",
    "wrong_final_action": "WRONG_ACTION",
    "action_execution_failure": "ACTION_ATTEMPT_FAILED",
    "action_failure": "ACTION_ATTEMPT_FAILED",
    "goal_not_fulfilled": "GOAL_NOT_COMPLETED",
    "claimed_action_not_executed": "CLAIMED_ACTION_WITHOUT_EXECUTION",
    "tool_loop_limit": "RECOVERY_NOT_COMPLETED",
}


@dataclass(frozen=True)
class FailureAttribution:
    """Stable primary causal attribution, separate from episode occurrence data."""

    sop_node: Optional[str]
    path_step_index: Optional[int]
    failure_stage: str
    violated_invariant: str
    primary_error: str
    attribution_reason: str
    confidence: str
    trigger_class: str = "UNCLASSIFIED_TRIGGER"
    service_decision_class: str = "UNCLASSIFIED_DECISION"

    def __post_init__(self):
        if self.failure_stage not in FAILURE_STAGES:
            raise ValueError(f"unsupported failure stage: {self.failure_stage}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FailureAttribution":
        return cls(**{key: value for key, value in data.items() if key in cls.__dataclass_fields__})


def _path(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, (list, tuple)) else []


def _canonical_path(value: Iterable[str], expected: bool = False) -> list[str]:
    path = [item for item in _path(value) if item not in {"start", "end"}]
    # Legacy PathList canonical paths omit action_* nodes. Preserve the scorer's
    # canonical representation when locating first divergence.
    if expected and not any(item.startswith("action_") for item in path):
        return path
    return path


def _diagnostics(report: Any) -> dict[str, Any]:
    details = getattr(report, "details", {}) or {}
    return details.get("diagnostics", {}) if isinstance(details, dict) else {}


def _attribution_for_error(
    error: str, node: Optional[str], index: Optional[int], reason: str, confidence: str
) -> FailureAttribution:
    stage, invariant = ERROR_STAGE_MAP.get(
        error, ("UNKNOWN", "LEGACY_UNSPECIFIED" if error else "EXPECTED_OUTCOME_NOT_REACHED")
    )
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", error).strip("_").upper() or "UNCLASSIFIED"
    return FailureAttribution(
        sop_node=node,
        path_step_index=index,
        failure_stage=stage,
        violated_invariant=invariant,
        primary_error=error,
        attribution_reason=reason,
        confidence=confidence,
        trigger_class=_TRIGGER_BY_ERROR.get(error, f"ERROR_{normalized}"),
        service_decision_class=_DECISION_BY_ERROR.get(error, f"DECISION_{normalized}"),
    )


def _failure_attribution(
    *,
    errors: set[str],
    expected: list[str],
    predicted: list[str],
    node: Optional[str],
    index: Optional[int],
    failed: bool,
    protocol_invalid: bool = False,
    diagnostic_node: Optional[str] = None,
    diagnostic_index: Optional[int] = None,
) -> FailureAttribution:
    if protocol_invalid and not predicted:
        return FailureAttribution(
            None, None, "UNKNOWN", "LEGACY_UNSPECIFIED", "", "protocol-invalid episode", "unknown"
        )

    # First canonical divergence is a deterministic decision-stage cause;
    # verification errors take priority because they can precede the decision.
    verification_errors = [
        error for error in errors
        if ERROR_STAGE_MAP.get(error, ("UNKNOWN", ""))[0] == "VERIFICATION"
    ]
    if verification_errors:
        primary = sorted(verification_errors)[0]
        return _attribution_for_error(
            primary, node, index, "authoritative verification failure", "deterministic"
        )

    divergence = False
    divergence_index = None
    divergence_node = None
    if expected and predicted:
        for path_index, (expected_node, predicted_node) in enumerate(zip(expected, predicted)):
            if expected_node != predicted_node:
                divergence = True
                divergence_index = path_index
                divergence_node = expected_node
                break
        if not divergence and len(expected) != len(predicted):
            divergence = True
            divergence_index = min(len(expected), len(predicted))
            divergence_node = expected[divergence_index] if divergence_index < len(expected) else None
    elif isinstance(diagnostic_index, int) or isinstance(diagnostic_node, str):
        divergence = True
        divergence_index = diagnostic_index
        divergence_node = diagnostic_node

    if divergence:
        return _attribution_for_error(
            "canonical_path_divergence",
            divergence_node or node,
            divergence_index if isinstance(divergence_index, int) else index,
            "first canonical policy-path divergence",
            "deterministic",
        )

    mapped = [error for error in errors if error in ERROR_STAGE_MAP]
    if mapped:
        primary = min(
            mapped,
            key=lambda error: (
                _STAGE_PRIORITY[ERROR_STAGE_MAP[error][0]],
                error,
            ),
        )
        return _attribution_for_error(
            primary,
            node,
            index,
            "earliest mapped evaluator error by deterministic stage priority",
            "deterministic",
        )

    if failed:
        return _attribution_for_error(
            "goal_not_fulfilled", node, index,
            "terminal goal failure without a more specific evaluator error", "heuristic",
        )
    return FailureAttribution(
        node, index, "UNKNOWN", "LEGACY_UNSPECIFIED", "", "no failure to attribute", "unknown"
    )


def infer_failure_attribution(
    report: Any,
    simulation: Any = None,
    path_config: Optional[dict[str, Any]] = None,
) -> FailureAttribution:
    """Infer the primary failure's SOP location and execution lifecycle stage."""
    details = getattr(report, "details", {}) or {}
    diagnostics = _diagnostics(report)
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

    expected = _canonical_path(expected, expected=True)
    predicted = _canonical_path(predicted, expected=False)
    if expected and not any(item.startswith("action_") for item in expected):
        predicted = [item for item in predicted if not item.startswith("action_")]

    errors = set(getattr(report, "error_categories", []) or [])
    protocol_invalid = (
        "json_parse_failed" in errors
        or "protocol_failure" in errors
        or bool(diagnostics.get("protocol_failure", False))
    )
    failed = not bool(getattr(report, "task_success", False))
    failed = failed or float(getattr(report, "required_verification_score", 1.0)) < 1.0
    failed = failed or float(getattr(report, "policy_compliance_score", 1.0)) < 1.0
    failed = failed or float(getattr(report, "action_execution_score", 1.0)) < 1.0
    failed = failed or float(getattr(report, "goal_fulfillment", 1.0)) < 1.0
    node = None
    index = None
    if expected and failed:
        node, index = expected[-1], len(expected) - 1
    elif isinstance(diagnostics.get("first_divergence_node"), str):
        node = diagnostics["first_divergence_node"]
        value = diagnostics.get("first_divergence_step")
        index = value if isinstance(value, int) else None
    return _failure_attribution(
        errors=errors,
        expected=expected,
        predicted=predicted,
        node=node,
        index=index,
        failed=failed,
        protocol_invalid=protocol_invalid,
        diagnostic_node=diagnostics.get("first_divergence_node"),
        diagnostic_index=diagnostics.get("first_divergence_step"),
    )


def infer_failure_attribution_for_episode(episode: Any) -> FailureAttribution:
    """Use persisted attribution when present, otherwise derive it from episode data."""
    metadata = getattr(episode, "metadata", {}) or {}
    saved = metadata.get("failure_attribution")
    if isinstance(saved, dict):
        try:
            return FailureAttribution.from_dict(saved)
        except (TypeError, ValueError):
            pass
    errors = set(getattr(episode, "error_types", []) or [])
    invalid = bool(
        getattr(episode, "is_evaluation_invalid", lambda: False)()
    )
    node = getattr(episode, "sop_node", None)
    index = getattr(episode, "path_step_index", None)
    failed = not bool(getattr(episode, "task_success", False))
    return _failure_attribution(
        errors=errors,
        expected=[],
        predicted=[],
        node=node,
        index=index,
        failed=failed,
        protocol_invalid=invalid,
    )


def infer_failure_location(
    report: Any,
    simulation: Any = None,
    path_config: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Compatibility view exposing the historical node/index/reason fields."""
    attribution = infer_failure_attribution(report, simulation, path_config)
    return {
        "path_step_index": attribution.path_step_index,
        "sop_node": attribution.sop_node,
        "reason": attribution.attribution_reason or None,
        "failure_stage": attribution.failure_stage,
        "violated_invariant": attribution.violated_invariant,
        "primary_error": attribution.primary_error,
        "confidence": attribution.confidence,
        "trigger_class": attribution.trigger_class,
        "service_decision_class": attribution.service_decision_class,
    }
