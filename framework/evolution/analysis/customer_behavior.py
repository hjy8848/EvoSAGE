"""Customer-side integrity checks, independent of business-claim truth."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ...backend.types import CaseSpec
from ..customer.policy import AdversaryPolicy, requests_benchmark_manipulation
from ..legacy.customer_policy import CustomerPolicyCompiler


@dataclass
class CustomerBehaviorAssessment:
    valid: bool
    hard_violation: bool
    # Natural-language goal preservation is prompt-guided, not inferred by
    # brittle lexical rules. None means it was deliberately not adjudicated.
    goal_preserving: bool | None = None
    reasons: list[str] = field(default_factory=list)
    checks: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_CUSTOMER_GOAL_KEYS = {
    "type", "intent", "user_intent", "goal", "customer_goal", "request",
}


def _case_spec(value: Any) -> CaseSpec | None:
    if isinstance(value, CaseSpec):
        return value
    if not isinstance(value, dict):
        return None
    required = {
        "case_id", "scenario", "backend_record", "user_goal", "user_knowledge",
        "user_policy", "initial_observation", "expected_outcome",
    }
    if not required.issubset(value):
        return None
    try:
        return CaseSpec(**value)
    except (TypeError, ValueError):
        return None


def _customer_goal(goal: Any) -> Any:
    if not isinstance(goal, dict):
        return goal
    return {key: value for key, value in goal.items() if str(key).lower() in _CUSTOMER_GOAL_KEYS}


def assess_customer_behavior(simulation: Any, customer_policy: Any) -> CustomerBehaviorAssessment:
    """Reject protocol/integrity violations, never false Customer business claims."""
    reasons: list[str] = []
    checks: dict[str, Any] = {}
    case = _case_spec(getattr(simulation, "case_spec", None))
    turns = getattr(simulation, "turns", []) or []
    messages = [str(getattr(turn, "user_message", "") or "") for turn in turns]

    nonempty = bool(messages) and all(message.strip() for message in messages)
    checks["customer_messages_nonempty"] = nonempty
    if not messages and (case is not None or simulation.__class__.__name__ == "SimulationResult"):
        reasons.append("customer_messages_missing")
    elif messages and not nonempty:
        reasons.append("customer_message_empty")

    checks["case_spec_available"] = case is not None
    if case is not None:
        try:
            if isinstance(customer_policy, AdversaryPolicy):
                customer_policy.validate_integrity()
            else:
                CustomerPolicyCompiler().compile(customer_policy, case)
            checks["adversary_strategy_integrity"] = True
        except Exception:
            checks["adversary_strategy_integrity"] = False
            reasons.append("customer_strategy_invalid")

        observed_state = getattr(simulation, "user_environment_state", {}) or {}
        if hasattr(observed_state, "to_dict"):
            observed_state = observed_state.to_dict()
        observed_goal = observed_state.get("goal") if isinstance(observed_state, dict) else None
        expected_goal = _customer_goal(case.user_goal)
        goal_unchanged = observed_goal is None or observed_goal == expected_goal
        checks["customer_goal_unchanged"] = goal_unchanged
        if not goal_unchanged:
            reasons.append("customer_task_mutation")
    # Goal preservation is instruction-followed, not post-hoc classified from
    # natural-language keywords. Keep the assigned goal in both Evolver and
    # Customer prompts, but do not use utterance text to invalidate an episode.
    checks["goal_preservation_checked"] = False

    transcript = "\n".join(messages)
    harness_tampering = requests_benchmark_manipulation(transcript)
    checks["no_harness_manipulation"] = not harness_tampering
    if harness_tampering:
        reasons.append("customer_harness_manipulation")

    # Deliberately visible for analysis: business claims are not compared with
    # CaseSpec/backend truth here. That comparison belongs to official scoring.
    checks["business_claim_truth_checked"] = False
    checks["false_or_contradictory_business_claims_are_valid"] = True
    reasons = list(dict.fromkeys(reasons))
    return CustomerBehaviorAssessment(
        valid=not reasons,
        hard_violation=bool(reasons),
        goal_preserving=None,
        reasons=reasons,
        checks=checks,
    )
