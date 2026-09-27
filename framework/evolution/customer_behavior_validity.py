"""Deterministic hard checks for Customer behavior attribution.

This module deliberately avoids semantic/LLM judging. It only validates
explicit contracts, immutable case state, and exact unobserved identifiers or
backend values. A failed check makes an episode non-attributable; it does not
rewrite its business score or discard its diagnostic trace.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import re
from typing import Any

from ..backend.types import CaseSpec
from ..core.customer_contract import get_customer_opening_contract
from .customer_policy import CustomerPolicyCompiler


@dataclass
class CustomerBehaviorAssessment:
    valid: bool
    hard_violation: bool
    reasons: list[str] = field(default_factory=list)
    checks: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_INTERNAL_FIELD = re.compile(
    r"\b(?:expected[_ ](?:path|action|outcome)|gold[_ -]?path|"
    r"backend[_ ]record|system[_ ]variables|evaluator[_ -]only)\b",
    re.IGNORECASE,
)
_SERVICE_VOICE = re.compile(
    r"(?:^|[。！？!?.\n])\s*(?:您好[,，：:]?\s*)?(?:请您提供|请提供).{0,16}"
    r"(?:我(?:来|会|可以)?为您|我帮您)(?:查询|办理|提交|处理)|"
    r"(?:^|[。！？!?.\n])\s*(?:我已|我已经|已)为您(?:办理|提交|退款|申请|处理)",
    re.IGNORECASE,
)
_IDENTIFIER_KEY = re.compile(
    r"(?:^|_)(?:order|customer|record|booking|ticket|account|delivery|"
    r"course|repair|claim|payment|user)?(?:_)?(?:id|identifier|number|no)(?:$|_)",
    re.IGNORECASE,
)
_ID_KEYS = {"order_id", "customer_id", "record_id", "booking_id", "ticket_id", "account_id"}
_KNOWS_FLAG = {
    "order_id": "knows_order_id",
    "customer_id": "knows_customer_id",
    "record_id": "knows_record_id",
    "booking_id": "knows_booking_id",
    "ticket_id": "knows_ticket_id",
    "account_id": "knows_account_id",
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
        return CaseSpec(**{key: value[key] for key in required | ({"metadata"} if "metadata" in value else set())})
    except (TypeError, ValueError):
        return None


def _scalar_strings(value: Any, path: str = "") -> list[tuple[str, str]]:
    values: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            values.extend(_scalar_strings(item, child))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            values.extend(_scalar_strings(item, f"{path}[{index}]"))
    elif isinstance(value, (str, int, float)) and not isinstance(value, bool):
        text = str(value).strip()
        if text:
            values.append((path, text))
    return values


def _known_values(case: CaseSpec, simulation: Any) -> set[str]:
    known: set[str] = set()
    knowledge = case.user_knowledge or {}
    for key, value in knowledge.items():
        if key in _ID_KEYS and not knowledge.get(_KNOWS_FLAG.get(key, ""), False):
            continue
        known.update(text.lower() for _, text in _scalar_strings(value))
    known.update(text.lower() for _, text in _scalar_strings(case.initial_observation or {}))
    for event in getattr(simulation, "backend_events", []) or []:
        # Events are the public Customer projection. State snapshots are not
        # consulted here; only the tool/action result can authorize disclosure.
        result = event.get("result") if isinstance(event, dict) else None
        known.update(text.lower() for _, text in _scalar_strings(result or {}))
    return known


def _hidden_candidates(case: CaseSpec) -> list[tuple[str, str, bool]]:
    candidates: list[tuple[str, str, bool]] = []
    metadata_values = (case.metadata or {}).get("backend_system_variables", {})
    for path, text in _scalar_strings(metadata_values, "backend_system_variables"):
        candidates.append((path.rsplit(".", 1)[-1], text, False))
    private_variables = ((case.backend_record or {}).get("private_state") or {}).get("system_variables", {})
    for path, text in _scalar_strings(private_variables, "private_state.system_variables"):
        candidates.append((path.rsplit(".", 1)[-1], text, False))

    # Collect authoritative identifiers from the record, but not arbitrary
    # business text that may include public labels or ordinary vocabulary.
    for path, text in _scalar_strings(case.backend_record or {}):
        leaf = path.rsplit(".", 1)[-1].split("[", 1)[0]
        normalized = re.sub(r"[^a-z0-9]+", "_", leaf.lower()).strip("_")
        if normalized in _ID_KEYS or _IDENTIFIER_KEY.search(normalized):
            candidates.append((leaf, text, True))

    deduped = {}
    for field_name, value, is_identifier in candidates:
        if value and value.lower() not in {"none", "null", "unknown"}:
            deduped[(field_name, value.lower())] = (field_name, value, is_identifier)
    return list(deduped.values())


def _value_mentioned(message: str, value: str, identifier: bool) -> bool:
    normalized = value.strip()
    if not normalized:
        return False
    # Very short enum values are too ambiguous for a deterministic exact-text
    # leakage test. Identifiers are checked even when short.
    if not identifier and len(normalized) < 3:
        return False
    if identifier or re.search(r"[0-9_-]", normalized):
        return normalized.casefold() in message.casefold()
    return re.search(rf"(?<!\w){re.escape(normalized)}(?!\w)", message, re.IGNORECASE) is not None


def assess_customer_behavior(simulation: Any, customer_policy: Any) -> CustomerBehaviorAssessment:
    """Assess hard Customer-side contracts from a completed simulation trace."""
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

    if case is None:
        checks["case_spec_available"] = False
    else:
        checks["case_spec_available"] = True
        try:
            CustomerPolicyCompiler().compile(customer_policy, case)
            checks["customer_policy_immutable_case"] = True
        except Exception:
            checks["customer_policy_immutable_case"] = False
            reasons.append("customer_policy_invalid")

        opening_contract = get_customer_opening_contract(case)
        opening_ok = True
        if opening_contract:
            opening = messages[0] if messages else ""
            opening_ok = all(str(item["value"]).casefold() in opening.casefold() for item in opening_contract)
        checks["mandatory_opening_disclosure"] = opening_ok
        if not opening_ok:
            reasons.append("mandatory_opening_disclosure_missing")

        environment_state = getattr(simulation, "user_environment_state", {}) or {}
        if hasattr(environment_state, "to_dict"):
            environment_state = environment_state.to_dict()
        observed_goal = environment_state.get("goal") if isinstance(environment_state, dict) else None
        goal_unchanged = observed_goal is None or observed_goal == case.user_goal
        checks["case_goal_unchanged"] = goal_unchanged
        if not goal_unchanged:
            reasons.append("customer_goal_mutated")

        transcript = "\n".join(messages)
        if _INTERNAL_FIELD.search(transcript):
            reasons.append("customer_evaluator_field_leakage")
        allowed = _known_values(case, simulation)
        leaked_fields = set()
        unknown_identifiers = set()
        for field_name, value, is_identifier in _hidden_candidates(case):
            if value.lower() in allowed or not any(_value_mentioned(message, value, is_identifier) for message in messages):
                continue
            if is_identifier:
                unknown_identifiers.add(field_name)
            else:
                leaked_fields.add(field_name)
        for field_name in sorted(leaked_fields):
            reasons.append(f"customer_hidden_value_leakage:{field_name}")
        for field_name in sorted(unknown_identifiers):
            reasons.append(f"customer_unknown_identifier_leakage:{field_name}")

    role_drift = any(_SERVICE_VOICE.search(message) for message in messages)
    checks["customer_not_speaking_as_service"] = not role_drift
    if role_drift:
        reasons.append("customer_role_drift")
    policy_text = json.dumps(
        customer_policy.to_dict() if hasattr(customer_policy, "to_dict") else customer_policy,
        ensure_ascii=False,
    )
    checks["role_drift_suspected"] = bool(_SERVICE_VOICE.search(policy_text))
    reasons = list(dict.fromkeys(reasons))
    return CustomerBehaviorAssessment(
        valid=not reasons,
        hard_violation=bool(reasons),
        reasons=reasons,
        checks=checks,
    )
