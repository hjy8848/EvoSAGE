"""Serializable schemas used by the co-evolution layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import re
from typing import Any, Dict, Iterable, List, Optional


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PolicyValidationError(ValueError):
    """Raised when an evolution candidate violates a hard validity gate."""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass
class CustomerPolicy:
    policy_id: str = "customer_policy_c0"
    generation: int = 0
    parent_policy_ids: List[str] = field(default_factory=list)
    name: str = "baseline"
    description: str = "Truthful customer interaction baseline."
    strategy_tags: List[str] = field(default_factory=lambda: ["truthful", "cooperative"])
    disclosure_strategy: str = "provide requested order information when asked"
    claim_strategy: str = "state the customer's belief without changing the underlying case facts"
    pressure_strategy: str = "remain polite and pursue the original request"
    contradiction_strategy: str = "ask for an explanation when a verified result conflicts with the claim"
    timing_strategy: str = "disclose information when the service workflow requires it"
    response_to_verification: str = "acknowledge authoritative results and continue the same goal"
    response_to_rejection: str = "ask for the reason and a legitimate next step"
    escalation_strategy: str = "escalate only after a failed or missing resolution"
    optional_target_failure_modes: List[str] = field(default_factory=list)
    optional_target_sop_node: Optional[str] = None
    mutation_rationale: str = "baseline"
    source_failure_ids: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)
    model_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CustomerPolicy":
        fields = {key: value for key, value in data.items() if key in cls.__dataclass_fields__}
        return cls(**fields)

    def runtime_guidance(self) -> str:
        """Compile only reusable interaction strategy, never a case answer."""
        return (
            "\n【客户交互策略】\n"
            f"披露：{self.disclosure_strategy}\n"
            f"陈述：{self.claim_strategy}\n"
            f"压力：{self.pressure_strategy}\n"
            f"矛盾处理：{self.contradiction_strategy}\n"
            f"时机：{self.timing_strategy}\n"
            f"核验后：{self.response_to_verification}\n"
            f"拒绝后：{self.response_to_rejection}\n"
            f"升级：{self.escalation_strategy}\n"
            "策略只能改变表达和交互方式，不得改变案例目标、身份、订单或后台事实。"
        )

    def validate_for_case(self, case_spec: Any) -> None:
        """Hard-gate sample leakage and immutable-case mutation."""
        text = _json(self.to_dict()).lower()
        forbidden = ["expected_path", "expected_action", "task_success", "evaluator"]
        if any(token in text for token in forbidden) or re.search(r"\bpath\s*\d+\b|gold[_ -]?path", text):
            raise PolicyValidationError("customer policy references evaluator-only information")
        if case_spec is None:
            return
        immutable_values = [
            case_spec.case_id,
            case_spec.scenario,
            case_spec.user_goal.get("desired_action"),
            case_spec.user_knowledge.get("order_id"),
            case_spec.user_knowledge.get("customer_id"),
            case_spec.user_knowledge.get("record_id"),
        ]
        for value in immutable_values:
            if value and str(value).lower() in text:
                raise PolicyValidationError("customer policy contains sample-specific identity or answer")
        # A customer may only claim facts that are in its legitimate knowledge
        # or use generic language about an unobserved field.
        hidden = dict(case_spec.backend_record.get("private_state", {}).get("system_variables", {}))
        hidden.update(case_spec.metadata.get("backend_system_variables", {}))
        # Ecommerce stores authoritative fields in nested order/customer
        # records rather than a generic private_state object.  Collect only
        # scalar values; the validator still allows a value when it is part
        # of the customer's explicit knowledge projection.
        def collect_scalars(value):
            if isinstance(value, dict):
                for child in value.values():
                    yield from collect_scalars(child)
            elif isinstance(value, (str, int, float, bool)):
                yield value
        hidden_values = list(hidden.values()) + list(collect_scalars(case_spec.backend_record))
        known = {str(value).lower() for value in case_spec.user_knowledge.values() if value is not None}
        for value in hidden_values:
            if value is not None and str(value).lower() in text and str(value).lower() not in known:
                raise PolicyValidationError("customer policy embeds an unobserved backend value")

    def assert_immutable_case(self, case_spec: Any, original: Any) -> None:
        """Verify that a policy application did not mutate the benchmark case."""
        if _json(case_spec.to_dict()) != _json(original.to_dict()):
            raise PolicyValidationError("CustomerPolicy attempted to mutate immutable CaseSpec")


@dataclass
class ServiceRule:
    rule_id: str
    category: str
    text: str
    rationale: str = ""
    source_failure_signatures: List[str] = field(default_factory=list)
    generation_added: int = 0
    active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ServiceRule":
        return cls(**{key: value for key, value in data.items() if key in cls.__dataclass_fields__})


@dataclass
class ServicePolicy:
    policy_id: str = "service_policy_s0"
    generation: int = 0
    parent_policy_id: Optional[str] = None
    rules: List[ServiceRule] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)
    mutation_history: List[Dict[str, Any]] = field(default_factory=list)
    model_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["rules"] = [rule.to_dict() for rule in self.rules]
        return value

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ServicePolicy":
        value = dict(data)
        value["rules"] = [ServiceRule.from_dict(item) for item in value.get("rules", [])]
        return cls(**{key: value[key] for key in cls.__dataclass_fields__ if key in value})

    def overlay_prompt(self) -> str:
        active = [rule.text for rule in self.rules if rule.active]
        if not active:
            return ""
        return "\n【当前服务策略补丁】\n" + "\n".join(f"- {item}" for item in active)

    def clone_with(self, rules: Iterable[ServiceRule], generation: Optional[int] = None) -> "ServicePolicy":
        return ServicePolicy(
            policy_id=f"service_policy_s{self.generation + 1 if generation is None else generation}",
            generation=self.generation + 1 if generation is None else generation,
            parent_policy_id=self.policy_id,
            rules=list(rules),
            mutation_history=list(self.mutation_history),
            model_metadata=dict(self.model_metadata),
        )


@dataclass
class ServicePatch:
    patch_id: str
    patch_type: str
    rules: List[ServiceRule] = field(default_factory=list)
    rationale: str = ""
    evidence_count: int = 0
    source_failure_ids: List[str] = field(default_factory=list)
    rejected_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["rules"] = [rule.to_dict() for rule in self.rules]
        return value

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ServicePatch":
        value = dict(data)
        value["rules"] = [ServiceRule.from_dict(item) for item in value.get("rules", [])]
        return cls(**{key: value[key] for key in cls.__dataclass_fields__ if key in value})


@dataclass
class FailureSignature:
    signature_id: str
    scenario: str
    sop_node: Optional[str] = None
    path_step_index: Optional[int] = None
    first_divergence_node: Optional[str] = None
    error_types: List[str] = field(default_factory=list)
    required_verification_score: float = 0.0
    policy_score: float = 0.0
    action_execution_score: float = 0.0
    goal_fulfillment_score: float = 0.0
    predicted_action: str = ""
    executed_action: str = ""
    tool_sequence_summary: List[str] = field(default_factory=list)
    claimed_action_not_executed: bool = False
    user_claim_backend_conflict: bool = False
    termination_reason: str = ""

    @classmethod
    def from_episode(cls, episode: "EpisodeResult") -> "FailureSignature":
        payload = {
            "scenario": episode.scenario,
            "errors": sorted(episode.error_types),
            "predicted_action": episode.predicted_action,
            "executed_action": episode.executed_action,
            "tools": episode.tool_sequence_summary,
            "termination": episode.termination_reason,
        }
        signature_id = "failure_" + hashlib.sha256(_json(payload).encode()).hexdigest()[:12]
        return cls(
            signature_id=signature_id,
            scenario=episode.scenario,
            sop_node=episode.sop_node,
            path_step_index=episode.path_step_index,
            error_types=sorted(set(episode.error_types)),
            required_verification_score=getattr(episode, "required_verification_score", episode.verification_score),
            policy_score=episode.policy_score,
            action_execution_score=episode.action_execution_score,
            goal_fulfillment_score=episode.goal_fulfillment_score,
            predicted_action=episode.predicted_action,
            executed_action=episode.executed_action,
            tool_sequence_summary=list(episode.tool_sequence_summary),
            claimed_action_not_executed="claimed_action_not_executed" in episode.error_types,
            user_claim_backend_conflict="authoritative_conflict" in episode.error_types,
            termination_reason=episode.termination_reason,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FailureSignature":
        return cls(**{key: value for key, value in data.items() if key in cls.__dataclass_fields__})


@dataclass
class EpisodeResult:
    episode_id: str
    scenario: str
    case_id: str
    customer_policy_id: str
    service_policy_id: str
    split: str
    generation: int
    task_success: bool
    execution_score: float
    sage_style_score: float = 0.0
    verification_score: float = 0.0
    policy_score: float = 0.0
    action_execution_score: float = 0.0
    goal_fulfillment_score: float = 0.0
    error_types: List[str] = field(default_factory=list)
    predicted_action: str = ""
    executed_action: str = ""
    tool_sequence_summary: List[str] = field(default_factory=list)
    termination_reason: str = ""
    sop_node: Optional[str] = None
    path_step_index: Optional[int] = None
    dialogue: List[Dict[str, Any]] = field(default_factory=list)
    trace_ref: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["failure_signature"] = FailureSignature.from_episode(self).to_dict()
        return value

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpisodeResult":
        value = {key: item for key, item in data.items() if key in cls.__dataclass_fields__}
        return cls(**value)


@dataclass
class DefenseRecord:
    defense_id: str
    service_policy_id: str
    generation_added: int
    rule_ids: List[str]
    addresses_failure_signatures: List[str]
    validation_delta: Dict[str, float]
    normal_user_delta: Dict[str, float]
    adversarial_delta: Dict[str, float]
    regression_cases: List[str] = field(default_factory=list)
    active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DefenseRecord":
        return cls(**{key: value for key, value in data.items() if key in cls.__dataclass_fields__})


def reject_forbidden_service_text(text: str) -> Optional[str]:
    """Return a leakage reason, or None when a service patch is general."""
    lowered = text.lower()
    if re.search(r"case-[a-z0-9-]+|ord-[a-z0-9-]+|cus-[a-z0-9-]+", lowered):
        return "contains concrete benchmark identity"
    if re.search(r"\bpath\s*\d+\b|expected_path|gold[_ -]?path|task[_ -]?success", lowered):
        return "contains benchmark-specific path or evaluator instruction"
    if "case_id" in lowered or "order_id" in lowered and "verify" not in lowered:
        return "contains sample-specific mapping language"
    if re.search(r"shippingstatus\s*[:=]|creditlevel\s*[:=]|refundeligibility\s*[:=]", lowered):
        return "copies a hidden backend field assignment"
    return None
