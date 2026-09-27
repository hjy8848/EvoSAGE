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
        tag_guidance = {
            "truthful": "只陈述自己确实知道的事实，不猜测后台字段。",
            "cooperative": "先配合必要核验，并在客服说明下一步后确认理解。",
            "withholding": "非必要信息在客服明确询问前暂不主动补充。",
            "pressure": "在首次拒绝或等待后提高紧迫感，但不改变业务目标。",
            "contradiction": "发现说明不一致时礼貌指出矛盾并要求重新核对。",
            "delayed_disclosure": "先确认客服需要的信息，再分阶段披露合法已知事实。",
            "authority_challenge": "对权威查询结果影响诉求的原因提出业务解释请求。",
            "delayed_contradiction": "在至少一轮核验后再提出与个人理解不同的说法。",
            "escalation": "只有处理失败或问题未解决时才请求转人工。",
            "paraphrase": "保持事实和目标不变，但使用不同自然句式表达。",
        }
        tag_lines = [tag_guidance[tag] for tag in self.strategy_tags if tag in tag_guidance]
        return (
            "\n【客户交互策略】\n"
            + "\n".join(f"- {line}" for line in tag_lines)
            + "\n"
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

        def contains_value(haystack: str, value: Any) -> bool:
            """Match a value as a token, not as an arbitrary substring.

            This prevents benign words such as ``workflow`` from matching a
            hidden value like ``Low`` and ``rejection`` from matching an
            action named ``Reject``.
            """
            if value is None:
                return False
            needle = str(value).strip().lower()
            if not needle:
                return False
            return re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", haystack, re.UNICODE) is not None

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
            if value and contains_value(text, value):
                raise PolicyValidationError("customer policy contains sample-specific identity or answer")
        # A customer may only claim facts that are in its legitimate knowledge
        # or use generic language about an unobserved field.
        hidden = dict(case_spec.backend_record.get("private_state", {}).get("system_variables", {}))
        hidden.update(case_spec.metadata.get("backend_system_variables", {}))
        # Do not scan every scalar in backend_record here.  That record also
        # contains semantic labels used to build the benchmark, such as
        # ``responsibility="User"``; rejecting the ordinary word "user" in a
        # reusable customer strategy would be a false-positive leakage gate.
        # Factory-created CaseSpecs explicitly publish authoritative hidden
        # variables through this metadata map, so validate only that contract.
        hidden_values = list(hidden.values())
        known = {str(value).lower() for value in case_spec.user_knowledge.values() if value is not None}
        for value in hidden_values:
            if contains_value(text, value) and str(value).lower() not in known:
                raise PolicyValidationError(
                    f"customer policy embeds an unobserved backend value: {value!r}"
                )

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
    """Legacy V1 signature payload retained only for artifact compatibility.

    New runtime code must use :class:`VulnerabilitySignature`; this V1 shape
    includes occurrence details and must never be deduplicated with V2 IDs.
    """

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
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("FailureSignature is the legacy V1 compatibility type")

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
        values = {key: value for key, value in data.items() if key in cls.__dataclass_fields__}
        values.setdefault("schema_version", 1)
        return cls(**values)


@dataclass
class VulnerabilitySignature:
    """Stable V2 identity for a vulnerability class, excluding occurrences."""

    scenario: str
    failure_type: str
    failure_stage: str
    sop_node: Optional[str]
    violated_invariant: str
    trigger_class: str
    service_decision_class: str
    consequence_class: str
    signature_id: str = ""
    schema_version: int = 2

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError("VulnerabilitySignature only represents schema version 2")
        if not self.signature_id:
            payload = self.canonical_payload()
            digest = hashlib.sha256(_json(payload).encode("utf-8")).hexdigest()[:16]
            self.signature_id = f"vuln_v2_{digest}"

    def canonical_payload(self) -> Dict[str, Any]:
        # This is the entire identity contract. Episode/case IDs, scores,
        # wording, exact tool sequence and termination are occurrence data.
        return {
            "schema_version": 2,
            "scenario": self.scenario,
            "failure_type": self.failure_type,
            "failure_stage": self.failure_stage,
            "sop_node": self.sop_node,
            "violated_invariant": self.violated_invariant,
            "trigger_class": self.trigger_class,
            "service_decision_class": self.service_decision_class,
            "consequence_class": self.consequence_class,
        }

    @classmethod
    def from_episode(
        cls, episode: "EpisodeResult", attribution: Any = None
    ) -> "VulnerabilitySignature":
        if episode.task_success or episode.is_evaluation_invalid():
            raise ValueError("VulnerabilitySignature requires a valid failed episode")
        if attribution is None:
            from .attribution import infer_failure_attribution_for_episode

            attribution = infer_failure_attribution_for_episode(episode)
        return cls(
            scenario=episode.scenario,
            failure_type=attribution.primary_error or "unknown_failure",
            failure_stage=attribution.failure_stage,
            sop_node=attribution.sop_node or episode.sop_node,
            violated_invariant=attribution.violated_invariant,
            trigger_class=attribution.trigger_class,
            service_decision_class=attribution.service_decision_class,
            consequence_class=(
                "GOAL_NOT_FULFILLED" if not episode.task_success
                else "PROCESS_FAILURE_WITH_GOAL_MET"
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VulnerabilitySignature":
        version = int(data.get("schema_version", 0) or 0)
        if version != 2:
            raise ValueError(
                "V1/unspecified signatures must be loaded with signature_from_dict"
            )
        values = {
            key: value for key, value in data.items()
            if key in cls.__dataclass_fields__
        }
        return cls(**values)


@dataclass
class FailureOccurrence:
    """V2 record of one observed episode that instantiated a vulnerability."""

    occurrence_id: str
    signature_id: str
    episode_id: str
    case_id: str
    customer_policy_id: str
    service_policy_id: str
    generation: int
    split: str
    path_step_index: Optional[int]
    error_types: List[str]
    predicted_action: str
    executed_action: str
    tool_sequence_summary: List[str]
    termination_reason: str
    verification_score: float
    policy_score: float
    action_execution_score: float
    goal_fulfillment_score: float
    failure_type: str = "unknown_failure"
    failure_stage: str = "UNKNOWN"
    violated_invariant: str = "LEGACY_UNSPECIFIED"
    trigger_class: str = "UNCLASSIFIED_TRIGGER"
    service_decision_class: str = "UNCLASSIFIED_DECISION"
    trace_ref: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_primary: bool = True
    schema_version: int = 2

    @classmethod
    def from_episode(
        cls, episode: "EpisodeResult", signature: VulnerabilitySignature
    ) -> "FailureOccurrence":
        if episode.task_success or episode.is_evaluation_invalid():
            raise ValueError("FailureOccurrence requires a valid failed episode")
        if signature.schema_version != 2:
            raise ValueError("FailureOccurrence requires a V2 vulnerability signature")
        from .attribution import infer_failure_attribution_for_episode

        attribution = infer_failure_attribution_for_episode(episode)
        identity = {
            "schema_version": 2,
            "signature_id": signature.signature_id,
            "episode_id": episode.episode_id,
            "case_id": episode.case_id,
            "customer_policy_id": episode.customer_policy_id,
            "service_policy_id": episode.service_policy_id,
            "generation": episode.generation,
            "split": episode.split,
        }
        digest = hashlib.sha256(_json(identity).encode("utf-8")).hexdigest()[:16]
        return cls(
            occurrence_id=f"occ_v2_{digest}",
            signature_id=signature.signature_id,
            episode_id=episode.episode_id,
            case_id=episode.case_id,
            customer_policy_id=episode.customer_policy_id,
            service_policy_id=episode.service_policy_id,
            generation=episode.generation,
            split=episode.split,
            path_step_index=episode.path_step_index,
            error_types=sorted(set(episode.error_types or [])),
            predicted_action=episode.predicted_action,
            executed_action=episode.executed_action,
            tool_sequence_summary=list(episode.tool_sequence_summary),
            termination_reason=episode.termination_reason,
            verification_score=float(episode.verification_score),
            policy_score=float(episode.policy_score),
            action_execution_score=float(episode.action_execution_score),
            goal_fulfillment_score=float(episode.goal_fulfillment_score),
            failure_type=signature.failure_type,
            failure_stage=signature.failure_stage,
            violated_invariant=signature.violated_invariant,
            trigger_class=signature.trigger_class,
            service_decision_class=signature.service_decision_class,
            trace_ref=episode.trace_ref,
            metadata={
                **dict(episode.metadata or {}),
                "attribution_reason": attribution.attribution_reason,
                "attribution_confidence": attribution.confidence,
            },
        )

    @property
    def required_verification_score(self) -> float:
        """Compatibility name consumed by existing service-patch prompts."""
        return self.verification_score

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FailureOccurrence":
        values = {key: value for key, value in data.items() if key in cls.__dataclass_fields__}
        version = int(values.get("schema_version", 0) or 0)
        if version != 2:
            raise ValueError("FailureOccurrence only supports schema version 2")
        return cls(**values)


def signature_from_dict(data: Dict[str, Any]) -> VulnerabilitySignature | FailureSignature:
    """Load legacy V1 artifacts without allowing them into V2 identity sets."""
    raw_version = data.get("schema_version", 1)
    version = int(1 if raw_version is None else raw_version)
    if version == 1:
        return FailureSignature.from_dict(data)
    if version == 2:
        return VulnerabilitySignature.from_dict(data)
    raise ValueError(f"unsupported failure signature schema_version: {version}")


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
    # ``valid`` means the episode produced an evaluable Agent decision.  An
    # invalid episode is kept for diagnostics, but must never be interpreted
    # as a substantive task failure by a policy gate.
    evaluation_status: str = "valid"
    invalid_reason: Optional[str] = None
    _signature_artifact_version: Optional[int] = field(default=None, repr=False, compare=False)
    _signature_artifact_payload: Optional[Dict[str, Any]] = field(default=None, repr=False, compare=False)
    _occurrence_artifact_payload: Optional[Dict[str, Any]] = field(default=None, repr=False, compare=False)

    def is_evaluation_invalid(self) -> bool:
        return (
            self.evaluation_status != "valid"
            or bool(self.invalid_reason)
            or bool((self.metadata or {}).get("evaluation_status") == "invalid")
            or bool((self.metadata or {}).get("protocol_failure", False))
            or "protocol_failure" in (self.error_types or [])
            or "json_parse_failed" in (self.error_types or [])
        )

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value.pop("_signature_artifact_version", None)
        value.pop("_signature_artifact_payload", None)
        value.pop("_occurrence_artifact_payload", None)
        if self._signature_artifact_version == 0:
            # Historical episodes with no signature fields remain unclassified
            # until an explicitly versioned offline re-analysis is requested.
            value["vulnerability_signature"] = None
            value["failure_occurrence"] = None
            value["failure_signature"] = None
            return value
        if self._signature_artifact_version == 1:
            # A V1 episode loaded from historical results stays V1 when it is
            # copied or reserialized; it is never silently reinterpreted.
            value["vulnerability_signature"] = None
            value["failure_occurrence"] = None
            value["failure_signature"] = self._signature_artifact_payload
            return value
        if self._signature_artifact_version == 2:
            value["vulnerability_signature"] = self._signature_artifact_payload
            value["failure_occurrence"] = self._occurrence_artifact_payload
            value["failure_signature"] = self._signature_artifact_payload
            return value

        signature = None
        occurrence = None
        if not self.task_success and not self.is_evaluation_invalid():
            signature = VulnerabilitySignature.from_episode(self)
            occurrence = FailureOccurrence.from_episode(self, signature)
        signature_value = signature.to_dict() if signature else None
        value["vulnerability_signature"] = signature_value
        value["failure_occurrence"] = occurrence.to_dict() if occurrence else None
        # Keep the historical key as a migration alias. Its schema_version
        # makes V2 identity explicit; readers treat unversioned historical
        # payloads as V1 and keep those counts separate.
        value["failure_signature"] = signature_value
        return value

    def vulnerability_signature_v2(self) -> Optional[VulnerabilitySignature]:
        """Return only a V2 identity; legacy-loaded episodes are deliberately isolated."""
        if self._signature_artifact_version in {0, 1} or self.task_success or self.is_evaluation_invalid():
            return None
        if self._signature_artifact_version == 2:
            if not isinstance(self._signature_artifact_payload, dict):
                return None
            return VulnerabilitySignature.from_dict(self._signature_artifact_payload)
        return VulnerabilitySignature.from_episode(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpisodeResult":
        value = {key: item for key, item in data.items() if key in cls.__dataclass_fields__}
        value.pop("_signature_artifact_version", None)
        value.pop("_signature_artifact_payload", None)
        value.pop("_occurrence_artifact_payload", None)
        episode = cls(**value)
        signature = data.get("vulnerability_signature")
        alias = data.get("failure_signature")
        occurrence = data.get("failure_occurrence")
        if signature is not None and not isinstance(signature, dict):
            raise ValueError("vulnerability_signature must be an object or null")
        if alias is not None and not isinstance(alias, dict):
            raise ValueError("failure_signature must be an object or null")
        if isinstance(signature, dict) and isinstance(alias, dict) and signature != alias:
            raise ValueError("vulnerability_signature and failure_signature alias disagree")

        payload = signature if isinstance(signature, dict) else alias
        if isinstance(payload, dict):
            raw_version = payload.get("schema_version", 1)
            version = int(1 if raw_version is None else raw_version)
        else:
            version = 0
        if version == 2:
            # Validate the complete signature contract at the artifact boundary.
            VulnerabilitySignature.from_dict(payload)
            episode._signature_artifact_version = 2
            episode._signature_artifact_payload = dict(payload)
            episode._occurrence_artifact_payload = dict(occurrence) if isinstance(occurrence, dict) else None
            if occurrence is not None:
                FailureOccurrence.from_dict(occurrence)
        elif version == 1:
            # V1 artifacts predate schema_version; absence is explicitly V1.
            episode._signature_artifact_version = 1
            episode._signature_artifact_payload = dict(payload)
        elif version == 0:
            episode._signature_artifact_version = 0
        else:
            raise ValueError(f"unsupported episode signature schema_version: {version}")
        return episode


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
    latest_adversary_delta: Dict[str, float] = field(default_factory=dict)
    replay_delta: Dict[str, float] = field(default_factory=dict)
    robust_delta: Dict[str, float] = field(default_factory=dict)
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
