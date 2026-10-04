"""Small runtime records for Customer search and scored benchmark episodes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any, Optional


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ServicePolicy:
    """Identity record for the fixed, non-evolving Service S0."""

    policy_id: str = "service_policy_s0"

    def to_dict(self) -> dict[str, str]:
        return {"policy_id": self.policy_id}

    def semantic_dict(self) -> dict[str, str]:
        return {"service": self.policy_id}

    def semantic_fingerprint(self) -> str:
        return _fingerprint(self.semantic_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ServicePolicy":
        policy = cls(policy_id=str(data.get("policy_id", "")))
        if policy != cls():
            raise ValueError("Customer search only supports the fixed ServicePolicy S0")
        return policy


@dataclass
class EpisodeResult:
    """Official episode outcome plus runtime-validity and trace provenance."""

    episode_id: str
    scenario: str
    case_id: str
    customer_policy_id: str
    service_policy_id: str
    split: str
    generation: int
    task_success: Optional[bool]
    execution_score: float
    sage_style_score: float = 0.0
    verification_score: float = 0.0
    policy_score: float = 0.0
    action_execution_score: float = 0.0
    goal_fulfillment_score: float = 0.0
    error_types: list[str] = field(default_factory=list)
    predicted_action: str = ""
    executed_action: str = ""
    tool_sequence_summary: list[str] = field(default_factory=list)
    termination_reason: str = ""
    dialogue: list[dict[str, Any]] = field(default_factory=list)
    trace_ref: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    evaluation_status: str = "valid"
    invalid_reason: Optional[str] = None
    protocol_valid: bool = True
    environment_valid: bool = True
    validity_reasons: list[str] = field(default_factory=list)

    def is_evaluation_invalid(self) -> bool:
        runtime_error_markers = {
            "protocol_failure", "json_parse_failed", "provider_error", "timeout",
            "llm_timeout", "transport_error", "tls_error", "output_truncated",
            "no_valid_agent_decision", "empty_output", "empty_message",
            "missing_official_score",
        }
        return (
            self.evaluation_status != "valid"
            or bool(self.invalid_reason)
            or self.metadata.get("evaluation_status") == "invalid"
            or bool(self.metadata.get("protocol_failure"))
            or bool(runtime_error_markers.intersection(self.error_types or []))
            or bool(self.metadata.get("provider_error"))
            or bool(self.metadata.get("timeout"))
            or bool(self.metadata.get("output_truncated"))
            or bool(self.metadata.get("no_valid_agent_decision"))
        )

    def is_runtime_evaluable(self) -> bool:
        """Only technical validity and an official boolean score define fitness evidence."""
        return (
            not self.is_evaluation_invalid()
            and self.protocol_valid
            and self.environment_valid
            and isinstance(self.task_success, bool)
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EpisodeResult":
        return cls(**data)
