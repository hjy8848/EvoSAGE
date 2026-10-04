"""Explicit boundary adapters for historical Customer policy artifacts."""

from __future__ import annotations

from typing import Any

from .policy import AdversaryPolicy


def adapt_legacy_customer_policy(policy: Any) -> Any:
    """Convert a historical policy once at a legacy-runner boundary."""
    from ..schemas import LegacyCustomerPolicy

    if isinstance(policy, AdversaryPolicy) or not isinstance(policy, LegacyCustomerPolicy):
        return policy
    return load_legacy_customer_policy(policy.to_dict())


class LegacyCustomerStrategyGeneratorAdapter:
    """Adapt historical generator outputs before they enter Customer core."""

    def __init__(self, generator):
        self.generator = generator

    def __getattr__(self, name):
        return getattr(self.generator, name)

    def generate(self, **kwargs):
        policies = self.generator.generate(**kwargs)
        if isinstance(policies, list):
            return [adapt_legacy_customer_policy(policy) for policy in policies]
        return policies


def load_legacy_customer_policy(data: dict[str, Any]) -> AdversaryPolicy:
    """Convert a pre-free-text Customer policy payload into the compact schema.

    Legacy tactic fields are concatenated into strategy text once at load time;
    they are not retained as attributes or interpreted by the core.
    """
    if data.get("strategy") or data.get("description"):
        strategy = data.get("strategy") or data.get("description")
    else:
        fields = (
            "disclosure_strategy", "claim_strategy", "pressure_strategy",
            "contradiction_strategy", "timing_strategy",
            "response_to_verification", "response_to_rejection", "escalation_strategy",
        )
        strategy = "\n".join(str(data[key]).strip() for key in fields if data.get(key))
    return AdversaryPolicy(
        policy_id=str(data.get("policy_id") or data.get("name") or "legacy_adversary"),
        strategy=strategy or "Use a free-form Customer interaction strategy.",
        hypothesis=str(data.get("hypothesis") or data.get("mutation_rationale") or "loaded from legacy artifact"),
        parent_id=data.get("parent_id") or next(iter(data.get("parent_policy_ids", []) or []), None),
        generation=int(data.get("generation", 0) or 0),
        model_metadata=dict(data.get("model_metadata") or {}),
    )
