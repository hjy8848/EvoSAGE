"""Minimal hard boundary for strategy schema and benchmark integrity."""

from __future__ import annotations

from .policy import AdversaryPolicy


class AdversaryPolicyValidator:
    """Reject malformed policies and explicit benchmark/harness manipulation only."""

    def validate(self, policy: AdversaryPolicy) -> None:
        if not isinstance(policy, AdversaryPolicy):
            raise TypeError("Customer search requires the compact AdversaryPolicy schema")
        policy.validate_integrity()
