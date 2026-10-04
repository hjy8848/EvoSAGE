"""Structural validation for the free-text Customer policy object."""

from __future__ import annotations

from .policy import AdversaryPolicy


class AdversaryPolicyValidator:
    """Validate structure and provenance, never Customer behavior or claims."""

    def validate(self, policy: AdversaryPolicy) -> None:
        if not isinstance(policy, AdversaryPolicy):
            raise TypeError("Customer search requires the compact AdversaryPolicy schema")
        policy.validate_integrity()
