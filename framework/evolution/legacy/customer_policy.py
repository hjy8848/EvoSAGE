"""Customer policy validation, compilation and deterministic simulation."""

from __future__ import annotations

import copy
from typing import Optional

from ...backend.types import CaseSpec
from ..schemas import CustomerPolicy, PolicyValidationError


class CustomerPolicyValidator:
    """Validate strategy executability and benchmark integrity only.

    Business-claim truthfulness, hidden-value matching, and tactic allowlists
    are deliberately not validity criteria: misleading the Service is the
    adversarial task, while backend/evaluator truth remains authoritative.
    """

    def __init__(self, allowed_tags=None):
        # Accept the obsolete constructor argument so old configs remain
        # readable; it no longer constrains the open strategy space.
        self.allowed_tags = None

    def validate(self, policy: CustomerPolicy, case_spec: Optional[CaseSpec] = None) -> None:
        # The new Customer-only core uses a compact schema with its own
        # integrity validator. This branch keeps the legacy combined runner
        # able to execute compact policies without restoring business checks.
        from ..customer.policy import AdversaryPolicy
        if isinstance(policy, AdversaryPolicy):
            policy.validate_integrity()
            return
        policy.validate_for_case(case_spec)
        if not policy.policy_id:
            raise PolicyValidationError("adversarial strategy needs a stable id")
        if not isinstance(policy.strategy, str) or not policy.strategy.strip():
            raise PolicyValidationError("adversarial strategy must be non-empty text")


class CustomerPolicyCompiler:
    def __init__(self, validator: Optional[CustomerPolicyValidator] = None):
        self.validator = validator or CustomerPolicyValidator()

    def compile(self, policy: CustomerPolicy, case_spec: CaseSpec) -> CustomerPolicy:
        original = copy.deepcopy(case_spec)
        self.validator.validate(policy, case_spec)
        compiled = copy.deepcopy(policy)
        compiled.assert_immutable_case(case_spec, original)
        return compiled

    def compile_runtime_guidance(self, policy: CustomerPolicy) -> str:
        self.validator.validate(policy)
        return policy.runtime_guidance()


def policy_for_case(policy: CustomerPolicy, case_spec: CaseSpec) -> CustomerPolicy:
    return CustomerPolicyCompiler().compile(policy, case_spec)
