"""Structured service-policy patches and leakage-safe compilation."""

from __future__ import annotations

import copy
from typing import Iterable, Optional

from .schemas import (
    PolicyValidationError,
    ServicePatch,
    ServicePolicy,
    ServiceRule,
    reject_forbidden_service_text,
)


class ServicePolicySanitizer:
    def __init__(self, allowed_categories: Optional[Iterable[str]] = None):
        self.allowed_categories = {item.upper() for item in (allowed_categories or {
            "VERIFICATION", "ACTION_GROUNDING", "RECOVERY", "TOOL_USE", "COMMUNICATION"
        })}

    def sanitize(self, patch: ServicePatch) -> ServicePatch:
        if not patch.patch_id or not patch.rules:
            raise PolicyValidationError("service patch must contain an id and at least one rule")
        combined = "\n".join(rule.text + "\n" + rule.rationale for rule in patch.rules)
        reason = reject_forbidden_service_text(combined + "\n" + patch.rationale)
        if reason:
            raise PolicyValidationError(reason)
        for rule in patch.rules:
            if rule.category.upper() not in self.allowed_categories:
                raise PolicyValidationError(f"unsupported service rule category: {rule.category}")
            text = rule.text.lower()
            if "always reject" in text or "always transfer" in text or "query every" in text:
                raise PolicyValidationError("degenerate always-reject/transfer/query rule")
            if not rule.text.strip():
                raise PolicyValidationError("empty service rule")
        return copy.deepcopy(patch)


class ServicePolicyValidator:
    """Validate an already materialized policy before it is shown to an Agent."""

    def __init__(self, sanitizer: Optional[ServicePolicySanitizer] = None):
        self.sanitizer = sanitizer or ServicePolicySanitizer()

    def validate(self, policy: ServicePolicy) -> None:
        for rule in policy.rules:
            self.sanitizer.sanitize(ServicePatch(
                patch_id=f"validation_{rule.rule_id}", patch_type="add", rules=[rule],
            ))


class ServicePolicyCompiler:
    def __init__(self, sanitizer: Optional[ServicePolicySanitizer] = None):
        self.sanitizer = sanitizer or ServicePolicySanitizer()

    def apply_patch(self, policy: ServicePolicy, patch: ServicePatch, generation: Optional[int] = None) -> ServicePolicy:
        patch = self.sanitizer.sanitize(patch)
        by_id = {rule.rule_id: copy.deepcopy(rule) for rule in policy.rules}
        if patch.patch_type.lower() in {"remove", "disable"}:
            for rule in patch.rules:
                if rule.rule_id in by_id:
                    by_id[rule.rule_id].active = False
        else:
            for rule in patch.rules:
                by_id[rule.rule_id] = copy.deepcopy(rule)
        next_policy = policy.clone_with(by_id.values(), generation=generation)
        next_policy.mutation_history.append({
            "patch_id": patch.patch_id,
            "patch_type": patch.patch_type,
            "rationale": patch.rationale,
            "source_failure_ids": list(patch.source_failure_ids),
        })
        return next_policy

    def compile_prompt(self, policy: ServicePolicy) -> str:
        for rule in policy.rules:
            if rule.active:
                reason = reject_forbidden_service_text(rule.text + "\n" + rule.rationale)
                if reason:
                    raise PolicyValidationError(reason)
        return policy.overlay_prompt()
