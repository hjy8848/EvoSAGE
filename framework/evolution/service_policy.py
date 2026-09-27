"""Structured service-policy patches and leakage-safe compilation."""

from __future__ import annotations

import copy
import re
from typing import Iterable, Optional

from .schemas import (
    PolicyValidationError,
    ServicePatch,
    ServicePolicy,
    ServiceRule,
    reject_forbidden_service_text,
)


class ServicePolicySanitizer:
    TRIGGERS = {
        "MISSING_REQUIRED_ARGUMENT", "AUTHORITATIVE_RESULT_AVAILABLE", "TOOL_FAILURE",
        "ACTION_FAILURE", "BEFORE_STATE_DEPENDENT_ACTION", "BEFORE_COMPLETION_CLAIM",
    }
    OBLIGATIONS = {
        "ASK_FOR_ARGUMENT", "VERIFY_WITH_TOOL", "GROUND_REPLY_IN_TOOL_RESULT",
        "EXPLAIN_FAILURE", "REQUEST_ONLY_MISSING_INFO",
    }
    PROHIBITIONS = {
        "DO_NOT_CLAIM_UNEXECUTED_ACTION", "DO_NOT_INFER_HIDDEN_STATE",
        "DO_NOT_EXECUTE_ACTION_BEFORE_REQUIRED_VERIFICATION",
    }
    ORDERING = {"VERIFY_BEFORE_ACTION", "ACTION_SUCCESS_BEFORE_COMPLETION_CLAIM"}
    ARGUMENTS = {
        "required_identifier", "order_identifier", "customer_identifier",
        "record_identifier", "booking_identifier", "ticket_identifier",
        "account_identifier",
    }
    RECOVERY = {
        "ASK_FOR_ARGUMENT", "EXPLAIN_FAILURE", "RETRY_AFTER_CORRECTION",
    }

    def __init__(self, allowed_categories: Optional[Iterable[str]] = None):
        self.allowed_categories = {item.upper() for item in (allowed_categories or {
            "VERIFICATION", "ACTION_GROUNDING", "RECOVERY", "TOOL_USE", "COMMUNICATION"
        })}

    def sanitize(self, patch: ServicePatch) -> ServicePatch:
        if not patch.patch_id or not patch.rules:
            raise PolicyValidationError("service patch must contain an id and at least one rule")
        normalized = copy.deepcopy(patch)
        for rule in normalized.rules:
            if rule.category.upper() not in self.allowed_categories:
                raise PolicyValidationError(f"unsupported service rule category: {rule.category}")
            rendered = ServicePolicyCompiler.compile_rule_text(rule)
            if not rendered.strip():
                raise PolicyValidationError("empty service rule")
            combined = "\n".join((rendered, rule.rationale, patch.rationale))
            reason = reject_forbidden_service_text(combined)
            if reason:
                raise PolicyValidationError(reason)
            lowered = combined.lower()
            if "always reject" in lowered or "always transfer" in lowered or "query every" in lowered:
                raise PolicyValidationError("degenerate always-reject/transfer/query rule")
            if re.search(r"(?:uncertain|uncertainty).{0,80}(?:reject|transfer|refund)|(?:reject|transfer|refund).{0,80}(?:uncertain|uncertainty)", lowered):
                raise PolicyValidationError("business outcome cannot be rewritten based on uncertainty")
            if rule.rule_schema_version == 2:
                # Text is a deterministic compiled artifact, never the LLM's
                # executable source of truth.
                rule.text = rendered
        return normalized


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
        active = []
        for rule in policy.rules:
            if not rule.active:
                continue
            self.sanitizer.sanitize(ServicePatch(
                patch_id=f"compile_{rule.rule_id}", patch_type="add", rules=[rule],
            ))
            active.append(self.compile_rule_text(rule))
        if not active:
            return ""
        return "\n【当前服务策略补丁】\n" + "\n".join(f"- {item}" for item in active)

    @staticmethod
    def _check_v2_structure(rule: ServiceRule) -> None:
        if rule.rule_schema_version != 2:
            if rule.rule_schema_version == 1 and rule.text.strip():
                return
            raise PolicyValidationError("unsupported service rule schema version")
        trigger = rule.trigger
        if not isinstance(trigger, dict) or set(trigger) - {"type", "argument"}:
            raise PolicyValidationError("trigger must contain only type and optional generic argument")
        trigger_type = str(trigger.get("type", "")).upper()
        if trigger_type not in ServicePolicySanitizer.TRIGGERS:
            raise PolicyValidationError(f"unsupported service trigger: {trigger_type or 'missing'}")
        if "argument" in trigger and trigger["argument"] not in ServicePolicySanitizer.ARGUMENTS:
            raise PolicyValidationError("trigger argument must be a generic semantic identifier")
        if trigger_type == "MISSING_REQUIRED_ARGUMENT" and "argument" not in trigger:
            raise PolicyValidationError("missing-required-argument trigger needs a generic argument")

        collections = (
            ("obligations", rule.obligations, ServicePolicySanitizer.OBLIGATIONS),
            ("prohibitions", rule.prohibitions, ServicePolicySanitizer.PROHIBITIONS),
            ("ordering_constraints", rule.ordering_constraints, ServicePolicySanitizer.ORDERING),
        )
        for field_name, items, allowed in collections:
            if not isinstance(items, list):
                raise PolicyValidationError(f"{field_name} must be a list")
            for item in items:
                if not isinstance(item, dict) or set(item) - {"type", "argument"}:
                    raise PolicyValidationError(f"{field_name} entries contain unsupported fields")
                operation = str(item.get("type", "")).upper()
                if operation not in allowed:
                    raise PolicyValidationError(f"unsupported {field_name} operation: {operation or 'missing'}")
                if "argument" in item and item["argument"] not in ServicePolicySanitizer.ARGUMENTS:
                    raise PolicyValidationError(f"{field_name} argument must be generic")
                if operation == "ASK_FOR_ARGUMENT" and "argument" not in item:
                    raise PolicyValidationError("ASK_FOR_ARGUMENT requires a generic argument")
        if not isinstance(rule.recovery, dict) or set(rule.recovery) - {"type"}:
            raise PolicyValidationError("recovery must contain only an allowed type")
        if rule.recovery and str(rule.recovery.get("type", "")).upper() not in ServicePolicySanitizer.RECOVERY:
            raise PolicyValidationError("unsupported recovery operation")
        if not (rule.obligations or rule.prohibitions or rule.ordering_constraints or rule.recovery):
            raise PolicyValidationError("structured service rule needs at least one safe operation")

    @classmethod
    def compile_rule_text(cls, rule: ServiceRule) -> str:
        """Deterministically render an allowlisted V2 rule; preserve V1 text."""
        cls._check_v2_structure(rule)
        if rule.rule_schema_version == 1:
            return rule.text.strip()

        trigger_text = {
            "MISSING_REQUIRED_ARGUMENT": "When a required argument is missing",
            "AUTHORITATIVE_RESULT_AVAILABLE": "When an authoritative tool result is available",
            "TOOL_FAILURE": "When a tool call fails",
            "ACTION_FAILURE": "When a business action fails",
            "BEFORE_STATE_DEPENDENT_ACTION": "Before a state-dependent action",
            "BEFORE_COMPLETION_CLAIM": "Before claiming that a request is complete",
        }[str(rule.trigger["type"]).upper()]
        argument = rule.trigger.get("argument")
        if argument:
            trigger_text += f" ({argument.replace('_', ' ')})"

        rendered_operations = []
        operation_text = {
            "ASK_FOR_ARGUMENT": "ask the customer for the missing required argument",
            "VERIFY_WITH_TOOL": "verify the relevant fact with the corresponding query tool",
            "GROUND_REPLY_IN_TOOL_RESULT": "ground decisions and replies in the actual tool result",
            "EXPLAIN_FAILURE": "explain the failure using the returned error",
            "REQUEST_ONLY_MISSING_INFO": "request only information needed for a valid next step",
            "DO_NOT_CLAIM_UNEXECUTED_ACTION": "do not claim an action succeeded unless its action tool succeeded",
            "DO_NOT_INFER_HIDDEN_STATE": "do not infer hidden backend state from a customer claim",
            "DO_NOT_EXECUTE_ACTION_BEFORE_REQUIRED_VERIFICATION": "do not execute a state-dependent action before required verification",
            "VERIFY_BEFORE_ACTION": "complete required verification before an action",
            "ACTION_SUCCESS_BEFORE_COMPLETION_CLAIM": "confirm completion only after the action tool reports success",
        }
        for item in rule.obligations + rule.prohibitions + rule.ordering_constraints:
            operation = str(item["type"]).upper()
            if (
                operation == "ASK_FOR_ARGUMENT"
                and str(rule.trigger["type"]).upper() == "MISSING_REQUIRED_ARGUMENT"
            ):
                sentence = (
                    "ask the customer for the missing required argument before invoking the "
                    "corresponding query tool; do not send an empty placeholder"
                )
            else:
                sentence = operation_text[operation]
            if item.get("argument"):
                sentence += f" ({item['argument'].replace('_', ' ')})"
            rendered_operations.append(sentence)
        if rule.recovery:
            recovery_text = {
                "ASK_FOR_ARGUMENT": "ask for the missing argument before retrying",
                "EXPLAIN_FAILURE": "explain the failure and the safe next step",
                "RETRY_AFTER_CORRECTION": "retry only after the missing or invalid input is corrected",
            }[str(rule.recovery["type"]).upper()]
            rendered_operations.append(recovery_text)
        rendered = trigger_text
        if rendered_operations:
            rendered += ": " + "; ".join(rendered_operations)
        rendered += "."
        return rendered
