"""Failure-driven structured service patch proposal and gated evolution."""

from __future__ import annotations

import json
import re

from .evaluator_adapter import aggregate_episode_metrics
from .schemas import ServicePatch, ServicePolicy, ServiceRule
from .service_gate import ServiceGate
from .service_policy import ServicePolicyCompiler, ServicePolicySanitizer


class ServiceEvolver:
    def __init__(self, seed=7, sanitizer=None, compiler=None, gate=None, patch_generator=None):
        self.seed = seed
        self.sanitizer = sanitizer or ServicePolicySanitizer()
        self.compiler = compiler or ServicePolicyCompiler(self.sanitizer)
        self.gate = gate or ServiceGate()
        self.patch_generator = patch_generator

    def propose(self, policy: ServicePolicy, failures, generation: int, count: int = 5):
        if self.patch_generator is not None:
            try:
                generated = self.patch_generator.generate(policy, failures, generation, count)
                valid = []
                for patch in generated:
                    try:
                        valid.append(self.sanitizer.sanitize(patch))
                    except Exception:
                        continue
                if valid:
                    return valid[:count]
            except Exception:
                pass
        errors = {error for failure in failures for error in failure.error_types}
        templates = []
        if {"authoritative_conflict", "claimed_action_not_executed", "wrong_final_action"} & errors:
            templates.append(("ACTION_GROUNDING", "Before claiming resolution, execute the applicable action tool and ground the reply in its successful result."))
        if errors:
            templates.append(("VERIFICATION", "Verify authoritative order and customer fields with the corresponding query tool before selecting a state-dependent action."))
        if "tool_loop_limit" in errors:
            templates.append(("TOOL_USE", "Use the smallest set of necessary queries, then either execute a supported action or explain the blocking error."))
        if not templates:
            templates.append(("RECOVERY", "After a failed action, inspect the returned error and request only the missing information needed for a valid next step."))
        patches = []
        for index in range(max(0, count)):
            category, text = templates[index % len(templates)]
            rule = ServiceRule(rule_id=f"service_rule_g{generation}_{index}", category=category, text=text,
                               rationale="derived from observed failure signatures", generation_added=generation,
                               source_failure_signatures=[failure.signature_id for failure in failures])
            patches.append(ServicePatch(patch_id=f"service_patch_g{generation}_{index}", patch_type="add",
                                        rules=[rule], rationale="failure-driven structured patch",
                                        evidence_count=len(failures), source_failure_ids=[failure.signature_id for failure in failures]))
        return patches

    def evolve(self, incumbent, failures, normal_cases, adversarial_cases, evaluator, generation, count=5,
               customer_policy=None, replay_policies=None, replay_attack_count=0):
        customer_policy = customer_policy or self._baseline_customer()
        replay_policies = list(replay_policies or [])[:replay_attack_count or None]
        suite_policies = [customer_policy] + replay_policies
        def evaluate_adversarial(service_policy, phase):
            values = []
            for policy in suite_policies:
                values.extend(evaluator.evaluate(policy, service_policy, adversarial_cases, "validation", generation, phase))
            return values
        baseline_adv = evaluate_adversarial(incumbent, "service_baseline")
        baseline_normal = evaluator.evaluate(self._baseline_customer(), incumbent, normal_cases, "validation", generation, "service_normal_baseline")
        accepted = []
        for patch in self.propose(incumbent, failures, generation, count):
            try:
                candidate = self.compiler.apply_patch(incumbent, patch, generation=generation)
                candidate_adv = evaluate_adversarial(candidate, "service_candidate")
                candidate_normal = evaluator.evaluate(self._baseline_customer(), candidate, normal_cases, "validation", generation, "service_normal_candidate")
                decision = self.gate.accept(baseline_adv, candidate_adv, baseline_normal, candidate_normal)
                if decision.accepted:
                    accepted.append((decision, candidate, patch, aggregate_episode_metrics(candidate_adv)))
            except Exception:
                continue
        if accepted:
            # Evaluate every candidate before selecting the largest robust
            # improvement; policy id is a deterministic final tie-breaker.
            decision, candidate, patch, metrics = sorted(
                accepted,
                key=lambda item: (-item[0].delta, -metrics_value(item[3], "execution_score"), item[2].patch_id),
            )[0]
            return candidate, decision, patch
        rejected = self.gate.evaluate(aggregate_episode_metrics(baseline_adv), aggregate_episode_metrics(baseline_adv))
        rejected.reason = "no_candidate_passed_validation_gate"
        return incumbent, rejected, None

    @staticmethod
    def _baseline_customer():
        from .schemas import CustomerPolicy
        return CustomerPolicy()


def metrics_value(metrics, key):
    return float(metrics.get(key, 0.0))


class LLMServicePatchGenerator:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def generate(self, policy, failures, generation, count):
        view = [{"errors": list(item.error_types), "node": item.sop_node,
                 "predicted_action": item.predicted_action, "executed_action": item.executed_action,
                 "termination": item.termination_reason} for item in failures]
        prompt = (
            "Analyze these abstract customer-service failure signatures and propose structured, "
            "general service rules. Do not mention case IDs, order IDs, expected paths/actions, "
            "hidden values, or evaluator manipulation. Return a JSON array only with objects "
            "containing category, text, rationale. Categories must be VERIFICATION, "
            "ACTION_GROUNDING, RECOVERY, TOOL_USE, or COMMUNICATION.\n"
            f"Failure signatures: {json.dumps(view, ensure_ascii=False)}\n"
            f"Existing active rules: {json.dumps([r.text for r in policy.rules if r.active], ensure_ascii=False)}\n"
            f"Generate up to {count} distinct patches."
        )
        response = self.llm_client.generate(prompt=prompt, temperature=0.3, max_tokens=1600)
        text = re.sub(r"^```(?:json)?|```$", "", response.text.strip(), flags=re.I | re.M).strip()
        value = json.loads(text)
        if isinstance(value, dict):
            value = value.get("patches", [value])
        source_ids = [item.signature_id for item in failures]
        patches = []
        for index, item in enumerate(value if isinstance(value, list) else []):
            if not isinstance(item, dict) or not item.get("text"):
                continue
            rule = ServiceRule(
                rule_id=f"service_rule_g{generation}_llm_{index}",
                category=str(item.get("category", "RECOVERY")).upper(),
                text=str(item["text"]),
                rationale=str(item.get("rationale", "LLM-derived from abstract failures")),
                source_failure_signatures=source_ids,
                generation_added=generation,
            )
            patches.append(ServicePatch(
                patch_id=f"service_patch_g{generation}_llm_{index}", patch_type="add",
                rules=[rule], rationale="LLM-derived structured patch", evidence_count=len(failures),
                source_failure_ids=source_ids,
            ))
        return patches
