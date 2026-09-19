"""Failure-driven structured service patch proposal and gated evolution."""

from __future__ import annotations

from .schemas import ServicePatch, ServicePolicy, ServiceRule
from .service_gate import ServiceGate
from .service_policy import ServicePolicyCompiler, ServicePolicySanitizer


class ServiceEvolver:
    def __init__(self, seed=7, sanitizer=None, compiler=None, gate=None):
        self.seed = seed
        self.sanitizer = sanitizer or ServicePolicySanitizer()
        self.compiler = compiler or ServicePolicyCompiler(self.sanitizer)
        self.gate = gate or ServiceGate()

    def propose(self, policy: ServicePolicy, failures, generation: int, count: int = 5):
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

    def evolve(self, incumbent, failures, normal_cases, adversarial_cases, evaluator, generation, count=5, customer_policy=None):
        customer_policy = customer_policy or self._baseline_customer()
        baseline_adv = evaluator.evaluate(customer_policy, incumbent, adversarial_cases, "validation", generation, "service_baseline")
        baseline_normal = evaluator.evaluate(self._baseline_customer(), incumbent, normal_cases, "validation", generation, "service_normal_baseline")
        for patch in self.propose(incumbent, failures, generation, count):
            try:
                candidate = self.compiler.apply_patch(incumbent, patch, generation=generation)
                candidate_adv = evaluator.evaluate(customer_policy, candidate, adversarial_cases, "validation", generation, "service_candidate")
                candidate_normal = evaluator.evaluate(self._baseline_customer(), candidate, normal_cases, "validation", generation, "service_normal_candidate")
                decision = self.gate.accept(baseline_adv, candidate_adv, baseline_normal, candidate_normal)
                if decision.accepted:
                    return candidate, decision, patch
            except Exception:
                continue
        return incumbent, self.gate.evaluate({"task_success": 0.0}, {"task_success": 0.0}), None

    @staticmethod
    def _baseline_customer():
        from .schemas import CustomerPolicy
        return CustomerPolicy()
