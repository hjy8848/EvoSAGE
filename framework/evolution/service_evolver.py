"""Failure-driven structured service patch proposal and gated evolution."""

from __future__ import annotations

import json
import re

from .evaluator_adapter import aggregate_episode_metrics
from .schemas import ServicePatch, ServicePolicy, ServiceRule
from .service_gate import ServiceGate
from .service_policy import ServicePolicyCompiler, ServicePolicySanitizer


class ServiceEvolver:
    def __init__(self, seed=7, sanitizer=None, compiler=None, gate=None, patch_generator=None,
                 require_patch_generator: bool = False):
        self.seed = seed
        self.sanitizer = sanitizer or ServicePolicySanitizer()
        self.compiler = compiler or ServicePolicyCompiler(self.sanitizer)
        self.gate = gate or ServiceGate()
        self.patch_generator = patch_generator
        self.require_patch_generator = require_patch_generator
        self.last_candidate_records = []
        self.last_baseline_metrics = {}
        self.last_selected_metrics = {}

    def propose(self, policy: ServicePolicy, failures, generation: int, count: int = 5, defense_summary=None, historical_summary=None):
        if self.patch_generator is not None:
            try:
                generated = self.patch_generator.generate(policy, failures, generation, count, defense_summary, historical_summary)
                valid = []
                for patch in generated:
                    try:
                        valid.append(self.sanitizer.sanitize(patch))
                    except Exception:
                        continue
                if valid:
                    return valid[:count]
            except Exception as exc:
                if self.require_patch_generator:
                    raise RuntimeError("LLM service patch generation failed in strict real mode") from exc
                pass
            if self.require_patch_generator:
                raise RuntimeError("LLM service patch generator returned no valid candidates")
        elif self.require_patch_generator:
            raise RuntimeError("strict real mode requires an LLM service patch generator")
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
               customer_policy=None, replay_policies=None, replay_attack_count=0, defense_summary=None, historical_summary=None):
        normal_cases = list(normal_cases)
        adversarial_cases = list(adversarial_cases)
        if any(getattr(case, "split", "") == "heldout_test" for case in normal_cases + adversarial_cases):
            raise AssertionError("ServiceEvolver cannot consume heldout cases")
        self.last_candidate_records = []
        customer_policy = customer_policy or self._baseline_customer()
        replay_policies = list(replay_policies or [])
        replay_policies = replay_policies[:replay_attack_count] if replay_attack_count > 0 else []
        def evaluate_latest(service_policy):
            return evaluator.evaluate(
                customer_policy, service_policy, adversarial_cases,
                "validation", generation, "service_candidate_latest"
            )

        def evaluate_suite(service_policy, phase):
            latest = evaluator.evaluate(
                customer_policy, service_policy, adversarial_cases,
                "validation", generation, phase + "_latest"
            )
            replay = []
            for policy in replay_policies:
                replay.extend(evaluator.evaluate(policy, service_policy, adversarial_cases, "validation", generation, phase + "_replay"))
            return latest, replay

        def summarize(latest, replay, normal=None):
            adversarial = list(latest) + list(replay)
            robust = aggregate_episode_metrics(adversarial)
            robust["latest_task_success"] = aggregate_episode_metrics(latest)["task_success"]
            robust["replay_task_success"] = aggregate_episode_metrics(replay)["task_success"] if replay else robust["latest_task_success"]
            robust["robust_task_success"] = robust["task_success"]
            if normal is not None:
                robust["normal_task_success"] = aggregate_episode_metrics(normal)["task_success"]
            return robust

        def invalid_reasons(episodes):
            reasons = []
            for episode in episodes or []:
                if episode.is_evaluation_invalid():
                    reasons.extend(
                        episode.metadata.get("invalid_reasons", [])
                        if isinstance(episode.metadata, dict) else []
                    )
                    if episode.invalid_reason:
                        reasons.append(episode.invalid_reason)
                    elif episode.error_types:
                        reasons.extend(
                            error for error in episode.error_types
                            if error in {"protocol_failure", "json_parse_failed", "output_truncated", "timeout", "provider_error", "no_valid_agent_decision"}
                        )
            return list(dict.fromkeys(reasons or ["protocol_failure"]))

        def attempt_count(*episode_groups):
            return max(
                [
                    int((episode.metadata or {}).get("evaluation_attempts", 1) or 1)
                    for group in episode_groups
                    for episode in (group or [])
                ]
                or [0]
            )

        baseline_latest, baseline_replay = evaluate_suite(incumbent, "service_baseline")
        baseline_normal = evaluator.evaluate(self._baseline_customer(), incumbent, normal_cases, "validation", generation, "service_normal_baseline")
        baseline_metrics = summarize(baseline_latest, baseline_replay, baseline_normal)
        self.last_baseline_metrics = baseline_metrics
        baseline_invalid = invalid_reasons(baseline_latest + baseline_replay + baseline_normal)
        baseline_has_invalid = any(
            episode.is_evaluation_invalid()
            for episode in baseline_latest + baseline_replay + baseline_normal
        )
        accepted = []
        invalid_candidate_reasons = []
        substantive_candidate_rejection = False
        for patch in self.propose(incumbent, failures, generation, count, defense_summary, historical_summary):
            candidate = None
            try:
                candidate = self.compiler.apply_patch(
                    incumbent, patch, generation=incumbent.generation + 1
                )
                # Candidate policies need independent provenance even when
                # several patches are evaluated within the same generation.
                candidate.policy_id = f"{candidate.policy_id}_{patch.patch_id}"
                if baseline_has_invalid:
                    reason = f"baseline_evaluation_invalid:{baseline_invalid[0]}"
                    invalid_candidate_reasons.append(reason)
                    self.last_candidate_records.append({
                        "patch_id": patch.patch_id,
                        "service_policy_id": candidate.policy_id,
                        "source_failure_ids": list(patch.source_failure_ids),
                        "patch": patch.to_dict(),
                        "candidate_policy": candidate.to_dict(),
                        "accepted": False,
                        "evaluation_status": "invalid",
                        "invalid_reason": reason,
                        "invalid_reasons": baseline_invalid,
                        "reason": reason,
                        "delta": None,
                        "metrics": None,
                        "attempt_count": attempt_count(baseline_latest, baseline_replay, baseline_normal),
                        "normal_regression_cases": [],
                    })
                    continue
                # Stage 1 intentionally evaluates only the current/latest
                # adversarial customer.  Historical replay and normal-user
                # regression are paid only for candidates that first improve
                # the attack currently driving the evolution.
                candidate_latest = evaluate_latest(candidate)
                # Stage 1: reject patches that do not improve the latest
                # adversarial attack before spending API calls on normal-user
                # regression and historical replay evaluation.
                candidate_latest_metrics = summarize(candidate_latest, [])
                latest_baseline_metrics = summarize(baseline_latest, [])
                latest_invalid = invalid_reasons(candidate_latest)
                if any(item.is_evaluation_invalid() for item in candidate_latest):
                    reason = f"candidate_evaluation_invalid:{latest_invalid[0]}"
                    invalid_candidate_reasons.append(reason)
                    self.last_candidate_records.append({
                        "patch_id": patch.patch_id,
                        "service_policy_id": candidate.policy_id,
                        "source_failure_ids": list(patch.source_failure_ids),
                        "patch": patch.to_dict(),
                        "candidate_policy": candidate.to_dict(),
                        "accepted": False,
                        "evaluation_status": "invalid",
                        "invalid_reason": latest_invalid[0],
                        "invalid_reasons": latest_invalid,
                        "reason": reason,
                        "delta": None,
                        "metrics": None,
                        "raw_metrics": candidate_latest_metrics,
                        "attempt_count": attempt_count(candidate_latest),
                        "stages": {"latest_attack": {"evaluation_status": "invalid", "invalid_reasons": latest_invalid}},
                        "normal_regression_cases": [],
                    })
                    continue
                latest_decision = self.gate.evaluate(latest_baseline_metrics, candidate_latest_metrics)
                if not latest_decision.accepted:
                    self.last_candidate_records.append({
                        "patch_id": patch.patch_id,
                        "service_policy_id": candidate.policy_id,
                        "source_failure_ids": list(patch.source_failure_ids),
                        # Persist the proposal even when the staged gate
                        # rejects it.  Rejected candidates are important
                        # research evidence: without the patch text and
                        # rationale we cannot tell whether the model's rule
                        # was ineffective, invalid, or aimed at the wrong
                        # failure mode.
                        "patch": patch.to_dict(),
                        "candidate_policy": candidate.to_dict(),
                        "accepted": False,
                        "evaluation_status": "valid",
                        "invalid_reasons": [],
                        "reason": f"latest_attack_filter:{latest_decision.reason}",
                        "delta": latest_decision.delta,
                        "metrics": candidate_latest_metrics,
                        "attempt_count": attempt_count(candidate_latest),
                        "stages": {"latest_attack": latest_decision.metrics},
                        "normal_regression_cases": [],
                    })
                    continue
                candidate_replay = []
                for policy in replay_policies:
                    candidate_replay.extend(evaluator.evaluate(
                        policy, candidate, adversarial_cases, "validation", generation,
                        "service_candidate_replay"
                    ))
                candidate_normal = evaluator.evaluate(self._baseline_customer(), candidate, normal_cases, "validation", generation, "service_normal_candidate")
                suite_invalid = invalid_reasons(candidate_replay + candidate_normal)
                if any(item.is_evaluation_invalid() for item in candidate_replay + candidate_normal):
                    reason = f"candidate_evaluation_invalid:{suite_invalid[0]}"
                    invalid_candidate_reasons.append(reason)
                    self.last_candidate_records.append({
                        "patch_id": patch.patch_id,
                        "service_policy_id": candidate.policy_id,
                        "source_failure_ids": list(patch.source_failure_ids),
                        "patch": patch.to_dict(),
                        "candidate_policy": candidate.to_dict(),
                        "accepted": False,
                        "evaluation_status": "invalid",
                        "invalid_reason": suite_invalid[0],
                        "invalid_reasons": suite_invalid,
                        "reason": reason,
                        "delta": None,
                        "metrics": None,
                        "raw_metrics": summarize(candidate_latest, candidate_replay, candidate_normal),
                        "attempt_count": attempt_count(candidate_latest, candidate_replay, candidate_normal),
                        "stages": {"replay_or_normal": {"evaluation_status": "invalid", "invalid_reasons": suite_invalid}},
                        "normal_regression_cases": [],
                    })
                    continue
                candidate_metrics = summarize(candidate_latest, candidate_replay, candidate_normal)
                decision = self.gate.evaluate(
                    baseline_metrics, candidate_metrics,
                    aggregate_episode_metrics(baseline_normal),
                    aggregate_episode_metrics(candidate_normal),
                )
                decision.metrics = dict(candidate_metrics, normal_delta=decision.metrics.get("normal_delta", 0.0))
                regression_cases = [
                    item.case_id for item in candidate_normal if not item.task_success
                ]
                record = {
                    "patch_id": patch.patch_id,
                    "service_policy_id": candidate.policy_id,
                    "source_failure_ids": list(patch.source_failure_ids),
                    "patch": patch.to_dict(),
                    "candidate_policy": candidate.to_dict(),
                    "accepted": decision.accepted,
                    "evaluation_status": "valid",
                    "invalid_reasons": [],
                    "reason": decision.reason,
                    "delta": decision.delta,
                    "metrics": candidate_metrics,
                    "attempt_count": attempt_count(candidate_latest, candidate_replay, candidate_normal),
                    "normal_regression_cases": regression_cases,
                }
                self.last_candidate_records.append(record)
                substantive_candidate_rejection = substantive_candidate_rejection or not decision.accepted
                if decision.accepted:
                    self.last_selected_metrics = candidate_metrics
                    accepted.append((decision, candidate, patch, candidate_metrics))
            except Exception as exc:
                # Keep the generated proposal available even if applying or
                # evaluating it fails.  The error is part of the candidate's
                # audit trail, not a reason to discard the candidate itself.
                self.last_candidate_records.append({
                    "patch_id": patch.patch_id,
                    "source_failure_ids": list(patch.source_failure_ids),
                    "patch": patch.to_dict(),
                    "candidate_policy": candidate.to_dict() if candidate is not None else None,
                    "accepted": False,
                    "evaluation_status": "invalid",
                    "invalid_reason": "provider_error",
                    "invalid_reasons": ["provider_error"],
                    "reason": "candidate_evaluation_invalid:provider_error",
                    "delta": None,
                    "metrics": None,
                    "attempt_count": 1,
                    "normal_regression_cases": [],
                    "error": f"{type(exc).__name__}: {exc}",
                })
                invalid_candidate_reasons.append("candidate_evaluation_invalid:provider_error")
                continue
        if accepted:
            # Evaluate every candidate before selecting the largest robust
            # improvement; policy id is a deterministic final tie-breaker.
            decision, candidate, patch, metrics = sorted(
                accepted,
                key=lambda item: (-item[0].delta, -metrics_value(item[3], "execution_score"), item[2].patch_id),
            )[0]
            return candidate, decision, patch
        rejected = self.gate.evaluate(baseline_metrics, baseline_metrics)
        if baseline_has_invalid:
            rejected.reason = f"baseline_evaluation_invalid:{baseline_invalid[0]}"
        elif invalid_candidate_reasons and not substantive_candidate_rejection:
            rejected.reason = invalid_candidate_reasons[0]
        else:
            rejected.reason = "no_candidate_passed_validation_gate"
        return incumbent, rejected, None

    @staticmethod
    def _baseline_customer():
        from .schemas import CustomerPolicy
        return CustomerPolicy()


def metrics_value(metrics, key):
    return float(metrics.get(key, 0.0))


def _top_k(items, limit: int = 5):
    """Keep evolution prompts bounded while retaining recent records."""
    values = list(items or [])
    return values[-max(1, limit):]


class LLMServicePatchGenerator:
    def __init__(self, llm_client, max_tokens: int = 4096, summary_limit: int = 5):
        self.llm_client = llm_client
        self.max_tokens = max_tokens
        self.summary_limit = max(1, int(summary_limit))

    def generate(self, policy, failures, generation, count, defense_summary=None, historical_summary=None):
        view = [{"errors": list(item.error_types), "node": item.sop_node,
                 "predicted_action": item.predicted_action, "executed_action": item.executed_action,
                 "termination": item.termination_reason,
                 "verification": item.required_verification_score,
                 "policy": item.policy_score,
                 "action": item.action_execution_score,
                 "goal": item.goal_fulfillment_score,
                 "tools": item.tool_sequence_summary} for item in _top_k(failures, self.summary_limit)]
        defense_summary = _top_k(defense_summary, self.summary_limit)
        historical_summary = _top_k(historical_summary, self.summary_limit)
        prompt = (
            "Analyze these abstract customer-service failure signatures and propose structured, "
            "general service rules. Do not mention case IDs, order IDs, expected paths/actions, "
            "hidden values, or evaluator manipulation. Return a JSON array only with objects "
            "containing category, text, rationale. Categories must be VERIFICATION, "
            "ACTION_GROUNDING, RECOVERY, TOOL_USE, or COMMUNICATION.\n"
            f"Failure signatures: {json.dumps(view, ensure_ascii=False)}\n"
            f"Existing active rules: {json.dumps([r.text for r in policy.rules if r.active], ensure_ascii=False)}\n"
            f"Defense archive summary: {json.dumps(defense_summary or [], ensure_ascii=False)[:6000]}\n"
            f"Historical regression summary: {json.dumps(historical_summary or [], ensure_ascii=False)[:6000]}\n"
            f"Generate up to {count} distinct patches."
        )
        response = self.llm_client.generate(
            prompt=prompt,
            temperature=0.3,
            max_tokens=self.max_tokens,
        )
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
