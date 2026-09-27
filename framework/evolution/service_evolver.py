"""Failure-driven structured service patch proposal and gated evolution."""

from __future__ import annotations

import json
import re
import copy

from .evaluator_adapter import aggregate_episode_metrics
from .generation_protocol import GenerationProtocolError, request_json_with_retry
from .schemas import AttackInstance, CustomerPolicy, ServicePatch, ServicePolicy, ServiceRule
from .service_gate import ServiceGate
from .service_policy import ServicePolicyCompiler, ServicePolicySanitizer
from .split_manager import dataset_case_from_attack_instance


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
        self.last_generation_record = None
        self.last_baseline_metrics = {}
        self.last_selected_metrics = {}

    def propose(self, policy: ServicePolicy, failures, generation: int, count: int = 5, defense_summary=None, historical_summary=None):
        if self.patch_generator is not None:
            try:
                generated = self.patch_generator.generate(policy, failures, generation, count, defense_summary, historical_summary)
                self.last_generation_record = copy.deepcopy(
                    getattr(self.patch_generator, "last_generation_record", None)
                )
                valid = []
                for patch in generated:
                    try:
                        if self.require_patch_generator and any(
                            rule.rule_schema_version != 2 for rule in patch.rules
                        ):
                            raise ValueError("strict real generation requires ServiceRule V2")
                        sanitized = self.sanitizer.sanitize(patch)
                        valid.append(sanitized)
                        self._mark_generation_candidate(sanitized.patch_id, accepted=True)
                    except Exception as exc:
                        self._mark_generation_candidate(
                            patch.patch_id,
                            accepted=False,
                            reason=f"service_candidate_rejected:{exc}",
                        )
                        continue
                if valid:
                    return valid[:count]
                if self.last_generation_record is not None:
                    self.last_generation_record["status"] = "candidate_rejected"
                    self.last_generation_record["reason"] = "service_candidate_rejected:all_candidates"
            except Exception as exc:
                self.last_generation_record = copy.deepcopy(
                    getattr(self.patch_generator, "last_generation_record", None)
                )
                if isinstance(exc, GenerationProtocolError):
                    if self.last_generation_record is not None:
                        self.last_generation_record["status"] = "inconclusive"
                    if self.require_patch_generator:
                        raise
                if self.require_patch_generator:
                    raise RuntimeError("LLM service patch generation failed in strict real mode") from exc
                pass
            if self.require_patch_generator:
                raise RuntimeError("LLM service patch generator returned no valid candidates")
        elif self.require_patch_generator:
            raise RuntimeError("strict real mode requires an LLM service patch generator")
        errors = {error for failure in failures for error in failure.error_types}
        templates: list[tuple[str, dict, list[dict], list[dict], dict]] = []
        if {"authoritative_conflict", "claimed_action_not_executed", "wrong_final_action"} & errors:
            templates.append((
                "ACTION_GROUNDING",
                {"type": "BEFORE_COMPLETION_CLAIM"},
                [{"type": "GROUND_REPLY_IN_TOOL_RESULT"}],
                [{"type": "DO_NOT_CLAIM_UNEXECUTED_ACTION"}],
                [],
                {},
            ))
        if errors:
            templates.append((
                "VERIFICATION",
                {"type": "BEFORE_STATE_DEPENDENT_ACTION"},
                [{"type": "VERIFY_WITH_TOOL"}],
                [],
                [{"type": "VERIFY_BEFORE_ACTION"}],
                {},
            ))
        if "tool_loop_limit" in errors:
            templates.append((
                "TOOL_USE",
                {"type": "TOOL_FAILURE"},
                [{"type": "EXPLAIN_FAILURE"}, {"type": "REQUEST_ONLY_MISSING_INFO"}],
                [],
                [],
                {"type": "EXPLAIN_FAILURE"},
            ))
        if not templates:
            templates.append((
                "RECOVERY",
                {"type": "ACTION_FAILURE"},
                [{"type": "EXPLAIN_FAILURE"}, {"type": "REQUEST_ONLY_MISSING_INFO"}],
                [],
                [],
                {"type": "EXPLAIN_FAILURE"},
            ))
        patches = []
        for index in range(max(0, count)):
            category, trigger, obligations, prohibitions, ordering, recovery = templates[index % len(templates)]
            rule = ServiceRule(rule_id=f"service_rule_g{generation}_{index}", category=category, text="",
                               rationale="derived from observed failure signatures", generation_added=generation,
                               source_failure_signatures=[failure.signature_id for failure in failures],
                               rule_schema_version=2, trigger=trigger, obligations=obligations,
                               prohibitions=prohibitions, ordering_constraints=ordering, recovery=recovery)
            rule.text = ServicePolicyCompiler.compile_rule_text(rule)
            patches.append(ServicePatch(patch_id=f"service_patch_g{generation}_{index}", patch_type="add",
                                        rules=[rule], rationale="failure-driven structured patch",
                                        evidence_count=len(failures), source_failure_ids=[failure.signature_id for failure in failures]))
        return patches

    def _mark_generation_candidate(self, patch_id: str, *, accepted: bool, reason: str | None = None) -> None:
        """Annotate the raw generation record without changing gate semantics."""
        if not self.last_generation_record:
            return
        for item in self.last_generation_record.get("candidates", []):
            if item.get("patch_id") == patch_id:
                item["accepted"] = accepted
                item["candidate_validation"] = {
                    "status": "PASS" if accepted else "FAIL",
                    "exact_reason": reason,
                }
                return

    def evolve(self, incumbent, failures, normal_cases, adversarial_cases, evaluator, generation, count=5,
               customer_policy=None, replay_policies=None, replay_attack_count=0,
               exact_replay_instances=None, transfer_replay_policies=None,
               defense_summary=None, historical_summary=None):
        normal_cases = list(normal_cases)
        adversarial_cases = list(adversarial_cases)
        if any(getattr(case, "split", "") == "heldout_test" for case in normal_cases + adversarial_cases):
            raise AssertionError("ServiceEvolver cannot consume heldout cases")
        self.last_candidate_records = []
        customer_policy = customer_policy or self._baseline_customer()
        if transfer_replay_policies is None:
            transfer_replay_policies = list(replay_policies or [])
            transfer_replay_policies = (
                transfer_replay_policies[:replay_attack_count]
                if replay_attack_count > 0 else []
            )
        else:
            transfer_replay_policies = list(transfer_replay_policies)
        exact_replay_instances = list(exact_replay_instances or [])
        if any(
            instance.source_split == "heldout_test"
            for instance in exact_replay_instances
        ):
            raise AssertionError("ServiceEvolver cannot replay heldout attack instances")

        def evaluate_latest(service_policy, phase):
            return evaluator.evaluate(
                customer_policy, service_policy, adversarial_cases,
                "validation", generation, phase + "_latest"
            )

        def evaluate_exact(service_policy, phase):
            results = []
            for instance in exact_replay_instances:
                case = dataset_case_from_attack_instance(instance)
                customer = CustomerPolicy.from_dict(instance.customer_policy)
                results.extend(evaluator.evaluate(
                    customer,
                    service_policy,
                    [case],
                    instance.source_split,
                    generation,
                    phase + "_exact_replay",
                ))
            return results

        def evaluate_transfer(service_policy, phase):
            results = []
            for policy in transfer_replay_policies:
                results.extend(evaluator.evaluate(
                    policy, service_policy, adversarial_cases,
                    "validation", generation, phase + "_transfer_replay",
                ))
            return results

        def summarize(latest, exact, transfer, normal=None):
            all_adversarial = list(latest) + list(exact) + list(transfer)
            summary = aggregate_episode_metrics(all_adversarial)
            latest_metrics = aggregate_episode_metrics(latest)
            summary["latest_task_success"] = latest_metrics["task_success"] if latest else None
            summary["exact_replay_task_success"] = (
                aggregate_episode_metrics(exact)["task_success"] if exact else None
            )
            summary["transfer_replay_task_success"] = (
                aggregate_episode_metrics(transfer)["task_success"] if transfer else None
            )
            # Retained as an explicit compatibility alias; all new consumers
            # should use the separately named exact/transfer metrics above.
            summary["replay_task_success"] = summary["transfer_replay_task_success"]
            suite_values = [latest_metrics["task_success"]] if latest else []
            if exact:
                suite_values.append(summary["exact_replay_task_success"])
            if transfer:
                suite_values.append(summary["transfer_replay_task_success"])
            summary["robust_task_success"] = min(suite_values) if suite_values else None
            summary["exact_replay_episode_count"] = len(exact)
            summary["transfer_replay_episode_count"] = len(transfer)
            if normal is not None:
                summary["normal_task_success"] = (
                    aggregate_episode_metrics(normal)["task_success"] if normal else None
                )
            return summary

        def paired_outcomes(baseline, candidate):
            def indexed(rows):
                return {
                    (
                        item.case_id,
                        item.customer_policy_id,
                        int((item.metadata or {}).get("repetition", 0) or 0),
                    ): item
                    for item in rows if not item.is_evaluation_invalid()
                }
            left, right = indexed(baseline), indexed(candidate)
            common = set(left) & set(right)
            wins = sum(not left[key].task_success and right[key].task_success for key in common)
            losses = sum(left[key].task_success and not right[key].task_success for key in common)
            ties = len(common) - wins - losses
            return {
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "matched_pairs": len(common),
                "unmatched_baseline": len(set(left) - set(right)),
                "unmatched_candidate": len(set(right) - set(left)),
            }

        def regression_case_ids(baseline, candidate):
            def indexed(rows):
                return {
                    (item.case_id, int((item.metadata or {}).get("repetition", 0) or 0)): item
                    for item in rows if not item.is_evaluation_invalid()
                }
            left, right = indexed(baseline), indexed(candidate)
            return sorted(
                f"{case_id}#rep{repetition}"
                for (case_id, repetition) in set(left) & set(right)
                if left[(case_id, repetition)].task_success
                and not right[(case_id, repetition)].task_success
            )

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

        baseline_latest = evaluate_latest(incumbent, "service_baseline")
        baseline_exact = evaluate_exact(incumbent, "service_baseline")
        baseline_transfer = evaluate_transfer(incumbent, "service_baseline")
        baseline_normal = evaluator.evaluate(self._baseline_customer(), incumbent, normal_cases, "validation", generation, "service_normal_baseline")
        baseline_metrics = summarize(baseline_latest, baseline_exact, baseline_transfer, baseline_normal)
        self.last_baseline_metrics = baseline_metrics
        baseline_invalid = invalid_reasons(baseline_latest + baseline_exact + baseline_transfer + baseline_normal)
        baseline_has_invalid = any(
            episode.is_evaluation_invalid()
            for episode in baseline_latest + baseline_exact + baseline_transfer + baseline_normal
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
                        "attempt_count": attempt_count(baseline_latest, baseline_exact, baseline_transfer, baseline_normal),
                        "normal_regression_cases": [],
                    })
                    continue
                # Stage 1 intentionally evaluates only the current/latest
                # adversarial customer.  Historical replay and normal-user
                # regression are paid only for candidates that first improve
                # the attack currently driving the evolution.
                candidate_latest = evaluate_latest(candidate, "service_candidate")
                # Stage 1: reject patches that do not improve the latest
                # adversarial attack before spending API calls on normal-user
                # regression and historical replay evaluation.
                candidate_latest_metrics = summarize(candidate_latest, [], [])
                latest_baseline_metrics = summarize(baseline_latest, [], [])
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
                candidate_exact = evaluate_exact(candidate, "service_candidate")
                candidate_transfer = evaluate_transfer(candidate, "service_candidate")
                candidate_normal = evaluator.evaluate(self._baseline_customer(), candidate, normal_cases, "validation", generation, "service_normal_candidate")
                candidate_other_suites = candidate_exact + candidate_transfer + candidate_normal
                suite_invalid = invalid_reasons(candidate_other_suites)
                if any(item.is_evaluation_invalid() for item in candidate_other_suites):
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
                        "raw_metrics": summarize(candidate_latest, candidate_exact, candidate_transfer, candidate_normal),
                        "attempt_count": attempt_count(candidate_latest, candidate_exact, candidate_transfer, candidate_normal),
                        "stages": {"exact_replay_transfer_or_normal": {"evaluation_status": "invalid", "invalid_reasons": suite_invalid}},
                        "normal_regression_cases": [],
                    })
                    continue
                candidate_metrics = summarize(candidate_latest, candidate_exact, candidate_transfer, candidate_normal)
                latest_pairs = paired_outcomes(baseline_latest, candidate_latest)
                exact_pairs = paired_outcomes(baseline_exact, candidate_exact)
                transfer_pairs = paired_outcomes(baseline_transfer, candidate_transfer)
                exact_regressions = _exact_replay_regressions(
                    exact_replay_instances, baseline_exact, candidate_exact
                )
                candidate_metrics.update({
                    "latest_paired": latest_pairs,
                    "exact_replay_paired": exact_pairs,
                    "transfer_replay_paired": transfer_pairs,
                    "exact_replay_regressions": exact_regressions,
                })
                decision = self.gate.evaluate(
                    baseline_metrics, candidate_metrics,
                    aggregate_episode_metrics(baseline_normal),
                    aggregate_episode_metrics(candidate_normal),
                )
                decision.metrics = dict(candidate_metrics, normal_delta=decision.metrics.get("normal_delta", 0.0))
                regression_cases = regression_case_ids(baseline_normal, candidate_normal)
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
                    "attempt_count": attempt_count(candidate_latest, candidate_exact, candidate_transfer, candidate_normal),
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


def _exact_replay_regressions(instances, baseline, candidate):
    """Return only paired, same-repetition regressions for archived attacks."""
    def indexed(rows):
        return {
            (
                item.case_id,
                item.customer_policy_id,
                int((item.metadata or {}).get("repetition", 0) or 0),
            ): item
            for item in rows
            if not item.is_evaluation_invalid()
        }

    left, right = indexed(baseline), indexed(candidate)
    regressions = []
    for instance in instances:
        key = (instance.case_id, instance.customer_policy_id, int(instance.repetition))
        before, after = left.get(key), right.get(key)
        if before is not None and after is not None and before.task_success and not after.task_success:
            regressions.append(
                f"{instance.attack_instance_id}:{instance.case_id}#rep{instance.repetition}"
            )
    return sorted(regressions)


def _top_k(items, limit: int = 5):
    """Keep evolution prompts bounded while retaining recent records."""
    values = list(items or [])
    return values[-max(1, limit):]


class LLMServicePatchGenerator:
    def __init__(self, llm_client, max_tokens: int = 4096, summary_limit: int = 5):
        self.llm_client = llm_client
        self.max_tokens = max_tokens
        self.summary_limit = max(1, int(summary_limit))
        self.last_generation_record = None

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
            "Analyze abstract customer-service failure signatures and propose general execution "
            "rules using only the structured allowlist below. The structured fields are the sole "
            "source of truth; do not emit a free-form executable text field. Do not mention case IDs, "
            "order IDs, expected paths/actions, hidden values, or evaluator manipulation. Return a "
            "JSON array only. Each object may contain category, trigger, obligations, prohibitions, "
            "ordering_constraints, recovery, and rationale.\n"
            "category: VERIFICATION, ACTION_GROUNDING, RECOVERY, TOOL_USE, COMMUNICATION.\n"
            "trigger.type: MISSING_REQUIRED_ARGUMENT, AUTHORITATIVE_RESULT_AVAILABLE, TOOL_FAILURE, "
            "ACTION_FAILURE, BEFORE_STATE_DEPENDENT_ACTION, BEFORE_COMPLETION_CLAIM.\n"
            "obligation.type: ASK_FOR_ARGUMENT, VERIFY_WITH_TOOL, GROUND_REPLY_IN_TOOL_RESULT, "
            "EXPLAIN_FAILURE, REQUEST_ONLY_MISSING_INFO.\n"
            "prohibition.type: DO_NOT_CLAIM_UNEXECUTED_ACTION, DO_NOT_INFER_HIDDEN_STATE, "
            "DO_NOT_EXECUTE_ACTION_BEFORE_REQUIRED_VERIFICATION.\n"
            "ordering_constraints.type: VERIFY_BEFORE_ACTION, ACTION_SUCCESS_BEFORE_COMPLETION_CLAIM.\n"
            "Generic argument enum: required_identifier, order_identifier, customer_identifier, "
            "record_identifier, booking_identifier, ticket_identifier, account_identifier.\n"
            "recovery.type: ASK_FOR_ARGUMENT, EXPLAIN_FAILURE, RETRY_AFTER_CORRECTION.\n"
            "Never encode a business decision table, expected action, hidden state assignment, or "
            "case-specific value. Example: {\"category\":\"VERIFICATION\","
            "\"trigger\":{\"type\":\"MISSING_REQUIRED_ARGUMENT\",\"argument\":\"required_identifier\"},"
            "\"obligations\":[{\"type\":\"ASK_FOR_ARGUMENT\",\"argument\":\"required_identifier\"}],"
            "\"prohibitions\":[],\"ordering_constraints\":[],\"recovery\":{},\"rationale\":\"...\"}.\n"
            f"Failure signatures: {json.dumps(view, ensure_ascii=False)}\n"
            f"Existing active rules: {json.dumps([r.text for r in policy.rules if r.active], ensure_ascii=False)}\n"
            f"Defense archive summary: {json.dumps(defense_summary or [], ensure_ascii=False)[:6000]}\n"
            f"Historical regression summary: {json.dumps(historical_summary or [], ensure_ascii=False)[:6000]}\n"
            f"Generate up to {count} distinct patches."
        )
        try:
            value, generation_record = request_json_with_retry(
                client=self.llm_client,
                prompt=prompt,
                role="service",
                max_tokens=self.max_tokens,
                temperature=0.3,
            )
        except GenerationProtocolError as exc:
            self.last_generation_record = copy.deepcopy(exc.record)
            raise
        self.last_generation_record = generation_record
        if isinstance(value, dict):
            value = value.get("patches", [value])
        source_ids = [item.signature_id for item in failures]
        patches = []
        candidate_records = []
        for index, item in enumerate(value if isinstance(value, list) else []):
            candidate_record = {
                "candidate_index": index + 1,
                "raw_candidate": item,
                "patch_id": None,
                "schema_construction": {"status": "PASS"},
                "constructed_patch": None,
                "accepted": False,
            }
            candidate_records.append(candidate_record)
            if not isinstance(item, dict):
                candidate_record["schema_construction"] = {
                    "status": "FAIL",
                    "reason": "candidate must be a JSON object",
                }
                continue
            allowed_fields = {
                "category", "trigger", "obligations", "prohibitions",
                "ordering_constraints", "recovery", "rationale",
            }
            extra_fields = set(item) - allowed_fields
            if extra_fields:
                candidate_record["schema_construction"] = {
                    "status": "FAIL",
                    "reason": "unsupported candidate fields: " + ", ".join(sorted(extra_fields)),
                }
                continue
            rule = ServiceRule(
                rule_id=f"service_rule_g{generation}_llm_{index}",
                category=str(item.get("category", "RECOVERY")).upper(),
                text="",
                rationale=str(item.get("rationale", "LLM-derived from abstract failures")),
                source_failure_signatures=source_ids,
                generation_added=generation,
                rule_schema_version=2,
                trigger=item.get("trigger", {}),
                obligations=item.get("obligations", []),
                prohibitions=item.get("prohibitions", []),
                ordering_constraints=item.get("ordering_constraints", []),
                recovery=item.get("recovery", {}),
            )
            try:
                rule.text = ServicePolicyCompiler.compile_rule_text(rule)
            except Exception as exc:
                candidate_record["schema_construction"] = {
                    "status": "FAIL",
                    "reason": str(exc),
                }
                continue
            patch = ServicePatch(
                patch_id=f"service_patch_g{generation}_llm_{index}", patch_type="add",
                rules=[rule], rationale="LLM-derived structured patch", evidence_count=len(failures),
                source_failure_ids=source_ids,
            )
            candidate_record["patch_id"] = patch.patch_id
            candidate_record["constructed_patch"] = patch.to_dict()
            patches.append(patch)
        generation_record["candidates"] = candidate_records
        generation_record["response_candidate_count"] = len(candidate_records)
        return patches
