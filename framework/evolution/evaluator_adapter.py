"""Adapters from EvoSAGE episodes to the co-evolution protocol.

The mock evaluator is deliberately deterministic and is used by tests and the
CLI unless ``--real`` is explicitly selected.  It models only a small,
documented vulnerability; it is not presented as a benchmark score.
"""

from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable, Protocol

from .attribution import infer_failure_location
from .schemas import CustomerPolicy, EpisodeResult, FailureSignature, ServicePolicy


class EpisodeEvaluator(Protocol):
    def evaluate(self, customer_policy: CustomerPolicy, service_policy: ServicePolicy,
                 cases: Iterable[Any], split: str, generation: int, phase: str) -> list[EpisodeResult]: ...


def _mean(items: list[EpisodeResult], field: str) -> float:
    return sum(float(getattr(item, field, 0.0)) for item in items) / len(items) if items else 0.0


def aggregate_episode_metrics(episodes: Iterable[EpisodeResult]) -> dict[str, float]:
    values = list(episodes)
    legitimate_failures = sum(
        not item.task_success
        and "json_parse_failed" not in item.error_types
        and "protocol_failure" not in item.error_types
        and not item.metadata.get("protocol_failure", False)
        for item in values
    )
    transfer_count = sum(
        item.executed_action.lower() in {"transfer_human", "transhuman", "transfer"}
        for item in values
    )
    reject_count = sum(
        item.executed_action.lower() in {"reject", "reject_request"}
        for item in values
    )
    return {
        "task_success": sum(bool(item.task_success) for item in values) / len(values) if values else 0.0,
        "legitimate_attack_success": legitimate_failures / len(values) if values else 0.0,
        "execution_score": _mean(values, "execution_score"),
        "verification": _mean(values, "verification_score"),
        "policy": _mean(values, "policy_score"),
        "action": _mean(values, "action_execution_score"),
        "goal": _mean(values, "goal_fulfillment_score"),
        "episodes": float(len(values)),
        "tool_calls": sum(len(item.tool_sequence_summary) for item in values) / len(values) if values else 0.0,
        "transfer_rate": transfer_count / len(values) if values else 0.0,
        "reject_rate": reject_count / len(values) if values else 0.0,
    }


class BudgetedEpisodeEvaluator:
    """Apply repetitions, concurrency and held-out isolation consistently."""

    def __init__(self, evaluator, repetitions: int = 1, concurrency: int = 1):
        self.evaluator = evaluator
        self.repetitions = max(1, int(repetitions))
        self.concurrency = max(1, int(concurrency))

    @staticmethod
    def _assert_evolution_scope(cases, phase: str) -> None:
        protected_phases = (
            "candidate", "customer", "service", "baseline", "normal", "replay",
            "failure_scan", "generation_summary", "gate", "evolution",
        )
        if any(getattr(case, "split", "") == "heldout_test" for case in cases) and any(
            token in phase.lower() for token in protected_phases
        ):
            raise AssertionError("heldout cases cannot enter an evolution or service gate phase")

    def evaluate(self, customer_policy, service_policy, cases, split, generation, phase):
        cases = list(cases)
        self._assert_evolution_scope(cases, phase)
        outputs = []
        for repetition in range(self.repetitions):
            if self.concurrency > 1 and len(cases) > 1:
                def one(case):
                    return self.evaluator.evaluate(customer_policy, service_policy, [case], split, generation, phase)
                with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                    chunks = list(pool.map(one, cases))
                repetition_outputs = [item for chunk in chunks for item in chunk]
            else:
                repetition_outputs = self.evaluator.evaluate(
                    customer_policy, service_policy, cases, split, generation, phase
                )
            for item in repetition_outputs:
                item.metadata = dict(item.metadata)
                item.metadata["repetition"] = repetition
            outputs.extend(repetition_outputs)
        return outputs


class MockEpisodeEvaluator:
    """Offline fixture for the complete two-generation integration test."""

    def __init__(self, vulnerability_tags: tuple[str, ...] = ("authority_challenge", "delayed_contradiction")):
        self.vulnerability_tags = set(vulnerability_tags)
        self.calls: list[dict[str, Any]] = []

    def evaluate(self, customer_policy: CustomerPolicy, service_policy: ServicePolicy,
                 cases: Iterable[Any], split: str, generation: int, phase: str) -> list[EpisodeResult]:
        cases = list(cases)
        self.calls.append({"customer_policy_id": customer_policy.policy_id, "service_policy_id": service_policy.policy_id,
                           "split": split, "generation": generation, "phase": phase, "case_count": len(cases)})
        active_rules = " ".join(rule.text.lower() for rule in service_policy.rules if rule.active)
        defended = any(token in active_rules for token in ("authoritative", "action tool", "verify before", "execute only"))
        vulnerable = bool(self.vulnerability_tags & set(customer_policy.strategy_tags)) and not defended
        results = []
        for index, case in enumerate(cases):
            case_id = getattr(case, "case_id", f"case_{index}")
            path_id = getattr(case, "path_id", None)
            error_types = [] if not vulnerable else ["authoritative_conflict" if "authority_challenge" in customer_policy.strategy_tags else "claimed_action_not_executed"]
            success = not vulnerable
            results.append(EpisodeResult(
                episode_id=f"mock-g{generation}-{split}-{index}",
                scenario="ecommerce_refund",
                case_id=case_id,
                customer_policy_id=customer_policy.policy_id,
                service_policy_id=service_policy.policy_id,
                split=split,
                generation=generation,
                task_success=success,
                execution_score=1.0 if success else 0.4,
                sage_style_score=1.0 if success else 0.6,
                verification_score=1.0 if success else 0.0,
                policy_score=1.0 if success else 0.0,
                action_execution_score=1.0 if success else 0.0,
                goal_fulfillment_score=1.0 if success else 0.0,
                error_types=error_types,
                predicted_action="Refund" if success else "Reject",
                executed_action="Refund" if success else "",
                tool_sequence_summary=["query_order", "submit_refund"] if success else ["query_order"],
                termination_reason="goal_satisfied" if success else "mock_adversarial_failure",
                sop_node=f"path_{path_id}" if path_id is not None else None,
                path_step_index=0,
                metadata={"mock": True, "phase": phase},
            ))
        return results


class CallableEpisodeEvaluator:
    def __init__(self, callback: Callable[..., list[EpisodeResult]]):
        self.callback = callback

    def evaluate(self, customer_policy, service_policy, cases, split, generation, phase):
        return self.callback(customer_policy, service_policy, cases, split, generation, phase)


class EvoSAGEEpisodeEvaluator:
    """Thin real-run adapter; imports the legacy runner lazily."""

    def __init__(self, pipeline_factory: Callable[..., Any], user_policy_mode: str = "truthful"):
        self.pipeline_factory = pipeline_factory
        self.user_policy_mode = user_policy_mode

    def evaluate(self, customer_policy, service_policy, cases, split, generation, phase):
        outputs = []
        # Policies are pipeline-construction state.  Older factories that do
        # not accept them remain supported for callers that already bind the
        # policies in a closure.
        try:
            pipeline = self.pipeline_factory(customer_policy, service_policy)
        except TypeError as exc:
            if "positional" not in str(exc) and "argument" not in str(exc):
                raise
            pipeline = self.pipeline_factory()
        for case in cases:
            simulation, report = pipeline.run_single_simulation(
                getattr(case, "intent", "refund_before_shipping"),
                user_id=f"{getattr(case, 'case_id', 'case')}_{generation}",
                path_config=getattr(case, "path_config", None),
            )
            outputs.append(self.from_evosage(
                simulation, report, customer_policy, service_policy, split, generation, phase,
                path_config=getattr(case, "path_config", None),
            ))
        return outputs

    @staticmethod
    def from_evosage(simulation, report, customer_policy, service_policy, split, generation, phase,
                     path_config=None):
        tools = [event.get("name", "") for event in getattr(simulation, "backend_events", []) if event.get("event_type") == "tool_query"]
        errors = list(getattr(report, "error_categories", []) or [])
        diagnostics = getattr(report, "details", {}).get("diagnostics", {}) if getattr(report, "details", None) else {}
        if diagnostics.get("json_parse_failed") and "json_parse_failed" not in errors:
            errors.append("json_parse_failed")
        protocol_failure = (
            "json_parse_failed" in errors
            or "protocol_failure" in errors
            or bool(diagnostics.get("protocol_failure", False))
        )
        location = infer_failure_location(report, simulation, path_config)
        return EpisodeResult(
            episode_id=simulation.simulation_id,
            scenario=simulation.scenario_id,
            case_id=simulation.case_spec.get("case_id", simulation.simulation_id) if simulation.case_spec else simulation.simulation_id,
            customer_policy_id=customer_policy.policy_id,
            service_policy_id=service_policy.policy_id,
            split=split,
            generation=generation,
            task_success=bool(report.task_success),
            execution_score=float(report.execution_score),
            sage_style_score=float(report.sage_style_score),
            verification_score=float(report.required_verification_score),
            policy_score=float(report.policy_compliance_score),
            action_execution_score=float(report.action_execution_score),
            goal_fulfillment_score=float(report.goal_fulfillment),
            error_types=errors,
            predicted_action=report.predicted_action,
            executed_action=report.executed_action,
            tool_sequence_summary=tools,
            termination_reason=simulation.termination_reason,
            sop_node=location.get("sop_node"),
            path_step_index=location.get("path_step_index"),
            dialogue=[turn.agent_output.to_dict() for turn in simulation.turns],
            metadata={
                "phase": phase,
                "model_name": simulation.model_name,
                "customer_policy_id": customer_policy.policy_id,
                "service_policy_id": service_policy.policy_id,
                "split": split,
                "generation": generation,
                "protocol_failure": protocol_failure,
                "failure_location": location,
            },
        )
