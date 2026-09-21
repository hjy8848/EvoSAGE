"""Adapters from EvoSAGE episodes to the co-evolution protocol.

The mock evaluator is deliberately deterministic and is used by tests and the
CLI unless ``--real`` is explicitly selected.  It models only a small,
documented vulnerability; it is not presented as a benchmark score.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
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

    _FAST_PHASES = frozenset({
        "customer_failure_scan",
        "customer_candidate",
        "customer_elite",
        "selected_customer",
        "service_failures",
        "service_baseline_latest",
        "service_baseline_replay",
        "service_normal_baseline",
        "service_candidate_latest",
        "service_candidate_replay",
        "service_normal_candidate",
        "generation_summary",
        "fresh_adaptation",
    })

    def __init__(self, pipeline_factory: Callable[..., Any], user_policy_mode: str = "truthful",
                 judge_in_evolution: bool = False, cache_namespace: str = "default",
                 cache_path: str | Path | None = None, reset_cache: bool = False):
        self.pipeline_factory = pipeline_factory
        self.user_policy_mode = user_policy_mode
        self.judge_in_evolution = judge_in_evolution
        self.cache_namespace = cache_namespace
        self._episode_cache: dict[str, EpisodeResult] = {}
        self._cache_path = Path(cache_path) if cache_path else None
        self._reset_cache = reset_cache
        self._cache_lock = threading.Lock()
        self._pipelines = []
        self._pipeline_lock = threading.Lock()
        self.cache_hits = 0
        self.cache_misses = 0
        self.real_episode_count = 0
        self._reset_persistent_cache()
        self._load_persistent_cache()

    def _reset_persistent_cache(self) -> None:
        if self._cache_path is None or not self._reset_cache:
            return
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text("", encoding="utf-8")
        except OSError:
            # Cache persistence is an optimization and must not block a fresh
            # run if the old checkpoint cannot be truncated.
            return

    def _use_llm_judge(self, phase: str) -> bool:
        if self.judge_in_evolution:
            return True
        return phase not in self._FAST_PHASES

    def _cache_key(self, customer_policy, service_policy, case, split, generation, judge_enabled):
        payload = {
            "version": "episode-cache-v2",
            "namespace": self.cache_namespace,
            "customer_policy_id": customer_policy.policy_id,
            "service_policy_id": service_policy.policy_id,
            "customer_policy": self._fingerprint(customer_policy),
            "service_policy": self._fingerprint(service_policy),
            "case_id": getattr(case, "case_id", None),
            "intent": getattr(case, "intent", None),
            # Do not persist path_config itself: it can contain hidden
            # backend truth.  The digest still prevents cross-case reuse.
            "path_config_digest": self._fingerprint(getattr(case, "path_config", None)),
            "split": split,
            "generation": generation,
            "judge_enabled": judge_enabled,
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)

    @staticmethod
    def _fingerprint(value) -> str:
        if hasattr(value, "to_dict"):
            value = value.to_dict()
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _load_persistent_cache(self) -> None:
        if self._cache_path is None or not self._cache_path.exists():
            return
        try:
            with self._cache_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        value = json.loads(line)
                        key = value.get("key")
                        episode_data = value.get("episode")
                        if key and isinstance(episode_data, dict):
                            self._episode_cache[key] = EpisodeResult.from_dict(episode_data)
                    except (TypeError, ValueError, json.JSONDecodeError):
                        # A truncated final JSONL line must not invalidate the
                        # completed episodes that precede it.
                        continue
        except OSError:
            # Disk caching is an optimization; an unavailable cache must not
            # prevent a fresh evaluation from running.
            return

    def _persist_episode(self, key: str, episode: EpisodeResult) -> None:
        if self._cache_path is None:
            return
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self._cache_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"key": key, "episode": episode.to_dict()}, ensure_ascii=False) + "\n")
        except (OSError, TypeError, ValueError):
            # Keep the episode result usable even if checkpoint storage is
            # unavailable or a provider returned non-serializable metadata.
            return

    @staticmethod
    def _with_phase(episode, phase, cache_hit):
        result = copy.deepcopy(episode)
        result.metadata = dict(result.metadata or {})
        result.metadata["phase"] = phase
        result.metadata["cache_hit"] = cache_hit
        return result

    def evaluate(self, customer_policy, service_policy, cases, split, generation, phase):
        cases = list(cases)
        judge_enabled = self._use_llm_judge(phase)
        outputs = [None] * len(cases)
        missing = []
        for index, case in enumerate(cases):
            key = self._cache_key(customer_policy, service_policy, case, split, generation, judge_enabled)
            with self._cache_lock:
                cached = self._episode_cache.get(key)
            if cached is None:
                missing.append((index, case, key))
                self.cache_misses += 1
            else:
                outputs[index] = self._with_phase(cached, phase, cache_hit=True)
                self.cache_hits += 1

        if not missing:
            return outputs

        # Policies are pipeline-construction state.  Older factories that do
        # not accept them remain supported for callers that already bind the
        # policies in a closure.
        try:
            pipeline = self.pipeline_factory(customer_policy, service_policy, judge_enabled=judge_enabled)
        except TypeError as exc:
            if "positional" not in str(exc) and "argument" not in str(exc):
                raise
            try:
                pipeline = self.pipeline_factory(customer_policy, service_policy)
            except TypeError as legacy_exc:
                if "positional" not in str(legacy_exc) and "argument" not in str(legacy_exc):
                    raise
                pipeline = self.pipeline_factory()
        with self._pipeline_lock:
            self._pipelines.append(pipeline)
        for index, case, key in missing:
            run_kwargs = {
                "user_id": f"{getattr(case, 'case_id', 'case')}_{generation}",
                "path_config": getattr(case, "path_config", None),
                "judge_enabled": judge_enabled,
            }
            try:
                simulation, report = pipeline.run_single_simulation(
                    getattr(case, "intent", "refund_before_shipping"), **run_kwargs
                )
            except TypeError as exc:
                if "judge_enabled" not in str(exc):
                    raise
                run_kwargs.pop("judge_enabled")
                simulation, report = pipeline.run_single_simulation(
                    getattr(case, "intent", "refund_before_shipping"), **run_kwargs
                )
            episode = self.from_evosage(
                simulation, report, customer_policy, service_policy, split, generation, phase,
                path_config=getattr(case, "path_config", None),
            )
            episode.metadata["cache_hit"] = False
            cached_episode = copy.deepcopy(episode)
            with self._cache_lock:
                self._episode_cache[key] = cached_episode
                self._persist_episode(key, cached_episode)
            self.real_episode_count += 1
            outputs[index] = episode
        return outputs

    def get_stats(self) -> dict[str, Any]:
        stats = {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "real_episode_count": 0,
            "user_requests": 0,
            "agent_requests": 0,
            "judge_requests": 0,
            "pipeline_requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_seconds": 0.0,
            "retries": 0,
            "attempts": 0,
            "successes": 0,
            "failures": 0,
            "timeouts": 0,
            "total_attempt_latency": 0.0,
            "max_attempt_latency": 0.0,
        }
        for role in ("user", "agent", "judge"):
            stats[f"{role}_input_tokens"] = 0
            stats[f"{role}_output_tokens"] = 0
            stats[f"{role}_latency_seconds"] = 0.0
        with self._pipeline_lock:
            pipelines = list(self._pipelines)
        for pipeline in pipelines:
            for role, key in (("user", "user_requests"), ("agent", "agent_requests"), ("judge", "judge_requests")):
                client = getattr(pipeline, f"{role}_llm_client", None)
                client_stats = client.request_stats() if hasattr(client, "request_stats") else {}
                count = int(client_stats.get("requests", getattr(client, "request_count", 0)) or 0)
                stats[key] += count
                stats["pipeline_requests"] += count
                stats["retries"] += int(client_stats.get("retries", getattr(client, "retry_count", 0)) or 0)
                stats["attempts"] += int(client_stats.get("attempts", 0) or 0)
                stats["successes"] += int(client_stats.get("successes", 0) or 0)
                stats["failures"] += int(client_stats.get("failures", 0) or 0)
                stats["timeouts"] += int(client_stats.get("timeouts", 0) or 0)
                stats["input_tokens"] += int(client_stats.get("input_tokens", 0) or 0)
                stats["output_tokens"] += int(client_stats.get("output_tokens", 0) or 0)
                stats["latency_seconds"] += float(client_stats.get("latency_seconds", 0.0) or 0.0)
                stats["total_attempt_latency"] += float(client_stats.get("total_attempt_latency", 0.0) or 0.0)
                stats["max_attempt_latency"] = max(
                    stats["max_attempt_latency"],
                    float(client_stats.get("max_attempt_latency", 0.0) or 0.0),
                )
                stats[f"{role}_input_tokens"] += int(client_stats.get("input_tokens", 0) or 0)
                stats[f"{role}_output_tokens"] += int(client_stats.get("output_tokens", 0) or 0)
                stats[f"{role}_latency_seconds"] += float(client_stats.get("latency_seconds", 0.0) or 0.0)
            stats["real_episode_count"] += int(getattr(pipeline, "_completed_episode_count", 0) or 0)
        # The pipeline list is also a place to collect completed runs, but a
        # pipeline can be reused for multiple cases.  Prefer the adapter's
        # authoritative count when available.
        stats["real_episode_count"] = getattr(self, "real_episode_count", stats["real_episode_count"])
        return stats

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
