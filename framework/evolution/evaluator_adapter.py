"""Adapters from EvoSAGE episodes to the co-evolution protocol.

The mock evaluator is deliberately deterministic and is used by tests and the
CLI unless ``--real`` is explicitly selected.  It models only a small,
documented vulnerability; it is not presented as a benchmark score.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
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
    # Invalid transport/protocol evaluations are diagnostic records, not
    # zero-valued business outcomes.  Excluding them from the denominator
    # prevents a truncated candidate response from looking like a failed
    # service policy.
    valid_values = [item for item in values if not item.is_evaluation_invalid()]
    legitimate_failures = sum(
        not item.task_success
        for item in valid_values
    )
    transfer_count = sum(
        item.executed_action.lower() in {"transfer_human", "transhuman", "transfer"}
        for item in valid_values
    )
    reject_count = sum(
        item.executed_action.lower() in {"reject", "reject_request"}
        for item in valid_values
    )
    return {
        "task_success": sum(bool(item.task_success) for item in valid_values) / len(valid_values) if valid_values else 0.0,
        "legitimate_attack_success": legitimate_failures / len(valid_values) if valid_values else 0.0,
        "execution_score": _mean(valid_values, "execution_score"),
        "verification": _mean(valid_values, "verification_score"),
        "policy": _mean(valid_values, "policy_score"),
        "action": _mean(valid_values, "action_execution_score"),
        "goal": _mean(valid_values, "goal_fulfillment_score"),
        "episodes": float(len(valid_values)),
        "invalid_episodes": float(len(values) - len(valid_values)),
        "tool_calls": sum(len(item.tool_sequence_summary) for item in valid_values) / len(valid_values) if valid_values else 0.0,
        "transfer_rate": transfer_count / len(valid_values) if valid_values else 0.0,
        "reject_rate": reject_count / len(valid_values) if valid_values else 0.0,
    }


def _exception_invalid_reason(exc: BaseException) -> str:
    """Map transport/provider exceptions to a stable diagnostic reason."""
    customer_reason = getattr(exc, "reason", None)
    if getattr(exc, "customer_simulator_protocol_invalid", False) and customer_reason:
        return str(customer_reason)
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    if "timeout" in name or "timed out" in text or "timeout" in text:
        return "timeout"
    if "json" in name or "parse" in name or "json" in text:
        return "json_parse_failed"
    return "provider_error"


def _response_finish_reason(response: Any) -> str:
    if not isinstance(response, dict):
        return ""
    choices = response.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return ""
    return str(choices[0].get("finish_reason") or "")


def _episode_invalid_reasons(simulation: Any, report: Any) -> list[str]:
    """Identify invalid protocol/provider outcomes without judging business quality."""
    reasons: list[str] = []
    error_categories = set(getattr(report, "error_categories", []) or []) if report else set()
    details = (getattr(report, "details", {}) or {}) if report else {}
    diagnostics = details.get("diagnostics", {}) if isinstance(details, dict) else {}
    if "json_parse_failed" in error_categories or diagnostics.get("json_parse_failed"):
        reasons.append("json_parse_failed")
    if "protocol_failure" in error_categories or diagnostics.get("protocol_failure"):
        reasons.append("protocol_failure")

    turns = getattr(simulation, "turns", []) or []
    for turn in turns:
        output = getattr(turn, "agent_output", None)
        if output is None:
            continue
        metadata = getattr(output, "metadata", {}) or {}
        if metadata.get("protocol_failure"):
            reasons.append(str(metadata.get("invalid_reason") or "protocol_failure"))
        if getattr(output, "json_parse_failed", False):
            reasons.append("json_parse_failed")
        if metadata.get("provider_error") or metadata.get("llm_error"):
            reasons.append("provider_error")
        if metadata.get("timeout") or metadata.get("llm_timeout"):
            reasons.append("timeout")
        request = metadata.get("llm_request", {}) or {}
        if request.get("finish_reason") == "length":
            reasons.append("output_truncated")
        for attempt in metadata.get("llm_attempts", []) or []:
            if attempt.get("finish_reason") == "length":
                reasons.append("output_truncated")
            if attempt.get("invalid_reason"):
                reasons.append(str(attempt["invalid_reason"]))
            raw = attempt.get("raw_provider_response")
            if _response_finish_reason(raw) == "length":
                reasons.append("output_truncated")

    # A few deterministic/unit adapters intentionally represent a valid
    # report without materializing dialogue turns.  Real SimulationResult
    # instances, however, always create a turn before a report is produced;
    # an empty real result is therefore a missing decision/protocol failure.
    is_real_simulation = simulation is not None and simulation.__class__.__name__ == "SimulationResult"
    if not turns and is_real_simulation:
        reasons.append("no_valid_agent_decision")
    elif turns and not any(
        getattr(getattr(turn, "agent_output", None), "classification_output", None) is not None
        or getattr(getattr(turn, "agent_output", None), "final_output", None) is not None
        or getattr(getattr(turn, "agent_output", None), "tool_calls", None)
        for turn in turns
    ):
        reasons.append("no_valid_agent_decision")
    unique = set(reasons)
    priority = (
        "timeout",
        "output_truncated",
        "json_parse_failed",
        "provider_error",
        "no_valid_agent_decision",
        "protocol_failure",
    )
    return [
        reason for reason in priority if reason in unique
    ] + sorted(unique.difference(priority))


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
                 cache_path: str | Path | None = None, reset_cache: bool = False,
                 invalid_evaluation_retries: int = 1):
        self.pipeline_factory = pipeline_factory
        self.user_policy_mode = user_policy_mode
        self.judge_in_evolution = judge_in_evolution
        self.cache_namespace = cache_namespace
        self._episode_cache: dict[str, EpisodeResult] = {}
        self._cache_path = Path(cache_path) if cache_path else None
        self._invalid_path = (
            self._cache_path.with_name("invalid_evaluations.jsonl")
            if self._cache_path else None
        )
        self._reset_cache = reset_cache
        self.invalid_evaluation_retries = max(0, int(invalid_evaluation_retries))
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
            if self._invalid_path is not None:
                self._invalid_path.write_text("", encoding="utf-8")
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
                            episode = EpisodeResult.from_dict(episode_data)
                            # Invalid records are retained in the diagnostic
                            # log, never reused as a substantive cache hit.
                            if not episode.is_evaluation_invalid():
                                self._episode_cache[key] = episode
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

    def _persist_invalid_attempt(self, key: str, episode: EpisodeResult, attempt: int) -> None:
        """Persist invalid attempts without making them cacheable outcomes."""
        if self._invalid_path is None:
            return
        try:
            self._invalid_path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "key": key,
                "attempt": attempt,
                "episode": episode.to_dict(),
            }
            with self._invalid_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except (OSError, TypeError, ValueError):
            return

    @staticmethod
    def _with_phase(episode, phase, cache_hit):
        result = copy.deepcopy(episode)
        result.metadata = dict(result.metadata or {})
        result.metadata["phase"] = phase
        result.metadata["cache_hit"] = cache_hit
        return result

    @staticmethod
    def _invalid_episode(
        customer_policy,
        service_policy,
        case,
        split,
        generation,
        phase,
        reason: str,
        attempt: int,
        error: BaseException | None = None,
    ) -> EpisodeResult:
        """Create a diagnostic-only result for a failed protocol attempt.

        This deliberately contains zero business scores, but callers must not
        aggregate it as a zero-valued task.  ``evaluation_status`` and the
        protocol error labels are the contract that keeps provider failures
        separate from genuine service failures.
        """
        metadata = {
            "phase": phase,
            "split": split,
            "generation": generation,
            "evaluation_status": "invalid",
            "protocol_failure": True,
            "invalid_reason": reason,
            "evaluation_attempt": attempt,
        }
        customer_provenance = getattr(error, "customer_simulator_provenance", None) if error else None
        if customer_provenance is not None:
            metadata["customer_simulator_protocol_invalid"] = True
            metadata["customer_simulator_provenance"] = copy.deepcopy(customer_provenance)
        if error is not None:
            metadata["exception"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
        return EpisodeResult(
            episode_id=f"invalid-{getattr(case, 'case_id', 'case')}-{generation}-{attempt}",
            scenario=getattr(case, "scenario", "ecommerce_refund"),
            case_id=getattr(case, "case_id", "unknown"),
            customer_policy_id=customer_policy.policy_id,
            service_policy_id=service_policy.policy_id,
            split=split,
            generation=generation,
            task_success=False,
            execution_score=0.0,
            error_types=["protocol_failure", reason],
            termination_reason=(
                "customer_simulator_invalid"
                if reason.startswith("customer_simulator_invalid:")
                else "evaluation_invalid"
            ),
            metadata=metadata,
            evaluation_status="invalid",
            invalid_reason=reason,
        )

    @staticmethod
    def _call_pipeline(pipeline, intent: str, run_kwargs: dict[str, Any]):
        """Call old and new pipeline signatures without hiding real errors."""
        method = pipeline.run_single_simulation
        # Resolve compatibility before invoking the method.  This matters for
        # wrappers that record a call and then delegate to an old signature:
        # probing by calling first would create a duplicate provider request.
        try:
            signature = inspect.signature(method)
            parameters = signature.parameters
            accepts_kwargs = any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            )
            if accepts_kwargs and "phase" not in parameters:
                # Do not pass phase through opaque wrappers: a common wrapper
                # accepts **kwargs but delegates to an older concrete method.
                # Passing it once would otherwise trigger duplicate calls
                # during compatibility fallback.
                run_kwargs = dict(run_kwargs)
                run_kwargs.pop("phase", None)
            if not accepts_kwargs:
                run_kwargs = {
                    key: value for key, value in run_kwargs.items()
                    if key in parameters
                }
            return method(intent, **run_kwargs)
        except (TypeError, ValueError) as exc:
            # Some proxy/c-extension callables do not expose a signature;
            # retain the conservative fallback for those only.
            if not isinstance(exc, TypeError) or "unexpected keyword" not in str(exc):
                raise
        candidates = [
            dict(run_kwargs),
            {key: value for key, value in run_kwargs.items() if key != "phase"},
            {key: value for key, value in run_kwargs.items() if key not in {"phase", "judge_enabled"}},
        ]
        last_error = None
        for kwargs in candidates:
            try:
                return method(intent, **kwargs)
            except TypeError as exc:
                last_error = exc
                text = str(exc)
                # Only retry signature compatibility errors.  A TypeError
                # raised inside the provider/pipeline is a real invalid
                # evaluation and must reach the protocol classifier.
                optional = [name for name in ("phase", "judge_enabled") if name in kwargs]
                if not optional or not any(
                    marker in text for marker in ("unexpected keyword", "got an unexpected keyword", "positional")
                ):
                    raise
        raise last_error  # pragma: no cover - defensive; candidates are non-empty

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
                "phase": phase,
            }
            max_attempts = 1 + self.invalid_evaluation_retries
            invalid_attempts = []
            episode = None
            for attempt in range(1, max_attempts + 1):
                simulation = None
                report = None
                terminal_customer_invalid = False
                try:
                    simulation, report = self._call_pipeline(
                        pipeline,
                        getattr(case, "intent", "refund_before_shipping"),
                        run_kwargs,
                    )
                    episode = self.from_evosage(
                        simulation, report, customer_policy, service_policy, split, generation, phase,
                        path_config=getattr(case, "path_config", None),
                    )
                except Exception as exc:
                    reason = _exception_invalid_reason(exc)
                    terminal_customer_invalid = bool(
                        getattr(exc, "customer_simulator_protocol_invalid", False)
                    )
                    episode = self._invalid_episode(
                        customer_policy, service_policy, case, split, generation, phase,
                        reason, attempt, error=exc,
                    )

                episode.metadata = dict(episode.metadata or {})
                episode.metadata["cache_hit"] = False
                if episode.is_evaluation_invalid():
                    reasons = []
                    if episode.invalid_reason:
                        reasons.append(episode.invalid_reason)
                    reasons.extend(
                        reason for reason in episode.metadata.get("invalid_reasons", [])
                        if reason
                    )
                    if not reasons:
                        reasons.append("protocol_failure")
                    reasons = list(dict.fromkeys(reasons))
                    episode.evaluation_status = "invalid"
                    episode.invalid_reason = reasons[0]
                    episode.metadata.update({
                        "evaluation_status": "invalid",
                        "invalid_reason": reasons[0],
                        "invalid_reasons": reasons,
                        "evaluation_attempt": attempt,
                    })
                    invalid_attempts.append({
                        "attempt": attempt,
                        "reason": reasons[0],
                        "reasons": reasons,
                    })
                    self._persist_invalid_attempt(key, episode, attempt)
                    if attempt < max_attempts and not terminal_customer_invalid:
                        continue
                break

            assert episode is not None
            episode.metadata["evaluation_attempts"] = len(invalid_attempts) + (
                1 if not episode.is_evaluation_invalid() else 0
            )
            episode.metadata["invalid_attempts"] = invalid_attempts
            if not episode.is_evaluation_invalid():
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
        # BackendEnvironment records both query and action tools as ``tool_call``
        # events.  Keep the co-evolution trace aligned with the execution
        # evaluator; filtering for the old ``tool_query`` name silently turned
        # every real episode into an empty tool trace.
        tools = [
            event.get("name", "")
            for event in getattr(simulation, "backend_events", [])
            if event.get("event_type") == "tool_call"
        ]
        errors = list(getattr(report, "error_categories", []) or [])
        diagnostics = getattr(report, "details", {}).get("diagnostics", {}) if getattr(report, "details", None) else {}
        if diagnostics.get("json_parse_failed") and "json_parse_failed" not in errors:
            errors.append("json_parse_failed")
        invalid_reasons = _episode_invalid_reasons(simulation, report)
        for reason in invalid_reasons:
            if reason not in errors:
                errors.append(reason)
        protocol_failure = bool(invalid_reasons) or (
            "json_parse_failed" in errors
            or "protocol_failure" in errors
            or bool(diagnostics.get("protocol_failure", False))
        )
        evaluation_status = "invalid" if protocol_failure else "valid"
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
                "evaluation_status": evaluation_status,
                "invalid_reason": invalid_reasons[0] if invalid_reasons else None,
                "invalid_reasons": invalid_reasons,
                "failure_location": location,
                "customer_simulator_provenance": copy.deepcopy(
                    getattr(simulation, "customer_simulator_provenance", []) or []
                ),
            },
            evaluation_status=evaluation_status,
            invalid_reason=invalid_reasons[0] if invalid_reasons else None,
        )
