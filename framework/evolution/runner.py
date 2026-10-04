"""Orchestrator for static, customer-only, service-only and coevolution modes."""

from __future__ import annotations

from dataclasses import asdict
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Optional

from .archives import AttackArchive, DefenseArchive
from .config import EvolutionConfig
from .customer_evolver import CustomerEvolver
from .legacy.customer_policy import CustomerPolicyValidator
from .customer_selector import CandidateScore, CustomerSelector
from .evaluator_adapter import BudgetedEpisodeEvaluator, MockEpisodeEvaluator, aggregate_episode_metrics
from .generation_protocol import GenerationProtocolError
from .persistence import RunStore, has_prior_run_state, resolve_run_dir
from .reporting import generate_report
from .schemas import (
    CustomerPolicy,
    DefenseRecord,
    FailureOccurrence,
    ServicePatch,
    ServicePolicy,
)
from .service_evolver import ServiceEvolver
from .service_policy import ServicePolicySanitizer
from .service_gate import ServiceGate
from .service_gate import GateDecision
from .split_manager import SplitManager
from .weakness_frontier import WeaknessFrontier
from .request_budget import APIRequestBudgetExceeded


class GenerationEvaluationInconclusive(RuntimeError):
    """Raised when a generation has no valid episode evidence to evaluate."""

    inconclusive = True


class EvolutionRunner:
    def __init__(self, config: Optional[EvolutionConfig] = None, evaluator=None, store: Optional[RunStore] = None,
                 split_manager: Optional[SplitManager] = None, customer_evolver=None, service_evolver=None,
                 run_dir: str | Path | None = None):
        self.config = config or EvolutionConfig()
        self.requested_output_dir = Path(self.config.persistence.output_dir)
        self.fresh_run_isolated = False
        if store is None:
            resolved_run_dir = Path(run_dir) if run_dir is not None else self.resolve_run_dir(self.config)[0]
            self.fresh_run_isolated = resolved_run_dir != self.requested_output_dir
            self.store = RunStore(resolved_run_dir)
        else:
            self.store = store
        self.split_manager = split_manager or SplitManager(
            self.config.splits,
            self.store.run_dir / "split_manifest",
            scenario=self.config.scenario,
        )
        self.base_evaluator = evaluator or MockEpisodeEvaluator()
        self.request_budget = getattr(self.base_evaluator, "request_budget", None)
        self.evaluator = BudgetedEpisodeEvaluator(
            self.base_evaluator,
            repetitions=self.config.evaluation.repetitions,
            concurrency=self.config.evaluation.concurrency,
        )
        self.customer_evolver = customer_evolver or CustomerEvolver(
            self.config.seed,
            validator=CustomerPolicyValidator(),
            selector=CustomerSelector(),
        )
        self.service_evolver = service_evolver or ServiceEvolver(
            self.config.seed,
            sanitizer=ServicePolicySanitizer(self.config.service.allowed_rule_categories),
            gate=ServiceGate(
                self.config.service.min_delta,
                self.config.service.normal_regression_tolerance,
                self.config.service.gate_min_paired_wins,
                self.config.service.max_normal_paired_losses,
            ),
        )
        # Stage 1 deliberately avoids cross-generation archives: Customer
        # candidates compete on official outcomes against the fixed Service S0.
        customer_only = self.config.experiment_mode == "customer_only"
        self.attack_archive = (
            None if customer_only
            else AttackArchive(self.store.run_dir / "archives" / "attacks.jsonl")
        )
        self.defense_archive = (
            None if customer_only
            else DefenseArchive(self.store.run_dir / "archives" / "defenses.jsonl")
        )
        self.frontier = WeaknessFrontier()
        self._generation_wall_times: dict[str, float] = {}
        self._active_generation: Optional[int] = None
        self._active_stage = "initializing"
        self._completed_phases: set[str] = set()
        self._resume_checkpoint: Optional[dict[str, Any]] = None
        self._generation_baseline_customer_policy: Optional[CustomerPolicy] = None
        self._generation_baseline_service_policy: Optional[ServicePolicy] = None

    def _checkpoint_path(self, generation: int) -> Path:
        return self.store.run_dir / "generations" / f"gen_{generation:03d}" / "CHECKPOINT.json"

    def _load_incomplete_checkpoint(self, generation: int) -> Optional[dict[str, Any]]:
        if not self.config.persistence.resume or generation in self.store.completed_generations():
            return None
        path = self._checkpoint_path(generation)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if int(value.get("generation", -1)) != generation or str(value.get("status", "")).startswith("complete"):
            return None
        return value

    def _write_incomplete_checkpoint(self, status: str = "incomplete", **details) -> None:
        generation = self._active_generation
        if generation is None:
            return
        customer = getattr(self, "_active_customer_policy", None)
        service = getattr(self, "_active_service_policy", None)
        checkpoint = {
            "generation": generation,
            "status": status,
            "stage": self._active_stage,
            "completed_phases": sorted(self._completed_phases),
            "completed_generations": self.store.completed_generations(),
            "active_customer_policy": customer.to_dict() if customer is not None else None,
            "active_service_policy": service.to_dict() if service is not None else None,
            "generation_baseline_customer_policy": (
                self._generation_baseline_customer_policy.to_dict()
                if self._generation_baseline_customer_policy is not None else None
            ),
            "generation_baseline_service_policy": (
                self._generation_baseline_service_policy.to_dict()
                if self._generation_baseline_service_policy is not None else None
            ),
            # Keep these aliases readable by tooling created before this checkpoint schema.
            "customer_policy": customer.to_dict() if customer is not None else None,
            "service_policy": service.to_dict() if service is not None else None,
            "episode_cache": (
                "environment/episode_cache.jsonl"
                if (self.store.run_dir / "environment/episode_cache.jsonl").exists()
                else None
            ),
        }
        checkpoint.update(details)
        self.store.write_json(
            f"generations/gen_{generation:03d}/CHECKPOINT.json", checkpoint
        )

    def _mark_phase_complete(self, phase: str, stage: Optional[str] = None) -> None:
        self._completed_phases.add(phase)
        if stage:
            self._active_stage = stage
        self._write_incomplete_checkpoint()

    def _persist_evolver_record(self, generation: int, kind: str, **extra) -> None:
        """Persist structured-generation provenance in the authoritative run dir."""
        evolver = self.customer_evolver if kind == "customer" else self.service_evolver
        record = getattr(evolver, "last_generation_record", None)
        if not record:
            return
        payload = json.loads(json.dumps(record, ensure_ascii=False))
        payload["generation"] = generation
        payload.update(extra)
        self.store.write_json(
            f"generations/gen_{generation:03d}/{kind}_generation.json",
            payload,
        )

    def _persist_pending_evolver_records(self) -> None:
        if self._active_generation is None:
            return
        self._persist_evolver_record(self._active_generation, "customer")
        self._persist_evolver_record(self._active_generation, "service")

    def _git_provenance(self) -> dict[str, Any]:
        """Resolve configured freeze tags against the exact checked-out commit."""
        repository_root = Path(__file__).resolve().parents[2]

        def git_value(*args: str) -> Optional[str]:
            try:
                result = subprocess.run(
                    ["git", *args], cwd=repository_root, check=False,
                    capture_output=True, text=True, timeout=5,
                )
            except (OSError, subprocess.TimeoutExpired):
                return None
            value = result.stdout.strip()
            return value if result.returncode == 0 else None

        metadata = self.config.model_metadata
        commit_sha = git_value("rev-parse", "HEAD")
        runtime_tag = metadata.get("runtime_freeze_tag")
        protocol_tag = metadata.get("formal_protocol_tag")

        def tag_commit(tag: Optional[str]) -> Optional[str]:
            if not isinstance(tag, str) or not tag.strip():
                return None
            return git_value("rev-parse", "--verify", f"refs/tags/{tag.strip()}^{{commit}}")

        runtime_tag_commit = tag_commit(runtime_tag)
        protocol_tag_commit = tag_commit(protocol_tag)
        worktree_status = git_value("status", "--porcelain")
        runtime_tag_observed = runtime_tag if runtime_tag_commit == commit_sha and commit_sha else None
        if not runtime_tag or not commit_sha or not runtime_tag_commit:
            runtime_match = "not_confirmed"
        else:
            runtime_match = "matched" if runtime_tag_commit == commit_sha else "mismatch"
        if not protocol_tag or not commit_sha or not protocol_tag_commit:
            protocol_match = "not_confirmed"
        else:
            protocol_match = "matched" if protocol_tag_commit == commit_sha else "mismatch"

        return {
            "commit_sha": commit_sha,
            "runtime_commit_sha": commit_sha,
            "runtime_freeze_commit": runtime_tag_commit,
            "runtime_freeze_tag": runtime_tag,
            "observed_runtime_freeze_commit": commit_sha,
            "observed_runtime_freeze_tag": runtime_tag_observed,
            "runtime_freeze_match_status": runtime_match,
            "formal_protocol_commit": protocol_tag_commit,
            "formal_protocol_tag": protocol_tag,
            "formal_protocol_match_status": protocol_match,
            "git_worktree_clean": None if worktree_status is None else worktree_status == "",
        }

    def _split_manifest_provenance(self) -> dict[str, Any]:
        manifest_dir = self.store.run_dir / "split_manifest"
        manifest = {
            "strategy": self.config.splits.strategy,
            "seed": self.config.splits.seed,
            "files": {},
        }
        for filename in ("evolution_cases.json", "validation_cases.json", "heldout_cases.json"):
            path = manifest_dir / filename
            if not path.exists():
                continue
            raw = path.read_bytes()
            content = json.loads(raw)
            manifest["files"][filename] = {
                "path": f"split_manifest/{filename}",
                "sha256": hashlib.sha256(raw).hexdigest(),
                "case_ids": [item.get("case_id") for item in content.get("cases", [])],
            }
        return manifest

    def _estimate_generation_workload(self, splits) -> dict[str, Any]:
        """A deliberately rough planning estimate; hard budgets remain authoritative."""
        cfg = self.config
        evolution_count = len(splits.evolution)
        validation_count = len(splits.validation)
        mode = cfg.experiment_mode
        episode_runs = validation_count  # generation summary
        evolver_calls = 0
        if mode in {"customer_only", "coevolution"}:
            candidate_cases = (
                evolution_count if mode == "customer_only"
                else min(evolution_count, max(1, cfg.customer.cases_per_candidate))
            )
            episode_runs += evolution_count  # baseline failure scan
            episode_runs += cfg.customer.candidate_count * candidate_cases
            episode_runs += evolution_count  # selected customer confirmation
            evolver_calls += 1
        if mode in {"service_only", "coevolution"}:
            episode_runs += evolution_count  # service failure scan
            service_cases_per_candidate = (
                2 * validation_count
                + cfg.service.exact_replay_instance_count
                + int(cfg.service.transfer_replay_policy_count or 0)
            )
            episode_runs += cfg.service.candidate_count * service_cases_per_candidate
            evolver_calls += 1
        assumed_user_calls = 2
        assumed_agent_calls = 3
        per_episode = assumed_user_calls + assumed_agent_calls
        requests = episode_runs * per_episode + evolver_calls
        upper_calls_per_episode = (
            (cfg.evaluation.max_turns + 1)
            + cfg.evaluation.max_turns * (cfg.evaluation.max_tool_steps + 1)
        )
        return {
            "estimate_type": "rough_expected_not_guaranteed",
            "mode": mode,
            "evolution_cases": evolution_count,
            "validation_cases": validation_count,
            "estimated_episode_runs_per_generation": episode_runs,
            "assumptions": {
                "user_requests_per_episode": assumed_user_calls,
                "agent_requests_per_episode": assumed_agent_calls,
                "evolver_generation_requests_per_generation": evolver_calls,
                "transport_attempts_per_logical_request": 1,
                "max_possible_user_and_agent_calls_per_episode": upper_calls_per_episode,
            },
            "estimated_provider_requests_per_generation": requests,
            "estimated_provider_requests_total": requests * cfg.max_generations,
            "configured_generation_hard_limit": cfg.evaluation.max_api_requests_per_generation,
            "configured_run_hard_limit": cfg.evaluation.max_api_requests_per_run,
        }

    @classmethod
    def resolve_run_dir(cls, config: EvolutionConfig) -> tuple[Path, bool]:
        """Resolve the one authoritative directory for an experiment.

        Real evaluators create their cache during construction, so callers
        must resolve the directory before constructing either the evaluator or
        the runner.  Returning the isolation flag also lets the runner record
        provenance without re-resolving and accidentally creating a second
        ``*_fresh_*`` directory.
        """
        return resolve_run_dir(config)

    @staticmethod
    def _has_prior_run_state(run_dir: Path) -> bool:
        """Detect an existing experiment without treating an empty directory as a run."""
        return has_prior_run_state(run_dir)

    @staticmethod
    def _client_request_stats(client) -> dict[str, Any]:
        if client is None:
            return {
                "requests": 0, "retries": 0, "attempts": 0, "successes": 0,
                "failures": 0, "timeouts": 0, "input_tokens": 0,
                "output_tokens": 0, "latency_seconds": 0.0,
                "total_attempt_latency": 0.0, "max_attempt_latency": 0.0,
            }
        if hasattr(client, "request_stats"):
            value = client.request_stats()
            return {
                "requests": int(value.get("requests", 0) or 0),
                "retries": int(value.get("retries", 0) or 0),
                "attempts": int(value.get("attempts", 0) or 0),
                "successes": int(value.get("successes", 0) or 0),
                "failures": int(value.get("failures", 0) or 0),
                "timeouts": int(value.get("timeouts", 0) or 0),
                "input_tokens": int(value.get("input_tokens", 0) or 0),
                "output_tokens": int(value.get("output_tokens", 0) or 0),
                "latency_seconds": float(value.get("latency_seconds", 0.0) or 0.0),
                "total_attempt_latency": float(value.get("total_attempt_latency", 0.0) or 0.0),
                "max_attempt_latency": float(value.get("max_attempt_latency", 0.0) or 0.0),
            }
        return {
            "requests": int(getattr(client, "request_count", 0) or 0),
            "retries": int(getattr(client, "retry_count", 0) or 0),
            "attempts": int(getattr(client, "attempts", 0) or 0),
            "successes": int(getattr(client, "successes", 0) or 0),
            "failures": int(getattr(client, "failures", 0) or 0),
            "timeouts": int(getattr(client, "timeouts", 0) or 0),
            "input_tokens": int(getattr(client, "input_tokens", 0) or 0),
            "output_tokens": int(getattr(client, "output_tokens", 0) or 0),
            "latency_seconds": float(getattr(client, "latency_seconds", 0.0) or 0.0),
            "total_attempt_latency": float(getattr(client, "total_attempt_latency", 0.0) or 0.0),
            "max_attempt_latency": float(getattr(client, "max_attempt_latency", 0.0) or 0.0),
        }

    def _runtime_stats(self) -> dict[str, Any]:
        """Collect execution-level request and cache counters without secrets."""
        stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "real_episode_count": 0,
            "pipeline_requests": 0,
            "user_requests": 0,
            "agent_requests": 0,
            "judge_requests": 0,
            "policy_generation_requests": 0,
            "retries": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_seconds": 0.0,
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
        adapter = self.base_evaluator
        if hasattr(adapter, "get_stats"):
            adapter_stats = adapter.get_stats()
            for key in ("cache_hits", "cache_misses", "real_episode_count", "pipeline_requests",
                        "user_requests", "agent_requests", "judge_requests"):
                stats[key] = int(adapter_stats.get(key, 0) or 0)
            stats["retries"] += int(adapter_stats.get("retries", 0) or 0)
            for key in ("attempts", "successes", "failures", "timeouts"):
                stats[key] += int(adapter_stats.get(key, 0) or 0)
            stats["input_tokens"] += int(adapter_stats.get("input_tokens", 0) or 0)
            stats["output_tokens"] += int(adapter_stats.get("output_tokens", 0) or 0)
            stats["latency_seconds"] += float(adapter_stats.get("latency_seconds", 0.0) or 0.0)
            stats["total_attempt_latency"] += float(adapter_stats.get("total_attempt_latency", 0.0) or 0.0)
            stats["max_attempt_latency"] = max(
                stats["max_attempt_latency"],
                float(adapter_stats.get("max_attempt_latency", 0.0) or 0.0),
            )
            for role in ("user", "agent", "judge"):
                stats[f"{role}_input_tokens"] += int(adapter_stats.get(f"{role}_input_tokens", 0) or 0)
                stats[f"{role}_output_tokens"] += int(adapter_stats.get(f"{role}_output_tokens", 0) or 0)
                stats[f"{role}_latency_seconds"] += float(adapter_stats.get(f"{role}_latency_seconds", 0.0) or 0.0)

        seen_clients = set()
        for evolver in (self.customer_evolver, self.service_evolver):
            generator = getattr(evolver, "strategy_generator", None) or getattr(evolver, "patch_generator", None)
            client = getattr(generator, "llm_client", None)
            if client is None or id(client) in seen_clients:
                continue
            seen_clients.add(id(client))
            client_stats = self._client_request_stats(client)
            stats["policy_generation_requests"] += client_stats["requests"]
            stats["retries"] += client_stats["retries"]
            for key in ("attempts", "successes", "failures", "timeouts"):
                stats[key] += client_stats.get(key, 0)
            stats["input_tokens"] += client_stats["input_tokens"]
            stats["output_tokens"] += client_stats["output_tokens"]
            stats["latency_seconds"] += client_stats["latency_seconds"]
            stats["total_attempt_latency"] += client_stats.get("total_attempt_latency", 0.0)
            stats["max_attempt_latency"] = max(
                stats["max_attempt_latency"], client_stats.get("max_attempt_latency", 0.0)
            )

        stats["llm_requests"] = stats["pipeline_requests"] + stats["policy_generation_requests"]
        stats["total_requests_including_retries"] = stats["llm_requests"]
        stats["generation_wall_times_seconds"] = dict(self._generation_wall_times)
        if self.request_budget is not None:
            stats["request_budget"] = self.request_budget.snapshot()
        return stats

    def _update_effective_thinking_provenance(self) -> None:
        path = self.store.run_dir / "environment" / "provenance.json"
        if not path.exists():
            return
        try:
            provenance = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        clients: dict[str, list[str]] = {key: [] for key in (
            "customer", "agent", "customer_evolver", "service_evolver"
        )}
        adapter = self.base_evaluator
        for pipeline in getattr(adapter, "_pipelines", []) or []:
            for role in ("customer", "agent"):
                client = getattr(pipeline, f"{role if role != 'customer' else 'user'}_llm_client", None)
                if client is not None:
                    clients[role].append(getattr(client, "thinking_effective", "unknown"))
        for role, evolver, attr in (
            ("customer_evolver", self.customer_evolver, "strategy_generator"),
            ("service_evolver", self.service_evolver, "patch_generator"),
        ):
            client = getattr(getattr(evolver, attr, None), "llm_client", None)
            if client is not None:
                clients[role].append(getattr(client, "thinking_effective", "unknown"))
        effective = {}
        for role, values in clients.items():
            if not values:
                effective[role] = "unknown"
            elif "unsupported/not_applied" in values:
                effective[role] = "unsupported/not_applied"
            elif all(value == "provider_default" for value in values):
                effective[role] = "provider_default"
            else:
                effective[role] = "unknown"
        provenance["effective_thinking_modes"] = effective
        self.store.write_json("environment/provenance.json", provenance)

    def run(self) -> dict[str, Any]:
        """Run an experiment and persist diagnostics even on a hard failure."""
        run_started = time.monotonic()
        try:
            return self._run_impl()
        except Exception as exc:
            # API timeouts and provider failures can abort a generation before
            # the normal end-of-run metrics write.  Preserve the counters and
            # the failure class in the same authoritative run directory so a
            # partial experiment remains auditable.
            try:
                self._persist_pending_evolver_records()
                self._update_effective_thinking_provenance()
                metrics = self._runtime_stats()
                checkpoint_status = (
                    "incomplete_budget_exhausted"
                    if isinstance(exc, APIRequestBudgetExceeded)
                    else "incomplete_inconclusive" if (
                        isinstance(exc, GenerationProtocolError)
                        or bool(getattr(exc, "inconclusive", False))
                    ) else "incomplete_failed"
                )
                checkpoint_details = {"reason": str(exc), "failure_type": type(exc).__name__}
                if isinstance(exc, APIRequestBudgetExceeded):
                    checkpoint_details.update({
                        "reason": "api_request_budget_exceeded",
                        "budget_scope": exc.scope,
                        "limit": exc.limit,
                        "used": exc.used,
                    })
                self._write_incomplete_checkpoint(checkpoint_status, **checkpoint_details)
                if isinstance(exc, APIRequestBudgetExceeded):
                    generation = self._active_generation
                    metrics.update({
                        "run_status": "budget_exhausted",
                        "inconclusive_reason": "api_request_budget_exceeded",
                        "budget_scope": exc.scope,
                        "budget_limit": exc.limit,
                        "budget_used": exc.used,
                        "wall_time_seconds": time.monotonic() - run_started,
                    })
                    self.store.write_json("analysis/orchestration_metrics.json", metrics)
                    if self.request_budget is not None:
                        self.store.write_json("analysis/request_budget.json", self.request_budget.snapshot())
                    return {
                        "run_dir": str(self.store.run_dir),
                        "report": None,
                        "history": [],
                        "completed_generations": self.store.completed_generations(),
                        "run_status": "budget_exhausted",
                        "orchestration_metrics": metrics,
                    }
                inconclusive = (
                    isinstance(exc, GenerationProtocolError)
                    or bool(getattr(exc, "inconclusive", False))
                )
                metrics.update({
                    "run_status": "inconclusive" if inconclusive else "failed",
                    "failure_type": type(exc).__name__,
                    "failure_message": str(exc),
                    "wall_time_seconds": time.monotonic() - run_started,
                })
                if inconclusive:
                    metrics["inconclusive_reason"] = str(exc)
                self.store.write_json("analysis/orchestration_metrics.json", metrics)
                if self.request_budget is not None:
                    self.store.write_json("analysis/request_budget.json", self.request_budget.snapshot())
            except Exception:
                # Never hide the original provider or evaluation exception if
                # diagnostics persistence itself is unavailable.
                pass
            raise

    def _run_impl(self) -> dict[str, Any]:
        run_started = time.monotonic()
        manifest_exists = (self.store.run_dir / "split_manifest" / "evolution_cases.json").exists()
        # A fresh run rebuilds the manifest from the current config.  Only an
        # explicit resume reuses an existing split, preventing a changed
        # instances_per_path/seed from silently running against stale data.
        splits = SplitManager.load(self.store.run_dir / "split_manifest") if (manifest_exists and self.config.persistence.resume) else self.split_manager.build()
        self.store.write_json("config/evolution.json", self.config.to_dict())
        estimate = self._estimate_generation_workload(splits)
        self.store.write_json("analysis/request_budget_estimate.json", estimate)
        print(
            "[API budget estimate; rough, not guaranteed] "
            f"evolution_cases={estimate['evolution_cases']} "
            f"validation_cases={estimate['validation_cases']} "
            f"episodes/gen≈{estimate['estimated_episode_runs_per_generation']} "
            f"provider_requests/gen≈{estimate['estimated_provider_requests_per_generation']} "
            f"hard_limits(gen/run)={self.config.evaluation.max_api_requests_per_generation}/"
            f"{self.config.evaluation.max_api_requests_per_run}"
        )
        model_metadata = self.config.model_metadata
        self.store.write_json("environment/provenance.json", {
            "scenario": self.config.scenario,
            "seed": self.config.seed,
            "mode": self.config.experiment_mode,
            "model": model_metadata.get("model"),
            "provider": model_metadata.get("provider"),
            "api_url": model_metadata.get("api_url"),
            "client": model_metadata.get("client"),
            "evaluator": "mock" if isinstance(self.base_evaluator, MockEpisodeEvaluator) else "real",
            "repetitions": self.config.evaluation.repetitions,
            "concurrency": self.config.evaluation.concurrency,
            "judge_in_evolution": self.config.evaluation.judge_in_evolution,
            "customer_generator": "llm" if getattr(self.customer_evolver, "strategy_generator", None) is not None else "template",
            "service_generator": "llm" if getattr(self.service_evolver, "patch_generator", None) is not None else "template",
            "strict_real_generation": bool(
                getattr(self.customer_evolver, "require_strategy_generator", False)
                or getattr(self.service_evolver, "require_patch_generator", False)
            ),
            "customer_adversary_access": "black_box",
            "customer_thinking_mode": self.config.evaluation.customer_thinking_mode or "default",
            "requested_thinking_modes": {
                "customer": self.config.evaluation.customer_thinking_mode or "default",
                "agent": self.config.evaluation.agent_thinking_mode or "default",
                "customer_evolver": self.config.evaluation.evolver_thinking_mode or "default",
                "service_evolver": self.config.evaluation.evolver_thinking_mode or "default",
            },
            "effective_thinking_modes": {
                role: ("unknown" if mode != "default" else "provider_default")
                for role, mode in {
                    "customer": self.config.evaluation.customer_thinking_mode or "default",
                    "agent": self.config.evaluation.agent_thinking_mode or "default",
                    "customer_evolver": self.config.evaluation.evolver_thinking_mode or "default",
                    "service_evolver": self.config.evaluation.evolver_thinking_mode or "default",
                }.items()
            },
            "tool_contract": asdict(self.config.evaluation.tool_contract),
            "token_budget": asdict(self.config.evaluation.token_budget),
            "customer_protocol_retry_limit": self.config.evaluation.customer_protocol_retries,
            "evolver_protocol_retry_limit": self.config.evaluation.evolver_protocol_retries,
            "agent_max_retries": self.config.evaluation.agent_max_retries,
            "customer_transport_max_retries": self.config.evaluation.customer_transport_max_retries,
            "evolver_max_retries": self.config.evaluation.evolver_max_retries,
            "judge_max_retries": self.config.evaluation.judge_max_retries,
            "judge_validation_retries": self.config.evaluation.judge_validation_retries,
            "invalid_evaluation_retries": self.config.evaluation.invalid_evaluation_retries,
            "max_tool_steps": self.config.evaluation.max_tool_steps,
            "max_api_requests_per_generation": self.config.evaluation.max_api_requests_per_generation,
            "max_api_requests_per_run": self.config.evaluation.max_api_requests_per_run,
            "rate_limit_backoff_seconds": self.config.evaluation.rate_limit_backoff_seconds,
            "max_generations": self.config.max_generations,
            "max_turns": self.config.evaluation.max_turns,
            "repetitions_per_case": self.config.evaluation.repetitions,
            "model_metadata": model_metadata,
            "estimated_workload": estimate,
            "split_manifest": self._split_manifest_provenance(),
            **self._git_provenance(),
            "requested_output_dir": str(self.requested_output_dir),
            "run_dir": str(self.store.run_dir),
            "fresh_run_isolated": self.fresh_run_isolated,
        })
        completed = self.store.completed_generations()
        start = (max(completed) + 1) if self.config.persistence.resume and completed else 0
        self._resume_checkpoint = self._load_incomplete_checkpoint(start)
        checkpoint_customer = (self._resume_checkpoint or {}).get(
            "active_customer_policy", (self._resume_checkpoint or {}).get("customer_policy")
        )
        checkpoint_service = (self._resume_checkpoint or {}).get(
            "active_service_policy", (self._resume_checkpoint or {}).get("service_policy")
        )
        baseline_customer = (
            (self._resume_checkpoint or {}).get("generation_baseline_customer_policy")
            or checkpoint_customer
        )
        baseline_service = (
            (self._resume_checkpoint or {}).get("generation_baseline_service_policy")
            or checkpoint_service
        )
        customer = (
            CustomerPolicy.from_dict(baseline_customer)
            if baseline_customer else self._load_policy("customer_policy", CustomerPolicy())
        )
        service = (
            ServicePolicy.from_dict(baseline_service)
            if baseline_service else self._load_policy("service_policy", ServicePolicy())
        )
        self._active_customer_policy = (
            CustomerPolicy.from_dict(checkpoint_customer) if checkpoint_customer else customer
        )
        self._active_service_policy = (
            ServicePolicy.from_dict(checkpoint_service) if checkpoint_service else service
        )
        if not self.config.persistence.resume or not (self.store.run_dir / "environment/initial_service_policy.json").exists():
            self.store.write_json("environment/initial_customer_policy.json", CustomerPolicy().to_dict())
            self.store.write_json("environment/initial_service_policy.json", ServicePolicy().to_dict())
        history = []
        for generation in range(start, self.config.max_generations):
            self._active_generation = generation
            self._completed_phases = set(
                (self._resume_checkpoint or {}).get("completed_phases", [])
                if int((self._resume_checkpoint or {}).get("generation", -1)) == generation
                else []
            )
            self._active_stage = (
                str((self._resume_checkpoint or {}).get("stage") or "generation_started")
                if int((self._resume_checkpoint or {}).get("generation", -1)) == generation
                else "generation_started"
            )
            baseline_customer = customer
            baseline_service = service
            self._generation_baseline_customer_policy = baseline_customer
            self._generation_baseline_service_policy = baseline_service
            self._write_incomplete_checkpoint()
            generation_started = time.monotonic()
            gen_dir = self.store.generation_dir(generation)
            service_evolution_summary = {
                "status": "not_applicable",
                "candidate_count": 0,
                "selected_policy_id": service.policy_id,
                "service_changed": False,
            }
            if self.config.experiment_mode in {"customer_only", "coevolution"} and self.config.experiment_mode != "static":
                self._active_stage = "customer_failure_scan"
                prior_customer_episodes = self.evaluator.evaluate(
                    customer, service, splits.evolution, "evolution", generation, "customer_failure_scan"
                )
                prior_failures = []
                for item in prior_customer_episodes:
                    if not item.is_attributable_service_failure():
                        continue
                    signature = item.vulnerability_signature_v2()
                    if signature is not None:
                        prior_failures.append(FailureOccurrence.from_episode(item, signature))
                self._mark_phase_complete("customer_failure_scan_completed", "customer_failure_scan_completed")
                candidates_path = gen_dir / "customer_candidates.json"
                if (
                    "customer_candidate_selection_completed" in self._completed_phases
                    and candidates_path.exists()
                ):
                    # A selected Customer candidate is a generation artifact, not an
                    # episode-cache entry. Reuse it after restart instead of paying
                    # for a fresh stochastic Evolver proposal.
                    candidate_payload = json.loads(candidates_path.read_text(encoding="utf-8"))
                    selected_data = candidate_payload.get("selected_policy") or checkpoint_customer
                    if not isinstance(selected_data, dict):
                        raise RuntimeError("checkpoint marks Customer selection complete but selected policy is missing")
                    customer = CustomerPolicy.from_dict(selected_data)
                    scores = [CandidateScore.from_dict(item) for item in candidate_payload.get("scores", [])]
                    candidate_records = []
                else:
                    self._active_stage = "customer_candidate_generation_and_evaluation"
                    customer, candidate_records, scores = self.customer_evolver.evolve(
                        customer, service, splits.evolution, self.evaluator,
                        None if self.config.experiment_mode == "customer_only" else self.attack_archive,
                        generation,
                        self.config.customer.candidate_count,
                        # Stage 1 compares every proposal on the same complete
                        # evolution panel. Legacy subsampling remains available
                        # only to the older coevolution mode.
                        cases_per_candidate=(
                            None if self.config.experiment_mode == "customer_only"
                            else self.config.customer.cases_per_candidate
                        ),
                        elite_count=self.config.customer.elite_count,
                        source_failures=prior_failures,
                        incumbent_episodes=prior_customer_episodes,
                    )
                self._active_customer_policy = customer
                customer_evaluation_inconclusive = bool(scores) and not any(
                    score.episodes > 0 for score in scores
                )
                customer_evaluation_status = (
                    "inconclusive" if customer_evaluation_inconclusive else "valid"
                )
                if "customer_candidate_selection_completed" not in self._completed_phases:
                    score_by_id = {score.policy_id: score.to_dict() for score in scores}
                    proposals = [
                        {
                            "policy": policy.to_dict(),
                            "score": score_by_id.get(policy.policy_id),
                            "episode_ids": [item.episode_id for item in items],
                            "valid_episode_count": sum(item.is_substantively_evaluable() for item in items),
                            "invalid_episode_count": sum(not item.is_substantively_evaluable() for item in items),
                            "protocol_invalid_count": sum(
                                item.is_evaluation_invalid() or not item.protocol_valid
                                for item in items
                            ),
                            "attributable_service_failure_count": sum(
                                item.is_attributable_service_failure() for item in items
                            ),
                        }
                        for policy, items in candidate_records
                        if policy.policy_id != baseline_customer.policy_id
                    ]
                    customer_generation_record = copy.deepcopy(
                        getattr(self.customer_evolver, "last_generation_record", None)
                    )
                    generation_attempts = list(
                        (customer_generation_record or {}).get("attempts", [])
                    )
                    evolver_request_usage = {
                        "provider_attempts": len(generation_attempts),
                        "input_tokens": sum(int(item.get("input_tokens") or 0) for item in generation_attempts),
                        "completion_tokens": sum(int(item.get("completion_tokens") or 0) for item in generation_attempts),
                        "reasoning_tokens": sum(int(item.get("reasoning_tokens") or 0) for item in generation_attempts),
                        "timeouts": sum(bool(item.get("timeout")) for item in generation_attempts),
                        "provider_failures": sum(bool(item.get("provider_error")) for item in generation_attempts),
                        "latency_seconds": sum(float(item.get("latency_seconds") or 0.0) for item in generation_attempts),
                    }
                    self.store.write_json(f"generations/gen_{generation:03d}/customer_candidates.json", {
                        "strategy_schema": self.config.customer.strategy_schema,
                        "parent_policy": baseline_customer.to_dict(),
                        "parent_score": score_by_id.get(baseline_customer.policy_id),
                        "parent_feedback": copy.deepcopy(
                            (customer_generation_record or {}).get("parent_feedback", {})
                        ),
                        "generation_status": (customer_generation_record or {}).get("status"),
                        "generation_reason": (customer_generation_record or {}).get("reason"),
                        "generation_record": customer_generation_record,
                        "evolver_request_usage": evolver_request_usage,
                        "selected_policy": customer.to_dict(), "scores": [score.to_dict() for score in scores],
                        "candidate_count": len(proposals),
                        "candidate_policies": proposals,
                        "proposal_candidates": copy.deepcopy(
                            (customer_generation_record or {}).get("candidates", [])
                        ),
                        "fixed_service_policy_id": service.policy_id,
                        "evaluation_case_ids": list(
                            getattr(self.customer_evolver, "last_evaluation_case_ids", [])
                            or [case.case_id for case in splits.evolution]
                        ),
                        "selection": copy.deepcopy(
                            getattr(self.customer_evolver, "last_selection_record", None)
                        ),
                        "candidate_episode_counts": [len(items) for policy, items in candidate_records if policy.policy_id != baseline_customer.policy_id],
                        "candidate_valid_episode_counts": [
                            sum(item.is_substantively_evaluable() for item in items)
                            for policy, items in candidate_records if policy.policy_id != baseline_customer.policy_id
                        ],
                        "candidate_invalid_episode_counts": [
                            sum(not item.is_substantively_evaluable() for item in items)
                            for policy, items in candidate_records if policy.policy_id != baseline_customer.policy_id
                        ],
                        "rejections": list(getattr(self.customer_evolver, "last_rejections", [])),
                        "source_failures": [failure.to_dict() for failure in prior_failures],
                        "evaluation_status": customer_evaluation_status,
                        "selection_status": customer_evaluation_status,
                        "reason": (
                            "customer_candidate_evaluation_invalid:no_valid_episode_evidence"
                            if customer_evaluation_inconclusive else None
                        ),
                    })
                    self._persist_evolver_record(
                        generation,
                        "customer",
                        selected_policy=customer.to_dict(),
                        selected_policy_id=customer.policy_id,
                        scores=[score.to_dict() for score in scores],
                        selection=copy.deepcopy(
                            getattr(self.customer_evolver, "last_selection_record", None)
                        ),
                        evaluation_status=customer_evaluation_status,
                        selection_status=customer_evaluation_status,
                    )
                    self._active_customer_policy = customer
                    self._mark_phase_complete(
                        "customer_candidate_selection_completed", "customer_candidate_selection_completed"
                    )
                if customer_evaluation_inconclusive:
                    raise GenerationEvaluationInconclusive(
                        "customer_candidate_evaluation_invalid:no_valid_episode_evidence"
                    )
                self._active_stage = "selected_customer_evaluation"
                selected_episodes = self.evaluator.evaluate(customer, service, splits.evolution, "evolution", generation, "selected_customer")
                signatures = [
                    signature
                    for item in selected_episodes
                    if item.is_attributable_service_failure()
                    for signature in [item.vulnerability_signature_v2()]
                    if signature is not None
                ]
                occurrence_values = [
                    FailureOccurrence.from_episode(item, item.vulnerability_signature_v2())
                    for item in selected_episodes
                    if item.is_attributable_service_failure()
                    and item.vulnerability_signature_v2() is not None
                ]
                case_specs_by_id = {
                    case.case_id: case for case in splits.evolution + splits.validation
                }
                if self.config.experiment_mode != "customer_only":
                    self.attack_archive.add(
                        customer,
                        signatures,
                        selected_episodes,
                        generation,
                        case_specs_by_id=case_specs_by_id,
                        occurrences=occurrence_values,
                        reproduction_seed=self.config.seed,
                    )
                    self.frontier.add(selected_episodes)
                self._mark_phase_complete(
                    "selected_customer_evaluation_completed", "selected_customer_evaluation_completed"
                )
            if self.config.experiment_mode in {"service_only", "coevolution"} and self.config.experiment_mode != "static":
                self._active_stage = "service_failure_scan"
                service_failure_episodes = self.evaluator.evaluate(
                    customer, service, splits.evolution, "evolution", generation, "service_failures"
                )
                failures = []
                for item in service_failure_episodes:
                    if not item.is_attributable_service_failure():
                        continue
                    signature = item.vulnerability_signature_v2()
                    if signature is not None:
                        failures.append(FailureOccurrence.from_episode(item, signature))
                self._mark_phase_complete("service_failure_scan_completed", "service_failure_scan_completed")
                service_gate_path = gen_dir / "service_gate.json"
                if (
                    (
                        "service_candidate_gate_completed" in self._completed_phases
                        or "service_evolution_completed" in self._completed_phases
                    )
                    and service_gate_path.exists()
                ):
                    # The Service-stage artifact contains the proposal/no-op and
                    # selected-policy provenance. Rehydrate it rather than
                    # regenerating a stochastic proposal after a restart.
                    gate_payload = json.loads(service_gate_path.read_text(encoding="utf-8"))
                    service_record_path = gen_dir / "service_generation.json"
                    service_record = (
                        json.loads(service_record_path.read_text(encoding="utf-8"))
                        if service_record_path.exists() else {}
                    )
                    selected_service = (
                        checkpoint_service
                        or service_record.get("selected_policy")
                        or service_record.get("candidate_policy")
                    )
                    if not isinstance(selected_service, dict):
                        raise RuntimeError("checkpoint marks Service gate complete but selected policy is missing")
                    service = ServicePolicy.from_dict(selected_service)
                    decision = GateDecision(
                        accepted=bool(gate_payload.get("accepted")),
                        reason=str(gate_payload.get("reason") or "unknown"),
                        delta=float(gate_payload.get("delta") or 0.0),
                        metrics=dict(gate_payload.get("decision_metrics") or {}),
                    )
                    patch = ServicePatch.from_dict(gate_payload["patch"]) if gate_payload.get("patch") else None
                    self.service_evolver.last_candidate_records = list(gate_payload.get("candidates") or [])
                    self.service_evolver.last_baseline_metrics = dict(gate_payload.get("baseline_metrics") or {})
                    self.service_evolver.last_selected_metrics = dict(gate_payload.get("selected_metrics") or {})
                    service_evolution_summary = dict(gate_payload.get("service_evolution") or {})
                    if not service_evolution_summary:
                        service_evolution_summary = {
                            "status": "candidate_accepted" if decision.accepted else "candidate_rejected",
                            "candidate_count": len(self.service_evolver.last_candidate_records),
                            "selected_policy_id": service.policy_id,
                            "service_changed": service.policy_id != baseline_service.policy_id,
                        }
                    self.service_evolver.last_evolution_status = service_evolution_summary["status"]
                    self.service_evolver.last_no_candidate_reason = service_evolution_summary.get(
                        "no_candidate_reason"
                    )
                else:
                    self._active_stage = "service_candidate_generation_and_gate"
                    service, decision, patch = self.service_evolver.evolve(
                        service, failures, splits.validation, splits.validation, self.evaluator, generation,
                        self.config.service.candidate_count,
                        customer_policy=customer,
                        exact_replay_instances=self.attack_archive.exact_instances(
                            self.config.service.exact_replay_instance_count
                        ),
                        transfer_replay_policies=self._replay_policies(
                            self.config.service.transfer_replay_policy_count or 0
                        ),
                        replay_attack_count=self.config.service.transfer_replay_policy_count or 0,
                        defense_summary=self.defense_archive.to_dicts()[-max(1, self.config.evaluation.summary_limit):],
                        historical_summary=self.service_evolver.last_candidate_records[-max(1, self.config.evaluation.summary_limit):],
                    )
                    self._active_service_policy = service
                    no_candidate = decision.reason == "no_candidate_proposed"
                    service_evolution_summary = {
                        "status": (
                            "no_candidate_proposed" if no_candidate
                            else getattr(self.service_evolver, "last_evolution_status", None)
                            or ("candidate_accepted" if decision.accepted else "candidate_rejected")
                        ),
                        "no_candidate_reason": decision.metrics.get("no_candidate_reason") if no_candidate else None,
                        "candidate_count": (
                            0 if no_candidate
                            else len(getattr(self.service_evolver, "last_candidate_records", []))
                        ),
                        "selected_policy_id": service.policy_id,
                        "service_changed": service.policy_id != baseline_service.policy_id,
                    }
                    self.store.write_json(f"generations/gen_{generation:03d}/service_gate.json", {
                        "accepted": decision.accepted, "reason": decision.reason, "delta": decision.delta,
                        "patch": patch.to_dict() if patch else None,
                        "gate_executed": not no_candidate,
                        "service_evolution": service_evolution_summary,
                        "decision_metrics": decision.metrics,
                        "baseline_metrics": getattr(self.service_evolver, "last_baseline_metrics", {}),
                        "selected_metrics": getattr(self.service_evolver, "last_selected_metrics", {}),
                        "candidates": list(getattr(self.service_evolver, "last_candidate_records", [])),
                    })
                    self._persist_evolver_record(
                        generation,
                        "service",
                        selected_policy=service.to_dict(),
                        selected_policy_id=service.policy_id,
                        service_evolution=service_evolution_summary,
                        gate={
                            "accepted": decision.accepted,
                            "reason": decision.reason,
                            "delta": decision.delta,
                        },
                    )
                    self._active_service_policy = service
                    self._mark_phase_complete(
                        "service_evolution_completed" if no_candidate else "service_candidate_gate_completed",
                        "service_evolution_completed" if no_candidate else "service_candidate_gate_completed",
                    )
                if decision.accepted and patch:
                    before = getattr(self.service_evolver, "last_baseline_metrics", {})
                    after = getattr(self.service_evolver, "last_selected_metrics", {})
                    self.defense_archive.add(DefenseRecord(
                        defense_id=f"defense_g{generation}", service_policy_id=service.policy_id,
                        generation_added=generation, rule_ids=[rule.rule_id for rule in patch.rules],
                        addresses_failure_signatures=[failure.signature_id for failure in failures],
                        validation_delta={"task_success": decision.delta, "robust_task_success": decision.delta},
                        normal_user_delta={"task_success": decision.metrics.get("normal_task_success", 0.0) - before.get("normal_task_success", 0.0)},
                        adversarial_delta={"task_success": decision.metrics.get("task_success", 0.0) - before.get("task_success", 0.0)},
                        latest_adversary_delta={"task_success": decision.metrics.get("latest_task_success", 0.0) - before.get("latest_task_success", 0.0)},
                        replay_delta={"task_success": decision.metrics.get("replay_task_success", 0.0) - before.get("replay_task_success", 0.0)},
                        exact_replay_delta={
                            "task_success": (decision.metrics.get("exact_replay_task_success") or 0.0)
                            - (before.get("exact_replay_task_success") or 0.0)
                        },
                        transfer_replay_delta={
                            "task_success": (decision.metrics.get("transfer_replay_task_success") or 0.0)
                            - (before.get("transfer_replay_task_success") or 0.0)
                        },
                        exact_replay_regressions=list(decision.metrics.get("exact_replay_regressions", [])),
                        robust_delta={"task_success": decision.metrics.get("robust_task_success", 0.0) - before.get("robust_task_success", 0.0)},
                        regression_cases=[case for item in getattr(self.service_evolver, "last_candidate_records", []) if item.get("accepted") and item.get("patch_id") == patch.patch_id for case in item.get("normal_regression_cases", [])],
                    ))
            self._active_stage = "generation_summary"
            episodes = self.evaluator.evaluate(customer, service, splits.validation, "validation", generation, "generation_summary")
            self.store.append_jsonl(f"generations/gen_{generation:03d}/episodes.jsonl", [item.to_dict() for item in episodes])
            self.store.write_json(f"generations/gen_{generation:03d}/customer_policy.json", customer.to_dict())
            self.store.write_json(f"generations/gen_{generation:03d}/service_policy.json", service.to_dict())
            if not episodes or not any(item.is_substantively_evaluable() for item in episodes):
                raise GenerationEvaluationInconclusive(
                    "generation_summary_invalid:no_valid_episode_evidence"
                )
            if self.config.experiment_mode != "customer_only":
                self.frontier.add(episodes)
            self._generation_wall_times[str(generation)] = time.monotonic() - generation_started
            self.store.mark_generation_complete(generation, {
                "episode_count": len(episodes),
                "service_policy_id": service.policy_id,
                "customer_policy_id": customer.policy_id,
                "service_evolution": service_evolution_summary,
                "orchestration": self._runtime_stats(),
            })
            self._completed_phases.add("generation_summary_completed")
            self._active_stage = "generation_completed"
            self._write_incomplete_checkpoint("complete")
            if self.request_budget is not None:
                gen_stats = self.request_budget.snapshot().get("generations", {}).get(str(generation), {})
                used = gen_stats.get("provider_attempts", 0)
                limit = self.config.evaluation.max_api_requests_per_generation
                print(
                    f"[G{generation} complete] provider_attempts={used} "
                    f"generation_budget={used}/{limit if limit is not None else 'unbounded'}"
                )
            generation_metrics = aggregate_episode_metrics(episodes)
            history.append({
                "generation": generation,
                "episode_count": len(episodes),
                "task_success": generation_metrics["task_success"],
                "invalid_episodes": generation_metrics["invalid_episodes"],
            })
        if self.config.experiment_mode != "customer_only":
            self.frontier.save(self.store.run_dir / "analysis" / "weakness_frontier.json", self.store.run_dir / "analysis" / "weakness_frontier.csv")
        if self.config.fresh_adversary.enabled and self.config.experiment_mode != "customer_only":
            self.fresh_adversary_evaluation(
                rounds=self.config.fresh_adversary.rounds,
                candidate_count=self.config.fresh_adversary.candidate_count,
                target_service=service,
                target_label="final_coevolved",
            )
        orchestration_metrics = self._runtime_stats()
        orchestration_metrics["wall_time_seconds"] = time.monotonic() - run_started
        self.store.write_json("analysis/orchestration_metrics.json", orchestration_metrics)
        if self.request_budget is not None:
            self.store.write_json("analysis/request_budget.json", self.request_budget.snapshot())
        self._update_effective_thinking_provenance()
        report = generate_report(self.store.run_dir)
        return {
            "run_dir": str(self.store.run_dir),
            "report": str(report),
            "history": history,
            "completed_generations": self.store.completed_generations(),
            "orchestration_metrics": orchestration_metrics,
        }

    def heldout_evaluation(self):
        splits = SplitManager.load(self.store.run_dir / "split_manifest")
        customer = self._load_policy("customer_policy", CustomerPolicy())
        service = self._load_policy("service_policy", ServicePolicy())
        generation = max(self.store.completed_generations(), default=0)
        results = self.evaluator.evaluate(customer, service, splits.heldout_test, "heldout_test", generation, "heldout")
        self.store.write_json("analysis/heldout_results.json", {"results": [item.to_dict() for item in results]})
        return results

    def fresh_adversary_evaluation(self, rounds: int = 2, candidate_count: int = 5,
                                   target_service=None, target_label: str = "final",
                                   mode: str | None = None, fresh_strategy_generator=None):
        """Adapt fresh customers on validation, then evaluate only on held-out.

        In-family reuses the training generator class/model with a fresh seed.
        Cross-generator requires an explicitly constructed, independently
        configured generator; it never silently falls back to the training one.
        """
        splits = SplitManager.load(self.store.run_dir / "split_manifest")
        service = target_service or self._load_policy("service_policy", ServicePolicy())
        mode = mode or self.config.fresh_adversary.mode
        if mode not in {"in_family", "cross_generator"}:
            raise ValueError("fresh adversary mode must be in_family or cross_generator")
        fresh_config = self.config.fresh_adversary
        if mode == "cross_generator":
            if fresh_strategy_generator is None:
                raise ValueError("cross_generator mode requires a separate fresh_strategy_generator")
            if not fresh_config.generator_model or not fresh_config.generator_provider:
                raise ValueError("cross_generator mode requires generator model/provider provenance")
            training_model = str(self.config.model_metadata.get("model") or "")
            training_provider = str(self.config.model_metadata.get("provider") or "")
            if (
                fresh_config.generator_model == training_model
                and fresh_config.generator_provider == training_provider
            ):
                raise ValueError("cross_generator must differ from the training model or provider")
            strategy_generator = fresh_strategy_generator
            require_generator = True
        else:
            strategy_generator = fresh_strategy_generator or getattr(
                self.customer_evolver, "strategy_generator", None
            )
            require_generator = bool(
                getattr(self.customer_evolver, "require_strategy_generator", False)
            ) if fresh_strategy_generator is None else True

        if strategy_generator is not None:
            from .customer.legacy import LegacyCustomerStrategyGeneratorAdapter
            if not isinstance(strategy_generator, LegacyCustomerStrategyGeneratorAdapter):
                strategy_generator = LegacyCustomerStrategyGeneratorAdapter(strategy_generator)

        incumbent = CustomerPolicy()
        results = []
        adaptation_results = []
        selector = CustomerSelector()
        fresh_evolver = CustomerEvolver(
            seed=self.config.seed + 10_000,
            validator=getattr(self.customer_evolver, "validator", None),
            selector=selector,
            strategy_generator=strategy_generator,
            require_strategy_generator=require_generator,
        )
        known_signatures: set[str] = set()
        round_records = []
        for generation in range(rounds):
            candidates = fresh_evolver.propose(incumbent, 10_000 + generation, candidate_count)
            evaluated = []
            for policy in candidates:
                # Adaptation receives validation feedback only.  Held-out is
                # evaluated after selecting the round's attacker and never
                # enters CustomerSelector or a training archive.
                episodes = self.evaluator.evaluate(policy, service, splits.validation, "validation", generation, "fresh_adaptation")
                for episode in episodes:
                    episode.metadata = dict(episode.metadata, fresh_round=generation, target_service=target_label)
                adaptation_results.extend(episodes)
                evaluated.append((policy, episodes))
            known_before = set(known_signatures)
            selected, scores = selector.select(evaluated)
            incumbent = selected or incumbent
            selected_validation = next(
                (episodes for policy, episodes in evaluated if selected is not None and policy.policy_id == selected.policy_id),
                [],
            )
            selected_signature_ids = sorted({
                signature.signature_id
                for episode in selected_validation
                if episode.is_attributable_service_failure()
                for signature in [episode.vulnerability_signature_v2()]
                if signature is not None
            })
            known_signatures.update(selected_signature_ids)
            heldout_episodes = self.evaluator.evaluate(incumbent, service, splits.heldout_test, "heldout_test", generation, "fresh_adversary_eval")
            for episode in heldout_episodes:
                episode.metadata = dict(episode.metadata, fresh_round=generation, target_service=target_label, adaptation_split="validation")
            results.extend(heldout_episodes)
            selected_scores = next((score.to_dict() for score in scores if score.policy_id == incumbent.policy_id), {})
            round_records.append({
                "round": generation,
                "target_service": target_label,
                "selected_policy": incumbent.to_dict(),
                "fitness": selected_scores,
                "known_signature_count_before": len(known_before),
                "selected_validation_signature_ids": selected_signature_ids,
                "known_signature_count_after": len(known_signatures),
                "novelty_reference": "prior selected fresh-round validation signatures",
            })
        training_model = self.config.model_metadata.get("model")
        training_provider = self.config.model_metadata.get("provider")
        generator_model = (
            fresh_config.generator_model if mode == "cross_generator"
            else training_model or getattr(getattr(strategy_generator, "llm_client", None), "model_name", None)
        )
        generator_provider = (
            fresh_config.generator_provider if mode == "cross_generator" else training_provider
        )
        generator_class = type(strategy_generator).__name__ if strategy_generator is not None else "template"
        heldout_metrics = aggregate_episode_metrics(results)
        robustness_value = (
            heldout_metrics["task_success"] if heldout_metrics["episodes"] > 0 else None
        )
        robustness_metric = (
            "fresh_in_family_robustness" if mode == "in_family"
            else "fresh_cross_generator_robustness"
        )
        output_name = f"fresh_adversary_{target_label}.json"
        if mode == "cross_generator":
            output_name = f"fresh_adversary_cross_generator_{target_label}.json"
        payload = {
            "evaluator": "mock" if isinstance(self.base_evaluator, MockEpisodeEvaluator) else "real",
            "fresh_mode": mode,
            "robustness_metric": robustness_metric,
            robustness_metric: robustness_value,
            "heldout_metric_status": "valid" if robustness_value is not None else "inconclusive",
            "heldout_valid_episode_count": int(heldout_metrics["episodes"]),
            "heldout_invalid_episode_count": int(heldout_metrics["invalid_episodes"]),
            "target_service": target_label,
            "results": [item.to_dict() for item in results],
            "adaptation_results": [item.to_dict() for item in adaptation_results],
            "rounds": round_records,
            "training_archive_used": False,
            "heldout_used_for_adaptation_or_selection": False,
            "customer_generator": "llm" if fresh_evolver.strategy_generator is not None else "template",
            "customer_generator_class": generator_class,
            "customer_generator_model": generator_model,
            "customer_generator_provider": generator_provider,
            "customer_generator_client": str(
                self.config.model_metadata.get("client") or "unknown"
            ),
            "training_generator_model": training_model,
            "training_generator_provider": training_provider,
            "fresh_generator_seed": self.config.seed + 10_000,
            "strict_real_generation": bool(fresh_evolver.require_strategy_generator),
        }
        self.store.write_json(f"analysis/{output_name}", payload)
        # Keep a stable aggregate path for existing tooling.
        self.store.write_json("analysis/fresh_adversary_results.json", {
            "results": [item.to_dict() for item in results],
            "target_service": target_label,
            "evaluator": "mock" if isinstance(self.base_evaluator, MockEpisodeEvaluator) else "real",
            "fresh_mode": mode,
            "robustness_metric": payload["robustness_metric"],
            robustness_metric: robustness_value,
            "heldout_metric_status": payload["heldout_metric_status"],
            "heldout_valid_episode_count": payload["heldout_valid_episode_count"],
            "heldout_invalid_episode_count": payload["heldout_invalid_episode_count"],
            "customer_generator_model": generator_model,
            "customer_generator_provider": generator_provider,
            "customer_generator": "llm" if fresh_evolver.strategy_generator is not None else "template",
            "strict_real_generation": bool(fresh_evolver.require_strategy_generator),
        })
        return results

    def cross_generation_evaluation(self):
        splits = SplitManager.load(self.store.run_dir / "split_manifest")
        generations = self.store.completed_generations()
        matrix = []
        customer_versions = [(0, CustomerPolicy.from_dict(json.loads((self.store.run_dir / "environment/initial_customer_policy.json").read_text(encoding="utf-8"))))]
        service_versions = [(0, ServicePolicy.from_dict(json.loads((self.store.run_dir / "environment/initial_service_policy.json").read_text(encoding="utf-8"))))]
        customer_versions.extend((generation + 1, CustomerPolicy.from_dict(self.store.read_generation(generation, "customer_policy"))) for generation in generations)
        service_versions.extend((generation + 1, ServicePolicy.from_dict(self.store.read_generation(generation, "service_policy"))) for generation in generations)
        for customer_generation, customer in customer_versions:
            for service_generation, service in service_versions:
                results = self.evaluator.evaluate(customer, service, splits.validation, "validation", customer_generation, "cross_generation")
                metrics = aggregate_episode_metrics(results)
                matrix.append({
                    "customer_generation": customer_generation,
                    "service_generation": service_generation,
                    "metrics": {
                        "task_success": metrics["task_success"],
                        "execution_score": metrics["execution_score"],
                        "verification": metrics["verification"],
                        "policy": metrics["policy"],
                        "action": metrics["action"],
                        "goal": metrics["goal"],
                    },
                })
        self.store.write_json("analysis/cross_generation_matrix.json", {
            "evaluator": "mock" if isinstance(self.base_evaluator, MockEpisodeEvaluator) else "real",
            "manifest": "validation_cases.json",
            "matrix": matrix,
        })
        return matrix

    def _load_policy(self, name, default):
        generations = self.store.completed_generations()
        if not generations:
            return default
        try:
            data = self.store.read_generation(generations[-1], name)
            return CustomerPolicy.from_dict(data) if name.startswith("customer") else ServicePolicy.from_dict(data)
        except FileNotFoundError:
            return default

    def _replay_policies(self, count: int):
        """Prefer recent, strategy/error-diverse archived attackers."""
        return self.attack_archive.policies(limit=count, diverse=True)
