"""Thin Customer-only evolutionary runner over a fixed Service S0.

This module intentionally has no imports of ServiceEvolver, ServiceGate,
AttackArchive, DefenseArchive, WeaknessFrontier, or failure attribution.
"""

from __future__ import annotations

import copy
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import time

from ..config import CustomerSearchConfig, EvolutionConfig
from ..customer_evolver import CustomerEvolver
from ..customer_selector import CustomerSelector
from ..evaluator_adapter import BudgetedEpisodeEvaluator, MockEpisodeEvaluator
from ..generation_protocol import GenerationProtocolError
from ..persistence import RunStore
from ..schemas import ServicePolicy
from ..split_manager import SplitManager
from .legacy import load_legacy_customer_policy
from .integrity import AdversaryPolicyValidator
from .policy import AdversaryPolicy


class CustomerEvolutionInconclusive(RuntimeError):
    """Raised when protocol/runtime invalidity leaves no incumbent score."""


class CustomerEvolutionRunner:
    """Evaluate incumbent + children once on E, then select by official score."""

    def __init__(self, config: CustomerSearchConfig | EvolutionConfig | None = None, evaluator=None,
                 customer_evolver: CustomerEvolver | None = None,
                 split_manager: SplitManager | None = None, run_dir: str | Path | None = None):
        if config is None:
            config = CustomerSearchConfig()
        elif isinstance(config, EvolutionConfig):
            if config.experiment_mode != "customer_only":
                raise ValueError("CustomerEvolutionRunner requires experiment_mode='customer_only'")
            config = CustomerSearchConfig.from_evolution_config(config)
        self.config = config
        if self.config.experiment_mode != "customer_only":
            raise ValueError("CustomerEvolutionRunner requires experiment_mode='customer_only'")
        self.run_dir = Path(run_dir or self.config.persistence.output_dir)
        self.store = RunStore(self.run_dir, create_archives=False)
        self.split_manager = split_manager or SplitManager(
            self.config.splits, self.run_dir / "split_manifest", scenario=self.config.scenario,
        )
        self.base_evaluator = evaluator or MockEpisodeEvaluator()
        self.evaluator = BudgetedEpisodeEvaluator(
            self.base_evaluator,
            repetitions=self.config.evaluation.repetitions,
            concurrency=self.config.evaluation.concurrency,
        )
        self.customer_evolver = customer_evolver or CustomerEvolver(
            seed=self.config.seed,
            selector=CustomerSelector(),
        )
        self.selector = self.customer_evolver.selector
        self.policy_validator = AdversaryPolicyValidator()
        self.request_budget = getattr(self.base_evaluator, "request_budget", None)

    @staticmethod
    def _git_value(*args) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args], capture_output=True, text=True, check=True,
                timeout=3,
            )
            value = result.stdout.strip()
            return value or None
        except (OSError, subprocess.SubprocessError):
            return None

    @staticmethod
    def _valid(rows):
        return [
            item for item in rows
            if item.is_runtime_evaluable() and isinstance(item.task_success, bool)
        ]

    @classmethod
    def _score_summary(cls, episodes):
        valid = cls._valid(episodes)
        invalid = len(episodes) - len(valid)
        success = (
            sum(bool(item.task_success) for item in valid) / len(valid)
            if valid else None
        )
        return {
            "episode_count": len(episodes),
            "valid_episode_count": len(valid),
            "invalid_episode_count": invalid,
            "official_task_success": success,
            "fitness": 1.0 - success if success is not None else None,
        }

    @staticmethod
    def _score_to_dict(scores):
        return [score.to_dict() for score in scores]

    def _request_metrics(self):
        base = {}
        if callable(getattr(self.base_evaluator, "get_stats", None)):
            base = dict(self.base_evaluator.get_stats() or {})
        generation_client = getattr(
            getattr(self.customer_evolver, "strategy_generator", None), "llm_client", None,
        )
        evolver = {}
        if callable(getattr(generation_client, "request_stats", None)):
            evolver = dict(generation_client.request_stats() or {})
        budget = self.request_budget.snapshot() if self.request_budget is not None else None
        return {
            "episode_evaluator": base,
            "customer_evolver": evolver,
            "request_budget": budget,
            "provider_attempts": (
                int((budget or {}).get("total_provider_attempts", 0))
                if budget is not None
                else int(base.get("attempts", 0) or 0) + int(evolver.get("attempts", 0) or 0)
            ),
            "input_tokens": int(base.get("input_tokens", 0) or 0)
            + int(evolver.get("input_tokens", 0) or 0),
            "output_tokens": int(base.get("output_tokens", 0) or 0)
            + int(evolver.get("output_tokens", 0) or 0),
            "timeouts": int(base.get("timeouts", 0) or 0)
            + int(evolver.get("timeouts", 0) or 0),
            "provider_failures": int(base.get("failures", 0) or 0)
            + int(evolver.get("failures", 0) or 0),
            "cache_hits": int(base.get("cache_hits", 0) or 0),
            "cache_misses": int(base.get("cache_misses", 0) or 0),
            "latency_seconds": float(base.get("latency_seconds", 0.0) or 0.0)
            + float(evolver.get("latency_seconds", 0.0) or 0.0),
        }

    def _write_run_provenance(self, splits, fresh_run_isolated=False):
        model = self.config.model_metadata or {}
        self.store.write_json("config/evolution.json", self.config.to_dict())
        self.store.write_json("environment/provenance.json", {
            "mode": "customer_only",
            "search_method": "open_ended_black_box_customer_search",
            "selection_objective": "fitness = 1 - mean(official task_success) over runtime-valid E episodes",
            "service_policy_id": "service_policy_s0",
            "model": model.get("model"),
            "provider": model.get("provider"),
            "api_url": model.get("api_url"),
            "client": model.get("client"),
            "evaluator": "mock" if isinstance(self.base_evaluator, MockEpisodeEvaluator) else "real",
            "seed": self.config.seed,
            "generation_count": self.config.max_generations,
            "customer_candidate_count": self.config.customer.candidate_count,
            "repetitions": self.config.evaluation.repetitions,
            "max_turns": self.config.evaluation.max_turns,
            "judge_in_evolution": self.config.evaluation.judge_in_evolution,
            "token_budget": asdict(self.config.evaluation.token_budget),
            "customer_protocol_retry_limit": self.config.evaluation.customer_protocol_retries,
            "evolver_protocol_retry_limit": self.config.evaluation.evolver_protocol_retries,
            "customer_hidden_truth_access": False,
            "evolver_case_or_trace_access": False,
            "customer_behavior_gate": False,
            "failure_attribution_in_fitness": False,
            "heldout_used_for_adaptation_or_selection": False,
            "split_strategy": splits.strategy,
            "split_seed": splits.seed,
            "split_counts": {
                "evolution": len(splits.evolution),
                "validation": len(splits.validation),
                "heldout_test": len(splits.heldout_test),
            },
            "runtime_commit": self._git_value("rev-parse", "HEAD"),
            "runtime_tag": self._git_value("describe", "--tags", "--exact-match"),
            "run_dir": str(self.run_dir),
            "fresh_run_isolated": bool(fresh_run_isolated),
        })

    def _load_incumbent(self, completed: list[int]) -> AdversaryPolicy:
        if completed:
            path = self.run_dir / "generations" / f"gen_{completed[-1]:03d}" / "customer_policy.json"
            if path.exists():
                policy = AdversaryPolicy.from_dict(json.loads(path.read_text(encoding="utf-8")))
                self.policy_validator.validate(policy)
                return policy
        initial_path = self.run_dir / "environment" / "initial_customer_policy.json"
        if self.config.persistence.resume and initial_path.exists():
            policy = load_legacy_customer_policy(json.loads(initial_path.read_text(encoding="utf-8")))
            self.policy_validator.validate(policy)
            return policy
        policy = AdversaryPolicy()
        self.policy_validator.validate(policy)
        return policy

    def _run_generation(self, generation: int, incumbent, service, evolution_cases):
        started = time.monotonic()
        request_metrics_before = self._request_metrics()
        self.store.generation_dir(generation)
        baseline = self.evaluator.evaluate(
            incumbent, service, evolution_cases, "evolution", generation,
            "customer_incumbent",
        )
        parent_score = self.selector.score(incumbent, baseline)
        try:
            proposals = self.customer_evolver.propose(
                incumbent,
                generation,
                count=self.config.customer.candidate_count,
                parent_reward=parent_score.fitness,
            )
        except GenerationProtocolError as exc:
            record = copy.deepcopy(exc.record)
            self.store.write_json(f"generations/gen_{generation:03d}/proposals.json", {
                "generation": generation,
                "parent_policy": incumbent.to_dict(),
                "parent_score": parent_score.to_dict(),
                "requested_candidate_count": self.config.customer.candidate_count,
                "proposed_candidate_count": 0,
                "generation_record": record,
                "candidate_policies": [],
            })
            self.store.append_jsonl(
                f"generations/gen_{generation:03d}/episodes.jsonl",
                [item.to_dict(include_analysis_artifacts=False) for item in baseline],
            )
            self.store.write_json(f"generations/gen_{generation:03d}/selection.json", {
                "generation": generation,
                "selection_status": "inconclusive",
                "reason": record.get("reason") or "customer_generation_protocol_invalid",
                "selected_policy_id": None,
                "candidate_scores": [parent_score.to_dict()],
            })
            raise CustomerEvolutionInconclusive(
                str(record.get("reason") or "customer_generation_protocol_invalid")
            ) from exc

        evaluated = [(incumbent, baseline)]
        for candidate in proposals:
            rows = self.evaluator.evaluate(
                candidate, service, evolution_cases, "evolution", generation,
                "customer_candidate",
            )
            evaluated.append((candidate, rows))

        selected, scores = self.selector.select(
            evaluated, incumbent_policy_id=incumbent.policy_id,
        )
        selected = selected or incumbent
        score_by_id = {item.policy_id: item for item in scores}
        if parent_score.fitness is None:
            evaluation_rows = [episode for _, rows in evaluated for episode in rows]
            generation_record = copy.deepcopy(self.customer_evolver.last_generation_record)
            self.store.write_json(f"generations/gen_{generation:03d}/proposals.json", {
                "generation": generation,
                "parent_policy": incumbent.to_dict(),
                "parent_score": parent_score.to_dict(),
                "requested_candidate_count": self.config.customer.candidate_count,
                "proposed_candidate_count": len(proposals),
                "generation_record": generation_record,
                "candidate_policies": [policy.to_dict() for policy in proposals],
            })
            self.store.append_jsonl(
                f"generations/gen_{generation:03d}/episodes.jsonl",
                [item.to_dict(include_analysis_artifacts=False) for item in evaluation_rows],
            )
            self.store.write_json(f"generations/gen_{generation:03d}/selection.json", {
                "generation": generation,
                "selection_status": "inconclusive",
                "reason": "incumbent_has_no_runtime_valid_official_score",
                "selected_policy_id": None,
                "candidate_scores": self._score_to_dict(scores),
            })
            raise CustomerEvolutionInconclusive(
                "incumbent_has_no_runtime_valid_official_score"
            )
        evaluation_rows = [episode for _, rows in evaluated for episode in rows]
        gen_metrics = self._score_summary(evaluation_rows)
        selected_score = score_by_id[selected.policy_id]
        phase_counts = {
            policy.policy_id: self._score_summary(rows)
            for policy, rows in evaluated
        }
        generation_record = copy.deepcopy(self.customer_evolver.last_generation_record)
        proposal_payload = {
            "generation": generation,
            "parent_policy": incumbent.to_dict(),
            "parent_score": parent_score.to_dict(),
            "requested_candidate_count": self.config.customer.candidate_count,
            "proposed_candidate_count": len(proposals),
            "generation_record": generation_record,
            "candidate_policies": [policy.to_dict() for policy in proposals],
        }
        self.store.write_json(f"generations/gen_{generation:03d}/proposals.json", proposal_payload)
        self.store.append_jsonl(
            f"generations/gen_{generation:03d}/episodes.jsonl",
            [item.to_dict(include_analysis_artifacts=False) for item in evaluation_rows],
        )
        selection_record = {
            **(copy.deepcopy(self.selector.last_selection_record) or {}),
            "generation": generation,
            "objective": "fitness = 1 - mean(official task_success) over runtime-evaluable episodes",
            "candidate_scores": self._score_to_dict(scores),
            "selected_policy": selected.to_dict(),
            "selected_score": selected_score.to_dict(),
            "per_policy_evaluation": phase_counts,
            "reused_selected_candidate_evaluation": True,
            "selected_episode_ids": [
                item.episode_id for policy, rows in evaluated
                if policy.policy_id == selected.policy_id for item in rows
            ],
        }
        self.store.write_json(f"generations/gen_{generation:03d}/selection.json", selection_record)
        self.store.write_json(f"generations/gen_{generation:03d}/customer_policy.json", selected.to_dict())
        self.store.write_json(f"generations/gen_{generation:03d}/service_policy.json", service.to_dict())
        request_metrics_after = self._request_metrics()
        generation_requests = max(
            0, request_metrics_after["provider_attempts"] - request_metrics_before["provider_attempts"]
        )
        generation_input_tokens = max(
            0, request_metrics_after["input_tokens"] - request_metrics_before["input_tokens"]
        )
        generation_output_tokens = max(
            0, request_metrics_after["output_tokens"] - request_metrics_before["output_tokens"]
        )
        generation_runtime_metrics = {
            "provider_requests": generation_requests,
            "input_tokens": generation_input_tokens,
            "output_tokens": generation_output_tokens,
            "tokens": generation_input_tokens + generation_output_tokens,
            "timeouts": max(0, request_metrics_after["timeouts"] - request_metrics_before["timeouts"]),
            "provider_failures": max(
                0, request_metrics_after["provider_failures"] - request_metrics_before["provider_failures"]
            ),
            "latency_seconds": max(
                0.0, request_metrics_after["latency_seconds"] - request_metrics_before["latency_seconds"]
            ),
        }
        payload = {
            "generation": generation,
            "status": "complete",
            "selected_policy_id": selected.policy_id,
            "incumbent_policy_id": incumbent.policy_id,
            "customer_changed": selected.policy_id != incumbent.policy_id,
            "selected_fitness": selected_score.fitness,
            "selected_official_task_success": selected_score.official_task_success,
            "requested_candidate_count": self.config.customer.candidate_count,
            "proposed_candidate_count": len(proposals),
            "evaluated_policy_count": len(evaluated),
            "evolution_evaluation": gen_metrics,
            "api_episode_runs": (1 + len(proposals)) * len(evolution_cases) * self.config.evaluation.repetitions,
            "runtime_metrics": generation_runtime_metrics,
            "wall_time_seconds": time.monotonic() - started,
        }
        self.store.write_json(f"generations/gen_{generation:03d}/SUMMARY.json", payload)
        self.store.mark_generation_complete(generation, payload)
        return selected, payload

    def _write_report(self, history, validation_summary, status):
        lines = [
            "# EvoSAGE Customer-only search run",
            "",
            f"- Status: `{status}`",
            f"- Run directory: `{self.run_dir}`",
            f"- Objective: `fitness = 1 - mean(official task_success)` on runtime-valid E episodes",
            f"- Fixed Service: `service_policy_s0`",
            f"- Validation: `{validation_summary.get('status', 'not_evaluated')}` (report-only)",
            "",
            "| Generation | Incumbent | Selected | Changed | Candidates | Runtime-valid episodes | Runtime-invalid episodes | Official task success | Fitness | Provider requests | Tokens |",
            "|---:|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for item in history:
            lines.append(
                f"| {item['generation']} | {item['incumbent_policy_id']} | {item['selected_policy_id']} | "
                f"{'yes' if item.get('customer_changed') else 'no'} | "
                f"{item['proposed_candidate_count']} | "
                f"{item.get('evolution_evaluation', {}).get('valid_episode_count', 0)} | "
                f"{item.get('evolution_evaluation', {}).get('invalid_episode_count', 0)} | "
                f"{_display(item.get('selected_official_task_success'))} | {_display(item.get('selected_fitness'))} | "
                f"{item.get('runtime_metrics', {}).get('provider_requests', 0)} | "
                f"{item.get('runtime_metrics', {}).get('tokens', 0)} |"
            )
        lines.append("")
        lines.append("Held-out data was not used or evaluated by this Customer-only runner.")
        (self.run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def run(self, fresh_run_isolated: bool = False):
        run_started = time.monotonic()
        completed = self.store.completed_generations() if self.config.persistence.resume else []
        start = max(completed) + 1 if completed else 0
        if self.config.persistence.resume:
            incomplete = sorted(
                int(path.name.split("_")[-1])
                for path in (self.run_dir / "generations").glob("gen_*")
                if path.is_dir()
                and (path / "COMPLETE.json").exists() is False
                and any(path.iterdir())
            )
            if incomplete:
                raise RuntimeError(
                    "incomplete Customer generation cannot be safely resumed; start a fresh run directory: "
                    + ",".join(map(str, incomplete))
                )
        manifest = self.run_dir / "split_manifest" / "evolution_cases.json"
        if completed and self.config.persistence.resume and manifest.exists():
            splits = SplitManager.load(self.run_dir / "split_manifest")
        else:
            splits = self.split_manager.build()
        evolution_cases = list(splits.evolution)
        validation_cases = list(splits.validation)
        if not evolution_cases:
            raise ValueError("Customer-only search requires at least one E/evolution case")
        self._write_run_provenance(splits, fresh_run_isolated=fresh_run_isolated)

        incumbent = self._load_incumbent(completed)
        service = ServicePolicy()
        initial_customer_path = self.run_dir / "environment" / "initial_customer_policy.json"
        initial_service_path = self.run_dir / "environment" / "initial_service_policy.json"
        if not initial_customer_path.exists():
            self.store.write_json("environment/initial_customer_policy.json", incumbent.to_dict())
        if not initial_service_path.exists():
            self.store.write_json("environment/initial_service_policy.json", service.to_dict())
        else:
            service = ServicePolicy.from_dict(json.loads(initial_service_path.read_text(encoding="utf-8")))
        baseline_service = ServicePolicy()
        if (
            service.policy_id != baseline_service.policy_id
            or service.semantic_fingerprint() != baseline_service.semantic_fingerprint()
        ):
            raise ValueError("Customer-only runs require the fixed baseline ServicePolicy S0")
        service_fingerprint = service.semantic_fingerprint()
        history = []
        for generation in range(start, self.config.max_generations):
            incumbent_before = incumbent
            try:
                incumbent, entry = self._run_generation(
                    generation, incumbent, service, evolution_cases,
                )
            except CustomerEvolutionInconclusive as exc:
                request_metrics = self._request_metrics()
                request_metrics.update({
                    "wall_time_seconds": time.monotonic() - run_started,
                    "run_status": "inconclusive",
                    "inconclusive_reason": str(exc),
                })
                self.store.write_json("analysis/orchestration_metrics.json", request_metrics)
                if self.request_budget is not None:
                    self.store.write_json("analysis/request_budget.json", self.request_budget.snapshot())
                self.store.write_json("analysis/run_status.json", {
                    "status": "inconclusive",
                    "completed_generations": self.store.completed_generations(),
                    "requested_generation_count": self.config.max_generations,
                    "inconclusive_generation": generation,
                    "reason": str(exc),
                    "heldout_evaluated": False,
                })
                self._write_report(history, {"status": "not_evaluated"}, "inconclusive")
                raise
            if service.semantic_fingerprint() != service_fingerprint:
                raise AssertionError("Customer-only runner must keep fixed Service S0")
            entry["incumbent_policy_id"] = incumbent_before.policy_id
            history.append(entry)
            self.store.write_json("analysis/trajectory.json", {
                "objective": "1 - mean(official task_success) over runtime-evaluable E episodes",
                "selection_uses_attribution_or_behavior_validity": False,
                "history": history,
            })

        validation_summary = {"status": "not_evaluated", "reason": "no validation cases"}
        if validation_cases:
            validation_rows = self.evaluator.evaluate(
                incumbent, service, validation_cases, "validation",
                max(0, self.config.max_generations - 1), "customer_validation_report_only",
            )
            self.store.append_jsonl(
                "analysis/validation_episodes.jsonl",
                [item.to_dict(include_analysis_artifacts=False) for item in validation_rows],
            )
            validation_summary = {"status": "reported_not_selected", **self._score_summary(validation_rows)}
            self.store.write_json("analysis/validation_summary.json", validation_summary)

        request_metrics = self._request_metrics()
        request_metrics["wall_time_seconds"] = time.monotonic() - run_started
        request_metrics["run_status"] = "complete"
        self.store.write_json("analysis/orchestration_metrics.json", request_metrics)
        if self.request_budget is not None:
            self.store.write_json("analysis/request_budget.json", self.request_budget.snapshot())
        self.store.write_json("analysis/run_status.json", {
            "status": "complete",
            "completed_generations": self.store.completed_generations(),
            "requested_generation_count": self.config.max_generations,
            "heldout_evaluated": False,
        })
        self._write_report(history, validation_summary, "complete")
        return {
            "run_dir": str(self.run_dir),
            "report": str(self.run_dir / "report.md"),
            "completed_generations": self.store.completed_generations(),
            "history": history,
            "orchestration_metrics": request_metrics,
            "validation_summary": validation_summary,
        }


def _display(value):
    return "—" if value is None else f"{float(value):.4f}"
