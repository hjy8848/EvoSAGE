"""Orchestrator for static, customer-only, service-only and coevolution modes."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any, Optional

from .archives import AttackArchive, DefenseArchive
from .config import EvolutionConfig
from .customer_evolver import CustomerEvolver
from .customer_policy import CustomerPolicyValidator
from .customer_selector import CustomerSelector
from .evaluator_adapter import BudgetedEpisodeEvaluator, MockEpisodeEvaluator, aggregate_episode_metrics
from .persistence import RunStore
from .reporting import generate_report
from .schemas import CustomerPolicy, DefenseRecord, FailureSignature, ServicePolicy
from .service_evolver import ServiceEvolver
from .service_policy import ServicePolicySanitizer
from .service_gate import ServiceGate
from .split_manager import SplitManager
from .weakness_frontier import WeaknessFrontier


class EvolutionRunner:
    def __init__(self, config: Optional[EvolutionConfig] = None, evaluator=None, store: Optional[RunStore] = None,
                 split_manager: Optional[SplitManager] = None, customer_evolver=None, service_evolver=None):
        self.config = config or EvolutionConfig()
        self.store = store or RunStore(self.config.persistence.output_dir)
        self.split_manager = split_manager or SplitManager(self.config.splits, self.store.run_dir / "split_manifest")
        self.base_evaluator = evaluator or MockEpisodeEvaluator()
        self.evaluator = BudgetedEpisodeEvaluator(
            self.base_evaluator,
            repetitions=self.config.evaluation.repetitions,
            concurrency=self.config.evaluation.concurrency,
        )
        self.customer_evolver = customer_evolver or CustomerEvolver(
            self.config.seed,
            validator=CustomerPolicyValidator(self.config.customer.allowed_strategy_tags),
            selector=CustomerSelector(self.config.customer.fitness_weights),
        )
        self.service_evolver = service_evolver or ServiceEvolver(
            self.config.seed,
            sanitizer=ServicePolicySanitizer(self.config.service.allowed_rule_categories),
            gate=ServiceGate(self.config.service.min_delta, self.config.service.normal_regression_tolerance),
        )
        self.attack_archive = AttackArchive(self.store.run_dir / "archives" / "attacks.jsonl")
        self.defense_archive = DefenseArchive(self.store.run_dir / "archives" / "defenses.jsonl")
        self.frontier = WeaknessFrontier()
        self._generation_wall_times: dict[str, float] = {}

    @staticmethod
    def _client_request_stats(client) -> dict[str, Any]:
        if client is None:
            return {"requests": 0, "retries": 0, "input_tokens": 0, "output_tokens": 0, "latency_seconds": 0.0}
        if hasattr(client, "request_stats"):
            value = client.request_stats()
            return {
                "requests": int(value.get("requests", 0) or 0),
                "retries": int(value.get("retries", 0) or 0),
                "input_tokens": int(value.get("input_tokens", 0) or 0),
                "output_tokens": int(value.get("output_tokens", 0) or 0),
                "latency_seconds": float(value.get("latency_seconds", 0.0) or 0.0),
            }
        return {
            "requests": int(getattr(client, "request_count", 0) or 0),
            "retries": int(getattr(client, "retry_count", 0) or 0),
            "input_tokens": int(getattr(client, "input_tokens", 0) or 0),
            "output_tokens": int(getattr(client, "output_tokens", 0) or 0),
            "latency_seconds": float(getattr(client, "latency_seconds", 0.0) or 0.0),
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
            stats["input_tokens"] += int(adapter_stats.get("input_tokens", 0) or 0)
            stats["output_tokens"] += int(adapter_stats.get("output_tokens", 0) or 0)
            stats["latency_seconds"] += float(adapter_stats.get("latency_seconds", 0.0) or 0.0)
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
            stats["input_tokens"] += client_stats["input_tokens"]
            stats["output_tokens"] += client_stats["output_tokens"]
            stats["latency_seconds"] += client_stats["latency_seconds"]

        stats["llm_requests"] = stats["pipeline_requests"] + stats["policy_generation_requests"]
        stats["total_requests_including_retries"] = stats["llm_requests"]
        stats["generation_wall_times_seconds"] = dict(self._generation_wall_times)
        return stats

    def run(self) -> dict[str, Any]:
        run_started = time.monotonic()
        manifest_exists = (self.store.run_dir / "split_manifest" / "evolution_cases.json").exists()
        # A fresh run rebuilds the manifest from the current config.  Only an
        # explicit resume reuses an existing split, preventing a changed
        # instances_per_path/seed from silently running against stale data.
        splits = SplitManager.load(self.store.run_dir / "split_manifest") if (manifest_exists and self.config.persistence.resume) else self.split_manager.build()
        self.store.write_json("config/evolution.json", self.config.to_dict())
        self.store.write_json("environment/provenance.json", {
            "scenario": self.config.scenario,
            "seed": self.config.seed,
            "mode": self.config.experiment_mode,
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
            "customer_adversary_access": self.config.customer.adversary_access,
        })
        completed = self.store.completed_generations()
        start = (max(completed) + 1) if self.config.persistence.resume and completed else 0
        customer = self._load_policy("customer_policy", CustomerPolicy())
        service = self._load_policy("service_policy", ServicePolicy())
        if not self.config.persistence.resume or not (self.store.run_dir / "environment/initial_service_policy.json").exists():
            self.store.write_json("environment/initial_customer_policy.json", CustomerPolicy().to_dict())
            self.store.write_json("environment/initial_service_policy.json", ServicePolicy().to_dict())
        history = []
        for generation in range(start, self.config.max_generations):
            generation_started = time.monotonic()
            gen_dir = self.store.generation_dir(generation)
            baseline_customer = customer
            baseline_service = service
            if self.config.experiment_mode in {"customer_only", "coevolution"} and self.config.experiment_mode != "static":
                prior_customer_episodes = self.evaluator.evaluate(
                    customer, service, splits.evolution, "evolution", generation, "customer_failure_scan"
                )
                prior_failures = [FailureSignature.from_episode(item) for item in prior_customer_episodes if not item.task_success]
                customer, candidate_records, scores = self.customer_evolver.evolve(
                    customer, service, splits.evolution, self.evaluator, self.attack_archive, generation,
                    self.config.customer.candidate_count,
                    cases_per_candidate=self.config.customer.cases_per_candidate,
                    elite_count=self.config.customer.elite_count,
                    source_failures=prior_failures,
                    frontier=self.frontier.to_dicts()[-max(1, self.config.evaluation.summary_limit):],
                    archive_summary=self.attack_archive.to_dicts()[-max(1, self.config.evaluation.summary_limit):],
                )
                self.store.write_json(f"generations/gen_{generation:03d}/customer_candidates.json", {
                    "selected_policy": customer.to_dict(), "scores": [score.to_dict() for score in scores],
                    "candidate_episode_counts": [len(items) for _, items in candidate_records],
                    "rejections": list(getattr(self.customer_evolver, "last_rejections", [])),
                    "source_failures": [failure.to_dict() for failure in prior_failures],
                })
                selected_episodes = self.evaluator.evaluate(customer, service, splits.evolution, "evolution", generation, "selected_customer")
                signatures = [FailureSignature.from_episode(item) for item in selected_episodes if not item.task_success]
                self.attack_archive.add(customer, signatures, selected_episodes, generation)
                self.frontier.add(selected_episodes)
            if self.config.experiment_mode in {"service_only", "coevolution"} and self.config.experiment_mode != "static":
                failures = [FailureSignature.from_episode(item) for item in self.evaluator.evaluate(customer, service, splits.evolution, "evolution", generation, "service_failures") if not item.task_success]
                service, decision, patch = self.service_evolver.evolve(
                    service, failures, splits.validation, splits.validation, self.evaluator, generation,
                    self.config.service.candidate_count,
                    customer_policy=customer,
                    replay_policies=self._replay_policies(self.config.service.replay_attack_count),
                    replay_attack_count=self.config.service.replay_attack_count,
                    defense_summary=self.defense_archive.to_dicts()[-max(1, self.config.evaluation.summary_limit):],
                    historical_summary=self.service_evolver.last_candidate_records[-max(1, self.config.evaluation.summary_limit):],
                )
                self.store.write_json(f"generations/gen_{generation:03d}/service_gate.json", {
                    "accepted": decision.accepted, "reason": decision.reason, "delta": decision.delta,
                    "patch": patch.to_dict() if patch else None,
                    "candidates": list(getattr(self.service_evolver, "last_candidate_records", [])),
                })
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
                        robust_delta={"task_success": decision.metrics.get("robust_task_success", 0.0) - before.get("robust_task_success", 0.0)},
                        regression_cases=[case for item in getattr(self.service_evolver, "last_candidate_records", []) if item.get("accepted") and item.get("patch_id") == patch.patch_id for case in item.get("normal_regression_cases", [])],
                    ))
            episodes = self.evaluator.evaluate(customer, service, splits.validation, "validation", generation, "generation_summary")
            self.frontier.add(episodes)
            self.store.append_jsonl(f"generations/gen_{generation:03d}/episodes.jsonl", [item.to_dict() for item in episodes])
            self.store.write_json(f"generations/gen_{generation:03d}/customer_policy.json", customer.to_dict())
            self.store.write_json(f"generations/gen_{generation:03d}/service_policy.json", service.to_dict())
            self._generation_wall_times[str(generation)] = time.monotonic() - generation_started
            self.store.mark_generation_complete(generation, {
                "episode_count": len(episodes),
                "service_policy_id": service.policy_id,
                "customer_policy_id": customer.policy_id,
                "orchestration": self._runtime_stats(),
            })
            history.append({"generation": generation, "episode_count": len(episodes), "task_success": sum(item.task_success for item in episodes) / len(episodes) if episodes else 0.0})
        self.frontier.save(self.store.run_dir / "analysis" / "weakness_frontier.json", self.store.run_dir / "analysis" / "weakness_frontier.csv")
        if self.config.fresh_adversary.enabled:
            self.fresh_adversary_evaluation(
                rounds=self.config.fresh_adversary.rounds,
                candidate_count=self.config.fresh_adversary.candidate_count,
                target_service=service,
                target_label="final_coevolved",
            )
        orchestration_metrics = self._runtime_stats()
        orchestration_metrics["wall_time_seconds"] = time.monotonic() - run_started
        self.store.write_json("analysis/orchestration_metrics.json", orchestration_metrics)
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
                                   target_service=None, target_label: str = "final"):
        """Evaluate newly proposed customer strategies without archive feedback."""
        splits = SplitManager.load(self.store.run_dir / "split_manifest")
        service = target_service or self._load_policy("service_policy", ServicePolicy())
        incumbent = CustomerPolicy()
        results = []
        adaptation_results = []
        selector = CustomerSelector(self.config.customer.fitness_weights)
        fresh_evolver = CustomerEvolver(
            seed=self.config.seed + 10_000,
            validator=getattr(self.customer_evolver, "validator", None),
            selector=selector,
            strategy_generator=getattr(self.customer_evolver, "strategy_generator", None),
            require_strategy_generator=getattr(self.customer_evolver, "require_strategy_generator", False),
        )
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
            selected, scores = selector.select(evaluated, set(), total_nodes=max(1, len(splits.validation)))
            incumbent = selected or incumbent
            heldout_episodes = self.evaluator.evaluate(incumbent, service, splits.heldout_test, "heldout_test", generation, "fresh_adversary_eval")
            for episode in heldout_episodes:
                episode.metadata = dict(episode.metadata, fresh_round=generation, target_service=target_label, adaptation_split="validation")
            results.extend(heldout_episodes)
            selected_scores = next((score.to_dict() for score in scores if score.policy_id == incumbent.policy_id), {})
            round_records.append({"round": generation, "target_service": target_label,
                                  "selected_policy": incumbent.to_dict(), "fitness": selected_scores})
        self.store.write_json(f"analysis/fresh_adversary_{target_label}.json", {
            "evaluator": "mock" if isinstance(self.base_evaluator, MockEpisodeEvaluator) else "real",
            "target_service": target_label,
            "results": [item.to_dict() for item in results],
            "adaptation_results": [item.to_dict() for item in adaptation_results],
            "rounds": round_records,
            "training_archive_used": False,
            "customer_generator": "llm" if fresh_evolver.strategy_generator is not None else "template",
            "strict_real_generation": bool(fresh_evolver.require_strategy_generator),
        })
        # Keep a stable aggregate path for existing tooling.
        self.store.write_json("analysis/fresh_adversary_results.json", {
            "results": [item.to_dict() for item in results],
            "target_service": target_label,
            "evaluator": "mock" if isinstance(self.base_evaluator, MockEpisodeEvaluator) else "real",
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
        records = sorted(self.attack_archive.to_dicts(), key=lambda item: item.get("generation_discovered", item.get("generation", 0)), reverse=True)
        selected = []
        seen = set()
        for record in records:
            policy_data = record.get("customer_policy")
            if not policy_data:
                continue
            signature = (tuple(sorted(record.get("strategy_tags", []))), tuple(sorted(record.get("induced_error_types", []))))
            if signature in seen and len(selected) < count:
                continue
            seen.add(signature)
            selected.append(CustomerPolicy.from_dict(policy_data))
            if len(selected) >= count:
                break
        return selected
