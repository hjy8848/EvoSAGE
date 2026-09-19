"""Orchestrator for static, customer-only, service-only and coevolution modes."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from .archives import AttackArchive, DefenseArchive
from .config import EvolutionConfig
from .customer_evolver import CustomerEvolver
from .evaluator_adapter import MockEpisodeEvaluator
from .persistence import RunStore
from .reporting import generate_report
from .schemas import CustomerPolicy, DefenseRecord, FailureSignature, ServicePolicy
from .service_evolver import ServiceEvolver
from .service_gate import ServiceGate
from .split_manager import SplitManager
from .weakness_frontier import WeaknessFrontier


class EvolutionRunner:
    def __init__(self, config: Optional[EvolutionConfig] = None, evaluator=None, store: Optional[RunStore] = None,
                 split_manager: Optional[SplitManager] = None, customer_evolver=None, service_evolver=None):
        self.config = config or EvolutionConfig()
        self.store = store or RunStore(self.config.persistence.output_dir)
        self.split_manager = split_manager or SplitManager(self.config.splits, self.store.run_dir / "split_manifest")
        self.evaluator = evaluator or MockEpisodeEvaluator()
        self.customer_evolver = customer_evolver or CustomerEvolver(self.config.seed)
        self.service_evolver = service_evolver or ServiceEvolver(
            self.config.seed,
            gate=ServiceGate(self.config.service.min_delta, self.config.service.normal_regression_tolerance),
        )
        self.attack_archive = AttackArchive(self.store.run_dir / "archives" / "attacks.jsonl")
        self.defense_archive = DefenseArchive(self.store.run_dir / "archives" / "defenses.jsonl")
        self.frontier = WeaknessFrontier()

    def run(self) -> dict[str, Any]:
        splits = self.split_manager.build() if not (self.store.run_dir / "split_manifest" / "evolution_cases.json").exists() else SplitManager.load(self.store.run_dir / "split_manifest")
        self.store.write_json("config/evolution.json", self.config.to_dict())
        self.store.write_json("environment/provenance.json", {"scenario": self.config.scenario, "seed": self.config.seed, "mode": self.config.experiment_mode, "note": "mock metrics are fixture metrics" if isinstance(self.evaluator, MockEpisodeEvaluator) else "real evaluator"})
        completed = self.store.completed_generations()
        start = (max(completed) + 1) if self.config.persistence.resume and completed else 0
        customer = self._load_policy("customer_policy", CustomerPolicy())
        service = self._load_policy("service_policy", ServicePolicy())
        history = []
        for generation in range(start, self.config.max_generations):
            gen_dir = self.store.generation_dir(generation)
            baseline_customer = customer
            baseline_service = service
            if self.config.experiment_mode in {"customer_only", "coevolution"} and self.config.experiment_mode != "static":
                customer, candidate_records, scores = self.customer_evolver.evolve(
                    customer, service, splits.evolution, self.evaluator, self.attack_archive, generation,
                    self.config.customer.candidate_count)
                self.store.write_json(f"generations/gen_{generation:03d}/customer_candidates.json", {
                    "selected_policy": customer.to_dict(), "scores": [score.to_dict() for score in scores],
                    "candidate_episode_counts": [len(items) for _, items in candidate_records],
                })
                selected_episodes = self.evaluator.evaluate(customer, service, splits.evolution, "evolution", generation, "selected_customer")
                signatures = [FailureSignature.from_episode(item) for item in selected_episodes if not item.task_success]
                self.attack_archive.add(customer, signatures, selected_episodes, generation)
                self.frontier.add(selected_episodes)
            if self.config.experiment_mode in {"service_only", "coevolution"} and self.config.experiment_mode != "static":
                failures = [FailureSignature.from_episode(item) for item in self.evaluator.evaluate(customer, service, splits.evolution, "evolution", generation, "service_failures") if not item.task_success]
                service, decision, patch = self.service_evolver.evolve(
                    service, failures, splits.validation, splits.evolution, self.evaluator, generation,
                    self.config.service.candidate_count, customer_policy=customer)
                self.store.write_json(f"generations/gen_{generation:03d}/service_gate.json", {
                    "accepted": decision.accepted, "reason": decision.reason, "delta": decision.delta,
                    "patch": patch.to_dict() if patch else None,
                })
                if decision.accepted and patch:
                    self.defense_archive.add(DefenseRecord(
                        defense_id=f"defense_g{generation}", service_policy_id=service.policy_id,
                        generation_added=generation, rule_ids=[rule.rule_id for rule in patch.rules],
                        addresses_failure_signatures=[failure.signature_id for failure in failures],
                        validation_delta={"task_success": decision.delta}, normal_user_delta=decision.metrics,
                        adversarial_delta=decision.metrics,
                    ))
            episodes = self.evaluator.evaluate(customer, service, splits.validation, "validation", generation, "generation_summary")
            self.frontier.add(episodes)
            self.store.append_jsonl(f"generations/gen_{generation:03d}/episodes.jsonl", [item.to_dict() for item in episodes])
            self.store.write_json(f"generations/gen_{generation:03d}/customer_policy.json", customer.to_dict())
            self.store.write_json(f"generations/gen_{generation:03d}/service_policy.json", service.to_dict())
            self.store.mark_generation_complete(generation, {"episode_count": len(episodes), "service_policy_id": service.policy_id, "customer_policy_id": customer.policy_id})
            history.append({"generation": generation, "episode_count": len(episodes), "task_success": sum(item.task_success for item in episodes) / len(episodes) if episodes else 0.0})
        self.frontier.save(self.store.run_dir / "analysis" / "weakness_frontier.json", self.store.run_dir / "analysis" / "weakness_frontier.csv")
        report = generate_report(self.store.run_dir)
        return {"run_dir": str(self.store.run_dir), "report": str(report), "history": history, "completed_generations": self.store.completed_generations()}

    def heldout_evaluation(self):
        splits = SplitManager.load(self.store.run_dir / "split_manifest")
        customer = self._load_policy("customer_policy", CustomerPolicy())
        service = self._load_policy("service_policy", ServicePolicy())
        generation = max(self.store.completed_generations(), default=0)
        results = self.evaluator.evaluate(customer, service, splits.heldout_test, "heldout_test", generation, "heldout")
        self.store.write_json("analysis/heldout_results.json", {"results": [item.to_dict() for item in results]})
        return results

    def fresh_adversary_evaluation(self, rounds: int = 2, candidate_count: int = 5):
        """Evaluate newly proposed customer strategies without archive feedback."""
        splits = SplitManager.load(self.store.run_dir / "split_manifest")
        service = self._load_policy("service_policy", ServicePolicy())
        incumbent = CustomerPolicy()
        results = []
        for generation in range(rounds):
            candidates = self.customer_evolver.propose(incumbent, 10_000 + generation, candidate_count)
            for policy in candidates:
                results.extend(self.evaluator.evaluate(policy, service, splits.heldout_test, "heldout_test", generation, "fresh_adversary"))
        self.store.write_json("analysis/fresh_adversary_results.json", {"results": [item.to_dict() for item in results]})
        return results

    def cross_generation_evaluation(self):
        splits = SplitManager.load(self.store.run_dir / "split_manifest")
        generations = self.store.completed_generations()
        matrix = []
        for customer_generation in generations:
            customer = CustomerPolicy.from_dict(self.store.read_generation(customer_generation, "customer_policy"))
            for service_generation in generations:
                service = ServicePolicy.from_dict(self.store.read_generation(service_generation, "service_policy"))
                results = self.evaluator.evaluate(customer, service, splits.validation, "validation", customer_generation, "cross_generation")
                matrix.append({"customer_generation": customer_generation, "service_generation": service_generation, "metrics": {"task_success": sum(item.task_success for item in results) / len(results) if results else 0.0}})
        self.store.write_json("analysis/cross_generation_matrix.json", matrix)
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
