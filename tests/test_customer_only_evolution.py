import json

from framework.evolution.config import (
    CustomerEvolutionConfig,
    EvolutionConfig,
    PersistenceConfig,
    SplitConfig,
)
from framework.evolution.customer_evolver import CustomerEvolver
from framework.evolution.evaluator_adapter import CallableEpisodeEvaluator
from framework.evolution.runner import EvolutionRunner
from framework.evolution.schemas import AdversaryPolicy, CustomerPolicy, EpisodeResult


class _FreeTextStrategyGenerator:
    def __init__(self):
        self.calls = []
        self.last_generation_record = None

    def generate(self, *, parent_strategy, parent_id, generation, count, parent_reward, **kwargs):
        self.calls.append({
            "generation": generation,
            "count": count,
            "parent_reward": parent_reward,
            "parent_strategy": parent_strategy,
            "parent_id": parent_id,
            "extra": dict(kwargs),
        })
        strategies = (
            [
                "Claim the order has not shipped; after a backend result contradicts this, insist the lookup was wrong.",
                "Ask for an update while remaining neutral and do not assert a backend fact.",
            ]
            if generation == 0 else [
                "Use a polite but persistent request for an exception without making a false status claim.",
                "Rephrase the request and ask the Service to explain its decision.",
            ]
        )[:count]
        self.last_generation_record = {
            "status": "valid",
            "attempts": [{"parse_status": "PASS"}],
            "candidates": [],
        }
        return [
            AdversaryPolicy(
                policy_id=f"free-g{generation}-c{index + 1}",
                strategy=strategy,
                hypothesis="test whether the fixed Service can be induced to fail",
                parent_id=parent_id,
                generation=generation,
            )
            for index, strategy in enumerate(strategies)
        ]


def test_customer_only_evolution_uses_fixed_service_and_same_full_evolution_panel(tmp_path):
    config = EvolutionConfig(
        experiment_mode="customer_only",
        max_generations=2,
        seed=23,
        customer=CustomerEvolutionConfig(
            candidate_count=2, elite_count=1, cases_per_candidate=1,
        ),
        splits=SplitConfig(seed=23, max_cases=8),
        persistence=PersistenceConfig(output_dir=str(tmp_path / "customer-only"), resume=False),
    )
    calls = []

    def evaluate(customer, service, cases, split, generation, phase):
        cases = list(cases)
        calls.append({
            "customer_policy_id": customer.policy_id,
            "service_policy_id": service.policy_id,
            "case_ids": [case.case_id for case in cases],
            "split": split,
            "generation": generation,
            "phase": phase,
        })
        attack_succeeds = "insist the lookup was wrong" in customer.strategy.lower()
        return [
            EpisodeResult(
                episode_id=f"{generation}-{phase}-{customer.policy_id}-{case.case_id}",
                scenario=case.case_spec["scenario"],
                case_id=case.case_id,
                customer_policy_id=customer.policy_id,
                service_policy_id=service.policy_id,
                split=split,
                generation=generation,
                task_success=not attack_succeeds,
                execution_score=0.0 if attack_succeeds else 1.0,
                error_types=["wrong_final_action"] if attack_succeeds else [],
                predicted_action="Refund" if attack_succeeds else "Refund",
                executed_action="" if attack_succeeds else "Refund",
                termination_reason="goal_not_fulfilled" if attack_succeeds else "goal_satisfied",
                service_failure_attributable=attack_succeeds,
                metadata={"phase": phase},
            )
            for case in cases
        ]

    generator = _FreeTextStrategyGenerator()
    evolver = CustomerEvolver(
        seed=23,
        strategy_generator=generator,
        require_strategy_generator=True,
    )
    runner = EvolutionRunner(
        config,
        evaluator=CallableEpisodeEvaluator(evaluate),
        customer_evolver=evolver,
        run_dir=tmp_path / "customer-only",
    )
    splits = runner.split_manager.build()
    evolution_ids = [case.case_id for case in splits.evolution]

    result = runner.run()

    assert result["completed_generations"] == [0, 1]
    assert len(generator.calls) == 2
    assert all(call["count"] == 2 for call in generator.calls)
    assert all(call["extra"] == {} for call in generator.calls)
    assert all(call["case_ids"] == evolution_ids for call in calls if call["phase"] in {
        "customer_failure_scan", "customer_candidate", "selected_customer",
    })
    assert all(call["service_policy_id"] == "service_policy_s0" for call in calls)
    assert not (tmp_path / "customer-only" / "archives" / "attacks.jsonl").exists()
    assert not (tmp_path / "customer-only" / "archives" / "defenses.jsonl").exists()
    assert not (tmp_path / "customer-only" / "analysis" / "weakness_frontier.json").exists()

    for generation in range(2):
        generation_dir = tmp_path / "customer-only" / "generations" / f"gen_{generation:03d}"
        candidates = json.loads((generation_dir / "customer_candidates.json").read_text())
        policy = json.loads((generation_dir / "service_policy.json").read_text())
        assert not (generation_dir / "service_gate.json").exists()
        assert candidates["fixed_service_policy_id"] == "service_policy_s0"
        assert candidates["evaluation_case_ids"] == evolution_ids
        assert candidates["candidate_count"] == 2
        assert all(item["policy"]["strategy"] for item in candidates["candidate_policies"])
        assert all(item["policy"]["provenance_hash"] for item in candidates["candidate_policies"])
        assert policy["policy_id"] == "service_policy_s0"
