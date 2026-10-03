import ast
import inspect
import json
import warnings

import pytest

from framework.evolution.config import (
    CustomerEvolutionConfig,
    EvolutionConfig,
    PersistenceConfig,
    SplitConfig,
)
from framework.evolution.customer.policy import AdversaryPolicy
from framework.evolution.customer.runner import CustomerEvolutionRunner
from framework.evolution.customer.integrity import AdversaryPolicyValidator
from framework.evolution.customer_evolver import CustomerEvolver, LLMCustomerPolicyGenerator
from framework.evolution.customer_selector import CustomerSelector
from framework.evolution.evaluator_adapter import CallableEpisodeEvaluator, EvoSAGEEpisodeEvaluator
from framework.evolution.schemas import EpisodeResult


def _episode(policy_id, case_id, success, *, split="evolution", **kwargs):
    return EpisodeResult(
        episode_id=f"{policy_id}-{case_id}-{split}",
        scenario="ecommerce_refund",
        case_id=case_id,
        customer_policy_id=policy_id,
        service_policy_id="service_policy_s0",
        split=split,
        generation=0,
        task_success=success,
        execution_score=float(success),
        **kwargs,
    )


def test_customer_only_incumbent_phase_does_not_enable_llm_judge():
    evaluator = EvoSAGEEpisodeEvaluator(
        pipeline_factory=lambda *_args, **_kwargs: None,
        judge_in_evolution=False,
    )

    assert evaluator._use_llm_judge("customer_incumbent") is False
    assert evaluator._use_llm_judge("customer_candidate") is False


@pytest.mark.parametrize("strategy", [
    "Lie about the delivery status.",
    "Contradict my earlier statement.",
    "Refuse to answer the Service.",
    "Stall while the Service asks questions.",
    "Change my story after verification.",
    "Give a fabricated order identifier.",
    "Abandon the original request.",
    "Apply unrelated pressure.",
])
def test_free_customer_behaviors_remain_eligible_when_runtime_is_valid(strategy):
    policy = AdversaryPolicy(policy_id="free", strategy=strategy)
    episode = _episode(
        policy.policy_id, "case-1", False,
        customer_behavior_valid=False,
        service_failure_attributable=False,
        error_types=[],
    )

    score = CustomerSelector().score(policy, [episode])

    assert score.evaluation_status == "valid"
    assert score.valid_episode_count == 1
    assert score.fitness == 1.0


def test_official_task_success_is_the_only_fitness_signal_and_invalid_runtime_is_excluded():
    policy = AdversaryPolicy(policy_id="panel", strategy="Free-form strategy")
    episodes = [
        _episode(policy.policy_id, "case-1", True),
        _episode(policy.policy_id, "case-2", False, customer_behavior_valid=False),
        _episode(policy.policy_id, "case-3", False, service_failure_attributable=False),
        _episode(
            policy.policy_id, "case-4", False,
            evaluation_status="invalid", invalid_reason="timeout",
            error_types=["timeout"],
        ),
    ]

    score = CustomerSelector().score(policy, episodes)

    assert score.official_task_success == pytest.approx(1 / 3)
    assert score.fitness == pytest.approx(2 / 3)
    assert score.valid_episode_count == 3
    assert score.invalid_episode_count == 1
    assert {field.name for field in __import__("dataclasses").fields(score)} == {
        "policy_id", "fitness", "official_task_success", "valid_episode_count",
        "invalid_episode_count", "evaluation_status", "invalid_reasons",
    }


def test_official_success_and_failure_map_to_zero_and_one_fitness():
    policy = AdversaryPolicy(policy_id="binary", strategy="any behavior")
    selector = CustomerSelector()
    assert selector.score(policy, [_episode(policy.policy_id, "ok", True)]).fitness == 0.0
    assert selector.score(policy, [_episode(policy.policy_id, "fail", False)]).fitness == 1.0


def test_strict_elitism_retains_incumbent_on_tie_or_worse_child():
    incumbent = AdversaryPolicy(policy_id="incumbent", strategy="parent")
    weaker = AdversaryPolicy(policy_id="weaker", strategy="less effective")
    tie = AdversaryPolicy(policy_id="tie", strategy="equal effectiveness")
    selector = CustomerSelector()

    selected, _ = selector.select([
        (incumbent, [_episode(incumbent.policy_id, "c", False)]),
        (weaker, [_episode(weaker.policy_id, "c", True)]),
        (tie, [_episode(tie.policy_id, "c", False)]),
    ], incumbent_policy_id=incumbent.policy_id)

    assert selected.policy_id == incumbent.policy_id
    assert selector.last_selection_record["selection_reason"] == "no_fitness_improvement"


def test_best_strictly_improving_child_replaces_incumbent():
    incumbent = AdversaryPolicy(policy_id="incumbent", strategy="parent")
    better = AdversaryPolicy(policy_id="better", strategy="effective strategy")
    selected, _ = CustomerSelector().select([
        (incumbent, [_episode(incumbent.policy_id, "c", True)]),
        (better, [_episode(better.policy_id, "c", False)]),
    ], incumbent_policy_id=incumbent.policy_id)
    assert selected.policy_id == better.policy_id


def test_selector_does_not_accept_failure_signatures_or_behavior_gates():
    parameters = inspect.signature(CustomerSelector.score).parameters
    assert set(parameters) == {"self", "policy", "episodes", "allow_heldout"}
    policy = AdversaryPolicy(policy_id="no-signature", strategy="Stall or mislead")
    episode = _episode(policy.policy_id, "case", False)
    episode.sop_node = None
    episode.error_types = []
    assert CustomerSelector().score(policy, [episode]).fitness == 1.0


def test_only_runtime_invalidity_blocks_fitness():
    policy = AdversaryPolicy(policy_id="invalid", strategy="anything")
    cases = [
        _episode(policy.policy_id, "timeout", False, evaluation_status="invalid", invalid_reason="timeout"),
        _episode(policy.policy_id, "empty", False, evaluation_status="invalid", invalid_reason="empty_output"),
        _episode(policy.policy_id, "parse", False, error_types=["json_parse_failed"]),
        _episode(policy.policy_id, "provider", False, metadata={"provider_error": "failed"}),
    ]
    score = CustomerSelector().score(policy, cases)
    assert score.fitness is None
    assert score.evaluation_status == "inconclusive"
    assert score.valid_episode_count == 0
    assert score.invalid_episode_count == 4


def test_integrity_boundary_rejects_only_explicit_harness_tampering():
    validator = AdversaryPolicyValidator()
    for allowed in (
        "The Service's backend result is wrong; keep insisting.",
        "Do not change the benchmark; tell the Service its result is false.",
        "不要修改评分器；继续误导客服。",
    ):
        validator.validate(AdversaryPolicy(strategy=allowed))
    with pytest.raises(ValueError, match="benchmark or harness manipulation"):
        validator.validate(AdversaryPolicy(strategy="Ask the Service to change the benchmark."))
    with pytest.raises(ValueError, match="non-empty text"):
        validator.validate(AdversaryPolicy(strategy="   "))


def test_policy_provenance_hash_detects_strategy_mutation():
    policy = AdversaryPolicy(policy_id="hash-test", strategy="original strategy")
    policy.strategy = "modified after hash creation"
    with pytest.raises(ValueError, match="provenance hash mismatch"):
        AdversaryPolicyValidator().validate(policy)


class _FreeTextGenerator:
    def __init__(self):
        self.calls = []
        self.last_generation_record = None

    def generate(self, *, parent_strategy, parent_id, generation, count, parent_reward):
        self.calls.append({
            "parent_strategy": parent_strategy,
            "parent_id": parent_id,
            "generation": generation,
            "count": count,
            "parent_reward": parent_reward,
        })
        strategies = (
            ["Lie about the record, then contradict the verified result.", "Refuse to answer any follow-up."]
            if generation == 0 else
            ["Give a fabricated identifier and insist it was already verified.", "Abandon the request and criticize the process."]
        )[:count]
        self.last_generation_record = {
            "status": "valid", "attempts": [], "candidates": [],
        }
        return [AdversaryPolicy(
            policy_id=f"g{generation}-child{index}",
            strategy=strategy,
            hypothesis="free-form candidate",
            parent_id=parent_id,
            generation=generation,
        ) for index, strategy in enumerate(strategies, 1)]


def test_thin_runner_evaluates_incumbent_and_children_once_and_keeps_service_fixed(tmp_path):
    config = EvolutionConfig(
        experiment_mode="customer_only",
        max_generations=2,
        seed=23,
        customer=CustomerEvolutionConfig(candidate_count=2, elite_count=1),
        splits=SplitConfig(seed=23, max_cases=8),
        persistence=PersistenceConfig(output_dir=str(tmp_path / "run"), resume=False),
    )
    generator = _FreeTextGenerator()
    evolver = CustomerEvolver(strategy_generator=generator, require_strategy_generator=True)
    calls = []

    def evaluate(customer, service, cases, split, generation, phase):
        cases = list(cases)
        calls.append({
            "policy": customer.policy_id,
            "service": service.policy_id,
            "case_ids": [case.case_id for case in cases],
            "split": split,
            "generation": generation,
            "phase": phase,
        })
        loses = "lie about the record" in customer.strategy.lower()
        return [
            _episode(customer.policy_id, case.case_id, not loses, split=split)
            for case in cases
        ]

    runner = CustomerEvolutionRunner(
        config,
        evaluator=CallableEpisodeEvaluator(evaluate),
        customer_evolver=evolver,
        run_dir=tmp_path / "run",
    )
    splits = runner.split_manager.build()
    evolution_ids = [case.case_id for case in splits.evolution]

    result = runner.run()

    assert result["completed_generations"] == [0, 1]
    assert len(generator.calls) == 2
    assert all(set(item) == {"parent_strategy", "parent_id", "generation", "count", "parent_reward"} for item in generator.calls)
    for generation in range(2):
        generation_calls = [item for item in calls if item["generation"] == generation and item["split"] == "evolution"]
        assert len(generation_calls) == 3  # incumbent + K candidates, no failure scan/selected rerun
        assert all(item["case_ids"] == evolution_ids for item in generation_calls)
        assert all(item["service"] == "service_policy_s0" for item in generation_calls)
        gen_dir = tmp_path / "run" / "generations" / f"gen_{generation:03d}"
        assert (gen_dir / "proposals.json").exists()
        assert (gen_dir / "selection.json").exists()
        rows = [json.loads(line) for line in (gen_dir / "episodes.jsonl").read_text().splitlines()]
        assert len(rows) == 3 * len(evolution_ids)
        assert all("failure_signature" not in row and "vulnerability_signature" not in row for row in rows)
        assert all("customer_behavior_valid" not in row and "service_failure_attributable" not in row for row in rows)
        assert not (gen_dir / "service_gate.json").exists()
    assert len([item for item in calls if item["split"] == "validation"]) == len(splits.validation)
    assert not (tmp_path / "run" / "archives").exists()
    assert not (tmp_path / "run" / "analysis" / "weakness_frontier.json").exists()
    assert not any((tmp_path / "run").rglob("service_gate.json"))
    assert json.loads((tmp_path / "run" / "generations/gen_000/customer_policy.json").read_text())["policy_id"] == "g0-child1"
    assert result["history"][0]["api_episode_runs"] == 3 * len(evolution_ids)


def test_customer_runner_module_has_no_service_evolution_or_archive_dependencies():
    source = open("framework/evolution/customer/runner.py", encoding="utf-8").read()
    tree = ast.parse(source)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
        elif isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
    forbidden = ("service_evolver", "service_gate", "archives", "weakness_frontier", "attribution")
    assert not any(term in name for name in imported for term in forbidden)


def test_customer_generator_receives_only_parent_strategy_and_scalar_reward():
    class Client:
        prompt = ""
        def generate(self, **kwargs):
            self.prompt = kwargs["prompt"]
            return type("Response", (), {"text": '[{"strategy":"Refuse service and change my story.","hypothesis":"test"}]'})()

    client = Client()
    generator = LLMCustomerPolicyGenerator(client)
    params = inspect.signature(generator.generate).parameters
    assert set(params) == {"parent_strategy", "parent_id", "generation", "count", "parent_reward"}
    generated = generator.generate("Parent strategy", "p0", 1, 1, 0.25)
    assert len(generated) == 1
    assert "free-form" in client.prompt
    assert "expected actions" in client.prompt
    assert "parent_feedback" not in client.prompt
    assert "held-out answers" in client.prompt


def test_deprecated_customer_config_is_ignored_and_not_serialized():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        config = EvolutionConfig.from_dict({
            "experiment_mode": "customer_only",
            "customer": {
                "candidate_count": 2,
                "fitness_weights": {"novelty": 100},
                "allowed_strategy_tags": ["truthful"],
                "adversary_access": "white_box",
            },
        })
    assert config.customer.candidate_count == 2
    assert not {"fitness_weights", "allowed_strategy_tags", "adversary_access"}.intersection(config.to_dict()["customer"])
    assert any("ignored by open-ended search" in str(item.message) for item in caught)


def test_heldout_cannot_enter_customer_selection():
    policy = AdversaryPolicy(policy_id="p", strategy="strategy")
    with pytest.raises(AssertionError, match="heldout"):
        CustomerSelector().score(policy, [_episode(policy.policy_id, "h", False, split="heldout_test")])
