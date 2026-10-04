"""End-to-end tests for Customer strategy search against fixed Service S0."""

import json
from types import SimpleNamespace

import pytest

from framework.evolution.config import (
    CustomerEvolutionConfig,
    CustomerSearchConfig,
    SplitConfig,
    load_customer_search_config,
)
from framework.evolution.customer.evolver import CustomerEvolver, LLMAdversaryStrategyGenerator
from framework.evolution.customer.integrity import AdversaryPolicyValidator
from framework.evolution.customer.policy import AdversaryPolicy
from framework.evolution.customer.runner import CustomerEvolutionRunner
from framework.evolution.customer.selector import CustomerSelector
from framework.evolution.evaluator_adapter import MockEpisodeEvaluator
from framework.evolution.schemas import EpisodeResult, ServicePolicy


def _episode(policy_id, case_id, success, split="evolution", **kwargs):
    values = dict(
        episode_id=f"{policy_id}-{case_id}-{split}",
        scenario="ecommerce_refund",
        case_id=case_id,
        customer_policy_id=policy_id,
        service_policy_id="service_policy_s0",
        split=split,
        generation=0,
        task_success=success,
        execution_score=float(bool(success)),
    )
    values.update(kwargs)
    return EpisodeResult(**values)


def test_customer_evolver_proposes_bounded_free_text_candidates():
    incumbent = AdversaryPolicy(policy_id="parent", strategy="parent strategy")
    evolver = CustomerEvolver(seed=11)
    proposals = evolver.propose(incumbent, generation=1, count=3, parent_reward=0.25)
    assert len(proposals) == 3
    assert all(item.parent_id == "parent" and item.generation == 1 for item in proposals)
    assert all(item.strategy.strip() for item in proposals)
    assert evolver.last_generation_record["status"] == "valid"


def test_llm_generator_receives_only_parent_strategy_and_scalar_reward():
    class Client:
        def __init__(self):
            self.prompt = ""

        def generate(self, prompt="", **kwargs):
            self.prompt = prompt
            return SimpleNamespace(text='[{"strategy":"Lie about the order state.","hypothesis":"test"}]')

    client = Client()
    generator = LLMAdversaryStrategyGenerator(client)
    candidates = generator.generate(
        parent_strategy="Parent strategy", parent_id="p0", generation=1,
        count=2, parent_reward=0.25,
    )
    assert len(candidates) == 1
    assert "Parent strategy" in client.prompt
    assert "0.25" in client.prompt
    for forbidden in ("case-1", "CaseSpec: CASE-SECRET", "tool_arguments: secret", "expected_path: gold-path", "heldout answer: gold", "failure category: private"):
        assert forbidden not in client.prompt


def test_generator_records_candidate_rejection_separately_from_protocol_failure():
    class Client:
        def generate(self, **kwargs):
            return SimpleNamespace(text='[{"strategy":"", "hypothesis":"bad"}]')

    generator = LLMAdversaryStrategyGenerator(Client(), protocol_retries=0)
    candidates = generator.generate("parent", "p0", 1, 1, 0.5)
    assert candidates == []
    assert generator.last_generation_record["status"] == "candidate_rejected"
    assert generator.last_generation_record["candidates"][0]["schema_construction"]["status"] == "FAIL"


def test_smoke_config_contains_only_customer_search_method():
    config = load_customer_search_config("configs/customer_search_smoke.yaml")
    assert config.scenario == "ecommerce_refund"
    assert config.customer.candidate_count == 1
    assert "service" not in config.to_dict()
    assert "fresh_adversary" not in config.to_dict()
    assert "service_evolver" not in config.to_dict()["evaluation"]["token_budget"]


def test_customer_runner_completes_two_generations_against_unchanged_s0(tmp_path):
    config = CustomerSearchConfig(
        scenario="ecommerce_refund",
        seed=7,
        max_generations=2,
        customer=CustomerEvolutionConfig(candidate_count=2),
        splits=SplitConfig(seed=7, max_cases=6),
    )
    evaluator = MockEpisodeEvaluator()
    runner = CustomerEvolutionRunner(
        config, evaluator=evaluator, run_dir=tmp_path / "customer-search",
    )
    result = runner.run()

    assert result["completed_generations"] == [0, 1]
    assert len([call for call in evaluator.calls if call["split"] == "evolution"]) == 6
    assert len([call for call in evaluator.calls if call["split"] == "validation"]) == 1
    assert all(call["service_policy_id"] == ServicePolicy().policy_id for call in evaluator.calls)
    assert not any(call["split"] == "heldout_test" for call in evaluator.calls)

    for generation in range(2):
        gen_dir = tmp_path / "customer-search" / "generations" / f"gen_{generation:03d}"
        proposals = json.loads((gen_dir / "proposals.json").read_text())
        episodes = [json.loads(line) for line in (gen_dir / "episodes.jsonl").read_text().splitlines()]
        selection = json.loads((gen_dir / "selection.json").read_text())
        summary = json.loads((gen_dir / "SUMMARY.json").read_text())
        service = json.loads((gen_dir / "service_policy.json").read_text())
        assert proposals["proposed_candidate_count"] == 2
        assert len(proposals["candidate_policies"]) == 2
        assert len(episodes) == 3 * len(runner.split_manager.load(runner.run_dir / "split_manifest").evolution)
        assert all(row["service_policy_id"] == ServicePolicy().policy_id for row in episodes)
        assert "failure_signature" not in episodes[0]
        assert selection["selected_policy_id"]
        assert summary["status"] == "complete"
        assert service == ServicePolicy().to_dict()
        assert (gen_dir / "COMPLETE.json").exists()
        assert not (gen_dir / "service_gate.json").exists()

    assert not (tmp_path / "customer-search" / "archives").exists()
    assert not (tmp_path / "customer-search" / "analysis" / "weakness_frontier.json").exists()
    assert json.loads((tmp_path / "customer-search" / "analysis" / "run_status.json").read_text())["status"] == "complete"


def test_mock_evaluator_is_policy_agnostic_and_never_keyword_scores_strategy():
    evaluator = MockEpisodeEvaluator()
    policy = AdversaryPolicy(policy_id="p", strategy="backend result is wrong")
    case = SimpleNamespace(case_id="c", scenario="ecommerce_refund")
    episode = evaluator.evaluate(policy, ServicePolicy(), [case], "evolution", 0, "test")[0]
    assert episode.task_success is True
    assert episode.error_types == []
    assert episode.metadata["mock"] is True


def test_customer_candidate_evaluation_never_scores_heldout():
    incumbent = AdversaryPolicy(policy_id="p", strategy="strategy")
    with pytest.raises(AssertionError, match="heldout"):
        CustomerSelector().score(incumbent, [
            _episode(incumbent.policy_id, "h", False, split="heldout_test"),
        ])
