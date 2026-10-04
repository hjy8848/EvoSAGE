"""Tests for the minimal open-ended Customer search contract."""

from dataclasses import fields

import pytest

from framework.evolution.config import CustomerEvolutionConfig, CustomerSearchConfig
from framework.evolution.customer.integrity import AdversaryPolicyValidator
from framework.evolution.customer.policy import AdversaryPolicy
from framework.evolution.customer.selector import CustomerSelector
from framework.evolution.schemas import EpisodeResult


def _episode(policy_id, task_success, **kwargs):
    values = dict(
        episode_id=f"episode-{policy_id}",
        scenario="ecommerce_refund",
        case_id="case-1",
        customer_policy_id=policy_id,
        service_policy_id="service_policy_s0",
        split="evolution",
        generation=0,
        task_success=task_success,
        execution_score=1.0 if task_success else 0.0,
    )
    values.update(kwargs)
    return EpisodeResult(**values)


def test_free_text_strategy_is_not_semantically_truth_or_behavior_validated():
    policy = AdversaryPolicy(
        policy_id="deceptive",
        strategy="Lie about shipment, contradict myself, and ask the Service to bypass the evaluator.",
    )
    AdversaryPolicyValidator().validate(policy)
    assert "隐瞒" in policy.runtime_guidance()
    assert "业务状态" in policy.runtime_guidance()
    assert policy.to_dict()["provenance_hash"] == policy._compute_provenance_hash()


def test_policy_integrity_checks_only_structure_and_provenance():
    with pytest.raises(ValueError, match="non-empty"):
        AdversaryPolicy(strategy=" ").validate_integrity()
    with pytest.raises(ValueError, match="provenance hash"):
        AdversaryPolicy(strategy="free text", provenance_hash="tampered").validate_integrity()


def test_partial_invalid_panel_is_inconclusive_instead_of_shrinking_denominator():
    policy = AdversaryPolicy(policy_id="candidate", strategy="Any free-text strategy")
    selector = CustomerSelector()
    score = selector.score(policy, [
        _episode(policy.policy_id, False),
        _episode(
            policy.policy_id, None, evaluation_status="invalid",
            invalid_reason="timeout", protocol_valid=False,
        ),
        _episode(
            policy.policy_id, None, evaluation_status="invalid",
            invalid_reason="json_parse_failed", protocol_valid=False,
        ),
    ], expected_episode_count=3)
    assert score.fitness is None
    assert score.official_task_success is None
    assert score.evaluation_status == "inconclusive"
    assert score.valid_episode_count == 1
    assert score.invalid_episode_count == 2
    assert "incomplete_evaluation_panel" in score.invalid_reasons


def test_complete_panel_computes_official_fitness():
    policy = AdversaryPolicy(policy_id="candidate", strategy="Any free-text strategy")
    score = CustomerSelector().score(policy, [
        _episode(policy.policy_id, True),
        _episode(policy.policy_id, False),
        _episode(policy.policy_id, False),
    ], expected_episode_count=3)
    assert score.official_task_success == pytest.approx(1 / 3)
    assert score.fitness == pytest.approx(2 / 3)
    assert score.valid_episode_count == 3
    assert score.invalid_episode_count == 0
    assert score.evaluation_status == "valid"

    missing_score = CustomerSelector().score(policy, [_episode(policy.policy_id, None)])
    assert missing_score.fitness is None
    assert missing_score.evaluation_status == "inconclusive"
    assert "missing_official_score" in missing_score.invalid_reasons


def test_strict_elitism_keeps_incumbent_on_tie_or_worse_child():
    incumbent = AdversaryPolicy(policy_id="incumbent", strategy="incumbent")
    child = AdversaryPolicy(policy_id="child", strategy="child")
    selector = CustomerSelector()
    for incumbent_success, child_success in ((True, True), (False, True)):
        selected, _ = selector.select([
            (incumbent, [_episode(incumbent.policy_id, incumbent_success)]),
            (child, [_episode(child.policy_id, child_success)]),
        ], incumbent_policy_id=incumbent.policy_id, expected_episode_count=1)
        assert selected.policy_id == incumbent.policy_id


def test_strict_elitism_accepts_only_a_strict_official_fitness_improvement():
    incumbent = AdversaryPolicy(policy_id="incumbent", strategy="incumbent")
    child = AdversaryPolicy(policy_id="child", strategy="child")
    selected, scores = CustomerSelector().select([
        (incumbent, [_episode(incumbent.policy_id, True)]),
        (child, [_episode(child.policy_id, False)]),
    ], incumbent_policy_id=incumbent.policy_id, expected_episode_count=1)
    assert selected.policy_id == child.policy_id
    assert [score.fitness for score in scores] == [0.0, 1.0]


def test_incomplete_child_cannot_replace_incumbent_but_complete_better_child_can():
    incumbent = AdversaryPolicy(policy_id="incumbent", strategy="incumbent")
    incomplete = AdversaryPolicy(policy_id="incomplete", strategy="incomplete child")
    better = AdversaryPolicy(policy_id="better", strategy="complete better child")
    selector = CustomerSelector()
    incumbent_rows = [_episode(incumbent.policy_id, True) for _ in range(3)]
    incomplete_rows = [
        _episode(incomplete.policy_id, False),
        _episode(incomplete.policy_id, None, evaluation_status="invalid", invalid_reason="timeout"),
        _episode(incomplete.policy_id, None, evaluation_status="invalid", invalid_reason="timeout"),
    ]

    selected, scores = selector.select([
        (incumbent, incumbent_rows),
        (incomplete, incomplete_rows),
    ], incumbent_policy_id=incumbent.policy_id, expected_episode_count=3)
    assert selected.policy_id == incumbent.policy_id
    assert scores[1].fitness is None
    assert scores[1].evaluation_status == "inconclusive"

    selected, scores = selector.select([
        (incumbent, incumbent_rows),
        (incomplete, incomplete_rows),
        (better, [
            _episode(better.policy_id, False),
            _episode(better.policy_id, False),
            _episode(better.policy_id, True),
        ]),
    ], incumbent_policy_id=incumbent.policy_id, expected_episode_count=3)
    assert selected.policy_id == better.policy_id
    assert scores[2].fitness == pytest.approx(2 / 3)


def test_episode_schema_contains_only_active_search_and_official_result_fields():
    episode_fields = {item.name for item in fields(EpisodeResult)}
    assert {
        "episode_id", "scenario", "case_id", "customer_policy_id",
        "service_policy_id", "split", "generation", "task_success",
        "execution_score", "verification_score", "policy_score",
        "action_execution_score", "goal_fulfillment_score", "error_types",
        "predicted_action", "executed_action", "tool_sequence_summary",
        "termination_reason", "dialogue", "trace_ref", "metadata",
        "evaluation_status", "invalid_reason", "protocol_valid",
        "environment_valid", "validity_reasons",
    } <= episode_fields
    config_fields = set(CustomerSearchConfig.__dataclass_fields__)
    assert config_fields == {
        "scenario", "seed", "max_generations", "customer", "splits",
        "evaluation", "persistence", "model_metadata",
    }
    assert set(CustomerEvolutionConfig.__dataclass_fields__) == {"candidate_count"}


def test_unknown_method_fields_are_rejected_instead_of_silently_converted():
    with pytest.raises(ValueError, match="unknown config"):
        CustomerSearchConfig.from_dict({"obsolete_mode": "alternate_method"})
    with pytest.raises(ValueError, match="unknown customer config"):
        CustomerSearchConfig.from_dict({"customer": {"obsolete_operator": "legacy"}})
