import pytest

from framework.evolution.evaluator_adapter import BudgetedEpisodeEvaluator
from framework.evolution.paired_stats import PairingSetMismatch, compare_paired_episodes
from framework.evolution.schemas import CustomerPolicy, EpisodeResult, ServicePolicy
from framework.evolution.service_evolver import candidate_selection_key
from framework.evolution.service_gate import ServiceGate


def _episode(case_id, success, *, repetition=0, customer="customer", valid=True,
             execution_score=None, service="service"):
    return EpisodeResult(
        episode_id=f"{case_id}-{repetition}-{service}",
        scenario="ecommerce_refund",
        case_id=case_id,
        customer_policy_id=customer,
        service_policy_id=service,
        split="validation",
        generation=0,
        task_success=success,
        execution_score=float(success) if execution_score is None else execution_score,
        customer_behavior_valid=valid,
        service_failure_attributable=not success,
        metadata={"repetition": repetition},
    )


def test_paired_gate_rejects_zero_net_even_when_aggregate_delta_is_equal():
    before = [_episode("a", False), _episode("b", True)]
    after = [_episode("a", True, service="candidate"), _episode("b", False, service="candidate")]
    paired = compare_paired_episodes(before, after)

    decision = ServiceGate(min_delta=0.0).evaluate(
        {"task_success": 0.5}, {"task_success": 0.5}, paired_latest=paired
    )

    assert paired.wins == paired.losses == 1
    assert decision.accepted is False
    assert decision.reason == "insufficient_paired_adversarial_improvement"
    assert decision.metrics["latest_paired"]["matched_pairs"] == 2


def test_pairing_requires_identical_valid_case_customer_repetition_set():
    baseline = [_episode("case-1", False), _episode("case-2", True)]
    candidate = [_episode("case-1", True, service="candidate")]
    with pytest.raises(PairingSetMismatch, match="episode sets do not match"):
        compare_paired_episodes(baseline, candidate)


def test_normal_pair_set_mismatch_raises_instead_of_comparing_intersection():
    normal_baseline = [_episode("normal-a", True), _episode("normal-b", False)]
    normal_candidate = [_episode("normal-a", True, service="candidate")]
    with pytest.raises(PairingSetMismatch, match="episode sets do not match"):
        compare_paired_episodes(normal_baseline, normal_candidate)


def test_invalid_episodes_are_excluded_from_paired_comparison():
    baseline = [
        _episode("case-1", False),
        _episode("case-2", True, valid=False),
    ]
    candidate = [
        _episode("case-1", True, service="candidate"),
        _episode("case-2", False, valid=False, service="candidate"),
    ]

    paired = compare_paired_episodes(baseline, candidate)

    assert paired.matched_pairs == 1
    assert paired.wins == 1
    assert paired.losses == 0


def test_normal_gate_rejects_paired_loss_even_when_mean_delta_is_zero():
    baseline = [_episode("normal-a", True), _episode("normal-b", False)]
    candidate = [
        _episode("normal-a", False, service="candidate"),
        _episode("normal-b", True, service="candidate"),
    ]
    normal_pairs = compare_paired_episodes(baseline, candidate)
    decision = ServiceGate(max_normal_paired_losses=0).evaluate(
        {"task_success": 0.5}, {"task_success": 0.5},
        {"task_success": 0.5}, {"task_success": 0.5},
        paired_normal=normal_pairs,
    )

    assert normal_pairs.delta_mean == 0.0
    assert normal_pairs.losses == 1
    assert decision.accepted is False
    assert decision.reason == "normal_user_paired_regression"


def test_candidate_tie_break_prefers_fewer_exact_replay_losses():
    safer = {
        "latest_paired": {"net_wins": 2},
        "exact_replay_paired": {"losses": 0},
        "robust_task_success": 0.6,
        "execution_score": 0.5,
    }
    riskier = {
        "latest_paired": {"net_wins": 2},
        "exact_replay_paired": {"losses": 1},
        "robust_task_success": 0.6,
        "execution_score": 0.5,
    }
    assert candidate_selection_key(safer, "p-z") < candidate_selection_key(riskier, "p-a")


def test_exact_replay_paired_loss_rejects_even_when_aggregate_is_unchanged():
    baseline = [_episode("exact-a", True), _episode("exact-b", False)]
    candidate = [
        _episode("exact-a", False, service="candidate"),
        _episode("exact-b", True, service="candidate"),
    ]
    exact_pairs = compare_paired_episodes(baseline, candidate)

    decision = ServiceGate().evaluate(
        {"task_success": 0.5, "exact_replay_task_success": 0.5},
        {"task_success": 0.5, "exact_replay_task_success": 0.5},
        paired_exact=exact_pairs,
    )

    assert exact_pairs.losses == 1
    assert decision.accepted is False
    assert decision.reason == "exact_replay_regression"


def test_budgeted_evaluator_preserves_repetition_and_pair_identity():
    class OneEpisodeEvaluator:
        def evaluate(self, customer_policy, service_policy, cases, split, generation, phase):
            case = cases[0]
            return [_episode(case.case_id, True, customer=customer_policy.policy_id,
                             service=service_policy.policy_id)]

    class Case:
        case_id = "same-case"

    outputs = BudgetedEpisodeEvaluator(OneEpisodeEvaluator(), repetitions=2).evaluate(
        CustomerPolicy(policy_id="customer-x"), ServicePolicy(policy_id="service-x"),
        [Case()], "validation", 0, "service_candidate_latest",
    )

    assert [item.metadata["repetition"] for item in outputs] == [0, 1]
    assert [item.metadata["pair_key"] for item in outputs] == [
        "same-case|customer=customer-x|rep=0",
        "same-case|customer=customer-x|rep=1",
    ]


def test_paired_gate_requires_minimum_wins_and_strict_net_win():
    baseline = [_episode("a", False), _episode("b", False), _episode("c", True)]
    candidate = [
        _episode("a", True, service="candidate"),
        _episode("b", True, service="candidate"),
        _episode("c", False, service="candidate"),
    ]
    paired = compare_paired_episodes(baseline, candidate)
    assert paired.wins == 2 and paired.losses == 1
    assert paired.net_wins == 1
    assert ServiceGate().evaluate(
        {"task_success": 1 / 3}, {"task_success": 2 / 3}, paired_latest=paired
    ).accepted
