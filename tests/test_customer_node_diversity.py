import pytest

from framework.evolution.config import CustomerEvolutionConfig
from framework.evolution.customer_selector import CustomerSelector
from framework.evolution.schemas import CustomerPolicy, EpisodeResult


def test_failure_node_diversity_is_normalized_by_candidate_cases_not_sop_graph_size():
    policy = CustomerPolicy(policy_id="node-diversity")
    episodes = [
        EpisodeResult(
            episode_id=f"episode-{index}", scenario="ecommerce_refund",
            case_id=f"case-{index}", customer_policy_id=policy.policy_id,
            service_policy_id="service", split="evolution", generation=0,
            task_success=False, execution_score=0.0,
            error_types=["wrong_tool_arguments"],
            sop_node=f"sop-node-{index % 3}",
            service_failure_attributable=True,
        )
        for index in range(6)
    ]
    # The old call-site pretended the case count was the number of SOP nodes.
    selector = CustomerSelector({
        "attack_success": 0.0, "novelty": 0.0, "node_diversity": 1.0,
    })
    with pytest.warns(DeprecationWarning, match="total_nodes.*ignored"):
        score = selector.score(policy, episodes, set(), total_nodes=20)
    assert score.node_diversity == 0.5  # 3 observed failure nodes / 6 evaluated cases
    assert "node_diversity" in score.to_dict()
    assert "coverage" not in score.to_dict()


def test_legacy_fitness_weights_are_retained_for_provenance_but_never_affect_selection():
    legacy = CustomerEvolutionConfig(fitness_weights={
        "attack_success": 0.0, "novelty": 100.0, "coverage": 100.0,
    })
    policy = CustomerPolicy(policy_id="official-fitness")
    episodes = [
        EpisodeResult(
            episode_id="success", scenario="ecommerce_refund", case_id="case-1",
            customer_policy_id=policy.policy_id, service_policy_id="service",
            split="evolution", generation=0, task_success=True,
            execution_score=1.0,
            service_failure_attributable=False,
        ),
        EpisodeResult(
            episode_id="failure", scenario="ecommerce_refund", case_id="case-2",
            customer_policy_id=policy.policy_id, service_policy_id="service",
            split="evolution", generation=0, task_success=False,
            execution_score=0.0,
            service_failure_attributable=True,
        ),
    ]
    score = CustomerSelector(legacy.fitness_weights).score(policy, episodes, set())
    assert legacy.fitness_weights["coverage"] == 100.0
    assert score.fitness == 0.5
