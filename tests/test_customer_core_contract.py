"""Regression guard: diversity/signature diagnostics do not gate Customer fitness."""

from framework.evolution.config import CustomerEvolutionConfig, CustomerSearchConfig
from framework.evolution.customer.policy import AdversaryPolicy
from framework.evolution.customer_selector import CustomerSelector
from framework.evolution.schemas import EpisodeResult


def test_failure_node_and_signature_metadata_cannot_change_official_fitness():
    policy = AdversaryPolicy(policy_id="candidate", strategy="Free adversarial behavior")
    base = EpisodeResult(
        episode_id="episode",
        scenario="ecommerce_refund",
        case_id="case",
        customer_policy_id=policy.policy_id,
        service_policy_id="service_policy_s0",
        split="evolution",
        generation=0,
        task_success=False,
        execution_score=0.0,
    )
    attributed = EpisodeResult.from_dict(base.to_dict())
    attributed.sop_node = "some-node"
    attributed.service_failure_attributable = True
    attributed.customer_behavior_valid = True
    unattributed = EpisodeResult.from_dict(base.to_dict())
    unattributed.sop_node = None
    unattributed.service_failure_attributable = False
    unattributed.customer_behavior_valid = False

    selector = CustomerSelector()
    first = selector.score(policy, [attributed])
    second = selector.score(policy, [unattributed])

    assert first.fitness == second.fitness == 1.0
    assert first.official_task_success == second.official_task_success == 0.0


def test_customer_config_has_no_behavior_taxonomy_or_fitness_weights():
    fields = set(CustomerEvolutionConfig.__dataclass_fields__)
    assert fields == {"strategy_schema", "candidate_count"}
    customer_fields = set(CustomerSearchConfig.__dataclass_fields__)
    assert not {"service", "fresh_adversary"}.intersection(customer_fields)
