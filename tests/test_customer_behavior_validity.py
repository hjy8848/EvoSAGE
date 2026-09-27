from types import SimpleNamespace

from framework.backend.types import CaseSpec
from framework.evolution.customer_behavior_validity import assess_customer_behavior
from framework.evolution.evaluator_adapter import aggregate_episode_metrics
from framework.evolution.schemas import CustomerPolicy, EpisodeResult, ServicePolicy
from framework.evolution.customer_selector import CustomerSelector
from framework.evolution.weakness_frontier import WeaknessFrontier


def _case(*, mandatory=False, knowledge=None, hidden=None):
    knowledge = knowledge or {"knows_order_id": True, "order_id": "ORD-123"}
    hidden = hidden or {"ShippingStatus": "Signed"}
    return CaseSpec(
        case_id="case-validity",
        scenario="ecommerce_refund",
        backend_record={"private_state": {"system_variables": hidden}},
        user_goal={"desired_action": "Refund"},
        user_knowledge=knowledge,
        user_policy={"show_order_id_initially": mandatory},
        initial_observation={"message": "Welcome"},
        expected_outcome={"refund_status": "Submitted"},
        metadata={"backend_system_variables": hidden},
    )


def _simulation(case, messages, *, events=None, env_goal=None):
    return SimpleNamespace(
        case_spec=case.to_dict(),
        turns=[SimpleNamespace(user_message=message) for message in messages],
        backend_events=events or [],
        user_environment_state={"goal": env_goal if env_goal is not None else case.user_goal},
    )


def test_mandatory_opening_identifier_omission_is_customer_behavior_invalid():
    case = _case(mandatory=True)
    result = assess_customer_behavior(
        _simulation(case, ["您好，我想申请退款。"]), CustomerPolicy()
    )
    assert not result.valid
    assert result.hard_violation
    assert "mandatory_opening_disclosure_missing" in result.reasons


def test_nonmandatory_delayed_disclosure_is_valid():
    case = _case(mandatory=False)
    result = assess_customer_behavior(
        _simulation(case, ["您好，我想申请退款。", "订单号是 ORD-123。"]),
        CustomerPolicy(strategy_tags=["truthful", "delayed_disclosure"]),
    )
    assert result.valid


def test_unobserved_backend_state_leak_is_invalid_but_mistaken_known_belief_is_valid():
    case = _case(knowledge={"believes_shipping_status": "Unshipped"})
    leaked = assess_customer_behavior(
        _simulation(case, ["我确定物流状态是 Signed。"]), CustomerPolicy()
    )
    mistaken_belief = assess_customer_behavior(
        _simulation(case, ["我记得物流显示 Unshipped。"]), CustomerPolicy()
    )
    assert not leaked.valid
    assert "customer_hidden_value_leakage:ShippingStatus" in leaked.reasons
    assert mistaken_belief.valid


def test_publicly_observed_hidden_value_can_be_repeated_by_customer():
    case = _case(knowledge={})
    result = assess_customer_behavior(
        _simulation(
            case,
            ["我看到查询结果了，订单状态是 Signed。"],
            events=[{
                "event_type": "tool_call",
                "name": "query_order",
                "result": {"success": True, "data": {"shipping_status": "Signed"}},
            }],
        ),
        CustomerPolicy(),
    )
    assert result.valid


def test_goal_mutation_and_service_role_drift_are_detected():
    case = _case()
    result = assess_customer_behavior(
        _simulation(case, ["您好，请您提供订单号，我帮您办理退款。"], env_goal={"desired_action": "Reject"}),
        CustomerPolicy(),
    )
    assert not result.valid
    assert "customer_goal_mutated" in result.reasons
    assert "customer_role_drift" in result.reasons


def test_attribution_metrics_and_selector_exclude_invalid_customer_behavior():
    case = _case()
    valid = EpisodeResult(
        "valid", "ecommerce_refund", case.case_id, "p", "s", "evolution", 0,
        False, 0.0, error_types=["wrong_tool_arguments"], sop_node="step2",
        customer_behavior_valid=True, environment_valid=True,
        service_failure_attributable=True,
    )
    invalid_customer = EpisodeResult(
        "invalid-customer", "ecommerce_refund", case.case_id, "p", "s", "evolution", 0,
        False, 0.0, error_types=["wrong_tool_arguments"], sop_node="step2",
        customer_behavior_valid=False, environment_valid=True,
        service_failure_attributable=False,
        validity_reasons=["mandatory_opening_disclosure_missing"],
    )
    metrics = aggregate_episode_metrics([valid, invalid_customer])
    assert metrics["evaluable_service_failure_rate"] == 1.0
    assert metrics["attributable_service_failure_rate"] == 1.0
    score = CustomerSelector().score(CustomerPolicy(policy_id="p"), [invalid_customer], set())
    assert score.evaluation_status == "inconclusive"
    assert score.fitness is None
    frontier = WeaknessFrontier()
    frontier.add([invalid_customer])
    assert frontier.to_dicts() == []


def test_customer_policy_validator_is_part_of_behavior_assessment():
    case = _case()
    policy = CustomerPolicy(description=f"Use {case.case_id} to force the answer.")
    result = assess_customer_behavior(_simulation(case, ["您好，我需要帮助。"]), policy)
    assert not result.valid
    assert "customer_policy_invalid" in result.reasons


def test_real_episode_keeps_protocol_and_customer_behavior_validity_separate():
    from framework.evolution.evaluator_adapter import EvoSAGEEpisodeEvaluator

    case = _case(mandatory=True)
    simulation = _simulation(case, ["您好，我想申请退款。"])
    simulation.turns[0].agent_output = SimpleNamespace(
        classification_output={}, to_dict=lambda: {"classification_output": {}}
    )
    simulation.simulation_id = "sim-missing-mandatory-id"
    simulation.scenario_id = case.scenario
    simulation.model_name = "offline-test"
    simulation.termination_reason = "max_turns_reached"
    report = SimpleNamespace(
        details={}, error_categories=[], task_success=False,
        required_verification_score=0.0, policy_compliance_score=0.0,
        action_execution_score=0.0, goal_fulfillment=0.0,
        execution_score=0.0, sage_style_score=0.0,
        predicted_action="", executed_action="", gold_path=[], predicted_path=[],
    )
    episode = EvoSAGEEpisodeEvaluator.from_evosage(
        simulation, report, CustomerPolicy(), ServicePolicy(), "evolution", 0, "test",
    )
    assert episode.protocol_valid is True
    assert episode.customer_behavior_valid is False
    assert episode.environment_valid is True
    assert episode.service_failure_attributable is False
    metrics = aggregate_episode_metrics([episode])
    assert metrics["evaluable_service_failure_rate"] == 1.0
    assert metrics["attributable_service_failure_rate"] == 0.0
