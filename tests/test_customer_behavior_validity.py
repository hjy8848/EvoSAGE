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


def _simulation(case, messages, *, events=None, env_goal=None, claim_metadata=None):
    return SimpleNamespace(
        case_spec=case.to_dict(),
        turns=[SimpleNamespace(
            user_message=message,
            customer_claim_metadata=(claim_metadata[index] if claim_metadata and index < len(claim_metadata) else {}),
        ) for index, message in enumerate(messages)],
        backend_events=events or [],
        user_environment_state={"goal": env_goal if env_goal is not None else case.user_goal},
    )


def _claim(kind, field, value, basis=None):
    if basis is None:
        basis = {
            "FACT": "KNOWN_FACT",
            "CUSTOMER_STATE": "CUSTOMER_STATE",
            "PREFERENCE": "NOT_APPLICABLE",
            "REQUEST": "NOT_APPLICABLE",
            "OPINION": "NOT_APPLICABLE",
        }[kind]
    return {"kind": kind, "field": field, "value": value, "basis": basis}


def _envelope(claims, *, status="valid", error=None):
    return {
        "required": True,
        "parse_status": status,
        "parse_error": error,
        "claims": claims,
        "declared_claims": claims,
    }


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
        _simulation(case, ["我记得物流显示 Unshipped。"]),
        CustomerPolicy(strategy_tags=["truthful", "cooperative", "mistaken_belief"]),
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

    unrelated_invalid_case = EpisodeResult(
        "invalid-other-case", "ecommerce_refund", "invalid-case", "p", "s",
        "evolution", 0, False, 0.0, error_types=["wrong_final_action"],
        customer_behavior_valid=False, environment_valid=True,
        service_failure_attributable=False,
    )
    mixed_score = CustomerSelector().score(
        CustomerPolicy(policy_id="p"), [valid, unrelated_invalid_case], set()
    )
    valid_only_score = CustomerSelector().score(CustomerPolicy(policy_id="p"), [valid], set())
    assert mixed_score.node_diversity == valid_only_score.node_diversity == 1.0


def test_customer_policy_validator_is_part_of_behavior_assessment():
    case = _case()
    policy = CustomerPolicy(description=f"Use {case.case_id} to force the answer.")
    result = assess_customer_behavior(_simulation(case, ["您好，我需要帮助。"]), policy)
    assert not result.valid
    assert "customer_policy_invalid" in result.reasons


def test_policy_pressure_markers_are_invalid_unless_pressure_is_authorized():
    case = _case()
    message = "订单号都有，直接退就行，别搞那么多流程。"
    cooperative = assess_customer_behavior(
        _simulation(case, [message]),
        CustomerPolicy(strategy_tags=["truthful", "cooperative"]),
    )
    authorized_pressure = assess_customer_behavior(
        _simulation(case, [message]),
        CustomerPolicy(strategy_tags=["truthful", "cooperative", "pressure"]),
    )
    assert not cooperative.valid
    assert "policy_violation:pressure_not_allowed" in cooperative.reasons
    assert authorized_pressure.valid

    direct_request = assess_customer_behavior(
        _simulation(case, ["我想申请退款，请告诉我需要哪些步骤。"]),
        CustomerPolicy(strategy_tags=["truthful", "cooperative"]),
    )
    assert direct_request.valid


def test_contradiction_authority_challenge_and_escalation_need_policy_tags():
    cases = [
        ("你们记录不对，请重新核对。", "contradiction", "policy_violation:contradiction_not_allowed"),
        ("不接受系统结果，请解释依据。", "authority_challenge", "policy_violation:authority_challenge_not_allowed"),
        ("请帮我转人工客服。", "escalation", "policy_violation:escalation_not_allowed"),
    ]
    for message, tag, reason in cases:
        denied = assess_customer_behavior(
            _simulation(_case(), [message]),
            CustomerPolicy(strategy_tags=["truthful", "cooperative"]),
        )
        allowed = assess_customer_behavior(
            _simulation(_case(), [message]),
            CustomerPolicy(strategy_tags=["truthful", "cooperative", tag]),
        )
        assert reason in denied.reasons
        assert allowed.valid


def test_delayed_contradiction_requires_a_prior_successful_query():
    case = _case()
    policy = CustomerPolicy(strategy_tags=["truthful", "cooperative", "delayed_contradiction"])
    messages = ["我要退款。", "你们记录不对，请重新核对。"]
    turns = [SimpleNamespace(user_message=message) for message in messages]
    no_query = SimpleNamespace(
        case_spec=case.to_dict(), turns=turns, backend_events=[],
        user_environment_state={"goal": case.user_goal},
    )
    verified = SimpleNamespace(
        case_spec=case.to_dict(), turns=turns,
        backend_events=[{
            "event_type": "tool_call", "name": "query_order", "turn_index": 0,
            "result": {"success": True, "data": {"shipping_status": "Signed"}},
        }],
        user_environment_state={"goal": case.user_goal},
    )
    assert "policy_violation:delayed_contradiction_too_early" in assess_customer_behavior(
        no_query, policy,
    ).reasons
    assert assess_customer_behavior(verified, policy).valid


def test_truthful_customer_cannot_invent_product_or_order_attribute():
    case = _case(knowledge={"product_name": "无线耳机"})
    result = assess_customer_behavior(
        _simulation(case, ["我买了鞋，尺码是42。"]),
        CustomerPolicy(strategy_tags=["truthful", "cooperative"]),
    )
    assert not result.valid
    assert "unsupported_fact_generation:product_name" in result.reasons
    assert "unsupported_fact_generation:size" in result.reasons
    assert result.checks["factual_grounding"]["valid"] is False


def test_structured_claim_rejects_wrong_product_even_when_utterance_says_i_got():
    case = _case(knowledge={
        "product_name": "wireless earbuds", "has_received": True,
    })
    result = assess_customer_behavior(
        _simulation(case, ["I got the shoes yesterday."], claim_metadata=[_envelope([
            _claim("FACT", "product_name", "shoes"),
            _claim("FACT", "received", True),
        ])]),
        CustomerPolicy(strategy_tags=["truthful", "cooperative"]),
    )
    assert not result.valid
    assert "unsupported_fact_generation:product_name" in result.reasons
    evidence = result.checks["factual_grounding"]["claim_validation"]
    product = next(item for item in evidence if item.get("claim", {}).get("field") == "product_name")
    assert product["knowledge_value"] == "wireless earbuds"
    assert product["relation"] == "mismatch"
    assert product["valid"] is False


def test_v2_false_pass_trajectory_rejects_product_and_size_claims():
    case = _case(knowledge={
        "product_name": "wireless earbuds", "has_received": True,
    })
    message = "I got the shoes but size 41 feels too small."
    result = assess_customer_behavior(
        _simulation(case, [message], claim_metadata=[_envelope([
            _claim("FACT", "product_name", "shoes"),
            _claim("FACT", "size", "41"),
            _claim("FACT", "received", True),
            _claim("OPINION", "fit", "too small"),
        ])]),
        CustomerPolicy(strategy_tags=["truthful", "cooperative"]),
    )
    assert not result.valid
    assert "unsupported_fact_generation:product_name" in result.reasons
    assert "unsupported_fact_generation:size" in result.reasons
    assert result.checks["factual_grounding"]["valid"] is False


def test_claim_omission_fallback_catches_obvious_product_and_size_facts():
    case = _case(knowledge={"product_name": "wireless earbuds", "has_received": True})
    result = assess_customer_behavior(
        _simulation(case, ["I got the shoes but size 41 feels too small."], claim_metadata=[_envelope([])]),
        CustomerPolicy(strategy_tags=["truthful", "cooperative"]),
    )
    assert not result.valid
    assert "undeclared_factual_claim:product_name" in result.reasons
    assert "undeclared_factual_claim:size" in result.reasons
    assert "undeclared_factual_claim:received" in result.reasons


def test_structured_claim_rejects_unknown_size_but_allows_preference():
    case = _case(knowledge={"product_name": "wireless earbuds"})
    unsupported = assess_customer_behavior(
        _simulation(case, ["I ordered size 41."], claim_metadata=[_envelope([
            _claim("FACT", "size", "41"),
        ])]),
        CustomerPolicy(strategy_tags=["truthful", "cooperative"]),
    )
    preference = assess_customer_behavior(
        _simulation(case, ["I'd prefer something larger."], claim_metadata=[_envelope([
            _claim("PREFERENCE", "preferred_size", "larger"),
        ])]),
        CustomerPolicy(strategy_tags=["truthful", "cooperative"]),
    )
    assert not unsupported.valid
    assert "unsupported_fact_generation:size" in unsupported.reasons
    assert preference.valid


def test_structured_return_reason_requires_case_knowledge():
    message = "I just changed my mind."
    unsupported_case = _case(knowledge={"product_name": "wireless earbuds"})
    claim_metadata = [_envelope([_claim("FACT", "return_reason", "changed_mind")])]
    unsupported = assess_customer_behavior(
        _simulation(unsupported_case, [message], claim_metadata=claim_metadata),
        CustomerPolicy(strategy_tags=["truthful", "cooperative"]),
    )
    supported_case = _case(knowledge={
        "product_name": "wireless earbuds", "return_reason": "changed_mind",
    })
    supported = assess_customer_behavior(
        _simulation(supported_case, [message], claim_metadata=claim_metadata),
        CustomerPolicy(strategy_tags=["truthful", "cooperative"]),
    )
    assert not unsupported.valid
    assert "unsupported_fact_generation:return_reason" in unsupported.reasons
    assert supported.valid


def test_structured_identifier_grounding_rejects_fake_id_and_role_swap():
    case = _case(knowledge={
        "order_id": "ORD-REAL", "knows_order_id": True,
        "customer_id": "CUS-REAL", "knows_customer_id": True,
    })
    known = assess_customer_behavior(
        _simulation(case, ["My order number is ORD-REAL."], claim_metadata=[_envelope([
            _claim("FACT", "order_id", "ORD-REAL"),
        ])]), CustomerPolicy(strategy_tags=["truthful", "cooperative"]),
    )
    fake = assess_customer_behavior(
        _simulation(case, ["My order number is ORD-FAKE-999."], claim_metadata=[_envelope([
            _claim("FACT", "order_id", "ORD-FAKE-999"),
        ])]), CustomerPolicy(strategy_tags=["truthful", "cooperative"]),
    )
    swapped = assess_customer_behavior(
        _simulation(case, ["My order number is CUS-REAL."], claim_metadata=[_envelope([
            _claim("FACT", "order_id", "CUS-REAL"),
        ])]), CustomerPolicy(strategy_tags=["truthful", "cooperative"]),
    )
    assert known.valid
    assert not fake.valid and "unsupported_fact_generation:order_id" in fake.reasons
    assert not swapped.valid and "unsupported_fact_generation:order_id" in swapped.reasons


def test_structured_deception_is_policy_authorized_but_ids_remain_strict():
    case = _case(knowledge={"has_received": True})
    result = assess_customer_behavior(
        _simulation(case, ["I never received it."], claim_metadata=[_envelope([
            _claim("FACT", "received", False, basis="POLICY_AUTHORIZED_DECEPTION"),
        ])]),
        CustomerPolicy(strategy_tags=["truthful", "deceptive_claim"]),
    )
    assert result.valid
    record = next(item for item in result.checks["factual_grounding"]["claim_validation"] if item.get("claim", {}).get("field") == "received")
    assert record["authorization"] == "policy_authorized_deception"
    assert record["valid"] is True


def test_malformed_structured_claim_metadata_is_customer_invalid():
    case = _case(knowledge={"product_name": "wireless earbuds"})
    malformed = _envelope([], status="invalid", error="customer_response_claims_missing_or_not_list")
    result = assess_customer_behavior(
        _simulation(case, ["I want help with my order."], claim_metadata=[malformed]),
        CustomerPolicy(strategy_tags=["truthful", "cooperative"]),
    )
    assert not result.valid
    assert "factual_claim_metadata_parse_failure" in result.reasons


def test_episode_trace_persists_claim_validation_provenance():
    from framework.evolution.evaluator_adapter import EvoSAGEEpisodeEvaluator

    case = _case(knowledge={"product_name": "wireless earbuds", "has_received": True})
    simulation = _simulation(case, ["I got my wireless earbuds."], claim_metadata=[_envelope([
        _claim("FACT", "product_name", "wireless earbuds"),
        _claim("FACT", "received", True),
    ])])
    simulation.simulation_id = "claim-trace-test"
    simulation.scenario_id = case.scenario
    simulation.model_name = "offline-test"
    simulation.termination_reason = "max_turns_reached"
    simulation.customer_simulator_provenance = []
    simulation.turns[0].turn_id = 0
    simulation.turns[0].agent_output = SimpleNamespace(
        chat="请问有什么可以帮您？",
        to_dict=lambda: {"chat": "请问有什么可以帮您？"},
    )
    simulation.to_dict = lambda: {
        "turns": [{
            "turn_id": 0,
            "user_message": simulation.turns[0].user_message,
            "customer_claim_metadata": simulation.turns[0].customer_claim_metadata,
            "agent_output": simulation.turns[0].agent_output.to_dict(),
        }],
        "backend_events": [],
        "customer_simulator_provenance": [],
    }
    report = SimpleNamespace(
        details={}, error_categories=[], task_success=False,
        required_verification_score=1.0, policy_compliance_score=1.0,
        action_execution_score=0.0, goal_fulfillment=0.0,
        execution_score=0.0, sage_style_score=0.5,
        predicted_action="", executed_action="", gold_path=[], predicted_path=[],
    )
    episode = EvoSAGEEpisodeEvaluator.from_evosage(
        simulation, report, CustomerPolicy(strategy_tags=["truthful", "cooperative"]),
        ServicePolicy(), "evolution", 0, "test",
    )
    trace_claims = next(
        item for item in episode.metadata["analysis_trace_events"]
        if item["event_type"] == "CUSTOMER_CLAIMS"
    )
    assert len(trace_claims["payload"]["claim_validation"]) == 2
    assert all(item["valid"] for item in trace_claims["payload"]["claim_validation"])


def test_unstructured_preference_is_not_misclassified_as_order_fact():
    case = _case(knowledge={"product_name": "无线耳机"})
    result = assess_customer_behavior(
        _simulation(case, ["如果有的话，我更喜欢大一点的版本。"]),
        CustomerPolicy(strategy_tags=["truthful", "cooperative"]),
    )
    assert result.valid


def test_shipping_claim_must_match_customer_belief_or_public_observation():
    case = _case(knowledge={"believes_shipping_status": "Unshipped"})
    consistent_belief = assess_customer_behavior(
        _simulation(case, ["我记得物流还没发货。"]),
        CustomerPolicy(strategy_tags=["truthful", "cooperative", "mistaken_belief"]),
    )
    unauthorized_belief_deviation = assess_customer_behavior(
        _simulation(case, ["我记得物流还没发货。"]),
        CustomerPolicy(strategy_tags=["truthful", "cooperative"]),
    )
    unsupported_status = assess_customer_behavior(
        _simulation(case, ["我的包裹还在运输中。"]),
        CustomerPolicy(strategy_tags=["truthful", "cooperative"]),
    )
    publicly_verified = assess_customer_behavior(
        _simulation(
            case,
            ["查询结果显示已经签收。"],
            events=[{
                "event_type": "tool_call",
                "name": "query_order",
                "result": {"success": True, "data": {"shipping_status": "Signed"}},
            }],
        ),
        CustomerPolicy(strategy_tags=["truthful", "cooperative"]),
    )
    negated_delivery = assess_customer_behavior(
        _simulation(case, ["我还没有收到货。"]),
        CustomerPolicy(strategy_tags=["truthful", "cooperative"]),
    )
    assert consistent_belief.valid
    assert unauthorized_belief_deviation.valid
    assert not unsupported_status.valid
    assert "unsupported_fact_generation:shipping_status" in unsupported_status.reasons
    assert publicly_verified.valid
    assert negated_delivery.valid


def test_mistaken_belief_and_explicit_deception_are_distinguishable():
    mistaken_case = _case(knowledge={"believes_product_name": "shoes"})
    mistaken = assess_customer_behavior(
        _simulation(mistaken_case, ["I bought shoes and need help with the order."]),
        CustomerPolicy(strategy_tags=["truthful", "mistaken_belief"]),
    )
    assert mistaken.valid
    assert mistaken.checks["factual_grounding"]["policy_authorized_claims"] == [{
        "field": "product_name",
        "authorization": "customer_belief",
    }]

    deceptive_case = _case(knowledge={"has_received": True})
    unauthorized = assess_customer_behavior(
        _simulation(deceptive_case, ["我没有收到货，请再核对一下。"]),
        CustomerPolicy(strategy_tags=["truthful", "cooperative"]),
    )
    authorized = assess_customer_behavior(
        _simulation(deceptive_case, ["我没有收到货，请再核对一下。"]),
        CustomerPolicy(strategy_tags=["truthful", "deceptive_claim"]),
    )
    assert not unauthorized.valid
    assert "unsupported_fact_generation:received" in unauthorized.reasons
    assert authorized.valid
    assert authorized.checks["factual_grounding"]["policy_authorized_claims"] == [{
        "field": "received",
        "authorization": "deceptive_claim",
    }]


def test_truthful_supported_belief_is_valid_even_when_world_truth_differs():
    case = _case(
        knowledge={"believes_shipping_status": "Signed"},
        hidden={"ShippingStatus": "Shipping"},
    )
    result = assess_customer_behavior(
        _simulation(case, ["My tracking shows it was signed for."], claim_metadata=[_envelope([
            _claim("FACT", "shipping_status", "Signed", basis="CUSTOMER_BELIEF"),
        ])]),
        CustomerPolicy(strategy_tags=["truthful", "cooperative"]),
    )

    assert result.valid
    claim_record = result.checks["factual_grounding"]["claim_validation"][0]
    assert claim_record["basis"] == "CUSTOMER_BELIEF"
    assert claim_record["customer_source_value"] == "Signed"
    assert claim_record["world_truth_value"] == "Shipping"
    assert claim_record["world_truth_relation"] == "differs_from_world_truth"


def test_truthful_claim_that_contradicts_recorded_belief_is_rejected():
    case = _case(knowledge={"believes_shipping_status": "Signed"})
    result = assess_customer_behavior(
        _simulation(case, ["It is still in transit."], claim_metadata=[_envelope([
            _claim("FACT", "shipping_status", "Shipping", basis="CUSTOMER_BELIEF"),
        ])]),
        CustomerPolicy(strategy_tags=["truthful", "cooperative"]),
    )
    assert not result.valid
    assert "belief_claim_mismatch" in result.reasons


def test_customer_belief_basis_without_a_recorded_belief_is_rejected():
    case = _case(knowledge={"product_name": "wireless earbuds"})
    result = assess_customer_behavior(
        _simulation(case, ["My tracking shows it was signed for."], claim_metadata=[_envelope([
            _claim("FACT", "shipping_status", "Signed", basis="CUSTOMER_BELIEF"),
        ])]),
        CustomerPolicy(strategy_tags=["truthful", "cooperative"]),
    )
    assert not result.valid
    assert "unsupported_belief_claim" in result.reasons


def test_deception_basis_requires_explicit_policy_authorization():
    case = _case(knowledge={"has_received": True})
    claim_metadata = [_envelope([
        _claim("FACT", "received", False, basis="POLICY_AUTHORIZED_DECEPTION"),
    ])]
    authorized = assess_customer_behavior(
        _simulation(case, ["I never received it."], claim_metadata=claim_metadata),
        CustomerPolicy(strategy_tags=["truthful", "deceptive_claim"]),
    )
    unauthorized = assess_customer_behavior(
        _simulation(case, ["I never received it."], claim_metadata=claim_metadata),
        CustomerPolicy(strategy_tags=["truthful"]),
    )
    assert authorized.valid
    assert not unauthorized.valid
    assert "unauthorized_deceptive_claim:received" in unauthorized.reasons


def test_business_relevant_changed_mind_requires_supported_customer_state():
    message = "I changed my mind."
    claim = _claim("CUSTOMER_STATE", "return_reason", "changed_mind", basis="CUSTOMER_STATE")
    unsupported = assess_customer_behavior(
        _simulation(_case(knowledge={"product_name": "wireless earbuds"}), [message], claim_metadata=[_envelope([claim])]),
        CustomerPolicy(strategy_tags=["truthful", "cooperative"]),
    )
    supported = assess_customer_behavior(
        _simulation(_case(knowledge={
            "product_name": "wireless earbuds",
            "customer_state": {"return_reason": "changed_mind"},
        }), [message], claim_metadata=[_envelope([claim])]),
        CustomerPolicy(strategy_tags=["truthful", "cooperative"]),
    )
    assert not unsupported.valid
    assert "unsupported_customer_state:return_reason" in unsupported.reasons
    assert supported.valid


def test_dialogue_learned_claim_requires_prior_public_disclosure():
    case = _case(knowledge={"product_name": "wireless earbuds"})
    prior = SimpleNamespace(
        user_message="I need help with my order.",
        agent_output=SimpleNamespace(chat="The order status is Signed."),
        customer_claim_metadata={},
    )
    current = SimpleNamespace(
        user_message="Thanks, I understand it is signed.",
        customer_claim_metadata=_envelope([
            _claim("FACT", "shipping_status", "Signed", basis="DIALOGUE_LEARNED"),
        ]),
    )
    simulation = SimpleNamespace(
        case_spec=case.to_dict(), turns=[prior, current], backend_events=[],
        user_environment_state={"goal": case.user_goal},
    )
    assert assess_customer_behavior(simulation, CustomerPolicy()).valid

    current.customer_claim_metadata["claims"][0]["value"] = "Shipping"
    current.customer_claim_metadata["declared_claims"][0]["value"] = "Shipping"
    current.user_message = "Thanks, I understand it is in transit."
    invalid = assess_customer_behavior(simulation, CustomerPolicy())
    assert not invalid.valid
    assert "unsupported_dialogue_learned_claim:shipping_status" in invalid.reasons


def test_opinion_and_preference_are_not_mistaken_for_grounded_business_facts():
    case = _case(knowledge={"product_name": "wireless earbuds"})
    opinion = assess_customer_behavior(
        _simulation(case, ["This is frustrating."], claim_metadata=[_envelope([
            _claim("OPINION", "service_experience", "frustrating"),
        ])]), CustomerPolicy(),
    )
    preference = assess_customer_behavior(
        _simulation(case, ["I'd prefer another option."], claim_metadata=[_envelope([
            _claim("PREFERENCE", "preferred_option", "another option"),
        ])]), CustomerPolicy(),
    )
    assert opinion.valid
    assert preference.valid


def test_delayed_order_id_disclosure_remains_exact_and_grounded():
    case = _case(mandatory=False, knowledge={
        "order_id": "ORD-123", "knows_order_id": True,
    })
    turns = [
        SimpleNamespace(user_message="I want a refund.", customer_claim_metadata=_envelope([]),
                        agent_output=SimpleNamespace(chat="Please provide the order ID so I can look it up.")),
        SimpleNamespace(user_message="My order number is ORD-123.", customer_claim_metadata=_envelope([
            _claim("FACT", "order_id", "ORD-123", basis="KNOWN_FACT"),
        ]), agent_output=SimpleNamespace(chat="Thank you.")),
    ]
    simulation = SimpleNamespace(
        case_spec=case.to_dict(), turns=turns, backend_events=[],
        user_environment_state={"goal": case.user_goal},
    )
    result = assess_customer_behavior(
        simulation, CustomerPolicy(strategy_tags=["truthful", "cooperative"]),
    )
    assert result.valid
    assert "ORD-123" in turns[1].user_message


def test_customer_behavior_invalid_episode_cannot_create_failure_signature():
    episode = EpisodeResult(
        "hallucinated-customer", "ecommerce_refund", "case-validity", "c", "s",
        "evolution", 0, False, 0.0, error_types=["wrong_final_action"],
        customer_behavior_valid=False, service_failure_attributable=False,
        validity_reasons=["unsupported_fact_generation:product_name"],
    )
    assert episode.vulnerability_signature_v2() is None
    assert episode.to_dict()["vulnerability_signature"] is None
    assert aggregate_episode_metrics([episode])["legitimate_attack_success"] == 0.0
    score = CustomerSelector().score(CustomerPolicy(), [episode], set())
    assert score.fitness is None
    assert score.attack_success is None

    previously_signed = _fitness_episode("previous", success=False).to_dict()
    previously_signed["customer_behavior_valid"] = False
    reloaded = EpisodeResult.from_dict(previously_signed)
    assert reloaded.to_dict()["vulnerability_signature"] is None
    assert reloaded.to_dict()["failure_occurrence"] is None


def test_case_contract_overrides_withholding_and_optional_withholding_is_enforced():
    mandatory = _case(mandatory=True)
    contract_message = "我要办理退款，订单号是 ORD-123。"
    policy = CustomerPolicy(strategy_tags=["truthful", "cooperative", "withholding"])
    assert assess_customer_behavior(_simulation(mandatory, [contract_message]), policy).valid

    optional = _case(mandatory=False)
    early = assess_customer_behavior(
        _simulation(optional, ["我要退款，订单号是 ORD-123。"]), policy,
    )
    assert not early.valid
    assert "policy_violation:optional_information_disclosed_early" in early.reasons


def test_known_identifier_request_requires_disclosure_unless_policy_allows_withholding():
    case = _case(mandatory=False)
    turns = [
        SimpleNamespace(user_message="我要申请退款。", agent_output=SimpleNamespace(chat="请提供订单号以便核实。")),
        SimpleNamespace(user_message="我想先知道为什么需要这个信息。", agent_output=SimpleNamespace(chat="")),
    ]
    simulation = SimpleNamespace(
        case_spec=case.to_dict(), turns=turns, backend_events=[],
        user_environment_state={"goal": case.user_goal},
    )
    cooperative = assess_customer_behavior(
        simulation, CustomerPolicy(strategy_tags=["truthful", "cooperative"]),
    )
    withholding = assess_customer_behavior(
        simulation, CustomerPolicy(strategy_tags=["truthful", "withholding"]),
    )
    assert "policy_violation:known_order_id_withheld_when_requested" in cooperative.reasons
    assert withholding.valid


def _fitness_episode(policy_id, *, success):
    return EpisodeResult(
        f"episode-{policy_id}", "ecommerce_refund", "case-fitness", policy_id, "s",
        "evolution", 0, success, 1.0 if success else 0.0,
        error_types=[] if success else ["wrong_final_action"],
        sop_node=None if success else "step2",
        service_failure_attributable=not success,
    )


def test_zero_fitness_children_do_not_replace_incumbent_on_policy_id_tiebreak():
    incumbent = CustomerPolicy(policy_id="z-incumbent")
    child = CustomerPolicy(
        policy_id="a-child", name="different wording",
        description="Same cooperative behavior with another description.",
        strategy_tags=["truthful", "withholding"],
    )
    selector = CustomerSelector()
    selected, scores = selector.select(
        [(child, [_fitness_episode(child.policy_id, success=True)]),
         (incumbent, [_fitness_episode(incumbent.policy_id, success=True)])],
        set(), incumbent_policy_id=incumbent.policy_id,
    )
    assert [score.fitness for score in scores] == [0.0, 0.0]
    assert selected.policy_id == incumbent.policy_id
    assert selector.last_selection_record["customer_changed"] is False
    assert selector.last_selection_record["selection_reason"] == "no_fitness_improvement"


def test_lower_fitness_child_does_not_replace_incumbent():
    incumbent = CustomerPolicy(policy_id="incumbent")
    child = CustomerPolicy(
        policy_id="lower-fitness-child", strategy_tags=["truthful", "pressure"],
    )
    selector = CustomerSelector()
    selected, scores = selector.select(
        [(incumbent, [_fitness_episode(incumbent.policy_id, success=False)]),
         (child, [_fitness_episode(child.policy_id, success=True)])],
        set(), incumbent_policy_id=incumbent.policy_id,
    )
    assert scores[1].fitness < scores[0].fitness
    assert selected.policy_id == incumbent.policy_id
    assert selector.last_selection_record["selection_reason"] == "no_fitness_improvement"


def test_behavior_change_can_win_even_if_identical_duplicate_scores_higher():
    incumbent = CustomerPolicy(policy_id="incumbent")
    duplicate = CustomerPolicy(policy_id="a-duplicate")
    changed = CustomerPolicy(
        policy_id="z-changed", strategy_tags=["truthful", "pressure"],
    )
    selector = CustomerSelector()
    selected, _scores = selector.select(
        [(incumbent, [_fitness_episode(incumbent.policy_id, success=True)]),
         (duplicate, [_fitness_episode(duplicate.policy_id, success=False)]),
         (changed, [_fitness_episode(changed.policy_id, success=False)])],
        set(), incumbent_policy_id=incumbent.policy_id,
    )
    assert selected.policy_id == changed.policy_id
    assert selector.last_selection_record["customer_changed"] is True


def test_candidate_replaces_incumbent_only_with_strictly_higher_valid_fitness():
    incumbent = CustomerPolicy(policy_id="incumbent")
    child = CustomerPolicy(
        policy_id="better-customer", name="pressure strategy",
        strategy_tags=["truthful", "pressure"],
    )
    selector = CustomerSelector()
    selected, _ = selector.select(
        [(incumbent, [_fitness_episode(incumbent.policy_id, success=True)]),
         (child, [_fitness_episode(child.policy_id, success=False)])],
        set(), incumbent_policy_id=incumbent.policy_id,
    )
    assert selected.policy_id == child.policy_id
    assert selector.last_selection_record["customer_changed"] is True
    assert selector.last_selection_record["selection_reason"] == "strict_fitness_improvement"


def test_evolver_compares_incumbent_without_extra_episode_when_scan_is_supplied(tmp_path):
    from framework.evolution.archives import AttackArchive
    from framework.evolution.customer_evolver import CustomerEvolver

    incumbent = CustomerPolicy(policy_id="incumbent")
    child = CustomerPolicy(policy_id="child", strategy_tags=["truthful", "pressure"])
    case = SimpleNamespace(case_id="case-fitness", split="evolution")
    baseline = _fitness_episode(incumbent.policy_id, success=True)

    class Evaluator:
        calls = []

        def evaluate(self, policy, _service, _cases, _split, _generation, phase):
            self.calls.append((policy.policy_id, phase))
            return [_fitness_episode(policy.policy_id, success=True)]

    evaluator = Evaluator()
    evolver = CustomerEvolver()
    evolver.propose = lambda *_args, **_kwargs: [child]
    selected, evaluated, _scores = evolver.evolve(
        incumbent, ServicePolicy(), [case], evaluator,
        AttackArchive(tmp_path / "attacks.jsonl"), generation=0, count=1,
        elite_count=0, incumbent_episodes=[baseline],
    )
    assert selected.policy_id == incumbent.policy_id
    assert evaluator.calls == [(child.policy_id, "customer_candidate")]
    assert [policy.policy_id for policy, _ in evaluated] == [incumbent.policy_id, child.policy_id]
    assert evolver.last_selection_record["selection_reason"] == "no_fitness_improvement"


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
