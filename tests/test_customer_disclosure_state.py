from framework.backend.factory import build_case_spec
from framework.evolution.config import SplitConfig
from framework.evolution.customer_policy import PolicyCustomerModel
from framework.evolution.schemas import CustomerPolicy
from framework.evolution.split_manager import SplitManager
from framework.models import UserProfile


def _profile():
    return UserProfile(
        user_id="disclosure-test", user_intent="refund_before_shipping",
        adversarial_intensity="weak_conflict", scenario_id="ecommerce_refund",
    )


def _case(variant="on_request"):
    from framework.sop.ecommerce_refund_PathList import generate_path_list
    return build_case_spec(
        "ecommerce_refund", "refund_before_shipping", generate_path_list()[5],
        user_id=f"disclosure-{variant}", disclosure_variant=variant,
    )


def test_mandatory_opening_identifier_overrides_delayed_customer_policy():
    case = _case("opening")
    policy = CustomerPolicy(strategy_tags=["truthful", "withholding", "delayed_disclosure"])
    user = PolicyCustomerModel(_profile(), case_spec=case, policy=policy)
    opening = user.generate_initial_message()
    assert case.user_knowledge["order_id"] in opening


def test_delayed_disclosure_reveals_identifier_on_second_request():
    case = _case("on_request")
    user = PolicyCustomerModel(
        _profile(), case_spec=case,
        policy=CustomerPolicy(strategy_tags=["truthful", "delayed_disclosure"]),
    )
    opening = user.generate_initial_message()
    assert case.user_knowledge["order_id"] not in opening
    first = user.generate_next_message("请提供订单号，以便核验。", 1)
    second = user.generate_next_message("请再提供一下订单号。", 2)
    assert case.user_knowledge["order_id"] not in first
    assert case.user_knowledge["order_id"] in second
    assert user._identifier_request_count["order_id"] == 2
    assert "order_id" in user._identifier_disclosed


def test_withholding_requires_an_explanation_and_is_distinct_from_delay():
    case = _case("on_request")
    user = PolicyCustomerModel(
        _profile(), case_spec=case,
        policy=CustomerPolicy(strategy_tags=["truthful", "withholding"]),
    )
    first = user.generate_next_message("请提供订单号。", 1)
    second = user.generate_next_message("需要订单号用于核验订单状态。", 2)
    assert case.user_knowledge["order_id"] not in first
    assert "说明" in first
    assert case.user_knowledge["order_id"] in second


def test_split_manager_builds_legal_on_request_disclosure_family():
    splits = SplitManager(SplitConfig(
        customer_disclosure_variants=["opening", "on_request"],
    )).build()
    all_cases = splits.all_cases
    assert {case.case_spec["metadata"].get("customer_disclosure_variant", "opening") for case in all_cases} == {
        "opening", "on_request",
    }
    on_request = next(case for case in all_cases if case.case_spec["metadata"].get("customer_disclosure_variant") == "on_request")
    assert on_request.case_spec["user_knowledge"]["knows_order_id"] is True
    assert on_request.case_spec["user_policy"]["show_order_id_initially"] is False
    assert on_request.case_spec["user_policy"]["reveal_order_id_on_request"] is True
