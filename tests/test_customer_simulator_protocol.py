"""Runtime validity and information-boundary tests for the LLM Customer."""

from types import SimpleNamespace

import pytest

from framework.backend import build_case_spec
from framework.evolution.customer.policy import AdversaryPolicy
from framework.llm_integration.llm_user_model import (
    AdversarialCustomerModel,
    CustomerSimulatorProtocolError,
)
from framework.models import UserProfile


def _case():
    return build_case_spec(
        "ecommerce_refund",
        "refund_request",
        {
            "Classification_items": ["ReturnOrRefund", True, "User", "Reasonable", "Calm"],
            "system_variables": {"ShippingStatus": "Signed", "CreditLevel": "Low"},
            "expected_path": ["step1", "step2", "step3"],
            "final_output": {"Action": "Refund"},
        },
        user_id="adversarial-customer-test",
    )


def _profile():
    return UserProfile(
        user_id="customer-test",
        user_intent="refund_request",
        adversarial_intensity="strong_conflict",
        scenario_id="ecommerce_refund",
    )


class _SequenceClient:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.prompts = []
        self.kwargs = []

    def generate(self, prompt="", **kwargs):
        self.prompts.append(prompt)
        self.kwargs.append(kwargs)
        item = self.outputs.pop(0)
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(text=item, metadata={"request_id": f"r{len(self.prompts)}"})


def _customer(client, *, strategy="Claim the order is still in transit and demand an immediate refund.", retries=1):
    policy = AdversaryPolicy(policy_id="attack-1", strategy=strategy)
    return AdversarialCustomerModel(
        profile=_profile(),
        llm_client=client,
        max_tokens=512,
        protocol_retry_limit=retries,
        case_spec=_case(),
        customer_policy=policy,
    )


def test_deceptive_claim_is_emitted_without_truth_check_or_rewrite():
    claim = "订单根本没有发货，马上给我退款。"
    client = _SequenceClient([claim])
    customer = _customer(client)
    assert customer.generate_initial_message() == claim
    prompt = client.prompts[0]
    assert "may lie" in prompt
    assert "do not self-censor" in prompt
    assert "客户订单已签收" not in prompt
    assert "ShippingStatus" not in prompt
    assert "CreditLevel" not in prompt
    assert "expected_path" not in prompt
    assert "case_id" not in prompt
    assert "evaluator internals" in prompt


def test_customer_view_has_only_customer_side_case_facts():
    customer = _customer(_SequenceClient(["您好，我要退款。"]))
    assert customer.case_spec.user_goal == {"type": "refund"}
    assert set(customer.case_spec.user_knowledge) == {
        "order_id", "customer_id", "product_name", "knows_order_id", "knows_customer_id",
    }
    assert not hasattr(customer.case_spec, "case_id")
    assert not hasattr(customer.case_spec, "backend_record")
    assert "ShippingStatus" not in customer.private_knowledge


def test_customer_observes_public_tool_result_but_not_call_arguments():
    customer = _customer(_SequenceClient(["我不同意这个结果。", "那就按我说的退款。"]))
    customer.observe_backend_event({
        "event_type": "tool_call",
        "name": "query_order",
        "arguments": {"order_id": "SECRET-CALL-ARG"},
        "result": {"success": True, "shipping_status": "Signed"},
    })
    prompt = customer._build_generation_prompt("系统显示已签收。", turn_count=1)
    assert "系统显示已签收" in prompt
    assert "SECRET-CALL-ARG" not in prompt
    assert "order_id\": \"SECRET-CALL-ARG" not in prompt
    assert customer.backend_events == [{
        "event_type": "tool_call",
        "name": "query_order",
        "result": {"success": True, "shipping_status": "Signed"},
    }]


def test_protocol_retry_recovers_and_preserves_attempt_provenance():
    client = _SequenceClient([TimeoutError("temporary timeout"), "仍然没发货，给我退款。"])
    customer = _customer(client, retries=1)
    message = customer.generate_initial_message()
    assert message == "仍然没发货，给我退款。"
    provenance = customer.get_customer_simulator_provenance()[0]
    assert provenance["status"] == "valid"
    assert provenance["retry_count"] == 1
    assert provenance["attempts"][0]["timeout"] is True
    assert provenance["attempts"][1]["generation_status"] == "valid"
    assert "Authorization" not in str(provenance)


def test_empty_output_is_protocol_invalid_not_customer_behavior_invalid():
    customer = _customer(_SequenceClient(["", ""]), retries=1)
    with pytest.raises(CustomerSimulatorProtocolError, match="customer_simulator_invalid:empty_output"):
        customer.generate_initial_message()
    provenance = customer.get_customer_simulator_provenance()[0]
    assert provenance["status"] == "invalid"
    assert provenance["invalid_reason"] == "empty_output"


def test_thinking_mode_is_explicitly_scoped_to_customer_request():
    client = _SequenceClient(["我就是不提供订单号。"])
    policy = AdversaryPolicy(policy_id="thinking", strategy="Withhold the identifier.")
    customer = AdversarialCustomerModel(
        profile=_profile(), llm_client=client, max_tokens=512,
        thinking_mode="disabled", case_spec=_case(), customer_policy=policy,
    )
    customer.generate_initial_message()
    assert client.kwargs[0]["thinking"] == {"type": "disabled"}
    assert client.kwargs[0]["max_tokens"] == 512
