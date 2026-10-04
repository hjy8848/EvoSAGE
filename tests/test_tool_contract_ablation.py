import json
from types import SimpleNamespace

from framework import get_sop_graph
from framework.backend import ToolCall, ToolContractConfig, build_case_spec, create_backend
from framework.evolution.config import EvaluationConfig
from framework.models.agent_model import AgentModel
from framework.prompts.ecommerce_refund_prompts import AGENT_SYSTEM_PROMPT


def _case():
    return build_case_spec(
        "ecommerce_refund",
        "refund_request",
        {
            "Classification_items": ["ReturnOrRefund", True, "User", "Reasonable", "Calm"],
            "system_variables": {"ShippingStatus": "Signed", "CreditLevel": "Low"},
            "expected_path": ["step1", "step2", "step3"],
            "final_output": {"Action": "CollectionService"},
        },
        user_id="tool-contract-test",
    )


class _QueryThenFinishClient:
    def __init__(self):
        self.requests = []

    def generate(self, prompt="", **kwargs):
        self.requests.append(kwargs)
        if len(self.requests) == 1:
            return SimpleNamespace(
                text="",
                tool_calls=[ToolCall("query-1", "query_order", {"order_id": ""})],
                metadata={},
            )
        return SimpleNamespace(
            text=json.dumps({
                "classification_output": {
                    "CoreIntention": "ReturnOrRefund",
                    "ProvidedDocument": True,
                    "Responsibility": "User",
                    "RefundReasonable": "Reasonable",
                    "EmotionStatus": "Calm",
                },
                "now_path": ["step1", "step2", "step3"],
                "finals": {"Action": "Supplementary"},
                "chat": "请提供订单号，我再为您核实。",
            }, ensure_ascii=False),
            tool_calls=[],
            metadata={},
        )


def _run(config):
    case = _case()
    backend = create_backend(case, config)
    client = _QueryThenFinishClient()
    agent = AgentModel(
        "ecommerce_refund",
        get_sop_graph("ecommerce_refund"),
        system_prompt=AGENT_SYSTEM_PROMPT,
        llm_client=client,
        use_llm_for_full_output=True,
        tool_contract_config=config,
    )
    output = agent.process_turn("我想退款。", backend_environment=backend)
    return backend, client, output


def _query_order_schema(client):
    tools = client.requests[0]["tools"]
    return next(item for item in tools if item["function"]["name"] == "query_order")


def test_provider_schema_ablation_exposes_min_length_without_runtime_blocking():
    backend, client, output = _run(ToolContractConfig(provider_schema_strict=True))

    order_schema = _query_order_schema(client)["function"]["parameters"]["properties"]["order_id"]
    assert order_schema["minLength"] == 1
    assert output.tool_results[0]["error_code"] == "order_not_found"
    assert backend.get_event_log()[0]["result"]["error_code"] == "order_not_found"
    assert backend.selected_records == {}


def test_runtime_contract_blocks_empty_order_id_before_domain_backend():
    config = ToolContractConfig(provider_schema_strict=True, runtime_schema_validation=True)
    backend, client, output = _run(config)

    assert _query_order_schema(client)["function"]["parameters"]["properties"]["order_id"]["minLength"] == 1
    assert output.tool_results[0]["error_code"] == "tool_argument_schema_invalid"
    assert "length must be at least 1" in output.tool_results[0]["error_message"]
    # The audit log records the rejected attempt, while domain lookup/state
    # selection never runs.
    assert len(backend.get_event_log()) == 1
    assert backend.get_event_log()[0]["event_type"] == "tool_call"
    assert backend.get_event_log()[0]["result"]["error_code"] == "tool_argument_schema_invalid"
    assert backend.selected_records == {}
    assert output.metadata["tool_contract_violations"][0]["name"] == "query_order"


def test_runtime_validation_only_enforces_constraints_present_in_schema():
    backend, _client, output = _run(ToolContractConfig(runtime_schema_validation=True))
    assert output.tool_results[0]["error_code"] == "order_not_found"
    assert backend.selected_records == {}


def test_default_tool_contract_preserves_existing_behavior_and_config_roundtrips():
    default = EvaluationConfig.from_dict({})
    assert default.tool_contract == ToolContractConfig()
    strict = EvaluationConfig.from_dict({
        "tool_contract": {
            "provider_schema_strict": True,
            "runtime_schema_validation": True,
        }
    })
    assert strict.tool_contract == ToolContractConfig(True, True)
    assert strict.__dict__["tool_contract"] == ToolContractConfig(True, True)
    assert {
        "provider_schema_strict": True,
        "runtime_schema_validation": True,
    } == {
        "provider_schema_strict": strict.tool_contract.provider_schema_strict,
        "runtime_schema_validation": strict.tool_contract.runtime_schema_validation,
    }
    try:
        ToolContractConfig.from_value({"runtime_schema_validaton": True})
    except ValueError as exc:
        assert "unknown tool contract options" in str(exc)
    else:
        raise AssertionError("typo in tool contract config must be rejected")
