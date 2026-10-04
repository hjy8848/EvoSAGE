import json
import sys
from types import SimpleNamespace

import pytest
import requests

from framework.evolution.config import CustomerSearchConfig, load_customer_search_config
from framework.evolution.evaluator_adapter import EvoSAGEEpisodeEvaluator
from framework.evolution.evaluator_adapter import _exception_invalid_reason
from framework.evolution.persistence import RunStore
from framework.evolution.request_budget import APIRequestBudget, APIRequestBudgetExceeded, request_context
from framework.evolution.customer.policy import AdversaryPolicy
from framework.evolution.schemas import ServicePolicy
from framework.backend.types import ToolCall, ToolResult
from framework.llm_integration.llm_client import LLMResponse, OpenAIAPIClient
from framework.backend import EcommerceBackend, build_case_spec
from framework import get_sop_graph
from framework.models.agent_model import AgentModel


class _FakeResponse:
    def __init__(self, *, status=200, body=None, text=""):
        self.status_code = status
        self._body = body or {
            "id": "req-test",
            "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 2},
        }
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}: {self.text}", response=self)

    def json(self):
        return self._body


def test_performance_config_roundtrip_and_safe_legacy_defaults(tmp_path):
    defaults = CustomerSearchConfig.from_dict({})
    assert defaults.evaluation.max_tool_steps == 8
    assert defaults.evaluation.invalid_evaluation_retries == 1
    assert defaults.evaluation.agent_max_retries == 1

    config = CustomerSearchConfig.from_dict({
        "evaluation": {
            "agent_thinking_mode": "disabled",
            "customer_evolver_thinking_mode": "enabled",
            "agent_max_retries": 1,
            "customer_transport_max_retries": 2,
            "customer_evolver_max_retries": 1,
            "judge_validation_retries": 0,
            "invalid_evaluation_retries": 0,
            "max_tool_steps": 3,
            "max_api_requests_per_generation": 30,
            "max_api_requests_per_run": 30,
            "rate_limit_backoff_seconds": 0,
        }
    })
    path = tmp_path / "roundtrip.json"
    path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    loaded = load_customer_search_config(path)
    assert loaded.evaluation == config.evaluation
    assert loaded.evaluation.agent_thinking_mode == "disabled"
    assert loaded.evaluation.customer_evolver_thinking_mode == "enabled"
    assert loaded.evaluation.max_tool_steps == 3
    assert loaded.evaluation.max_api_requests_per_run == 30


def test_transport_attempt_one_means_one_provider_request_and_no_429_sleep(monkeypatch):
    calls = []
    sleeps = []

    def post(*args, **kwargs):
        calls.append(kwargs)
        return _FakeResponse(status=429, text="rate limit")

    monkeypatch.setattr("framework.llm_integration.llm_client.requests.post", post)
    monkeypatch.setattr("framework.llm_integration.llm_client.time.sleep", sleeps.append)
    client = OpenAIAPIClient(
        api_key="never-record-this",
        base_url="https://provider.invalid/v1",
        model_name="test-model",
        max_retries=1,
        rate_limit_backoff_seconds=0,
    )
    with pytest.raises(requests.HTTPError):
        client.generate("hello", max_tokens=8)
    assert len(calls) == 1
    assert sleeps == []
    assert client.request_stats()["attempts"] == 1
    assert client.request_stats()["failures"] == 1


def test_configured_rate_limit_backoff_is_used_only_between_retries(monkeypatch):
    calls = []
    sleeps = []

    def post(*args, **kwargs):
        calls.append(kwargs)
        return _FakeResponse(status=429, text="rate limit")

    monkeypatch.setattr("framework.llm_integration.llm_client.requests.post", post)
    monkeypatch.setattr("framework.llm_integration.llm_client.time.sleep", sleeps.append)
    client = OpenAIAPIClient(
        api_key="never-record-this", base_url="https://provider.invalid/v1",
        model_name="test-model", max_retries=2, rate_limit_backoff_seconds=0.25,
    )
    with pytest.raises(requests.HTTPError):
        client.generate("hello", max_tokens=8)
    assert len(calls) == 2
    assert sleeps == [0.25]
    assert client.request_stats()["attempts"] == 2
    assert client.request_stats()["retries"] == 1


def test_litellm_429_uses_configured_backoff(monkeypatch):
    calls = []
    sleeps = []

    class RateLimitError(RuntimeError):
        status_code = 429

    def completion(**_kwargs):
        calls.append(1)
        raise RateLimitError("provider rate limited")

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(
        completion=completion, suppress_debug_info=False,
    ))
    monkeypatch.setattr("framework.llm_integration.llm_client.time.sleep", sleeps.append)
    from framework.llm_integration.llm_client import LiteLLMClient
    client = LiteLLMClient(
        api_key="never-record-this", base_url="https://provider.invalid/v1",
        model_name="test-model", max_retries=2, rate_limit_backoff_seconds=0.4,
    )
    with pytest.raises(RateLimitError):
        client.generate("hello", max_tokens=8)
    assert len(calls) == 2
    assert sleeps == [0.4]
    assert client.request_stats()["attempts"] == 2


def test_request_budget_blocks_before_http_and_accounts_by_phase(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "framework.llm_integration.llm_client.requests.post",
        lambda *args, **kwargs: calls.append(kwargs) or _FakeResponse(),
    )
    budget = APIRequestBudget(max_per_generation=1, max_per_run=1,
                              persist_path=tmp_path / "request_budget.json")
    client = OpenAIAPIClient(
        api_key="never-record-this", base_url="https://provider.invalid/v1",
        model_name="test-model", max_retries=1, request_budget=budget,
        request_role="agent",
    )
    with request_context(0, "agent_candidate"):
        client.generate("first", max_tokens=8)
        with pytest.raises(APIRequestBudgetExceeded, match="api_request_budget_exceeded"):
            client.generate("must not reach provider", max_tokens=8)
    assert len(calls) == 1
    snapshot = json.loads((tmp_path / "request_budget.json").read_text())
    phase = snapshot["generations"]["0"]["phases"]["agent_candidate"]
    assert phase["provider_attempts"] == 1
    assert phase["logical_requests"] == 1
    assert phase["roles"]["agent"]["input_tokens"] == 7
    assert snapshot["blocked_attempts"] == 1


def test_unsupported_thinking_falls_back_once_and_records_effective_status(monkeypatch):
    payloads = []

    def post(*args, **kwargs):
        # Capture a snapshot: the client removes the rejected optional field
        # from its reusable payload before the compatibility fallback request.
        payloads.append(json.loads(json.dumps(kwargs["json"])))
        if len(payloads) == 1:
            return _FakeResponse(status=400, text="unknown parameter thinking")
        return _FakeResponse()

    monkeypatch.setattr("framework.llm_integration.llm_client.requests.post", post)
    client = OpenAIAPIClient(
        api_key="never-record-this", base_url="https://provider.invalid/v1",
        model_name="test-model", max_retries=1,
    )
    response = client.generate("hello", max_tokens=8, thinking={"type": "disabled"})
    assert len(payloads) == 2
    assert payloads[0]["thinking"] == {"type": "disabled"}
    assert "thinking" not in payloads[1]
    assert response.metadata["thinking_requested"] == "disabled"
    assert response.metadata["thinking_effective"] == "unsupported/not_applied"
    assert client.request_stats()["attempts"] == 2


def test_configured_max_tool_steps_caps_executed_tool_rounds():
    case = build_case_spec(
        "ecommerce_refund",
        "refund_request",
        {
            "Classification_items": ["ReturnOrRefund", True, "User", "Reasonable", "Calm"],
            "system_variables": {"ShippingStatus": "Unshipped", "CreditLevel": "High"},
            "expected_path": ["step1", "step2"],
            "final_output": {"Action": "Refund", "PLAN": "none"},
        },
        user_id="tool-limit-test",
    )
    backend = EcommerceBackend(case)
    order_id = case.user_knowledge["order_id"]

    class AlwaysCallsTool:
        model_name = "fake"

        def __init__(self):
            self.calls = 0

        def generate(self, prompt, **kwargs):
            self.calls += 1
            return SimpleNamespace(
                text="",
                tool_calls=[ToolCall(f"call-{self.calls}", "query_order", {"order_id": order_id})],
                metadata={"assistant_message": {
                    "role": "assistant", "content": None,
                    "tool_calls": [{
                        "id": f"call-{self.calls}", "type": "function",
                        "function": {"name": "query_order", "arguments": json.dumps({"order_id": order_id})},
                    }],
                }},
            )

    client = AlwaysCallsTool()
    agent = AgentModel(
        "ecommerce_refund", get_sop_graph("ecommerce_refund"),
        system_prompt="Use tools", llm_client=client, use_llm_for_full_output=True,
    )
    output = agent.process_turn("我要退款", backend_environment=backend, max_tool_steps=3)

    executed_tool_events = [event for event in backend.get_event_log() if event["event_type"] == "tool_call"]
    assert len(executed_tool_events) == 3
    assert len(output.tool_calls) == 3
    assert output.metadata["tool_loop_limit"] is True
    assert output.metadata["termination_reason"] == "max_tool_steps"
    # A final provider response is needed to learn whether the Agent is done;
    # the fourth response is not allowed to execute its requested tool.
    assert client.calls == 4


def test_real_evaluator_cache_namespace_changes_with_runtime_semantics(tmp_path):
    from framework.evolution.real_factory import make_real_evaluator

    common = {
        "model": "test-model",
        "api_url": "https://provider.invalid/v1",
        "api_key": "never-record-this",
        "resume": True,
    }
    baseline = make_real_evaluator(output_dir=tmp_path / "base", max_tool_steps=3, **common)
    different_tool_cap = make_real_evaluator(
        output_dir=tmp_path / "tool-cap", max_tool_steps=2, **common
    )
    different_thinking = make_real_evaluator(
        output_dir=tmp_path / "thinking", max_tool_steps=3,
        agent_thinking_mode="disabled", **common
    )
    assert baseline.cache_namespace != different_tool_cap.cache_namespace
    assert baseline.cache_namespace != different_thinking.cache_namespace


@pytest.mark.parametrize("invalid_retries,expected_calls", [(0, 1), (1, 2)])
def test_invalid_episode_retry_count_is_configurable_and_not_business_failure(
    invalid_retries, expected_calls
):
    calls = []

    class Pipeline:
        def run_single_simulation(
            self, intent, user_id=None, path_config=None, phase="", case_spec_override=None
        ):
            calls.append(intent)
            raise RuntimeError("HTTP 503 provider unavailable")

    evaluator = EvoSAGEEpisodeEvaluator(
        lambda *_args, **_kwargs: Pipeline(),
        invalid_evaluation_retries=invalid_retries,
    )
    case = SimpleNamespace(
        case_id="case-1", scenario="ecommerce_refund", intent="refund_before_shipping",
        path_config={}, case_spec=None, split="evolution", path_id=1,
    )
    result = evaluator.evaluate(
        AdversaryPolicy(), ServicePolicy(), [case], "evolution", 0, "customer_candidate"
    )[0]
    assert len(calls) == expected_calls
    assert result.is_evaluation_invalid()
    assert not result.is_runtime_evaluable()
    assert result.invalid_reason == "provider_error"


def test_local_harness_type_error_is_not_misreported_as_provider_failure():
    assert _exception_invalid_reason(TypeError("unexpected local keyword")) == "environment_execution_error"
    assert _exception_invalid_reason(TimeoutError("request timed out")) == "timeout"
    assert _exception_invalid_reason(requests.Timeout("wrapped timeout")) == "timeout"
    assert _exception_invalid_reason(requests.HTTPError("HTTP 503")) == "provider_error"


def test_local_harness_error_is_invalid_but_not_a_provider_or_protocol_failure():
    class BrokenPipeline:
        def run_single_simulation(self, intent, user_id=None, path_config=None,
                                  phase="", case_spec_override=None):
            raise TypeError("unexpected local keyword")

    evaluator = EvoSAGEEpisodeEvaluator(
        lambda *_args, **_kwargs: BrokenPipeline(), invalid_evaluation_retries=0,
    )
    case = SimpleNamespace(
        case_id="case-1", scenario="ecommerce_refund", intent="refund_before_shipping",
        path_config={}, case_spec=None, split="evolution", path_id=1,
    )
    result = evaluator.evaluate(
        AdversaryPolicy(), ServicePolicy(), [case], "evolution", 0, "customer_candidate"
    )[0]
    assert result.invalid_reason == "environment_execution_error"
    assert result.error_types == ["environment_execution_error"]
    assert result.protocol_valid is True
    assert result.environment_valid is False
    assert result.termination_reason == "environment_execution_error"


def test_request_budget_artifact_records_role_and_phase_counters(tmp_path):
    budget = APIRequestBudget(max_per_generation=2, max_per_run=3,
                              persist_path=tmp_path / "request_budget.json")
    with request_context(0, "customer_candidate"):
        budget.before_attempt("agent")
        budget.record_result("agent", input_tokens=5, output_tokens=3)
    snapshot = json.loads((tmp_path / "request_budget.json").read_text())
    assert snapshot["generations"]["0"]["phases"]["customer_candidate"]["provider_attempts"] == 1
    assert snapshot["generations"]["0"]["phases"]["customer_candidate"]["roles"]["agent"]["input_tokens"] == 5
