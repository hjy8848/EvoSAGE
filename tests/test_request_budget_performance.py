import json
import sys
from types import SimpleNamespace

import pytest
import requests

from framework.evolution.config import EvolutionConfig, load_config, save_config
from framework.evolution.evaluator_adapter import EvoSAGEEpisodeEvaluator
from framework.evolution.persistence import RunStore
from framework.evolution.request_budget import APIRequestBudget, APIRequestBudgetExceeded, request_context
from framework.evolution.runner import EvolutionRunner
from framework.evolution.schemas import CustomerPolicy, ServicePolicy
from framework.evolution.config import PersistenceConfig, SplitConfig
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
    legacy = EvolutionConfig.from_dict({})
    assert legacy.evaluation.max_tool_steps == 8
    assert legacy.evaluation.invalid_evaluation_retries == 1
    assert legacy.evaluation.agent_max_retries == 1

    config = EvolutionConfig.from_dict({
        "evaluation": {
            "agent_thinking_mode": "disabled",
            "evolver_thinking_mode": "enabled",
            "agent_max_retries": 1,
            "customer_transport_max_retries": 2,
            "evolver_max_retries": 1,
            "judge_validation_retries": 0,
            "invalid_evaluation_retries": 0,
            "max_tool_steps": 3,
            "max_api_requests_per_generation": 30,
            "max_api_requests_per_run": 30,
            "rate_limit_backoff_seconds": 0,
        }
    })
    path = tmp_path / "roundtrip.json"
    save_config(config, path)
    loaded = load_config(path)
    assert loaded.evaluation == config.evaluation
    assert loaded.evaluation.agent_thinking_mode == "disabled"
    assert loaded.evaluation.evolver_thinking_mode == "enabled"
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
        def run_single_simulation(self, intent, **kwargs):
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
        CustomerPolicy(), ServicePolicy(), [case], "evolution", 0, "customer_candidate"
    )[0]
    assert len(calls) == expected_calls
    assert result.is_evaluation_invalid()
    assert not result.is_attributable_service_failure()
    assert result.invalid_reason == "provider_error"


def test_budget_exhaustion_persists_checkpoint_and_is_not_a_business_episode(tmp_path):
    def stop_with_budget(*_args, **_kwargs):
        raise APIRequestBudgetExceeded("generation", 1, 1)

    config = EvolutionConfig.from_dict({
        "experiment_mode": "static",
        "max_generations": 1,
        "splits": {
            "strategy": "instance_holdout", "seed": 7,
            "evolution_ratio": 0.5, "validation_ratio": 0.25,
            "heldout_ratio": 0.25, "max_cases": 3,
        },
        "persistence": {"output_dir": str(tmp_path / "budget-run"), "resume": False},
    })
    runner = EvolutionRunner(
        config,
        evaluator=type("Evaluator", (), {"evaluate": staticmethod(stop_with_budget)})(),
    )
    result = runner.run()
    assert result["run_status"] == "budget_exhausted"
    metrics = json.loads((runner.store.run_dir / "analysis/orchestration_metrics.json").read_text())
    checkpoint = json.loads(
        (runner.store.run_dir / "generations/gen_000/CHECKPOINT.json").read_text()
    )
    assert metrics["run_status"] == "budget_exhausted"
    assert metrics["inconclusive_reason"] == "api_request_budget_exceeded"
    assert checkpoint["status"] == "incomplete_budget_exhausted"
    assert checkpoint["reason"] == "api_request_budget_exceeded"
    assert checkpoint["active_customer_policy"]["policy_id"] == "customer_policy_c0"
    assert checkpoint["active_service_policy"]["policy_id"] == "service_policy_s0"
    assert checkpoint["stage"] == "generation_summary"
    assert checkpoint["completed_phases"] == []
