import json
from types import SimpleNamespace

import pytest

from framework.backend import CaseSpec, create_backend
from framework.core.simulator import DialogueSimulator
from framework.core.customer_contract import get_customer_opening_contract
from framework.evolution.archives import AttackArchive
from framework.evolution.config import (
    CustomerEvolutionConfig,
    EvolutionConfig,
    PersistenceConfig,
    ServiceEvolutionConfig,
    SplitConfig,
)
from framework.evolution.customer_selector import CustomerSelector
from framework.evolution.customer_policy import CustomerPolicyCompiler
from framework.evolution.evaluator_adapter import EvoSAGEEpisodeEvaluator, aggregate_episode_metrics
from framework.evolution.runner import EvolutionRunner, GenerationEvaluationInconclusive
from framework.evolution.schemas import CustomerPolicy, EpisodeResult, FailureSignature, ServicePolicy
from framework.evolution.service_gate import GateDecision
from framework.evolution.split_manager import SplitManager
from framework.llm_integration.llm_client import LLMResponse
from framework.llm_integration.llm_user_model import (
    CustomerSimulatorProtocolError,
    LLMUserModel,
)
from framework.models import AgentTurnOutput, UserProfile


def _case(*, show_order_id_initially=True):
    return CaseSpec(
        case_id="case-customer-protocol",
        scenario="ecommerce_refund",
        backend_record={
            "order": {
                "order_id": "ORD-123456",
                "shipping_status": "Signed",
                "payment_status": "Paid",
                "refund_status": "None",
                "return_window_open": True,
            },
            "customer": {"customer_id": "CUS-123"},
        },
        user_goal={"type": "refund"},
        user_knowledge={
            "order_id": "ORD-123456",
            "knows_order_id": True,
            "customer_id": "CUS-123",
            "knows_customer_id": True,
        },
        user_policy={
            "show_order_id_initially": show_order_id_initially,
            "reveal_order_id_on_request": True,
        },
        initial_observation={},
        expected_outcome={},
    )


def _response(content, *, finish_reason="stop", request_id="req-1", usage=None):
    usage = usage or {"prompt_tokens": 23, "completion_tokens": 7}
    raw = {
        "id": request_id,
        "choices": [{
            "finish_reason": finish_reason,
            "message": {"role": "assistant", "content": content},
        }],
        "usage": usage,
    }
    return LLMResponse(
        text=content or "",
        model="fake-customer",
        tokens=usage.get("completion_tokens", 0),
        metadata={
            "finish_reason": finish_reason,
            "usage": usage,
            "latency_seconds": 0.01,
        },
        raw_response=raw,
    )


class _SequenceClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []
        self.calls = 0

    def generate(self, **kwargs):
        self.calls += 1
        self.prompts.append(kwargs["prompt"])
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def _profile():
    return UserProfile(
        user_id="customer-test",
        user_intent="refund_request",
        adversarial_intensity="weak_conflict",
        scenario_id="ecommerce_refund",
    )


def test_mandatory_opening_contract_overrides_customer_withholding_policy():
    case = _case(show_order_id_initially=True)
    policy = CustomerPolicy(
        policy_id="withholding-opening-test",
        strategy_tags=["truthful", "withholding", "delayed_disclosure"],
        disclosure_strategy="withhold optional known details until asked",
    )
    client = _SequenceClient([_response("我想办理换货，订单号是 ORD-123456。")])
    user = LLMUserModel(
        _profile(),
        "Customer prompt base.",
        llm_client=client,
        max_tokens=1536,
        case_spec=case,
    )
    user.system_prompt += CustomerPolicyCompiler().compile(policy, case).runtime_guidance()

    prompt = user._build_initial_message_prompt()
    contract = get_customer_opening_contract(case)
    assert contract == [{
        "field": "order_id",
        "value": "ORD-123456",
        "timing": "opening_turn",
    }]
    assert "强制首轮披露约定" in prompt
    assert "ORD-123456" in prompt
    assert "优先于 CustomerPolicy" in prompt
    assert user._mandatory_opening_order_id() == contract[0]["value"]

    message = user.generate_initial_message()
    assert "ORD-123456" in message
    assert user.get_customer_simulator_provenance()[0]["status"] == "valid"


def test_non_mandatory_opening_without_identifier_remains_valid():
    case = _case(show_order_id_initially=False)
    client = _SequenceClient([_response("I want to return my earbuds.")])
    user = LLMUserModel(_profile(), "Customer prompt base.", client, max_tokens=1536, case_spec=case)

    assert get_customer_opening_contract(case) == []
    assert user._mandatory_opening_order_id() is None
    prompt = user._build_initial_message_prompt()
    assert "强制首轮披露约定" not in prompt
    assert "ORD-123456" not in prompt
    assert user.generate_initial_message() == "I want to return my earbuds."
    assert user.get_customer_simulator_provenance()[0]["status"] == "valid"


def test_customer_opening_prompt_never_exposes_backend_only_values():
    case = _case(show_order_id_initially=False)
    case.user_knowledge = {
        "knows_order_id": False,
        "knows_customer_id": False,
        "product_name": "customer-known product",
    }
    case.backend_record["customer"]["credit_level"] = "PRIVATE-CREDIT-SENTINEL"
    case.backend_record["private_state"] = {
        "system_variables": {"private_marker": "PRIVATE-BACKEND-SENTINEL"}
    }
    case.expected_outcome = {"private_expected_marker": "PRIVATE-GT-SENTINEL"}
    user = LLMUserModel(_profile(), "Customer prompt base.", case_spec=case)

    prompt = user._build_initial_message_prompt()
    assert "PRIVATE-CREDIT-SENTINEL" not in prompt
    assert "PRIVATE-BACKEND-SENTINEL" not in prompt
    assert "PRIVATE-GT-SENTINEL" not in prompt
    assert "credit_level" not in prompt
    assert "expected_outcome" not in prompt


def test_prompt_and_post_generation_guard_use_the_same_opening_contract():
    mandatory_case = _case(show_order_id_initially=True)
    optional_case = _case(show_order_id_initially=False)
    mandatory_user = LLMUserModel(_profile(), "base", case_spec=mandatory_case)
    optional_user = LLMUserModel(_profile(), "base", case_spec=optional_case)

    contract = get_customer_opening_contract(mandatory_case)
    assert mandatory_user._mandatory_opening_order_id() == contract[0]["value"]
    assert contract[0]["value"] in mandatory_user._build_initial_message_prompt()
    assert optional_user._mandatory_opening_order_id() is None
    assert get_customer_opening_contract(optional_case) == []
    assert "强制首轮披露约定" not in optional_user._build_initial_message_prompt()


class _CapturingAgent:
    scenario_id = "ecommerce_refund"
    path_taken = []

    def __init__(self, backend=None):
        self.messages = []
        self.backend = backend

    def process_turn(self, user_message, backend_environment=None, **_kwargs):
        self.messages.append(user_message)
        output = AgentTurnOutput(turn_id=0, chat="请问您能提供更多信息吗？")
        if self.backend is not None:
            result = backend_environment.execute_tool(
                "query_order", {"order_id": ""}, turn_index=0, call_id="query-empty"
            )
            output.tool_calls = [{
                "call_id": "query-empty",
                "name": "query_order",
                "arguments": {"order_id": ""},
            }]
            output.tool_results = [result.to_dict()]
        return output


def test_empty_customer_opening_retries_and_only_valid_message_reaches_agent():
    client = _SequenceClient([
        _response("", request_id="empty-1"),
        _response(
            "我要申请退货，订单号是 ORD-123456。",
            request_id="valid-2",
            usage={
                "prompt_tokens": 23,
                "completion_tokens": 17,
                "completion_tokens_details": {"reasoning_tokens": 9},
            },
        ),
    ])
    user = LLMUserModel(_profile(), llm_client=client, case_spec=_case())

    initial_message = user.generate_initial_message()
    assert client.calls == 2
    assert client.prompts[0] == client.prompts[1]
    assert initial_message == "我要申请退货，订单号是 ORD-123456。"
    assert user.dialogue_history == []
    provenance = user.get_customer_simulator_provenance()[0]
    assert provenance["status"] == "valid"
    assert provenance["retry_count"] == 1
    assert provenance["attempts"][0]["invalid_reason"] == "empty_message"
    assert provenance["attempts"][1]["generation_status"] == "valid"
    assert provenance["attempts"][0]["provider_request_id"] == "empty-1"
    assert provenance["attempts"][1]["input_tokens"] == 23
    assert provenance["attempts"][1]["completion_tokens"] == 17
    assert provenance["attempts"][1]["reasoning_tokens"] == 9

    agent = _CapturingAgent()
    result = DialogueSimulator(
        user_model=user,
        agent_model=agent,
        max_turns=1,
        case_spec=_case(),
    ).run(initial_message)
    assert agent.messages == [initial_message]
    assert result.turns[0].user_message == initial_message
    assert result.to_dict()["customer_simulator_provenance"] == user.get_customer_simulator_provenance()


def test_persistent_empty_customer_generation_is_invalid_and_excluded_everywhere(tmp_path):
    client = _SequenceClient([_response(""), _response("   ", request_id="empty-2")])
    customer_case = _case()
    evaluation_case = SplitManager().build().evolution[0]
    user_model = {}

    class Pipeline:
        user_llm_client = client

        def run_single_simulation(self, *_args, **_kwargs):
            user = LLMUserModel(_profile(), llm_client=client, case_spec=customer_case)
            user_model["value"] = user
            user.generate_initial_message()

    evaluator = EvoSAGEEpisodeEvaluator(
        lambda *_args, **_kwargs: Pipeline(),
        cache_path=tmp_path / "episode_cache.jsonl",
        invalid_evaluation_retries=1,
    )
    policy = CustomerPolicy(policy_id="empty-customer")
    episode = evaluator.evaluate(
        policy, ServicePolicy(), [evaluation_case], "evolution", 0, "customer_candidate"
    )[0]

    assert client.calls == 2  # no second adapter-level retry
    assert episode.evaluation_status == "invalid"
    assert episode.invalid_reason == "customer_simulator_invalid:empty_message"
    assert episode.termination_reason == "customer_simulator_invalid"
    assert episode.metadata["customer_simulator_protocol_invalid"] is True
    attempts = episode.metadata["customer_simulator_provenance"][0]["attempts"]
    assert [item["raw_content"] for item in attempts] == ["", "   "]
    assert all(item["generation_status"] == "invalid" for item in attempts)
    assert user_model["value"].last_courtesy_turn == -1
    assert user_model["value"].environment_state.resolution_status == "unsolved"
    assert user_model["value"].environment_state.revealed_facts == []
    assert not (tmp_path / "episode_cache.jsonl").exists()
    invalid_record = json.loads((tmp_path / "invalid_evaluations.jsonl").read_text().splitlines()[0])
    assert invalid_record["episode"]["evaluation_status"] == "invalid"
    assert len(invalid_record["episode"]["metadata"]["customer_simulator_provenance"][0]["attempts"]) == 2

    aggregate = aggregate_episode_metrics([episode])
    assert aggregate["episodes"] == 0
    score = CustomerSelector().score(policy, [episode], set(), total_nodes=1)
    assert score.attack_success is None
    assert score.novelty is None
    assert score.coverage is None
    assert score.fitness is None
    assert score.evaluation_status == "inconclusive"
    assert score.invalid_episode_count == 1
    assert score.episodes == 0
    selected, _ = CustomerSelector().select([(policy, [episode])], set())
    assert selected is None

    signature = FailureSignature.from_episode(episode)
    archive = AttackArchive(tmp_path / "attacks.jsonl")
    assert archive.add(policy, [signature], [episode]) == 0
    assert len(archive) == 0


def test_invalid_customer_episode_is_not_a_service_source_failure(tmp_path):
    evaluation_case = SplitManager(SplitConfig(max_cases=3)).build().evolution[0]

    class InvalidEvaluator:
        def evaluate(self, customer_policy, service_policy, cases, split, generation, phase):
            return [EpisodeResult(
                episode_id=f"invalid-{case.case_id}-{phase}",
                scenario="ecommerce_refund",
                case_id=case.case_id,
                customer_policy_id=customer_policy.policy_id,
                service_policy_id=service_policy.policy_id,
                split=split,
                generation=generation,
                task_success=False,
                execution_score=0,
                error_types=["protocol_failure", "customer_simulator_invalid:empty_message"],
                evaluation_status="invalid",
                invalid_reason="customer_simulator_invalid:empty_message",
                metadata={"protocol_failure": True},
            ) for case in cases]

    class ServiceEvolverSpy:
        last_candidate_records = []
        last_generation_record = None
        last_baseline_metrics = {}
        last_selected_metrics = {}

        def __init__(self):
            self.seen_failures = None

        def evolve(self, incumbent, failures, *_args, **_kwargs):
            self.seen_failures = list(failures)
            return incumbent, GateDecision(False, "no_candidate"), None

    service_evolver = ServiceEvolverSpy()
    config = EvolutionConfig(
        experiment_mode="service_only",
        max_generations=1,
        splits=SplitConfig(max_cases=3),
        service=ServiceEvolutionConfig(candidate_count=0, replay_attack_count=0),
        persistence=PersistenceConfig(output_dir=str(tmp_path / "service-only")),
    )
    runner = EvolutionRunner(config, evaluator=InvalidEvaluator(), service_evolver=service_evolver)
    with pytest.raises(GenerationEvaluationInconclusive):
        runner.run()

    assert service_evolver.seen_failures == []
    generation_episodes = json.loads(
        (tmp_path / "service-only/generations/gen_000/episodes.jsonl").read_text().splitlines()[0]
    )
    assert generation_episodes["evaluation_status"] == "invalid"
    assert generation_episodes["invalid_reason"] == "customer_simulator_invalid:empty_message"
    assert not (tmp_path / "service-only/generations/gen_000/COMPLETE.json").exists()
    metrics = json.loads(
        (tmp_path / "service-only/analysis/orchestration_metrics.json").read_text()
    )
    assert metrics["run_status"] == "inconclusive"
    assert metrics["inconclusive_reason"] == "generation_summary_invalid:no_valid_episode_evidence"


def test_all_invalid_customer_candidate_evaluations_are_inconclusive_without_complete(tmp_path):
    class InvalidEvaluator:
        def evaluate(self, customer_policy, service_policy, cases, split, generation, phase):
            return [EpisodeResult(
                episode_id=f"invalid-{case.case_id}-{phase}",
                scenario="ecommerce_refund",
                case_id=case.case_id,
                customer_policy_id=customer_policy.policy_id,
                service_policy_id=service_policy.policy_id,
                split=split,
                generation=generation,
                task_success=False,
                execution_score=0,
                error_types=["protocol_failure", "customer_simulator_invalid:output_truncated"],
                evaluation_status="invalid",
                invalid_reason="customer_simulator_invalid:output_truncated",
            ) for case in cases]

    config = EvolutionConfig(
        experiment_mode="customer_only",
        max_generations=1,
        customer=CustomerEvolutionConfig(candidate_count=1, elite_count=1),
        splits=SplitConfig(max_cases=3),
        persistence=PersistenceConfig(output_dir=str(tmp_path / "customer-only")),
    )
    runner = EvolutionRunner(config, evaluator=InvalidEvaluator())
    with pytest.raises(GenerationEvaluationInconclusive, match="customer_candidate_evaluation_invalid"):
        runner.run()

    candidates = json.loads(
        (tmp_path / "customer-only/generations/gen_000/customer_candidates.json").read_text()
    )
    assert candidates["evaluation_status"] == "inconclusive"
    assert candidates["selection_status"] == "inconclusive"
    assert all(score["fitness"] is None for score in candidates["scores"])
    assert all(score["episodes"] == 0 for score in candidates["scores"])
    assert not (tmp_path / "customer-only/generations/gen_000/COMPLETE.json").exists()
    metrics = json.loads(
        (tmp_path / "customer-only/analysis/orchestration_metrics.json").read_text()
    )
    assert metrics["run_status"] == "inconclusive"
    assert metrics["inconclusive_reason"] == "customer_candidate_evaluation_invalid:no_valid_episode_evidence"


def test_customer_score_uses_only_valid_episodes_when_invalids_are_mixed():
    policy = CustomerPolicy(policy_id="mixed-customer")
    valid_failure = EpisodeResult(
        episode_id="valid-business-failure",
        scenario="ecommerce_refund",
        case_id="case-valid",
        customer_policy_id=policy.policy_id,
        service_policy_id="service",
        split="evolution",
        generation=0,
        task_success=False,
        execution_score=0.0,
        error_types=["order_not_found"],
        sop_node="shipping_status",
    )
    invalid = EpisodeResult(
        episode_id="invalid-customer-generation",
        scenario="ecommerce_refund",
        case_id="case-invalid",
        customer_policy_id=policy.policy_id,
        service_policy_id="service",
        split="evolution",
        generation=0,
        task_success=False,
        execution_score=0.0,
        error_types=["protocol_failure", "customer_simulator_invalid:empty_message"],
        evaluation_status="invalid",
        invalid_reason="customer_simulator_invalid:empty_message",
    )

    score = CustomerSelector().score(policy, [valid_failure, invalid], set(), total_nodes=1)
    assert score.evaluation_status == "valid"
    assert score.episodes == 1
    assert score.invalid_episode_count == 1
    assert score.attack_success == 1.0


def test_nonmandatory_opening_without_order_id_remains_valid_business_failure():
    case = _case(show_order_id_initially=False)
    client = _SequenceClient([_response("我想退耳机，请帮我安排上门取件。")])
    user = LLMUserModel(_profile(), llm_client=client, case_spec=case)
    initial_message = user.generate_initial_message()
    assert initial_message and "ORD-123456" not in initial_message
    assert user.get_customer_simulator_provenance()[0]["status"] == "valid"

    backend = create_backend(case)
    backend.reset(case)
    agent = _CapturingAgent(backend)
    simulation = DialogueSimulator(
        user_model=user,
        agent_model=agent,
        max_turns=1,
        backend_environment=backend,
        case_spec=case,
    ).run(initial_message)
    assert agent.messages == [initial_message]
    assert simulation.backend_events[0]["name"] == "query_order"
    assert simulation.backend_events[0]["arguments"] == {"order_id": ""}
    assert simulation.backend_events[0]["result"]["error_code"] == "order_not_found"

    report = SimpleNamespace(
        details={},
        error_categories=["order_not_found"],
        task_success=False,
        required_verification_score=0.0,
        policy_compliance_score=0.0,
        action_execution_score=0.0,
        goal_fulfillment=0.0,
        execution_score=0.0,
        sage_style_score=0.0,
        predicted_action="",
        executed_action="",
        gold_path=[],
        predicted_path=[],
    )
    episode = EvoSAGEEpisodeEvaluator.from_evosage(
        simulation, report, CustomerPolicy(), ServicePolicy(), "evolution", 0, "test"
    )
    assert episode.evaluation_status == "valid"
    assert episode.task_success is False
    assert "order_not_found" in episode.error_types
    assert not episode.is_evaluation_invalid()


def test_mandatory_opening_order_id_is_enforced_but_only_for_explicit_case_contract():
    client = _SequenceClient([
        _response("我要退货，请帮我处理。"),
        _response("我需要退货，请帮忙处理一下。", request_id="still-missing"),
    ])
    user = LLMUserModel(_profile(), llm_client=client, case_spec=_case(show_order_id_initially=True))
    with pytest.raises(CustomerSimulatorProtocolError, match="customer_simulator_invalid:missing_mandatory_order_id"):
        user.generate_initial_message()
    assert client.calls == 2
    assert user.get_customer_simulator_provenance()[0]["status"] == "invalid"
    assert [item["invalid_reason"] for item in user.get_customer_simulator_provenance()[0]["attempts"]] == [
        "missing_mandatory_order_id",
        "missing_mandatory_order_id",
    ]


def test_customer_simulator_timeout_retries_once_and_records_provider_provenance():
    client = _SequenceClient([
        TimeoutError("provider read timed out"),
        _response("您好，我想咨询退款。", request_id="after-timeout"),
    ])
    user = LLMUserModel(_profile(), llm_client=client, case_spec=_case(show_order_id_initially=False))
    assert user.generate_initial_message() == "您好，我想咨询退款。"
    attempts = user.get_customer_simulator_provenance()[0]["attempts"]
    assert client.calls == 2
    assert attempts[0]["invalid_reason"] == "timeout"
    assert attempts[0]["timeout"] is True
    assert attempts[0]["provider_error"].startswith("TimeoutError:")
    assert attempts[1]["provider_request_id"] == "after-timeout"


def test_customer_simulator_provider_error_retries_once_and_records_error():
    client = _SequenceClient([
        RuntimeError("gateway unavailable"),
        _response("您好，我想咨询退款。", request_id="after-provider-error"),
    ])
    user = LLMUserModel(_profile(), llm_client=client, case_spec=_case(show_order_id_initially=False))
    assert user.generate_initial_message() == "您好，我想咨询退款。"
    attempts = user.get_customer_simulator_provenance()[0]["attempts"]
    assert client.calls == 2
    assert attempts[0]["invalid_reason"] == "provider_error"
    assert attempts[0]["provider_error"] == "RuntimeError: gateway unavailable"
    assert attempts[0]["timeout"] is False
    assert attempts[1]["provider_request_id"] == "after-provider-error"


def test_customer_simulator_truncated_output_retries_and_redacts_credentials_from_provenance():
    client = _SequenceClient([
        _response("截断", finish_reason="length", request_id="truncated"),
        _response("用户提供的令牌 sk-123456789012345678901234567890", request_id="valid"),
    ])
    user = LLMUserModel(_profile(), llm_client=client, case_spec=_case(show_order_id_initially=False))
    user.generate_initial_message()
    attempts = user.get_customer_simulator_provenance()[0]["attempts"]
    assert attempts[0]["invalid_reason"] == "output_truncated"
    assert attempts[1]["generation_status"] == "valid"
    assert "sk-123456789012345678901234567890" not in attempts[1]["raw_content"]
    assert "[REDACTED_API_KEY]" in attempts[1]["raw_content"]


def test_pipeline_disables_hidden_transport_retries_for_customer_only(tmp_path, monkeypatch):
    import run_evaluation_with_llm

    created = []

    class FakeClient:
        def __init__(self, client_type, config):
            self.client_type = client_type
            self.config = config

    def fake_get_llm_client(client_type, **config):
        client = FakeClient(client_type, config)
        created.append(client)
        return client

    monkeypatch.setattr(run_evaluation_with_llm, "get_llm_client", fake_get_llm_client)
    run_evaluation_with_llm.LLMEvaluationPipeline(
        scenario_id="ecommerce_refund",
        model_name="deepseek-v4-flash",
        output_dir=str(tmp_path / "pipeline"),
        eval_mode="api",
        api_key="test-key",
        api_url="https://inferaiapi.com/v1",
        user_model_name="deepseek-v4-flash",
        agent_model_type="api",
        agent_model_name="deepseek-v4-flash",
        judge_model_name="deepseek-v4-flash",
        client_type="openai_api",
        verbose=False,
    )

    user, agent, judge = created
    assert user.config["max_retries"] == 1
    assert "max_retries" not in agent.config
    assert "max_retries" not in judge.config
