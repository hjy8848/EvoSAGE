import json
import re
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
from framework.evolution.customer_behavior_validity import assess_customer_behavior
from framework.evolution.evaluator_adapter import EvoSAGEEpisodeEvaluator, aggregate_episode_metrics
from framework.evolution.runner import EvolutionRunner, GenerationEvaluationInconclusive
from framework.evolution.schemas import CustomerPolicy, EpisodeResult, FailureSignature, ServicePolicy
from framework.evolution.service_gate import GateDecision
from framework.evolution.split_manager import SplitManager
from framework.llm_integration.llm_client import LLMResponse, OpenAIAPIClient
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


def _response(content, *, finish_reason="stop", request_id="req-1", usage=None,
              reasoning_content=None, claims=None, raw_response_text=False):
    usage = usage or {"prompt_tokens": 23, "completion_tokens": 7}
    if content and str(content).strip() and not raw_response_text:
        if claims is None:
            claims = []
            for field, pattern in {
                "order_id": r"(?:订单号|order number|order id)\s*(?:是|为|is|[:：])?\s*([A-Za-z0-9_-]{4,})",
                "customer_id": r"(?:客户号|customer number|customer id)\s*(?:是|为|is|[:：])?\s*([A-Za-z0-9_-]{4,})",
            }.items():
                match = re.search(pattern, str(content), re.I)
                if match:
                    claims.append({"kind": "FACT", "field": field, "value": match.group(1), "basis": "KNOWN_FACT"})
        content = json.dumps({"utterance": str(content), "claims": claims}, ensure_ascii=False)
    raw = {
        "id": request_id,
        "choices": [{
            "finish_reason": finish_reason,
            "message": {
                "role": "assistant", "content": content,
                **({"reasoning_content": reasoning_content} if reasoning_content is not None else {}),
            },
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
        self.request_kwargs = []
        self.calls = 0

    def generate(self, **kwargs):
        self.calls += 1
        self.prompts.append(kwargs["prompt"])
        self.request_kwargs.append(dict(kwargs))
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
        customer_policy=policy,
    )

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
    assert user.get_customer_simulator_provenance()[0]["policy_constraint_overrides"] == [{
        "constraint": "mandatory_opening_disclosure",
        "field": "order_id",
        "reason": "policy_constraint_overridden_by_case_contract",
    }]


def test_customer_policy_is_injected_in_opening_and_followup_requests():
    case = _case(show_order_id_initially=False)
    policy = CustomerPolicy(
        policy_id="persistent-guidance",
        strategy_tags=["truthful", "cooperative"],
        pressure_strategy="do not pressure the agent",
    )
    client = _SequenceClient([
        _response("我想申请退款，请帮我核实。"),
        _response("我会配合核验，请告知下一步。", request_id="reply-2"),
    ])
    user = LLMUserModel(
        _profile(), "Customer prompt base.", llm_client=client,
        case_spec=case, customer_policy=policy,
    )
    user.generate_initial_message()
    user.dialogue_history = [
        {"role": "user", "content": "我想申请退款，请帮我核实。"},
        {"role": "assistant", "content": "我正在查询。"},
    ]
    user.generate_next_message("我正在查询。", turn_count=1)

    guidance = policy.runtime_guidance()
    assert len(client.prompts) == 2
    assert all(guidance in prompt for prompt in client.prompts)
    assert all("策略优先级" in prompt for prompt in client.prompts)
    provenance = user.get_customer_simulator_provenance()
    assert [record["stage"] for record in provenance] == ["opening", "reply"]
    assert all(record["policy_guidance_injected"] for record in provenance)
    assert all(record["customer_policy_id"] == policy.policy_id for record in provenance)


def test_non_mandatory_opening_without_identifier_remains_valid():
    case = _case(show_order_id_initially=False)
    client = _SequenceClient([_response("I want to return my earbuds.")])
    user = LLMUserModel(_profile(), "Customer prompt base.", client, max_tokens=1536, case_spec=case)

    assert get_customer_opening_contract(case) == []
    assert user._mandatory_opening_order_id() is None
    prompt = user._build_initial_message_prompt()
    assert "强制首轮披露约定" not in prompt
    assert user.private_knowledge["order_id"] == "ORD-123456"
    assert "你私下知道的信息" in prompt
    assert '"order_id": "ORD-123456"' in prompt
    assert "不要主动说出订单号" in prompt
    assert user.generate_initial_message() == "I want to return my earbuds."
    assert user.get_customer_simulator_provenance()[0]["status"] == "valid"


def test_same_call_customer_envelope_keeps_claims_internal_and_traceable():
    from framework.evolution.trace import flatten_simulation

    client = _SequenceClient([_response(
        "I got my wireless earbuds.",
        claims=[
            {"kind": "FACT", "field": "product_name", "value": "wireless earbuds", "basis": "KNOWN_FACT"},
            {"kind": "FACT", "field": "received", "value": True, "basis": "KNOWN_FACT"},
        ],
    )])
    case = _case(show_order_id_initially=False)
    case.user_knowledge["product_name"] = "wireless earbuds"
    case.user_knowledge["has_received"] = True
    user = LLMUserModel(_profile(), llm_client=client, case_spec=case)
    utterance = user.generate_initial_message()
    claim_metadata = user.get_last_customer_claim_metadata()

    assert utterance == "I got my wireless earbuds."
    assert client.calls == 1
    assert claim_metadata["parse_status"] == "valid"
    assert [claim["field"] for claim in claim_metadata["claims"]] == ["product_name", "received"]
    assert "claims" in client.prompts[0][-900:]

    agent = _CapturingAgent()
    result = DialogueSimulator(
        user_model=user, agent_model=agent, max_turns=1, case_spec=case,
    ).run(utterance)
    assert agent.messages == [utterance]
    assert result.turns[0].customer_claim_metadata["claims"] == claim_metadata["claims"]
    claim_events = [event for event in flatten_simulation(result) if event.event_type == "CUSTOMER_CLAIMS"]
    assert len(claim_events) == 1
    assert claim_events[0].payload["claims"] == claim_metadata["claims"]
    assert "FACT" not in agent.messages[0]


def test_customer_envelope_parses_epistemic_basis_and_customer_state():
    raw = json.dumps({
        "utterance": "My tracking shows it was signed for; I changed my mind.",
        "claims": [
            {"kind": "FACT", "field": "shipping_status", "value": "Signed", "basis": "CUSTOMER_BELIEF"},
            {"kind": "CUSTOMER_STATE", "field": "return_reason", "value": "changed_mind", "basis": "CUSTOMER_STATE"},
            {"kind": "OPINION", "field": "experience", "value": "frustrating", "basis": "NOT_APPLICABLE"},
        ],
    }, ensure_ascii=False)
    utterance, metadata = LLMUserModel._parse_customer_response(raw)
    assert utterance == "My tracking shows it was signed for; I changed my mind."
    assert metadata["parse_status"] == "valid"
    assert [claim["basis"] for claim in metadata["claims"]] == [
        "CUSTOMER_BELIEF", "CUSTOMER_STATE", "NOT_APPLICABLE",
    ]


def test_grounded_customer_claim_without_basis_is_protocol_invalid():
    raw = json.dumps({
        "utterance": "The package was signed for.",
        "claims": [{"kind": "FACT", "field": "shipping_status", "value": "Signed"}],
    })
    utterance, metadata = LLMUserModel._parse_customer_response(raw)
    assert utterance == "The package was signed for."
    assert metadata["parse_status"] == "invalid"
    assert metadata["parse_error"] == "customer_response_claim_0_invalid_or_missing_basis"


def test_malformed_claims_metadata_does_not_hide_or_validate_utterance():
    malformed = json.dumps({"utterance": "I want to return the item.", "claims": "not-a-list"})
    client = _SequenceClient([_response(malformed, raw_response_text=True)])
    user = LLMUserModel(_profile(), llm_client=client, case_spec=_case(show_order_id_initially=False))
    assert user.generate_initial_message() == "I want to return the item."
    metadata = user.get_last_customer_claim_metadata()
    assert metadata["parse_status"] == "invalid"
    assert metadata["parse_error"] == "customer_response_claims_missing_or_not_list"
    assert user.get_customer_simulator_provenance()[0]["claim_metadata_status"] == "invalid"


def test_private_known_order_id_survives_opening_withholding_and_is_available_on_request():
    case = _case(show_order_id_initially=False)
    policy = CustomerPolicy(
        policy_id="truthful-private-order-id",
        strategy_tags=["truthful", "cooperative"],
    )
    client = _SequenceClient([
        _response("我想申请退货，请帮我处理。"),
        _response("我的订单号是 ORD-123456，请帮我核实。", request_id="asked-id"),
    ])
    user = LLMUserModel(
        _profile(), llm_client=client, case_spec=case, customer_policy=policy,
    )
    opening = user.generate_initial_message()
    assert "ORD-123456" not in opening
    user.add_user_message(opening)
    user.add_assistant_message("请提供订单号，以便我核实订单状态。")
    reply = user.generate_next_message("请提供订单号，以便我核实订单状态。", 1)

    assert reply == "我的订单号是 ORD-123456，请帮我核实。"
    reply_prompt = client.prompts[1]
    assert '"order_id": "ORD-123456"' in reply_prompt
    assert "客服本轮询问订单号" in reply_prompt
    assert "必须使用私有知识中的准确值 ORD-123456" in reply_prompt


def test_private_identifier_prompt_keeps_order_and_customer_id_roles_distinct():
    case = _case(show_order_id_initially=False)
    client = _SequenceClient([_response("订单号是 ORD-123456。")])
    user = LLMUserModel(_profile(), llm_client=client, case_spec=case)
    user.dialogue_history = [{"role": "assistant", "content": "请提供订单号。"}]
    reply = user.generate_next_message("请提供订单号。", 1)

    assert reply == "订单号是 ORD-123456。"
    prompt = client.prompts[0]
    assert '"order_id": "ORD-123456"' in prompt
    assert '"customer_id": "CUS-123"' in prompt
    assert "不能猜测或替换成其他编号" in prompt
    assert user._requested_identifier_field("请提供订单号。") == "order_id"


def test_initial_visibility_changes_disclosure_instruction_not_private_knowledge():
    hidden_at_opening = _case(show_order_id_initially=False)
    visible_at_opening = _case(show_order_id_initially=True)
    hidden_user = LLMUserModel(_profile(), case_spec=hidden_at_opening)
    visible_user = LLMUserModel(_profile(), case_spec=visible_at_opening)

    assert hidden_user.private_knowledge == visible_user.private_knowledge
    assert hidden_user._known_case_facts_text() == visible_user._known_case_facts_text()
    assert "不要主动说出订单号" in hidden_user._disclosure_timing_text("opening")
    assert "必须逐字使用该值" in visible_user._disclosure_timing_text("opening")


def test_withholding_changes_turn_disclosure_but_preserves_exact_private_identifier():
    case = _case(show_order_id_initially=False)
    policy = CustomerPolicy(
        policy_id="withhold-but-know",
        strategy_tags=["truthful", "withholding"],
    )
    client = _SequenceClient([
        _response("可以先说明需要订单号核验什么吗？"),
        _response("订单号是 ORD-123456。"),
    ])
    user = LLMUserModel(_profile(), llm_client=client, case_spec=case, customer_policy=policy)
    first_instruction = user._disclosure_timing_text("reply", "请提供订单号。")
    assert "订单号" in first_instruction and "不要说出编号" in first_instruction
    assert user.private_knowledge["order_id"] == "ORD-123456"

    user.dialogue_history = [
        {"role": "user", "content": "我想申请退货。"},
        {"role": "assistant", "content": "请提供订单号。"},
        {"role": "user", "content": "可以先说明需要订单号核验什么吗？"},
    ]
    second_prompt = user._build_generation_prompt(
        "需要订单号用于核验订单状态。", turn_count=2,
    )
    assert '"order_id": "ORD-123456"' in second_prompt
    assert "必须使用私有知识中的准确值 ORD-123456" in second_prompt


def test_unknown_order_id_is_not_added_to_private_knowledge_or_guessed():
    case = _case(show_order_id_initially=False)
    case.user_knowledge["knows_order_id"] = False
    user = LLMUserModel(_profile(), case_spec=case)

    assert "order_id" not in user.private_knowledge
    assert "order_id" not in user.environment_state.known_facts
    instruction = user._disclosure_timing_text("reply", "请提供订单号。")
    assert "你并不知道该字段的值" in instruction
    assert "ORD-123456" not in user._known_case_facts_text()


def test_unsupported_identifier_remains_customer_behavior_invalid():
    case = _case(show_order_id_initially=False)
    simulation = SimpleNamespace(
        case_spec=case.to_dict(),
        turns=[
            SimpleNamespace(
                user_message="我想申请退货。",
                agent_output=SimpleNamespace(chat="请提供订单号，以便核实。"),
            ),
            SimpleNamespace(
                user_message="我的订单号是 ORD-FAKE-999。",
                agent_output=SimpleNamespace(chat=""),
            ),
        ],
        backend_events=[],
        user_environment_state={"goal": case.user_goal},
    )
    assessment = assess_customer_behavior(
        simulation, CustomerPolicy(strategy_tags=["truthful", "cooperative"]),
    )

    assert assessment.valid is False
    assert "unsupported_fact_generation:order_id" in assessment.reasons


def test_customer_thinking_mode_is_opt_in_and_saved_in_generation_provenance():
    default_client = _SequenceClient([_response("我要申请退款，订单号是 ORD-123456。")])
    default_user = LLMUserModel(
        _profile(), llm_client=default_client, case_spec=_case(), max_tokens=1536
    )
    assert default_user.generate_initial_message()
    assert "thinking" not in default_client.request_kwargs[0]
    assert default_user.get_customer_simulator_provenance()[0]["thinking_mode"] == "default"

    disabled_client = _SequenceClient([_response(
        "我要申请退款，订单号是 ORD-123456。",
        reasoning_content="test-only hidden reasoning",
        usage={"prompt_tokens": 23, "completion_tokens": 17,
               "completion_tokens_details": {"reasoning_tokens": 9}},
    )])
    disabled_user = LLMUserModel(
        _profile(), llm_client=disabled_client, case_spec=_case(), max_tokens=1536,
        thinking_mode="disabled",
    )
    assert disabled_user.generate_initial_message()
    assert disabled_client.request_kwargs[0]["thinking"] == {"type": "disabled"}
    generation = disabled_user.get_customer_simulator_provenance()[0]
    assert generation["thinking_mode"] == "disabled"
    assert generation["attempts"][0]["thinking_mode"] == "disabled"
    assert generation["attempts"][0]["reasoning_content_present"] is True
    assert "reasoning_content" not in generation["attempts"][0]


def test_openai_api_client_forwards_only_explicit_thinking_control(monkeypatch):
    sent_payloads = []

    class FakeHTTPResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "id": "req-thinking-test",
                "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1},
            }

    def fake_post(_url, *, json, headers, timeout):
        sent_payloads.append(json)
        return FakeHTTPResponse()

    monkeypatch.setattr("framework.llm_integration.llm_client.requests.post", fake_post)
    client = OpenAIAPIClient(
        api_key="test-only", base_url="https://inferaiapi.com/v1",
        model_name="deepseek-v4-flash", max_retries=1,
    )
    client.generate("Say hello briefly.", max_tokens=1536)
    client.generate(
        "Say hello briefly.", max_tokens=1536, thinking={"type": "disabled"}
    )

    assert sent_payloads[0]["max_tokens"] == 1536
    assert "thinking" not in sent_payloads[0]
    assert sent_payloads[1]["thinking"] == {"type": "disabled"}


def test_pipeline_thinking_setting_is_customer_scoped(tmp_path, monkeypatch):
    import run_evaluation_with_llm

    created_clients = []
    created_users = []
    created_agents = []

    class FakeClient:
        def __init__(self, role, config):
            self.role = role
            self.config = config

    def fake_get_llm_client(_client_type, **config):
        client = FakeClient(len(created_clients), config)
        created_clients.append(client)
        return client

    class CapturingUser(LLMUserModel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created_users.append(self)

        def generate_initial_message(self):
            order_id = self._mandatory_opening_order_id()
            return f"我想咨询一下订单 {order_id or 'unknown'}。"

    class FakeAgent:
        def __init__(self, *_args, **kwargs):
            self.kwargs = kwargs
            created_agents.append(self)

    class FakeSimulationResult:
        simulation_id = "thinking-mode-test"
        turns = []
        model_name = ""

        def to_dict(self):
            return {"simulation_id": self.simulation_id}

    class FakeSimulator:
        def __init__(self, **_kwargs):
            pass

        def run(self, **_kwargs):
            return FakeSimulationResult()

    class FakeEvaluator:
        def __init__(self, **_kwargs):
            pass

        def evaluate_simulation(self, **_kwargs):
            return SimpleNamespace(to_dict=lambda: {})

    monkeypatch.setattr(run_evaluation_with_llm, "get_llm_client", fake_get_llm_client)
    monkeypatch.setattr(run_evaluation_with_llm, "LLMUserModel", CapturingUser)
    monkeypatch.setattr(run_evaluation_with_llm, "AgentModel", FakeAgent)
    monkeypatch.setattr(run_evaluation_with_llm, "DialogueSimulator", FakeSimulator)
    monkeypatch.setattr(run_evaluation_with_llm, "Evaluator", FakeEvaluator)

    pipeline = run_evaluation_with_llm.LLMEvaluationPipeline(
        scenario_id="ecommerce_refund", model_name="deepseek-v4-flash",
        output_dir=str(tmp_path / "pipeline"), eval_mode="api",
        api_key="test-only", api_url="https://inferaiapi.com/v1",
        user_model_name="deepseek-v4-flash", agent_model_type="api",
        agent_model_name="deepseek-v4-flash", judge_model_name="deepseek-v4-flash",
        user_max_tokens=1536, customer_thinking_mode="disabled", use_llm_judge=False,
        verbose=False,
    )
    pipeline.run_single_simulation("exchange_product", user_id="thinking-test-user")

    assert created_users[-1].thinking_mode == "disabled"
    # Customer thinking is role-scoped; Agent keeps provider-default thinking
    # unless the caller explicitly configures an Agent mode.
    assert created_agents[-1].kwargs["thinking_mode"] is None
    assert len(created_clients) == 3
    assert all("thinking" not in client.config for client in created_clients)


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
    assert "PRIVATE-CREDIT-SENTINEL" not in user.private_knowledge.values()
    assert "PRIVATE-BACKEND-SENTINEL" not in user.private_knowledge.values()
    assert "PRIVATE-GT-SENTINEL" not in user.private_knowledge.values()
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
    score = CustomerSelector().score(policy, [episode], set())
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
        service_failure_attributable=True,
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

    score = CustomerSelector().score(policy, [valid_failure, invalid], set())
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
    # All API roles use the explicit one-attempt default instead of silently
    # inheriting the OpenAI-compatible client's retry=3 default.
    assert agent.config["max_retries"] == 1
    assert judge.config["max_retries"] == 1
