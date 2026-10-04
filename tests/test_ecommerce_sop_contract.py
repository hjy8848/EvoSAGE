"""Cross-layer consistency checks for the ecommerce refund benchmark contract."""

import json
from copy import deepcopy
from types import SimpleNamespace

from framework import get_sop_graph
from framework.backend import build_case_spec
from framework.backend.types import CaseSpec
from framework.config import get_scenario_config
from framework.evolution.config import SplitConfig
from framework.evolution.split_manager import SplitManager
from framework.llm_integration.llm_user_model import LLMUserModel
from framework.models.agent_model import AgentModel
from framework.models.user_model import UserProfile
from framework.prompts.ecommerce_refund_prompts import (
    AGENT_SYSTEM_PROMPT,
    get_user_prompt_for_intent,
)
from framework.sop.ecommerce_refund_PathList import (
    generate_path_list,
    get_intent_path_mapping,
)
from framework.sop.sop_rule_engine import EcommerceRefundSOPRuleEngine


EXPECTED_PATH_INTENTS = {
    "exchange_product": [1, 2, 3, 4, 5],
    "refund_before_shipping": [6],
    "refund_on_the_way": [7],
    "merchant_compensation_high_credit": [8],
    "merchant_compensation_low_credit": [9, 10],
    "user_return_high_credit": [11],
    "user_return_medium_credit": [12],
    "user_return_low_credit_with_doc": [13],
    "user_return_low_credit_no_doc": [14],
    "unreasonable_refund": [15],
}


def _classification_by_name(path_config):
    fields = list(get_scenario_config("ecommerce_refund").classification_fields)
    return dict(zip(fields, path_config["Classification_items"]))


def test_every_path_has_one_semantically_compatible_customer_intent():
    paths = generate_path_list()
    mapping = get_intent_path_mapping()

    assert {
        name: sorted(item["possible_paths"])
        for name, item in mapping.items()
    } == EXPECTED_PATH_INTENTS

    mapped_ids = [
        path_id
        for item in mapping.values()
        for path_id in item["possible_paths"]
    ]
    assert len(mapped_ids) == len(set(mapped_ids))
    assert set(mapped_ids) == set(range(1, len(paths) + 1))

    for intent, item in mapping.items():
        for path_id in item["possible_paths"]:
            config = paths[path_id - 1]
            values = _classification_by_name(config)
            values.update(config.get("system_variables", {}))
            for field, expected in item.get("required_conditions", {}).items():
                assert values.get(field) == expected, (path_id, intent, field)
            for field, forbidden in item.get("impossible_conditions", {}).items():
                assert values.get(field) != forbidden, (path_id, intent, field)


def test_formal_seed7_case_selection_uses_matching_intent_and_case_goal():
    splits = SplitManager(
        SplitConfig(seed=7, max_cases=10, instances_per_path=1)
    ).build()
    cases = splits.all_cases
    path_to_intent = {
        path_id: intent
        for intent, item in get_intent_path_mapping().items()
        for path_id in item["possible_paths"]
    }

    assert len(cases) == 10
    assert {case.path_id for case in cases} == {2, 4, 5, 8, 9, 10, 12, 13, 14, 15}
    for case in cases:
        assert case.intent == path_to_intent[case.path_id]
        expected_goal_type = (
            "refund" if "refund" in case.intent else case.intent
        )
        assert case.case_spec["user_goal"]["type"] == expected_goal_type
        assert case.case_spec["user_goal"]["desired_action"] == (
            case.path_config["final_output"]["Action"]
        )
        declared_emotion = case.path_config["Classification_items"][4]
        expected_emotion = (
            declared_emotion
            if declared_emotion in {"Calm", "Dissatisfied"}
            else None
        )
        assert case.case_spec["user_policy"].get("initial_emotion") == expected_emotion


def test_rule_engine_gold_matches_all_fifteen_pathlist_cases():
    engine = EcommerceRefundSOPRuleEngine()
    for path_id, config in enumerate(generate_path_list(), start=1):
        classification = _classification_by_name(config)
        if "step7" in config["expected_path"]:
            assert classification["EmotionStatus"] in {"Calm", "Dissatisfied"}, path_id
        result = engine.compute_correct_path_and_finals(
            classification,
            {"system_info": config.get("system_variables", {})},
        )
        assert result.now_path == config["expected_path"], path_id
        assert result.finals == config["final_output"], path_id


def test_explicit_dissatisfied_label_preserves_path_action_and_backend_goal():
    paths = generate_path_list()
    fields = list(get_scenario_config("ecommerce_refund").classification_fields)
    engine = EcommerceRefundSOPRuleEngine()

    for path_id in (9, 10):
        after = deepcopy(paths[path_id - 1])
        before = deepcopy(after)
        before["Classification_items"][fields.index("EmotionStatus")] = None

        def outcome(config):
            result = engine.compute_correct_path_and_finals(
                _classification_by_name(config),
                {"system_info": config["system_variables"]},
            )
            case = build_case_spec(
                "ecommerce_refund",
                "merchant_compensation_low_credit",
                config,
                user_id=f"explicit-emotion-{path_id}",
            )
            return result.now_path, result.finals, case.expected_outcome

        assert outcome(before) == outcome(after)
        assert after["Classification_items"][fields.index("EmotionStatus")] == "Dissatisfied"


def test_customer_uses_only_case_declared_emotion_not_adversarial_intensity():
    paths = generate_path_list()
    mapping = get_intent_path_mapping()
    path_to_intent = {
        path_id: intent
        for intent, item in mapping.items()
        for path_id in item["possible_paths"]
    }

    for path_id in (8, 9, 10, 15):
        intent = path_to_intent[path_id]
        case = build_case_spec(
            "ecommerce_refund", intent, paths[path_id - 1], user_id=f"emotion-{path_id}"
        )
        profile = UserProfile(
            user_id=f"emotion-{path_id}",
            user_intent=intent,
            adversarial_intensity="strong_conflict",
            scenario_id="ecommerce_refund",
        )
        model = LLMUserModel(
            profile,
            get_user_prompt_for_intent(intent, profile.user_id),
            llm_client=None,
            case_spec=case,
        )
        explicit_emotion = path_id in {9, 10}
        assert case.user_policy.get("initial_emotion") == (
            "Dissatisfied" if explicit_emotion else None
        )
        expected_internal_emotion = "angry" if explicit_emotion else "calm"
        assert model.environment_state.emotion == expected_internal_emotion
        initial_prompt = model._build_initial_message_prompt()
        assert ("情感状态: angry" in initial_prompt) is explicit_emotion
        assert "情感状态: calm" not in initial_prompt
        next_prompt = model._build_generation_prompt("请提供订单号", 1)
        assert ("情感状态: angry" in next_prompt) is explicit_emotion
        assert "情感状态: calm" not in next_prompt

def test_explicit_calm_customer_contract_initializes_state_and_prompt():
    case = CaseSpec(
        case_id="CASE-EXPLICIT-CALM",
        scenario="ecommerce_refund",
        backend_record={},
        user_goal={"type": "refund"},
        user_knowledge={},
        user_policy={"initial_emotion": "Calm"},
        initial_observation={},
        expected_outcome={},
    )
    profile = UserProfile(
        user_id="explicit-calm",
        user_intent="refund_before_shipping",
        adversarial_intensity="weak_conflict",
        scenario_id="ecommerce_refund",
    )
    model = LLMUserModel(profile, llm_client=None, case_spec=case)

    assert model.emotion_state.value == "calm"
    assert model.environment_state.emotion == "calm"
    assert "情感状态: calm" in model._build_initial_message_prompt()
    assert "情感状态: calm" in model._build_generation_prompt("请提供订单号", 1)


def test_agent_request_contains_the_correct_step5_route():
    class CapturingClient:
        messages = None

        def generate(self, **kwargs):
            self.messages = kwargs["messages"]
            return SimpleNamespace(
                text=json.dumps({
                    "classification_output": {},
                    "now_path": ["step1", "step2", "step3", "step5", "step4"],
                    "finals": {"Action": "CollectionService"},
                    "chat": "我已核验相关信息。",
                }, ensure_ascii=False),
                model="fake",
                tool_calls=[],
            )

    client = CapturingClient()
    agent = AgentModel(
        "ecommerce_refund",
        get_sop_graph("ecommerce_refund"),
        system_prompt=AGENT_SYSTEM_PROMPT,
        llm_client=client,
        use_llm_for_full_output=True,
    )
    agent.process_turn("我想申请退货。")

    system_prompt = client.messages[0]["content"]
    assert "User 和 Merchant 均继续到 step4" in system_prompt
    assert "User→step6" not in system_prompt


def test_customer_prompt_uses_corrected_intent_without_hidden_case_data():
    paths = generate_path_list()
    mapping = get_intent_path_mapping()
    path_to_intent = {
        path_id: intent
        for intent, item in mapping.items()
        for path_id in item["possible_paths"]
    }
    config = paths[12]  # Path 13: user return, low credit, document provided.
    intent = path_to_intent[13]
    case = build_case_spec("ecommerce_refund", intent, config, user_id="prompt-audit")
    case.backend_record["audit_secret"] = "BACKEND_PRIVATE_SENTINEL"
    case.metadata["audit_secret"] = "GOLD_PATH_PRIVATE_SENTINEL"

    profile = UserProfile(
        user_id="prompt-audit",
        user_intent=intent,
        adversarial_intensity="weak_conflict",
        scenario_id="ecommerce_refund",
    )
    model = LLMUserModel(
        profile,
        get_user_prompt_for_intent(intent, "prompt-audit"),
        llm_client=None,
        case_spec=case,
    )
    prompt = model._build_initial_message_prompt()

    assert "user_return_low_credit_with_doc" in prompt
    assert "BACKEND_PRIVATE_SENTINEL" not in prompt
    assert "GOLD_PATH_PRIVATE_SENTINEL" not in prompt
    assert "expected_path" not in prompt
