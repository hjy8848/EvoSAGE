import copy
import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from framework import get_sop_graph
from framework.backend import ToolCall, build_case_spec, create_backend
from framework.backend.factory import _stable_id
from framework.core.simulator import DialogueSimulator
from framework.evaluator.evaluator import Evaluator
from framework.models import UserModel, UserProfile
from framework.models.agent_model import AgentModel
from framework.prompts import ecommerce_refund_prompts
import run_evaluation_with_llm as runner_module


class ScriptedClient:
    """Deterministic OpenAI-compatible client for the environment E2E tests."""

    def __init__(self, steps):
        self.steps = list(steps)
        self.requests = []

    def generate(self, prompt, **kwargs):
        self.requests.append({
            "messages": copy.deepcopy(kwargs.get("messages", [])),
            "tools": copy.deepcopy(kwargs.get("tools", [])),
        })
        step = self.steps.pop(0)
        if callable(step):
            return step(self.requests[-1])
        return step


def response_with_tools(*calls):
    return SimpleNamespace(
        text="",
        model="fake",
        tool_calls=list(calls),
        metadata={},
    )


def response_with_json(action, path=None, chat="已根据后台结果完成处理。"):
    return SimpleNamespace(
        text=json.dumps({
            "classification_output": {
                "CoreIntention": "ReturnOrRefund",
                "ProvidedDocument": True,
                "Responsibility": "User",
                "RefundReasonable": "Reasonable",
                "EmotionStatus": "Calm",
            },
            "now_path": path or ["step1", "step2", "step3"],
            "finals": {"Action": action},
            "chat": chat,
        }, ensure_ascii=False),
        model="fake",
        tool_calls=[],
        metadata={},
    )


def make_case(shipping_status="Signed", credit_level="Low", expected_action="CollectionService"):
    path = {
        "Classification_items": ["ReturnOrRefund", True, "User", "Reasonable", "Calm"],
        "system_variables": {
            "ShippingStatus": shipping_status,
            "CreditLevel": credit_level,
        },
        "expected_path": ["step1", "step2", "step3"],
        "final_output": {"Action": expected_action},
    }
    case = build_case_spec(
        "ecommerce_refund", "refund_request", path, user_id="e2e-user"
    )
    return case


def make_agent(client):
    return AgentModel(
        "ecommerce_refund",
        get_sop_graph("ecommerce_refund"),
        system_prompt=ecommerce_refund_prompts.AGENT_SYSTEM_PROMPT,
        llm_client=client,
        use_llm_for_full_output=True,
    )


class EcommerceEnvironmentE2ETests(unittest.TestCase):
    def test_runner_main_entry_wires_case_backend_and_legacy_gt(self):
        path_config = {
            "Classification_items": ["ReturnOrRefund", True, "User", "Reasonable", "Calm"],
            "system_variables": {"ShippingStatus": "Signed", "CreditLevel": "Low"},
            "expected_path": ["step1", "step2", "step3"],
            "final_output": {"Action": "TransHuman"},
        }
        captured = {}
        case_id = _stable_id("CASE", "ecommerce_refund:refund_before_shipping:runner-test")
        order_id = _stable_id("ORD", case_id)
        client = ScriptedClient([
            response_with_tools(ToolCall("q", "query_order", {"order_id": order_id})),
            response_with_tools(ToolCall("a", "transfer_human", {"order_id": order_id})),
            response_with_json("TransHuman", path_config["expected_path"], "已为您转接人工客服。"),
        ])

        class TestPipeline(runner_module.LLMEvaluationPipeline):
            def _init_llm_clients(self, *args, **kwargs):
                self.user_llm_client = client
                self.agent_llm_client = client
                self.judge_llm_client = None

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.judge_model = SimpleNamespace(
                    evaluate_turn_comprehensive=lambda **_kwargs: {
                        "classification": {
                            "CoreIntention": "ReturnOrRefund",
                            "ProvidedDocument": True,
                            "Responsibility": "User",
                            "RefundReasonable": "Reasonable",
                            "EmotionStatus": "Calm",
                        },
                        "chat_quality_score": 0.8,
                        "chat_quality_dimensions": {},
                    },
                    evaluate_chat_quality=lambda **_kwargs: (0.8, {}),
                )

        original_simulator = runner_module.DialogueSimulator

        class CapturingSimulator(original_simulator):
            def __init__(self, *args, **kwargs):
                captured.update(kwargs)
                super().__init__(*args, **kwargs)

            def run(self, *args, **kwargs):
                captured["run_context_data"] = kwargs.get("context_data", {})
                return super().run(*args, **kwargs)

        with tempfile.TemporaryDirectory() as output_dir, patch.object(
            runner_module, "DialogueSimulator", CapturingSimulator
        ):
            pipeline = TestPipeline(
                scenario_id="ecommerce_refund",
                model_name="fake",
                output_dir=output_dir,
                max_turns=1,
                verbose=False,
                user_simulator_mode="rule",
            )
            result, report = pipeline.run_single_simulation(
                "refund_before_shipping", user_id="runner-test", path_config=path_config
            )

        self.assertIsNotNone(captured["backend_environment"])
        self.assertIsNotNone(captured["case_spec"])
        self.assertIs(captured["backend_environment"].case_spec, captured["case_spec"])
        self.assertNotIn("system_info", captured["run_context_data"])
        self.assertNotIn("expected_outcome", captured["run_context_data"])
        self.assertEqual(result.case_spec["metadata"]["legacy_gt"]["finals"]["Action"], "TransHuman")
        self.assertTrue(result.case_spec["metadata"]["legacy_gt"]["classification"])
        self.assertTrue(result.case_spec["metadata"]["legacy_gt"]["expected_path"])
        first_messages = json.dumps(client.requests[0]["messages"], ensure_ascii=False)
        self.assertNotIn("Signed", first_messages)
        self.assertNotIn("Low", first_messages)
        self.assertTrue(any(message.get("role") == "tool" for message in client.requests[1]["messages"]))
        self.assertIn("Signed", json.dumps(client.requests[1]["messages"], ensure_ascii=False))
        self.assertEqual([event["name"] for event in result.backend_events], [
            "query_order", "transfer_human", "TransHuman"
        ])
        self.assertEqual(result.backend_final_state["order"]["human_transfer_status"], "Requested")
        self.assertGreater(report.legacy_score, 0.0)
        self.assertGreater(report.environment_score, 0.0)
        self.assertEqual(report.environment_goal_fulfillment, 1.0)

    def test_hidden_backend_state_is_not_in_initial_agent_messages(self):
        case = make_case("Signed", "Low")
        client = ScriptedClient([response_with_json("Refund")])
        agent = make_agent(client)
        agent.process_turn(
            "订单 ORD-1 还没发货，我要退款。",
            context_data={"initial_observation": {}},
            backend_environment=create_backend(case),
        )
        first_request = json.dumps(client.requests[0]["messages"], ensure_ascii=False)
        self.assertNotIn("Signed", first_request)
        self.assertNotIn("Low", first_request)
        self.assertNotIn("backend_record", first_request)

    def test_current_user_message_is_not_injected_twice(self):
        client = ScriptedClient([response_with_json("Refund")])
        message = "我要退款，请帮我查一下订单。"
        make_agent(client).process_turn(
            message,
            context_data={"initial_observation": {}},
            backend_environment=create_backend(make_case()),
        )
        user_messages = [
            item.get("content", "")
            for item in client.requests[0]["messages"]
            if item.get("role") == "user"
        ]
        assert sum(content.count(message) for content in user_messages) == 1

    def test_same_user_message_has_same_initial_agent_input_for_two_backends(self):
        case_a = make_case("Signed", "High")
        case_b = make_case("Signed", "Low")
        input_text = "商品有问题，我想退款。"
        clients = []
        for case in (case_a, case_b):
            client = ScriptedClient([response_with_json("Refund")])
            clients.append(client)
            make_agent(client).process_turn(
                input_text,
                context_data={"initial_observation": {}},
                backend_environment=create_backend(case),
            )
        self.assertEqual(
            clients[0].requests[0]["messages"], clients[1].requests[0]["messages"]
        )

    def test_mistaken_user_claim_requires_backend_query(self):
        case = make_case("Signed", "Low")
        case.user_knowledge["believes_shipping_status"] = "Unshipped"
        client = ScriptedClient([response_with_json("Refund", chat="已经为您退款。")])
        agent = make_agent(client)
        profile = UserProfile("u", "refund_request", "strong_conflict", "ecommerce_refund")
        user = UserModel(profile)
        simulator = DialogueSimulator(
            user, agent, max_turns=1, backend_environment=create_backend(case), case_spec=case
        )
        result = simulator.run("订单还没发货，我要退款。", {"initial_observation": {}})
        report = Evaluator("ecommerce_refund", get_sop_graph("ecommerce_refund")).evaluate_simulation(result)
        self.assertIn("missed_backend_verification", report.error_categories)
        self.assertIn("user_claim_overtrusted", report.error_categories)

    def test_query_result_misinterpretation_is_distinguished_from_missed_query(self):
        case = make_case("Signed", "Low")
        order_id = case.user_knowledge["order_id"]
        customer_id = case.user_knowledge["customer_id"]
        client = ScriptedClient([
            response_with_tools(ToolCall("q", "query_order", {"order_id": order_id})),
            response_with_tools(ToolCall("c", "query_customer_profile", {"customer_id": customer_id})),
            response_with_tools(ToolCall("a", "submit_refund", {"order_id": order_id})),
            response_with_json("Refund", chat="已经为您退款。"),
        ])
        profile = UserProfile("u", "refund_request", "strong_conflict", "ecommerce_refund")
        result = DialogueSimulator(
            user_model=UserModel(profile),
            agent_model=make_agent(client),
            max_turns=1,
            backend_environment=create_backend(case),
            case_spec=case,
        ).run("订单还没发货，我要退款。", {"initial_observation": {}})
        report = Evaluator("ecommerce_refund", get_sop_graph("ecommerce_refund")).evaluate_simulation(result)
        self.assertIn("tool_result_misinterpretation", report.error_categories)
        self.assertIn("action_execution_failure", report.error_categories)
        self.assertNotIn("missed_backend_verification", report.error_categories)

    def test_credit_level_requires_customer_profile_verification(self):
        case = make_case("Signed", "Low", "CollectionService")
        order_id = case.user_knowledge["order_id"]
        client = ScriptedClient([
            response_with_tools(ToolCall("q", "query_order", {"order_id": order_id})),
            response_with_tools(ToolCall("a", "schedule_pickup", {"order_id": order_id})),
            response_with_json("CollectionService", chat="已为您安排上门取件。"),
        ])
        profile = UserProfile("u", "refund_request", "strong_conflict", "ecommerce_refund")
        result = DialogueSimulator(
            UserModel(profile), make_agent(client), max_turns=1,
            backend_environment=create_backend(case), case_spec=case,
        ).run("我要退货。", {"initial_observation": {}})
        report = Evaluator("ecommerce_refund", get_sop_graph("ecommerce_refund")).evaluate_simulation(result)
        self.assertEqual(result.backend_final_state["order"]["return_status"], "PickupScheduled")
        self.assertEqual(report.environment_goal_fulfillment, 1.0)
        self.assertLess(report.environment_score, 1.0)
        self.assertIn("missed_backend_verification", report.error_categories)
        self.assertEqual(
            report.details["environment_metrics"]["required_verification_coverage"],
            0.5,
        )

    def test_query_order_does_not_expose_gt_derived_eligibility(self):
        case = make_case("Unshipped", "High", "Refund")
        order_id = case.user_knowledge["order_id"]
        result = create_backend(case).execute_tool(
            "query_order", {"order_id": order_id}, turn_index=0
        )
        self.assertTrue(result.success)
        self.assertNotIn("refund_eligible", result.data)

    def test_pretending_refund_success_does_not_change_backend_goal(self):
        case = make_case("Unshipped", "High", "Refund")
        client = ScriptedClient([response_with_json("Refund", chat="已经为您退款。")])
        profile = UserProfile("u", "refund_request", "strong_conflict", "ecommerce_refund")
        result = DialogueSimulator(
            UserModel(profile), make_agent(client), max_turns=1,
            backend_environment=create_backend(case), case_spec=case,
        ).run("我要退款。", {"initial_observation": {}})
        report = Evaluator("ecommerce_refund", get_sop_graph("ecommerce_refund")).evaluate_simulation(result)
        self.assertEqual(result.backend_final_state["order"]["refund_status"], "None")
        self.assertEqual(report.environment_goal_fulfillment, 0.0)
        self.assertIn("claimed_action_not_executed", report.error_categories)

    def test_real_refund_success_is_a_backend_state_transition(self):
        case = make_case("Unshipped", "High", "Refund")
        order_id = case.user_knowledge["order_id"]
        client = ScriptedClient([
            response_with_tools(ToolCall("q", "query_order", {"order_id": order_id})),
            response_with_tools(ToolCall("p", "query_payment", {"order_id": order_id})),
            response_with_tools(ToolCall("a", "submit_refund", {"order_id": order_id})),
            response_with_json("Refund", ["step1", "step2", "step3"], "退款已提交。"),
        ])
        profile = UserProfile("u", "refund_request", "strong_conflict", "ecommerce_refund")
        result = DialogueSimulator(
            UserModel(profile), make_agent(client), max_turns=1,
            backend_environment=create_backend(case), case_spec=case,
        ).run("我要退款。", {"initial_observation": {}})
        report = Evaluator("ecommerce_refund", get_sop_graph("ecommerce_refund")).evaluate_simulation(result)
        action_events = [e for e in result.backend_events if e["event_type"] == "action_execution"]
        self.assertEqual(result.executed_action, "Refund")
        self.assertEqual(result.termination_reason, "goal_fulfilled")
        self.assertEqual(action_events[-1]["result"]["action_name"], "Refund")
        self.assertEqual(result.backend_final_state["order"]["refund_status"], "Approved")
        self.assertEqual(report.environment_goal_fulfillment, 1.0)

    def test_evaluator_does_not_trust_model_executed_path(self):
        case = make_case("Unshipped", "High", "Refund")
        client = ScriptedClient([
            response_with_json("Refund", ["refund_success"], "已经为您退款。")
        ])
        profile = UserProfile("u", "refund_request", "strong_conflict", "ecommerce_refund")
        result = DialogueSimulator(
            UserModel(profile), make_agent(client), max_turns=1,
            backend_environment=create_backend(case), case_spec=case,
        ).run("我要退款。", {"initial_observation": {}})
        self.assertEqual(result.predicted_path, ["refund_success"])
        self.assertEqual(result.executed_path, [])
        self.assertNotEqual(result.predicted_path, result.executed_path)

    def test_full_trace_uses_one_backend_instance(self):
        case = make_case("Unshipped", "High", "Refund")
        order_id = case.user_knowledge["order_id"]
        backend = create_backend(case)
        backend.reset(case)
        client = ScriptedClient([
            response_with_tools(ToolCall("q", "query_order", {"order_id": order_id})),
            response_with_tools(ToolCall("a", "submit_refund", {"order_id": order_id})),
            response_with_json("Refund", chat="退款已提交。"),
        ])
        profile = UserProfile("u", "refund_request", "strong_conflict", "ecommerce_refund")
        result = DialogueSimulator(
            UserModel(profile), make_agent(client), max_turns=1,
            backend_environment=backend, case_spec=case,
        ).run("订单还没发货，我要退款。", {"initial_observation": {}})
        self.assertIsNotNone(result.backend_final_state)
        self.assertEqual(result.backend_final_state, backend.get_state_snapshot())
        self.assertEqual(result.backend_events, backend.get_event_log())
        self.assertEqual(result.turns[0].backend_state_before["order"]["refund_status"], "None")
        self.assertEqual(result.turns[0].backend_state_after["order"]["refund_status"], "Approved")

    def test_json_parse_failure_preserves_raw_response_and_error(self):
        case = make_case("Unshipped", "High", "Refund")
        client = ScriptedClient([
            SimpleNamespace(
                text="{this is not valid json",
                model="fake",
                tool_calls=[],
                metadata={
                    "finish_reason": "stop",
                    "assistant_message": {
                        "role": "assistant",
                        "content": "{this is not valid json",
                    },
                },
            )
        ])
        output = make_agent(client).process_turn(
            "我要退款。",
            context_data={"initial_observation": {}},
            backend_environment=create_backend(case),
        )
        self.assertTrue(output.json_parse_failed)
        self.assertEqual(output.metadata["raw_llm_response"], "{this is not valid json")
        self.assertEqual(
            output.metadata["raw_final_assistant_message"]["content"],
            "{this is not valid json",
        )
        self.assertEqual(output.metadata["parse_error"]["type"], "ValueError")
        self.assertIn("does not contain JSON", output.metadata["parse_error"]["message"])


if __name__ == "__main__":
    unittest.main()
