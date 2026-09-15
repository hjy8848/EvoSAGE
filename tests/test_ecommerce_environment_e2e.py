import copy
import json
import unittest
from types import SimpleNamespace

from framework import get_sop_graph
from framework.backend import ToolCall, build_case_spec, create_backend
from framework.core.simulator import DialogueSimulator
from framework.evaluator.evaluator import Evaluator
from framework.models import UserModel, UserProfile
from framework.models.agent_model import AgentModel
from framework.prompts import ecommerce_refund_prompts


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
    case.metadata.update({
        "classification_dict": {
            "CoreIntention": "ReturnOrRefund",
            "ProvidedDocument": True,
            "Responsibility": "User",
            "RefundReasonable": "Reasonable",
            "EmotionStatus": "Calm",
        },
        "expected_path": path["expected_path"],
        "finals": path["final_output"],
    })
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
        client = ScriptedClient([
            response_with_tools(ToolCall("q", "query_order", {"order_id": order_id})),
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


if __name__ == "__main__":
    unittest.main()
