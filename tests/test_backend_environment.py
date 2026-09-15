import unittest
import json
from types import SimpleNamespace

from framework.backend import EcommerceBackend, build_case_spec, create_backend
from framework.models.agent_model import AgentModel
from framework import get_sop_graph


class EcommerceBackendTests(unittest.TestCase):
    def _case(self):
        return build_case_spec(
            "ecommerce_refund",
            "refund_request",
            {
                "Classification_items": ["ReturnOrRefund", True, "User", "Reasonable", "Calm"],
                "system_variables": {"ShippingStatus": "Unshipped", "CreditLevel": "High"},
                "expected_path": ["step1", "step2"],
                "final_output": {"Action": "Refund", "PLAN": "none"},
            },
            user_id="test-user",
        )

    def test_query_returns_authoritative_safe_fields(self):
        backend = EcommerceBackend(self._case())
        order_id = backend.case_spec.user_knowledge["order_id"]
        result = backend.execute_tool("query_order", {"order_id": order_id}, turn_index=0)

        self.assertTrue(result.success)
        self.assertEqual(result.data["shipping_status"], "Unshipped")
        self.assertEqual(result.data["responsibility"], "User")
        self.assertEqual(backend.get_event_log()[0]["event_type"], "tool_call")

    def test_action_requires_verification_and_mutates_state(self):
        backend = EcommerceBackend(self._case())
        blocked = backend.execute_action("Refund", turn_index=0)
        self.assertFalse(blocked.success)
        self.assertEqual(blocked.error_code, "order_not_verified")

        order_id = backend.case_spec.user_knowledge["order_id"]
        backend.execute_tool("query_order", {"order_id": order_id}, turn_index=1)
        applied = backend.execute_action("Refund", turn_index=1)

        self.assertTrue(applied.success)
        self.assertTrue(backend.goal_satisfied())
        self.assertEqual(backend.state["order"]["refund_status"], "Approved")

    def test_agent_tool_loop_feeds_result_back_before_final_json(self):
        case = self._case()
        backend = EcommerceBackend(case)
        order_id = case.user_knowledge["order_id"]

        class FakeClient:
            def __init__(self):
                self.calls = 0

            def generate(self, prompt, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    from framework.backend import ToolCall
                    call = ToolCall("call-1", "query_order", {"order_id": order_id})
                    return SimpleNamespace(
                        text="",
                        model="fake",
                        tool_calls=[call],
                        metadata={
                            "assistant_message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [{
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "query_order",
                                        "arguments": json.dumps({"order_id": order_id}),
                                    },
                                }],
                            }
                        },
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
                        "now_path": ["step1"],
                        "finals": {"Action": "Refund", "PLAN": "none"},
                        "chat": "已核验订单，可以为您提交退款。",
                    }, ensure_ascii=False),
                    model="fake",
                )

        agent = AgentModel(
            "ecommerce_refund",
            get_sop_graph("ecommerce_refund"),
            system_prompt="return JSON",
            llm_client=FakeClient(),
            use_llm_for_full_output=True,
        )
        output = agent.process_turn("我要退款", backend_environment=backend)

        self.assertEqual(len(output.tool_calls), 1)
        self.assertEqual(output.tool_results[0]["data"]["order_id"], order_id)
        self.assertEqual(output.action, "Refund")
        self.assertEqual(backend.state["order"]["refund_status"], "None")

    def test_formal_action_tool_changes_state_once(self):
        case = self._case()
        backend = EcommerceBackend(case)
        order_id = case.user_knowledge["order_id"]

        class ActionClient:
            def __init__(self):
                self.calls = 0

            def generate(self, prompt, **kwargs):
                self.calls += 1
                from framework.backend import ToolCall
                if self.calls == 1:
                    return SimpleNamespace(
                        text="",
                        tool_calls=[ToolCall("q", "query_order", {"order_id": order_id})],
                        metadata={},
                    )
                if self.calls == 2:
                    return SimpleNamespace(
                        text="",
                        tool_calls=[ToolCall("a", "submit_refund", {"order_id": order_id})],
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
                        "finals": {"Action": "Refund"},
                        "chat": "退款已提交。",
                    }, ensure_ascii=False),
                    tool_calls=[],
                    metadata={},
                )

        agent = AgentModel(
            "ecommerce_refund", get_sop_graph("ecommerce_refund"),
            system_prompt="return JSON", llm_client=ActionClient(),
            use_llm_for_full_output=True,
        )
        output = agent.process_turn("我要退款", backend_environment=backend)
        action_events = [event for event in backend.get_event_log() if event["event_type"] == "action_execution"]
        self.assertEqual(len(action_events), 1)
        self.assertTrue(backend.goal_satisfied())
        self.assertEqual(output.action, "Refund")

if __name__ == "__main__":
    unittest.main()
