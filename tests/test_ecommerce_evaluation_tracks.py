import unittest

from framework import get_sop_graph
from framework.core.simulator import SimulationResult, SimulationTurn
from framework.evaluator.evaluator import Evaluator
from framework.models.agent_model import (
    AgentTurnOutput,
    EcommerceRefundClassification,
    FinalOutput,
)


class EcommerceEvaluationTrackTests(unittest.TestCase):
    """Regression tests for decision, execution and strict-success semantics."""

    GOLD_PATH = ["step1", "step2", "step3"]

    def _tool(self, name, arguments, data=None, success=True):
        return {
            "event_type": "tool_call",
            "name": name,
            "arguments": arguments,
            "result": {
                "success": success,
                "tool_name": name,
                "data": data or {},
            },
        }

    def _action(self, name="Refund", success=True):
        return {
            "event_type": "action_execution",
            "name": name,
            "arguments": {"order_id": "order-1"},
            "result": {
                "success": success,
                "action_name": name,
                "data": {},
            },
        }

    def _simulation(
        self,
        system_variables=None,
        events=None,
        final_state=None,
        predicted_path=None,
        predicted_action="Refund",
        chat="已为您处理退款。",
        action_expected="Refund",
    ):
        system_variables = system_variables or {"ShippingStatus": "Unshipped"}
        expected_outcome = {
            "order.last_action": action_expected,
            "order.refund_status": "Approved",
        }
        classification = EcommerceRefundClassification(
            CoreIntention="ReturnOrRefund",
            ProvidedDocument=False,
            Responsibility="Merchant",
            RefundReasonable="Reasonable",
            EmotionStatus="Calm",
        )
        output = AgentTurnOutput(
            turn_id=0,
            classification_output=classification,
            expected_path=list(predicted_path or self.GOLD_PATH),
            final_output=FinalOutput(Action=predicted_action),
            action=predicted_action,
            chat=chat,
        )
        case = {
            "case_id": "case-1",
            "scenario": "ecommerce_refund",
            "backend_record": {},
            "user_goal": {"type": "refund"},
            "user_knowledge": {"order_id": "order-1", "customer_id": "customer-1"},
            "user_policy": {},
            "initial_observation": {},
            "expected_outcome": expected_outcome,
            "metadata": {
                "classification_dict": classification.to_dict(),
                "expected_path": self.GOLD_PATH,
                "finals": {"Action": action_expected},
                "path_config": {"system_variables": system_variables},
            },
        }
        return SimulationResult(
            simulation_id="eval-test",
            scenario_id="ecommerce_refund",
            model_name="test",
            user_intent="refund_request",
            adversarial_intensity="zero_conflict",
            turns=[SimulationTurn(0, "我要退款", output)],
            case_spec=case,
            backend_events=events or [],
            backend_final_state=final_state or {},
        )

    def _evaluate(self, simulation):
        return Evaluator(
            "ecommerce_refund", get_sop_graph("ecommerce_refund")
        ).evaluate_simulation(simulation, eval_turns=[0])

    def _successful_refund_trace(self, include_extra=False):
        events = [self._tool(
            "query_order", {"order_id": "order-1"},
            {"shipping_status": "Unshipped"},
        )]
        if include_extra:
            events.append(self._tool("query_payment", {"order_id": "order-1"}, {"status": "Paid"}))
        events.append(self._action())
        return events

    def test_predicted_path_uses_canonical_declaration_not_executed_trace(self):
        simulation = self._simulation(
            events=self._successful_refund_trace(include_extra=True),
            final_state={"order": {"last_action": "Refund", "refund_status": "Approved"}},
        )
        report = self._evaluate(simulation)
        self.assertEqual(report.canonical_path_correctness, 1.0)
        self.assertTrue(report.task_success)
        self.assertEqual(report.executed_trace, ["query_order", "query_payment", "Refund"])

    def test_predicted_action_without_actual_action_fails_execution(self):
        simulation = self._simulation(
            events=[self._tool("query_order", {"order_id": "order-1"}, {"shipping_status": "Unshipped"})],
        )
        report = self._evaluate(simulation)
        self.assertEqual(report.predicted_action_correctness, 1.0)
        self.assertEqual(report.action_execution_score, 0.0)
        self.assertEqual(report.goal_fulfillment, 0.0)
        self.assertFalse(report.task_success)

    def test_legal_extra_trace_does_not_fail_strict_task_success(self):
        simulation = self._simulation(
            predicted_path=["step1", "step2", "step3", "step4"],
            events=self._successful_refund_trace(include_extra=True),
            final_state={"order": {"last_action": "Refund", "refund_status": "Approved"}},
        )
        report = self._evaluate(simulation)
        self.assertLess(report.canonical_path_correctness, 1.0)
        self.assertEqual(report.action_execution_score, 1.0)
        self.assertTrue(report.task_success)

    def test_missing_credit_verification_blocks_task_success(self):
        simulation = self._simulation(
            system_variables={"ShippingStatus": "Signed", "CreditLevel": "High"},
            events=[
                self._tool("query_order", {"order_id": "order-1"}, {"shipping_status": "Signed"}),
                self._action(),
            ],
            final_state={"order": {"last_action": "Refund", "refund_status": "Approved"}},
        )
        report = self._evaluate(simulation)
        self.assertEqual(report.predicted_action_correctness, 1.0)
        self.assertLess(report.required_verification_score, 1.0)
        self.assertIn("missed_backend_verification", report.error_categories)
        self.assertFalse(report.task_success)

    def test_authoritative_result_cannot_be_overridden_by_user_claim(self):
        simulation = self._simulation(
            events=[
                self._tool("query_order", {"order_id": "order-1"}, {"shipping_status": "Signed"}),
                self._action(),
            ],
            final_state={"order": {"last_action": "Refund", "refund_status": "Approved"}},
        )
        report = self._evaluate(simulation)
        self.assertIn("user_claim_overtrusted", report.error_categories)
        self.assertEqual(report.policy_compliance_score, 0.0)
        self.assertFalse(report.task_success)

    def test_text_claim_without_backend_action_is_not_success(self):
        simulation = self._simulation(
            chat="已经退款成功，请留意到账。",
            events=[self._tool("query_order", {"order_id": "order-1"}, {"shipping_status": "Unshipped"})],
        )
        report = self._evaluate(simulation)
        self.assertIn("claimed_action_not_executed", report.error_categories)
        self.assertEqual(report.action_execution_score, 0.0)
        self.assertEqual(report.goal_fulfillment, 0.0)

    def test_successful_refund_satisfies_action_and_goal(self):
        report = self._evaluate(self._simulation(
            events=self._successful_refund_trace(),
            final_state={"order": {"last_action": "Refund", "refund_status": "Approved"}},
        ))
        self.assertEqual(report.action_execution_score, 1.0)
        self.assertEqual(report.goal_fulfillment, 1.0)
        self.assertTrue(report.task_success)

    def test_successful_wrong_action_is_not_action_execution_success(self):
        report = self._evaluate(self._simulation(
            events=[
                self._tool("query_order", {"order_id": "order-1"}, {"shipping_status": "Unshipped"}),
                self._action("Interception"),
            ],
            final_state={"order": {"last_action": "Interception"}},
        ))
        self.assertEqual(report.action_execution_score, 0.0)
        self.assertEqual(report.policy_compliance_score, 0.0)
        self.assertIn("wrong_final_action", report.error_categories)
        self.assertFalse(report.task_success)

    def test_decision_right_execution_wrong_is_not_task_success(self):
        report = self._evaluate(self._simulation(
            events=[self._tool("query_order", {"order_id": "order-1"}, {"shipping_status": "Unshipped"})],
        ))
        self.assertGreater(report.sage_style_score, report.execution_score)
        self.assertFalse(report.task_success)

    def test_execution_right_but_noncanonical_prediction_keeps_execution_high(self):
        report = self._evaluate(self._simulation(
            predicted_path=["step1", "step2", "step3", "step4"],
            events=self._successful_refund_trace(),
            final_state={"order": {"last_action": "Refund", "refund_status": "Approved"}},
        ))
        self.assertLess(report.canonical_path_correctness, 1.0)
        self.assertEqual(report.execution_score, 1.0)
        self.assertTrue(report.task_success)


if __name__ == "__main__":
    unittest.main()
