import unittest

from framework import get_sop_graph
from framework.core.simulator import SimulationResult, SimulationTurn
from framework.evaluator.evaluator import Evaluator
from framework.models.agent_model import (
    AgentTurnOutput,
    ClassificationOutput,
    FinalOutput,
)


class EvaluationRegressionTests(unittest.TestCase):
    def _make_simulation(self, chat):
        classification = ClassificationOutput(
            DescriptionClear=True,
            QuestionRelevance=True,
            EmotionTendency="Calm",
            ResolveDependency="MediumDependency",
            RepeatedRaised=False,
            RegardingRefund=False,
        )
        expected_path = [
            "step1", "step2", "step3", "step4",
            "step5", "step6", "step7", "action_plan",
        ]
        output = AgentTurnOutput(
            turn_id=0,
            classification_output=classification,
            # The model may declare a different path; it must not be used as
            # the executed path for scoring.
            expected_path=["wrong", "path"],
            final_output=FinalOutput(Action="PLAN", PLAN="PLAN_B"),
            action="PLAN",
            chat=chat,
            path_taken=expected_path,
        )
        case = {
            "classification": classification.to_dict(),
            "now_path": expected_path,
            "finals": {"Action": "PLAN", "PLAN": "PLAN_B"},
        }
        return SimulationResult(
            simulation_id="regression",
            scenario_id="online_education",
            model_name="test",
            user_intent="seek_answer",
            adversarial_intensity="weak_conflict",
            context_data={"benchmark_case": case},
            turns=[SimulationTurn(0, "如何设置参数？", output)],
        )

    def test_path_score_uses_executed_path_and_action_schema_is_stable(self):
        simulation = self._make_simulation(
            "默认参数会在调用时省略该参数时生效。例如 def f(x=1): return x。"
        )
        report = Evaluator(
            "online_education", get_sop_graph("online_education")
        ).evaluate_simulation(simulation)
        metrics = {metric.metric_name: metric.score for metric in report.metric_scores}
        self.assertEqual(metrics["path_correctness"], 1.0)
        self.assertEqual(metrics["action_correctness"], 1.0)
        self.assertIn("goal_fulfillment", metrics)

    def test_promise_only_reply_does_not_count_as_goal_fulfillment(self):
        simulation = self._make_simulation(
            "好的，我来为您详细讲解Python函数参数默认值的实际应用场景。"
        )
        report = Evaluator(
            "online_education", get_sop_graph("online_education")
        ).evaluate_simulation(simulation)
        metrics = {metric.metric_name: metric.score for metric in report.metric_scores}
        self.assertEqual(metrics["goal_fulfillment"], 0.0)


if __name__ == "__main__":
    unittest.main()
