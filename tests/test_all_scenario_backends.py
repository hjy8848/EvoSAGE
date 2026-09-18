import unittest

from framework.backend import build_case_spec, create_backend
from framework.core.simulator import SimulationResult
from framework.evaluator.evaluator import Evaluator
from run_evaluation_with_llm import get_path_list_by_scenario


MIGRATED_SCENARIOS = (
    "telecom_package",
    "property_service",
    "logistics_delivery",
    "airline_refund",
    "online_education",
)


class AllScenarioBackendTests(unittest.TestCase):
    def _path_cases(self, scenario):
        generate_paths, get_mapping = get_path_list_by_scenario(scenario)
        paths = generate_paths()
        mapping = get_mapping()
        path_to_intent = {
            path_id: intent
            for intent, config in mapping.items()
            for path_id in config.get("possible_paths", [])
        }
        self.assertEqual(set(path_to_intent), set(range(1, len(paths) + 1)))
        for path_id, intent in sorted(path_to_intent.items()):
            config = dict(paths[path_id - 1])
            config["pilot_path_id"] = path_id
            yield path_id, intent, config

    def test_every_path_has_public_queries_and_stateful_action(self):
        for scenario in MIGRATED_SCENARIOS:
            for path_id, intent, config in self._path_cases(scenario):
                with self.subTest(scenario=scenario, path_id=path_id):
                    case = build_case_spec(
                        scenario,
                        intent,
                        config,
                        user_id=f"backend-test-{scenario}-{path_id}",
                    )
                    backend = create_backend(case)
                    record_id = case.user_knowledge["record_id"]
                    action = config["final_output"]["Action"]

                    self.assertTrue(backend.get_tool_definitions())
                    self.assertNotIn("system_info", case.backend_record)
                    self.assertIn("private_state", case.backend_record)

                    invalid = backend.execute_tool(
                        backend.get_tool_definitions()[0]["function"]["name"],
                        {"record_id": "invalid-record"},
                        turn_index=0,
                    )
                    self.assertFalse(invalid.success)

                    action_tool = next(
                        name for name, canonical in backend.get_action_tool_map().items()
                        if canonical == action
                    )
                    before_action = create_backend(case)
                    failed_action = before_action.execute_tool(
                        action_tool, {"record_id": record_id}, turn_index=0
                    )
                    self.assertFalse(failed_action.success)
                    self.assertEqual(failed_action.error_code, "record_not_verified")

                    for requirement in case.metadata["required_backend_verifications"]:
                        query = backend.execute_tool(
                            requirement["tool"],
                            {requirement["argument"]: record_id},
                            turn_index=0,
                        )
                        self.assertTrue(query.success)
                        self.assertIn(requirement["result_field"], query.data)
                        self.assertNotIn("private_state", query.data)

                    result = backend.execute_tool(
                        action_tool, {"record_id": record_id}, turn_index=1
                    )
                    self.assertTrue(result.success)
                    self.assertEqual(result.data["last_action"], action)
                    self.assertTrue(backend.goal_satisfied())

    def test_public_projection_does_not_expose_private_backend_state(self):
        for scenario in MIGRATED_SCENARIOS:
            path_id, intent, config = next(self._path_cases(scenario))
            case = build_case_spec(scenario, intent, config, user_id=f"projection-{scenario}")
            backend = create_backend(case)
            query_name = backend.get_tool_definitions()[0]["function"]["name"]
            result = backend.execute_tool(
                query_name,
                {"record_id": case.user_knowledge["record_id"]},
            )
            self.assertTrue(result.success)
            self.assertNotIn("private_state", result.data)
            self.assertNotIn("backend_record", result.data)

    def test_execution_track_uses_backend_events_for_all_scenarios(self):
        for scenario in MIGRATED_SCENARIOS:
            path_id, intent, config = next(self._path_cases(scenario))
            case = build_case_spec(
                scenario, intent, config, user_id=f"evaluation-{scenario}"
            )
            backend = create_backend(case)
            record_id = case.user_knowledge["record_id"]
            for requirement in case.metadata["required_backend_verifications"]:
                backend.execute_tool(
                    requirement["tool"],
                    {requirement["argument"]: record_id},
                    turn_index=0,
                )
            expected_action = config["final_output"]["Action"]
            action_tool = next(
                name for name, canonical in backend.get_action_tool_map().items()
                if canonical == expected_action
            )
            backend.execute_tool(action_tool, {"record_id": record_id}, turn_index=1)
            simulation = SimulationResult(
                simulation_id=f"eval-{scenario}",
                scenario_id=scenario,
                model_name="test",
                user_intent=intent,
                adversarial_intensity="zero_conflict",
                case_spec=case.to_dict(),
                backend_events=backend.get_event_log(),
                backend_final_state=backend.get_state_snapshot(),
                goal_solved=backend.goal_satisfied(),
            )
            assessment = Evaluator._execution_assessment(simulation)
            with self.subTest(scenario=scenario, path_id=path_id):
                self.assertEqual(assessment["verification"], 1.0)
                self.assertEqual(assessment["policy"], 1.0)
                self.assertEqual(assessment["action_execution"], 1.0)
                self.assertEqual(assessment["goal_fulfillment"], 1.0)
                self.assertEqual(assessment["errors"], [])


if __name__ == "__main__":
    unittest.main()
