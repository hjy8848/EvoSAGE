"""Static closure checks for every canonical PathList case.

This test deliberately uses no LLM.  It verifies that each canonical path can
be represented by a CaseSpec, queried through its declared public tools, and
completed through the formal action tool that the backend exposes.
"""

from run_evaluation_with_llm import get_path_list_by_scenario
from framework.backend import build_case_spec, create_backend


SCENARIOS = (
    "ecommerce_refund",
    "telecom_package",
    "property_service",
    "logistics_delivery",
    "airline_refund",
    "online_education",
)


def _lookup(state, dotted_key):
    value = state
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _path_cases(scenario):
    generate_paths, get_mapping = get_path_list_by_scenario(scenario)
    paths = generate_paths()
    mapping = get_mapping()
    path_to_intent = {
        path_id: intent
        for intent, config in mapping.items()
        for path_id in config.get("possible_paths", [])
    }
    assert set(path_to_intent) == set(range(1, len(paths) + 1))
    for path_id, intent in sorted(path_to_intent.items()):
        config = dict(paths[path_id - 1])
        config["pilot_path_id"] = path_id
        yield path_id, intent, config


def test_all_122_pathlist_cases_have_a_closed_backend_execution_path():
    total = 0

    for scenario in SCENARIOS:
        for path_id, intent, config in _path_cases(scenario):
            total += 1
            case = build_case_spec(
                scenario,
                intent,
                config,
                user_id=f"path-sanity-{scenario}-{path_id}",
            )
            backend = create_backend(case)
            record_id = case.user_knowledge.get("order_id") or case.user_knowledge["record_id"]
            expected_action = config["final_output"]["Action"]

            requirements = case.metadata["required_backend_verifications"]
            assert requirements, (scenario, path_id)
            for requirement in requirements:
                query_identifier = case.user_knowledge[requirement["knowledge_key"]]
                result = backend.execute_tool(
                    requirement["tool"],
                    {requirement["argument"]: query_identifier},
                    turn_index=0,
                )
                assert result.success, (scenario, path_id, requirement, result.to_dict())
                assert requirement["result_field"] in result.data
                assert "private_state" not in result.data
                assert "system_info" not in result.data

            action_tool = next(
                (
                    name
                    for name, canonical in backend.get_action_tool_map().items()
                    if canonical == expected_action
                ),
                None,
            )
            assert action_tool is not None, (scenario, path_id, expected_action)
            action_result = backend.execute_tool(
                action_tool,
                {"order_id": record_id} if scenario == "ecommerce_refund" else {"record_id": record_id},
                turn_index=1,
            )
            assert action_result.success, (
                scenario,
                path_id,
                expected_action,
                action_result.to_dict(),
            )
            assert backend.goal_satisfied(), (scenario, path_id, backend.get_state_snapshot())

            final_state = backend.get_state_snapshot()
            for key, expected_value in case.expected_outcome.items():
                assert _lookup(final_state, key) == expected_value, (
                    scenario,
                    path_id,
                    key,
                    expected_value,
                    final_state,
                )

            action_events = [
                event
                for event in backend.get_event_log()
                if event["event_type"] == "action_execution"
            ]
            assert len(action_events) == 1, (scenario, path_id, action_events)

    assert total == 122
