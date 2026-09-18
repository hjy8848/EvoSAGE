"""End-to-end smoke tests for the five non-ecommerce backend adapters."""

import copy
import json
from types import SimpleNamespace

from framework import get_sop_graph
from framework.backend import ToolCall, build_case_spec, create_backend
from framework.config import get_scenario_config
from framework.core.simulator import DialogueSimulator
from framework.evaluator.evaluator import Evaluator
from framework.models import UserModel, UserProfile
from framework.models.agent_model import AgentModel
from run_evaluation_with_llm import (
    get_agent_system_prompt_by_scenario,
    get_path_list_by_scenario,
)


SCENARIOS = [
    "telecom_package",
    "property_service",
    "logistics_delivery",
    "airline_refund",
    "online_education",
]


class ScriptedClient:
    """Minimal OpenAI-compatible client for deterministic tool-loop tests."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def generate(self, prompt, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        return self.responses.pop(0)


def tool_response(call):
    return SimpleNamespace(
        text="",
        tool_calls=[call],
        metadata={},
        model="fake",
    )


def final_response(scenario, path_config):
    field_names = list(get_scenario_config(scenario).classification_fields.keys())
    values = path_config.get("Classification_items", [])
    classification = {
        field: values[index] if index < len(values) else None
        for index, field in enumerate(field_names)
    }
    return SimpleNamespace(
        text=json.dumps(
            {
                "classification_output": classification,
                "now_path": path_config["expected_path"],
                "finals": {
                    "Action": path_config["final_output"]["Action"],
                    "PLAN": path_config["final_output"].get("PLAN", "none"),
                },
                "chat": "已根据后台查询结果完成处理。",
            },
            ensure_ascii=False,
        ),
        tool_calls=[],
        metadata={},
        model="fake",
    )


def build_scripted_run(scenario, path_config):
    case = build_case_spec(scenario, "backend_e2e", path_config, user_id="backend-e2e")
    backend = create_backend(case)
    record_id = case.user_knowledge["record_id"]
    responses = []

    # Reproduce the minimum authoritative verification plan declared for the
    # selected PathList case.  The Agent must earn each public fact by calling
    # its formal query tool.
    for index, requirement in enumerate(
        case.metadata["required_backend_verifications"]
    ):
        responses.append(
            tool_response(
                ToolCall(
                    f"query-{index}",
                    requirement["tool"],
                    {requirement["argument"]: record_id},
                )
            )
        )

    expected_action = path_config["final_output"]["Action"]
    action_tool = next(
        name
        for name, action in backend.get_action_tool_map().items()
        if action == expected_action
    )
    responses.append(
        tool_response(
            ToolCall("action", action_tool, {"record_id": record_id})
        )
    )
    responses.append(final_response(scenario, path_config))

    client = ScriptedClient(responses)
    agent = AgentModel(
        scenario,
        get_sop_graph(scenario),
        system_prompt=get_agent_system_prompt_by_scenario(scenario),
        llm_client=client,
        use_llm_for_full_output=True,
    )
    user = UserModel(
        UserProfile("backend-e2e", "backend_e2e", "strong_conflict", scenario)
    )
    result = DialogueSimulator(
        user_model=user,
        agent_model=agent,
        max_turns=1,
        backend_environment=backend,
        case_spec=case,
    ).run("请处理我的业务。", {"initial_observation": {}})
    report = Evaluator(scenario, get_sop_graph(scenario)).evaluate_simulation(result)
    return case, backend, client, result, report


def test_all_five_scenarios_complete_one_authoritative_tool_loop():
    """Every migrated scenario must complete an actual backend transition."""

    for scenario in SCENARIOS:
        generate_paths, _ = get_path_list_by_scenario(scenario)
        path_config = generate_paths()[0]
        case, backend, client, result, report = build_scripted_run(
            scenario, path_config
        )

        assert result.goal_solved, scenario
        assert result.executed_action == path_config["final_output"]["Action"]
        assert backend.goal_satisfied(), scenario
        assert report.required_verification_score == 1.0, scenario
        assert report.policy_compliance_score == 1.0, scenario
        assert report.action_execution_score == 1.0, scenario
        assert report.environment_goal_fulfillment == 1.0, scenario
        assert not report.error_categories, (scenario, report.error_categories)

        event_names = [event["name"] for event in result.backend_events]
        assert any(event["event_type"] == "action_execution" for event in result.backend_events)
        assert event_names[-1] == path_config["final_output"]["Action"]
        assert client.responses == []


def test_migrated_scenarios_keep_backend_truth_out_of_agent_prompt():
    """Private backend fields are not injected before a formal query."""

    for scenario in SCENARIOS:
        generate_paths, _ = get_path_list_by_scenario(scenario)
        path_config = generate_paths()[0]
        case, _backend, client, _result, _report = build_scripted_run(
            scenario, path_config
        )
        first_request = json.dumps(client.requests[0], ensure_ascii=False)
        assert case.user_knowledge["record_id"] not in first_request
        assert "backend_record" not in first_request
        assert "private_state" not in first_request

        tool_names = [
            tool["function"]["name"] for tool in client.requests[0]["tools"]
        ]
        assert case.metadata["required_backend_verifications"][0]["tool"] in tool_names

