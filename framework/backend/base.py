"""Scenario-independent backend environment interfaces."""

from typing import Any, Dict, List, Optional
import copy
import json

from .types import ActionResult, BackendEvent, CaseSpec, ToolResult


class BackendEnvironment:
    """Authoritative environment for one benchmark case.

    The full state is available to the evaluator, but only tool results are
    exposed to the Agent. Scenario adapters implement the actual business
    tools and transitions.
    """

    scenario_id = "generic"

    def __init__(self, case_spec: Optional[CaseSpec] = None):
        self.case_spec: Optional[CaseSpec] = None
        self.state: Dict[str, Any] = {}
        self.event_log: List[BackendEvent] = []
        self.selected_records: Dict[str, Any] = {}
        if case_spec is not None:
            self.reset(case_spec)

    def reset(self, case_spec: CaseSpec) -> None:
        self.case_spec = case_spec
        self.state = copy.deepcopy(case_spec.backend_record)
        self.event_log = []
        self.selected_records = {}

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        return []

    def get_action_tool_map(self) -> Dict[str, str]:
        """Return formal-tool-name -> canonical SOP action name."""
        return {}

    def is_action_tool(self, tool_name: str) -> bool:
        return tool_name in self.get_action_tool_map()

    def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        turn_index: Optional[int] = None,
        call_id: Optional[str] = None,
    ) -> ToolResult:
        if self.is_action_tool(tool_name):
            action_name = self.get_action_tool_map()[tool_name]
            action_result = self.execute_action(
                action_name,
                arguments or {},
                turn_index=turn_index,
            )
            return ToolResult(
                success=action_result.success,
                tool_name=tool_name,
                data=copy.deepcopy(action_result.data),
                error_code=action_result.error_code,
                error_message=action_result.error_message,
                call_id=call_id,
            )
        before = self.get_state_snapshot()
        try:
            result = self._execute_tool(tool_name, arguments or {})
        except Exception as exc:
            result = ToolResult(
                success=False,
                tool_name=tool_name,
                error_code="backend_exception",
                error_message=str(exc),
                call_id=call_id,
            )
        result.call_id = call_id
        self.event_log.append(
            BackendEvent(
                event_type="tool_call",
                name=tool_name,
                arguments=copy.deepcopy(arguments or {}),
                result=result.to_dict(),
                state_before=before,
                state_after=self.get_state_snapshot(),
                turn_index=turn_index,
            )
        )
        return result

    def execute_action(
        self,
        action_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        turn_index: Optional[int] = None,
    ) -> ActionResult:
        before = self.get_state_snapshot()
        try:
            result = self._execute_action(action_name, arguments or {})
        except Exception as exc:
            result = ActionResult(
                success=False,
                action_name=action_name,
                error_code="backend_exception",
                error_message=str(exc),
            )
        self.event_log.append(
            BackendEvent(
                event_type="action_execution",
                name=action_name,
                arguments=copy.deepcopy(arguments or {}),
                result=result.to_dict(),
                state_before=before,
                state_after=self.get_state_snapshot(),
                turn_index=turn_index,
            )
        )
        return result

    def get_state_snapshot(self) -> Dict[str, Any]:
        return copy.deepcopy(self.state)

    def get_event_log(self) -> List[Dict[str, Any]]:
        return [event.to_dict() for event in self.event_log]

    def get_rule_context(self) -> Dict[str, Any]:
        """Compatibility view for the existing rule engine.

        New evaluators should derive this from the authoritative state instead
        of constructing a separate hidden system_info object.
        """
        return {"system_info": copy.deepcopy(self.state)}

    def get_public_state(self) -> Dict[str, Any]:
        """Public state explicitly allowed as an initial Agent observation."""
        if self.case_spec is None:
            return {}
        return copy.deepcopy(self.case_spec.initial_observation)

    def goal_satisfied(self) -> bool:
        if self.case_spec is None:
            return False
        expected = self.case_spec.expected_outcome
        if not expected:
            return False
        if (
            self.case_spec.scenario == "online_education"
            and self._lookup(self.state, "interaction.last_action") == "PLAN"
            and not self._lookup(self.state, "interaction.answer_completed")
        ):
            # Resource allocation is not itself an answer to a knowledge question.
            return False
        return all(self._lookup(self.state, key) == value for key, value in expected.items())

    @staticmethod
    def _lookup(data: Dict[str, Any], dotted_key: str) -> Any:
        current: Any = data
        for part in dotted_key.split("."):
            if not isinstance(current, dict) or part not in current:
                return None
            current = current[part]
        return current

    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        return ToolResult(
            success=False,
            tool_name=tool_name,
            error_code="unknown_tool",
            error_message=f"Unknown tool: {tool_name}",
        )

    def _execute_action(self, action_name: str, arguments: Dict[str, Any]) -> ActionResult:
        return ActionResult(
            success=False,
            action_name=action_name,
            error_code="unsupported_action",
            error_message=f"Unsupported action: {action_name}",
        )


def tool_definition(
    name: str,
    description: str,
    properties: Dict[str, Any],
    required: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build an OpenAI-compatible function tool definition."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required or [],
                "additionalProperties": False,
            },
        },
    }
