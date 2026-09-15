"""Shared case, tool and backend event data structures."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import copy


@dataclass
class CaseSpec:
    """One fixed benchmark case shared by every participant in a simulation."""

    case_id: str
    scenario: str
    backend_record: Dict[str, Any]
    user_goal: Dict[str, Any]
    user_knowledge: Dict[str, Any]
    user_policy: Dict[str, Any]
    initial_observation: Dict[str, Any]
    expected_outcome: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "scenario": self.scenario,
            "backend_record": copy.deepcopy(self.backend_record),
            "user_goal": copy.deepcopy(self.user_goal),
            "user_knowledge": copy.deepcopy(self.user_knowledge),
            "user_policy": copy.deepcopy(self.user_policy),
            "initial_observation": copy.deepcopy(self.initial_observation),
            "expected_outcome": copy.deepcopy(self.expected_outcome),
            "metadata": copy.deepcopy(self.metadata),
        }


@dataclass
class ToolCall:
    """A normalized tool call independent of provider-specific response shape."""

    call_id: str
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    arguments_valid: bool = True
    argument_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "call_id": self.call_id,
            "name": self.name,
            "arguments": copy.deepcopy(self.arguments),
            "arguments_valid": self.arguments_valid,
            "argument_error": self.argument_error,
        }


@dataclass
class ToolResult:
    """A safe, observable result returned by a backend tool."""

    success: bool
    tool_name: str
    data: Dict[str, Any] = field(default_factory=dict)
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    call_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "tool_name": self.tool_name,
            "data": copy.deepcopy(self.data),
            "error_code": self.error_code,
            "error_message": self.error_message,
            "call_id": self.call_id,
        }


@dataclass
class ActionResult:
    """The result of a business action, including state transition outcome."""

    success: bool
    action_name: str
    data: Dict[str, Any] = field(default_factory=dict)
    error_code: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "action_name": self.action_name,
            "data": copy.deepcopy(self.data),
            "error_code": self.error_code,
            "error_message": self.error_message,
        }


@dataclass
class BackendEvent:
    """Auditable trace entry for every query and state-changing action."""

    event_type: str
    name: str
    arguments: Dict[str, Any]
    result: Dict[str, Any]
    state_before: Optional[Dict[str, Any]] = None
    state_after: Optional[Dict[str, Any]] = None
    turn_index: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "name": self.name,
            "arguments": copy.deepcopy(self.arguments),
            "result": copy.deepcopy(self.result),
            "state_before": copy.deepcopy(self.state_before),
            "state_after": copy.deepcopy(self.state_after),
            "turn_index": self.turn_index,
        }


@dataclass
class UserEnvironmentState:
    """Explicit customer-side state; hidden facts never enter Agent prompts."""

    goal: Dict[str, Any] = field(default_factory=dict)
    facts: Dict[str, Any] = field(default_factory=dict)
    known_facts: Dict[str, Any] = field(default_factory=dict)
    revealed_facts: List[str] = field(default_factory=list)
    emotion: str = "calm"
    satisfaction: float = 0.5
    resolution_status: str = "unsolved"
    escalation_status: str = "none"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": copy.deepcopy(self.goal),
            "facts": copy.deepcopy(self.facts),
            "known_facts": copy.deepcopy(self.known_facts),
            "revealed_facts": list(self.revealed_facts),
            "emotion": self.emotion,
            "satisfaction": self.satisfaction,
            "resolution_status": self.resolution_status,
            "escalation_status": self.escalation_status,
        }
