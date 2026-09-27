"""Read-only normalization of an EvoSAGE simulation into one timeline.

The underlying simulator/backend/provider artifacts remain authoritative; this
module creates a deterministic analysis view and is never a scoring input.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class TraceEvent:
    seq: int
    turn_index: int | None
    event_type: str
    actor: str
    name: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "to_dict"):
        converted = value.to_dict()
        return converted if isinstance(converted, dict) else {}
    return {}


def _sequence(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def flatten_simulation(simulation: Any) -> list[TraceEvent]:
    """Flatten a SimulationResult (or serialized mapping) without hidden truth."""
    source = _mapping(simulation)
    if not source and simulation is not None:
        return []
    turns = _sequence(source.get("turns"))
    backend_events = _sequence(source.get("backend_events"))
    customer_records = _sequence(source.get("customer_simulator_provenance"))
    customer_by_turn: dict[int | None, list[dict[str, Any]]] = {}
    backend_by_turn: dict[int | None, list[dict[str, Any]]] = {}

    def normalized_turn(value: Any) -> int | None:
        return int(value) if isinstance(value, int) else None

    for record in customer_records:
        record = _mapping(record)
        customer_by_turn.setdefault(normalized_turn(record.get("turn_index")), []).append(record)
    for event in backend_events:
        event = _mapping(event)
        backend_by_turn.setdefault(normalized_turn(event.get("turn_index")), []).append(event)

    events: list[TraceEvent] = []

    def append(turn: int | None, kind: str, actor: str, name: str, payload: dict[str, Any]) -> None:
        events.append(TraceEvent(
            seq=len(events), turn_index=turn, event_type=kind, actor=actor,
            name=name, payload=payload,
        ))

    def add_customer_attempts(turn: int | None) -> None:
        for record in customer_by_turn.pop(turn, []):
            stage = str(record.get("stage") or "message")
            for attempt in _sequence(record.get("attempts")):
                attempt = _mapping(attempt)
                fields = (
                    "attempt_index", "thinking_mode", "finish_reason", "max_tokens",
                    "input_tokens", "completion_tokens", "reasoning_tokens", "raw_content",
                    "provider_request_id", "latency_seconds", "provider_error", "timeout",
                    "parse_status", "generation_status", "invalid_reason",
                )
                append(turn, "CUSTOMER_SIMULATOR_ATTEMPT", "customer_simulator", stage, {
                    key: attempt[key] for key in fields if key in attempt
                })
            if record.get("status") == "invalid":
                append(turn, "PROTOCOL_ERROR", "customer_simulator", stage, {
                    "reason": record.get("invalid_reason") or "customer_simulator_invalid",
                })

    def add_backend_events(turn: int | None, model_calls: list[dict[str, Any]], model_results: list[dict[str, Any]]) -> None:
        turn_events = backend_by_turn.pop(turn, [])
        matched_model_calls: set[int] = set()
        matched_result_ids: set[str] = set()
        for event in turn_events:
            event_type = event.get("event_type")
            name = str(event.get("name") or "unknown")
            arguments = event.get("arguments", {})
            result = _mapping(event.get("result"))
            call_id = result.get("call_id")
            matching_call = next((
                (index, call) for index, call in enumerate(model_calls)
                if index not in matched_model_calls
                and (not call_id or str(call.get("id") or call.get("call_id") or "") == str(call_id))
                and str(call.get("name") or call.get("tool_name") or "unknown") == name
                and call.get("arguments", call.get("args", {})) == arguments
            ), None)
            if event_type == "tool_call":
                if matching_call is not None:
                    matched_model_calls.add(matching_call[0])
                append(turn, "TOOL_CALL", "agent" if matching_call else "backend", name, {
                    "call_id": call_id,
                    "arguments": arguments,
                })
                append(turn, "TOOL_RESULT", "backend", name, result)
                if call_id:
                    matched_result_ids.add(str(call_id))
            elif event_type == "action_execution":
                append(turn, "ACTION_EXECUTION", "backend", name, result)
                before = _mapping(event.get("state_before"))
                after = _mapping(event.get("state_after"))
                changed = sorted(
                    key for key in set(before) | set(after)
                    if before.get(key) != after.get(key)
                )
                if changed:
                    # Do not copy authoritative state values into this display
                    # projection; retain only the fact and field names changed.
                    append(turn, "STATE_CHANGE", "backend", name, {
                        "changed_top_level_keys": changed,
                    })
            elif event_type == "state_change":
                append(turn, "STATE_CHANGE", "backend", name, {
                    key: event[key] for key in ("changed_fields", "change", "success") if key in event
                })

        # Preserve parsed calls that never reached Backend (for example a
        # protocol/tool-loop failure), without pretending that they executed.
        for index, call in enumerate(model_calls):
            if index in matched_model_calls:
                continue
            append(turn, "TOOL_CALL", "agent", str(call.get("name") or call.get("tool_name") or "unknown"), {
                "call_id": call.get("id") or call.get("call_id"),
                "arguments": call.get("arguments", call.get("args")),
                "backend_event_recorded": False,
            })
        for result in model_results:
            result_id = str(result.get("call_id") or "")
            if result_id and result_id in matched_result_ids:
                continue
            if any(
                event.get("event_type") == "tool_call"
                and event.get("name") == result.get("tool_name")
                and _mapping(event.get("result")).get("call_id") == result.get("call_id")
                for event in turn_events
            ):
                continue
            append(turn, "TOOL_RESULT", "backend", str(result.get("tool_name") or "unknown"), result)

    for turn_record in turns:
        turn_record = _mapping(turn_record)
        turn = normalized_turn(turn_record.get("turn_id", turn_record.get("turn_index")))
        add_customer_attempts(turn)
        append(turn, "USER_MESSAGE", "user", "message", {
            "text": str(turn_record.get("user_message") or ""),
        })
        output = _mapping(turn_record.get("agent_output"))
        append(turn, "AGENT_MESSAGE", "agent", "message", {
            "text": str(output.get("chat") or ""),
            "predicted_action": turn_record.get("predicted_action") or output.get("predicted_action"),
            "predicted_path": turn_record.get("predicted_path") or output.get("predicted_path") or [],
        })
        metadata = _mapping(output.get("metadata"))
        request = _mapping(metadata.get("llm_request"))
        attempts = _sequence(metadata.get("llm_attempts"))
        invalid_reason = (
            metadata.get("invalid_reason")
            or ("json_parse_failed" if output.get("json_parse_failed") else None)
            or ("timeout" if metadata.get("timeout") or metadata.get("llm_timeout") else None)
            or ("provider_error" if metadata.get("provider_error") or metadata.get("llm_error") else None)
            or ("output_truncated" if request.get("finish_reason") == "length" else None)
            or ("output_truncated" if any(_mapping(item).get("finish_reason") == "length" for item in attempts) else None)
        )
        if invalid_reason or metadata.get("protocol_failure"):
            append(turn, "PROTOCOL_ERROR", "agent", "response", {
                "reason": str(invalid_reason or "protocol_failure"),
            })
        model_calls = [
            dict(item) for item in _sequence(turn_record.get("tool_calls") or output.get("tool_calls"))
            if isinstance(item, dict)
        ]
        model_results = [
            dict(item) for item in _sequence(turn_record.get("tool_results") or output.get("tool_results"))
            if isinstance(item, dict)
        ]
        add_backend_events(turn, model_calls, model_results)

    # Keep attempts/errors that occur before a turn was materialized.
    add_customer_attempts(None)
    for turn in sorted(backend_by_turn, key=lambda value: -1 if value is None else value):
        add_backend_events(turn, [], [])
    return events
