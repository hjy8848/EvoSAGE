from framework.evolution.schemas import EpisodeResult
from framework.evolution.trace import TraceEvent, flatten_simulation


def _sample_simulation():
    return {
        "simulation_id": "sim_trace_1",
        "customer_simulator_provenance": [{
            "turn_index": 0,
            "stage": "initial_message",
            "status": "valid",
            "attempts": [{
                "attempt_index": 1,
                "finish_reason": "stop",
                "raw_content": "我要查询订单。",
                "completion_tokens": 8,
            }],
        }],
        "turns": [{
            "turn_id": 0,
            "user_message": "我要查询订单。",
            "agent_output": {
                "chat": "请提供订单号。",
                "predicted_action": "Supplementary",
                "tool_calls": [{
                    "call_id": "call-1", "name": "query_order",
                    "arguments": {"order_id": ""},
                }],
            },
            "tool_calls": [{
                "call_id": "call-1", "name": "query_order",
                "arguments": {"order_id": ""},
            }],
            "tool_results": [{
                "call_id": "call-1", "tool_name": "query_order",
                "success": False, "error_code": "order_not_found",
            }],
        }],
        "backend_events": [
            {
                "event_type": "tool_call", "name": "query_order", "turn_index": 0,
                "arguments": {"order_id": ""},
                "result": {"call_id": "call-1", "tool_name": "query_order",
                           "success": False, "error_code": "order_not_found"},
                "state_before": {"secret": "hidden-a"},
                "state_after": {"secret": "hidden-a"},
            },
            {
                "event_type": "action_execution", "name": "Refund", "turn_index": 0,
                "result": {"success": False, "error_code": "not_authorized"},
                "state_before": {"refund_status": "None"},
                "state_after": {"refund_status": "None"},
            },
            {
                "event_type": "action_execution", "name": "Register", "turn_index": 0,
                "result": {"success": True},
                "state_before": {"ticket_status": "None"},
                "state_after": {"ticket_status": "Registered"},
            },
        ],
    }


def test_flatten_simulation_orders_customer_agent_tool_and_state_events():
    events = flatten_simulation(_sample_simulation())
    assert all(isinstance(event, TraceEvent) for event in events)
    assert [event.seq for event in events] == list(range(len(events)))
    assert [event.event_type for event in events] == [
        "CUSTOMER_SIMULATOR_ATTEMPT",
        "USER_MESSAGE",
        "AGENT_MESSAGE",
        "TOOL_CALL",
        "TOOL_RESULT",
        "ACTION_EXECUTION",
        "ACTION_EXECUTION",
        "STATE_CHANGE",
    ]
    query_call = events[3]
    assert query_call.name == "query_order"
    assert query_call.payload["arguments"] == {"order_id": ""}
    assert events[4].payload["error_code"] == "order_not_found"
    assert events[-1].payload == {"changed_top_level_keys": ["ticket_status"]}
    assert "hidden-a" not in repr([event.to_dict() for event in events])


def test_flatten_simulation_marks_agent_protocol_failure_and_unmaterialized_customer_attempt():
    simulation = {
        "turns": [{
            "turn_id": 0,
            "user_message": "Hi",
            "agent_output": {
                "chat": "",
                "json_parse_failed": True,
                "metadata": {"invalid_reason": "json_parse_failed"},
            },
        }],
        "customer_simulator_provenance": [{
            "stage": "opening", "status": "invalid", "invalid_reason": "output_truncated",
            "attempts": [{"attempt_index": 1, "finish_reason": "length"}],
        }],
    }
    events = flatten_simulation(simulation)
    assert [(event.event_type, event.payload.get("reason")) for event in events if event.event_type == "PROTOCOL_ERROR"] == [
        ("PROTOCOL_ERROR", "json_parse_failed"),
        ("PROTOCOL_ERROR", "output_truncated"),
    ]


def test_failure_occurrence_keeps_trace_reference_and_sequence_bounds():
    episode = EpisodeResult(
        episode_id="episode-trace-1",
        scenario="ecommerce_refund",
        case_id="case-1",
        customer_policy_id="customer-1",
        service_policy_id="service-1",
        split="evolution",
        generation=0,
        task_success=False,
        execution_score=0.0,
        error_types=["wrong_tool_arguments"],
        trace_ref="simulation:sim_trace_1",
        service_failure_attributable=True,
        metadata={"analysis_trace_events": [{"seq": 0}, {"seq": 1}, {"seq": 2}]},
    )
    occurrence = episode.to_dict()["failure_occurrence"]
    assert occurrence["trace_ref"] == "simulation:sim_trace_1"
    assert occurrence["trace_seq_start"] == 0
    assert occurrence["trace_seq_end"] == 2
    assert occurrence["metadata"]["trace_range_scope"] == "whole_episode"
    assert "analysis_trace_events" not in occurrence["metadata"]
