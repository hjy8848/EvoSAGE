"""Read-only exporter tests for current Customer-search artifacts."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


EXPORTER_PATH = Path(__file__).parents[1] / "scripts" / "export_dashboard_data.py"
SPEC = importlib.util.spec_from_file_location("evosage_dashboard_exporter", EXPORTER_PATH)
assert SPEC and SPEC.loader
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_jsonl(path: Path, values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(value) for value in values) + "\n", encoding="utf-8")


def make_run(tmp_path: Path, evaluator="real") -> Path:
    run = tmp_path / "run"
    write_json(run / "config" / "customer_search.json", {
        "scenario": "ecommerce_refund",
        "seed": 17,
        "max_generations": 1,
        "customer": {"candidate_count": 2},
        "splits": {"strategy": "instance_holdout", "seed": 5},
        "evaluation": {"max_turns": 4},
        "model_metadata": {"model": "deepseek-v4-flash", "provider": "InferAI"},
    })
    write_json(run / "environment" / "provenance.json", {
        "method": "customer_search",
        "evaluator": evaluator,
        "model": "deepseek-v4-flash",
        "provider": "InferAI",
        "seed": 17,
        "runtime_commit": "actual-runtime-sha",
        "runtime_tag": "runtime-tag",
        "heldout_used_for_adaptation_or_selection": False,
        "service_policy_id": "service_policy_s0",
    })
    for name in ("evolution_cases", "validation_cases", "heldout_cases"):
        write_json(run / "split_manifest" / f"{name}.json", {
            "cases": [{"case_id": f"{name}-1"}],
        })
    write_json(run / "analysis" / "run_status.json", {"status": "complete"})
    write_json(run / "analysis" / "orchestration_metrics.json", {
        "provider_attempts": 9,
        "input_tokens": 900,
        "output_tokens": 120,
        "latency_seconds": 12.5,
        "timeouts": 1,
        "provider_failures": 0,
        "customer_evolver": {"attempts": 1, "requests": 1},
        "episode_evaluator": {"attempts": 8, "pipeline_requests": 8},
    })
    generation = run / "generations" / "gen_000"
    write_json(generation / "COMPLETE.json", {"generation": 0})
    write_json(generation / "SUMMARY.json", {
        "generation": 0,
        "status": "complete",
        "selected_policy_id": "customer-1",
        "incumbent_policy_id": "customer-0",
        "customer_changed": True,
        "selected_fitness": 1.0,
        "selected_official_task_success": 0.0,
        "proposed_candidate_count": 2,
        "evolution_evaluation": {"valid_episode_count": 2, "invalid_episode_count": 1},
    })
    write_json(generation / "proposals.json", {
        "requested_candidate_count": 2,
        "proposed_candidate_count": 2,
        "candidate_policies": [{
            "policy_id": "customer-1",
            "strategy": "Claim the delivery record is wrong.",
            "hypothesis": "This may induce rechecking.",
            "provenance_hash": "policy-hash",
        }],
        "generation_record": {"status": "valid", "candidates": []},
    })
    write_json(generation / "selection.json", {
        "selected_policy_id": "customer-1",
        "incumbent_policy_id": "customer-0",
        "selected_policy": {"policy_id": "customer-1", "strategy": "Claim the delivery record is wrong."},
        "selected_score": {"fitness": 1.0, "official_task_success": 0.0},
        "selected_episode_ids": ["failure-1"],
        "candidate_scores": [{"policy_id": "customer-1", "fitness": 1.0, "official_task_success": 0.0}],
        "selection_reason": "strict_official_fitness_improvement",
    })
    write_json(generation / "service_policy.json", {"policy_id": "service_policy_s0"})
    write_jsonl(generation / "episodes.jsonl", [
        {
            "episode_id": "failure-1",
            "scenario": "ecommerce_refund",
            "case_id": "CASE-1",
            "customer_policy_id": "customer-1",
            "service_policy_id": "service_policy_s0",
            "split": "evolution",
            "generation": 0,
            "task_success": False,
            "execution_score": 0.25,
            "action_execution_score": 0.0,
            "goal_fulfillment_score": 0.0,
            "error_types": ["goal_not_fulfilled"],
            "predicted_action": "Supplementary",
            "executed_action": "",
            "tool_sequence_summary": ["query_order"],
            "trace_ref": "simulation:sim-1",
            "metadata": {
                "phase": "customer_candidate",
                "model_name": "deepseek-v4-flash",
                "trace_events": [
                    {"seq": 0, "turn_index": 0, "event_type": "USER_MESSAGE", "actor": "user", "payload": {"text": "I want a refund."}},
                    {"seq": 1, "turn_index": 0, "event_type": "TOOL_CALL", "actor": "agent", "name": "query_order", "payload": {"arguments": {"order_id": "ORD-123"}}},
                    {"seq": 2, "turn_index": 0, "event_type": "TOOL_RESULT", "actor": "backend", "name": "query_order", "payload": {"success": False, "error_code": "order_not_found", "state_before": {"secret": "must-not-export"}}},
                ],
            },
        },
        {
            "episode_id": "invalid-1",
            "scenario": "ecommerce_refund",
            "case_id": "CASE-2",
            "customer_policy_id": "customer-1",
            "service_policy_id": "service_policy_s0",
            "split": "evolution",
            "generation": 0,
            "task_success": False,
            "evaluation_status": "invalid",
            "invalid_reason": "output_truncated",
            "protocol_valid": False,
            "environment_valid": True,
        },
    ])
    return run


def test_exporter_copies_official_result_and_display_trace_without_rescoring(tmp_path):
    run = make_run(tmp_path)
    exported = exporter.export_run("customer-search", run)

    assert exported["metadata"]["method"] == "customer_search"
    assert exported["metadata"]["source_kind"] == "REAL"
    assert exported["metadata"]["runtime_commit"] == "actual-runtime-sha"
    assert exported["metadata"]["runtime_freeze_match_status"] == "not_confirmed"
    assert exported["metrics"]["task_success"] == 0.0
    assert exported["metrics"]["invalid_rate"] == 0.5
    assert exported["metrics"]["requests"] == 9
    assert "service_candidates" not in exported
    assert "robustness" not in exported
    assert "unique_failure_signatures" not in exported["metrics"]

    failure = next(item for item in exported["episodes"] if item["episode_id"] == "failure-1")
    assert failure["status"] == "VALID FAILURE"
    assert failure["task_success"] is False
    tool_call = next(item for item in failure["trace_events"] if item["kind"] == "tool_call")
    backend_result = next(item for item in failure["trace_events"] if item["kind"] == "backend_result")
    assert tool_call["label"] == "query_order"
    assert tool_call["payload"]["arguments"] == {"order_id": "ORD-123"}
    assert backend_result["payload"]["error_code"] == "order_not_found"
    assert "state_before" not in backend_result["payload"]
    invalid = next(item for item in exported["episodes"] if item["episode_id"] == "invalid-1")
    assert invalid["status"] == "INVALID EVALUATION"
    assert invalid["task_success"] is None


def test_current_runtime_metrics_count_evaluator_and_evolver_requests():
    metrics = exporter._runtime_metrics({
        "provider_attempts": 27,
        "episode_evaluator": {"pipeline_requests": 26},
        "customer_evolver": {"requests": 1},
    })
    assert metrics["requests"] == 27
    assert metrics["attempts"] == 27


def test_trace_events_are_read_from_episode_metadata_trace_events():
    events = exporter.trace_events({
        "metadata": {
            "trace_events": [
                {"seq": 1, "event_type": "TOOL_CALL", "actor": "agent",
                 "name": "query_order", "payload": {"arguments": {"order_id": "ORD-1"}}},
            ],
        },
    })
    assert len(events) == 1
    assert events[0]["kind"] == "tool_call"
    assert events[0]["payload"]["arguments"] == {"order_id": "ORD-1"}


def test_mock_source_is_explicit_and_candidate_provenance_is_preserved(tmp_path):
    run = make_run(tmp_path, evaluator="mock")
    exported = exporter.export_run("mock-search", run)
    assert exported["metadata"]["source_kind"] == "MOCK"
    candidate = exported["customer_candidates"][0]["candidates"][0]
    assert candidate["strategy"] == "Claim the delivery record is wrong."
    assert candidate["provenance_hash"] == "policy-hash"
    assert exported["customer_candidates"][0]["selected_policy_id"] == "customer-1"


def test_trajectory_analyzer_reads_summary_and_does_not_rescore(tmp_path):
    run = make_run(tmp_path)
    analyzer_path = Path(__file__).parents[1] / "scripts" / "analyze_evolution_trajectory.py"
    analyzer_spec = importlib.util.spec_from_file_location("evosage_trajectory_analyzer", analyzer_path)
    assert analyzer_spec and analyzer_spec.loader
    analyzer = importlib.util.module_from_spec(analyzer_spec)
    analyzer_spec.loader.exec_module(analyzer)
    result = analyzer.analyze_run("run", run)
    assert result["generations"][0]["selected_fitness"] == 1.0
    assert result["generations"][0]["selected_official_task_success"] == 0.0
    assert result["valid_episode_count"] == 1
    assert result["invalid_episode_count"] == 1
