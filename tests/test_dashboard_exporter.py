"""Read-only correctness tests for the EvoSAGE dashboard exporter."""

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


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def make_run(tmp_path: Path, *, source_kind: str = "REAL") -> Path:
    run = tmp_path / "run"
    write_json(
        run / "config" / "evolution.json",
        {
            "experiment_mode": "coevolution",
            "model": "deepseek-v4-flash",
            "seed": 7,
            "max_generations": 1,
            "runtime_freeze_commit": "runtime-sha",
            "runtime_freeze_tag": "runtime-tag",
            "formal_protocol": {"commit": "protocol-sha", "tag": "protocol-tag"},
            "splits": {"strategy": "instance_holdout", "seed": 17, "max_cases": 3},
            "customer": {"candidate_count": 3, "elite_count": 1},
            "service": {"candidate_count": 3, "replay_attack_count": 1},
            "evaluation": {"max_turns": 5},
        },
    )
    write_json(
        run / "environment" / "provenance.json",
        {
            "source_kind": source_kind,
            "provider": "InferAI",
            "commit_sha": "runtime-sha",
            "freeze_tag": "runtime-tag",
            "observed_runtime_freeze_commit": "runtime-sha",
            "observed_runtime_freeze_tag": "runtime-tag",
        },
    )
    write_json(
        run / "analysis" / "orchestration_metrics.json",
        {
            "llm_requests": 11,
            "input_tokens": 1000,
            "output_tokens": 200,
            "latency_seconds": 12.5,
            "timeouts": 0,
            "failures": 0,
            "retries": 1,
            "attempts": 12,
            "successes": 11,
            "average_attempt_latency": 1.04,
            "max_attempt_latency": 2.2,
        },
    )
    for name in ("evolution_cases.json", "validation_cases.json", "heldout_cases.json"):
        write_json(run / "split_manifest" / name, [])
    write_json(run / "generations" / "gen_000" / "COMPLETE.json", {"complete": True})
    episodes = [
        {
            "episode_id": "valid-episode",
            "generation": 0,
            "split": "evolution",
            "case_id": "CASE-1",
            "task_success": True,
            "strict_process_success": True,
            "eventual_goal_success": 1.0,
            "recovery_success": False,
            "protocol_valid": True,
            "environment_valid": True,
            "customer_behavior_valid": True,
            "service_failure_attributable": False,
            "vulnerability_signature": None,
            "failure_occurrence": None,
            "expected_action": "Refund",
            "predicted_action": "Refund",
            "executed_action": "Refund",
            "latency_seconds": 2.5,
            "failure_signature": {},
        },
        {
            "episode_id": "legitimate-failure",
            "generation": 0,
            "split": "evolution",
            "case_id": "CASE-2",
            "task_success": False,
            "strict_process_success": False,
            "eventual_goal_success": 0.0,
            "recovery_success": False,
            "protocol_valid": True,
            "environment_valid": True,
            "customer_behavior_valid": True,
            "service_failure_attributable": True,
            "vulnerability_signature": {
                "schema_version": 2,
                "signature_id": "v2-missing-order-id",
                "failure_stage": "VERIFICATION",
            },
            "failure_occurrence": {"occurrence_id": "occurrence-1"},
            "error_types": ["missed_backend_verification"],
            "failure_signature": {
                "signature_id": "missing_order_id",
                "sop_node": "shipping_verification",
            },
        },
        {
            "episode_id": "invalid-episode",
            "generation": 0,
            "split": "evolution",
            "case_id": "CASE-3",
            "task_success": False,
            "evaluation_status": "invalid",
            "invalid_reason": "output_truncated",
            "metadata": {"protocol_failure": True},
            "protocol_valid": False,
            "environment_valid": True,
            "customer_behavior_valid": True,
            "service_failure_attributable": False,
            "vulnerability_signature": None,
            "failure_occurrence": None,
        },
    ]
    write_jsonl(run / "generations" / "gen_000" / "episodes.jsonl", episodes)
    write_jsonl(
        run / "real_traces" / "trace.jsonl",
        [
            {
                "trace_file": "real_traces/trace.jsonl",
                "simulation": {
                    "case_spec": {"case_id": "CASE-2"},
                    "simulation_id": "sim-2",
                    "model_name": "deepseek-v4-flash",
                    "duration_seconds": 3.25,
                    "turns": [
                        {
                            "turn_id": 0,
                            "user_message": "I need a refund but do not have the order number.",
                            "agent_output": {"chat": "I will check the order."},
                            "tool_calls": [
                                {"name": "query_order", "arguments": {"order_id": ""}}
                            ],
                            "tool_results": [{"error": "order_not_found", "state_before": "hidden"}],
                        }
                    ],
                },
            }
        ],
    )
    write_json(
        run / "generations" / "gen_000" / "customer_generation.json",
        {
            "status": "valid",
            "response_candidate_count": 3,
            "candidates": [
                {
                    "candidate_index": i,
                    "policy_id": f"customer-{i}",
                    "constructed_policy": {"policy_id": f"customer-{i}", "strategy_tags": ["withholding"]},
                    "raw_candidate": {"strategy_tags": ["withholding"]},
                    "accepted": True,
                }
                for i in range(3)
            ],
        },
    )
    write_json(
        run / "generations" / "gen_000" / "customer_candidates.json",
        {
            "selected_policy": {"policy_id": "customer-0"},
            "scores": [
                {"policy_id": "incumbent", "fitness": 0.2},
                {"policy_id": "customer-0", "fitness": 0.8},
                {"policy_id": "customer-1", "fitness": 0.4},
                {"policy_id": "customer-2", "fitness": 0.3},
            ],
        },
    )
    write_json(
        run / "generations" / "gen_000" / "service_generation.json",
        {"status": "valid", "candidates": [], "selected_policy_id": "service-0"},
    )
    write_json(
        run / "generations" / "gen_000" / "service_gate.json",
        {
            "candidates": [
                {
                    "patch_id": "patch-1",
                    "accepted": True,
                    "evaluation_status": "valid",
                    "delta": 0.4,
                    "reason": "improved",
                    "source_failure_ids": ["missing_order_id"],
                    "patch": {"rule_category": "VERIFICATION", "rule_text": "Ask for order id."},
                    "metrics": {
                        "latest_task_success": 1.0,
                        "exact_replay_task_success": 1.0,
                        "transfer_replay_task_success": 0.5,
                        "normal_task_success": 1.0,
                        "latest_paired": {"wins": 1, "losses": 0, "ties": 0},
                        "exact_replay_paired": {"wins": 0, "losses": 0, "ties": 1},
                        "transfer_replay_paired": {"wins": 1, "losses": 0, "ties": 0},
                        "normal_paired": {"wins": 0, "losses": 0, "ties": 1},
                    },
                },
                {
                    "patch_id": "patch-2",
                    "accepted": False,
                    "evaluation_status": "valid",
                    "delta": 0.0,
                    "reason": "insufficient_improvement",
                    "source_failure_ids": ["missing_order_id"],
                    "patch": {"rule_category": "VERIFICATION", "rule_text": "Ask for order id."},
                },
                {
                    "patch_id": "patch-3",
                    "accepted": False,
                    "evaluation_status": "invalid",
                    "invalid_reasons": ["output_truncated"],
                    "reason": "candidate_evaluation_invalid:output_truncated",
                    "patch": {"rule_category": "TOOL_USE", "rule_text": "Use grounded args."},
                },
            ]
        },
    )
    write_jsonl(
        run / "archives" / "attack_instances.jsonl",
        [{"attack_instance_id": "instance-1", "vulnerability_signature_id": "v2-missing-order-id"}],
    )
    return run


def test_exporter_preserves_statuses_candidates_and_trace_provenance(tmp_path: Path):
    run = make_run(tmp_path)
    result = exporter.export_run("real", run)

    assert result["metadata"]["source_kind"] == "REAL"
    assert result["metadata"]["runtime_freeze_match_status"] == "matched"
    assert result["metadata"]["formal_protocol_commit"] == "protocol-sha"
    assert result["metadata"]["candidate_counts"] == {
        "customer": 3,
        "customer_elite": 1,
        "service": 3,
        "service_replay": 1,
    }
    assert result["metrics"]["valid_episodes"] == 2
    assert result["metrics"]["legitimate_failures"] == 1
    assert result["metrics"]["invalid_episodes"] == 1
    assert result["metrics"]["protocol_invalid_rate"] == 1 / 3
    assert result["metrics"]["customer_behavior_invalid_rate"] == 0.0
    assert result["metrics"]["attributable_service_failure_rate"] == 0.5
    assert result["metrics"]["strict_process_success"] == 0.5
    assert result["metrics"]["eventual_goal_success"] == 0.5
    assert result["metrics"]["recovery_success_rate"] == 0.0
    assert result["metrics"]["unique_vulnerability_signatures_v2"] == 1
    assert result["metrics"]["attack_instance_count"] == 1
    assert result["metrics"]["latest_task_success"] == 1.0
    assert result["metrics"]["exact_replay_task_success"] == 1.0
    assert result["metrics"]["transfer_replay_task_success"] == 0.5
    assert result["metrics"]["normal_task_success"] == 1.0
    assert result["metrics"]["paired_outcomes"]["latest"]["wins"] == 1
    assert result["failure_analysis"]["unique_count"] == 1
    assert result["failure_analysis"]["legitimate_failure_count"] == 1
    assert result["customer_candidates"][0]["candidate_count"] == 3
    assert result["customer_candidates"][0]["evaluated_candidate_count"] == 4

    episode = next(item for item in result["episodes"] if item["episode_id"] == "legitimate-failure")
    assert episode["status"] == "LEGITIMATE FAILURE"
    assert episode["attribution_status"] == "attributable"
    assert episode["latency_seconds"] == 3.25
    tool = next(item for item in episode["trace_events"] if item["kind"] == "tool_call")
    backend = next(item for item in episode["trace_events"] if item["kind"] == "backend_result")
    assert tool["payload"] == {"name": "query_order", "arguments": {"order_id": ""}}
    assert backend["payload"] == {"error": "order_not_found"}

    invalid = next(item for item in result["episodes"] if item["episode_id"] == "invalid-episode")
    assert invalid["status"] == "INVALID EVALUATION"
    assert invalid["invalid_reason"] == "output_truncated"
    assert result["repair_analysis"]["statistics"]["total_service_proposals"] == 3
    assert result["repair_analysis"]["statistics"]["unique_service_proposals"] == 2
    assert result["repair_analysis"]["statistics"]["repeated_service_proposals"] == 1
    assert result["robustness"]["heldout"] is None
    assert result["robustness"]["fresh_adversary"] is None


def test_exporter_marks_mock_without_reclassifying_business_failures(tmp_path: Path):
    run = make_run(tmp_path, source_kind="MOCK")
    result = exporter.export_run("mock", run)
    assert result["metadata"]["source_kind"] == "MOCK"
    assert result["metrics"]["legitimate_failures"] == 1
    assert result["metrics"]["invalid_episodes"] == 1


def test_legacy_failures_are_not_silently_reclassified_as_attributable(tmp_path: Path):
    run = make_run(tmp_path)
    episodes_path = run / "generations" / "gen_000" / "episodes.jsonl"
    rows = [json.loads(line) for line in episodes_path.read_text().splitlines()]
    for row in rows:
        for key in (
            "strict_process_success", "eventual_goal_success", "recovery_success",
            "protocol_valid", "environment_valid", "customer_behavior_valid",
            "service_failure_attributable", "vulnerability_signature", "failure_occurrence",
        ):
            row.pop(key, None)
    write_jsonl(episodes_path, rows)
    (run / "archives" / "attack_instances.jsonl").unlink()

    result = exporter.export_run("legacy", run)
    failed = next(item for item in result["episodes"] if item["episode_id"] == "legitimate-failure")
    assert failed["status"] == "UNATTRIBUTED LEGACY FAILURE"
    assert failed["attribution_status"] == "legacy_unavailable"
    assert result["metrics"]["attributable_service_failure_rate"] is None
    assert result["metrics"]["strict_process_success"] is None
    assert result["metrics"]["unique_vulnerability_signatures_v2"] is None
    assert result["metrics"]["attack_instance_count"] is None
    assert result["failure_analysis"]["legitimate_failure_count"] == 0
    assert result["failure_analysis"]["legacy_unattributed_failure_count"] == 1


def test_customer_behavior_invalid_failure_is_not_a_service_failure(tmp_path: Path):
    run = make_run(tmp_path)
    episodes_path = run / "generations" / "gen_000" / "episodes.jsonl"
    rows = [json.loads(line) for line in episodes_path.read_text().splitlines()]
    failed = next(row for row in rows if row["episode_id"] == "legitimate-failure")
    failed["customer_behavior_valid"] = False
    failed["service_failure_attributable"] = False
    write_jsonl(episodes_path, rows)

    result = exporter.export_run("invalid-customer", run)
    failed_view = next(item for item in result["episodes"] if item["episode_id"] == "legitimate-failure")
    assert failed_view["status"] == "NON-ATTRIBUTABLE FAILURE"
    assert failed_view["validity_status"] == "customer_behavior_invalid"
    assert result["failure_analysis"]["legitimate_failure_count"] == 0


def test_demo_fixture_is_explicitly_demo():
    fixture = json.loads((Path(__file__).parents[1] / "dashboard" / "public" / "data" / "demo.json").read_text())
    assert fixture["dataset_kind"] == "DEMO"
    assert fixture["demo_notice"]
    assert all(run["metadata"]["source_kind"] == "DEMO" for run in fixture["runs"])
