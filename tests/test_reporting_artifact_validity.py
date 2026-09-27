import importlib.util
import json
from pathlib import Path

from framework.evolution.reporting import generate_report, summarize_episode_reporting


ROOT = Path(__file__).parents[1]


def _load_script(name):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analyzer = _load_script("analyze_evolution_trajectory.py")
pilot_summary = _load_script("summarize_pilot_runs.py")


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_reporting_separates_protocol_customer_and_service_attribution():
    rows = [
        {
            "evaluation_status": "valid",
            "protocol_valid": True,
            "environment_valid": True,
            "customer_behavior_valid": True,
            "service_failure_attributable": False,
            "task_success": True,
            "strict_process_success": True,
            "eventual_goal_success": 1.0,
            "recovery_success": False,
            "vulnerability_signature": None,
        },
        {
            "evaluation_status": "valid",
            "protocol_valid": True,
            "environment_valid": True,
            "customer_behavior_valid": True,
            "service_failure_attributable": True,
            "task_success": False,
            "strict_process_success": False,
            "eventual_goal_success": 0.0,
            "recovery_success": True,
            "vulnerability_signature": {"schema_version": 2, "signature_id": "sig-v2"},
        },
        {
            "evaluation_status": "valid",
            "protocol_valid": True,
            "environment_valid": True,
            "customer_behavior_valid": False,
            "service_failure_attributable": False,
            "task_success": False,
            "strict_process_success": False,
            "eventual_goal_success": 0.0,
            "recovery_success": False,
            "vulnerability_signature": {"schema_version": 2, "signature_id": "must-not-count"},
        },
        {
            "evaluation_status": "invalid",
            "protocol_valid": False,
            "environment_valid": True,
            "customer_behavior_valid": True,
            "service_failure_attributable": False,
            "task_success": False,
            "vulnerability_signature": None,
        },
    ]

    metrics = summarize_episode_reporting(rows, attack_instance_count=4)
    assert metrics["valid_episodes"] == 3
    assert metrics["protocol_invalid_rate"] == 0.25
    assert metrics["customer_behavior_invalid_rate"] == 1 / 3
    assert metrics["attributable_service_failure_rate"] == 0.5
    assert metrics["task_success"] == 0.5
    assert metrics["strict_process_success"] == 0.5
    assert metrics["eventual_goal_success"] == 0.5
    assert metrics["recovery_success_rate"] == 0.5
    assert metrics["unique_vulnerability_signatures_v2"] == 1
    assert metrics["attack_instance_count"] == 4


def test_legacy_reporting_metrics_remain_unavailable_not_zero(tmp_path):
    row = {"evaluation_status": "valid", "task_success": False, "failure_signature": {"signature_id": "v1"}}
    metrics = summarize_episode_reporting([row])
    assert metrics["attributable_service_failure_rate"] is None
    assert metrics["customer_behavior_invalid_rate"] is None
    assert metrics["strict_process_success"] is None
    assert metrics["unique_vulnerability_signatures_v2"] is None
    generation = tmp_path / "generations" / "gen_000"
    _json(generation / "COMPLETE.json", {"generation": 0})
    _json(generation / "customer_candidates.json", {"scores": []})
    _jsonl(generation / "episodes.jsonl", [row])
    report = generate_report(tmp_path)
    report_text = report.read_text(encoding="utf-8")
    assert "N/A (legacy schema)" in report_text
    assert "Unique V2 vulnerability signatures: N/A (legacy schema)" in report_text


def test_partial_validity_provenance_does_not_dilute_rates():
    metrics = summarize_episode_reporting([
        {"evaluation_status": "valid", "protocol_valid": True, "customer_behavior_valid": True},
        {"evaluation_status": "valid", "protocol_valid": True, "customer_behavior_valid": False},
        {"task_success": True},
    ])
    assert metrics["protocol_evaluated_episodes"] == 2
    assert metrics["protocol_invalid_rate"] == 0.0
    assert metrics["customer_behavior_evaluated_episodes"] == 2
    assert metrics["customer_behavior_invalid_rate"] == 0.5


def test_default_mock_outcome_fields_are_not_treated_as_evaluated_metrics():
    row = {
        "evaluation_status": "valid",
        "protocol_valid": True,
        "environment_valid": True,
        "customer_behavior_valid": True,
        "service_failure_attributable": False,
        "task_success": True,
        "strict_process_success": None,
        "eventual_goal_success": 0.0,
        "recovery_success": False,
        "vulnerability_signature": None,
    }
    metrics = summarize_episode_reporting([row])
    assert metrics["outcome_metrics_status"] == "not_evaluated"
    assert metrics["strict_process_success"] is None
    assert metrics["eventual_goal_success"] is None
    assert metrics["recovery_success_rate"] is None
    assert metrics["attack_instance_count"] == 0


def test_report_shows_paired_gate_evidence_and_staged_not_evaluated(tmp_path):
    generation = tmp_path / "generations" / "gen_000"
    _json(generation / "COMPLETE.json", {"generation": 0})
    _json(generation / "customer_candidates.json", {"scores": []})
    _jsonl(generation / "episodes.jsonl", [])
    _json(
        generation / "service_gate.json",
        {
            "candidates": [
                {
                    "patch_id": "patch-latest",
                    "accepted": False,
                    "evaluation_status": "valid",
                    "reason": "latest_attack_filter:insufficient_paired_wins",
                    "metrics": {"latest_paired": {"wins": 0, "losses": 1, "ties": 0}},
                },
                {
                    "patch_id": "patch-full",
                    "accepted": True,
                    "evaluation_status": "valid",
                    "reason": "accepted",
                    "delta": 0.2,
                    "metrics": {
                        "latest_paired": {"wins": 2, "losses": 0, "ties": 1},
                        "exact_replay_paired": {"wins": 1, "losses": 0, "ties": 0},
                        "transfer_replay_paired": {"wins": 0, "losses": 0, "ties": 1},
                        "normal_paired": {"wins": 0, "losses": 0, "ties": 2},
                    },
                },
            ]
        },
    )

    report_text = generate_report(tmp_path).read_text(encoding="utf-8")
    assert "Service candidate paired evidence" in report_text
    assert "0/1/0" in report_text
    assert "Not evaluated" in report_text
    assert "2/0/1" in report_text
    assert "1/0/0" in report_text


def test_pilot_summary_exposes_accepted_candidate_paired_counts(tmp_path):
    generation = tmp_path / "generations" / "gen_000"
    _jsonl(generation / "episodes.jsonl", [])
    _json(
        generation / "service_gate.json",
        {
            "patch": {"patch_id": "p1"},
            "candidates": [
                {
                    "patch_id": "p1",
                    "accepted": True,
                    "metrics": {
                        "latest_task_success": 0.8,
                        "latest_paired": {"wins": 2, "losses": 1, "ties": 0},
                    },
                }
            ],
        },
    )
    summary = pilot_summary.summarize("coevolution", tmp_path)
    assert summary["latest_paired_wlt"] == "2/1/0"
    assert summary["exact_replay_paired_wlt"] is None
    assert summary["paired_outcomes"]["latest"]["wins"] == 2


def test_trajectory_analyzer_uses_boolean_outcomes_and_v2_attribution(tmp_path):
    run = tmp_path / "run"
    rows = [
        {
            "evaluation_status": "valid", "protocol_valid": True, "environment_valid": True,
            "customer_behavior_valid": True, "service_failure_attributable": False,
            "task_success": True, "action_execution_score": 1.0, "goal_fulfillment_score": 1.0,
            "vulnerability_signature": None,
        },
        {
            "evaluation_status": "valid", "protocol_valid": True, "environment_valid": True,
            "customer_behavior_valid": True, "service_failure_attributable": True,
            "task_success": False, "action_execution_score": 0.0, "goal_fulfillment_score": 0.0,
            "vulnerability_signature": {"schema_version": 2, "signature_id": "sig"},
        },
        {
            "evaluation_status": "invalid", "protocol_valid": False, "task_success": False,
        },
    ]
    _jsonl(run / "generations" / "gen_000" / "episodes.jsonl", rows)

    result = analyzer.analyze("test", run)["episode_metrics"][0]
    assert result["task_success"] == 0.5
    assert result["legitimate_failure_count"] == 1
    assert result["unique_failure_signatures"] == 1
    assert result["invalid_episodes"] == 1
    assert analyzer.mean([{"x": True}, {"x": False}], "x") == 0.5


def test_pilot_summary_does_not_call_legacy_failures_attributable(tmp_path):
    run = tmp_path / "legacy"
    _jsonl(
        run / "generations" / "gen_000" / "episodes.jsonl",
        [{"evaluation_status": "valid", "task_success": False, "failure_signature": {"signature_id": "v1"}}],
    )
    result = pilot_summary.summarize("static", run)
    assert result["legitimate_failures"] is None
    assert result["unique_failure_signatures"] is None
    assert result["task_success"] == 0.0
