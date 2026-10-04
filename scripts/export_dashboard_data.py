#!/usr/bin/env python3
"""Export read-only Customer-search artifacts for the research dashboard.

This script reads structured run artifacts only. It does not call a model,
evaluator, or scorer, and it never recomputes scores from dialogue text.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


HIDDEN_TRACE_KEYS = {
    "backend_record", "backend_final_state", "system_info", "hidden_state",
    "state_before", "state_after", "case_spec", "gold_answer", "expected_path",
}


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    result = []
    for line in lines:
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            result.append(value)
    return result


def _safe_trace_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _safe_trace_value(item)
            for key, item in value.items()
            if str(key).lower() not in HIDDEN_TRACE_KEYS
        }
    if isinstance(value, list):
        return [_safe_trace_value(item) for item in value]
    return value


def trace_events(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize already-structured trace events for display only."""
    metadata = row.get("metadata") or {}
    source = metadata.get("trace_events") or row.get("trace_events") or []
    events = []
    for index, event in enumerate(source):
        if not isinstance(event, dict):
            continue
        event_type = str(event.get("event_type") or event.get("kind") or "").upper()
        actor = str(event.get("actor") or "").lower()
        name = str(event.get("name") or "")
        payload = _safe_trace_value(event.get("payload", {}))
        if event_type in {"USER_MESSAGE", "CUSTOMER_MESSAGE"}:
            kind, label = "user", "Customer"
            if isinstance(payload, dict) and "text" in payload:
                payload = payload["text"]
        elif event_type in {"AGENT_MESSAGE", "SERVICE_MESSAGE"}:
            kind, label = "agent", "Service"
            if isinstance(payload, dict) and "text" in payload:
                payload = payload["text"]
        elif event_type in {"TOOL_CALL", "TOOL_QUERY", "ACTION_TOOL_CALL"}:
            kind, label = "tool_call", name or "Tool call"
        elif event_type in {"TOOL_RESULT", "TOOL_QUERY_RESULT", "ACTION_RESULT"}:
            kind, label = "backend_result", name or "Backend result"
        elif event_type in {"ACTION_EXECUTION", "ACTION_EXECUTED"}:
            kind, label = "action_result", name or "Action execution"
        elif event_type in {"STATE_CHANGE", "STATE_CHANGED"}:
            kind, label = "state_change", name or "State change"
        elif event_type in {"PROTOCOL_ERROR", "CUSTOMER_SIMULATOR_ATTEMPT"}:
            kind, label = "diagnostic", name or event_type.replace("_", " ").title()
        elif actor == "user":
            kind, label = "user", name or "Customer"
        elif actor == "agent":
            kind, label = "agent", name or "Service"
        else:
            continue
        events.append({
            "id": event.get("seq", index),
            "kind": kind,
            "label": label,
            "payload": payload,
            "turn": event.get("turn_index", event.get("turn")),
        })
    return events


def is_runtime_valid(row: dict[str, Any]) -> bool:
    metadata = row.get("metadata") or {}
    return (
        row.get("evaluation_status", "valid") == "valid"
        and row.get("protocol_valid", True) is not False
        and row.get("environment_valid", metadata.get("environment_valid", True)) is not False
        and not metadata.get("protocol_failure", False)
        and isinstance(row.get("task_success"), bool)
    )


def _mean(rows: list[dict[str, Any]], field: str) -> float | None:
    values = []
    for row in rows:
        value = row.get(field)
        if isinstance(value, bool):
            values.append(float(value))
        elif isinstance(value, (int, float)):
            values.append(float(value))
    return sum(values) / len(values) if values else None


def _generation_paths(run_dir: Path) -> list[Path]:
    paths = [path for path in (run_dir / "generations").glob("gen_*") if path.is_dir()]
    return sorted(paths, key=lambda path: int(path.name.split("_")[-1]))


def _generation_artifacts(directory: Path) -> dict[str, Any]:
    return {
        "proposals": read_json(directory / "proposals.json", {}) or {},
        "selection": read_json(directory / "selection.json", {}) or {},
        "summary": read_json(directory / "SUMMARY.json", {}) or {},
        "episodes": read_jsonl(directory / "episodes.jsonl"),
    }


def _source_kind(provenance: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    evaluator = str(provenance.get("evaluator") or "").lower()
    if evaluator == "mock":
        return "MOCK"
    if evaluator == "real":
        return "REAL"
    for row in rows:
        metadata = row.get("metadata") or {}
        if metadata.get("source_kind") in {"MOCK", "REAL"}:
            return str(metadata["source_kind"])
    return "UNKNOWN"


def _customer_generation(generation: int, artifacts: dict[str, Any]) -> dict[str, Any]:
    proposal_file = artifacts["proposals"]
    selection = artifacts["selection"]
    summary = artifacts["summary"]
    generation_record = proposal_file.get("generation_record") or {}
    selected_policy = selection.get("selected_policy") or {}
    candidate_scores = selection.get("candidate_scores") or []
    score_by_id = {
        item.get("policy_id"): item for item in candidate_scores if isinstance(item, dict)
    }
    policies = proposal_file.get("candidate_policies") or []
    candidate_records = generation_record.get("candidates") or []
    candidate_by_id = {
        item.get("policy_id"): item for item in candidate_records if isinstance(item, dict)
    }
    candidates = []
    for index, policy in enumerate(policies, start=1):
        if not isinstance(policy, dict):
            continue
        policy_id = policy.get("policy_id")
        record = candidate_by_id.get(policy_id, {})
        candidates.append({
            "candidate_index": index,
            "policy_id": policy_id,
            "strategy": policy.get("strategy"),
            "hypothesis": policy.get("hypothesis"),
            "provenance_hash": policy.get("provenance_hash"),
            "score": score_by_id.get(policy_id),
            "selected": policy_id == selection.get("selected_policy_id"),
            "proposal_status": record.get("proposal_status"),
        })
    return {
        "generation": generation,
        "candidate_count": len(policies),
        "requested_candidate_count": proposal_file.get("requested_candidate_count"),
        "selected_policy_id": selection.get("selected_policy_id") or summary.get("selected_policy_id"),
        "selected_policy": selected_policy,
        "selected_fitness": summary.get("selected_fitness"),
        "selected_official_task_success": summary.get("selected_official_task_success"),
        "incumbent_policy_id": selection.get("incumbent_policy_id") or summary.get("incumbent_policy_id"),
        "incumbent_retained": not bool(summary.get("customer_changed")),
        "customer_changed": summary.get("customer_changed"),
        "selection_reason": selection.get("selection_reason"),
        "scores": candidate_scores,
        "candidates": candidates,
        "generation_status": generation_record.get("status"),
    }


def _normalize_episode(row: dict[str, Any]) -> dict[str, Any]:
    valid = is_runtime_valid(row)
    if not valid:
        status = "INVALID EVALUATION"
    else:
        status = "VALID SUCCESS" if row.get("task_success") is True else "VALID FAILURE"
    metadata = row.get("metadata") or {}
    scores = {
        "execution_score": row.get("execution_score"),
        "sage_style_score": row.get("sage_style_score"),
        "verification_score": row.get("verification_score"),
        "policy_score": row.get("policy_score"),
        "action_execution_score": row.get("action_execution_score"),
        "goal_fulfillment_score": row.get("goal_fulfillment_score"),
    }
    return {
        "method": "customer_search",
        "episode_id": row.get("episode_id"),
        "scenario": row.get("scenario"),
        "case_id": row.get("case_id"),
        "generation": row.get("generation"),
        "split": row.get("split"),
        "customer_policy_id": row.get("customer_policy_id"),
        "service_policy_id": row.get("service_policy_id"),
        "task_success": row.get("task_success") if valid else None,
        "status": status,
        "validity_status": "valid" if valid else "invalid",
        "invalid_reason": row.get("invalid_reason") or (metadata.get("invalid_reason") if not valid else None),
        "error_types": list(row.get("error_types") or []),
        "predicted_action": row.get("predicted_action"),
        "executed_action": row.get("executed_action"),
        "termination_reason": row.get("termination_reason"),
        "tool_sequence": list(row.get("tool_sequence_summary") or []),
        "trace_events": trace_events(row),
        "trace_ref": row.get("trace_ref"),
        "scores": scores,
        "provenance": {
            "model": metadata.get("model_name"),
            "phase": metadata.get("phase"),
        },
    }


def _runtime_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    budget = metrics.get("request_budget") or {}
    attempts = metrics.get("provider_attempts")
    if attempts is None:
        attempts = budget.get("total_provider_attempts")
    evaluator = metrics.get("episode_evaluator") or {}
    evolver = metrics.get("customer_evolver") or {}
    successes = [item.get("successes") for item in (evaluator, evolver) if isinstance(item.get("successes"), (int, float))]
    reasoning = [item.get("reasoning_tokens") for item in (evaluator, evolver) if isinstance(item.get("reasoning_tokens"), (int, float))]
    evaluator_requests = evaluator.get("pipeline_requests")
    evolver_requests = evolver.get("requests")
    requests = None
    if isinstance(evaluator_requests, (int, float)) or isinstance(evolver_requests, (int, float)):
        requests = sum(
            value for value in (evaluator_requests, evolver_requests)
            if isinstance(value, (int, float))
        )
    if requests is None:
        for key in ("requests", "request_count", "total_requests"):
            values = [value.get(key) for value in (evaluator, evolver) if isinstance(value, dict)]
            if values and any(isinstance(value, (int, float)) for value in values):
                requests = sum(value for value in values if isinstance(value, (int, float)))
                break
    return {
        "requests": requests if requests is not None else attempts,
        "attempts": attempts,
        "successes": sum(successes) if successes else None,
        "input_tokens": metrics.get("input_tokens"),
        "output_tokens": metrics.get("output_tokens"),
        "tokens": (metrics.get("input_tokens") or 0) + (metrics.get("output_tokens") or 0),
        "reasoning_tokens": sum(reasoning) if reasoning else None,
        "latency_seconds": metrics.get("latency_seconds"),
        "average_attempt_latency": metrics.get("average_attempt_latency"),
        "max_attempt_latency": metrics.get("max_attempt_latency"),
        "retries": metrics.get("retries", budget.get("retries")),
        "timeouts": metrics.get("timeouts", 0),
        "provider_failures": metrics.get("provider_failures", metrics.get("failures", 0)),
    }


def export_run(label: str, run_dir: str | Path, model_override: str | None = None,
               provider_override: str | None = None) -> dict[str, Any]:
    root = Path(run_dir)
    config = read_json(root / "config" / "customer_search.json", {}) or {}
    provenance = read_json(root / "environment" / "provenance.json", {}) or {}
    run_status = read_json(root / "analysis" / "run_status.json", {}) or {}
    orchestration = read_json(root / "analysis" / "orchestration_metrics.json", {}) or {}
    generation_dirs = _generation_paths(root)
    generation_data = [
        (int(path.name.split("_")[-1]), path, _generation_artifacts(path))
        for path in generation_dirs
    ]
    raw_rows = [row for _, _, artifacts in generation_data for row in artifacts["episodes"]]
    episodes = [_normalize_episode(row) for row in raw_rows]
    valid_rows = [row for row in raw_rows if is_runtime_valid(row)]
    invalid_count = len(raw_rows) - len(valid_rows)

    latest_selection = generation_data[-1][2]["selection"] if generation_data else {}
    latest_summary = generation_data[-1][2]["summary"] if generation_data else {}
    selected_ids = set(latest_selection.get("selected_episode_ids") or [])
    selected_id = latest_selection.get("selected_policy_id") or latest_summary.get("selected_policy_id")
    selected_rows = [
        row for row in raw_rows
        if (selected_ids and row.get("episode_id") in selected_ids)
        or (not selected_ids and row.get("generation") == (generation_data[-1][0] if generation_data else None)
            and row.get("customer_policy_id") == selected_id)
    ]
    selected_valid = [row for row in selected_rows if is_runtime_valid(row)]
    generation_metrics = []
    customer_generations = []
    for generation, _, artifacts in generation_data:
        summary = artifacts["summary"]
        evaluation = summary.get("evolution_evaluation") or {}
        generation_metrics.append({
            "generation": generation,
            "task_success": summary.get("selected_official_task_success"),
            "fitness": summary.get("selected_fitness"),
            "valid_episode_count": evaluation.get("valid_episode_count"),
            "invalid_episode_count": evaluation.get("invalid_episode_count"),
        })
        customer_generations.append(_customer_generation(generation, artifacts))

    split_manifest = root / "split_manifest"
    manifest_present = all((split_manifest / f"{name}.json").exists() for name in (
        "evolution_cases", "validation_cases", "heldout_cases",
    ))
    evolution_manifest = read_json(split_manifest / "evolution_cases.json", {}) or {}
    evolution_cases = evolution_manifest.get("cases") or []
    artifacts_present = {
        "config": "confirmed" if (root / "config" / "customer_search.json").exists() else "missing",
        "split_manifest": "confirmed" if manifest_present else "missing",
        "generation_proposals": "confirmed" if generation_dirs and all((path / "proposals.json").exists() for path in generation_dirs) else "missing",
        "generation_selection": "confirmed" if generation_dirs and all((path / "selection.json").exists() for path in generation_dirs) else "missing",
        "episodes": "confirmed" if generation_dirs and all((path / "episodes.jsonl").exists() for path in generation_dirs) else "missing",
        "service_s0": "confirmed" if (root / "environment" / "initial_service_policy.json").exists() else "missing",
        "raw_traces": "confirmed" if any(item.get("trace_events") or item.get("trace_ref") for item in episodes) else "missing",
        "orchestration_metrics": "confirmed" if (root / "analysis" / "orchestration_metrics.json").exists() else "missing",
    }
    expected_freeze = provenance.get("expected_runtime_freeze_commit") or provenance.get("runtime_freeze_commit")
    observed_commit = provenance.get("runtime_commit") or provenance.get("commit_sha")
    expected_tag = provenance.get("expected_runtime_freeze_tag") or provenance.get("runtime_freeze_tag")
    observed_tag = provenance.get("runtime_tag") or provenance.get("freeze_tag")
    freeze_status = "not_confirmed"
    if expected_freeze:
        freeze_status = "matched" if expected_freeze == observed_commit else "mismatch"
    elif expected_tag:
        freeze_status = "matched" if expected_tag == observed_tag else "mismatch"
    source_kind = _source_kind(provenance, raw_rows)
    runtime_metrics = _runtime_metrics(orchestration)
    status = run_status.get("status") or orchestration.get("run_status")
    if not status and generation_dirs:
        status = "complete" if all((path / "COMPLETE.json").exists() for path in generation_dirs) else "incomplete"
    mode = "Customer Search"
    model_metadata = config.get("model_metadata") or {}
    model = model_override or provenance.get("model") or config.get("model") or model_metadata.get("model")
    provider = provider_override or provenance.get("provider") or config.get("provider") or model_metadata.get("provider")
    seed = provenance.get("seed", config.get("seed"))
    split_cfg = config.get("splits") or {}
    evaluation_cfg = config.get("evaluation") or {}
    customer_cfg = config.get("customer") or {}
    final_eval = latest_summary.get("evolution_evaluation") or {}
    final_success = latest_summary.get("selected_official_task_success")
    if final_success is None:
        final_success = latest_summary.get("selected_score", {}).get("official_task_success")
    metrics = {
        "task_success": final_success,
        "action_execution": _mean(selected_valid, "action_execution_score"),
        "goal_fulfillment": _mean(selected_valid, "goal_fulfillment_score"),
        "valid_episode_rate": len(valid_rows) / len(raw_rows) if raw_rows else None,
        "invalid_rate": invalid_count / len(raw_rows) if raw_rows else None,
        "invalid_episodes": invalid_count,
        "protocol_invalid_rate": sum(not row.get("protocol_valid", True) for row in raw_rows) / len(raw_rows) if raw_rows else None,
        "requests": runtime_metrics["requests"],
        "attempts": runtime_metrics["attempts"],
        "successes": runtime_metrics["successes"],
        "input_tokens": runtime_metrics["input_tokens"],
        "output_tokens": runtime_metrics["output_tokens"],
        "tokens": runtime_metrics["tokens"],
        "reasoning_tokens": runtime_metrics["reasoning_tokens"],
        "latency_seconds": runtime_metrics["latency_seconds"],
        "average_attempt_latency": runtime_metrics["average_attempt_latency"],
        "max_attempt_latency": runtime_metrics["max_attempt_latency"],
        "retries": runtime_metrics["retries"],
        "timeouts": runtime_metrics["timeouts"],
        "provider_failures": runtime_metrics["provider_failures"],
    }
    return {
        "id": label,
        "metadata": {
            "method": "customer_search",
            "mode": mode,
            "source_kind": source_kind,
            "model": model,
            "provider": provider,
            "seed": seed,
            "case_count": len(evolution_cases),
            "generations": len(generation_dirs),
            "requested_generations": config.get("max_generations") or provenance.get("generation_count"),
            "run_status": status or "not_recorded",
            "run_dir": str(root),
            "runtime_commit": observed_commit,
            "runtime_tag": observed_tag,
            "runtime_freeze_commit": expected_freeze,
            "runtime_freeze_tag": expected_tag,
            "runtime_freeze_match_status": freeze_status,
            "formal_protocol_commit": provenance.get("formal_protocol_commit"),
            "formal_protocol_tag": provenance.get("formal_protocol_tag"),
            "split_strategy": provenance.get("split_strategy") or split_cfg.get("strategy"),
            "split_seed": provenance.get("split_seed", split_cfg.get("seed")),
            "max_turns": provenance.get("max_turns", evaluation_cfg.get("max_turns")),
            "fixed_service_policy_id": provenance.get("service_policy_id", "service_policy_s0"),
            "candidate_count": customer_cfg.get("candidate_count", provenance.get("customer_candidate_count")),
            "validity_schema_status": "current",
            "request_count_semantics": "provider attempts when successful-request count is unavailable",
        },
        "metrics": metrics,
        "generation_metrics": generation_metrics,
        "customer_candidates": customer_generations,
        "episodes": episodes,
        "completeness": {
            "split_manifest": manifest_present,
            "valid_invalid_separated": all(item["validity_status"] in {"valid", "invalid"} for item in episodes) if episodes else None,
            "candidate_provenance": all((path / "proposals.json").exists() and (path / "selection.json").exists() for path in generation_dirs) if generation_dirs else None,
            "traces": artifacts_present["raw_traces"] == "confirmed",
            "heldout_not_used_by_evolver": provenance.get("heldout_used_for_adaptation_or_selection") is False,
            "artifacts": artifacts_present,
        },
        "notes": [
            "Customer-search run against fixed Service S0; no Service evolution is performed.",
            "Official scores are copied from structured evaluator artifacts; no dialogue is rescored.",
            "Generations are trajectory steps, not independent statistical samples.",
        ],
    }


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("run must be LABEL=PATH")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("run must be LABEL=PATH")
    return label, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=parse_run, required=True, metavar="LABEL=PATH")
    parser.add_argument("--model")
    parser.add_argument("--provider")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = {
        "schema_version": 2,
        "dataset_kind": "EXPORTED_ARTIFACTS",
        "read_only": True,
        "scoring_recomputed": False,
        "runs": [
            export_run(label, path, args.model, args.provider)
            for label, path in args.run
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {len(output['runs'])} run(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
