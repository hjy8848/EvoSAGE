#!/usr/bin/env python3
"""Export EvoSAGE artifacts into a read-only dashboard dataset.

This exporter is deliberately separate from the benchmark runtime. It reads
structured JSON/JSONL files, normalizes them for the dashboard, and never
recomputes an evaluation score from raw dialogue. Trace files are used only to
provide provenance and a display timeline.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


FREEZE_COMMIT = "4dfc56d6c92fff86f8310e7fcf2fda12b41df42c"
FREEZE_TAG = "exp-freeze-2026-09-22"
INVALID_TERMS = {
    "json_parse_failed",
    "json_parse_error",
    "provider_error",
    "provider_failure",
    "timeout",
    "timed_out",
    "output_truncated",
    "protocol_failure",
}


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return rows
    for line in lines:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def first_value(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value not in (None, "", []):
            return value
    return default


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def generation_number(path: Path) -> int:
    match = re.search(r"gen_(\d+)", path.name)
    return int(match.group(1)) if match else 0


def load_generations(run_dir: Path) -> List[int]:
    values = []
    for path in sorted((run_dir / "generations").glob("gen_*")):
        if path.is_dir():
            values.append(generation_number(path))
    return sorted(set(values))


def read_generation_file(run_dir: Path, generation: int, name: str, default: Any) -> Any:
    return read_json(run_dir / "generations" / f"gen_{generation:03d}" / name, default)


def contains_mock(rows: Iterable[Dict[str, Any]]) -> bool:
    for row in rows:
        metadata = row.get("metadata") or {}
        if metadata.get("mock") is True or metadata.get("source_kind") == "MOCK":
            return True
        if "mock" in str(metadata).lower():
            return True
    return False


def infer_provider(config: Dict[str, Any], provenance: Dict[str, Any], model: str) -> str:
    explicit = first_value(
        config.get("provider"),
        config.get("model_provider"),
        provenance.get("provider"),
        provenance.get("provider_name"),
    )
    if explicit:
        return str(explicit)
    url = str(first_value(config.get("api_url"), provenance.get("api_url"), default=""))
    if "inferai" in url.lower():
        return "InferAI"
    if "openai" in url.lower():
        return "OpenAI-compatible"
    if str(model).lower().startswith("dashscope/"):
        return "DashScope-compatible"
    return "Unknown"


def infer_model(config: Dict[str, Any], provenance: Dict[str, Any], traces: List[Dict[str, Any]]) -> str:
    model_metadata = config.get("model_metadata") or {}
    for trace in traces:
        simulation = trace.get("simulation") or {}
        value = first_value(simulation.get("model_name"), trace.get("model"))
        if value:
            return str(value)
    value = first_value(
        config.get("model"),
        config.get("model_name"),
        model_metadata.get("model"),
        provenance.get("model"),
        default="Not recorded",
    )
    return str(value)


def trace_events_from_record(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Create display-only events from a real trace; this never scores it."""
    simulation = record.get("simulation") or {}
    events: List[Dict[str, Any]] = []
    event_index = 0

    def append(kind: str, label: str, payload: Any = None, turn: Any = None) -> None:
        nonlocal event_index
        events.append({
            "id": f"trace-{event_index}",
            "kind": kind,
            "label": label,
            "payload": payload,
            "turn": turn,
        })
        event_index += 1

    for turn in simulation.get("turns", []) or []:
        turn_id = turn.get("turn_id")
        if turn.get("user_message"):
            append("user", "USER", turn.get("user_message"), turn_id)
        output = turn.get("agent_output") or {}
        if output.get("chat"):
            append("agent", "AGENT", output.get("chat"), turn_id)
        for call in (turn.get("tool_calls") or output.get("tool_calls") or []):
            if isinstance(call, dict):
                append("tool_call", "TOOL CALL", {
                    "name": first_value(call.get("name"), call.get("tool_name"), default="unknown"),
                    "arguments": call.get("arguments", call.get("args", {})),
                }, turn_id)
        for result in (turn.get("tool_results") or output.get("tool_results") or []):
            if isinstance(result, dict):
                append("backend_result", "BACKEND RESULT", result, turn_id)
        for result in (turn.get("action_results") or output.get("action_results") or []):
            if isinstance(result, dict):
                append("state_change", "STATE CHANGE", result, turn_id)
    for event in record.get("backend_events", []) or []:
        if not isinstance(event, dict):
            continue
        event_type = event.get("event_type") or "backend_result"
        if event_type == "tool_call":
            append("tool_call", "TOOL CALL", event, event.get("turn_index"))
        elif event_type == "state_change":
            append("state_change", "STATE CHANGE", event, event.get("turn_index"))
        else:
            append("backend_result", "BACKEND RESULT", event, event.get("turn_index"))
    return events


def load_trace_records(run_dir: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for directory_name in ("real_traces", "traces"):
        directory = run_dir / directory_name
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*.jsonl")):
            records.extend(read_jsonl(path))
    return records


def trace_index(records: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    for record in records:
        simulation = record.get("simulation") or {}
        keys = [
            simulation.get("simulation_id"),
            simulation.get("case_id"),
            (record.get("evaluation") or {}).get("simulation_id"),
        ]
        payload = {
            "trace_file": record.get("trace_file"),
            "trace_events": trace_events_from_record(record),
            "simulation_id": simulation.get("simulation_id"),
            "model": simulation.get("model_name"),
            "adversarial_intensity": simulation.get("adversarial_intensity"),
        }
        for key in keys:
            if key:
                indexed[str(key)] = payload
    return indexed


def all_episode_rows(run_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted((run_dir / "generations").glob("gen_*/episodes.jsonl")):
        for row in read_jsonl(path):
            row = dict(row)
            row.setdefault("generation", generation_number(path.parent))
            rows.append(row)
    return rows


def error_values(row: Dict[str, Any]) -> List[str]:
    values = list(row.get("error_types") or [])
    diagnostics = row.get("diagnostics") or {}
    values.extend(diagnostics.get("error_categories") or [])
    metadata = row.get("metadata") or {}
    if metadata.get("json_parse_failed") is True:
        values.append("json_parse_failed")
    if metadata.get("protocol_failure") is True:
        values.append("protocol_failure")
    return sorted(set(str(value) for value in values if value))


def invalid_reason(row: Dict[str, Any]) -> Optional[str]:
    if row.get("evaluation_status") == "invalid":
        return str(row.get("invalid_reason") or "evaluation_invalid")
    metadata = row.get("metadata") or {}
    if metadata.get("protocol_failure"):
        return str(metadata.get("invalid_reason") or "protocol_failure")
    if metadata.get("json_parse_failed"):
        return "json_parse_failed"
    for value in error_values(row):
        lowered = value.lower()
        if lowered in INVALID_TERMS or any(term in lowered for term in ("timeout", "provider", "truncat", "parse")):
            return value
    return None


def sanitize_case_metadata(row: Dict[str, Any]) -> Dict[str, Any]:
    metadata = row.get("case_metadata") or {}
    if not metadata:
        metadata = {
            "case_id": row.get("case_id"),
            "scenario": row.get("scenario"),
            "split": row.get("split"),
        }
    # Do not export hidden backend records into the browser dataset. The
    # dashboard needs case identity and outcome labels, not evaluator secrets.
    return {
        key: value
        for key, value in metadata.items()
        if key not in {"backend_record", "backend_system_variables", "system_info"}
    }


def normalize_episode(row: Dict[str, Any], traces: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    episode_id = str(first_value(row.get("episode_id"), row.get("id"), default="episode-unknown"))
    reason = invalid_reason(row)
    status = "INVALID EVALUATION" if reason else ("VALID SUCCESS" if row.get("task_success") else "LEGITIMATE FAILURE")
    trace = first_value(traces.get(episode_id), traces.get(str(row.get("case_id"))))
    trace = trace or {}
    tool_sequence = row.get("tool_sequence_summary") or []
    events = row.get("trace_events") or trace.get("trace_events") or []
    return {
        "episode_id": episode_id,
        "generation": safe_int(row.get("generation"), 0),
        "phase": row.get("phase") or row.get("metadata", {}).get("phase") or "unknown",
        "split": row.get("split") or "unknown",
        "case_id": row.get("case_id") or "unknown",
        "scenario": row.get("scenario") or "unknown",
        "customer_policy_id": row.get("customer_policy_id"),
        "service_policy_id": row.get("service_policy_id"),
        "status": status,
        "evaluation_status": row.get("evaluation_status") or ("invalid" if reason else "valid"),
        "invalid_reason": reason,
        "task_success": bool(row.get("task_success")) if reason is None else None,
        "expected_action": first_value(row.get("expected_action"), (row.get("finals") or {}).get("Action")),
        "predicted_action": row.get("predicted_action"),
        "executed_action": row.get("executed_action"),
        "termination_reason": row.get("termination_reason"),
        "error_types": error_values(row),
        "sop_node": row.get("sop_node"),
        "path_step_index": row.get("path_step_index"),
        "tool_sequence": tool_sequence,
        "scores": {
            "execution": safe_float(row.get("execution_score")),
            "sage_style": safe_float(row.get("sage_style_score")),
            "verification": safe_float(row.get("verification_score")),
            "policy": safe_float(row.get("policy_score")),
            "action_execution": safe_float(row.get("action_execution_score")),
            "goal_fulfillment": safe_float(row.get("goal_fulfillment_score")),
        },
        "failure_signature": row.get("failure_signature") or {},
        "case_metadata": sanitize_case_metadata(row),
        "trace_ref": first_value(row.get("trace_ref"), trace.get("trace_file")),
        "trace_events": events,
        "provenance": {
            "source": "structured episode artifact",
            "trace_available": bool(events),
            "trace_simulation_id": trace.get("simulation_id"),
            "model": trace.get("model"),
            "adversarial_intensity": trace.get("adversarial_intensity"),
        },
    }


def group_generation_metrics(episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        grouped[int(episode.get("generation", 0))].append(episode)
    result = []
    for generation in sorted(grouped):
        rows = grouped[generation]
        valid = [row for row in rows if row.get("evaluation_status") == "valid"]
        failures = [row for row in valid if not row.get("task_success")]
        signatures = {
            (row.get("failure_signature") or {}).get("signature_id")
            for row in failures
            if (row.get("failure_signature") or {}).get("signature_id")
        }
        result.append({
            "generation": generation,
            "episodes": len(rows),
            "valid_rate": len(valid) / len(rows) if rows else None,
            "task_success": sum(bool(row.get("task_success")) for row in valid) / len(valid) if valid else None,
            "action_execution": _mean_score(valid, "action_execution"),
            "goal_fulfillment": _mean_score(valid, "goal_fulfillment"),
            "unique_failure_signatures": len(signatures),
            "legitimate_failures": len(failures),
        })
    return result


def _mean_score(rows: List[Dict[str, Any]], name: str) -> Optional[float]:
    values = [row.get("scores", {}).get(name) for row in rows]
    values = [float(value) for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _metric_mean(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
    values = [row.get(key) for row in rows if row.get(key) is not None]
    return sum(float(value) for value in values) / len(values) if values else None


def summarize_metrics(episodes: List[Dict[str, Any]], metrics: Dict[str, Any]) -> Dict[str, Any]:
    valid = [row for row in episodes if row.get("evaluation_status") == "valid"]
    invalid = [row for row in episodes if row.get("evaluation_status") != "valid"]
    failures = [row for row in valid if not row.get("task_success")]
    signatures = {
        (row.get("failure_signature") or {}).get("signature_id")
        for row in failures
        if (row.get("failure_signature") or {}).get("signature_id")
    }
    return {
        "episodes": len(episodes),
        "valid_episodes": len(valid),
        "invalid_episodes": len(invalid),
        "valid_episode_rate": len(valid) / len(episodes) if episodes else None,
        "invalid_rate": len(invalid) / len(episodes) if episodes else None,
        "task_success": _metric_mean(valid, "task_success"),
        "action_execution": _mean_score(valid, "action_execution"),
        "goal_fulfillment": _mean_score(valid, "goal_fulfillment"),
        "legitimate_failures": len(failures),
        "unique_failure_signatures": len(signatures),
        "requests": first_value(metrics.get("llm_requests"), metrics.get("pipeline_requests")),
        "input_tokens": metrics.get("input_tokens"),
        "output_tokens": metrics.get("output_tokens"),
        "tokens": (
            safe_int(metrics.get("input_tokens"), 0) + safe_int(metrics.get("output_tokens"), 0)
            if metrics else None
        ),
        "latency_seconds": metrics.get("latency_seconds"),
        "timeouts": metrics.get("timeouts"),
        "provider_failures": metrics.get("failures"),
        "retries": metrics.get("retries"),
    }


def candidate_data(run_dir: Path, generations: List[int]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    customer: List[Dict[str, Any]] = []
    service: List[Dict[str, Any]] = []
    for generation in generations:
        customer_file = read_generation_file(run_dir, generation, "customer_candidates.json", {}) or {}
        selected = customer_file.get("selected_policy") or {}
        scores = customer_file.get("scores") or []
        customer.append({
            "generation": generation,
            "selected_policy": selected,
            "candidate_count": len(scores),
            "scores": scores,
            "source_failure_ids": selected.get("source_failure_ids", []),
        })
        gate = read_generation_file(run_dir, generation, "service_gate.json", {}) or {}
        candidates = gate.get("candidates") or []
        if not candidates and any(key in gate for key in ("accepted", "reason", "patch")):
            candidates = [{
                "accepted": gate.get("accepted"),
                "delta": gate.get("delta"),
                "patch": gate.get("patch"),
                "reason": gate.get("reason"),
                "evaluation_status": gate.get("evaluation_status", "valid"),
                "metrics": gate.get("metrics", {}),
            }]
        service.append({
            "generation": generation,
            "accepted": gate.get("accepted"),
            "delta": gate.get("delta"),
            "reason": gate.get("reason"),
            "candidates": candidates,
            "candidate_count": len(candidates),
            "provenance_present": bool(candidates) and all(
                isinstance(item, dict) and (
                    item.get("patch") is not None or item.get("candidate_policy") is not None
                ) for item in candidates
            ),
        })
    return customer, service


def completeness(run_dir: Path, episodes: List[Dict[str, Any]], service: List[Dict[str, Any]]) -> Dict[str, Any]:
    metrics_path = run_dir / "analysis" / "orchestration_metrics.json"
    split_files = [
        run_dir / "split_manifest" / "evolution_cases.json",
        run_dir / "split_manifest" / "validation_cases.json",
        run_dir / "split_manifest" / "heldout_cases.json",
    ]
    trace_files = [
        path for directory_name in ("real_traces", "traces")
        for path in (run_dir / directory_name).rglob("*")
        if path.is_file()
    ] if any((run_dir / name).exists() for name in ("real_traces", "traces")) else []
    return {
        "metrics": metrics_path.exists(),
        "split_manifest": all(path.exists() for path in split_files),
        "episodes": bool(episodes),
        "traces": bool(trace_files) or any(ep.get("trace_events") for ep in episodes),
        "candidate_provenance": all(item.get("provenance_present") for item in service) if service else None,
        "valid_invalid_separated": bool(episodes) and any(
            ep.get("evaluation_status") == "valid" for ep in episodes
        ) and any(ep.get("evaluation_status") != "valid" for ep in episodes) or bool(episodes),
        "heldout_not_used_by_evolver": None,
    }


def export_run(label: str, run_dir: Path, model_override: Optional[str] = None,
               provider_override: Optional[str] = None) -> Dict[str, Any]:
    config = read_json(run_dir / "config" / "evolution.json", {}) or {}
    provenance = read_json(run_dir / "environment" / "provenance.json", {}) or {}
    metrics = read_json(run_dir / "analysis" / "orchestration_metrics.json", {}) or {}
    raw_episodes = all_episode_rows(run_dir)
    raw_trace_records = load_trace_records(run_dir)
    traces = trace_index(raw_trace_records)
    episodes = [normalize_episode(row, traces) for row in raw_episodes]
    generations = load_generations(run_dir)
    customer, service = candidate_data(run_dir, generations)
    frontier = read_json(run_dir / "analysis" / "weakness_frontier.json", []) or []
    model = model_override or infer_model(config, provenance, raw_trace_records)
    is_mock = contains_mock(raw_episodes) or "mock" in json.dumps(provenance, ensure_ascii=False).lower()
    source_kind = "MOCK" if is_mock else "REAL"
    metadata = {
        "mode": label or config.get("experiment_mode", "unknown"),
        "source_kind": source_kind,
        "model": model,
        "provider": provider_override or infer_provider(config, provenance, model),
        "seed": first_value(config.get("seed"), (config.get("splits") or {}).get("seed")),
        "commit_sha": first_value(provenance.get("commit_sha"), provenance.get("git_commit"), default=FREEZE_COMMIT),
        "freeze_tag": first_value(provenance.get("freeze_tag"), default=FREEZE_TAG),
        "generations": first_value(config.get("max_generations"), len(generations), default=0),
        "run_dir": str(run_dir),
        "scenario": config.get("scenario") or provenance.get("scenario"),
        "split_strategy": (config.get("splits") or {}).get("strategy"),
        "max_cases": (config.get("splits") or {}).get("max_cases"),
        "max_turns": (config.get("evaluation") or {}).get("max_turns"),
        "judge_in_evolution": (config.get("evaluation") or {}).get("judge_in_evolution"),
        "candidate_counts": {
            "customer": (config.get("customer") or {}).get("candidate_count"),
            "service": (config.get("service") or {}).get("candidate_count"),
        },
    }
    all_signatures = {
        (ep.get("failure_signature") or {}).get("signature_id")
        for ep in episodes
        if (ep.get("failure_signature") or {}).get("signature_id")
    }
    if not all_signatures:
        for item in read_jsonl(run_dir / "archives" / "attacks.jsonl"):
            signature = item.get("failure_signature") or {}
            if signature.get("signature_id"):
                all_signatures.add(signature["signature_id"])
    return {
        "id": label or run_dir.name,
        "metadata": metadata,
        "metrics": summarize_metrics(episodes, metrics),
        "generation_metrics": group_generation_metrics(episodes),
        "episodes": episodes,
        "customer_candidates": customer,
        "service_candidates": service,
        "weakness_frontier": frontier,
        "failure_signatures": sorted(all_signatures),
        "heldout": read_json(run_dir / "analysis" / "heldout_results.json", None),
        "fresh_adversary": [
            read_json(path, {}) for path in sorted((run_dir / "analysis").glob("fresh_adversary_*.json"))
        ],
        "completeness": completeness(run_dir, episodes, service),
        "notes": [
            "Scores and statuses are read from structured artifacts.",
            "Trace data is display-only provenance; raw dialogue is not re-scored.",
        ],
    }


def parse_run(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must use MODE=RUN_DIR")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("--run must use MODE=RUN_DIR")
    return label, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", action="append", required=True, type=parse_run,
        help="run mapping in the form mode=run_dir; repeat for each experiment",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", help="optional model label when the run config does not record it")
    parser.add_argument("--provider", help="optional provider label when the run config does not record it")
    parser.add_argument("--freeze-commit", default=FREEZE_COMMIT)
    parser.add_argument("--freeze-tag", default=FREEZE_TAG)
    args = parser.parse_args()
    runs = [export_run(label, path, model_override=args.model, provider_override=args.provider) for label, path in args.run]
    payload = {
        "schema_version": 1,
        "dataset_kind": "EXPORTED_ARTIFACTS",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "freeze": {"commit_sha": args.freeze_commit, "tag": args.freeze_tag},
        "read_only": True,
        "scoring_recomputed": False,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Exported {len(runs)} run(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
