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
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


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


def normalized_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def artifact_status(path: Path, *, applicable: bool = True) -> str:
    if not applicable:
        return "not_applicable"
    return "confirmed" if path.exists() else "missing"


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

    def public_result(value: Any) -> Any:
        """Keep displayable tool output while excluding hidden backend state."""
        if not isinstance(value, dict):
            return value
        hidden = {"state_before", "state_after", "backend_final_state", "backend_record", "system_info"}
        return {key: item for key, item in value.items() if key not in hidden}

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
                append("backend_result", "BACKEND RESULT", public_result(result), turn_id)
        for result in (turn.get("action_results") or output.get("action_results") or []):
            if isinstance(result, dict):
                append("state_change", "STATE CHANGE", public_result(result), turn_id)

    # Some trace writers store backend events under simulation rather than at
    # the record root. Turn-level calls/results are preferred because they
    # preserve the actual interaction order and do not duplicate events.
    has_tool_events = any(event.get("kind") in {"tool_call", "backend_result"} for event in events)
    backend_events = record.get("backend_events") or simulation.get("backend_events") or []
    if not has_tool_events:
        for event in backend_events:
            if not isinstance(event, dict):
                continue
            event_type = event.get("event_type") or "backend_result"
            if event_type == "tool_call":
                append("tool_call", "TOOL CALL", {
                    "name": first_value(event.get("name"), event.get("tool_name"), default="unknown"),
                    "arguments": event.get("arguments") if "arguments" in event else None,
                }, event.get("turn_index"))
                result = event.get("result")
                if result is not None:
                    append("backend_result", "BACKEND RESULT", public_result(result), event.get("turn_index"))
            elif event_type == "state_change":
                append("state_change", "STATE CHANGE", public_result(event), event.get("turn_index"))
            else:
                append("backend_result", "BACKEND RESULT", public_result(event), event.get("turn_index"))
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
        case_spec = simulation.get("case_spec") or {}
        keys = [
            simulation.get("simulation_id"),
            simulation.get("case_id"),
            case_spec.get("case_id"),
            (record.get("evaluation") or {}).get("simulation_id"),
        ]
        payload = {
            "trace_file": record.get("trace_file"),
            "trace_events": trace_events_from_record(record),
            "simulation_id": simulation.get("simulation_id"),
            "model": simulation.get("model_name"),
            "adversarial_intensity": simulation.get("adversarial_intensity"),
            "duration_seconds": safe_float(
                first_value(
                    simulation.get("duration_seconds"),
                    simulation.get("latency_seconds"),
                    record.get("duration_seconds"),
                )
            ),
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


def protocol_invalid_reason(row: Dict[str, Any]) -> Optional[str]:
    reason = invalid_reason(row)
    return reason if reason else None


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
        "latency_seconds": safe_float(
            first_value(
                row.get("latency_seconds"),
                row.get("duration_seconds"),
                trace.get("duration_seconds"),
            )
        ),
        "trace_ref": first_value(row.get("trace_ref"), trace.get("trace_file")),
        "trace_events": events,
        "provenance": {
            "source": "structured episode artifact",
            "trace_available": bool(events),
            "trace_simulation_id": trace.get("simulation_id"),
            "model": trace.get("model"),
            "adversarial_intensity": trace.get("adversarial_intensity"),
        },
        "protocol_invalid_reason": protocol_invalid_reason(row),
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
        "legitimate_failure_rate": len(failures) / len(valid) if valid else None,
        "unique_failure_signatures": len(signatures),
        "requests": first_value(metrics.get("llm_requests"), metrics.get("pipeline_requests")),
        "input_tokens": metrics.get("input_tokens"),
        "output_tokens": metrics.get("output_tokens"),
        "reasoning_tokens": metrics.get("reasoning_tokens"),
        "tokens": (
            safe_int(metrics.get("input_tokens"), 0) + safe_int(metrics.get("output_tokens"), 0)
            if metrics else None
        ),
        "latency_seconds": metrics.get("latency_seconds"),
        "timeouts": metrics.get("timeouts"),
        "provider_failures": metrics.get("failures"),
        "retries": metrics.get("retries"),
        "attempts": metrics.get("attempts"),
        "successes": metrics.get("successes"),
        "average_attempt_latency": metrics.get("average_attempt_latency") or (
            safe_float(metrics.get("total_attempt_latency")) / safe_float(metrics.get("attempts"))
            if safe_float(metrics.get("total_attempt_latency")) is not None and safe_float(metrics.get("attempts"), 0) else None
        ),
        "max_attempt_latency": metrics.get("max_attempt_latency"),
    }


def candidate_data(run_dir: Path, generations: List[int], config: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    customer: List[Dict[str, Any]] = []
    service: List[Dict[str, Any]] = []
    for generation in generations:
        customer_file = read_generation_file(run_dir, generation, "customer_candidates.json", {}) or {}
        customer_generation = read_generation_file(run_dir, generation, "customer_generation.json", {}) or {}
        selected = customer_file.get("selected_policy") or {}
        scores = customer_file.get("scores") or []
        score_by_id = {item.get("policy_id"): item for item in scores if item.get("policy_id")}
        candidate_records = []
        for item in customer_generation.get("candidates", []) or []:
            if not isinstance(item, dict):
                continue
            policy = item.get("constructed_policy") or item.get("candidate_policy") or item.get("policy") or item.get("raw_candidate") or {}
            policy_id = item.get("policy_id") or policy.get("policy_id")
            candidate_records.append({
                "candidate_index": item.get("candidate_index"),
                "policy_id": policy_id,
                "policy": policy,
                "raw_candidate": item.get("raw_candidate"),
                "schema_construction": item.get("schema_construction"),
                "validation": item.get("validator") or item.get("candidate_validation"),
                "accepted": item.get("accepted"),
                "selected": policy_id == selected.get("policy_id"),
                "elite": bool(item.get("elite") or item.get("is_elite")),
                "score": score_by_id.get(policy_id, {}),
                "rejection_reason": item.get("rejection_reason") or item.get("reason"),
            })
        if not candidate_records:
            candidate_records = [
                {
                    "candidate_index": index,
                    "policy_id": item.get("policy_id"),
                    "policy": item,
                    "raw_candidate": item,
                    "schema_construction": None,
                    "validation": None,
                    "accepted": item.get("policy_id") == selected.get("policy_id"),
                    "selected": item.get("policy_id") == selected.get("policy_id"),
                    "elite": False,
                    "score": item,
                    "rejection_reason": None,
                }
                for index, item in enumerate(scores)
                if isinstance(item, dict)
            ]
        response_candidate_count = first_value(
            customer_generation.get("response_candidate_count"),
            len(candidate_records),
            default=None,
        )
        customer.append({
            "generation": generation,
            "selected_policy": selected,
            # scores may include the incumbent in addition to newly generated
            # candidates. Keep both counts explicit so the dashboard does not
            # overstate the configured proposal population.
            "candidate_count": safe_int(response_candidate_count),
            "evaluated_candidate_count": len(scores),
            "scores": scores,
            "candidates": candidate_records,
            "generation_status": customer_generation.get("status"),
            "evaluation_status": customer_generation.get("status"),
            "attempts": customer_generation.get("attempts", []),
            "retry_count": customer_generation.get("retry_count", 0),
            "incumbent_policy_id": customer_generation.get("incumbent_policy_id"),
            "elite_policy_ids": customer_generation.get("elite_policy_ids", []),
            "selected_policy_id": customer_generation.get("selected_policy_id") or selected.get("policy_id"),
            "source_failure_ids": selected.get("source_failure_ids", []),
        })
        service_generation = read_generation_file(run_dir, generation, "service_generation.json", {}) or {}
        gate = read_generation_file(run_dir, generation, "service_gate.json", {}) or {}
        candidates = gate.get("candidates") or []
        generated_candidates = {
            item.get("patch_id") or (item.get("constructed_patch") or {}).get("patch_id"): item
            for item in service_generation.get("candidates", []) or []
            if isinstance(item, dict)
        }
        normalized_candidates = []
        for candidate in candidates:
            candidate = dict(candidate)
            patch_id = candidate.get("patch_id") or (candidate.get("patch") or {}).get("patch_id")
            generated = generated_candidates.get(patch_id, {})
            patch = candidate.get("patch") or generated.get("constructed_patch") or generated.get("patch")
            candidate["patch"] = patch
            candidate["raw_candidate"] = candidate.get("raw_candidate") or generated.get("raw_candidate")
            candidate["schema_construction"] = candidate.get("schema_construction") or generated.get("schema_construction")
            candidate["candidate_validation"] = candidate.get("candidate_validation") or generated.get("candidate_validation")
            candidate["candidate_policy"] = candidate.get("candidate_policy") or generated.get("candidate_policy")
            candidate["selected"] = patch_id == service_generation.get("selected_policy_id")
            candidate["latest_filter_rejected"] = str(candidate.get("reason") or "").startswith("latest_attack_filter:")
            candidate["evaluation_scope"] = "latest_only" if candidate["latest_filter_rejected"] else "latest_replay_normal"
            normalized_candidates.append(candidate)
        candidates = normalized_candidates
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
            "selected_policy": service_generation.get("selected_policy") or {},
            "selected_policy_id": service_generation.get("selected_policy_id"),
            "generation_status": service_generation.get("status"),
            "evaluation_status": service_generation.get("status"),
            "attempts": service_generation.get("attempts", []),
            "retry_count": service_generation.get("retry_count", 0),
            "provenance_present": bool(candidates) and all(
                isinstance(item, dict) and (
                    item.get("patch") is not None or item.get("candidate_policy") is not None
                ) for item in candidates
            ),
        })
    return customer, service


def failure_analysis(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, Dict[str, Any]] = {}
    generation_signatures: Dict[int, set] = defaultdict(set)
    for episode in episodes:
        if episode.get("status") != "LEGITIMATE FAILURE":
            continue
        signature = episode.get("failure_signature") or {}
        signature_id = signature.get("signature_id") or "unlabelled_failure"
        generation = int(episode.get("generation", 0))
        generation_signatures[generation].add(signature_id)
        record = grouped.setdefault(signature_id, {
            "signature_id": signature_id,
            "error_types": set(),
            "sop_nodes": set(),
            "tools": set(),
            "count": 0,
            "generations": set(),
            "customer_policy_ids": set(),
            "service_policy_ids": set(),
            "representative_episode_ids": [],
        })
        record["count"] += 1
        record["generations"].add(generation)
        record["error_types"].update(episode.get("error_types") or signature.get("error_types") or [])
        if episode.get("sop_node"):
            record["sop_nodes"].add(episode["sop_node"])
        record["tools"].update(episode.get("tool_sequence") or [])
        if episode.get("customer_policy_id"):
            record["customer_policy_ids"].add(episode["customer_policy_id"])
        if episode.get("service_policy_id"):
            record["service_policy_ids"].add(episode["service_policy_id"])
        if len(record["representative_episode_ids"]) < 3:
            record["representative_episode_ids"].append(episode.get("episode_id"))
    first_seen = {signature: min(gens) for signature, gens in ((key, value["generations"]) for key, value in grouped.items()) if gens}
    failures = []
    for signature_id, record in sorted(grouped.items()):
        generations = sorted(record["generations"])
        failures.append({
            **record,
            "error_types": sorted(record["error_types"]),
            "sop_nodes": sorted(record["sop_nodes"]),
            "tools": sorted(record["tools"]),
            "generations": generations,
            "first_seen_generation": generations[0] if generations else None,
            "last_seen_generation": generations[-1] if generations else None,
            "customer_policy_ids": sorted(record["customer_policy_ids"]),
            "service_policy_ids": sorted(record["service_policy_ids"]),
            "new": True,
            "repeated": record["count"] > 1,
        })
    trajectory = []
    seen = set()
    for generation in sorted(generation_signatures):
        current = generation_signatures[generation]
        new = sorted(current - seen)
        repeated = sorted(current & seen)
        trajectory.append({
            "generation": generation,
            "new_signatures": new,
            "repeated_signatures": repeated,
            "new_count": len(new),
            "repeated_count": len(repeated),
        })
        seen.update(current)
    return {
        "signatures": failures,
        "trajectory": trajectory,
        "unique_count": len(failures),
        "legitimate_failure_count": sum(item["count"] for item in failures),
    }


def repair_fingerprint(patch: Any) -> Optional[str]:
    if not isinstance(patch, dict):
        return None
    rules = patch.get("rules") or []
    if rules:
        parts = [
            f"{normalized_text(rule.get('category'))}:{normalized_text(rule.get('text'))}"
            for rule in rules if isinstance(rule, dict)
        ]
        return "|".join(parts) or None
    category = normalized_text(patch.get("category") or patch.get("rule_category"))
    text = normalized_text(patch.get("text") or patch.get("rule_text"))
    return f"{category}:{text}" if category or text else None


def repair_analysis(service: List[Dict[str, Any]]) -> Dict[str, Any]:
    all_candidates = []
    by_generation = []
    occurrences = Counter()
    rejected_occurrences = Counter()
    for item in service:
        generation = item.get("generation", 0)
        generation_candidates = []
        for candidate in item.get("candidates", []) or []:
            fingerprint = repair_fingerprint(candidate.get("patch"))
            view = {
                "generation": generation,
                "patch_id": candidate.get("patch_id") or (candidate.get("patch") or {}).get("patch_id"),
                "fingerprint": fingerprint,
                "accepted": candidate.get("accepted") is True,
                "evaluation_status": candidate.get("evaluation_status", "valid"),
                "reason": candidate.get("reason"),
                "delta": candidate.get("delta"),
            }
            generation_candidates.append(view)
            all_candidates.append(view)
            if fingerprint:
                occurrences[fingerprint] += 1
                if not view["accepted"] and view["evaluation_status"] not in {"invalid", "inconclusive"}:
                    rejected_occurrences[fingerprint] += 1
        by_generation.append({"generation": generation, "candidates": generation_candidates})
    total = len(all_candidates)
    accepted = sum(item["accepted"] for item in all_candidates)
    valid = sum(item["evaluation_status"] not in {"invalid", "inconclusive"} for item in all_candidates)
    return {
        "statistics": {
            "total_service_proposals": total,
            "unique_service_proposals": len(occurrences),
            "repeated_service_proposals": sum(value - 1 for value in occurrences.values() if value > 1),
            "repeated_rejected_proposals": sum(value - 1 for value in rejected_occurrences.values() if value > 1),
            "accepted_service_proposals": accepted,
            "rejected_service_proposals": sum(not item["accepted"] and item["evaluation_status"] not in {"invalid", "inconclusive"} for item in all_candidates),
            "invalid_service_proposals": sum(item["evaluation_status"] in {"invalid", "inconclusive"} for item in all_candidates),
            "service_acceptance_rate": accepted / total if total else None,
            "repair_recurrence_rate": sum(value - 1 for value in occurrences.values() if value > 1) / total if total else None,
        },
        "by_generation": by_generation,
    }


def metric_bundle(value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, dict):
        return None
    metrics = value.get("metrics") if isinstance(value.get("metrics"), dict) else value
    result = {}
    for key in ("task_success", "action_execution", "goal_fulfillment", "robust_task_success", "latest_task_success", "replay_task_success", "normal_task_success"):
        if key in metrics:
            result[key] = safe_float(metrics.get(key))
    return result or None


def robustness_data(service: List[Dict[str, Any]], heldout: Any, fresh: List[Dict[str, Any]]) -> Dict[str, Any]:
    trajectory = []
    for item in service:
        candidates = [candidate for candidate in item.get("candidates", []) if candidate.get("evaluation_status", "valid") not in {"invalid", "inconclusive"}]
        metrics = [candidate.get("metrics") or {} for candidate in candidates]
        def best(key: str) -> Optional[float]:
            values = [safe_float(entry.get(key)) for entry in metrics]
            values = [value for value in values if value is not None]
            return max(values) if values else None
        trajectory.append({
            "generation": item.get("generation"),
            "latest": best("latest_task_success"),
            "replay": best("replay_task_success"),
            "normal": best("normal_task_success"),
            "robust": best("robust_task_success"),
        })
    fresh_metrics = [metric_bundle(item) for item in fresh if isinstance(item, dict)]
    fresh_metrics = [item for item in fresh_metrics if item]
    return {
        "latest": trajectory[-1].get("latest") if trajectory else None,
        "replay": trajectory[-1].get("replay") if trajectory else None,
        "normal": trajectory[-1].get("normal") if trajectory else None,
        "heldout": metric_bundle(heldout),
        "fresh_adversary": fresh_metrics[0] if fresh_metrics else None,
        "trajectory": trajectory,
    }


def completeness(run_dir: Path, episodes: List[Dict[str, Any]], service: List[Dict[str, Any]], generations: List[int], expected_generations: Optional[int]) -> Dict[str, Any]:
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
    complete_generations = [
        generation for generation in generations
        if (run_dir / "generations" / f"gen_{generation:03d}" / "COMPLETE.json").exists()
    ]
    customer_artifact_present = any(
        (run_dir / "generations" / f"gen_{generation:03d}" / name).exists()
        for generation in generations
        for name in ("customer_candidates.json", "customer_generation.json")
    )
    service_artifact_present = any(
        (run_dir / "generations" / f"gen_{generation:03d}" / name).exists()
        for generation in generations
        for name in ("service_gate.json", "service_generation.json")
    )
    service_applicable = bool(service_artifact_present or any(item.get("candidate_count") for item in service))
    return {
        "metrics": metrics_path.exists(),
        "split_manifest": all(path.exists() for path in split_files),
        "episodes": bool(episodes),
        "traces": bool(trace_files) or any(ep.get("trace_events") for ep in episodes),
        "candidate_provenance": all(item.get("provenance_present") for item in service) if service_applicable else None,
        "valid_invalid_separated": bool(episodes) and any(
            ep.get("evaluation_status") == "valid" for ep in episodes
        ) and any(ep.get("evaluation_status") != "valid" for ep in episodes) or bool(episodes),
        "heldout_not_used_by_evolver": None,
        "artifacts": {
            "provenance": artifact_status(run_dir / "environment" / "provenance.json"),
            "split_manifest": "confirmed" if all(path.exists() for path in split_files) else "missing",
            "generations": "confirmed" if expected_generations is None or len(complete_generations) >= expected_generations else "missing",
            "customer_candidates": "confirmed" if customer_artifact_present else "missing",
            "service_gate": "confirmed" if service_artifact_present and service_applicable else ("not_applicable" if not service_applicable else "missing"),
            "raw_trace": "confirmed" if bool(trace_files) or any(ep.get("trace_events") for ep in episodes) else "missing",
            "orchestration_metrics": "confirmed" if metrics_path.exists() else "missing",
            "heldout": "confirmed" if (run_dir / "analysis" / "heldout_results.json").exists() else "not_evaluated",
            "fresh_adversary": "confirmed" if list((run_dir / "analysis").glob("fresh_adversary_*.json")) else "not_evaluated",
        },
    }


def export_run(label: str, run_dir: Path, model_override: Optional[str] = None,
               provider_override: Optional[str] = None,
               analysis_override: Optional[Path] = None,
               formal_protocol_commit: Optional[str] = None,
               formal_protocol_tag: Optional[str] = None) -> Dict[str, Any]:
    config = read_json(run_dir / "config" / "evolution.json", {}) or {}
    provenance = read_json(run_dir / "environment" / "provenance.json", {}) or {}
    metrics = read_json(run_dir / "analysis" / "orchestration_metrics.json", {}) or {}
    raw_episodes = all_episode_rows(run_dir)
    raw_trace_records = load_trace_records(run_dir)
    traces = trace_index(raw_trace_records)
    episodes = [normalize_episode(row, traces) for row in raw_episodes]
    generations = load_generations(run_dir)
    customer, service = candidate_data(run_dir, generations, config)
    frontier = read_json(run_dir / "analysis" / "weakness_frontier.json", []) or []
    heldout = read_json(run_dir / "analysis" / "heldout_results.json", None)
    fresh_adversary = [
        read_json(path, {}) for path in sorted((run_dir / "analysis").glob("fresh_adversary_*.json"))
    ]
    trajectory_analysis = read_json(analysis_override, {}) if analysis_override else None
    if trajectory_analysis is None:
        candidates = sorted((run_dir / "analysis").glob("*trajectory*.json"))
        trajectory_analysis = read_json(candidates[-1], {}) if candidates else None
    # ``analyze_evolution_trajectory.py`` emits a multi-run envelope. Select
    # the matching analysis record without changing the analyzer's artifact
    # contract or recomputing anything here.
    if isinstance(trajectory_analysis, dict) and isinstance(trajectory_analysis.get("runs"), list):
        matching = next(
            (
                item for item in trajectory_analysis["runs"]
                if isinstance(item, dict)
                and (
                    item.get("label") == label
                    or Path(str(item.get("run_dir", ""))).resolve() == run_dir.resolve()
                )
            ),
            None,
        )
        trajectory_analysis = matching
    model = model_override or infer_model(config, provenance, raw_trace_records)
    is_mock = contains_mock(raw_episodes) or "mock" in json.dumps(provenance, ensure_ascii=False).lower()
    source_kind = "MOCK" if is_mock else "REAL"
    model_metadata = config.get("model_metadata") or {}
    runtime_freeze_commit = first_value(
        provenance.get("runtime_freeze_commit"),
        config.get("runtime_freeze_commit"),
        model_metadata.get("parent_freeze_commit"),
    )
    runtime_freeze_tag = first_value(
        provenance.get("runtime_freeze_tag"),
        config.get("runtime_freeze_tag"),
        model_metadata.get("parent_freeze_tag"),
    )
    formal_protocol = config.get("formal_protocol") or provenance.get("formal_protocol") or {}
    protocol_commit = first_value(
        formal_protocol_commit,
        formal_protocol.get("commit"),
        formal_protocol.get("commit_sha"),
        provenance.get("formal_protocol_commit"),
        model_metadata.get("formal_protocol_commit"),
    )
    protocol_tag = first_value(
        formal_protocol_tag,
        formal_protocol.get("tag"),
        formal_protocol.get("freeze_tag"),
        provenance.get("formal_protocol_tag"),
        model_metadata.get("formal_protocol_tag"),
    )
    observed_runtime_commit = first_value(
        provenance.get("observed_runtime_freeze_commit"),
        provenance.get("commit_sha"),
        provenance.get("git_commit"),
    )
    observed_runtime_tag = first_value(
        provenance.get("observed_runtime_freeze_tag"),
        provenance.get("freeze_tag"),
    )
    if runtime_freeze_commit and runtime_freeze_tag and observed_runtime_commit and observed_runtime_tag:
        runtime_freeze_match_status = (
            "matched"
            if runtime_freeze_commit == observed_runtime_commit and runtime_freeze_tag == observed_runtime_tag
            else "mismatch"
        )
    else:
        runtime_freeze_match_status = "not_confirmed"
    metadata = {
        "mode": label or config.get("experiment_mode", "unknown"),
        "source_kind": source_kind,
        "model": model,
        "provider": provider_override or infer_provider(config, provenance, model),
        "run_status": first_value(
            metrics.get("run_status"),
            "complete" if all((run_dir / "generations" / f"gen_{generation:03d}" / "COMPLETE.json").exists() for generation in generations) and generations else "inconclusive",
        ),
        "case_count": first_value((config.get("splits") or {}).get("max_cases"), len({ep.get("case_id") for ep in episodes if ep.get("case_id")})),
        "seed": first_value(config.get("seed"), (config.get("splits") or {}).get("seed")),
        "commit_sha": first_value(provenance.get("commit_sha"), provenance.get("git_commit")),
        "freeze_tag": first_value(provenance.get("freeze_tag")),
        "runtime_freeze_commit": runtime_freeze_commit,
        "runtime_freeze_tag": runtime_freeze_tag,
        "runtime_freeze_match_status": runtime_freeze_match_status,
        "observed_runtime_freeze_commit": observed_runtime_commit,
        "observed_runtime_freeze_tag": observed_runtime_tag,
        "formal_protocol_commit": protocol_commit,
        "formal_protocol_tag": protocol_tag,
        "formal_protocol_id": first_value(model_metadata.get("formal_case_set_id"), (config.get("splits") or {}).get("strategy")),
        "generations": first_value(config.get("max_generations"), len(generations), default=0),
        "run_dir": str(run_dir),
        "scenario": config.get("scenario") or provenance.get("scenario"),
        "split_strategy": (config.get("splits") or {}).get("strategy"),
        "max_cases": (config.get("splits") or {}).get("max_cases"),
        "split_seed": (config.get("splits") or {}).get("seed"),
        "max_turns": (config.get("evaluation") or {}).get("max_turns"),
        "judge_in_evolution": (config.get("evaluation") or {}).get("judge_in_evolution"),
        "candidate_counts": {
            "customer": (config.get("customer") or {}).get("candidate_count"),
            "customer_elite": (config.get("customer") or {}).get("elite_count"),
            "service": (config.get("service") or {}).get("candidate_count"),
            "service_replay": (config.get("service") or {}).get("replay_attack_count"),
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
        "failure_analysis": failure_analysis(episodes),
        "repair_analysis": repair_analysis(service),
        "robustness": robustness_data(service, heldout, fresh_adversary),
        "heldout": heldout,
        "fresh_adversary": fresh_adversary,
        "trajectory_analysis": trajectory_analysis,
        "completeness": completeness(run_dir, episodes, service, generations, safe_int(config.get("max_generations"))),
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


def parse_mapping(value: str, flag: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"{flag} must use LABEL=PATH")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError(f"{flag} must use LABEL=PATH")
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
    parser.add_argument("--freeze-commit", help="optional expected runtime freeze commit")
    parser.add_argument("--freeze-tag", help="optional expected runtime freeze tag")
    parser.add_argument("--analysis", action="append", type=lambda value: parse_mapping(value, "--analysis"),
                        help="optional trajectory analysis mapping LABEL=PATH")
    parser.add_argument("--formal-protocol-commit", help="optional formal protocol commit recorded in the export")
    parser.add_argument("--formal-protocol-tag", help="optional formal protocol tag recorded in the export")
    args = parser.parse_args()
    analysis_by_label = dict(args.analysis or [])
    runs = [
        export_run(
            label,
            path,
            model_override=args.model,
            provider_override=args.provider,
            analysis_override=analysis_by_label.get(label),
            formal_protocol_commit=args.formal_protocol_commit,
            formal_protocol_tag=args.formal_protocol_tag,
        )
        for label, path in args.run
    ]
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
