#!/usr/bin/env python3
"""Read-only summary for the three EvoSAGE pilot run directories.

This script never invokes an evaluator and never recomputes scores from raw
chat text. It consumes only structured artifacts emitted by a run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _read_json(path: Path, default=None):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return default


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except (ValueError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _episodes(run_dir: Path) -> List[Dict[str, Any]]:
    rows = []
    for path in sorted((run_dir / "generations").glob("gen_*/episodes.jsonl")):
        rows.extend(_read_jsonl(path))
    return rows


def _valid(row: Dict[str, Any]) -> bool:
    metadata = row.get("metadata") or {}
    return (
        row.get("evaluation_status", "valid") == "valid"
        and not metadata.get("protocol_failure", False)
    )


def _mean(rows: Iterable[Dict[str, Any]], field: str) -> Optional[float]:
    values = [float(row.get(field, 0.0) or 0.0) for row in rows]
    return sum(values) / len(values) if values else None


def _signature_keys(run_dir: Path, episodes: List[Dict[str, Any]]) -> set[str]:
    keys = set()
    for record in _read_jsonl(run_dir / "archives" / "attacks.jsonl"):
        signature = record.get("failure_signature") or {}
        if signature.get("signature_id"):
            keys.add(str(signature["signature_id"]))
    for row in episodes:
        if not _valid(row) or row.get("task_success", True):
            continue
        signature = row.get("failure_signature") or {}
        if signature.get("signature_id"):
            keys.add(str(signature["signature_id"]))
        else:
            keys.add(json.dumps({
                "node": row.get("sop_node"),
                "step": row.get("path_step_index"),
                "errors": sorted(row.get("error_types") or []),
            }, sort_keys=True))
    return keys


def _service_candidates(run_dir: Path) -> Dict[str, int]:
    accepted = rejected = invalid = total = 0
    for path in sorted((run_dir / "generations").glob("gen_*/service_gate.json")):
        data = _read_json(path, {}) or {}
        for candidate in data.get("candidates", []) or []:
            total += 1
            if candidate.get("evaluation_status") == "invalid":
                invalid += 1
            elif candidate.get("accepted"):
                accepted += 1
            else:
                rejected += 1
    return {
        "service_candidates": total,
        "accepted_service_patches": accepted,
        "rejected_patches": rejected,
        "invalid_candidates": invalid,
    }


def _customer_candidates(run_dir: Path) -> int:
    count = 0
    for path in sorted((run_dir / "generations").glob("gen_*/customer_candidates.json")):
        data = _read_json(path, {}) or {}
        count += len(data.get("scores", []) or [])
    return count


def summarize(mode: str, run_dir: Path) -> Dict[str, Any]:
    rows = _episodes(run_dir)
    valid_rows = [row for row in rows if _valid(row)]
    invalid_rows = [row for row in rows if not _valid(row)]
    metrics = _read_json(run_dir / "analysis" / "orchestration_metrics.json", {}) or {}
    service = _service_candidates(run_dir)
    trace_files = []
    for directory_name in ("real_traces", "traces"):
        directory = run_dir / directory_name
        if directory.exists():
            trace_files.extend(path for path in directory.rglob("*") if path.is_file())
    artifacts_complete = all([
        bool(rows),
        (run_dir / "analysis" / "orchestration_metrics.json").exists(),
        (run_dir / "split_manifest" / "evolution_cases.json").exists(),
        (run_dir / "split_manifest" / "validation_cases.json").exists(),
        (run_dir / "split_manifest" / "heldout_cases.json").exists(),
        bool(trace_files),
    ])
    result = {
        "mode": mode,
        "run_dir": str(run_dir),
        "valid_episodes": len(valid_rows),
        "invalid_episodes": len(invalid_rows),
        "invalid_rate": (len(invalid_rows) / len(rows)) if rows else None,
        "task_success": _mean(valid_rows, "task_success"),
        "action_execution": _mean(valid_rows, "action_execution_score"),
        "goal_fulfillment": _mean(valid_rows, "goal_fulfillment_score"),
        "legitimate_failures": sum(not bool(row.get("task_success")) for row in valid_rows),
        "unique_failure_signatures": len(_signature_keys(run_dir, rows)),
        "customer_candidates": _customer_candidates(run_dir),
        **service,
        # ``llm_requests`` includes both episode-pipeline calls and Customer/
        # Service policy-generation calls.  ``pipeline_requests`` alone would
        # under-report the actual pilot request volume for customer-only and
        # coevolution runs.
        "requests": metrics.get("llm_requests"),
        "tokens": (
            int(metrics.get("input_tokens", 0) or 0)
            + int(metrics.get("output_tokens", 0) or 0)
            if metrics else None
        ),
        "timeouts": metrics.get("timeouts"),
        "provider_failures": metrics.get("failures"),
        "latency_seconds": metrics.get("latency_seconds"),
        "artifacts_complete": artifacts_complete,
        "raw_trace_present": bool(trace_files),
    }
    return result


def _display(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static", type=Path, required=True)
    parser.add_argument("--customer-only", type=Path, required=True)
    parser.add_argument("--coevolution", type=Path, required=True)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    rows = [
        summarize("static", args.static),
        summarize("customer-only", args.customer_only),
        summarize("coevolution", args.coevolution),
    ]
    if args.as_json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0
    fields = [
        "mode", "valid_episodes", "invalid_episodes", "invalid_rate",
        "task_success", "action_execution", "goal_fulfillment",
        "legitimate_failures", "unique_failure_signatures",
        "customer_candidates", "service_candidates", "accepted_service_patches",
        "rejected_patches", "invalid_candidates", "requests", "tokens",
        "timeouts", "provider_failures", "latency_seconds",
    ]
    print("| " + " | ".join(fields) + " |")
    print("| " + " | ".join("---" for _ in fields) + " |")
    for row in rows:
        print("| " + " | ".join(_display(row.get(field)) for field in fields) + " |")
    print()
    for row in rows:
        print(
            f"{row['mode']}: artifacts_complete={row['artifacts_complete']} "
            f"raw_trace_present={row['raw_trace_present']} run_dir={row['run_dir']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
