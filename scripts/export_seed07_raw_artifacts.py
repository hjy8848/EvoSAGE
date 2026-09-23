#!/usr/bin/env python3
"""Export Seed 7 conversations and raw model outputs without evaluator truth."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_RUNS = {
    "static": "results/formal_20260923_static_seed07",
    "customer_only": (
        "results/formal_20260923_customer_only_seed07_fresh_20260923_011241_230e88"
    ),
    "coevolution": "results/formal_20260923_coevolution_seed07",
}
DEFAULT_OUTPUT = "experiments/seed07_raw_artifacts"

SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "headers",
    "request_headers",
    "access_token",
    "refresh_token",
    "client_secret",
    "case_spec",
    "context_data",
    "backend_final_state",
    "backend_state_before",
    "backend_state_after",
    "user_environment_state",
    "judge_evaluation",
    "evaluation",
    "ground_truth",
    "ground_truth_data",
    "gold_path",
    "hidden_state",
    "hidden_facts",
    "system_info",
    "backend_record",
    "prompt",
    "messages",
    "llm_request",
}
SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{12,}"),
)


def _key_id(key: Any) -> str:
    return re.sub(r"[^a-z0-9_]", "", str(key).lower())


def clean(value: Any) -> Any:
    """Recursively remove credential, prompt, and evaluator-only fields."""
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            if _key_id(key) in SENSITIVE_KEYS:
                continue
            result[key] = clean(child)
        return result
    if isinstance(value, list):
        return [clean(item) for item in value]
    if isinstance(value, str):
        for pattern in SECRET_PATTERNS:
            value = pattern.sub("[REDACTED]", value)
    return value


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_attempt(attempt: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "attempt_index",
        "turn_id",
        "phase",
        "episode_id",
        "turn",
        "finish_reason",
        "content",
        "reasoning",
        "reasoning_content",
        "tool_calls",
        "parsed_arguments",
        "parse_diagnostics",
        "tool_results",
        "request_id",
        "latency_seconds",
        "raw_provider_response",
        "max_tokens",
        "input_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "parse_status",
        "parse_error",
        "provider_error",
        "timeout",
    )
    return clean({key: attempt[key] for key in fields if key in attempt})


def agent_output(agent: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "turn_id",
        "timestamp",
        "classification_output",
        "cot",
        "expected_path",
        "predicted_path",
        "final_output",
        "predicted_action",
        "chat",
        "path_taken",
        "executed_path",
        "executed_action",
        "json_parse_failed",
        "tool_calls",
        "tool_results",
        "action_result",
        "raw_llm_response",
        "raw_provider_response",
        "raw_final_assistant_message",
    )
    result = {key: agent[key] for key in fields if key in agent}
    metadata = agent.get("metadata") or {}
    attempts = metadata.get("llm_attempts") if isinstance(metadata, dict) else None
    if attempts:
        result["model_attempts"] = [model_attempt(item) for item in attempts]
    return clean(result)


def episode_record(mode: str, run: Path, simulation: dict[str, Any], generation_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    turns = []
    for turn in simulation.get("turns") or []:
        turns.append(
            clean(
                {
                    "turn_id": turn.get("turn_id"),
                    "timestamp": turn.get("timestamp"),
                    "user_message": turn.get("user_message"),
                    "agent": agent_output(turn.get("agent_output") or {}),
                    "tool_calls": turn.get("tool_calls") or [],
                    "tool_results": turn.get("tool_results") or [],
                    "action_calls": turn.get("action_calls") or [],
                    "action_results": turn.get("action_results") or [],
                    "predicted_classification": turn.get("predicted_classification"),
                    "predicted_path": turn.get("predicted_path"),
                    "predicted_action": turn.get("predicted_action"),
                    "executed_path": turn.get("executed_path"),
                    "executed_action": turn.get("executed_action"),
                }
            )
        )
    sid = simulation.get("simulation_id")
    generation_meta = generation_index.get(str(sid), {})
    return {
        "artifact_type": "raw_episode_transcript",
        "data_label": "REAL",
        "mode": mode,
        "source_run": run.name,
        "simulation_id": sid,
        "scenario_id": simulation.get("scenario_id"),
        "model_name": simulation.get("model_name"),
        "generation": generation_meta.get("generation"),
        "split": generation_meta.get("split"),
        "customer_policy_id": generation_meta.get("customer_policy_id"),
        "service_policy_id": generation_meta.get("service_policy_id"),
        "user_intent": clean(simulation.get("user_intent")),
        "adversarial_intensity": simulation.get("adversarial_intensity"),
        "dialogue_length": simulation.get("dialogue_length"),
        "duration_seconds": simulation.get("duration_seconds"),
        "conversation": turns,
        "environment_outcome": clean(
            {
                "actions_taken": simulation.get("actions_taken"),
                "predicted_action": simulation.get("predicted_action"),
                "executed_action": simulation.get("executed_action"),
                "predicted_path": simulation.get("predicted_path"),
                "executed_path": simulation.get("executed_path"),
                "termination_reason": simulation.get("termination_reason"),
                "final_status": simulation.get("final_status"),
                "goal_solved": simulation.get("goal_solved"),
            }
        ),
    }


def generation_index(run: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for path in sorted(run.glob("generations/gen_*/episodes.jsonl")):
        generation = int(path.parent.name.split("_")[-1])
        for row in read_jsonl(path):
            meta = row.get("metadata") or {}
            index[str(row.get("episode_id"))] = {
                "generation": generation,
                "split": meta.get("split", row.get("split")),
                "customer_policy_id": row.get("customer_policy_id"),
                "service_policy_id": row.get("service_policy_id"),
            }
    return index


def export_episodes(run_dirs: dict[str, Path], output: Path) -> dict[str, Any]:
    counts: dict[str, Any] = {}
    destination = output / "episodes.jsonl"
    seen_global: set[tuple[str, str]] = set()
    with destination.open("w", encoding="utf-8") as out:
        for mode, run in run_dirs.items():
            index = generation_index(run)
            seen: set[str] = set()
            trace_paths = sorted((run / "real_traces").glob("*.jsonl"))
            for path in trace_paths:
                for row in read_jsonl(path):
                    simulation = row.get("simulation") or {}
                    sid = str(simulation.get("simulation_id") or "")
                    if not sid or sid in seen:
                        continue
                    seen.add(sid)
                    seen_global.add((mode, sid))
                    record = episode_record(mode, run, simulation, index)
                    out.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            counts[mode] = {"episodes": len(seen), "trace_files": len(trace_paths)}
    return counts


def generation_record(mode: str, run: Path, path: Path, data: dict[str, Any], role: str) -> dict[str, Any]:
    # The generation prompt is intentionally omitted; only generated output and provenance
    # are included. Candidate validation/gate records are artifact-derived, not recomputed.
    record = {
        "artifact_type": f"{role}_generation_output",
        "data_label": "REAL",
        "mode": mode,
        "source_run": run.name,
        "generation": data.get("generation", int(path.parent.name.split("_")[-1])),
        "role": role,
        "status": data.get("status"),
        "reason": data.get("reason"),
        "max_tokens": data.get("max_tokens"),
        "retry_count": data.get("retry_count"),
        "retry_limit": data.get("retry_limit"),
        "response_candidate_count": data.get("response_candidate_count"),
        "accepted_candidate_count": data.get("accepted_candidate_count"),
        "attempts": [model_attempt(item) for item in data.get("attempts", [])],
        "candidates": clean(data.get("candidates", [])),
        "selected_policy": clean(data.get("selected_policy")),
        "selected_policy_id": data.get("selected_policy_id"),
        "scores": clean(data.get("scores")),
    }
    return clean(record)


def export_generations(run_dirs: dict[str, Path], output: Path) -> int:
    destination = output / "evolution_outputs.jsonl"
    count = 0
    with destination.open("w", encoding="utf-8") as out:
        for mode, run in run_dirs.items():
            for role in ("customer", "service"):
                for path in sorted(run.glob(f"generations/gen_*/{role}_generation.json")):
                    data = json.loads(path.read_text(encoding="utf-8"))
                    out.write(json.dumps(generation_record(mode, run, path, data, role), ensure_ascii=False, sort_keys=True) + "\n")
                    count += 1
            for path in sorted(run.glob("generations/gen_*/service_gate.json")):
                data = json.loads(path.read_text(encoding="utf-8"))
                record = {
                    "artifact_type": "service_candidate_gate_output",
                    "data_label": "REAL",
                    "mode": mode,
                    "source_run": run.name,
                    "generation": int(path.parent.name.split("_")[-1]),
                    "gate": clean(data),
                }
                out.write(json.dumps(clean(record), ensure_ascii=False, sort_keys=True) + "\n")
                count += 1
    return count


def verify_no_secrets(output: Path) -> None:
    for path in output.glob("*.jsonl"):
        for line_number, line in enumerate(path.open(encoding="utf-8"), 1):
            if any(pattern.search(line) for pattern in SECRET_PATTERNS):
                raise ValueError(f"Possible credential remains in {path.name}:{line_number}")
            row = json.loads(line)
            forbidden = SENSITIVE_KEYS.intersection(_key_id(key) for key in row)
            if forbidden:
                raise ValueError(f"Forbidden top-level fields in {path.name}:{line_number}: {forbidden}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static", default=DEFAULT_RUNS["static"])
    parser.add_argument("--customer-only", default=DEFAULT_RUNS["customer_only"])
    parser.add_argument("--coevolution", default=DEFAULT_RUNS["coevolution"])
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    run_dirs = {
        "static": Path(args.static),
        "customer_only": Path(args.customer_only),
        "coevolution": Path(args.coevolution),
    }
    for mode, run in run_dirs.items():
        if not run.is_dir() or not (run / "real_traces").is_dir():
            raise FileNotFoundError(f"Missing completed run/real_traces for {mode}: {run}")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    episode_counts = export_episodes(run_dirs, output)
    generation_count = export_generations(run_dirs, output)
    verify_no_secrets(output)

    source_files = {}
    for mode, run in run_dirs.items():
        files = sorted(
            list((run / "real_traces").glob("*.jsonl"))
            + list(run.glob("generations/gen_*/episodes.jsonl"))
            + list(run.glob("generations/gen_*/customer_generation.json"))
            + list(run.glob("generations/gen_*/service_generation.json"))
            + list(run.glob("generations/gen_*/service_gate.json"))
        )
        source_files[mode] = [
            {"path": str(path.relative_to(run)), "sha256": sha256_file(path)} for path in files
        ]
    manifest = {
        "dataset_label": "REAL",
        "description": "Seed 7 raw dialogue/model outputs; evaluator-only gold and hidden state excluded.",
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "runs": {mode: {"run_id": run.name, **episode_counts[mode]} for mode, run in run_dirs.items()},
        "evolution_output_records": generation_count,
        "source_artifacts": source_files,
        "export_files": {
            name: {"sha256": sha256_file(output / name), "bytes": (output / name).stat().st_size}
            for name in ("episodes.jsonl", "evolution_outputs.jsonl")
        },
        "excluded": [
            "Evaluator records and scores",
            "Gold paths and ground-truth labels",
            "CaseSpec, hidden backend snapshots, and simulator private state",
            "Full prompts and request headers/credentials",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output), "episodes": episode_counts, "evolution_output_records": generation_count, "manifest": str(output / 'manifest.json')}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
