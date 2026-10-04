#!/usr/bin/env python3
"""Summarize Customer-search trajectory artifacts without rescoring episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


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
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict):
            result.append(item)
    return result


def generations(run_dir: Path) -> list[Path]:
    paths = [path for path in (run_dir / "generations").glob("gen_*") if path.is_dir()]
    return sorted(paths, key=lambda item: int(item.name.split("_")[-1]))


def analyze_run(label: str, run_dir: Path) -> dict[str, Any]:
    rows = []
    episodes = []
    for directory in generations(run_dir):
        generation = int(directory.name.split("_")[-1])
        summary = read_json(directory / "SUMMARY.json", {}) or {}
        selection = read_json(directory / "selection.json", {}) or {}
        proposals = read_json(directory / "proposals.json", {}) or {}
        episode_rows = read_jsonl(directory / "episodes.jsonl")
        episodes.extend(episode_rows)
        selected_score = selection.get("selected_score") or {}
        proposal_record = proposals.get("generation_record") or {}
        rows.append({
            "generation": generation,
            "status": summary.get("status", "incomplete"),
            "incumbent_policy_id": summary.get("incumbent_policy_id") or selection.get("incumbent_policy_id"),
            "selected_policy_id": summary.get("selected_policy_id") or selection.get("selected_policy_id"),
            "selected_fitness": summary.get("selected_fitness", selected_score.get("fitness")),
            "selected_official_task_success": summary.get(
                "selected_official_task_success", selected_score.get("official_task_success")
            ),
            "customer_changed": summary.get("customer_changed", selection.get("customer_changed")),
            "requested_candidate_count": summary.get(
                "requested_candidate_count", proposals.get("requested_candidate_count")
            ),
            "proposed_candidate_count": summary.get(
                "proposed_candidate_count", proposals.get("proposed_candidate_count", 0)
            ),
            "evaluated_policy_count": summary.get("evaluated_policy_count"),
            "candidate_scores": selection.get("candidate_scores", []),
            "runtime_metrics": summary.get("runtime_metrics", {}),
            "valid_episode_count": (summary.get("evolution_evaluation") or {}).get("valid_episode_count"),
            "invalid_episode_count": (summary.get("evolution_evaluation") or {}).get("invalid_episode_count"),
            "generation_protocol_status": proposal_record.get("status"),
            "wall_time_seconds": summary.get("wall_time_seconds"),
        })

    valid = [
        item for item in episodes
        if item.get("evaluation_status", "valid") == "valid"
        and item.get("protocol_valid", True) is not False
        and item.get("environment_valid", True) is not False
        and isinstance(item.get("task_success"), bool)
    ]
    invalid_count = len(episodes) - len(valid)
    return {
        "label": label,
        "run_dir": str(run_dir),
        "generation_count": len(rows),
        "generations": rows,
        "episode_count": len(episodes),
        "valid_episode_count": len(valid),
        "invalid_episode_count": invalid_count,
        "official_task_success_over_recorded_episodes": (
            sum(bool(item["task_success"]) for item in valid) / len(valid) if valid else None
        ),
        "source_kind": (read_json(run_dir / "environment" / "provenance.json", {}) or {}).get("evaluator"),
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
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    document = {
        "schema_version": 1,
        "analysis": "customer_search_trajectory",
        "read_only": True,
        "scoring_recomputed": False,
        "runs": [analyze_run(label, path) for label, path in args.run],
    }
    text = json.dumps(document, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
