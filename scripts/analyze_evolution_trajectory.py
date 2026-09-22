#!/usr/bin/env python3
"""Read-only trajectory and service-repair analysis for EvoSAGE runs.

The analyzer consumes structured artifacts only.  It never invokes a model,
the evaluator, or a scorer, and it never recomputes a score from raw dialogue.
Use ``--run LABEL=PATH`` for one or more independent run directories.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return default


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
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


def generation_dirs(run_dir: Path) -> List[Tuple[int, Path]]:
    result = []
    for path in sorted((run_dir / "generations").glob("gen_*/")):
        try:
            result.append((int(path.name.split("_")[-1]), path))
        except ValueError:
            continue
    return result


def valid_episode(row: Dict[str, Any]) -> bool:
    metadata = row.get("metadata") or {}
    return row.get("evaluation_status", "valid") == "valid" and not metadata.get("protocol_failure", False)


def legitimate_failure(row: Dict[str, Any]) -> bool:
    return valid_episode(row) and not bool(row.get("task_success"))


def numeric(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def best_metric(candidates: Iterable[Dict[str, Any]], field: str) -> Optional[float]:
    values = [numeric((candidate.get("metrics") or {}).get(field)) for candidate in candidates]
    values = [value for value in values if value is not None]
    return max(values) if values else None


def mean(rows: Iterable[Dict[str, Any]], field: str) -> Optional[float]:
    values = [numeric(row.get(field)) for row in rows]
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def signature_id(row: Dict[str, Any]) -> Optional[str]:
    signature = row.get("failure_signature") or {}
    if isinstance(signature, dict) and signature.get("signature_id"):
        return str(signature["signature_id"])
    return None


def episode_metrics(run_dir: Path, generation: int, directory: Path) -> Dict[str, Any]:
    rows = read_jsonl(directory / "episodes.jsonl")
    valid = [row for row in rows if valid_episode(row)]
    failures = [row for row in valid if not bool(row.get("task_success"))]
    signatures = {value for row in failures if (value := signature_id(row))}
    return {
        "generation": generation,
        "episodes": len(rows),
        "valid_episodes": len(valid),
        "invalid_episodes": len(rows) - len(valid),
        "task_success": mean(valid, "task_success"),
        "action_execution": mean(valid, "action_execution_score"),
        "goal_fulfillment": mean(valid, "goal_fulfillment_score"),
        "legitimate_failure_count": len(failures),
        "unique_failure_signatures": len(signatures),
        "failure_signature_ids": sorted(signatures),
    }


def customer_trajectory(run_dir: Path, generations: List[Tuple[int, Path]]) -> List[Dict[str, Any]]:
    result = []
    previous_policy = (read_json(run_dir / "environment" / "initial_customer_policy.json", {}) or {}).get("policy_id")
    best_so_far: Optional[float] = None
    for generation, directory in generations:
        data = read_json(directory / "customer_candidates.json", {}) or {}
        generation_record = read_json(directory / "customer_generation.json", {}) or {}
        scores = data.get("scores", []) or []
        selected = data.get("selected_policy", {}) or {}
        selected_id = selected.get("policy_id") or generation_record.get("selected_policy_id")
        selected_score = next((item for item in scores if item.get("policy_id") == selected_id), {})
        fitness = numeric(selected_score.get("fitness"))
        best_so_far = fitness if best_so_far is None or (fitness is not None and fitness > best_so_far) else best_so_far
        selected_id = selected_id or "unknown"
        result.append({
            "generation": generation,
            "candidate_proposals": generation_record.get("response_candidate_count", len(data.get("candidate_episode_counts", []) or [])),
            "evaluated_candidates": len(scores),
            "accepted_candidates": generation_record.get("accepted_candidate_count"),
            "selected_policy_id": selected_id,
            "selected_fitness": fitness,
            "best_so_far_fitness": best_so_far,
            "selected_attack_success": numeric(selected_score.get("attack_success")),
            "selected_novelty": numeric(selected_score.get("novelty")),
            "selected_coverage": numeric(selected_score.get("coverage")),
            "incumbent_policy_id": previous_policy,
            "incumbent_retained": bool(previous_policy and selected_id == previous_policy),
            "incumbent_replaced": bool(previous_policy and selected_id != previous_policy),
            "source_failure_ids": [item.get("signature_id") for item in (data.get("source_failures", []) or []) if item.get("signature_id")],
        })
        previous_policy = selected_id
    return result


def normalize_repair(candidate: Dict[str, Any]) -> Optional[str]:
    patch = candidate.get("patch") or {}
    if not isinstance(patch, dict):
        return None
    rules = patch.get("rules") or []
    if not rules:
        return None
    categories = []
    texts = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        categories.append(str(rule.get("category", "")).strip().lower())
        texts.append(str(rule.get("text", "")).strip().lower())
    if not texts:
        return None
    collapse = lambda value: re.sub(r"\s+", " ", value).strip()
    return "|".join(collapse(value) for value in categories) + "::" + "|".join(collapse(value) for value in texts)


def service_trajectory(run_dir: Path, generations: List[Tuple[int, Path]]) -> Dict[str, Any]:
    all_candidates: List[Dict[str, Any]] = []
    by_generation = []
    for generation, directory in generations:
        gate = read_json(directory / "service_gate.json", {}) or {}
        candidates = list(gate.get("candidates", []) or [])
        for candidate in candidates:
            candidate = dict(candidate)
            candidate["_generation"] = generation
            candidate["_normalized_repair"] = normalize_repair(candidate)
            all_candidates.append(candidate)
        valid = [item for item in candidates if item.get("evaluation_status", "valid") != "invalid"]
        accepted = [item for item in candidates if item.get("accepted") is True]
        rejected = [item for item in candidates if item.get("accepted") is not True and item.get("evaluation_status", "valid") != "invalid"]
        invalid = [item for item in candidates if item.get("evaluation_status") == "invalid"]
        metrics = []
        for item in candidates:
            values = item.get("metrics") or {}
            metrics.append({
                "patch_id": (item.get("patch") or {}).get("patch_id") or item.get("patch_id"),
                "evaluation_status": item.get("evaluation_status", "valid"),
                "accepted": bool(item.get("accepted")),
                "delta": numeric(item.get("delta")),
                "robust_task_success": numeric(values.get("robust_task_success")),
                "latest_task_success": numeric(values.get("latest_task_success")),
                "replay_task_success": numeric(values.get("replay_task_success")),
                "normal_task_success": numeric(values.get("normal_task_success")),
                "normal_regression_cases": len(item.get("normal_regression_cases", []) or []),
                "reason": item.get("reason"),
            })
        by_generation.append({
            "generation": generation,
            "gate": {
                "accepted": gate.get("accepted"),
                "reason": gate.get("reason"),
                "delta": numeric(gate.get("delta")),
            },
            "proposals": len(candidates),
            "accepted": len(accepted),
            "rejected": len(rejected),
            "invalid": len(invalid),
            "valid": len(valid),
            "candidate_metrics": metrics,
            "best_robust_task_success": best_metric(candidates, "robust_task_success"),
            "best_latest_task_success": best_metric(candidates, "latest_task_success"),
            "best_replay_task_success": best_metric(candidates, "replay_task_success"),
            "best_normal_task_success": best_metric(candidates, "normal_task_success"),
        })

    occurrences = Counter(item["_normalized_repair"] for item in all_candidates if item.get("_normalized_repair"))
    rejected_occurrences = Counter(
        item["_normalized_repair"]
        for item in all_candidates
        if item.get("_normalized_repair") and item.get("accepted") is not True and item.get("evaluation_status", "valid") != "invalid"
    )
    total = len(all_candidates)
    accepted_count = sum(item.get("accepted") is True for item in all_candidates)
    valid_count = sum(item.get("evaluation_status", "valid") != "invalid" for item in all_candidates)
    return {
        "by_generation": by_generation,
        "repair_statistics": {
            "total_service_proposals": total,
            "unique_service_proposals": len(occurrences),
            "repeated_service_proposals": sum(count - 1 for count in occurrences.values() if count > 1),
            "accepted_service_proposals": accepted_count,
            "rejected_service_proposals": sum(item.get("accepted") is not True and item.get("evaluation_status", "valid") != "invalid" for item in all_candidates),
            "invalid_service_proposals": sum(item.get("evaluation_status") == "invalid" for item in all_candidates),
            "repeated_rejected_proposals": sum(count - 1 for count in rejected_occurrences.values() if count > 1),
            "service_acceptance_rate": (accepted_count / total) if total else None,
            "service_acceptance_rate_over_valid": (accepted_count / valid_count) if valid_count else None,
            "normalized_repair_occurrences": dict(sorted(occurrences.items())),
        },
    }


def analyze(label: str, run_dir: Path) -> Dict[str, Any]:
    generations = generation_dirs(run_dir)
    episodes = [episode_metrics(run_dir, generation, directory) for generation, directory in generations]
    customer = customer_trajectory(run_dir, generations)
    service = service_trajectory(run_dir, generations)
    return {
        "label": label,
        "run_dir": str(run_dir),
        "config": read_json(run_dir / "config" / "evolution.json", {}) or {},
        "generations": [generation for generation, _ in generations],
        "episode_metrics": episodes,
        "customer_trajectory": customer,
        "service_trajectory": service["by_generation"],
        "repair_statistics": service["repair_statistics"],
        "artifacts": {
            "orchestration_metrics": (run_dir / "analysis" / "orchestration_metrics.json").exists(),
            "weakness_frontier": (run_dir / "analysis" / "weakness_frontier.json").exists(),
            "split_manifest": (run_dir / "split_manifest").exists(),
            "attack_archive": (run_dir / "archives" / "attacks.jsonl").exists(),
        },
    }


def parse_run(value: str) -> Tuple[str, Path]:
    if "=" in value:
        label, path = value.split("=", 1)
        return label, Path(path)
    path = Path(value)
    return path.name, path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=RUN_DIR",
                        help="Repeat for each run; label is optional when using a bare path")
    parser.add_argument("--output", type=Path, help="Optional JSON output path outside the run directories")
    args = parser.parse_args()
    payload = {"runs": [analyze(label, path) for label, path in (parse_run(value) for value in args.run)]}
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
