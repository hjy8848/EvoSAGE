"""Research report generation without inventing missing metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def _metadata(row: Dict[str, Any]) -> Dict[str, Any]:
    value = row.get("metadata")
    return value if isinstance(value, dict) else {}


def _protocol_invalid(row: Dict[str, Any]) -> bool:
    metadata = _metadata(row)
    return (
        row.get("evaluation_status") == "invalid"
        or row.get("protocol_valid") is False
        or metadata.get("protocol_failure") is True
    )


def _environment_invalid(row: Dict[str, Any]) -> bool:
    metadata = _metadata(row)
    return row.get("environment_valid") is False or metadata.get("environment_valid") is False


def _evaluable(row: Dict[str, Any]) -> bool:
    return not _protocol_invalid(row) and not _environment_invalid(row)


def _explicit_customer_valid(row: Dict[str, Any]) -> Optional[bool]:
    if isinstance(row.get("customer_behavior_valid"), bool):
        return row["customer_behavior_valid"]
    assessment = _metadata(row).get("customer_behavior_assessment")
    if isinstance(assessment, dict) and isinstance(assessment.get("valid"), bool):
        return assessment["valid"]
    return None


def _explicit_attribution(row: Dict[str, Any]) -> Optional[bool]:
    value = row.get("service_failure_attributable")
    return value if isinstance(value, bool) else None


def _mean(rows: Iterable[Dict[str, Any]], field: str) -> Optional[float]:
    values = []
    for row in rows:
        value = row.get(field)
        if isinstance(value, bool):
            values.append(float(value))
        elif isinstance(value, (int, float)):
            values.append(float(value))
    return sum(values) / len(values) if values else None


def summarize_episode_reporting(
    rows: Iterable[Dict[str, Any]], *, attack_instance_count: Optional[int] = None
) -> Dict[str, Any]:
    """Summarize already-scored episode artifacts without rescoring dialogue.

    New validity/attribution fields remain ``None`` for legacy artifacts when
    their provenance is absent. Customer-invalid episodes remain inspectable,
    but are excluded from success and attribution denominators.
    """
    episodes = list(rows)
    evaluable = [row for row in episodes if _evaluable(row)]
    behavior_rows = [row for row in evaluable if _explicit_customer_valid(row) is not False]
    protocol_rows = [
        row for row in episodes
        if isinstance(row.get("protocol_valid"), bool)
        or row.get("evaluation_status") in {"valid", "invalid"}
        or isinstance(_metadata(row).get("protocol_failure"), bool)
    ]
    customer_marked = any(_explicit_customer_valid(row) is not None for row in episodes)
    customer_validity_rows = [
        row for row in evaluable if _explicit_customer_valid(row) is not None
    ]
    attribution_marked = any(_explicit_attribution(row) is not None for row in episodes)
    attribution_rows = [
        row for row in behavior_rows if _explicit_attribution(row) is not None
    ]
    v2_schema_marked = any(
        "vulnerability_signature" in row or "failure_occurrence" in row
        for row in episodes
    )
    v2_signatures = {
        str(signature["signature_id"])
        for row in episodes
        if _explicit_attribution(row) is True
        and isinstance((signature := row.get("vulnerability_signature")), dict)
        and signature.get("schema_version") == 2
        and signature.get("signature_id")
    }
    # Treat the three process/recovery metrics as one versioned outcome bundle.
    # Some older/mock EpisodeResult serializers emitted default eventual=0 and
    # recovery=False while leaving strict_process_success unset; those defaults
    # are not evidence that the metrics were actually evaluated.
    explicit_strict = [row for row in behavior_rows if isinstance(row.get("strict_process_success"), bool)]
    explicit_eventual = [
        row for row in behavior_rows
        if isinstance(row.get("strict_process_success"), bool)
        and isinstance(row.get("eventual_goal_success"), (int, float))
    ]
    explicit_recovery = [
        row for row in behavior_rows
        if isinstance(row.get("strict_process_success"), bool)
        and isinstance(row.get("recovery_success"), bool)
    ]
    invalid_protocol_count = sum(_protocol_invalid(row) for row in episodes)
    invalid_environment_count = sum(_environment_invalid(row) for row in episodes)
    customer_invalid_count = sum(
        _explicit_customer_valid(row) is False for row in evaluable
    ) if customer_marked else None
    attributable_failures = sum(_explicit_attribution(row) is True for row in attribution_rows)
    outcome_schema_available = any(
        any(key in row for key in ("strict_process_success", "eventual_goal_success", "recovery_success"))
        for row in episodes
    )
    outcome_metrics_status = (
        "available" if explicit_strict else "not_evaluated" if outcome_schema_available else "legacy"
    )
    if (
        attack_instance_count is None
        and v2_schema_marked
        and attribution_marked
        and attributable_failures == 0
    ):
        # A current-schema run with no attributable failure has a known empty
        # attack-instance set even if the append-only archive file was never
        # materialized.
        attack_instance_count = 0
    return {
        "episodes": len(episodes),
        "valid_episodes": len(evaluable),
        "invalid_episodes": len(episodes) - len(evaluable),
        "valid_episode_rate": len(evaluable) / len(episodes) if episodes else None,
        "protocol_invalid_episodes": invalid_protocol_count if protocol_rows else None,
        "protocol_invalid_rate": invalid_protocol_count / len(protocol_rows) if protocol_rows else None,
        "protocol_evaluated_episodes": len(protocol_rows) if protocol_rows else None,
        "environment_invalid_episodes": invalid_environment_count if any(
            "environment_valid" in row or "environment_valid" in _metadata(row) for row in episodes
        ) else None,
        "customer_behavior_validity_available": customer_marked,
        "customer_behavior_evaluated_episodes": len(customer_validity_rows) if customer_marked else None,
        "customer_behavior_invalid_episodes": customer_invalid_count,
        "customer_behavior_invalid_rate": (
            customer_invalid_count / len(customer_validity_rows)
            if customer_marked and customer_validity_rows else None
        ),
        "attributable_episodes": len(attribution_rows) if attribution_marked else None,
        "attributable_service_failure_count": attributable_failures if attribution_marked else None,
        "attributable_service_failure_rate": (
            attributable_failures / len(attribution_rows)
            if attribution_marked and attribution_rows else (0.0 if attribution_marked else None)
        ),
        "task_success": _mean(behavior_rows, "task_success"),
        "strict_process_success": _mean(explicit_strict, "strict_process_success"),
        "eventual_goal_success": _mean(explicit_eventual, "eventual_goal_success"),
        "recovery_success_rate": _mean(explicit_recovery, "recovery_success"),
        "outcome_metrics_status": outcome_metrics_status,
        "outcome_metrics_evaluated_episodes": len(explicit_strict) if explicit_strict else None,
        "unique_vulnerability_signatures_v2": len(v2_signatures) if v2_schema_marked and attribution_marked else None,
        "attack_instance_count": attack_instance_count,
    }


def generate_report(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    generations = sorted((run_dir / "generations").glob("gen_*/COMPLETE.json"))
    provenance_path = run_dir / "environment" / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.exists() else {}
    evaluator = provenance.get("evaluator", "not evaluated")
    lines = ["# EvoSAGE adversarial co-evolution report", "", f"Run directory: `{run_dir}`", f"Evaluator: **{evaluator}**", ""]
    if not generations:
        lines += ["No completed generation was found; metrics are not evaluated.", ""]
    else:
        lines += [
            "## Completed generations", "",
            "| Generation | Customer | Fitness | Service gate | Evaluable | Task success | Protocol invalid | Customer behavior invalid | Attributable failure rate | Strict process success | Eventual goal success | Recovery success |",
            "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for marker in generations:
            data = json.loads(marker.read_text(encoding="utf-8"))
            generation = data.get("generation", marker.parent.name)
            directory = marker.parent
            selected = json.loads((directory / "customer_candidates.json").read_text(encoding="utf-8")) if (directory / "customer_candidates.json").exists() else {}
            fitness = selected.get("scores", [])
            selected_id = selected.get("selected_policy", {}).get("policy_id", "not evaluated")
            selected_fitness = next((item.get("fitness") for item in fitness if item.get("policy_id") == selected_id), None)
            gate = json.loads((directory / "service_gate.json").read_text(encoding="utf-8")) if (directory / "service_gate.json").exists() else {}
            episodes = list(_read_jsonl(directory / "episodes.jsonl"))
            summary = summarize_episode_reporting(episodes)
            lines.append(
                f"| {generation} | `{selected_id}` | {_format_metric(selected_fitness)} | {gate.get('reason', 'not evaluated')} "
                f"| {_format_metric(summary['valid_episodes'])}/{_format_metric(summary['episodes'])} "
                f"| {_format_metric(summary['task_success'], percent=True)} "
                f"| {_format_metric(summary['protocol_invalid_rate'], percent=True)} "
                f"| {_format_metric(summary['customer_behavior_invalid_rate'], percent=True)} "
                f"| {_format_metric(summary['attributable_service_failure_rate'], percent=True)} "
                f"| {_format_outcome_metric(summary['strict_process_success'], episodes, 'strict_process_success')} "
                f"| {_format_outcome_metric(summary['eventual_goal_success'], episodes, 'eventual_goal_success')} "
                f"| {_format_outcome_metric(summary['recovery_success_rate'], episodes, 'recovery_success')} |"
            )
        lines.append("")
        service_rows = []
        for marker in generations:
            data = json.loads(marker.read_text(encoding="utf-8"))
            directory = marker.parent
            gate_path = directory / "service_gate.json"
            if not gate_path.exists():
                continue
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            generation = data.get("generation", directory.name)
            for candidate in gate.get("candidates", []) or []:
                metrics = candidate.get("metrics") or {}
                candidate_reason = str(candidate.get("reason") or "")
                candidate_status = (
                    "INVALID / INCONCLUSIVE"
                    if candidate.get("evaluation_status") in {"invalid", "inconclusive"}
                    else "ACCEPTED" if candidate.get("accepted") is True else "REJECTED"
                )
                not_reached_replay = candidate_reason.startswith("latest_attack_filter:")
                service_rows.append((generation, candidate, metrics, candidate_status, not_reached_replay))
        if service_rows:
            lines += [
                "## Service candidate paired evidence", "",
                "Paired W/L/T values are read from persisted gate metrics. A staged candidate that did not reach a suite is shown as `Not evaluated`; absent legacy fields are not interpreted as zero.", "",
                "| Generation | Patch | Status | Gate reason | Delta | Latest W/L/T | Exact replay W/L/T | Transfer replay W/L/T | Normal W/L/T |",
                "|---:|---|---|---|---:|---:|---:|---:|---:|",
            ]
            for generation, candidate, metrics, candidate_status, not_reached_replay in service_rows:
                patch_id = candidate.get("patch_id") or (candidate.get("patch") or {}).get("patch_id") or "not recorded"
                pairs = ("latest_paired", "exact_replay_paired", "transfer_replay_paired", "normal_paired")
                cells = [
                    _format_pair(
                        metrics.get(key),
                        not_reached_replay=not_reached_replay and key != "latest_paired",
                        invalid=candidate_status == "INVALID / INCONCLUSIVE",
                    )
                    for key in pairs
                ]
                delta = candidate.get("delta")
                lines.append(
                    f"| {generation} | `{patch_id}` | {candidate_status} | "
                    f"{_markdown_cell(candidate.get('reason') or 'not recorded')} | {_format_metric(delta)} | "
                    + " | ".join(cells) + " |"
                )
            lines.append("")
        all_episodes = [
            row for marker in generations
            for row in _read_jsonl(marker.parent / "episodes.jsonl")
        ]
        attack_instances = run_dir / "archives" / "attack_instances.jsonl"
        overall = summarize_episode_reporting(
            all_episodes,
            attack_instance_count=_count_jsonl(attack_instances) if attack_instances.exists() else None,
        )
        legacy_note = "" if overall["unique_vulnerability_signatures_v2"] is not None else " (N/A: legacy schema)"
        lines += [
            "## Validity, attribution, and outcomes", "",
            "These are aggregates of structured episode fields; dialogue is not rescored. `N/A (legacy schema)` means the source artifact does not contain the required provenance.", "",
            f"- Evaluable episodes: {overall['valid_episodes']}/{overall['episodes']}",
            f"- Protocol-invalid rate: {_format_metric(overall['protocol_invalid_rate'], percent=True)}",
            f"- Protocol-evaluated episodes: {_format_metric(overall['protocol_evaluated_episodes'])}/{overall['episodes']}",
            f"- Customer-behavior-invalid rate: {_format_metric(overall['customer_behavior_invalid_rate'], percent=True)} ({_format_metric(overall['customer_behavior_evaluated_episodes'])} assessed)",
            f"- Attributable Service failure rate: {_format_metric(overall['attributable_service_failure_rate'], percent=True)}",
            f"- Attribution-evaluated episodes: {_format_metric(overall['attributable_episodes'])}",
            f"- Process-outcome-evaluated episodes: {_format_metric(overall['outcome_metrics_evaluated_episodes'])}",
            f"- Strict process success: {_format_outcome_metric(overall['strict_process_success'], all_episodes, 'strict_process_success')}",
            f"- Eventual goal success: {_format_outcome_metric(overall['eventual_goal_success'], all_episodes, 'eventual_goal_success')}",
            f"- Recovery success rate: {_format_outcome_metric(overall['recovery_success_rate'], all_episodes, 'recovery_success')}",
            f"- Unique V2 vulnerability signatures: {_format_metric(overall['unique_vulnerability_signatures_v2'])}{legacy_note}",
            f"- Exact attack instances: {_format_metric(overall['attack_instance_count'])}",
            "",
        ]
    heldout = run_dir / "analysis" / "heldout_results.json"
    if heldout.exists():
        lines += ["## Held-out evaluation", "", "Held-out metrics are reported only after evolution and are not fed back to an evolver.", ""]
    else:
        lines += ["## Held-out evaluation", "", "Not evaluated.", ""]
    cross = run_dir / "analysis" / "cross_generation_matrix.json"
    if cross.exists():
        data = json.loads(cross.read_text(encoding="utf-8"))
        lines += ["## Cross-generation evaluation", "", f"Evaluator: **{data.get('evaluator', 'unknown')}**; cells: {len(data.get('matrix', []))}.", ""]
    else:
        lines += ["## Cross-generation evaluation", "", "Not evaluated.", ""]
    fresh = sorted((run_dir / "analysis").glob("fresh_adversary_*.json"))
    if fresh:
        lines += ["## Fresh adaptive adversary", ""]
        for path in fresh:
            data = json.loads(path.read_text(encoding="utf-8"))
            lines.append(f"- `{data.get('target_service', path.stem)}`: evaluator **{data.get('evaluator', 'unknown')}**, rounds {len(data.get('rounds', []))}.")
        lines.append("")
    else:
        lines += ["## Fresh adaptive adversary", "", "Not evaluated.", ""]
    attack_path = run_dir / "archives" / "attacks.jsonl"
    defense_path = run_dir / "archives" / "defenses.jsonl"
    lines += ["## Archives", "", f"AttackArchive records: {_count_jsonl(attack_path)}", f"DefenseArchive records: {_count_jsonl(defense_path)}", ""]
    target = run_dir / "report.md"
    target.write_text("\n".join(lines), encoding="utf-8")
    return target


def _read_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _count_jsonl(path: Path) -> int:
    return len(_read_jsonl(path))


def _format_metric(value: Any, *, percent: bool = False) -> str:
    if value is None:
        return "N/A (legacy schema)"
    if percent:
        return f"{float(value):.1%}"
    return str(value)


def _format_outcome_metric(value: Any, rows: list[dict[str, Any]], field: str) -> str:
    if not any(field in row for row in rows):
        return "N/A (legacy schema)"
    if not any(isinstance(row.get("strict_process_success"), bool) for row in rows):
        return "Not evaluated"
    if value is None:
        return "Not recorded"
    return f"{float(value):.1%}"


def _format_pair(value: Any, *, not_reached_replay: bool = False, invalid: bool = False) -> str:
    if isinstance(value, dict):
        wins, losses, ties = (value.get(key) for key in ("wins", "losses", "ties"))
        if all(isinstance(item, int) for item in (wins, losses, ties)):
            return f"{wins}/{losses}/{ties}"
    if not_reached_replay:
        return "Not evaluated"
    if invalid:
        return "Invalid evaluation"
    return "Not recorded"


def _markdown_cell(value: Any) -> str:
    return " ".join(str(value).replace("|", "\\|").splitlines())
