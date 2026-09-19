"""Research report generation without inventing missing metrics."""

from __future__ import annotations

import json
from pathlib import Path


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
        lines += ["## Completed generations", "", "| Generation | Customer | Customer fitness | Service gate | Validation |", "|---:|---|---:|---|---:|"]
        for marker in generations:
            data = json.loads(marker.read_text(encoding="utf-8"))
            generation = data.get("generation", marker.parent.name)
            directory = marker.parent
            selected = json.loads((directory / "customer_candidates.json").read_text(encoding="utf-8")) if (directory / "customer_candidates.json").exists() else {}
            fitness = selected.get("scores", [])
            selected_id = selected.get("selected_policy", {}).get("policy_id", "not evaluated")
            selected_fitness = next((item.get("fitness", 0.0) for item in fitness if item.get("policy_id") == selected_id), "not evaluated")
            gate = json.loads((directory / "service_gate.json").read_text(encoding="utf-8")) if (directory / "service_gate.json").exists() else {}
            episodes = list(_read_jsonl(directory / "episodes.jsonl"))
            validation = sum(bool(item.get("task_success")) for item in episodes) / len(episodes) if episodes else "not evaluated"
            lines.append(f"| {generation} | `{selected_id}` | {selected_fitness} | {gate.get('reason', 'not evaluated')} | {validation} |")
        lines.append("")
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
