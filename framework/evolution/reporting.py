"""Research report generation without inventing missing metrics."""

from __future__ import annotations

import json
from pathlib import Path


def generate_report(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    generations = sorted((run_dir / "generations").glob("gen_*/COMPLETE.json"))
    lines = ["# EvoSAGE adversarial co-evolution report", "", f"Run directory: `{run_dir}`", ""]
    if not generations:
        lines += ["No completed generation was found; metrics are not evaluated.", ""]
    else:
        lines += ["## Completed generations", "", "| Generation | Status |", "|---:|---|"]
        for marker in generations:
            data = json.loads(marker.read_text(encoding="utf-8"))
            lines.append(f"| {data.get('generation', marker.parent.name)} | complete |")
        lines.append("")
    heldout = run_dir / "analysis" / "heldout_results.json"
    if heldout.exists():
        lines += ["## Held-out evaluation", "", "Held-out metrics are reported only after evolution and are not fed back to an evolver.", ""]
    else:
        lines += ["## Held-out evaluation", "", "Not evaluated.", ""]
    target = run_dir / "report.md"
    target.write_text("\n".join(lines), encoding="utf-8")
    return target
