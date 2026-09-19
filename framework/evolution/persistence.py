"""Filesystem persistence for resumable co-evolution experiments."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable


def _now():
    return datetime.now(timezone.utc).isoformat()


class RunStore:
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        for name in ("config", "environment", "split_manifest", "generations", "archives", "traces", "analysis"):
            (self.run_dir / name).mkdir(parents=True, exist_ok=True)

    def write_json(self, relative: str, value: Any) -> Path:
        target = self.run_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        return target

    def append_jsonl(self, relative: str, values: Iterable[Any]) -> Path:
        target = self.run_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            for value in values:
                handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        return target

    def generation_dir(self, generation: int) -> Path:
        path = self.run_dir / "generations" / f"gen_{generation:03d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def mark_generation_complete(self, generation: int, payload: dict[str, Any] | None = None) -> None:
        value = {"generation": generation, "completed_at": _now(), **(payload or {})}
        self.write_json(f"generations/gen_{generation:03d}/COMPLETE.json", value)

    def completed_generations(self) -> list[int]:
        result = []
        root = self.run_dir / "generations"
        for marker in root.glob("gen_*/COMPLETE.json"):
            try:
                result.append(int(marker.parent.name.split("_")[-1]))
            except ValueError:
                continue
        return sorted(result)

    def latest_policy(self, name: str, default=None):
        generations = self.completed_generations()
        if not generations:
            return default
        path = self.run_dir / "generations" / f"gen_{generations[-1]:03d}" / f"{name}.json"
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))

    def read_generation(self, generation: int, name: str) -> dict:
        path = self.run_dir / "generations" / f"gen_{generation:03d}" / f"{name}.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def latest_policy_at(self, generation: int, name: str) -> dict:
        return self.read_generation(generation, name)
