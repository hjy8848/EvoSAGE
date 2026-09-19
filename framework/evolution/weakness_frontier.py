"""Weakness-frontier aggregation and portable CSV/JSON export."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .schemas import EpisodeResult


class WeaknessFrontier:
    def __init__(self):
        self.rows: dict[tuple, dict] = {}

    def add(self, episodes: Iterable[EpisodeResult]) -> None:
        grouped = defaultdict(list)
        for episode in episodes:
            for error in (episode.error_types or (["success"] if episode.task_success else ["unknown"])):
                node = episode.sop_node
                if not node and episode.path_step_index is not None:
                    node = f"path_step_{episode.path_step_index}"
                grouped[(episode.generation, node or "unknown", error)].append(episode)
        for (generation, node, failure_type), values in grouped.items():
            count = len(values)
            failures = sum(not value.task_success for value in values)
            key = (generation, node, failure_type)
            self.rows[key] = {
                "generation": generation,
                "sop_node": node,
                "failure_type": failure_type,
                "episodes": count,
                "failures": failures,
                "failure_rate": failures / count if count else 0.0,
                "attack_success_rate": failures / count if count else 0.0,
                "verification_failure_rate": sum(value.verification_score < 1 for value in values) / count,
                "action_failure_rate": sum(value.action_execution_score < 1 for value in values) / count,
                "goal_failure_rate": sum(value.goal_fulfillment_score < 1 for value in values) / count,
                "unique_customer_strategies": len({value.customer_policy_id for value in values}),
            }

    def to_dicts(self) -> list[dict]:
        return [self.rows[key] for key in sorted(self.rows)]

    def save(self, json_path: str | Path, csv_path: str | Path | None = None) -> None:
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(self.to_dicts(), ensure_ascii=False, indent=2), encoding="utf-8")
        if csv_path:
            csv_path = Path(csv_path)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            rows = self.to_dicts()
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["generation", "sop_node", "failure_type"])
                writer.writeheader()
                writer.writerows(rows)
