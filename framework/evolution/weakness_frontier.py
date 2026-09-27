"""V2 weakness-frontier aggregation and portable CSV/JSON export."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .schemas import EpisodeResult, FailureOccurrence


class WeaknessFrontier:
    def __init__(self):
        self.rows: dict[tuple, dict] = {}

    def add(self, episodes: Iterable[EpisodeResult]) -> None:
        grouped = defaultdict(list)
        for episode in episodes:
            if episode.is_evaluation_invalid():
                continue
            if episode.task_success:
                node = episode.sop_node or (
                    f"path_step_{episode.path_step_index}"
                    if episode.path_step_index is not None else "unknown"
                )
                grouped[(episode.generation, node, "UNKNOWN", "NO_FAILURE")].append(
                    (episode, "success", None)
                )
                continue

            signature = episode.vulnerability_signature_v2()
            if signature is None:
                # V1 or unclassified historical rows must not silently acquire
                # V2 identity during a new frontier aggregation.
                continue
            occurrence = FailureOccurrence.from_episode(episode, signature)
            node = signature.sop_node or (
                f"path_step_{episode.path_step_index}"
                if episode.path_step_index is not None else "unknown"
            )
            grouped[(
                episode.generation,
                node,
                signature.failure_stage,
                signature.violated_invariant,
            )].append((episode, signature.failure_type, occurrence.occurrence_id))

        for (generation, node, stage, invariant), values in grouped.items():
            count = len(values)
            failures = sum(not value.task_success for value, _, _ in values)
            primary_errors = sorted({error for _, error, _ in values if error != "success"})
            primary_error = primary_errors[0] if len(primary_errors) == 1 else (
                "MULTIPLE" if primary_errors else "success"
            )
            key = (generation, node, stage, invariant)
            self.rows[key] = {
                "schema_version": 2,
                "generation": generation,
                "sop_node": node,
                "failure_stage": stage,
                "violated_invariant": invariant,
                "primary_error": primary_error,
                "primary_errors": primary_errors,
                "failure_type": primary_error,  # legacy reporting alias
                "episodes": count,
                "failures": failures,
                "failure_rate": failures / count if count else 0.0,
                "attack_success_rate": failures / count if count else 0.0,
                "verification_failure_rate": sum(value.verification_score < 1 for value, _, _ in values) / count,
                "action_failure_rate": sum(value.action_execution_score < 1 for value, _, _ in values) / count,
                "goal_failure_rate": sum(value.goal_fulfillment_score < 1 for value, _, _ in values) / count,
                "unique_customer_strategies": len({value.customer_policy_id for value, _, _ in values}),
                "unique_attack_instances": len({occurrence for _, _, occurrence in values if occurrence}),
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
            writer_fields = list(rows[0]) if rows else [
                "generation", "sop_node", "failure_stage", "violated_invariant"
            ]
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=writer_fields)
                writer.writeheader()
                writer.writerows(rows)
