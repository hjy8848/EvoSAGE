"""Persistent, deduplicated attack and defense archives."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Iterable, Optional

from .schemas import CustomerPolicy, DefenseRecord, EpisodeResult, FailureSignature


def _key(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]


class AttackArchive:
    def __init__(self, path: Optional[str | Path] = None):
        self.path = Path(path) if path else None
        self.attacks: dict[str, dict] = {}
        if self.path and self.path.exists():
            self.load_jsonl(self.path)

    def add(self, policy: CustomerPolicy, failure_signatures: Iterable[FailureSignature],
            episodes: Iterable[EpisodeResult] = (), generation: Optional[int] = None) -> int:
        episodes = list(episodes)
        if any(episode.split == "heldout_test" for episode in episodes):
            raise AssertionError("AttackArchive cannot ingest heldout episodes")
        added = 0
        for signature in failure_signatures:
            if "json_parse_failed" in signature.error_types or "protocol_failure" in signature.error_types:
                continue
            key = _key({"tags": sorted(policy.strategy_tags), "node": signature.sop_node,
                        "errors": sorted(signature.error_types), "action": signature.predicted_action})
            record = {
                "attack_id": "attack_" + key,
                "customer_policy_id": policy.policy_id,
                "generation_discovered": policy.generation if generation is None else generation,
                "customer_policy": policy.to_dict(),
                "generation": policy.generation if generation is None else generation,
                "strategy_tags": list(policy.strategy_tags),
                "target_sop_node": signature.sop_node,
                "target_path_step_index": signature.path_step_index,
                "induced_error_types": list(signature.error_types),
                "failure_signature": signature.to_dict(),
                "source_case_ids": sorted({episode.case_id for episode in episodes if not episode.task_success}),
                "attack_success_rate": sum(not episode.task_success for episode in episodes) / len(episodes) if episodes else 0.0,
                "novelty_signature": key,
                "transfer_success_rate": None,
                "active": True,
            }
            if key not in self.attacks:
                self.attacks[key] = record
                added += 1
        if self.path and added:
            self.save_jsonl(self.path)
        return added

    def contains_signature(self, signature: FailureSignature) -> bool:
        return any(item.get("failure_signature", {}).get("signature_id") == signature.signature_id for item in self.attacks.values())

    def signatures(self) -> list[dict]:
        return [item["failure_signature"] for item in self.attacks.values()]

    def __len__(self) -> int:
        return len(self.attacks)

    def to_dicts(self) -> list[dict]:
        return list(self.attacks.values())

    def save_jsonl(self, path: Optional[str | Path] = None) -> None:
        target = Path(path or self.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in self.attacks.values()), encoding="utf-8")

    def load_jsonl(self, path: Optional[str | Path] = None) -> None:
        target = Path(path or self.path)
        for line in target.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                self.attacks[item.get("attack_id", _key(item))] = item


class DefenseArchive:
    def __init__(self, path: Optional[str | Path] = None):
        self.path = Path(path) if path else None
        self.defenses: dict[str, dict] = {}
        if self.path and self.path.exists():
            self.load_jsonl(self.path)

    def add(self, record: DefenseRecord) -> bool:
        key = record.defense_id or _key(record.to_dict())
        if key in self.defenses:
            return False
        self.defenses[key] = record.to_dict()
        if self.path:
            self.save_jsonl(self.path)
        return True

    def __len__(self) -> int:
        return len(self.defenses)

    def to_dicts(self) -> list[dict]:
        return list(self.defenses.values())

    def save_jsonl(self, path: Optional[str | Path] = None) -> None:
        target = Path(path or self.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in self.defenses.values()), encoding="utf-8")

    def load_jsonl(self, path: Optional[str | Path] = None) -> None:
        target = Path(path or self.path)
        for line in target.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                self.defenses[item["defense_id"]] = item
