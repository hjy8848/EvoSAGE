"""Persistent, deduplicated attack and defense archives."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Iterable, Optional

from .schemas import (
    CustomerPolicy,
    DefenseRecord,
    EpisodeResult,
    FailureOccurrence,
    FailureSignature,
    VulnerabilitySignature,
)


def _key(value) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


class AttackArchive:
    def __init__(self, path: Optional[str | Path] = None):
        self.path = Path(path) if path else None
        self.attacks: dict[str, dict] = {}
        if self.path and self.path.exists():
            self.load_jsonl(self.path)

    def add(self, policy: CustomerPolicy, failure_signatures: Iterable[VulnerabilitySignature | FailureSignature],
            episodes: Iterable[EpisodeResult] = (), generation: Optional[int] = None) -> int:
        episodes = list(episodes)
        if any(episode.split == "heldout_test" for episode in episodes):
            raise AssertionError("AttackArchive cannot ingest heldout episodes")
        legitimate_episodes = [
            episode for episode in episodes
            if not episode.is_evaluation_invalid()
        ]
        legitimate_signature_ids = {
            signature.signature_id
            for episode in legitimate_episodes
            if not episode.task_success
            for signature in [episode.vulnerability_signature_v2()]
            if signature is not None
        }
        added = 0
        for signature in failure_signatures:
            # V1 signatures contain occurrence fields in their hash. Keep them
            # readable in existing archives, but never merge them into V2.
            if not isinstance(signature, VulnerabilitySignature) or signature.schema_version != 2:
                continue
            if episodes and not legitimate_episodes:
                continue
            if episodes and signature.signature_id not in legitimate_signature_ids:
                continue
            matching_episodes = [
                episode for episode in legitimate_episodes
                if not episode.task_success
                and episode.vulnerability_signature_v2() is not None
                and episode.vulnerability_signature_v2().signature_id == signature.signature_id
            ]
            occurrences = [
                FailureOccurrence.from_episode(episode, signature).to_dict()
                for episode in matching_episodes
            ]
            key = _key({
                "schema_version": 2,
                "signature_id": signature.signature_id,
                "customer_policy_id": policy.policy_id,
            })
            record = {
                "schema_version": 2,
                "attack_id": "attack_" + key,
                "customer_policy_id": policy.policy_id,
                "generation_discovered": policy.generation if generation is None else generation,
                "customer_policy": policy.to_dict(),
                "generation": policy.generation if generation is None else generation,
                "strategy_tags": list(policy.strategy_tags),
                "target_sop_node": signature.sop_node,
                "target_path_step_index": matching_episodes[0].path_step_index if matching_episodes else None,
                "induced_error_types": sorted({error for item in occurrences for error in item["error_types"]}),
                "vulnerability_signature": signature.to_dict(),
                "failure_signature": signature.to_dict(),
                "failure_occurrences": occurrences,
                "source_case_ids": sorted({episode.case_id for episode in matching_episodes}),
                "attack_success_rate": sum(
                    not episode.task_success
                    for episode in legitimate_episodes
                ) / len(legitimate_episodes) if legitimate_episodes else 0.0,
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

    def contains_signature(self, signature: VulnerabilitySignature | FailureSignature) -> bool:
        if not isinstance(signature, VulnerabilitySignature) or signature.schema_version != 2:
            return False
        return any(
            item.get("schema_version") == 2
            and (item.get("vulnerability_signature") or {}).get("signature_id") == signature.signature_id
            for item in self.attacks.values()
        )

    def signatures(self) -> list[dict]:
        return [
            item.get("vulnerability_signature") or item.get("failure_signature")
            for item in self.attacks.values()
            if item.get("schema_version") == 2
            and isinstance(item.get("vulnerability_signature") or item.get("failure_signature"), dict)
        ]

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
