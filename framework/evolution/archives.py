"""Persistent, deduplicated attack and defense archives."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Iterable, Optional

from .schemas import (
    AttackInstance,
    CustomerPolicy,
    DefenseRecord,
    EpisodeResult,
    FailureOccurrence,
    FailureSignature,
    VulnerabilityArchiveEntry,
    VulnerabilitySignature,
)


def _key(value) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


class AttackArchive:
    """Versioned vulnerability index plus replayable historical instances.

    ``path`` remains the compatibility summary archive. New exact replay data
    is persisted separately so legacy policy summaries cannot masquerade as
    case-grounded attack instances.
    """

    def __init__(self, path: Optional[str | Path] = None,
                 instance_path: Optional[str | Path] = None,
                 vulnerability_path: Optional[str | Path] = None):
        self.path = Path(path) if path else None
        root = self.path.parent if self.path else None
        self.instance_path = Path(instance_path) if instance_path else (root / "attack_instances.jsonl" if root else None)
        self.vulnerability_path = Path(vulnerability_path) if vulnerability_path else (root / "vulnerabilities.jsonl" if root else None)
        self.attacks: dict[str, dict] = {}
        self.instances: dict[str, AttackInstance] = {}
        self.vulnerabilities: dict[str, VulnerabilityArchiveEntry] = {}
        if self.path and self.path.exists():
            self.load_jsonl(self.path)
        if self.instance_path and self.instance_path.exists():
            self._load_instances(self.instance_path)
        if self.vulnerability_path and self.vulnerability_path.exists():
            self._load_vulnerabilities(self.vulnerability_path)

    @staticmethod
    def _case_payload(value):
        if value is None:
            return None, None
        if hasattr(value, "to_dict"):
            descriptor = value.to_dict()
            return descriptor.get("case_spec"), descriptor
        if isinstance(value, dict):
            if isinstance(value.get("case_spec"), dict):
                return value["case_spec"], value
            return value, None
        return None, None

    def add(self, policy: CustomerPolicy,
            failure_signatures: Iterable[VulnerabilitySignature | FailureSignature],
            episodes: Iterable[EpisodeResult] = (), generation: Optional[int] = None,
            case_specs_by_id: Optional[dict[str, Any]] = None,
            occurrences: Optional[Iterable[FailureOccurrence]] = None,
            reproduction_seed: int = 0) -> int:
        episodes = list(episodes)
        if any(episode.split == "heldout_test" for episode in episodes):
            raise AssertionError("AttackArchive cannot ingest heldout episodes")
        valid_episodes = [episode for episode in episodes if not episode.is_evaluation_invalid()]
        signatures_by_id = {
            item.signature_id: item for item in failure_signatures
            if isinstance(item, VulnerabilitySignature) and item.schema_version == 2
        }
        occurrences_by_episode = {item.episode_id: item for item in (occurrences or [])}
        added = 0
        changed = False
        cohort_failure_count = sum(not episode.task_success for episode in valid_episodes)
        for episode in valid_episodes:
            if episode.task_success:
                continue
            signature = episode.vulnerability_signature_v2()
            if signature is None or signature.signature_id not in signatures_by_id:
                continue
            signature = signatures_by_id[signature.signature_id]
            occurrence = occurrences_by_episode.get(episode.episode_id)
            if occurrence is None:
                occurrence = FailureOccurrence.from_episode(episode, signature)

            case_value = (case_specs_by_id or {}).get(episode.case_id)
            case_spec, dataset_case = self._case_payload(case_value)
            instance = None
            if isinstance(case_spec, dict) and case_spec.get("case_id") == episode.case_id:
                instance = AttackInstance.from_episode(
                    episode, signature, occurrence, policy, case_spec,
                    reproduction_seed=reproduction_seed,
                    dataset_case=dataset_case,
                    generation=generation,
                )
                if instance.attack_instance_id not in self.instances:
                    self.instances[instance.attack_instance_id] = instance

            vulnerability = self.vulnerabilities.get(signature.signature_id)
            if vulnerability is None:
                vulnerability = VulnerabilityArchiveEntry(signature=signature.to_dict())
                self.vulnerabilities[signature.signature_id] = vulnerability
                changed = True
            if instance is not None and instance.attack_instance_id not in vulnerability.instance_ids:
                vulnerability.instance_ids.append(instance.attack_instance_id)
                changed = True
            associated = [
                value for value in self.instances.values()
                if value.vulnerability_signature_id == signature.signature_id
            ]
            vulnerability.distinct_case_count = len({value.case_id for value in associated})
            vulnerability.distinct_customer_policy_count = len({value.customer_policy_id for value in associated})
            previous_incidence = vulnerability.signature_incidence_rate
            # This is incidence within the evaluated policy cohort, not the
            # probability that the policy fails for any reason.
            signature_episode_count = sum(
                not item.task_success
                and (item.vulnerability_signature_v2() is not None)
                and item.vulnerability_signature_v2().signature_id == signature.signature_id
                for item in valid_episodes
            )
            vulnerability.signature_incidence_rate = (
                signature_episode_count / len(valid_episodes) if valid_episodes else None
            )
            if vulnerability.signature_incidence_rate != previous_incidence:
                changed = True

            summary_key = _key({
                "schema_version": 2,
                "signature_id": signature.signature_id,
                "customer_policy_id": policy.policy_id,
            })
            occurrence_dict = occurrence.to_dict()
            summary = {
                "schema_version": 2,
                "attack_id": "attack_" + summary_key,
                "customer_policy_id": policy.policy_id,
                "generation_discovered": policy.generation if generation is None else generation,
                "customer_policy": policy.to_dict(),
                "generation": policy.generation if generation is None else generation,
                "strategy_tags": list(policy.strategy_tags),
                "target_sop_node": signature.sop_node,
                "target_path_step_index": episode.path_step_index,
                "induced_error_types": list(occurrence.error_types),
                "vulnerability_signature": signature.to_dict(),
                "failure_signature": signature.to_dict(),
                "failure_occurrences": [occurrence_dict],
                "attack_instance_ids": list(vulnerability.instance_ids),
                "source_case_ids": sorted({item.case_id for item in associated}),
                "policy_attack_success_rate": cohort_failure_count / len(valid_episodes) if valid_episodes else None,
                "signature_incidence_rate": vulnerability.signature_incidence_rate,
                "attack_success_rate": cohort_failure_count / len(valid_episodes) if valid_episodes else None,
                "novelty_signature": summary_key,
                "transfer_success_rate": None,
                "active": True,
            }
            if summary_key not in self.attacks:
                self.attacks[summary_key] = summary
                added += 1
                changed = True
            else:
                existing = self.attacks[summary_key]
                known_occurrences = {
                    item.get("occurrence_id") for item in existing.get("failure_occurrences", [])
                }
                if occurrence.occurrence_id not in known_occurrences:
                    existing.setdefault("failure_occurrences", []).append(occurrence_dict)
                    changed = True
        self._refresh_summary_instances()
        if changed:
            self._persist()
        return added

    def _refresh_summary_instances(self) -> None:
        by_id = {key: value.to_dict() for key, value in self.vulnerabilities.items()}
        for record in self.attacks.values():
            sid = (record.get("vulnerability_signature") or {}).get("signature_id")
            vulnerability = by_id.get(sid)
            if vulnerability:
                policy_id = record.get("customer_policy_id")
                matching_instances = [
                    self.instances[item]
                    for item in vulnerability.get("instance_ids", [])
                    if item in self.instances
                    and self.instances[item].customer_policy_id == policy_id
                ]
                record["attack_instance_ids"] = [
                    item.attack_instance_id for item in matching_instances
                ]
                record["source_case_ids"] = sorted({
                    item.case_id for item in matching_instances
                })

    def _persist(self) -> None:
        if self.path:
            self.save_jsonl(self.path)
        if self.instance_path:
            self._save_records(self.instance_path, (item.to_dict() for item in self.instances.values()))
        if self.vulnerability_path:
            self._save_records(self.vulnerability_path, (item.to_dict() for item in self.vulnerabilities.values()))

    @staticmethod
    def _save_records(path: Path, records) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records), encoding="utf-8")

    def contains_signature(self, signature: VulnerabilitySignature | FailureSignature) -> bool:
        return isinstance(signature, VulnerabilitySignature) and signature.signature_id in self.vulnerabilities

    def signatures(self) -> list[VulnerabilitySignature]:
        return [
            VulnerabilitySignature.from_dict(entry.signature)
            for entry in self.vulnerabilities.values()
        ]

    def exact_instances(self, limit: Optional[int] = None, diverse: bool = True) -> list[AttackInstance]:
        candidates = sorted(
            (item for item in self.instances.values() if item.active and item.source_split != "heldout_test"),
            key=lambda item: (-item.generation_discovered, item.attack_instance_id),
        )
        if not diverse:
            selected = candidates
        else:
            selected = []
            remaining = list(candidates)
            seen_signatures, seen_stages, seen_strategies, seen_cases = set(), set(), set(), set()
            while remaining and (limit is None or len(selected) < limit):
                index = max(range(len(remaining)), key=lambda i: (
                    remaining[i].vulnerability_signature_id not in seen_signatures,
                    remaining[i].failure_stage not in seen_stages,
                    tuple(sorted(remaining[i].strategy_tags)) not in seen_strategies,
                    remaining[i].case_id not in seen_cases,
                    remaining[i].generation_discovered,
                    remaining[i].attack_instance_id,
                ))
                item = remaining.pop(index)
                selected.append(item)
                seen_signatures.add(item.vulnerability_signature_id)
                seen_stages.add(item.failure_stage)
                seen_strategies.add(tuple(sorted(item.strategy_tags)))
                seen_cases.add(item.case_id)
        return selected[:max(0, limit)] if limit is not None else selected

    def policies(self, limit: Optional[int] = None, diverse: bool = True) -> list[CustomerPolicy]:
        records = list(self.attacks.values()) + [item.to_dict() for item in self.instances.values()]
        by_policy = {}
        for record in sorted(records, key=lambda item: (-int(item.get("generation_discovered", 0)), str(item.get("customer_policy_id", "")))):
            payload = record.get("customer_policy")
            if isinstance(payload, dict) and payload.get("policy_id"):
                by_policy.setdefault(payload["policy_id"], payload)
        values = [CustomerPolicy.from_dict(item) for item in by_policy.values()]
        if limit is not None:
            values = values[:max(0, limit)]
        return values

    def __len__(self) -> int:
        return len(self.attacks)

    def to_dicts(self) -> list[dict]:
        return list(self.attacks.values())

    def save_jsonl(self, path: Optional[str | Path] = None) -> None:
        target = Path(path or self.path)
        self._save_records(target, self.attacks.values())

    def load_jsonl(self, path: Optional[str | Path] = None) -> None:
        target = Path(path or self.path)
        for line in target.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                self.attacks[item.get("attack_id", _key(item))] = item

    def _load_instances(self, path: Path) -> None:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                instance = AttackInstance.from_dict(json.loads(line))
                self.instances[instance.attack_instance_id] = instance

    def _load_vulnerabilities(self, path: Path) -> None:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entry = VulnerabilityArchiveEntry.from_dict(json.loads(line))
                if entry.signature.get("schema_version") == 2:
                    self.vulnerabilities[entry.signature_id] = entry


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
