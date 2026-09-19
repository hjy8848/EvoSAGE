"""Deterministic, persisted experiment/validation/held-out manifests."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import copy
import json
import random
from pathlib import Path
from typing import Any, Iterable, Optional

from ..backend.factory import build_case_spec
from .config import SplitConfig


@dataclass
class DatasetCase:
    case_id: str
    split: str
    path_id: int
    instance_index: int
    intent: str
    path_config: dict[str, Any]
    case_spec: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetSplits:
    evolution: list[DatasetCase]
    validation: list[DatasetCase]
    heldout_test: list[DatasetCase]
    seed: int
    strategy: str

    @property
    def heldout(self) -> list[DatasetCase]:
        return self.heldout_test

    @property
    def all_cases(self) -> list[DatasetCase]:
        return self.evolution + self.validation + self.heldout_test

    def assert_no_heldout(self, cases: Iterable[DatasetCase | str]) -> None:
        heldout_ids = {item.case_id for item in self.heldout_test}
        found = {item if isinstance(item, str) else item.case_id for item in cases}
        overlap = heldout_ids & found
        if overlap:
            raise AssertionError(f"heldout cases entered an evolver: {sorted(overlap)}")


class SplitManager:
    def __init__(self, config: Optional[SplitConfig] = None, manifest_dir: Optional[str | Path] = None):
        self.config = config or SplitConfig()
        self.manifest_dir = Path(manifest_dir) if manifest_dir else None

    def build(self) -> DatasetSplits:
        if self.config.strategy not in {"instance_holdout", "path_holdout"}:
            raise ValueError(f"unknown split strategy: {self.config.strategy}")
        from ..sop import ecommerce_refund_PathList
        paths = ecommerce_refund_PathList.generate_path_list()
        mapping = ecommerce_refund_PathList.get_intent_path_mapping()
        path_to_intent = {}
        for intent, item in mapping.items():
            for path_id in item.get("possible_paths", []):
                path_to_intent.setdefault(path_id, intent)
        cases: list[DatasetCase] = []
        for path_id, path_config in enumerate(paths, start=1):
            intent = path_to_intent.get(path_id, "refund_before_shipping")
            for instance_index in range(max(1, self.config.instances_per_path)):
                user_id = f"evolution_case_{path_id}_{instance_index}"
                case_spec = build_case_spec("ecommerce_refund", intent, copy.deepcopy(path_config), user_id=user_id)
                cases.append(DatasetCase(
                    case_id=case_spec.case_id,
                    split="",
                    path_id=path_id,
                    instance_index=instance_index,
                    intent=intent,
                    path_config=copy.deepcopy(path_config),
                    case_spec=case_spec.to_dict(),
                ))
        rng = random.Random(self.config.seed)
        if self.config.max_cases is not None:
            rng.shuffle(cases)
            cases = cases[:max(0, self.config.max_cases)]
        if self.config.strategy == "path_holdout":
            heldout_paths = set(self.config.holdout_paths or [len(paths)])
            heldout = [item for item in cases if item.path_id in heldout_paths]
            pool = [item for item in cases if item.path_id not in heldout_paths]
            rng.shuffle(pool)
            evolution, validation = self._ratio_split(pool)
        else:
            rng.shuffle(cases)
            evolution, validation, heldout = self._three_way_split(cases)
        for split, values in (("evolution", evolution), ("validation", validation), ("heldout_test", heldout)):
            for item in values:
                item.split = split
        result = DatasetSplits(evolution, validation, heldout, self.config.seed, self.config.strategy)
        if self.manifest_dir:
            self.persist(result, self.manifest_dir)
        return result

    def _ratio_split(self, cases: list[DatasetCase]) -> tuple[list[DatasetCase], list[DatasetCase]]:
        cut = round(len(cases) * self.config.evolution_ratio / max(0.01, self.config.evolution_ratio + self.config.validation_ratio))
        return cases[:cut], cases[cut:]

    def _three_way_split(self, cases: list[DatasetCase]) -> tuple[list[DatasetCase], list[DatasetCase], list[DatasetCase]]:
        n = len(cases)
        evolution_end = int(n * self.config.evolution_ratio)
        validation_end = evolution_end + int(n * self.config.validation_ratio)
        if n and evolution_end == n:
            evolution_end -= 1
        if n > 1 and validation_end <= evolution_end:
            validation_end = min(n - 1, evolution_end + 1)
        return cases[:evolution_end], cases[evolution_end:validation_end], cases[validation_end:]

    @staticmethod
    def persist(splits: DatasetSplits, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name, values in (("evolution_cases", splits.evolution), ("validation_cases", splits.validation), ("heldout_cases", splits.heldout_test)):
            (directory / f"{name}.json").write_text(json.dumps({
                "seed": splits.seed, "strategy": splits.strategy,
                "cases": [item.to_dict() for item in values],
            }, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def load(directory: str | Path) -> DatasetSplits:
        directory = Path(directory)
        values = {}
        for name, attr in (("evolution_cases", "evolution"), ("validation_cases", "validation"), ("heldout_cases", "heldout_test")):
            data = json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))
            values[attr] = [DatasetCase(**item) for item in data.get("cases", [])]
            values.setdefault("seed", data.get("seed", 7))
            values.setdefault("strategy", data.get("strategy", "instance_holdout"))
        return DatasetSplits(**values)

    @staticmethod
    def case_spec(case: DatasetCase):
        from ..backend.types import CaseSpec
        return CaseSpec(**copy.deepcopy(case.case_spec))
