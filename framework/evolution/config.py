"""Configuration for reproducible adversarial co-evolution runs.

The configuration is intentionally plain dataclasses so experiments can be
serialized without importing a framework-specific configuration library.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class CustomerEvolutionConfig:
    candidate_count: int = 5
    elite_count: int = 1
    cases_per_candidate: int = 4
    fitness_weights: Dict[str, float] = field(default_factory=lambda: {
        "attack_success": 0.70,
        "novelty": 0.15,
        "coverage": 0.15,
    })
    allowed_strategy_tags: list[str] = field(default_factory=lambda: [
        "truthful", "cooperative", "withholding", "pressure", "contradiction",
        "delayed_disclosure", "authority_challenge", "delayed_contradiction", "escalation", "paraphrase",
    ])
    adversary_access: str = "black_box"


@dataclass
class ServiceEvolutionConfig:
    candidate_count: int = 5
    min_delta: float = 0.01
    normal_regression_tolerance: float = 0.03
    replay_attack_count: int = 5
    allowed_rule_categories: list[str] = field(default_factory=lambda: [
        "VERIFICATION", "ACTION_GROUNDING", "RECOVERY", "TOOL_USE", "COMMUNICATION",
    ])


@dataclass
class SplitConfig:
    strategy: str = "instance_holdout"
    seed: int = 7
    evolution_ratio: float = 0.6
    validation_ratio: float = 0.2
    heldout_ratio: float = 0.2
    holdout_paths: list[int] = field(default_factory=list)
    instances_per_path: int = 1
    max_cases: Optional[int] = None


@dataclass
class EvaluationConfig:
    repetitions: int = 1
    max_turns: int = 10
    concurrency: int = 1
    api_timeout: int = 300


@dataclass
class PersistenceConfig:
    output_dir: str = "results/adversarial_coevolution"
    resume: bool = False


@dataclass
class FreshAdversaryConfig:
    enabled: bool = False
    rounds: int = 2
    candidate_count: int = 5


@dataclass
class EvolutionConfig:
    scenario: str = "ecommerce_refund"
    experiment_mode: str = "coevolution"
    seed: int = 7
    max_generations: int = 2
    customer: CustomerEvolutionConfig = field(default_factory=CustomerEvolutionConfig)
    service: ServiceEvolutionConfig = field(default_factory=ServiceEvolutionConfig)
    splits: SplitConfig = field(default_factory=SplitConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)
    fresh_adversary: FreshAdversaryConfig = field(default_factory=FreshAdversaryConfig)
    model_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Optional[Dict[str, Any]]) -> "EvolutionConfig":
        value = dict(value or {})
        nested = {
            "customer": CustomerEvolutionConfig,
            "service": ServiceEvolutionConfig,
            "splits": SplitConfig,
            "evaluation": EvaluationConfig,
            "persistence": PersistenceConfig,
            "fresh_adversary": FreshAdversaryConfig,
        }
        for key, constructor in nested.items():
            if isinstance(value.get(key), dict):
                value[key] = constructor(**value[key])
        return cls(**{key: item for key, item in value.items() if key in cls.__dataclass_fields__})


def load_config(path: str | Path) -> EvolutionConfig:
    """Load JSON or YAML when PyYAML is installed; JSON is always supported."""
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ValueError("Config is not JSON and PyYAML is not installed") from exc
        data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError("Evolution config must be a mapping")
    return EvolutionConfig.from_dict(data)


def save_config(config: EvolutionConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
