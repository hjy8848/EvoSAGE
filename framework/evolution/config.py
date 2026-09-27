"""Configuration for reproducible adversarial co-evolution runs.

The configuration is intentionally plain dataclasses so experiments can be
serialized without importing a framework-specific configuration library.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Optional
import warnings

from ..backend.tool_contract import ToolContractConfig


@dataclass
class CustomerEvolutionConfig:
    candidate_count: int = 5
    elite_count: int = 1
    cases_per_candidate: int = 4
    fitness_weights: Dict[str, float] = field(default_factory=lambda: {
        "attack_success": 0.70,
        "novelty": 0.15,
        "node_diversity": 0.15,
    })
    allowed_strategy_tags: list[str] = field(default_factory=lambda: [
        "truthful", "cooperative", "withholding", "pressure", "contradiction",
        "delayed_disclosure", "authority_challenge", "delayed_contradiction", "escalation", "paraphrase",
    ])
    adversary_access: str = "black_box"

    def __post_init__(self) -> None:
        if "coverage" in self.fitness_weights:
            if "node_diversity" not in self.fitness_weights:
                self.fitness_weights["node_diversity"] = self.fitness_weights["coverage"]
            self.fitness_weights.pop("coverage", None)
            warnings.warn(
                "customer fitness weight 'coverage' is deprecated; interpreted as 'node_diversity'",
                DeprecationWarning,
                stacklevel=2,
            )


@dataclass
class ServiceEvolutionConfig:
    candidate_count: int = 5
    min_delta: float = 0.01
    normal_regression_tolerance: float = 0.03
    gate_min_paired_wins: int = 1
    max_normal_paired_losses: int = 0
    replay_attack_count: int = 5
    exact_replay_instance_count: int = 5
    transfer_replay_policy_count: Optional[int] = None
    allowed_rule_categories: list[str] = field(default_factory=lambda: [
        "VERIFICATION", "ACTION_GROUNDING", "RECOVERY", "TOOL_USE", "COMMUNICATION",
    ])

    def __post_init__(self) -> None:
        if self.gate_min_paired_wins < 1:
            raise ValueError("gate_min_paired_wins must be at least 1")
        if self.max_normal_paired_losses < 0:
            raise ValueError("max_normal_paired_losses cannot be negative")
        if self.transfer_replay_policy_count is None:
            self.transfer_replay_policy_count = int(self.replay_attack_count)


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
    customer_disclosure_variants: list[str] = field(default_factory=lambda: ["opening"])

    def __post_init__(self) -> None:
        supported = {"opening", "on_request"}
        if not self.customer_disclosure_variants or any(
            variant not in supported for variant in self.customer_disclosure_variants
        ):
            raise ValueError("customer_disclosure_variants must contain opening/on_request variants")


@dataclass
class TokenBudgetConfig:
    """Role-specific completion budgets for API calls.

    These limits keep orchestration requests bounded without changing the
    benchmark's SOP or tool semantics.
    """

    user: int = 512
    agent: int = 1536
    judge: int = 1024
    customer_evolver: int = 4096
    service_evolver: int = 4096


@dataclass
class EvaluationConfig:
    repetitions: int = 1
    max_turns: int = 10
    concurrency: int = 1
    api_timeout: int = 300
    # Evolution uses objective Backend/V/P/A/G signals by default.  Full LLM
    # Judge scoring remains available for final/held-out evaluation phases.
    judge_in_evolution: bool = False
    token_budget: TokenBudgetConfig = field(default_factory=TokenBudgetConfig)
    summary_limit: int = 5
    # None preserves provider defaults; explicit values currently target the
    # Customer simulator only, leaving Agent/Evolver/Judge requests unchanged.
    customer_thinking_mode: Optional[str] = None
    tool_contract: ToolContractConfig = field(default_factory=ToolContractConfig)

    def __post_init__(self) -> None:
        if self.customer_thinking_mode not in {None, "enabled", "disabled"}:
            raise ValueError("customer_thinking_mode must be None, 'enabled', or 'disabled'")

    @classmethod
    def from_dict(cls, value: Optional[Dict[str, Any]]) -> "EvaluationConfig":
        value = dict(value or {})
        token_budget = value.get("token_budget")
        if isinstance(token_budget, dict):
            value["token_budget"] = TokenBudgetConfig(**token_budget)
        tool_contract = value.get("tool_contract")
        if tool_contract is not None:
            value["tool_contract"] = ToolContractConfig.from_value(tool_contract)
        return cls(**{key: item for key, item in value.items() if key in cls.__dataclass_fields__})


@dataclass
class PersistenceConfig:
    output_dir: str = "results/adversarial_coevolution"
    resume: bool = False


@dataclass
class FreshAdversaryConfig:
    enabled: bool = False
    rounds: int = 2
    candidate_count: int = 5
    mode: str = "in_family"
    generator_model: Optional[str] = None
    generator_provider: Optional[str] = None

    def __post_init__(self) -> None:
        if self.rounds < 1:
            raise ValueError("fresh_adversary.rounds must be at least 1")
        if self.candidate_count < 1:
            raise ValueError("fresh_adversary.candidate_count must be at least 1")
        if self.mode not in {"in_family", "cross_generator"}:
            raise ValueError("fresh_adversary.mode must be in_family or cross_generator")
        if self.mode == "cross_generator" and (
            not self.generator_model or not self.generator_provider
        ):
            raise ValueError(
                "cross_generator mode requires generator_model and generator_provider"
            )


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
            if key == "evaluation" and isinstance(value.get(key), dict):
                value[key] = EvaluationConfig.from_dict(value[key])
            elif key == "service" and isinstance(value.get(key), dict):
                service_value = dict(value[key])
                if "replay_attack_count" in service_value and "transfer_replay_policy_count" not in service_value:
                    service_value["transfer_replay_policy_count"] = service_value["replay_attack_count"]
                    warnings.warn(
                        "service.replay_attack_count is deprecated; interpreted as "
                        "transfer_replay_policy_count",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                value[key] = constructor(**service_value)
            elif isinstance(value.get(key), dict):
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
