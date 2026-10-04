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
    """Only the proposal parameters used by the Customer search core."""

    strategy_schema: str = "deceptive_free_text_v1"
    candidate_count: int = 5


@dataclass
class LegacyCustomerEvolutionConfig(CustomerEvolutionConfig):
    """Customer knobs retained only for the older combined runner."""

    elite_count: int = 1
    cases_per_candidate: int = 4

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
class CustomerSearchTokenBudgetConfig:
    """Role budgets needed by Customer-only runtime; no Service-Evolver slot."""

    user: int = 512
    agent: int = 1536
    judge: int = 1024
    customer_evolver: int = 4096

    @classmethod
    def from_value(cls, value) -> "CustomerSearchTokenBudgetConfig":
        if hasattr(value, "__dataclass_fields__"):
            value = asdict(value)
        value = dict(value or {})
        return cls(**{
            key: item for key, item in value.items()
            if key in cls.__dataclass_fields__
        })


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
    # roles that accept provider-specific thinking controls.
    customer_thinking_mode: Optional[str] = None
    agent_thinking_mode: Optional[str] = None
    evolver_thinking_mode: Optional[str] = None
    # These are maximum total transport attempts per client.generate call
    # (max_retries=1 therefore means one provider request, not one retry).
    agent_max_retries: int = 1
    customer_transport_max_retries: int = 1
    evolver_max_retries: int = 1
    judge_max_retries: int = 1
    judge_validation_retries: int = 2
    customer_protocol_retries: int = 1
    evolver_protocol_retries: int = 1
    rate_limit_backoff_seconds: float = 0.0
    # Preserve historical evaluator retry and tool-loop behavior for configs
    # that predate these fields. Fast profiles opt down explicitly.
    invalid_evaluation_retries: int = 1
    max_tool_steps: int = 8
    max_api_requests_per_generation: Optional[int] = None
    max_api_requests_per_run: Optional[int] = None
    tool_contract: ToolContractConfig = field(default_factory=ToolContractConfig)

    def __post_init__(self) -> None:
        for field_name in ("customer_thinking_mode", "agent_thinking_mode", "evolver_thinking_mode"):
            if getattr(self, field_name) not in {None, "enabled", "disabled"}:
                raise ValueError(f"{field_name} must be None, 'enabled', or 'disabled'")
        for field_name in (
            "agent_max_retries", "customer_transport_max_retries",
            "evolver_max_retries", "judge_max_retries",
        ):
            if int(getattr(self, field_name)) < 1:
                raise ValueError(f"{field_name} is a total-attempt count and must be at least 1")
        if self.invalid_evaluation_retries < 0:
            raise ValueError("invalid_evaluation_retries cannot be negative")
        if self.customer_protocol_retries < 0 or self.evolver_protocol_retries < 0:
            raise ValueError("protocol retries cannot be negative")
        if self.judge_validation_retries < 0:
            raise ValueError("judge_validation_retries cannot be negative")
        if self.max_tool_steps < 1:
            raise ValueError("max_tool_steps must be at least 1")
        if self.rate_limit_backoff_seconds < 0:
            raise ValueError("rate_limit_backoff_seconds cannot be negative")
        for field_name in ("max_api_requests_per_generation", "max_api_requests_per_run"):
            value = getattr(self, field_name)
            if value is not None and int(value) < 1:
                raise ValueError(f"{field_name} must be positive or null")

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
class CustomerSearchEvaluationConfig(EvaluationConfig):
    """Runtime controls for Customer-only runs without Service-Evolver budget."""

    token_budget: CustomerSearchTokenBudgetConfig = field(
        default_factory=CustomerSearchTokenBudgetConfig
    )

    @classmethod
    def from_dict(cls, value: Optional[Dict[str, Any]]) -> "CustomerSearchEvaluationConfig":
        value = dict(value or {})
        value["token_budget"] = CustomerSearchTokenBudgetConfig.from_value(
            value.get("token_budget")
        )
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
    customer: LegacyCustomerEvolutionConfig = field(default_factory=LegacyCustomerEvolutionConfig)
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
            "customer": LegacyCustomerEvolutionConfig,
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
                nested_value = dict(value[key])
                if key == "customer":
                    deprecated = {"fitness_weights", "allowed_strategy_tags", "adversary_access"}
                    removed = sorted(deprecated.intersection(nested_value))
                    if removed:
                        warnings.warn(
                            "deprecated Customer config fields are ignored by open-ended search: "
                            + ", ".join(removed),
                            DeprecationWarning,
                            stacklevel=2,
                        )
                        for field_name in removed:
                            nested_value.pop(field_name, None)
                value[key] = constructor(**nested_value)
        return cls(**{key: item for key, item in value.items() if key in cls.__dataclass_fields__})


@dataclass
class CustomerSearchConfig:
    """Small configuration surface for open-ended Customer black-box search."""

    scenario: str = "ecommerce_refund"
    seed: int = 7
    max_generations: int = 2
    customer: CustomerEvolutionConfig = field(default_factory=CustomerEvolutionConfig)
    splits: SplitConfig = field(default_factory=SplitConfig)
    evaluation: CustomerSearchEvaluationConfig = field(default_factory=CustomerSearchEvaluationConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)
    model_metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def experiment_mode(self) -> str:
        return "customer_only"

    def to_dict(self) -> Dict[str, Any]:
        return {"experiment_mode": self.experiment_mode, **asdict(self)}

    @classmethod
    def from_dict(cls, value: Optional[Dict[str, Any]]) -> "CustomerSearchConfig":
        value = dict(value or {})
        mode = value.get("experiment_mode", "customer_only")
        if mode != "customer_only":
            raise ValueError("CustomerSearchConfig only accepts experiment_mode='customer_only'")

        for legacy_block in ("service", "fresh_adversary"):
            if legacy_block in value:
                warnings.warn(
                    f"legacy {legacy_block} config is ignored by CustomerSearchConfig",
                    DeprecationWarning,
                    stacklevel=2,
                )

        customer_value = dict(value.get("customer") or {})
        supported_customer = set(CustomerEvolutionConfig.__dataclass_fields__)
        ignored_customer = sorted(set(customer_value) - supported_customer)
        if ignored_customer:
            warnings.warn(
                "deprecated Customer fields are ignored by CustomerSearchConfig: "
                + ", ".join(ignored_customer),
                DeprecationWarning,
                stacklevel=2,
            )
        customer = CustomerEvolutionConfig(**{
            key: item for key, item in customer_value.items()
            if key in supported_customer
        })

        splits_value = dict(value.get("splits") or {})
        splits = SplitConfig(**{
            key: item for key, item in splits_value.items()
            if key in SplitConfig.__dataclass_fields__
        })
        persistence_value = dict(value.get("persistence") or {})
        persistence = PersistenceConfig(**{
            key: item for key, item in persistence_value.items()
            if key in PersistenceConfig.__dataclass_fields__
        })
        evaluation = CustomerSearchEvaluationConfig.from_dict(value.get("evaluation"))
        return cls(
            scenario=str(value.get("scenario", "ecommerce_refund")),
            seed=int(value.get("seed", 7)),
            max_generations=int(value.get("max_generations", 2)),
            customer=customer,
            splits=splits,
            evaluation=evaluation,
            persistence=persistence,
            model_metadata=dict(value.get("model_metadata") or {}),
        )

    @classmethod
    def from_evolution_config(cls, config: EvolutionConfig) -> "CustomerSearchConfig":
        """Explicitly drop legacy Service/search fields at the CLI/API boundary."""
        evaluation = CustomerSearchEvaluationConfig.from_dict(asdict(config.evaluation))
        return cls(
            scenario=config.scenario,
            seed=config.seed,
            max_generations=config.max_generations,
            customer=CustomerEvolutionConfig(
                strategy_schema=config.customer.strategy_schema,
                candidate_count=config.customer.candidate_count,
            ),
            splits=SplitConfig(**asdict(config.splits)),
            evaluation=evaluation,
            persistence=PersistenceConfig(**asdict(config.persistence)),
            model_metadata=dict(config.model_metadata or {}),
        )


def _read_config_data(path: str | Path) -> dict[str, Any]:
    """Read JSON or YAML configuration data without choosing a runtime schema."""
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
    return data


def load_config(path: str | Path) -> EvolutionConfig:
    """Load a legacy/general Service or co-evolution configuration."""
    return EvolutionConfig.from_dict(_read_config_data(path))


def load_customer_search_config(path: str | Path) -> CustomerSearchConfig:
    """Load the intentionally small Customer-only black-box search config."""
    return CustomerSearchConfig.from_dict(_read_config_data(path))


def save_config(config: EvolutionConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
