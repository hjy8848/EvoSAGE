"""Configuration for open-ended Customer strategy search."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Optional

from ..backend.tool_contract import ToolContractConfig


@dataclass
class CustomerEvolutionConfig:
    """Number of free-text Customer strategies proposed per generation."""

    candidate_count: int = 5

    def __post_init__(self) -> None:
        if self.candidate_count < 1:
            raise ValueError("customer.candidate_count must be at least 1")


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
class TokenBudgetConfig:
    """Completion budgets for roles used by Customer search."""

    user: int = 512
    agent: int = 4096
    judge: int = 1024
    customer_evolver: int = 8192


@dataclass
class EvaluationConfig:
    repetitions: int = 1
    max_turns: int = 10
    concurrency: int = 1
    api_timeout: int = 300
    judge_validation_enabled: bool = False
    token_budget: TokenBudgetConfig = field(default_factory=TokenBudgetConfig)
    customer_thinking_mode: Optional[str] = None
    agent_thinking_mode: Optional[str] = None
    customer_evolver_thinking_mode: Optional[str] = None
    agent_max_retries: int = 1
    customer_transport_max_retries: int = 1
    customer_evolver_max_retries: int = 1
    judge_max_retries: int = 1
    judge_validation_retries: int = 2
    customer_protocol_retries: int = 1
    customer_evolver_protocol_retries: int = 1
    rate_limit_backoff_seconds: float = 0.0
    invalid_evaluation_retries: int = 1
    max_tool_steps: int = 8
    max_api_requests_per_generation: Optional[int] = None
    max_api_requests_per_run: Optional[int] = None
    tool_contract: ToolContractConfig = field(default_factory=ToolContractConfig)

    def __post_init__(self) -> None:
        for field_name in (
            "customer_thinking_mode", "agent_thinking_mode",
            "customer_evolver_thinking_mode",
        ):
            if getattr(self, field_name) not in {None, "enabled", "disabled"}:
                raise ValueError(f"{field_name} must be None, 'enabled', or 'disabled'")
        for field_name in (
            "agent_max_retries", "customer_transport_max_retries",
            "customer_evolver_max_retries", "judge_max_retries",
        ):
            if int(getattr(self, field_name)) < 1:
                raise ValueError(f"{field_name} is a total-attempt count and must be at least 1")
        if self.invalid_evaluation_retries < 0:
            raise ValueError("invalid_evaluation_retries cannot be negative")
        if self.customer_protocol_retries < 0 or self.customer_evolver_protocol_retries < 0:
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
    def from_dict(cls, value: Optional[dict[str, Any]]) -> "EvaluationConfig":
        value = dict(value or {})
        _reject_unknown(value, cls.__dataclass_fields__, "evaluation")
        if isinstance(value.get("token_budget"), dict):
            value["token_budget"] = _construct(
                TokenBudgetConfig, value["token_budget"], "evaluation.token_budget",
            )
        if value.get("tool_contract") is not None:
            value["tool_contract"] = ToolContractConfig.from_value(value["tool_contract"])
        return cls(**value)


@dataclass
class PersistenceConfig:
    output_dir: str = "results/customer_search"
    resume: bool = False


@dataclass
class CustomerSearchConfig:
    """Complete active experiment configuration; no legacy mode conversion."""

    scenario: str = "ecommerce_refund"
    seed: int = 7
    max_generations: int = 2
    customer: CustomerEvolutionConfig = field(default_factory=CustomerEvolutionConfig)
    splits: SplitConfig = field(default_factory=SplitConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)
    model_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Optional[dict[str, Any]]) -> "CustomerSearchConfig":
        value = dict(value or {})
        _reject_unknown(value, cls.__dataclass_fields__, "config")
        return cls(
            scenario=str(value.get("scenario", "ecommerce_refund")),
            seed=int(value.get("seed", 7)),
            max_generations=int(value.get("max_generations", 2)),
            customer=_construct(CustomerEvolutionConfig, value.get("customer"), "customer"),
            splits=_construct(SplitConfig, value.get("splits"), "splits"),
            evaluation=EvaluationConfig.from_dict(value.get("evaluation")),
            persistence=_construct(PersistenceConfig, value.get("persistence"), "persistence"),
            model_metadata=dict(value.get("model_metadata") or {}),
        )


def _reject_unknown(value: dict[str, Any], fields, section: str) -> None:
    unknown = sorted(set(value) - set(fields))
    if unknown:
        raise ValueError(f"unknown {section} config field(s): {', '.join(unknown)}")


def _construct(constructor, value, section: str):
    value = dict(value or {})
    _reject_unknown(value, constructor.__dataclass_fields__, section)
    return constructor(**value)


def load_customer_search_config(path: str | Path) -> CustomerSearchConfig:
    """Load one strict Customer-search JSON/YAML config."""
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ValueError("Config is not JSON and PyYAML is not installed") from exc
        value = yaml.safe_load(raw)
    if not isinstance(value, dict):
        raise ValueError("Customer-search config must be a mapping")
    return CustomerSearchConfig.from_dict(value)
