"""Adversarial customer--service co-evolution for EvoSAGE.

The package deliberately sits above the existing benchmark.  It owns policy
versions, candidate selection, dataset manifests and experiment artifacts; it
does not redefine CaseSpec, BackendEnvironment or evaluator semantics.
"""

from .schemas import (
    CustomerPolicy,
    DefenseRecord,
    EpisodeResult,
    FailureOccurrence,
    FailureSignature,
    ServicePatch,
    ServicePolicy,
    ServiceRule,
    VulnerabilitySignature,
    signature_from_dict,
)
from .config import EvolutionConfig, load_config
from .archives import AttackArchive, DefenseArchive
from .split_manager import DatasetSplits, SplitManager
from .runner import EvolutionRunner
from .customer_policy import CustomerPolicyCompiler, CustomerPolicyValidator, PolicyCustomerModel
from .service_policy import ServicePolicyCompiler, ServicePolicySanitizer, ServicePolicyValidator
from .service_gate import GateDecision, ServiceGate
from .evaluator_adapter import BudgetedEpisodeEvaluator, EvoSAGEEpisodeEvaluator, MockEpisodeEvaluator, aggregate_episode_metrics
from .weakness_frontier import WeaknessFrontier
from .attribution import FailureAttribution, infer_failure_attribution, infer_failure_location

__all__ = [
    "AttackArchive",
    "CustomerPolicy",
    "DatasetSplits",
    "DefenseArchive",
    "DefenseRecord",
    "EpisodeResult",
    "FailureAttribution",
    "FailureOccurrence",
    "EvolutionConfig",
    "EvolutionRunner",
    "CustomerPolicyCompiler",
    "CustomerPolicyValidator",
    "PolicyCustomerModel",
    "FailureSignature",
    "VulnerabilitySignature",
    "ServicePatch",
    "ServicePolicy",
    "ServiceRule",
    "SplitManager",
    "ServicePolicyCompiler",
    "ServicePolicySanitizer",
    "ServicePolicyValidator",
    "GateDecision",
    "ServiceGate",
    "EvoSAGEEpisodeEvaluator",
    "BudgetedEpisodeEvaluator",
    "MockEpisodeEvaluator",
    "aggregate_episode_metrics",
    "WeaknessFrontier",
    "infer_failure_location",
    "infer_failure_attribution",
    "signature_from_dict",
    "load_config",
]
