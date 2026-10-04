"""Public API for EvoSAGE's open-ended Customer search core.

Combined Service/co-evolution code stays available through its existing
submodules and is imported lazily only when a legacy API is explicitly used.
"""

from .config import (
    CustomerEvolutionConfig,
    CustomerSearchConfig,
    EvolutionConfig,
    load_config,
    load_customer_search_config,
)
from .customer.policy import AdversaryPolicy
from .customer.runner import CustomerEvolutionRunner
from .customer_evolver import CustomerEvolver, LLMCustomerPolicyGenerator
from .customer_selector import CandidateScore, CustomerSelector
from .evaluator_adapter import BudgetedEpisodeEvaluator, EvoSAGEEpisodeEvaluator, MockEpisodeEvaluator
from .schemas import EpisodeResult
from .split_manager import DatasetSplits, SplitManager

__all__ = [
    "AdversaryPolicy",
    "CandidateScore",
    "CustomerEvolutionConfig",
    "CustomerEvolver",
    "CustomerEvolutionRunner",
    "CustomerSearchConfig",
    "CustomerSelector",
    "DatasetSplits",
    "EpisodeResult",
    "EvolutionConfig",
    "LLMCustomerPolicyGenerator",
    "SplitManager",
    "BudgetedEpisodeEvaluator",
    "EvoSAGEEpisodeEvaluator",
    "MockEpisodeEvaluator",
    "load_config",
    "load_customer_search_config",
]


_LEGACY_EXPORTS = {
    "AttackArchive": (".archives", "AttackArchive"),
    "AttackInstance": (".schemas", "AttackInstance"),
    "CustomerPolicy": (".schemas", "CustomerPolicy"),
    "DefenseArchive": (".archives", "DefenseArchive"),
    "DefenseRecord": (".schemas", "DefenseRecord"),
    "EvolutionRunner": (".runner", "EvolutionRunner"),
    "FailureAttribution": (".attribution", "FailureAttribution"),
    "FailureOccurrence": (".schemas", "FailureOccurrence"),
    "FailureSignature": (".schemas", "FailureSignature"),
    "ServicePatch": (".schemas", "ServicePatch"),
    "ServicePolicy": (".schemas", "ServicePolicy"),
    "ServiceRule": (".schemas", "ServiceRule"),
    "VulnerabilitySignature": (".schemas", "VulnerabilitySignature"),
    "VulnerabilityArchiveEntry": (".schemas", "VulnerabilityArchiveEntry"),
    "ServicePolicyCompiler": (".service_policy", "ServicePolicyCompiler"),
    "ServicePolicySanitizer": (".service_policy", "ServicePolicySanitizer"),
    "ServicePolicyValidator": (".service_policy", "ServicePolicyValidator"),
    "GateDecision": (".service_gate", "GateDecision"),
    "ServiceGate": (".service_gate", "ServiceGate"),
    "WeaknessFrontier": (".weakness_frontier", "WeaknessFrontier"),
    "infer_failure_location": (".attribution", "infer_failure_location"),
    "infer_failure_attribution": (".attribution", "infer_failure_attribution"),
    "signature_from_dict": (".schemas", "signature_from_dict"),
    "aggregate_episode_metrics": (".evaluator_adapter", "aggregate_episode_metrics"),
}


def __getattr__(name):
    target = _LEGACY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    from importlib import import_module
    module = import_module(target[0], __name__)
    value = getattr(module, target[1])
    globals()[name] = value
    return value
