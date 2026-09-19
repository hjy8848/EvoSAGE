"""Deterministic customer candidate scoring and selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .config import CustomerEvolutionConfig
from .evaluator_adapter import aggregate_episode_metrics
from .schemas import CustomerPolicy, EpisodeResult


def _is_protocol_failure(episode: EpisodeResult) -> bool:
    return (
        "json_parse_failed" in (episode.error_types or [])
        or "protocol_failure" in (episode.error_types or [])
        or bool(episode.metadata.get("protocol_failure", False))
    )


@dataclass
class CandidateScore:
    policy_id: str
    attack_success: float
    novelty: float
    coverage: float
    fitness: float
    episodes: int

    def to_dict(self):
        return self.__dict__.copy()


class CustomerSelector:
    def __init__(self, weights=None):
        self.weights = dict(CustomerEvolutionConfig().fitness_weights)
        if weights:
            self.weights.update(weights)

    def score(self, policy: CustomerPolicy, episodes: Iterable[EpisodeResult], known_signatures: set[str], total_nodes: int = 1, allow_heldout: bool = False) -> CandidateScore:
        episodes = list(episodes)
        if not allow_heldout and any(item.split == "heldout_test" for item in episodes):
            raise AssertionError("CustomerSelector cannot score heldout episodes")
        # Only legitimate business-process failures are useful adversarial
        # signal.  Parser/protocol breakage must not improve Customer fitness.
        attack_success = aggregate_episode_metrics(episodes)["legitimate_attack_success"]
        signatures = {
            signature.signature_id
            for episode in episodes
            if not episode.task_success and not _is_protocol_failure(episode)
            for signature in [FailureSignatureProxy.from_episode(episode)]
        }
        novelty = len(signatures - known_signatures) / max(1, len(signatures))
        coverage = len({e.sop_node for e in episodes if e.sop_node}) / max(1, total_nodes)
        fitness = sum(self.weights[key] * value for key, value in (("attack_success", attack_success), ("novelty", novelty), ("coverage", coverage)))
        return CandidateScore(policy.policy_id, attack_success, novelty, coverage, fitness, len(episodes))

    def select(self, candidates: list[tuple[CustomerPolicy, list[EpisodeResult]]], known_signatures: set[str], total_nodes: int = 1, allow_heldout: bool = False):
        scores = [self.score(policy, episodes, known_signatures, total_nodes, allow_heldout=allow_heldout) for policy, episodes in candidates]
        order = sorted(range(len(candidates)), key=lambda i: (-scores[i].fitness, -scores[i].attack_success, -scores[i].novelty, candidates[i][0].policy_id))
        index = order[0] if order else None
        return (candidates[index][0] if index is not None else None), scores


class FailureSignatureProxy:
    @staticmethod
    def from_episode(episode):
        from .schemas import FailureSignature
        return FailureSignature.from_episode(episode)
