"""Deterministic customer candidate scoring and selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from .config import CustomerEvolutionConfig
from .evaluator_adapter import aggregate_episode_metrics
from .schemas import CustomerPolicy, EpisodeResult


def _is_fitness_eligible(episode: EpisodeResult) -> bool:
    """Only protocol-, environment-, and Customer-valid evidence may score."""
    return episode.is_substantively_evaluable()


@dataclass
class CandidateScore:
    policy_id: str
    attack_success: Optional[float]
    novelty: Optional[float]
    coverage: Optional[float]
    fitness: Optional[float]
    episodes: int
    evaluation_status: str = "valid"
    invalid_episode_count: int = 0
    invalid_reasons: Optional[List[str]] = None

    def to_dict(self):
        value = self.__dict__.copy()
        value["invalid_reasons"] = list(self.invalid_reasons or [])
        return value


class CustomerSelector:
    def __init__(self, weights=None):
        self.weights = dict(CustomerEvolutionConfig().fitness_weights)
        if weights:
            self.weights.update(weights)

    def score(self, policy: CustomerPolicy, episodes: Iterable[EpisodeResult], known_signatures: set[str], total_nodes: int = 1, allow_heldout: bool = False) -> CandidateScore:
        episodes = list(episodes)
        if not allow_heldout and any(item.split == "heldout_test" for item in episodes):
            raise AssertionError("CustomerSelector cannot score heldout episodes")
        valid_episodes = [item for item in episodes if _is_fitness_eligible(item)]
        invalid_episodes = [item for item in episodes if not _is_fitness_eligible(item)]
        invalid_reasons = sorted({
            str(reason)
            for item in invalid_episodes
            for reason in ([item.invalid_reason] if item.invalid_reason else []) + list(item.validity_reasons or [])
        })
        if not valid_episodes:
            return CandidateScore(
                policy_id=policy.policy_id,
                attack_success=None,
                novelty=None,
                coverage=None,
                fitness=None,
                episodes=0,
                evaluation_status="inconclusive",
                invalid_episode_count=len(invalid_episodes),
                invalid_reasons=invalid_reasons or (["no_valid_episode_evidence"] if not episodes else []),
            )
        # Customer behavior that violates the fixed CaseSpec is diagnostic,
        # not attack fitness. Only attributable service failures contribute.
        attack_success = aggregate_episode_metrics(valid_episodes)["attributable_service_failure_rate"]
        signatures = {
            signature.signature_id
            for episode in valid_episodes
            if episode.is_attributable_service_failure()
            for signature in [episode.vulnerability_signature_v2()]
            if signature is not None
        }
        novelty = len(signatures - known_signatures) / max(1, len(signatures))
        coverage = len({e.sop_node for e in valid_episodes if e.sop_node}) / max(1, total_nodes)
        fitness = sum(self.weights[key] * value for key, value in (("attack_success", attack_success), ("novelty", novelty), ("coverage", coverage)))
        return CandidateScore(
            policy.policy_id, attack_success, novelty, coverage, fitness,
            len(valid_episodes), "valid", len(invalid_episodes), invalid_reasons,
        )

    def select(self, candidates: list[tuple[CustomerPolicy, list[EpisodeResult]]], known_signatures: set[str], total_nodes: int = 1, allow_heldout: bool = False):
        scores = [self.score(policy, episodes, known_signatures, total_nodes, allow_heldout=allow_heldout) for policy, episodes in candidates]
        eligible = [index for index, score in enumerate(scores) if score.episodes > 0]
        order = sorted(eligible, key=lambda i: (-scores[i].fitness, -scores[i].attack_success, -scores[i].novelty, candidates[i][0].policy_id))
        index = order[0] if order else None
        return (candidates[index][0] if index is not None else None), scores
