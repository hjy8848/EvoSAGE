"""Deterministic customer candidate scoring and selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from .config import CustomerEvolutionConfig
from .evaluator_adapter import aggregate_episode_metrics
from .schemas import CustomerPolicy, EpisodeResult


def _is_protocol_failure(episode: EpisodeResult) -> bool:
    return episode.is_evaluation_invalid() or (
        "json_parse_failed" in (episode.error_types or [])
        or "protocol_failure" in (episode.error_types or [])
        or bool(episode.metadata.get("protocol_failure", False))
    )


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
        valid_episodes = [item for item in episodes if not _is_protocol_failure(item)]
        invalid_episodes = [item for item in episodes if _is_protocol_failure(item)]
        invalid_reasons = sorted({
            str(item.invalid_reason)
            for item in invalid_episodes
            if item.invalid_reason
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
        # Only legitimate business-process failures are useful adversarial
        # signal.  Simulator/provider protocol-invalid episodes are excluded
        # from every fitness component, including SOP coverage.
        attack_success = aggregate_episode_metrics(valid_episodes)["legitimate_attack_success"]
        signatures = {
            signature.signature_id
            for episode in valid_episodes
            if not episode.task_success
            for signature in [FailureSignatureProxy.from_episode(episode)]
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


class FailureSignatureProxy:
    @staticmethod
    def from_episode(episode):
        from .schemas import FailureSignature
        return FailureSignature.from_episode(episode)
