"""Deterministic customer candidate scoring and selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional
import warnings

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
    node_diversity: Optional[float]
    fitness: Optional[float]
    episodes: int
    evaluation_status: str = "valid"
    invalid_episode_count: int = 0
    invalid_reasons: Optional[List[str]] = None

    @property
    def coverage(self) -> Optional[float]:
        """Deprecated API alias; this is case-normalized failure-node diversity."""
        return self.node_diversity

    def to_dict(self):
        value = self.__dict__.copy()
        value["invalid_reasons"] = list(self.invalid_reasons or [])
        return value


class CustomerSelector:
    def __init__(self, weights=None, tie_tolerance: float = 1e-12):
        self.weights = dict(CustomerEvolutionConfig().fitness_weights)
        self.tie_tolerance = max(0.0, float(tie_tolerance))
        self.last_selection_record = None
        if weights:
            weights = dict(weights)
            if "coverage" in weights:
                weights.setdefault("node_diversity", weights["coverage"])
                weights.pop("coverage", None)
                warnings.warn(
                    "customer fitness weight 'coverage' is deprecated; interpreted as 'node_diversity'",
                    DeprecationWarning,
                    stacklevel=2,
                )
            self.weights.update(weights)

    def score(self, policy: CustomerPolicy, episodes: Iterable[EpisodeResult], known_signatures: set[str], total_nodes: Optional[int] = None, allow_heldout: bool = False) -> CandidateScore:
        episodes = list(episodes)
        if total_nodes is not None:
            warnings.warn(
                "CustomerSelector.total_nodes is deprecated and ignored; node_diversity uses evaluated case count",
                DeprecationWarning,
                stacklevel=2,
            )
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
                node_diversity=None,
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
        candidate_case_count = len({episode.case_id for episode in valid_episodes})
        failure_nodes = {
            episode.sop_node
            for episode in valid_episodes
            if episode.is_attributable_service_failure() and episode.sop_node
        }
        node_diversity = min(1.0, len(failure_nodes) / max(1, candidate_case_count))
        fitness = sum(self.weights[key] * value for key, value in (
            ("attack_success", attack_success),
            ("novelty", novelty),
            ("node_diversity", node_diversity),
        ))
        return CandidateScore(
            policy.policy_id, attack_success, novelty, node_diversity, fitness,
            len(valid_episodes), "valid", len(invalid_episodes), invalid_reasons,
        )

    def select(
        self,
        candidates: list[tuple[CustomerPolicy, list[EpisodeResult]]],
        known_signatures: set[str],
        total_nodes: Optional[int] = None,
        allow_heldout: bool = False,
        incumbent_policy_id: Optional[str] = None,
    ):
        scores = [self.score(policy, episodes, known_signatures, total_nodes, allow_heldout=allow_heldout) for policy, episodes in candidates]
        eligible = [index for index, score in enumerate(scores) if score.episodes > 0]
        incumbent_index = next((
            index for index, (policy, _) in enumerate(candidates)
            if incumbent_policy_id is not None and policy.policy_id == incumbent_policy_id
        ), None)
        order = sorted(
            eligible,
            key=lambda i: (
                -float(scores[i].fitness),
                -float(scores[i].attack_success),
                -float(scores[i].novelty),
                candidates[i][0].policy_id,
            ),
        )

        selected_index = order[0] if order else None
        reason = "best_valid_candidate" if selected_index is not None else "no_valid_candidate"
        if incumbent_index is not None:
            incumbent = candidates[incumbent_index][0]
            incumbent_score = scores[incumbent_index]
            child_order = [
                index for index in order
                if index != incumbent_index
                and candidates[index][0].semantic_fingerprint() != incumbent.semantic_fingerprint()
            ]
            best_child_index = next((
                index for index in child_order
                if incumbent_score.fitness is not None
                and scores[index].fitness > incumbent_score.fitness + self.tie_tolerance
            ), None)
            selected_index = incumbent_index
            if not incumbent_score.episodes:
                reason = "incumbent_unscored"
            elif best_child_index is None:
                if not child_order and any(index != incumbent_index for index in eligible):
                    reason = "no_behavioral_change"
                elif not child_order and not any(index != incumbent_index for index in eligible):
                    reason = "all_candidates_invalid" if len(candidates) > 1 else "no_valid_candidate"
                else:
                    reason = "no_fitness_improvement"
            else:
                selected_index = best_child_index
                reason = "strict_fitness_improvement"
            selected = candidates[selected_index][0]
            self.last_selection_record = {
                "incumbent_policy_id": incumbent.policy_id,
                "incumbent_score": incumbent_score.to_dict(),
                "candidate_scores": [
                    score.to_dict() for index, score in enumerate(scores)
                    if index != incumbent_index
                ],
                "selected_policy_id": selected.policy_id,
                "customer_changed": (
                    selected.semantic_fingerprint() != incumbent.semantic_fingerprint()
                ),
                "selection_reason": reason,
                "tie_tolerance": self.tie_tolerance,
            }
            return selected, scores

        self.last_selection_record = {
            "incumbent_policy_id": incumbent_policy_id,
            "incumbent_score": None,
            "candidate_scores": [score.to_dict() for score in scores],
            "selected_policy_id": candidates[selected_index][0].policy_id if selected_index is not None else None,
            "customer_changed": selected_index is not None,
            "selection_reason": reason,
            "tie_tolerance": self.tie_tolerance,
        }
        return (candidates[selected_index][0] if selected_index is not None else None), scores
