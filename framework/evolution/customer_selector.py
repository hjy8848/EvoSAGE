"""Official-score-only selection for open-ended Customer search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .customer.policy import AdversaryPolicy
from .schemas import EpisodeResult


@dataclass
class CandidateScore:
    policy_id: str
    fitness: float | None
    official_task_success: float | None
    valid_episode_count: int
    invalid_episode_count: int
    evaluation_status: str
    invalid_reasons: list[str] | None = None

    @property
    def episodes(self) -> int:
        """Legacy runner compatibility; counts runtime-valid score evidence."""
        return self.valid_episode_count

    def to_dict(self) -> dict:
        value = dict(self.__dict__)
        value["invalid_reasons"] = list(self.invalid_reasons or [])
        return value

    @classmethod
    def from_dict(cls, value: dict) -> "CandidateScore":
        """Load both compact scores and pre-refactor persisted score rows."""
        valid_count = value.get("valid_episode_count", value.get("episodes", 0))
        fitness = value.get("fitness")
        official_success = value.get("official_task_success")
        if official_success is None and fitness is not None:
            official_success = 1.0 - float(fitness)
        return cls(
            policy_id=str(value.get("policy_id", "")),
            fitness=float(fitness) if fitness is not None else None,
            official_task_success=(
                float(official_success) if official_success is not None else None
            ),
            valid_episode_count=int(valid_count or 0),
            invalid_episode_count=int(value.get("invalid_episode_count", 0) or 0),
            evaluation_status=value.get(
                "evaluation_status", "valid" if valid_count else "inconclusive"
            ),
            invalid_reasons=list(value.get("invalid_reasons") or []),
        )


class CustomerSelector:
    """Score only runtime-evaluable official outcomes; retain incumbents on ties."""

    def score(
        self,
        policy: AdversaryPolicy,
        episodes: Iterable[EpisodeResult],
        *,
        allow_heldout: bool = False,
    ) -> CandidateScore:
        values = list(episodes)
        if not allow_heldout and any(item.split == "heldout_test" for item in values):
            raise AssertionError("CustomerSelector cannot score heldout episodes")
        valid = [
            item for item in values
            if item.is_runtime_evaluable() and isinstance(item.task_success, bool)
        ]
        invalid = [item for item in values if item not in valid]
        reasons = sorted({
            str(reason)
            for item in invalid
            for reason in ([item.invalid_reason] if item.invalid_reason else [])
            + list(item.validity_reasons or [])
            + list((item.metadata or {}).get("invalid_reasons", []) or [])
        })
        if any(
            item.is_runtime_evaluable() and not isinstance(item.task_success, bool)
            for item in invalid
        ):
            reasons = sorted(set(reasons) | {"missing_official_score"})
        if not valid:
            return CandidateScore(
                policy_id=policy.policy_id,
                fitness=None,
                official_task_success=None,
                valid_episode_count=0,
                invalid_episode_count=len(invalid),
                evaluation_status="inconclusive",
                invalid_reasons=reasons or (["no_episode_evidence"] if not values else []),
            )
        official_success = sum(bool(item.task_success) for item in valid) / len(valid)
        return CandidateScore(
            policy_id=policy.policy_id,
            fitness=1.0 - official_success,
            official_task_success=official_success,
            valid_episode_count=len(valid),
            invalid_episode_count=len(invalid),
            evaluation_status="valid",
            invalid_reasons=reasons,
        )

    def select(
        self,
        candidates: list[tuple[AdversaryPolicy, list[EpisodeResult]]],
        *,
        incumbent_policy_id: str | None = None,
        allow_heldout: bool = False,
    ):
        scores = [
            self.score(policy, episodes, allow_heldout=allow_heldout)
            for policy, episodes in candidates
        ]
        incumbent_index = next((
            index for index, (policy, _) in enumerate(candidates)
            if incumbent_policy_id is not None and policy.policy_id == incumbent_policy_id
        ), None)
        eligible = [index for index, score in enumerate(scores) if score.valid_episode_count]

        if incumbent_index is not None:
            selected_index = incumbent_index
            reason = "incumbent_unscored"
            incumbent_fitness = scores[incumbent_index].fitness
            if incumbent_fitness is not None:
                reason = "no_fitness_improvement"
                improved = [
                    index for index in eligible
                    if index != incumbent_index
                    and scores[index].fitness is not None
                    and scores[index].fitness > incumbent_fitness
                ]
                if improved:
                    selected_index = max(improved, key=lambda index: float(scores[index].fitness))
                    reason = "strict_official_fitness_improvement"
                elif not any(index != incumbent_index for index in eligible):
                    reason = "all_children_inconclusive"
            selected = candidates[selected_index][0]
            self.last_selection_record = {
                "incumbent_policy_id": candidates[incumbent_index][0].policy_id,
                "incumbent_score": scores[incumbent_index].to_dict(),
                "candidate_scores": [
                    scores[index].to_dict() for index in range(len(scores))
                    if index != incumbent_index
                ],
                "selected_policy_id": selected.policy_id,
                "customer_changed": selected_index != incumbent_index,
                "selection_reason": reason,
                "objective": "1 - mean(official task_success over runtime-evaluable episodes)",
            }
            return selected, scores

        if eligible:
            selected_index = max(eligible, key=lambda index: float(scores[index].fitness))
            selected = candidates[selected_index][0]
            reason = "highest_official_fitness"
        else:
            selected = None
            reason = "no_runtime_evaluable_candidate"
        self.last_selection_record = {
            "incumbent_policy_id": incumbent_policy_id,
            "incumbent_score": None,
            "candidate_scores": [item.to_dict() for item in scores],
            "selected_policy_id": selected.policy_id if selected else None,
            "customer_changed": selected is not None,
            "selection_reason": reason,
            "objective": "1 - mean(official task_success over runtime-evaluable episodes)",
        }
        return selected, scores
