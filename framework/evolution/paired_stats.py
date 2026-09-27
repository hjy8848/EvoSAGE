"""Deterministic paired comparisons for Service candidate evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from .schemas import EpisodeResult


class PairingSetMismatch(ValueError):
    """Raised when baseline and candidate do not cover the same valid pairs."""


@dataclass(frozen=True)
class PairedComparison:
    wins: int
    losses: int
    ties: int
    matched_pairs: int
    delta_mean: float
    baseline_success: float
    candidate_success: float
    execution_delta_mean: float

    @property
    def net_wins(self) -> int:
        return self.wins - self.losses

    def to_dict(self) -> dict:
        return {**asdict(self), "net_wins": self.net_wins}


def episode_pair_key(episode: EpisodeResult) -> str:
    metadata = episode.metadata or {}
    repetition = int(metadata.get("repetition", 0) or 0)
    canonical = f"{episode.case_id}|customer={episode.customer_policy_id}|rep={repetition}"
    explicit = metadata.get("pair_key")
    if explicit is not None and str(explicit) != canonical:
        raise PairingSetMismatch(
            f"pair_key mismatch for episode {episode.episode_id}: "
            f"expected {canonical!r}, got {str(explicit)!r}"
        )
    return canonical


def compare_paired_episodes(
    baseline: Iterable[EpisodeResult],
    candidate: Iterable[EpisodeResult],
    *,
    require_matching: bool = True,
) -> PairedComparison:
    """Compare task outcomes and execution scores over identical valid samples.

    Invalid/protocol/customer/environment episodes are excluded from the
    substantive comparison. Duplicate identities or asymmetric valid coverage
    are integrity errors, never silently reduced to an intersection.
    """

    def index(rows: Iterable[EpisodeResult], label: str) -> dict[str, EpisodeResult]:
        result: dict[str, EpisodeResult] = {}
        for episode in rows:
            if not episode.is_substantively_evaluable():
                continue
            key = episode_pair_key(episode)
            if key in result:
                raise PairingSetMismatch(f"duplicate {label} pair key: {key}")
            result[key] = episode
        return result

    left = index(baseline, "baseline")
    right = index(candidate, "candidate")
    left_keys, right_keys = set(left), set(right)
    if require_matching and left_keys != right_keys:
        missing = sorted(left_keys - right_keys)
        extra = sorted(right_keys - left_keys)
        raise PairingSetMismatch(
            f"paired episode sets do not match (candidate missing={missing}, extra={extra})"
        )
    common = sorted(left_keys & right_keys)
    if not common:
        return PairedComparison(0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0)

    wins = losses = ties = 0
    execution_deltas: list[float] = []
    baseline_successes: list[float] = []
    candidate_successes: list[float] = []
    for key in common:
        before, after = left[key], right[key]
        before_success, after_success = bool(before.task_success), bool(after.task_success)
        baseline_successes.append(float(before_success))
        candidate_successes.append(float(after_success))
        if not before_success and after_success:
            wins += 1
        elif before_success and not after_success:
            losses += 1
        else:
            ties += 1
        execution_deltas.append(float(after.execution_score) - float(before.execution_score))

    count = len(common)
    baseline_mean = sum(baseline_successes) / count
    candidate_mean = sum(candidate_successes) / count
    return PairedComparison(
        wins=wins,
        losses=losses,
        ties=ties,
        matched_pairs=count,
        delta_mean=candidate_mean - baseline_mean,
        baseline_success=baseline_mean,
        candidate_success=candidate_mean,
        execution_delta_mean=sum(execution_deltas) / count,
    )
