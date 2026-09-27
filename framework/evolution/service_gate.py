"""Acceptance gate for structured service-policy patches."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import ServiceEvolutionConfig
from .evaluator_adapter import aggregate_episode_metrics
from .paired_stats import PairedComparison, compare_paired_episodes


@dataclass
class GateDecision:
    accepted: bool
    reason: str
    delta: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)


class ServiceGate:
    def __init__(self, min_delta: float | None = None, normal_regression_tolerance: float | None = None,
                 gate_min_paired_wins: int | None = None, max_normal_paired_losses: int | None = None):
        defaults = ServiceEvolutionConfig()
        self.min_delta = defaults.min_delta if min_delta is None else min_delta
        self.normal_regression_tolerance = defaults.normal_regression_tolerance if normal_regression_tolerance is None else normal_regression_tolerance
        self.gate_min_paired_wins = defaults.gate_min_paired_wins if gate_min_paired_wins is None else gate_min_paired_wins
        self.max_normal_paired_losses = defaults.max_normal_paired_losses if max_normal_paired_losses is None else max_normal_paired_losses

    def evaluate(self, baseline: dict[str, float], candidate: dict[str, float], normal_baseline=None,
                 normal_candidate=None, sanitizer_passed=True, *, paired_latest: PairedComparison | None = None,
                 paired_exact: PairedComparison | None = None,
                 paired_transfer: PairedComparison | None = None,
                 paired_normal: PairedComparison | None = None) -> GateDecision:
        if not sanitizer_passed:
            return GateDecision(False, "sanitizer_rejected")
        score_key = "robust_task_success" if "robust_task_success" in candidate else "task_success"
        delta = candidate.get(score_key, 0.0) - baseline.get(score_key, 0.0)
        pair_metrics = {
            key: value.to_dict()
            for key, value in {
                "latest_paired": paired_latest,
                "exact_replay_paired": paired_exact,
                "transfer_replay_paired": paired_transfer,
                "normal_paired": paired_normal,
            }.items()
            if value is not None
        }
        normal_delta = 0.0
        if normal_baseline is not None and normal_candidate is not None:
            normal_delta = normal_candidate.get("task_success", 0.0) - normal_baseline.get("task_success", 0.0)
            if paired_normal is not None:
                if paired_normal.losses > self.max_normal_paired_losses:
                    return GateDecision(False, "normal_user_paired_regression", delta,
                                        {"normal_delta": normal_delta, **pair_metrics})
            elif normal_delta < -self.normal_regression_tolerance:
                return GateDecision(False, "normal_user_regression", delta,
                                    {"normal_delta": normal_delta, **pair_metrics})
        if paired_latest is not None and (
            paired_latest.wins < self.gate_min_paired_wins
            or paired_latest.wins <= paired_latest.losses
        ):
            return GateDecision(False, "insufficient_paired_adversarial_improvement", delta,
                                {"normal_delta": normal_delta, **pair_metrics})
        if paired_exact is not None and paired_exact.losses:
            return GateDecision(False, "exact_replay_regression", delta,
                                {"exact_replay_paired_losses": paired_exact.losses, **pair_metrics})
        if paired_transfer is not None and paired_transfer.losses:
            return GateDecision(False, "transfer_replay_regression", delta,
                                {"transfer_replay_paired_losses": paired_transfer.losses, **pair_metrics})
        exact_before = baseline.get("exact_replay_task_success")
        exact_after = candidate.get("exact_replay_task_success")
        if exact_before is not None and exact_after is not None and exact_after < exact_before:
            return GateDecision(False, "exact_replay_regression", delta, {
                "exact_replay_delta": exact_after - exact_before,
                "exact_replay_regressions": candidate.get("exact_replay_regressions", []),
            })
        transfer_before = baseline.get("transfer_replay_task_success")
        transfer_after = candidate.get("transfer_replay_task_success")
        if transfer_before is not None and transfer_after is not None and transfer_after < transfer_before:
            return GateDecision(False, "transfer_replay_regression", delta, {
                "transfer_replay_delta": transfer_after - transfer_before,
            })
        if candidate.get("exact_replay_regressions"):
            return GateDecision(False, "exact_replay_regression", delta, {
                "exact_replay_regressions": candidate["exact_replay_regressions"],
            })
        if delta < self.min_delta:
            return GateDecision(False, "insufficient_adversarial_improvement", delta,
                                {"normal_delta": normal_delta, **pair_metrics})
        if candidate.get("tool_calls", 0) > max(20.0, baseline.get("tool_calls", 0) * 3 + 1):
            return GateDecision(False, "degenerate_tool_query_increase", delta, pair_metrics)
        if candidate.get("reject_rate", 0.0) >= 0.98 or candidate.get("transfer_rate", 0.0) >= 0.98:
            return GateDecision(False, "degenerate_reject_or_transfer", delta, pair_metrics)
        return GateDecision(True, "accepted", delta, {"normal_delta": normal_delta, **pair_metrics})

    def accept(self, baseline_episodes, candidate_episodes, normal_baseline=None, normal_candidate=None,
               sanitizer_passed=True, *, exact_baseline=None, exact_candidate=None,
               transfer_baseline=None, transfer_candidate=None):
        baseline_episodes = list(baseline_episodes)
        candidate_episodes = list(candidate_episodes)
        normal_baseline = list(normal_baseline) if normal_baseline is not None else None
        normal_candidate = list(normal_candidate) if normal_candidate is not None else None
        exact_baseline = list(exact_baseline or [])
        exact_candidate = list(exact_candidate or [])
        transfer_baseline = list(transfer_baseline or [])
        transfer_candidate = list(transfer_candidate or [])
        return self.evaluate(
            aggregate_episode_metrics(baseline_episodes),
            aggregate_episode_metrics(candidate_episodes),
            aggregate_episode_metrics(normal_baseline) if normal_baseline is not None else None,
            aggregate_episode_metrics(normal_candidate) if normal_candidate is not None else None,
            sanitizer_passed,
            paired_latest=compare_paired_episodes(baseline_episodes, candidate_episodes),
            paired_exact=compare_paired_episodes(exact_baseline, exact_candidate),
            paired_transfer=compare_paired_episodes(transfer_baseline, transfer_candidate),
            paired_normal=(
                compare_paired_episodes(normal_baseline, normal_candidate)
                if normal_baseline is not None and normal_candidate is not None else None
            ),
        )
