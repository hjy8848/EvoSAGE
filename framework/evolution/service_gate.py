"""Acceptance gate for structured service-policy patches."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import ServiceEvolutionConfig
from .evaluator_adapter import aggregate_episode_metrics


@dataclass
class GateDecision:
    accepted: bool
    reason: str
    delta: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)


class ServiceGate:
    def __init__(self, min_delta: float | None = None, normal_regression_tolerance: float | None = None):
        defaults = ServiceEvolutionConfig()
        self.min_delta = defaults.min_delta if min_delta is None else min_delta
        self.normal_regression_tolerance = defaults.normal_regression_tolerance if normal_regression_tolerance is None else normal_regression_tolerance

    def evaluate(self, baseline: dict[str, float], candidate: dict[str, float], normal_baseline=None, normal_candidate=None, sanitizer_passed=True) -> GateDecision:
        if not sanitizer_passed:
            return GateDecision(False, "sanitizer_rejected")
        score_key = "robust_task_success" if "robust_task_success" in candidate else "task_success"
        delta = candidate.get(score_key, 0.0) - baseline.get(score_key, 0.0)
        normal_delta = 0.0
        if normal_baseline is not None and normal_candidate is not None:
            normal_delta = normal_candidate.get("task_success", 0.0) - normal_baseline.get("task_success", 0.0)
            if normal_delta < -self.normal_regression_tolerance:
                return GateDecision(False, "normal_user_regression", delta, {"normal_delta": normal_delta})
        if delta < self.min_delta:
            return GateDecision(False, "insufficient_adversarial_improvement", delta, {"normal_delta": normal_delta})
        if candidate.get("tool_calls", 0) > max(20.0, baseline.get("tool_calls", 0) * 3 + 1):
            return GateDecision(False, "degenerate_tool_query_increase", delta)
        if candidate.get("reject_rate", 0.0) >= 0.98 or candidate.get("transfer_rate", 0.0) >= 0.98:
            return GateDecision(False, "degenerate_reject_or_transfer", delta)
        return GateDecision(True, "accepted", delta, {"normal_delta": normal_delta})

    def accept(self, baseline_episodes, candidate_episodes, normal_baseline=None, normal_candidate=None, sanitizer_passed=True):
        return self.evaluate(aggregate_episode_metrics(baseline_episodes), aggregate_episode_metrics(candidate_episodes),
                             aggregate_episode_metrics(normal_baseline) if normal_baseline is not None else None,
                             aggregate_episode_metrics(normal_candidate) if normal_candidate is not None else None,
                             sanitizer_passed)
