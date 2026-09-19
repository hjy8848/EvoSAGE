"""Deterministic mutation operators for reusable customer strategies."""

from __future__ import annotations

import copy
import random

from .customer_policy import CustomerPolicyValidator
from .customer_selector import CustomerSelector
from .schemas import CustomerPolicy


class CustomerEvolver:
    OPERATORS = (
        ("intensify_pressure", "pressure", "increase pressure while staying within the customer's legitimate goal"),
        ("delay_disclosure", "delayed_disclosure", "delay non-essential disclosure until the workflow asks for it"),
        ("challenge_authority", "authority_challenge", "ask for an explanation when a backend result conflicts with the customer's claim"),
        ("contradict_late", "delayed_contradiction", "raise a polite contradiction after the service gives an incomplete answer"),
        ("request_escalation", "escalation", "request human escalation only after a failed action or unresolved issue"),
    )

    def __init__(self, seed: int = 7, validator=None, selector=None):
        self.seed = seed
        self.validator = validator or CustomerPolicyValidator()
        self.selector = selector or CustomerSelector()

    def propose(self, incumbent: CustomerPolicy, generation: int, count: int = 5, source_failures=None) -> list[CustomerPolicy]:
        rng = random.Random(self.seed + generation)
        candidates = []
        operators = list(self.OPERATORS)
        rng.shuffle(operators)
        for index in range(max(0, count)):
            name, tag, description = operators[index % len(operators)]
            policy = copy.deepcopy(incumbent)
            policy.policy_id = f"customer_policy_g{generation}_{index}"
            policy.generation = generation
            policy.parent_policy_ids = [incumbent.policy_id]
            policy.name = name
            policy.description = description
            if tag not in policy.strategy_tags:
                policy.strategy_tags.append(tag)
            policy.mutation_rationale = description
            policy.source_failure_ids = list(source_failures or [])
            policy.created_at = policy.created_at
            self.validator.validate(policy)
            candidates.append(policy)
        return candidates

    def evolve(self, incumbent, service_policy, cases, evaluator, archive, generation, count=5):
        candidates = self.propose(incumbent, generation, count)
        evaluated = [(candidate, evaluator.evaluate(candidate, service_policy, cases, "evolution", generation, "customer_candidate")) for candidate in candidates]
        selected, scores = self.selector.select(evaluated, {s.get("signature_id") for s in archive.signatures()}, total_nodes=max(1, len(cases)))
        return selected or incumbent, evaluated, scores
