"""Deterministic mutation operators for reusable customer strategies."""

from __future__ import annotations

import copy
import json
import random
import re

from .customer_policy import CustomerPolicyValidator
from .customer_selector import CustomerSelector
from .schemas import CustomerPolicy


def _top_k(items, limit: int = 5):
    """Keep prompts bounded while retaining the most recent observations."""
    values = list(items or [])
    return values[-max(1, limit):]


class CustomerEvolver:
    OPERATORS = (
        ("intensify_pressure", "pressure", "increase pressure while staying within the customer's legitimate goal"),
        ("delay_disclosure", "delayed_disclosure", "delay non-essential disclosure until the workflow asks for it"),
        ("challenge_authority", "authority_challenge", "ask for an explanation when a backend result conflicts with the customer's claim"),
        ("contradict_late", "delayed_contradiction", "raise a polite contradiction after the service gives an incomplete answer"),
        ("request_escalation", "escalation", "request human escalation only after a failed action or unresolved issue"),
    )

    def __init__(self, seed: int = 7, validator=None, selector=None, strategy_generator=None,
                 require_strategy_generator: bool = False):
        self.seed = seed
        self.validator = validator or CustomerPolicyValidator()
        self.selector = selector or CustomerSelector()
        self.strategy_generator = strategy_generator
        self.require_strategy_generator = require_strategy_generator
        self.last_rejections = []

    def propose(self, incumbent: CustomerPolicy, generation: int, count: int = 5, source_failures=None, service_policy=None, frontier=None, archive_summary=None) -> list[CustomerPolicy]:
        failures = list(source_failures or [])
        self.last_rejections = []
        if self.strategy_generator is not None:
            try:
                generated = self.strategy_generator.generate(
                    incumbent=incumbent, failures=failures, service_policy=service_policy,
                    frontier=frontier, archive_summary=archive_summary,
                    generation=generation, count=count,
                )
            except Exception as exc:
                if self.require_strategy_generator:
                    raise RuntimeError("LLM customer policy generation failed in strict real mode") from exc
                generated = []
            valid = []
            for policy in generated:
                try:
                    self.validator.validate(policy)
                    valid.append(policy)
                except Exception as exc:
                    self.last_rejections.append({"policy_id": policy.policy_id, "reason": str(exc)})
                    continue
            if valid:
                return valid[:count]
            if self.require_strategy_generator:
                raise RuntimeError("LLM customer policy generator returned no valid candidates")
        elif self.require_strategy_generator:
            raise RuntimeError("strict real mode requires an LLM customer policy generator")
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
            policy.source_failure_ids = [getattr(item, "signature_id", str(item)) for item in failures]
            policy.created_at = policy.created_at
            try:
                self.validator.validate(policy)
                candidates.append(policy)
            except Exception as exc:
                self.last_rejections.append({"policy_id": policy.policy_id, "reason": str(exc)})
        return candidates

    def evolve(self, incumbent, service_policy, cases, evaluator, archive, generation, count=5,
               cases_per_candidate=None, elite_count=0, source_failures=None, frontier=None, archive_summary=None):
        candidate_cases = list(cases)
        if any(getattr(case, "split", "") == "heldout_test" for case in candidate_cases):
            raise AssertionError("CustomerEvolver cannot consume heldout cases")
        if cases_per_candidate is not None and cases_per_candidate > 0:
            candidate_cases = candidate_cases[:cases_per_candidate]
        candidates = self.propose(incumbent, generation, count, source_failures, service_policy, frontier, archive_summary)
        evaluated = [(candidate, evaluator.evaluate(candidate, service_policy, candidate_cases, "evolution", generation, "customer_candidate")) for candidate in candidates]
        if elite_count:
            evaluated.insert(0, (incumbent, evaluator.evaluate(incumbent, service_policy, candidate_cases, "evolution", generation, "customer_elite")))
        selected, scores = self.selector.select(evaluated, {s.get("signature_id") for s in archive.signatures()}, total_nodes=max(1, len(cases)))
        return selected or incumbent, evaluated, scores


class LLMCustomerPolicyGenerator:
    """Generate reusable policy JSON from abstract failure signatures.

    The prompt contains no case IDs, user IDs, expected paths or hidden
    backend values.  Invalid or non-JSON generations are discarded by the
    caller and the deterministic fallback remains available.
    """

    def __init__(self, llm_client, adversary_access: str = "black_box", max_tokens: int = 4096):
        self.llm_client = llm_client
        self.max_tokens = max_tokens
        if adversary_access not in {"black_box", "white_box"}:
            raise ValueError("adversary_access must be black_box or white_box")
        self.adversary_access = adversary_access

    def generate(self, incumbent, failures, service_policy, generation, count, frontier=None, archive_summary=None):
        failure_view = [
            {"errors": list(item.error_types), "sop_node": item.sop_node,
             "predicted_action": item.predicted_action, "executed_action": item.executed_action}
            for item in _top_k(failures)
        ]
        frontier = _top_k(frontier)
        archive_summary = _top_k(archive_summary)
        prompt = (
            "Design reusable customer interaction strategies for a customer-service benchmark. "
            "Do not mention case IDs, order IDs, expected paths/actions, hidden values, evaluators, "
            "or parser manipulation. Return a JSON array only. Each item must contain name, "
            "description, strategy_tags, disclosure_strategy, pressure_strategy, "
            "contradiction_strategy, response_to_verification, response_to_rejection.\n"
            f"Current strategy tags: {json.dumps(incumbent.strategy_tags)}\n"
            f"Observed abstract failures: {json.dumps(failure_view, ensure_ascii=False)}\n"
            f"Service rule summary: {json.dumps([r.text for r in service_policy.rules if r.active], ensure_ascii=False) if self.adversary_access == 'white_box' else 'WITHHELD_IN_BLACK_BOX_MODE'}\n"
            f"Weakness frontier summary: {json.dumps(frontier or [], ensure_ascii=False)[:6000]}\n"
            f"Historical attack summary: {json.dumps(archive_summary or [], ensure_ascii=False)[:6000]}\n"
            f"Generate up to {count} distinct candidates."
        )
        response = self.llm_client.generate(
            prompt=prompt,
            temperature=0.7,
            max_tokens=self.max_tokens,
        )
        text = re.sub(r"^```(?:json)?|```$", "", response.text.strip(), flags=re.I | re.M).strip()
        value = json.loads(text)
        if isinstance(value, dict):
            value = value.get("candidates", [value])
        policies = []
        for index, item in enumerate(value if isinstance(value, list) else []):
            if not isinstance(item, dict):
                continue
            data = incumbent.to_dict()
            data.update({key: item[key] for key in item if key in data and key not in {"policy_id", "generation", "parent_policy_ids"}})
            data.update({
                "policy_id": f"customer_policy_g{generation}_llm_{index}",
                "generation": generation,
                "parent_policy_ids": [incumbent.policy_id],
                "mutation_rationale": "LLM-generated from abstract failure signatures",
                "source_failure_ids": [item.signature_id for item in failures],
            })
            policies.append(CustomerPolicy.from_dict(data))
        return policies
