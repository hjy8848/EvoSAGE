"""Deterministic mutation operators for reusable customer strategies."""

from __future__ import annotations

import copy
import json
import random
import re

from .customer_policy import CustomerPolicyValidator
from .customer_selector import CustomerSelector
from .generation_protocol import GenerationProtocolError, request_json_with_retry
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
        self.last_candidate_records = []
        self.last_generation_record = None

    def propose(self, incumbent: CustomerPolicy, generation: int, count: int = 5, source_failures=None,
                service_policy=None, frontier=None, archive_summary=None, validation_cases=None) -> list[CustomerPolicy]:
        failures = list(source_failures or [])
        self.last_rejections = []
        self.last_candidate_records = []
        if self.strategy_generator is not None:
            try:
                generated = self.strategy_generator.generate(
                    incumbent=incumbent, failures=failures, service_policy=service_policy,
                    frontier=frontier, archive_summary=archive_summary,
                    generation=generation, count=count,
                )
                self.last_generation_record = copy.deepcopy(
                    getattr(self.strategy_generator, "last_generation_record", None)
                )
            except GenerationProtocolError:
                self.last_generation_record = copy.deepcopy(
                    getattr(self.strategy_generator, "last_generation_record", None)
                )
                if self.require_strategy_generator:
                    raise
                generated = []
            except Exception as exc:
                if self.require_strategy_generator:
                    raise RuntimeError("LLM customer policy generation failed in strict real mode") from exc
                generated = []
            valid = []
            candidate_records = list(
                (self.last_generation_record or {}).get("candidates", [])
            )
            # Preserve schema-invalid candidates from the generator even
            # though they do not produce a CustomerPolicy object.  They are
            # protocol/candidate evidence, not silent drops.
            self.last_candidate_records = list(candidate_records)
            records_by_policy = {
                item.get("policy_id"): item
                for item in candidate_records
                if item.get("policy_id")
            }
            for policy in generated:
                candidate_record = records_by_policy.setdefault(
                    policy.policy_id,
                    {
                        "candidate_index": len(candidate_records) + 1,
                        "raw_candidate": None,
                        "policy_id": policy.policy_id,
                        "schema_construction": {"status": "PASS"},
                    },
                )
                checks = []
                try:
                    self.validator.validate(policy)
                    checks.append({"check": "generic", "status": "PASS"})
                except Exception as exc:
                    checks.append({
                        "check": "generic",
                        "status": "FAIL",
                        "exact_reason": str(exc),
                    })
                for case in validation_cases or []:
                    case_spec = getattr(case, "case_spec", None)
                    if isinstance(case_spec, dict):
                        from ..backend.types import CaseSpec
                        case_spec = CaseSpec(**copy.deepcopy(case_spec))
                    try:
                        self.validator.validate(policy, case_spec)
                        checks.append({"check": "same_case", "status": "PASS"})
                    except Exception as exc:
                        checks.append({
                            "check": "same_case",
                            "status": "FAIL",
                            "exact_reason": str(exc),
                        })
                candidate_record["constructed_policy"] = policy.to_dict()
                candidate_record["validator"] = checks
                candidate_record["accepted"] = all(
                    check["status"] == "PASS" for check in checks
                )
                if candidate_record not in self.last_candidate_records:
                    self.last_candidate_records.append(candidate_record)
                if candidate_record["accepted"]:
                    valid.append(policy)
                else:
                    failed_check = next(
                        (check for check in checks if check.get("status") == "FAIL"),
                        None,
                    )
                    self.last_rejections.append({
                        "policy_id": policy.policy_id,
                        "reason": (failed_check or {}).get("exact_reason", "candidate validation failed"),
                    })
            if valid:
                if self.last_generation_record is not None:
                    self.last_generation_record["status"] = "valid"
                    self.last_generation_record["accepted_candidate_count"] = len(valid)
                    self.last_generation_record["candidates"] = self.last_candidate_records
                return valid[:count]
            if self.require_strategy_generator:
                if self.last_generation_record is not None:
                    self.last_generation_record["status"] = "candidate_rejected"
                    self.last_generation_record["reason"] = "customer_candidate_rejected:all_candidates"
                    self.last_generation_record["candidates"] = self.last_candidate_records
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
        validation_cases = list(cases)
        candidate_cases = list(validation_cases)
        if any(getattr(case, "split", "") == "heldout_test" for case in validation_cases):
            raise AssertionError("CustomerEvolver cannot consume heldout cases")
        if cases_per_candidate is not None and cases_per_candidate > 0:
            candidate_cases = candidate_cases[:cases_per_candidate]
        candidates = self.propose(
            incumbent, generation, count, source_failures, service_policy,
            frontier, archive_summary, validation_cases=validation_cases,
        )
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

    def __init__(self, llm_client, adversary_access: str = "black_box", max_tokens: int = 4096,
                 summary_limit: int = 5, allowed_strategy_tags=None):
        self.llm_client = llm_client
        self.max_tokens = max_tokens
        self.summary_limit = max(1, int(summary_limit))
        self.allowed_strategy_tags = sorted(
            allowed_strategy_tags or CustomerPolicyValidator.DEFAULT_TAGS
        )
        if adversary_access not in {"black_box", "white_box"}:
            raise ValueError("adversary_access must be black_box or white_box")
        self.adversary_access = adversary_access
        self.last_generation_record = None

    def generate(self, incumbent, failures, service_policy, generation, count, frontier=None, archive_summary=None):
        failure_view = [
            {"errors": list(item.error_types), "sop_node": item.sop_node,
             "predicted_action": item.predicted_action, "executed_action": item.executed_action}
            for item in _top_k(failures, self.summary_limit)
        ]
        frontier = _top_k(frontier, self.summary_limit)
        archive_summary = _top_k(archive_summary, self.summary_limit)
        prompt = (
            "You are designing reusable behavior policies for the simulated CUSTOMER only, "
            "not for the service agent. Every strategy must describe what the customer says "
            "or does during the interaction. Never prescribe service-agent behavior such as "
            "classifying, querying tools, checking records, following SOP steps, approving, "
            "or rejecting requests. The response_to_verification and response_to_rejection "
            "fields must describe the customer's reaction or utterance after those events, "
            "not instructions for what the agent should do. "
            "Design reusable customer interaction strategies for a customer-service benchmark. "
            "Do not mention case IDs, order IDs, expected paths/actions, hidden values, evaluators, "
            "or parser manipulation. Do not mention any concrete backend field names, status values, "
            "identifiers, or case-specific facts. "
            "Use only generic interaction language such as authoritative result, required identifier, "
            "or failed action. strategy_tags must be a subset of this exact list: "
            f"{json.dumps(self.allowed_strategy_tags)}. Do not invent tags such as transparent. "
            "Return a JSON array only. Each item must contain name, "
            "description, strategy_tags, disclosure_strategy, pressure_strategy, "
            "contradiction_strategy, response_to_verification, response_to_rejection.\n"
            f"Current strategy tags: {json.dumps(incumbent.strategy_tags)}\n"
            f"Observed abstract failures: {json.dumps(failure_view, ensure_ascii=False)}\n"
            f"Service rule summary: {json.dumps([r.text for r in service_policy.rules if r.active], ensure_ascii=False) if self.adversary_access == 'white_box' else 'WITHHELD_IN_BLACK_BOX_MODE'}\n"
            f"Weakness frontier summary: {json.dumps(frontier or [], ensure_ascii=False)[:6000]}\n"
            f"Historical attack summary: {json.dumps(archive_summary or [], ensure_ascii=False)[:6000]}\n"
            f"Generate up to {count} distinct candidates."
        )
        try:
            value, generation_record = request_json_with_retry(
                client=self.llm_client,
                prompt=prompt,
                role="customer",
                max_tokens=self.max_tokens,
                temperature=0.7,
            )
        except GenerationProtocolError as exc:
            # The helper raises after the final attempt, so retain the full
            # attempt provenance before propagating the inconclusive result.
            self.last_generation_record = copy.deepcopy(exc.record)
            raise
        self.last_generation_record = generation_record
        if isinstance(value, dict):
            value = value.get("candidates", [value])
        policies = []
        candidate_records = []
        required_fields = {
            "name", "description", "strategy_tags", "disclosure_strategy",
            "pressure_strategy", "contradiction_strategy",
            "response_to_verification", "response_to_rejection",
        }
        for index, item in enumerate(value if isinstance(value, list) else []):
            candidate_record = {
                "candidate_index": index + 1,
                "raw_candidate": item,
                "policy_id": None,
                "schema_construction": {"status": "PASS"},
                "constructed_policy": None,
                "validator": [],
                "accepted": False,
            }
            candidate_records.append(candidate_record)
            if not isinstance(item, dict):
                candidate_record["schema_construction"] = {
                    "status": "FAIL",
                    "reason": "candidate must be a JSON object",
                }
                continue
            missing_fields = sorted(required_fields - set(item))
            wrong_types = {}
            if "strategy_tags" in item and not isinstance(item["strategy_tags"], list):
                wrong_types["strategy_tags"] = type(item["strategy_tags"]).__name__
            for field in required_fields - {"strategy_tags"}:
                if field in item and not isinstance(item[field], str):
                    wrong_types[field] = type(item[field]).__name__
                elif field in item and not item[field].strip():
                    wrong_types[field] = "empty_string"
            if missing_fields or wrong_types:
                candidate_record["schema_construction"] = {
                    "status": "FAIL",
                    "missing_fields": missing_fields,
                    "wrong_types": wrong_types,
                }
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
            policy = CustomerPolicy.from_dict(data)
            candidate_record["policy_id"] = policy.policy_id
            candidate_record["constructed_policy"] = policy.to_dict()
            policies.append(policy)
        generation_record["candidates"] = candidate_records
        generation_record["response_candidate_count"] = len(candidate_records)
        return policies
