from __future__ import annotations

import json
from types import SimpleNamespace

from framework.evolution.archives import AttackArchive
from framework.evolution.attribution import FailureAttribution, infer_failure_attribution
from framework.evolution.schemas import (
    CustomerPolicy,
    EpisodeResult,
    FailureSignature,
    VulnerabilitySignature,
    signature_from_dict,
)
from framework.evolution.customer_selector import CustomerSelector
from framework.evolution.weakness_frontier import WeaknessFrontier


def _episode(**overrides):
    values = {
        "episode_id": "episode-a",
        "scenario": "ecommerce_refund",
        "case_id": "case-a",
        "customer_policy_id": "customer-a",
        "service_policy_id": "service-a",
        "split": "evolution",
        "generation": 0,
        "task_success": False,
        "execution_score": 0.25,
        "error_types": ["wrong_tool_arguments"],
        "predicted_action": "Supplementary",
        "executed_action": "",
        "tool_sequence_summary": ["query_order"],
        "termination_reason": "tool_failure",
        "sop_node": "step3",
        "path_step_index": 2,
        "trace_ref": "real_traces/episode-a.json",
    }
    values.update(overrides)
    return EpisodeResult(**values)


def _attribution(**overrides):
    values = {
        "sop_node": "step3",
        "path_step_index": 2,
        "failure_stage": "VERIFICATION",
        "violated_invariant": "TOOL_ARGUMENT_PRECONDITION_VIOLATED",
        "primary_error": "wrong_tool_arguments",
        "attribution_reason": "required tool argument was invalid",
        "confidence": "deterministic",
        "trigger_class": "INVALID_TOOL_ARGUMENT",
        "service_decision_class": "INVALID_QUERY_ARGUMENT",
    }
    values.update(overrides)
    return FailureAttribution(**values)


def test_occurrence_details_do_not_change_vulnerability_identity():
    attribution = _attribution()
    first = VulnerabilitySignature.from_episode(_episode(), attribution)
    second = VulnerabilitySignature.from_episode(
        _episode(
            episode_id="episode-b",
            case_id="case-b",
            tool_sequence_summary=["query_order", "query_payment"],
            termination_reason="max_turns",
            predicted_action="Refund",
            executed_action="Refund",
            execution_score=0.5,
        ),
        attribution,
    )

    assert first.signature_id == second.signature_id


def test_different_sop_nodes_produce_different_vulnerabilities():
    first = VulnerabilitySignature.from_episode(_episode(), _attribution(sop_node="step3"))
    second = VulnerabilitySignature.from_episode(_episode(), _attribution(sop_node="step5"))

    assert first.signature_id != second.signature_id


def test_different_violated_invariants_produce_different_vulnerabilities():
    first = VulnerabilitySignature.from_episode(
        _episode(), _attribution(violated_invariant="TOOL_ARGUMENT_PRECONDITION_VIOLATED")
    )
    second = VulnerabilitySignature.from_episode(
        _episode(), _attribution(violated_invariant="REQUIRED_BACKEND_FACT_NOT_VERIFIED")
    )

    assert first.signature_id != second.signature_id


def test_confirmation_and_action_failures_are_distinct_vulnerabilities():
    action = VulnerabilitySignature.from_episode(
        _episode(error_types=["action_execution_failure"]),
        _attribution(
            failure_stage="ACTION",
            violated_invariant="FAILED_ACTION_NOT_RECOVERED",
            primary_error="action_execution_failure",
            trigger_class="ACTION_FAILURE",
            service_decision_class="ACTION_ATTEMPT_FAILED",
        ),
    )
    confirmation = VulnerabilitySignature.from_episode(
        _episode(error_types=["claimed_action_not_executed"]),
        _attribution(
            failure_stage="CONFIRMATION",
            violated_invariant="ACTION_CLAIM_WITHOUT_SUCCESSFUL_EXECUTION",
            primary_error="claimed_action_not_executed",
            trigger_class="FALSE_COMPLETION_CLAIM",
            service_decision_class="CLAIM_WITHOUT_ACTION",
        ),
    )

    assert action.signature_id != confirmation.signature_id


def test_v1_signature_loads_as_legacy_and_is_not_admitted_to_v2_archive(tmp_path):
    legacy_payload = {
        "signature_id": "failure_legacy",
        "scenario": "ecommerce_refund",
        "error_types": ["wrong_tool_arguments"],
        "tool_sequence_summary": ["query_order"],
        "termination_reason": "tool_failure",
    }
    legacy = signature_from_dict(legacy_payload)

    assert isinstance(legacy, FailureSignature)
    assert legacy.schema_version == 1
    assert legacy.to_dict()["schema_version"] == 1

    episode = _episode()
    v2 = VulnerabilitySignature.from_episode(episode, _attribution())
    archive = AttackArchive(tmp_path / "attacks.jsonl")
    assert archive.add(CustomerPolicy(), [legacy, v2], [episode]) == 1
    assert [item.signature_id for item in archive.signatures()] == [v2.signature_id]


def test_loading_legacy_archive_keeps_v1_records_out_of_v2_signature_set(tmp_path):
    path = tmp_path / "mixed-archive.jsonl"
    path.write_text(json.dumps({
        "attack_id": "legacy-attack",
        "failure_signature": {"signature_id": "failure_legacy"},
    }) + "\n", encoding="utf-8")
    archive = AttackArchive(path)

    assert len(archive) == 1  # readable legacy record remains preserved
    assert archive.signatures() == []


def test_episode_serializes_signature_and_occurrence_as_separate_v2_objects():
    episode = _episode()
    value = episode.to_dict()

    signature = value["vulnerability_signature"]
    occurrence = value["failure_occurrence"]
    assert signature["schema_version"] == 2
    assert occurrence["schema_version"] == 2
    assert value["failure_signature"] == signature  # migration alias
    assert occurrence["signature_id"] == signature["signature_id"]
    assert occurrence["episode_id"] == episode.episode_id
    assert occurrence["tool_sequence_summary"] == episode.tool_sequence_summary
    assert occurrence["termination_reason"] == episode.termination_reason
    assert occurrence["occurrence_id"] != signature["signature_id"]


def test_legacy_episode_roundtrip_remains_v1_and_is_not_promoted():
    original = _episode()
    data = original.to_dict()
    legacy_signature = FailureSignature.from_episode(original).to_dict()
    data["vulnerability_signature"] = None
    data["failure_occurrence"] = None
    data["failure_signature"] = legacy_signature

    loaded = EpisodeResult.from_dict(data)
    restored = loaded.to_dict()

    assert loaded.vulnerability_signature_v2() is None
    assert restored["vulnerability_signature"] is None
    assert restored["failure_occurrence"] is None
    assert restored["failure_signature"] == legacy_signature
    assert restored["failure_signature"]["schema_version"] == 1


def test_unclassified_historical_episode_does_not_gain_v2_signature():
    data = _episode().to_dict()
    data.pop("vulnerability_signature")
    data.pop("failure_occurrence")
    data.pop("failure_signature")

    loaded = EpisodeResult.from_dict(data)

    assert loaded.vulnerability_signature_v2() is None
    assert loaded.to_dict()["vulnerability_signature"] is None


def test_v2_episode_roundtrip_preserves_signature_and_occurrence():
    data = _episode().to_dict()
    loaded = EpisodeResult.from_dict(data)
    restored = loaded.to_dict()

    assert restored["vulnerability_signature"] == data["vulnerability_signature"]
    assert restored["failure_occurrence"] == data["failure_occurrence"]
    assert restored["failure_signature"] == data["failure_signature"]


def test_v1_episode_is_excluded_from_v2_novelty_and_archive(tmp_path):
    original = _episode()
    data = original.to_dict()
    legacy_signature = FailureSignature.from_episode(original).to_dict()
    data["vulnerability_signature"] = None
    data["failure_occurrence"] = None
    data["failure_signature"] = legacy_signature
    legacy_episode = EpisodeResult.from_dict(data)
    v2_signature = VulnerabilitySignature.from_episode(_episode(), _attribution())

    score = CustomerSelector().score(CustomerPolicy(), [legacy_episode], set())
    archive = AttackArchive(tmp_path / "legacy-attacks.jsonl")

    assert score.attack_success == 1.0
    assert score.novelty == 0.0
    assert archive.add(CustomerPolicy(), [v2_signature], [legacy_episode]) == 0
    assert archive.signatures() == []


def test_weakness_frontier_uses_stage_and_invariant_and_skips_legacy_failures():
    v1_data = _episode().to_dict()
    v1_data["vulnerability_signature"] = None
    v1_data["failure_occurrence"] = None
    v1_data["failure_signature"] = FailureSignature.from_episode(_episode()).to_dict()
    legacy = EpisodeResult.from_dict(v1_data)
    first = _episode(episode_id="v2-verification")
    first.metadata["failure_attribution"] = _attribution().to_dict()
    second = _episode(episode_id="v2-action")
    second.metadata["failure_attribution"] = _attribution(
        failure_stage="ACTION",
        violated_invariant="EXPECTED_OUTCOME_NOT_REACHED",
        primary_error="action_execution_failure",
        trigger_class="ACTION_FAILURE",
        service_decision_class="ACTION_ATTEMPT_FAILED",
    ).to_dict()

    frontier = WeaknessFrontier()
    frontier.add([legacy, first, second])

    rows = frontier.to_dicts()
    assert len(rows) == 2
    assert {(row["failure_stage"], row["violated_invariant"]) for row in rows} == {
        ("VERIFICATION", "TOOL_ARGUMENT_PRECONDITION_VIOLATED"),
        ("ACTION", "EXPECTED_OUTCOME_NOT_REACHED"),
    }


def test_failure_attribution_reports_path_and_lifecycle_stage_deterministically():
    base = {
        "details": {},
        "gold_path": ["step1", "step2", "step3"],
        "predicted_path": ["step1", "different", "step3"],
        "error_categories": ["wrong_final_action"],
        "task_success": False,
        "required_verification_score": 1.0,
        "policy_compliance_score": 1.0,
        "action_execution_score": 0.0,
        "goal_fulfillment": 0.0,
    }
    decision = infer_failure_attribution(SimpleNamespace(**base))
    assert (decision.failure_stage, decision.sop_node, decision.path_step_index) == (
        "DECISION", "step2", 1
    )

    base["predicted_path"] = ["step1", "step2", "step3"]
    base["error_categories"] = ["claimed_action_not_executed"]
    confirmation = infer_failure_attribution(SimpleNamespace(**base))
    assert confirmation.failure_stage == "CONFIRMATION"
    assert confirmation.violated_invariant == "ACTION_CLAIM_WITHOUT_SUCCESSFUL_EXECUTION"
