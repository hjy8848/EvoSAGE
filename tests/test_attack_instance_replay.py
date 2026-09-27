from __future__ import annotations

from framework.evolution.service_evolver import _exact_replay_regressions
from framework.evolution.archives import AttackArchive
from framework.evolution.schemas import (
    AttackInstance,
    CustomerPolicy,
    EpisodeResult,
    FailureOccurrence,
    ServicePatch,
    ServicePolicy,
    ServiceRule,
    VulnerabilitySignature,
)
from framework.evolution.service_evolver import ServiceEvolver
from framework.evolution.split_manager import DatasetCase, dataset_case_from_attack_instance


def _case(case_id, split="evolution", hidden_marker="original-secret"):
    case_spec = {
        "case_id": case_id,
        "scenario": "ecommerce_refund",
        "backend_record": {"private_state": {"marker": hidden_marker}},
        "user_goal": {"type": "refund", "desired_action": "Refund"},
        "user_knowledge": {"order_id": "ORD-ORIGINAL"},
        "user_policy": {"show_order_id_initially": False},
        "initial_observation": {},
        "expected_outcome": {"order.refund_status": "Approved"},
        "metadata": {"user_intent": "refund_before_shipping", "path_config": {"path_id": 3}},
    }
    return DatasetCase(
        case_id=case_id,
        split=split,
        path_id=3,
        instance_index=2,
        intent="refund_before_shipping",
        path_config={"path_id": 3},
        case_spec=case_spec,
    )


def _failed_episode(case, policy, episode_id, error="wrong_tool_arguments", repetition=0):
    return EpisodeResult(
        episode_id=episode_id,
        scenario="ecommerce_refund",
        case_id=case.case_id,
        customer_policy_id=policy.policy_id,
        service_policy_id="service-old",
        split=case.split,
        generation=0,
        task_success=False,
        execution_score=0.0,
        error_types=[error],
        sop_node="step3",
        path_step_index=2,
        metadata={"repetition": repetition},
    )


def _instance(case, policy, episode):
    signature = VulnerabilitySignature.from_episode(episode)
    occurrence = FailureOccurrence.from_episode(episode, signature)
    return AttackInstance.from_episode(
        episode,
        signature,
        occurrence,
        policy,
        case.case_spec,
        reproduction_seed=17,
        dataset_case=case.to_dict(),
    )


def test_exact_replay_uses_original_case_spec_and_original_customer_policy():
    original_case = _case("case-original")
    original_customer = CustomerPolicy(policy_id="customer-original", strategy_tags=["withholding"])
    episode = _failed_episode(original_case, original_customer, "episode-original")
    instance = _instance(original_case, original_customer, episode)

    rehydrated = dataset_case_from_attack_instance(instance)

    assert rehydrated.case_id == "case-original"
    assert rehydrated.case_spec == original_case.case_spec
    assert rehydrated.case_spec["backend_record"]["private_state"]["marker"] == "original-secret"
    assert rehydrated.path_config == original_case.path_config
    assert CustomerPolicy.from_dict(instance.customer_policy).policy_id == "customer-original"


def test_archive_persists_occurrence_case_and_vulnerability_indexes(tmp_path):
    policy = CustomerPolicy(policy_id="customer-archive")
    first_case = _case("case-a")
    second_case = _case("case-b")
    first = _failed_episode(first_case, policy, "episode-a", "wrong_tool_arguments")
    second = _failed_episode(second_case, policy, "episode-b", "wrong_final_action")
    first_sig = VulnerabilitySignature.from_episode(first)
    second_sig = VulnerabilitySignature.from_episode(second)
    archive = AttackArchive(tmp_path / "attacks.jsonl")

    added = archive.add(
        policy,
        [first_sig, second_sig],
        [first, second],
        generation=0,
        case_specs_by_id={first_case.case_id: first_case, second_case.case_id: second_case},
        occurrences=[
            FailureOccurrence.from_episode(first, first_sig),
            FailureOccurrence.from_episode(second, second_sig),
        ],
        reproduction_seed=17,
    )

    assert added == 2
    assert len(archive.exact_instances()) == 2
    assert (tmp_path / "attack_instances.jsonl").exists()
    assert (tmp_path / "vulnerabilities.jsonl").exists()
    assert {item.case_id for item in archive.exact_instances()} == {"case-a", "case-b"}
    assert all(item.case_spec["case_id"] == item.case_id for item in archive.exact_instances())
    assert {item.signature["signature_id"] for item in archive.vulnerabilities.values()} == {
        first_sig.signature_id, second_sig.signature_id,
    }
    reloaded = AttackArchive(tmp_path / "attacks.jsonl")
    assert {item.attack_instance_id for item in reloaded.exact_instances()} == {
        item.attack_instance_id for item in archive.exact_instances()
    }
    assert {item.signature_id for item in reloaded.signatures()} == {
        first_sig.signature_id, second_sig.signature_id,
    }


def test_archive_case_ids_are_signature_specific_and_incidence_is_not_policy_failure_rate(tmp_path):
    policy = CustomerPolicy(policy_id="customer-mixed")
    cases = [_case("case-a"), _case("case-b"), _case("case-c")]
    failures = [
        _failed_episode(cases[0], policy, "episode-a", "wrong_tool_arguments"),
        _failed_episode(cases[1], policy, "episode-b", "wrong_tool_arguments"),
        _failed_episode(cases[2], policy, "episode-c", "wrong_final_action"),
    ]
    signatures = [VulnerabilitySignature.from_episode(item) for item in failures]
    archive = AttackArchive(tmp_path / "attacks.jsonl")
    archive.add(
        policy,
        signatures,
        failures,
        case_specs_by_id={case.case_id: case for case in cases},
        reproduction_seed=7,
    )

    rows = archive.to_dicts()
    row_by_sig = {row["vulnerability_signature"]["signature_id"]: row for row in rows}
    sig_a, sig_b = signatures[0].signature_id, signatures[2].signature_id
    assert set(row_by_sig[sig_a]["source_case_ids"]) == {"case-a", "case-b"}
    assert set(row_by_sig[sig_b]["source_case_ids"]) == {"case-c"}
    assert row_by_sig[sig_a]["policy_attack_success_rate"] == 1.0
    assert row_by_sig[sig_a]["signature_incidence_rate"] == 2 / 3
    assert row_by_sig[sig_b]["signature_incidence_rate"] == 1 / 3


class RecordingEvaluator:
    def __init__(self):
        self.calls = []

    def evaluate(self, customer, service, cases, split, generation, phase):
        cases = list(cases)
        self.calls.append({
            "customer": customer.policy_id,
            "service": service.policy_id,
            "case_ids": [case.case_id for case in cases],
            "case_specs": [case.case_spec for case in cases],
            "split": split,
            "phase": phase,
        })
        defended = any(rule.active for rule in service.rules)
        return [EpisodeResult(
            episode_id=f"{phase}-{index}-{service.policy_id}",
            scenario="ecommerce_refund",
            case_id=case.case_id,
            customer_policy_id=customer.policy_id,
            service_policy_id=service.policy_id,
            split=split,
            generation=generation,
            task_success=defended,
            execution_score=float(defended),
            verification_score=float(defended),
            policy_score=float(defended),
            action_execution_score=float(defended),
            goal_fulfillment_score=float(defended),
            error_types=[] if defended else ["goal_not_fulfilled"],
        ) for index, case in enumerate(cases)]


def _patch_generator(policy, failures, generation, count, *_args):
    return [ServicePatch(
        patch_id="candidate-patch",
        patch_type="add",
        rules=[ServiceRule("candidate-rule", "RECOVERY", "Explain the failed operation and retry only after valid input")],
    )]


def test_service_evolver_separates_exact_and_transfer_replay_suites():
    archived_case = _case("case-original")
    archived_customer = CustomerPolicy(policy_id="customer-original")
    archived_episode = _failed_episode(archived_case, archived_customer, "episode-original")
    instance = _instance(archived_case, archived_customer, archived_episode)
    validation_case = _case("case-validation", split="validation", hidden_marker="validation-secret")
    current_customer = CustomerPolicy(policy_id="customer-current")
    evaluator = RecordingEvaluator()
    evolver = ServiceEvolver(patch_generator=type("Generator", (), {"generate": staticmethod(_patch_generator)})())

    _, decision, _ = evolver.evolve(
        ServicePolicy(),
        [FailureOccurrence.from_episode(
            archived_episode, VulnerabilitySignature.from_episode(archived_episode)
        )],
        [validation_case],
        [validation_case],
        evaluator,
        generation=1,
        count=1,
        customer_policy=current_customer,
        exact_replay_instances=[instance],
        transfer_replay_policies=[archived_customer],
    )

    assert decision.accepted
    exact_calls = [call for call in evaluator.calls if call["phase"].endswith("_exact_replay")]
    transfer_calls = [call for call in evaluator.calls if call["phase"].endswith("_transfer_replay")]
    latest_calls = [call for call in evaluator.calls if call["phase"].endswith("_latest")]
    assert exact_calls and all(call["case_ids"] == ["case-original"] for call in exact_calls)
    assert all(call["case_specs"][0]["backend_record"]["private_state"]["marker"] == "original-secret" for call in exact_calls)
    assert all(call["customer"] == "customer-original" and call["split"] == "evolution" for call in exact_calls)
    assert transfer_calls and all(call["case_ids"] == ["case-validation"] for call in transfer_calls)
    assert all(call["customer"] == "customer-original" and call["split"] == "validation" for call in transfer_calls)
    assert latest_calls and all(call["customer"] == "customer-current" for call in latest_calls)
    metrics = evolver.last_selected_metrics
    assert metrics["latest_task_success"] == 1.0
    assert metrics["exact_replay_task_success"] == 1.0
    assert metrics["transfer_replay_task_success"] == 1.0
    assert "replay_task_success" in metrics  # legacy alias is not the only replay metric


def test_exact_replay_regression_detection_pairs_same_repetition_only():
    case = _case("paired-case")
    policy = CustomerPolicy(policy_id="paired-customer")
    source = _failed_episode(case, policy, "source-episode", repetition=1)
    instance = _instance(case, policy, source)

    baseline_rep0 = _failed_episode(case, policy, "baseline-r0", repetition=0)
    baseline_rep0.task_success = True
    baseline_rep1 = _failed_episode(case, policy, "baseline-r1", repetition=1)
    candidate_rep0 = _failed_episode(case, policy, "candidate-r0", repetition=0)
    candidate_rep1 = _failed_episode(case, policy, "candidate-r1", repetition=1)
    candidate_rep1.task_success = True

    # A success at rep0 must not be paired with a failure at rep1.
    assert _exact_replay_regressions(
        [instance], [baseline_rep0, baseline_rep1], [candidate_rep0, candidate_rep1]
    ) == []
    # A regression at the archived instance's repetition is retained.
    assert _exact_replay_regressions(
        [instance], [baseline_rep1], [candidate_rep1]
    ) == []
    baseline_rep1.task_success = True
    candidate_rep1.task_success = False
    assert _exact_replay_regressions(
        [instance], [baseline_rep1], [candidate_rep1]
    ) == [f"{instance.attack_instance_id}:paired-case#rep1"]


def test_heldout_attack_instance_is_rejected_before_archive_or_replay(tmp_path):
    policy = CustomerPolicy(policy_id="heldout-policy")
    heldout_case = _case("heldout", split="heldout_test")
    episode = _failed_episode(heldout_case, policy, "heldout-episode")
    signature = VulnerabilitySignature.from_episode(episode)
    occurrence = FailureOccurrence.from_episode(episode, signature)

    try:
        AttackInstance.from_episode(
            episode, signature, occurrence, policy, heldout_case.case_spec, 7,
            dataset_case=heldout_case.to_dict(),
        )
    except AssertionError:
        pass
    else:
        raise AssertionError("heldout attack instance must not be created")

    archive = AttackArchive(tmp_path / "attacks.jsonl")
    try:
        archive.add(
            policy, [signature], [episode],
            case_specs_by_id={heldout_case.case_id: heldout_case},
        )
    except AssertionError:
        pass
    else:
        raise AssertionError("heldout episode must not enter the archive")
