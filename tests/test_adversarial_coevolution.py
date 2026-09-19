import json

import pytest

from framework.backend.factory import build_case_spec
from framework.evolution.archives import AttackArchive
from framework.evolution.config import EvolutionConfig, PersistenceConfig
from framework.evolution.customer_policy import CustomerPolicyValidator
from framework.evolution.evaluator_adapter import MockEpisodeEvaluator
from framework.evolution.runner import EvolutionRunner
from framework.evolution.schemas import CustomerPolicy, PolicyValidationError, ServicePatch, ServiceRule
from framework.evolution.service_gate import ServiceGate
from framework.evolution.service_policy import ServicePolicySanitizer
from framework.evolution.split_manager import SplitManager


def ecommerce_case():
    from framework.sop.ecommerce_refund_PathList import generate_path_list
    return build_case_spec("ecommerce_refund", "refund_before_shipping", generate_path_list()[5], "test-user")


def test_customer_policy_roundtrip_and_leakage_gate():
    case = ecommerce_case()
    policy = CustomerPolicy(strategy_tags=["truthful"], description="delay disclosure until asked")
    CustomerPolicyValidator().validate(policy, case)
    assert CustomerPolicy.from_dict(policy.to_dict()).to_dict() == policy.to_dict()
    bad = CustomerPolicy(description=f"use {case.case_id} and expected_action")
    with pytest.raises(PolicyValidationError):
        CustomerPolicyValidator().validate(bad, case)


def test_service_sanitizer_rejects_case_specific_rules():
    patch = ServicePatch("p1", "add", [ServiceRule("r1", "VERIFICATION", "For CASE-123 use expected_path")])
    with pytest.raises(PolicyValidationError):
        ServicePolicySanitizer().sanitize(patch)


def test_split_reproducibility_and_heldout_isolation(tmp_path):
    manager = SplitManager(manifest_dir=tmp_path / "manifest")
    one = manager.build()
    two = SplitManager(manifest_dir=tmp_path / "manifest2").build()
    assert [item.case_id for item in one.all_cases] == [item.case_id for item in two.all_cases]
    one.assert_no_heldout(one.evolution + one.validation)
    with pytest.raises(AssertionError):
        one.assert_no_heldout(one.heldout_test)


def test_gate_accepts_improvement_and_rejects_regression():
    gate = ServiceGate(min_delta=.01, normal_regression_tolerance=.03)
    accepted = gate.evaluate({"task_success": .2}, {"task_success": .3})
    rejected = gate.evaluate({"task_success": .2}, {"task_success": .3}, {"task_success": 1.0}, {"task_success": .9})
    assert accepted.accepted
    assert not rejected.accepted


def test_two_generation_mock_run_resume_and_matrix(tmp_path):
    config = EvolutionConfig(max_generations=2, persistence=PersistenceConfig(output_dir=str(tmp_path / "run")))
    runner = EvolutionRunner(config, evaluator=MockEpisodeEvaluator())
    result = runner.run()
    assert result["completed_generations"] == [0, 1]
    assert (tmp_path / "run" / "archives" / "attacks.jsonl").exists()
    assert (tmp_path / "run" / "analysis" / "weakness_frontier.json").exists()
    matrix = runner.cross_generation_evaluation()
    assert len(matrix) == 4
    config.persistence.resume = True
    assert EvolutionRunner(config, evaluator=MockEpisodeEvaluator()).run()["completed_generations"] == [0, 1]
