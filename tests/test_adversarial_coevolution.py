import json
from types import SimpleNamespace

import pytest

from framework.backend.factory import build_case_spec
from framework.evolution.archives import AttackArchive
from framework.evolution.config import EvolutionConfig, PersistenceConfig
from framework.evolution.customer_policy import CustomerPolicyValidator
from framework.evolution.evaluator_adapter import MockEpisodeEvaluator
from framework.evolution.evaluator_adapter import EvoSAGEEpisodeEvaluator
from framework.evolution.customer_evolver import LLMCustomerPolicyGenerator
from framework.evolution.service_evolver import LLMServicePatchGenerator
from framework.evolution.runner import EvolutionRunner
from framework.evolution.schemas import CustomerPolicy, PolicyValidationError, ServicePatch, ServicePolicy, ServiceRule
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


def test_real_adapter_binds_policies_at_pipeline_construction():
    seen = {}

    class Pipeline:
        def run_single_simulation(self, intent, user_id=None, path_config=None):
            simulation = SimpleNamespace(
                simulation_id="sim1", scenario_id="ecommerce_refund", case_spec={"case_id": "c1"},
                backend_events=[], termination_reason="goal_satisfied", turns=[], model_name="fake",
            )
            report = SimpleNamespace(
                task_success=True, execution_score=1.0, sage_style_score=1.0,
                required_verification_score=1.0, policy_compliance_score=1.0,
                action_execution_score=1.0, goal_fulfillment=1.0,
                error_categories=[], predicted_action="Refund", executed_action="Refund",
            )
            return simulation, report

    def factory(customer_policy, service_policy):
        seen["customer"] = customer_policy.policy_id
        seen["service"] = service_policy.policy_id
        return Pipeline()

    from framework.evolution.schemas import ServicePolicy
    customer = CustomerPolicy(policy_id="c-policy")
    service = ServicePolicy(policy_id="s-policy")
    case = SplitManager().build().evolution[0]
    episodes = EvoSAGEEpisodeEvaluator(factory).evaluate(customer, service, [case], "validation", 0, "test")
    assert seen == {"customer": "c-policy", "service": "s-policy"}
    assert episodes[0].task_success is True


def test_llm_generators_return_valid_structured_candidates_without_api():
    class Response:
        def __init__(self, text):
            self.text = text

    class FakeClient:
        def __init__(self, text):
            self.text = text

        def generate(self, **kwargs):
            return Response(self.text)

    customer = CustomerPolicy()
    generated = LLMCustomerPolicyGenerator(FakeClient('[{"name":"authority","description":"ask for an explanation","strategy_tags":["authority_challenge"],"disclosure_strategy":"answer necessary questions","pressure_strategy":"remain firm","contradiction_strategy":"ask for clarification","response_to_verification":"acknowledge the result","response_to_rejection":"request a reason"}]')).generate(customer, [], ServicePolicy(), 1, 1)
    patch = LLMServicePatchGenerator(FakeClient('[{"category":"VERIFICATION","text":"Verify authoritative results before deciding.","rationale":"failure-driven"}]')).generate(ServicePolicy(), [], 1, 1)
    assert generated[0].strategy_tags == ["authority_challenge"]
    assert patch[0].rules[0].category == "VERIFICATION"
