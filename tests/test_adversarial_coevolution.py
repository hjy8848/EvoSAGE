import json
from types import SimpleNamespace

import pytest

from framework.backend.factory import build_case_spec
from framework.evolution.archives import AttackArchive
from framework.evolution.attribution import infer_failure_location
from framework.evolution.config import EvolutionConfig, PersistenceConfig, SplitConfig
from framework.evolution.customer_policy import CustomerPolicyValidator, PolicyCustomerModel
from framework.evolution.evaluator_adapter import MockEpisodeEvaluator
from framework.evolution.evaluator_adapter import EvoSAGEEpisodeEvaluator
from framework.evolution.customer_evolver import LLMCustomerPolicyGenerator
from framework.evolution.customer_evolver import CustomerEvolver
from framework.evolution.customer_selector import CustomerSelector
from framework.evolution.service_evolver import LLMServicePatchGenerator
from framework.evolution.service_evolver import ServiceEvolver
from framework.evolution.runner import EvolutionRunner
from framework.evolution.schemas import CustomerPolicy, EpisodeResult, FailureSignature, PolicyValidationError, ServicePatch, ServicePolicy, ServiceRule
from framework.evolution.service_gate import ServiceGate
from framework.evolution.service_policy import ServicePolicySanitizer
from framework.evolution.split_manager import SplitManager
from framework.evolution.weakness_frontier import WeaknessFrontier
from framework.models import UserProfile


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


def test_customer_policy_compilation_changes_runtime_guidance_not_case_goal():
    case = ecommerce_case()
    profile = UserProfile(user_id="u", user_intent="refund_before_shipping", adversarial_intensity="weak_conflict", scenario_id="ecommerce_refund")
    cooperative = CustomerPolicy(strategy_tags=["truthful", "cooperative"])
    challenging = CustomerPolicy(strategy_tags=["authority_challenge", "delayed_contradiction"])
    first = PolicyCustomerModel(profile, case_spec=case, policy=cooperative)
    second = PolicyCustomerModel(profile, case_spec=case, policy=challenging)
    assert cooperative.runtime_guidance() != challenging.runtime_guidance()
    assert first.environment_state.goal == second.environment_state.goal == case.user_goal
    assert first.generate_initial_message() != second.generate_initial_message()


def _fake_real_report(**overrides):
    values = {
        "details": {},
        "error_categories": [],
        "task_success": True,
        "required_verification_score": 1.0,
        "policy_compliance_score": 1.0,
        "action_execution_score": 1.0,
        "goal_fulfillment": 1.0,
        "execution_score": 1.0,
        "sage_style_score": 1.0,
        "predicted_action": "Refund",
        "executed_action": "Refund",
        "gold_path": ["step1", "step2", "step3"],
        "predicted_path": ["step1", "step2", "step3"],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _fake_real_simulation():
    return SimpleNamespace(
        simulation_id="sim-attribution",
        scenario_id="ecommerce_refund",
        case_spec={"case_id": "CASE-ATTR", "metadata": {}},
        backend_events=[],
        termination_reason="goal_satisfied",
        turns=[],
        model_name="fake",
    )


def test_real_episode_attribution_uses_first_canonical_divergence():
    from framework.evolution.evaluator_adapter import EvoSAGEEpisodeEvaluator
    episode = EvoSAGEEpisodeEvaluator.from_evosage(
        _fake_real_simulation(),
        _fake_real_report(predicted_path=["step1", "wrong", "step3"]),
        CustomerPolicy(), ServicePolicy(), "validation", 0, "test",
    )
    assert episode.path_step_index == 1
    assert episode.sop_node == "step2"


def test_real_episode_attribution_uses_final_action_stage_for_execution_failure():
    from framework.evolution.evaluator_adapter import EvoSAGEEpisodeEvaluator
    report = _fake_real_report(
        task_success=False,
        action_execution_score=0.0,
        error_categories=["wrong_final_action"],
    )
    episode = EvoSAGEEpisodeEvaluator.from_evosage(
        _fake_real_simulation(), report, CustomerPolicy(), ServicePolicy(), "validation", 0, "test",
    )
    assert episode.path_step_index == 2
    assert episode.sop_node == "step3"


def test_missing_real_path_attribution_is_not_fabricated():
    from framework.evolution.evaluator_adapter import EvoSAGEEpisodeEvaluator
    report = _fake_real_report(gold_path=[], predicted_path=[])
    episode = EvoSAGEEpisodeEvaluator.from_evosage(
        _fake_real_simulation(), report, CustomerPolicy(), ServicePolicy(), "validation", 0, "test",
    )
    assert episode.path_step_index is None
    assert episode.sop_node is None


def test_weakness_frontier_falls_back_to_path_step_then_unknown():
    path_episode = EpisodeResult(
        "path", "ecommerce_refund", "case-path", "c", "s", "validation", 0, False, 0.0,
        error_types=["wrong_final_action"], path_step_index=3,
    )
    unknown_episode = EpisodeResult(
        "unknown", "ecommerce_refund", "case-unknown", "c", "s", "validation", 0, False, 0.0,
        error_types=["action_failure"],
    )
    frontier = WeaknessFrontier()
    frontier.add([path_episode, unknown_episode])
    labels = {row["sop_node"] for row in frontier.to_dicts()}
    assert "path_step_3" in labels
    assert "unknown" in labels


def test_customer_fitness_uses_legitimate_failures_only():
    protocol = EpisodeResult(
        "protocol", "ecommerce_refund", "case-protocol", "c", "s", "validation", 0, False, 0.0,
        error_types=["json_parse_failed"], metadata={"protocol_failure": True},
    )
    legitimate = EpisodeResult(
        "legitimate", "ecommerce_refund", "case-legitimate", "c", "s", "validation", 0, False, 0.0,
        error_types=["wrong_final_action"],
    )
    selector = CustomerSelector()
    protocol_score = selector.score(CustomerPolicy(), [protocol], set())
    legitimate_score = selector.score(CustomerPolicy(), [legitimate], set())
    assert protocol_score.attack_success == 0.0
    assert protocol_score.novelty == 0.0
    assert legitimate_score.attack_success == 1.0
    assert legitimate_score.novelty > 0.0


def test_attack_archive_excludes_protocol_failure_novelty(tmp_path):
    policy = CustomerPolicy(policy_id="protocol-policy")
    episode = EpisodeResult(
        "protocol-archive", "ecommerce_refund", "case-protocol-archive", policy.policy_id, "s", "evolution", 0, False, 0.0,
        error_types=["protocol_failure"], metadata={"protocol_failure": True},
    )
    signature = FailureSignature.from_episode(episode)
    archive = AttackArchive(tmp_path / "protocol-attacks.jsonl")
    assert archive.add(policy, [signature], [episode], generation=0) == 0
    assert len(archive) == 0


def test_fresh_real_mode_propagates_strict_customer_generation(tmp_path):
    class BrokenGenerator:
        def generate(self, **kwargs):
            raise ValueError("provider unavailable")

    config = EvolutionConfig(
        splits=SplitConfig(max_cases=3),
        persistence=PersistenceConfig(output_dir=str(tmp_path / "fresh")),
    )
    source = CustomerEvolver(strategy_generator=BrokenGenerator(), require_strategy_generator=True)
    runner = EvolutionRunner(config, evaluator=MockEpisodeEvaluator(), customer_evolver=source)
    runner.split_manager.build()
    with pytest.raises(RuntimeError, match="strict real mode"):
        runner.fresh_adversary_evaluation(rounds=1, candidate_count=1)


def test_unknown_customer_strategy_is_rejected():
    with pytest.raises(PolicyValidationError):
        CustomerPolicyValidator().validate(CustomerPolicy(strategy_tags=["not_a_business_strategy"]))


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


def test_multiple_instances_per_path_are_unique_and_deterministic():
    config = SplitConfig(seed=11, instances_per_path=3)
    one = SplitManager(config).build()
    two = SplitManager(config).build()
    assert len(one.all_cases) == 45
    assert len({item.case_id for item in one.all_cases}) == 45
    assert [item.case_id for item in one.all_cases] == [item.case_id for item in two.all_cases]
    assert len({item.path_id for item in one.all_cases}) == 15
    limited = SplitManager(SplitConfig(seed=11, instances_per_path=3, max_cases=3)).build()
    assert len(limited.all_cases) == 3


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
    assert len(matrix) == 9
    config.persistence.resume = True
    assert EvolutionRunner(config, evaluator=MockEpisodeEvaluator()).run()["completed_generations"] == [0, 1]


def test_fresh_adversary_updates_incumbent_without_training_archive(tmp_path):
    config = EvolutionConfig(max_generations=1, persistence=PersistenceConfig(output_dir=str(tmp_path / "run")))
    runner = EvolutionRunner(config, evaluator=MockEpisodeEvaluator())
    runner.run()
    runner.fresh_adversary_evaluation(rounds=2, candidate_count=1)
    data = json.loads((tmp_path / "run" / "analysis" / "fresh_adversary_final.json").read_text())
    assert len(data["rounds"]) == 2
    assert data["training_archive_used"] is False
    assert data["rounds"][0]["selected_policy"]["policy_id"] != data["rounds"][1]["selected_policy"]["policy_id"]


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


def test_real_mode_does_not_silently_fallback_to_templates():
    class BrokenGenerator:
        def generate(self, **kwargs):
            raise ValueError("provider unavailable")

    with pytest.raises(RuntimeError, match="strict real mode"):
        CustomerEvolver(strategy_generator=BrokenGenerator(), require_strategy_generator=True).propose(
            CustomerPolicy(), 0, count=1
        )
    with pytest.raises(RuntimeError, match="strict real mode"):
        ServiceEvolver(patch_generator=BrokenGenerator(), require_patch_generator=True).propose(
            ServicePolicy(), [], 0, count=1
        )


def test_service_evaluates_all_candidates_and_replays_archived_attacker():
    from framework.evolution.schemas import FailureSignature

    class PatchGenerator:
        def generate(self, policy, failures, generation, count, *args):
            return [ServicePatch(
                f"patch-{index}", "add", [ServiceRule(f"rule-{index}", "ACTION_GROUNDING", "Use the authoritative action tool before claiming success")]
            ) for index in range(count)]

    split = SplitManager().build()
    customer = CustomerPolicy(strategy_tags=["authority_challenge"])
    mock = MockEpisodeEvaluator()
    failures = [FailureSignature.from_episode(item) for item in mock.evaluate(customer, ServicePolicy(), split.evolution, "evolution", 0, "failure_scan") if not item.task_success]
    evolver = ServiceEvolver(patch_generator=PatchGenerator())
    _, decision, _ = evolver.evolve(
        ServicePolicy(), failures, split.validation, split.validation, mock, 0, count=3,
        customer_policy=customer, replay_policies=[customer], replay_attack_count=1,
    )
    assert decision.accepted is True
    assert len(evolver.last_candidate_records) == 3
    assert any(call["phase"] == "service_candidate_replay" for call in mock.calls)


def test_attack_archive_reconstructs_customer_policy(tmp_path):
    from framework.evolution.schemas import FailureSignature, EpisodeResult
    policy = CustomerPolicy(policy_id="archived", strategy_tags=["authority_challenge"])
    episode = EpisodeResult("e", "ecommerce_refund", "c", policy.policy_id, "s", "evolution", 0, False, 0.0, error_types=["authoritative_conflict"])
    signature = FailureSignature.from_episode(episode)
    archive = AttackArchive(tmp_path / "attacks.jsonl")
    archive.add(policy, [signature], [episode], 0)
    loaded = AttackArchive(tmp_path / "attacks.jsonl")
    reconstructed = CustomerPolicy.from_dict(loaded.to_dicts()[0]["customer_policy"])
    assert reconstructed.policy_id == policy.policy_id
