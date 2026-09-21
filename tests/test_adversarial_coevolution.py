import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import requests

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
from framework.llm_integration.llm_client import LiteLLMClient
from framework.models import UserProfile


def ecommerce_case():
    from framework.sop.ecommerce_refund_PathList import generate_path_list
    return build_case_spec("ecommerce_refund", "refund_before_shipping", generate_path_list()[5], "test-user")


def test_role_token_budgets_load_as_nested_config():
    config = EvolutionConfig.from_dict({
        "evaluation": {
            "token_budget": {"user": 256, "agent": 1024, "judge": 512,
                              "customer_evolver": 2048, "service_evolver": 3072},
            "summary_limit": 3,
        }
    })
    assert config.evaluation.token_budget.agent == 1024
    assert config.evaluation.token_budget.service_evolver == 3072
    assert config.evaluation.summary_limit == 3


def test_customer_policy_roundtrip_and_leakage_gate():
    case = ecommerce_case()
    policy = CustomerPolicy(strategy_tags=["truthful"], description="delay disclosure until asked")
    CustomerPolicyValidator().validate(policy, case)
    assert CustomerPolicy.from_dict(policy.to_dict()).to_dict() == policy.to_dict()
    bad = CustomerPolicy(description=f"use {case.case_id} and expected_action")
    with pytest.raises(PolicyValidationError):
        CustomerPolicyValidator().validate(bad, case)


def test_customer_policy_validator_uses_value_boundaries():
    # ``Reject`` must not match the ordinary word ``rejection`` and ``Low``
    # must not match the word ``workflow`` in the baseline policy text.
    CustomerPolicyValidator().validate(CustomerPolicy(), ecommerce_case())


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


def test_real_episode_preserves_backend_tool_call_trace():
    from framework.evolution.evaluator_adapter import EvoSAGEEpisodeEvaluator
    simulation = _fake_real_simulation()
    simulation.backend_events = [
        {"event_type": "tool_call", "name": "query_order"},
        {"event_type": "tool_call", "name": "submit_refund"},
        # Non-tool audit events must not be counted as tool calls.
        {"event_type": "state_change", "name": "refund_submitted"},
    ]
    episode = EvoSAGEEpisodeEvaluator.from_evosage(
        simulation, _fake_real_report(), CustomerPolicy(), ServicePolicy(), "validation", 0, "test",
    )
    assert episode.tool_sequence_summary == ["query_order", "submit_refund"]


def test_real_evaluator_and_runner_share_one_fresh_run_dir(tmp_path):
    """The real evaluator cache must never be created in the pre-isolation dir."""
    from framework.evolution.persistence import RunStore
    from framework.evolution.real_factory import make_real_evaluator

    requested_dir = tmp_path / "run"
    old_store = RunStore(requested_dir)
    old_store.write_json("config/evolution.json", {"old_run": True})
    old_files_before = {
        path.relative_to(requested_dir): path.read_bytes()
        for path in requested_dir.rglob("*")
        if path.is_file()
    }

    config = EvolutionConfig(
        max_generations=0,
        persistence=PersistenceConfig(output_dir=str(requested_dir), resume=False),
    )
    resolved_dir, isolated = EvolutionRunner.resolve_run_dir(config)
    assert isolated is True

    evaluator = make_real_evaluator(
        model="test-model",
        api_url="http://127.0.0.1:1/v1",
        api_key="test-key",
        output_dir=resolved_dir,
        max_turns=1,
        api_timeout=1,
        resume=False,
    )
    runner = EvolutionRunner(config, evaluator=evaluator, run_dir=resolved_dir)
    result = runner.run()

    assert Path(result["run_dir"]) == resolved_dir
    assert runner.store.run_dir == resolved_dir
    assert evaluator._cache_path == resolved_dir / "environment" / "episode_cache.jsonl"
    assert (resolved_dir / "environment" / "episode_cache.jsonl").exists()
    assert (resolved_dir / "split_manifest" / "evolution_cases.json").exists()
    assert (resolved_dir / "analysis" / "orchestration_metrics.json").exists()
    assert (resolved_dir / "report.md").exists()
    assert list(tmp_path.glob("run_fresh_*")) == [resolved_dir]

    old_files_after = {
        path.relative_to(requested_dir): path.read_bytes()
        for path in requested_dir.rglob("*")
        if path.is_file()
    }
    assert old_files_after == old_files_before


@pytest.mark.parametrize(
    ("error_factory", "expected_timeout"),
    [
        (TimeoutError, True),
        (requests.exceptions.Timeout, True),
        (type("WrappedTimeoutError", (Exception,), {}), True),
        (ValueError, False),
    ],
)
def test_litellm_timeout_diagnostics_classify_wrapped_errors(
    monkeypatch, error_factory, expected_timeout
):
    class FakeLiteLLM:
        suppress_debug_info = False

        @staticmethod
        def completion(**kwargs):
            raise error_factory("simulated failure")

    monkeypatch.setitem(sys.modules, "litellm", FakeLiteLLM)
    client = LiteLLMClient(
        api_key="test-key",
        base_url="http://127.0.0.1:1/v1",
        model_name="test-model",
        timeout=1,
        max_retries=1,
    )
    with pytest.raises(Exception):
        client.generate("ping")

    stats = client.request_stats()
    assert stats["attempts"] == 1
    assert stats["failures"] == 1
    assert stats["timeouts"] == int(expected_timeout)


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


def test_fresh_run_isolates_existing_run_state(tmp_path):
    config = EvolutionConfig(
        max_generations=0,
        persistence=PersistenceConfig(output_dir=str(tmp_path / "run")),
    )
    first = EvolutionRunner(config, evaluator=MockEpisodeEvaluator())
    first.run()
    second = EvolutionRunner(config, evaluator=MockEpisodeEvaluator())
    assert second.fresh_run_isolated is True
    assert second.store.run_dir != first.store.run_dir


def test_explicit_resolved_run_dir_is_shared_without_second_isolation(tmp_path):
    config = EvolutionConfig(
        max_generations=0,
        persistence=PersistenceConfig(output_dir=str(tmp_path / "run")),
    )
    first = EvolutionRunner(config, evaluator=MockEpisodeEvaluator())
    first.run()

    resolved, isolated = EvolutionRunner.resolve_run_dir(config)
    assert isolated is True
    second = EvolutionRunner(config, evaluator=MockEpisodeEvaluator(), run_dir=resolved)
    assert second.store.run_dir == resolved
    assert second.fresh_run_isolated is True


def test_failed_run_persists_orchestration_diagnostics(tmp_path):
    class FailingEvaluator:
        def evaluate(self, *args, **kwargs):
            raise RuntimeError("synthetic provider failure")

    config = EvolutionConfig(
        max_generations=1,
        persistence=PersistenceConfig(output_dir=str(tmp_path / "failed-run")),
    )
    runner = EvolutionRunner(config, evaluator=FailingEvaluator())
    with pytest.raises(RuntimeError, match="synthetic provider failure"):
        runner.run()

    metrics_path = runner.store.run_dir / "analysis" / "orchestration_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["run_status"] == "failed"
    assert metrics["failure_type"] == "RuntimeError"
    assert metrics["failure_message"] == "synthetic provider failure"


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


def test_real_adapter_caches_episode_but_separates_judge_modes(tmp_path):
    calls = []

    class Pipeline:
        def run_single_simulation(self, intent, user_id=None, path_config=None, judge_enabled=True):
            calls.append({"intent": intent, "judge_enabled": judge_enabled})
            return _fake_real_simulation(), _fake_real_report()

    def factory(customer_policy, service_policy, judge_enabled=True):
        calls.append({"factory_judge_enabled": judge_enabled})
        return Pipeline()

    cache_path = tmp_path / "episode_cache.jsonl"
    evaluator = EvoSAGEEpisodeEvaluator(factory, cache_namespace="cache-test", cache_path=cache_path)
    customer = CustomerPolicy(policy_id="cache-customer")
    service = ServicePolicy(policy_id="cache-service")
    case = SplitManager().build().evolution[0]

    first = evaluator.evaluate(customer, service, [case], "evolution", 0, "customer_candidate")
    second = evaluator.evaluate(customer, service, [case], "evolution", 0, "selected_customer")
    assert len(calls) == 2  # one factory call plus one real episode
    assert first[0].metadata["cache_hit"] is False
    assert second[0].metadata["cache_hit"] is True
    assert calls[-1]["judge_enabled"] is False

    heldout = evaluator.evaluate(customer, service, [case], "heldout_test", 0, "heldout")
    assert heldout[0].metadata["cache_hit"] is False
    assert calls[-2] == {"factory_judge_enabled": True}
    assert calls[-1]["judge_enabled"] is True

    restarted_calls = []

    class RestartedPipeline(Pipeline):
        def run_single_simulation(self, *args, **kwargs):
            restarted_calls.append(True)
            return super().run_single_simulation(*args, **kwargs)

    restarted = EvoSAGEEpisodeEvaluator(
        lambda *args, **kwargs: RestartedPipeline(),
        cache_namespace="cache-test",
        cache_path=cache_path,
    )
    restored = restarted.evaluate(customer, service, [case], "evolution", 0, "customer_candidate")
    assert restored[0].metadata["cache_hit"] is True
    assert restarted_calls == []

    fresh_calls = []

    class FreshPipeline(Pipeline):
        def run_single_simulation(self, *args, **kwargs):
            fresh_calls.append(True)
            return super().run_single_simulation(*args, **kwargs)

    fresh = EvoSAGEEpisodeEvaluator(
        lambda *args, **kwargs: FreshPipeline(),
        cache_namespace="cache-test",
        cache_path=cache_path,
        reset_cache=True,
    )
    fresh_result = fresh.evaluate(customer, service, [case], "evolution", 0, "customer_candidate")
    assert fresh_result[0].metadata["cache_hit"] is False
    assert fresh_calls == [True]


def test_service_latest_filter_skips_replay_and_normal_for_rejected_patch():
    class NonImprovingEvaluator:
        def __init__(self):
            self.phases = []

        def evaluate(self, customer_policy, service_policy, cases, split, generation, phase):
            self.phases.append(phase)
            improved = any("defensive" in rule.text.lower() for rule in service_policy.rules if rule.active)
            return [EpisodeResult(
                episode_id=f"{phase}-{index}", scenario="ecommerce_refund",
                case_id=getattr(case, "case_id", str(index)),
                customer_policy_id=customer_policy.policy_id,
                service_policy_id=service_policy.policy_id,
                split=split, generation=generation, task_success=improved,
                execution_score=1.0 if improved else 0.0,
                sage_style_score=1.0 if improved else 0.0,
                verification_score=1.0 if improved else 0.0,
                policy_score=1.0 if improved else 0.0,
                action_execution_score=1.0 if improved else 0.0,
                goal_fulfillment_score=1.0 if improved else 0.0,
            ) for index, case in enumerate(cases)]

    class PatchGenerator:
        def generate(self, *args, **kwargs):
            return [ServicePatch(
                "non-improving", "add",
                [ServiceRule("polite", "COMMUNICATION", "Be polite to the customer.")],
            )]

    split = SplitManager().build()
    evaluator = NonImprovingEvaluator()
    evolver = ServiceEvolver(patch_generator=PatchGenerator())
    _, decision, _ = evolver.evolve(
        ServicePolicy(), [], split.validation, split.validation, evaluator, 0,
        count=1, customer_policy=CustomerPolicy(),
        replay_policies=[CustomerPolicy()], replay_attack_count=1,
    )
    assert decision.accepted is False
    assert "service_candidate_latest" in evaluator.phases
    assert "service_candidate_replay" not in evaluator.phases
    assert "service_normal_candidate" not in evaluator.phases
    assert evolver.last_candidate_records[0]["reason"].startswith("latest_attack_filter:")


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
    assert len({item["service_policy_id"] for item in evolver.last_candidate_records}) == 3
    assert all(item["service_policy_id"].startswith("service_policy_s1_") for item in evolver.last_candidate_records)
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
