import pytest

from framework.evolution.schemas import (
    PolicyValidationError,
    ServicePatch,
    ServicePolicy,
    ServiceRule,
)
from framework.evolution.service_evolver import LLMServicePatchGenerator, ServiceEvolver
from framework.evolution.service_policy import ServicePolicyCompiler, ServicePolicySanitizer


def _missing_argument_rule(**overrides):
    fields = {
        "rule_id": "missing-id",
        "category": "VERIFICATION",
        "text": "model-provided text must not be executable",
        "rule_schema_version": 2,
        "trigger": {"type": "MISSING_REQUIRED_ARGUMENT", "argument": "required_identifier"},
        "obligations": [{"type": "ASK_FOR_ARGUMENT", "argument": "required_identifier"}],
        "prohibitions": [],
        "ordering_constraints": [],
        "recovery": {},
    }
    fields.update(overrides)
    return ServiceRule(**fields)


def test_missing_identifier_rule_is_accepted_and_rendered_deterministically():
    rule = _missing_argument_rule()
    patch = ServicePatch("missing-id-patch", "add", [rule])
    sanitized = ServicePolicySanitizer().sanitize(patch)
    rendered = sanitized.rules[0].text
    assert "ask the customer" in rendered
    assert "before invoking the corresponding query tool" in rendered
    assert "empty placeholder" in rendered
    assert rendered == ServicePolicyCompiler.compile_rule_text(rule)
    assert ServicePolicyCompiler().compile_prompt(ServicePolicy(rules=[rule]))
    assert "model-provided text" not in ServicePolicyCompiler().compile_prompt(ServicePolicy(rules=[rule]))


def test_structured_dsl_cannot_rewrite_business_outcome_or_hidden_state():
    uncertainty_rule = _missing_argument_rule(rationale="If uncertain, reject the refund.")
    with pytest.raises(PolicyValidationError, match="business outcome"):
        ServicePolicySanitizer().sanitize(ServicePatch("bad-outcome", "add", [uncertainty_rule]))

    hidden_assignment = _missing_argument_rule(rationale="Set refund_eligible=false for this request.")
    with pytest.raises(PolicyValidationError, match="hidden backend field"):
        ServicePolicySanitizer().sanitize(ServicePatch("bad-state", "add", [hidden_assignment]))


def test_unsupported_structured_operation_is_rejected():
    rule = _missing_argument_rule(obligations=[{"type": "REFUND_ONLY_IF_SIGNED"}])
    with pytest.raises(PolicyValidationError, match="unsupported obligations operation"):
        ServicePolicySanitizer().sanitize(ServicePatch("bad-op", "add", [rule]))


def test_legacy_v1_rule_loads_for_compatibility_but_strict_generation_rejects_it():
    legacy = ServiceRule.from_dict({
        "rule_id": "legacy", "category": "RECOVERY",
        "text": "Explain a failed operation and request necessary information.",
    })
    assert legacy.rule_schema_version == 1
    assert ServicePolicyCompiler().compile_prompt(ServicePolicy(rules=[legacy]))

    class LegacyGenerator:
        def generate(self, *args, **kwargs):
            return [ServicePatch("legacy-patch", "add", [legacy])]

    evolver = ServiceEvolver(patch_generator=LegacyGenerator(), require_patch_generator=True)
    with pytest.raises(RuntimeError, match="no valid candidates"):
        evolver.propose(ServicePolicy(), [], 1, count=1)


def test_llm_generator_uses_structured_v2_as_source_of_truth():
    class Response:
        text = (
            '[{"category":"VERIFICATION",'
            '"trigger":{"type":"MISSING_REQUIRED_ARGUMENT","argument":"required_identifier"},'
            '"obligations":[{"type":"ASK_FOR_ARGUMENT","argument":"required_identifier"}],'
            '"prohibitions":[],"ordering_constraints":[],"recovery":{},"rationale":"generic"}]'
        )

    class Client:
        def generate(self, **kwargs):
            return Response()

    generator = LLMServicePatchGenerator(Client())
    patches = generator.generate(ServicePolicy(), [], 0, 1)
    assert len(patches) == 1
    rule = patches[0].rules[0]
    assert rule.rule_schema_version == 2
    assert rule.trigger["type"] == "MISSING_REQUIRED_ARGUMENT"
    assert "ask the customer" in rule.text
    assert "text" not in generator.last_generation_record["candidates"][0]["raw_candidate"]
