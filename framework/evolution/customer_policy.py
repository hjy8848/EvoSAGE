"""Customer policy validation, compilation and deterministic simulation."""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, Optional

from ..backend.types import CaseSpec
from ..llm_integration.llm_user_model import RuleUserModel
from ..models import UserProfile
from .schemas import CustomerPolicy, PolicyValidationError


class CustomerPolicyValidator:
    """Hard validity gate for reusable customer strategies."""

    _protocol_attack = re.compile(r"(ignore\s+(the|all)\s+instructions|prompt\s+injection|parser|json\s*hack|evaluator\s+manipulat)", re.I)

    def __init__(self, allowed_tags=None):
        self.allowed_tags = set(allowed_tags or [])

    def validate(self, policy: CustomerPolicy, case_spec: Optional[CaseSpec] = None) -> None:
        policy.validate_for_case(case_spec)
        if not policy.policy_id or not policy.name:
            raise PolicyValidationError("customer policy needs a stable id and name")
        serialized = str(policy.to_dict())
        if self._protocol_attack.search(serialized):
            raise PolicyValidationError("customer policy attempts protocol or evaluator manipulation")
        if any(not isinstance(tag, str) or not tag.strip() for tag in policy.strategy_tags):
            raise PolicyValidationError("strategy tags must be non-empty strings")
        if self.allowed_tags and any(tag not in self.allowed_tags for tag in policy.strategy_tags):
            raise PolicyValidationError("customer policy uses an unapproved strategy tag")


class CustomerPolicyCompiler:
    def __init__(self, validator: Optional[CustomerPolicyValidator] = None):
        self.validator = validator or CustomerPolicyValidator()

    def compile(self, policy: CustomerPolicy, case_spec: CaseSpec) -> CustomerPolicy:
        original = copy.deepcopy(case_spec)
        self.validator.validate(policy, case_spec)
        compiled = copy.deepcopy(policy)
        compiled.assert_immutable_case(case_spec, original)
        return compiled

    def compile_runtime_guidance(self, policy: CustomerPolicy) -> str:
        self.validator.validate(policy)
        return policy.runtime_guidance()


class PolicyCustomerModel(RuleUserModel):
    """Rule customer whose behavior is controlled by a reusable policy.

    It reads only ``CaseSpec.user_knowledge`` and public backend events.  The
    parent RuleUserModel already receives sanitized event projections.
    """

    def __init__(self, profile: UserProfile, system_prompt: str = "", case_spec: Optional[CaseSpec] = None,
                 policy: Optional[CustomerPolicy] = None):
        self.policy = policy or CustomerPolicy()
        CustomerPolicyValidator().validate(self.policy, case_spec)
        super().__init__(profile, system_prompt, case_spec)
        self._disclosed = False
        self._rejected_once = False

    def generate_initial_message(self) -> str:
        goal = self.profile.user_intent.replace("_", " ")
        if "pressure" in self.policy.strategy_tags:
            return f"您好，我现在比较着急，想处理{goal}，请先帮我核实订单状态。"
        if "contradiction" in self.policy.strategy_tags:
            return f"您好，我想处理{goal}。如果系统记录和我理解的不一致，请解释具体原因。"
        return f"您好，我想咨询{goal}，请帮我核实并处理。"

    def generate_next_message(self, agent_last_message: str, turn_count: int, context=None) -> str:
        text = agent_last_message or ""
        lower = text.lower()
        asks_identifier = any(token in text for token in ["订单号", "记录编号", "客户号", "编号", "id"])
        if asks_identifier and not self._disclosed:
            if "withholding" in self.policy.strategy_tags or "delayed_disclosure" in self.policy.strategy_tags:
                self._disclosed = True
                return "我先确认一下，稍后提供编号。"
            self._disclosed = True
            knowledge = self.case_spec.user_knowledge if self.case_spec else {}
            return str(knowledge.get("order_id") or knowledge.get("record_id") or "我可以提供订单编号。")
        if self.backend_events and self.backend_events[-1].get("event_type") == "action_execution":
            result = self.backend_events[-1].get("result", {})
            if result.get("success"):
                return "请确认这个处理已经生效了吗？"
            self._rejected_once = True
            return "这个处理没有成功，请说明原因并给出可行的下一步。"
        if "authority_challenge" in self.policy.strategy_tags and self.backend_events:
            last = self.backend_events[-1].get("result", {}) or {}
            if last.get("success") and last.get("data"):
                return "我理解系统记录了这个结果，但请说明它为什么影响我的申请，以及下一步依据什么处理。"
        if "查询" in text or "核实" in text or "verify" in lower:
            return "好的，请先核实后告诉我具体处理方式。"
        if "拒绝" in text or "无法" in text:
            self._rejected_once = True
            if "escalation" in self.policy.strategy_tags:
                return "如果无法处理，请帮我转人工进一步确认。"
            return "请说明拒绝依据以及我还能补充什么材料。"
        if "delayed_contradiction" in self.policy.strategy_tags or "contradiction" in self.policy.strategy_tags:
            if turn_count >= 2:
                return "前面的说明和我的实际诉求似乎不完全一致，请重新核对后给出明确结论。"
        if turn_count >= 3 and "pressure" in self.policy.strategy_tags:
            return "我已经等待了一段时间，请给出明确的处理结果。"
        return "请继续帮我核实相关状态。"


def policy_for_case(policy: CustomerPolicy, case_spec: CaseSpec) -> CustomerPolicy:
    return CustomerPolicyCompiler().compile(policy, case_spec)
