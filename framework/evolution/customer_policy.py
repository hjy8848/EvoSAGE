"""Customer policy validation, compilation and deterministic simulation."""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, Optional

from ..backend.types import CaseSpec
from ..core.customer_contract import get_customer_opening_contract
from ..llm_integration.llm_user_model import RuleUserModel
from ..models import UserProfile
from .schemas import CustomerPolicy, PolicyValidationError


class CustomerPolicyValidator:
    """Hard validity gate for reusable customer strategies."""

    _protocol_attack = re.compile(r"(ignore\s+(the|all)\s+instructions|prompt\s+injection|parser|json\s*hack|evaluator\s+manipulat)", re.I)

    DEFAULT_TAGS = {
        "truthful", "cooperative", "withholding", "pressure", "contradiction",
        "delayed_disclosure", "authority_challenge", "delayed_contradiction",
        "escalation", "paraphrase",
    }

    def __init__(self, allowed_tags=None):
        self.allowed_tags = set(allowed_tags) if allowed_tags is not None else set(self.DEFAULT_TAGS)

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
        self._identifier_request_count = {"order_id": 0, "record_id": 0, "customer_id": 0}
        self._identifier_disclosed: set[str] = set()
        self._rejected_once = False

    def generate_initial_message(self) -> str:
        goal = self.profile.user_intent.replace("_", " ")
        opening_contract = get_customer_opening_contract(self.case_spec)
        required_order_id = next((item["value"] for item in opening_contract if item["field"] == "order_id"), None)
        if required_order_id:
            # CaseSpec's opening disclosure is a hard contract and overrides
            # any reusable policy tag that would otherwise delay disclosure.
            if "cooperative" in self.policy.strategy_tags:
                return f"您好，我愿意配合核验{goal}，订单号是{required_order_id}，请帮我处理。"
            if {"authority_challenge", "contradiction"} & set(self.policy.strategy_tags):
                return f"您好，我想处理{goal}，订单号是{required_order_id}；若记录和我的理解不一致，请帮我核实并说明。"
            return f"您好，我想处理{goal}，订单号是{required_order_id}，请帮我核实并处理。"
        if "paraphrase" in self.policy.strategy_tags:
            return f"您好，关于{goal}这件事，我想请您先帮我查一下具体状态。"
        if "cooperative" in self.policy.strategy_tags:
            return f"您好，我愿意配合核验{goal}所需的信息，请帮我处理。"
        if "pressure" in self.policy.strategy_tags:
            return f"您好，我现在比较着急，想处理{goal}，请先帮我核实订单状态。"
        if "contradiction" in self.policy.strategy_tags:
            return f"您好，我想处理{goal}。如果系统记录和我理解的不一致，请解释具体原因。"
        return f"您好，我想咨询{goal}，请帮我核实并处理。"

    def generate_next_message(self, agent_last_message: str, turn_count: int, context=None) -> str:
        text = agent_last_message or ""
        lower = text.lower()
        requested_field = self._requested_identifier_field(text)
        if requested_field and requested_field not in self._identifier_disclosed:
            self._identifier_request_count[requested_field] += 1
            value = self._known_identifier(requested_field)
            if not value:
                return "我目前没有这个编号，请先告诉我还能用什么信息核验。"
            tags = set(self.policy.strategy_tags)
            if "withholding" in tags and not self._agent_explained_identifier_need(lower):
                return "能否先说明需要这个编号核验什么，以便我确认后再提供？"
            if "delayed_disclosure" in tags and self._identifier_request_count[requested_field] == 1:
                return "我确认一下，下一条把编号提供给您。"
            self._identifier_disclosed.add(requested_field)
            return value
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
        if "truthful" in self.policy.strategy_tags and turn_count >= 2:
            return "我只补充我确认过的信息，请依据系统核验结果继续处理。"
        if "paraphrase" in self.policy.strategy_tags:
            return "麻烦换一种方式说明当前核验结果和下一步。"
        if turn_count >= 3 and "pressure" in self.policy.strategy_tags:
            return "我已经等待了一段时间，请给出明确的处理结果。"
        return "请继续帮我核实相关状态。"

    @staticmethod
    def _requested_identifier_field(text: str) -> Optional[str]:
        lower = text.lower()
        if any(token in lower for token in ("订单号", "订单编号", "order id", "order_id")):
            return "order_id"
        if any(token in lower for token in ("记录编号", "记录号", "record id", "record_id")):
            return "record_id"
        if any(token in lower for token in ("客户号", "客户编号", "customer id", "customer_id")):
            return "customer_id"
        if "编号" in text or " id" in lower:
            return "order_id" if "order_id" in (getattr(self.case_spec, "user_knowledge", {}) or {}) else "record_id"
        return None

    def _known_identifier(self, field: str) -> Optional[str]:
        if self.case_spec is None:
            return None
        knowledge = self.case_spec.user_knowledge or {}
        knows_flag = {"order_id": "knows_order_id", "record_id": "knows_record_id", "customer_id": "knows_customer_id"}[field]
        reveal_flag = {"order_id": "reveal_order_id_on_request", "record_id": "reveal_record_id_on_request", "customer_id": "reveal_customer_id_on_request"}[field]
        value = knowledge.get(field)
        if not (knowledge.get(knows_flag) and value and (self.case_spec.user_policy or {}).get(reveal_flag, False)):
            return None
        return str(value)

    @staticmethod
    def _agent_explained_identifier_need(lower_message: str) -> bool:
        return any(token in lower_message for token in (
            "为了核验", "用于核验", "需要核实", "用于查询", "订单核验", "验证订单",
        ))


def policy_for_case(policy: CustomerPolicy, case_spec: CaseSpec) -> CustomerPolicy:
    return CustomerPolicyCompiler().compile(policy, case_spec)
