#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM集成的用户模型
LLM-powered User Model

使用LLM生成更自然的用户消息
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import copy
import json
import logging
import re
import time

from ..models import UserModel, UserProfile
from .llm_client import LLMClient
from ..backend.types import CaseSpec
from ..backend.types import UserEnvironmentState

logger = logging.getLogger(__name__)

CUSTOMER_SIMULATOR_PROTOCOL_RETRY_LIMIT = 1

_CUSTOMER_IDENTIFIER_KNOWS_FLAGS = {
    "order_id": "knows_order_id",
    "customer_id": "knows_customer_id",
    "record_id": "knows_record_id",
    "booking_id": "knows_booking_id",
    "ticket_id": "knows_ticket_id",
    "account_id": "knows_account_id",
}
@dataclass(frozen=True)
class CustomerCaseView:
    """Only the case information the simulated Customer may receive."""

    scenario: str
    user_goal: Dict[str, Any]
    user_knowledge: Dict[str, Any]
    user_policy: Dict[str, Any]


class CustomerSimulatorProtocolError(RuntimeError):
    """Raised when a Customer message remains invalid after one retry."""

    customer_simulator_protocol_invalid = True
    retry_exhausted = True

    def __init__(self, reason: str, provenance: list[dict[str, Any]]):
        self.reason = (
            reason if reason.startswith("customer_simulator_invalid:")
            else f"customer_simulator_invalid:{reason}"
        )
        self.customer_simulator_provenance = copy.deepcopy(provenance)
        super().__init__(self.reason)


class LLMUserModel(UserModel):
    """
    LLM驱动的用户模型
    
    使用LLM生成用户的下一条消息，而不是使用固定的模板
    """
    
    def __init__(
        self,
        profile: UserProfile,
        system_prompt: str = "",
        llm_client: Optional[LLMClient] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = 512,
        case_spec: Optional[CaseSpec] = None,
        thinking_mode: Optional[str] = None,
        protocol_retry_limit: int = CUSTOMER_SIMULATOR_PROTOCOL_RETRY_LIMIT,
        customer_policy: Optional[Any] = None,
    ):
        """
        初始化LLM用户模型
        
        Args:
            profile: 用户画像
            system_prompt: 系统提示词
            llm_client: LLM客户端
            temperature: 采样温度
            max_tokens: 最大生成token数
        """
        super().__init__(profile, system_prompt)
        self.llm_client = llm_client
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.case_spec = self._customer_visible_case(case_spec)
        self.customer_policy = customer_policy
        self.customer_policy_guidance = (
            customer_policy.runtime_guidance()
            if customer_policy is not None
            and callable(getattr(customer_policy, "runtime_guidance", None))
            else ""
        )
        self.customer_policy_id = getattr(customer_policy, "policy_id", None)
        self.customer_policy_fingerprint = (
            customer_policy.semantic_fingerprint()
            if customer_policy is not None
            and callable(getattr(customer_policy, "semantic_fingerprint", None))
            else None
        )
        if thinking_mode not in {None, "enabled", "disabled"}:
            raise ValueError("thinking_mode must be None, 'enabled', or 'disabled'")
        self.thinking_mode = thinking_mode
        self.generation_retry_limit = max(0, int(protocol_retry_limit))
        self.customer_simulator_provenance: list[dict[str, Any]] = []
        self.backend_events = []
        # Private knowledge is sourced only from the explicit CaseSpec
        # customer-knowledge contract. Disclosure flags affect turn behavior,
        # never whether a known fact exists in this state.
        self.private_knowledge = self._extract_private_knowledge(self.case_spec)
        self.environment_state = UserEnvironmentState(
            goal=self.case_spec.user_goal if self.case_spec else {"type": profile.user_intent},
            facts=copy.deepcopy(self.private_knowledge),
            known_facts=copy.deepcopy(self.private_knowledge),
        )
        if self.case_spec:
            self.initialize_emotion_from_case(
                (self.case_spec.user_policy or {}).get("initial_emotion")
            )
        
        # 新增: 追踪问题是否已解决
        self.problem_status = "unsolved"  # unsolved, partially_solved, solved
        self.last_courtesy_turn = -1  # 记录上次礼貌性结束的轮次
        
        if llm_client is None:
            logger.warning("No LLM client provided. User message generation may be limited.")

    def _emotion_prompt_line(self) -> str:
        """Render affect only when it is case-declared or changed in-dialogue."""
        if self.initial_emotion_was_explicit or self.emotion_state.value != "calm":
            return f"- 情感状态: {self.emotion_state.value}\n"
        return ""

    def attach_case_spec(self, case_spec: CaseSpec) -> None:
        """Bind to a sanitized customer-side projection, never full CaseSpec."""
        self.case_spec = self._customer_visible_case(case_spec)
        self.private_knowledge = self._extract_private_knowledge(self.case_spec)
        self.environment_state.facts = copy.deepcopy(self.private_knowledge)
        self.environment_state.known_facts = copy.deepcopy(self.private_knowledge)

    @staticmethod
    def _customer_visible_case(case_spec: Optional[CaseSpec]) -> Optional[CustomerCaseView]:
        if case_spec is None:
            return None
        if isinstance(case_spec, CustomerCaseView):
            return case_spec
        if isinstance(case_spec, dict):
            case_spec = CaseSpec(**case_spec)
        source_goal = dict(case_spec.user_goal or {})
        customer_goal_keys = {
            "type", "intent", "user_intent", "goal", "customer_goal", "request",
        }
        public_goal = {
            key: copy.deepcopy(value)
            for key, value in source_goal.items()
            if key.lower() in customer_goal_keys
        }
        # Retain only the Customer's own affective state. Disclosure flags and
        # truthfulness modes are legacy behavior controls, not adversary limits.
        source_policy = case_spec.user_policy or {}
        public_policy = (
            {"initial_emotion": source_policy["initial_emotion"]}
            if "initial_emotion" in source_policy else {}
        )
        return CustomerCaseView(
            scenario=str(case_spec.scenario),
            user_goal=public_goal,
            user_knowledge=copy.deepcopy(case_spec.user_knowledge or {}),
            user_policy=public_policy,
        )

    @staticmethod
    def _extract_private_knowledge(case_spec: Optional[CaseSpec]) -> Dict[str, Any]:
        """Copy facts explicitly known by the Customer, never backend truth.

        Identifier values are included only when the corresponding ``knows_*``
        flag is true. ``show_*_initially`` and ``reveal_*_on_request`` are
        deliberately not consulted here; they are disclosure policy, not
        knowledge state.
        """
        if case_spec is None:
            return {}
        source = case_spec.user_knowledge or {}
        private: Dict[str, Any] = {}
        for field, value in source.items():
            if str(field).startswith("knows_") or value is None:
                continue
            knows_flag = _CUSTOMER_IDENTIFIER_KNOWS_FLAGS.get(str(field))
            if knows_flag and not source.get(knows_flag):
                continue
            private[str(field)] = copy.deepcopy(value)
        return private

    def observe_backend_event(self, event: Dict[str, Any]) -> None:
        """Observe only the public result of a backend event.

        The customer simulator may know its own case facts, but it should not
        receive evaluator-only state snapshots through this method.
        """
        public_event = {
            "event_type": event.get("event_type"),
            "name": event.get("name"),
            "result": event.get("result", {}),
        }
        self.backend_events.append(public_event)
        if public_event["event_type"] == "action_execution":
            result = public_event["result"] or {}
            if result.get("success"):
                action_name = result.get("action_name")
                # This is only the Customer's observation that an action
                # reported success. The official evaluator alone determines
                # whether that action satisfied the hidden task outcome.
                self.problem_status = "solved"
                self.problem_resolved = True
                self.environment_state.resolution_status = "solved"
                self.environment_state.satisfaction = min(
                    1.0, self.environment_state.satisfaction + 0.2
                )
                if action_name in {"TransHuman", "TRANSFER_HUMAN"}:
                    self.environment_state.escalation_status = "requested"
    
    def generate_initial_message(self) -> str:
        """
        生成对话的第一条消息
        
        Returns:
            str: 生成的初始消息
        """
        if not self.llm_client:
            # 没有LLM客户端，返回默认消息
            return "您好，我有个问题想咨询一下。"
        
        prompt = self._build_initial_message_prompt()
        return self._generate_protocol_checked_message(
            prompt, stage="opening", turn_index=0,
        )
    
    def generate_next_message(
        self,
        agent_last_message: str,
        turn_count: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        使用LLM生成用户的下一条消息
        
        Args:
            agent_last_message: 客服最后的消息
            turn_count: 当前轮次
            context: 额外的上下文信息
            
        Returns:
            str: 生成的用户消息
        """
        if not self.llm_client:
            # 没有LLM客户端，返回默认消息
            return self._default_next_message(agent_last_message, turn_count)
        
        # 构建生成提示词
        prompt = self._build_generation_prompt(
            agent_last_message=agent_last_message,
            turn_count=turn_count,
            context=context,
        )
        
        return self._generate_protocol_checked_message(
            prompt, stage="reply", turn_index=turn_count,
        )

    def get_customer_simulator_provenance(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self.customer_simulator_provenance)

    @staticmethod
    def _raw_response_content(response: Any) -> str:
        raw = getattr(response, "raw_response", None)
        if isinstance(raw, dict):
            choices = raw.get("choices") or []
            if choices and isinstance(choices[0], dict):
                message = choices[0].get("message") or {}
                if isinstance(message, dict) and message.get("content") is not None:
                    return str(message.get("content"))
                if choices[0].get("text") is not None:
                    return str(choices[0]["text"])
        metadata = getattr(response, "metadata", {}) or {}
        assistant_message = metadata.get("assistant_message")
        if isinstance(assistant_message, dict) and assistant_message.get("content") is not None:
            return str(assistant_message["content"])
        return str(getattr(response, "text", "") or "")

    @staticmethod
    def _safe_provenance_text(value: Any) -> str:
        text = str(value or "")
        text = re.sub(r"(?i)\bBearer\s+\S+", "Bearer [REDACTED]", text)
        text = re.sub(r"\bsk-[A-Za-z0-9_-]{16,}\b", "[REDACTED_API_KEY]", text)
        return text

    @classmethod
    def _parse_customer_response(cls, raw_content: str) -> tuple[str, dict[str, Any]]:
        """Accept plain user text; optionally unwrap historical JSON envelopes.

        Historical JSON envelopes may be unwrapped for compatibility, but any
        claim fields are ignored. Business claims are not annotated or
        truth-checked; the official backend/evaluator defines world truth.
        """
        raw = str(raw_content or "").strip()
        candidate = raw
        if candidate.startswith("```"):
            candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.I)
        if not raw:
            return "", {"parse_status": "invalid", "parse_error": "empty_output"}
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError) as exc:
            if candidate.startswith(("{", "[")):
                return "", {
                    "parse_status": "invalid",
                    "parse_error": f"malformed_customer_json:{type(exc).__name__}",
                }
            return raw, {
                "parse_status": "not_applicable",
                "parse_error": None,
            }
        if isinstance(parsed, str) and parsed.strip():
            utterance = parsed.strip()
        elif isinstance(parsed, dict):
            utterance = parsed.get("utterance")
            if not isinstance(utterance, str) or not utterance.strip():
                return "", {
                    "parse_status": "invalid",
                    "parse_error": "customer_json_missing_utterance",
                }
            utterance = utterance.strip()
        else:
            return "", {
                    "parse_status": "invalid",
                    "parse_error": "customer_json_not_text_or_envelope",
                }
        return utterance, {
            "parse_status": "valid_envelope",
            "parse_error": None,
        }

    @staticmethod
    def _response_finish_reason(response: Any) -> Optional[str]:
        metadata = getattr(response, "metadata", {}) or {}
        finish_reason = metadata.get("finish_reason")
        raw = getattr(response, "raw_response", None)
        if not finish_reason and isinstance(raw, dict):
            choices = raw.get("choices") or []
            if choices and isinstance(choices[0], dict):
                finish_reason = choices[0].get("finish_reason")
        return str(finish_reason) if finish_reason is not None else None

    @staticmethod
    def _timeout_exception(error: BaseException) -> bool:
        return (
            "timeout" in type(error).__name__.lower()
            or "timed out" in str(error).lower()
            or "timeout" in str(error).lower()
        )

    def _generate_protocol_checked_message(
        self,
        prompt: str,
        stage: str,
        turn_index: int,
    ) -> str:
        attempts: list[dict[str, Any]] = []
        invalid_reason = "empty_message"
        max_attempts = 1 + self.generation_retry_limit

        for attempt_index in range(1, max_attempts + 1):
            started = time.perf_counter()
            response = None
            provider_error = None
            timed_out = False
            thinking_effective = "unknown" if self.thinking_mode else "provider_default"
            try:
                request_options = {}
                if self.thinking_mode is not None:
                    request_options["thinking"] = {"type": self.thinking_mode}
                response = self.llm_client.generate(
                    prompt=prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    **request_options,
                )
                latency = float(
                    (getattr(response, "metadata", {}) or {}).get("latency_seconds")
                    or (time.perf_counter() - started)
                )
                raw_content = self._raw_response_content(response)
                finish_reason = self._response_finish_reason(response)
                metadata = getattr(response, "metadata", {}) or {}
                thinking_effective = metadata.get("thinking_effective", thinking_effective)
                raw_response = getattr(response, "raw_response", None)
                assistant_message = metadata.get("assistant_message")
                if not isinstance(assistant_message, dict) and isinstance(raw_response, dict):
                    choices = raw_response.get("choices") or []
                    if choices and isinstance(choices[0], dict):
                        assistant_message = choices[0].get("message")
                reasoning_content_present = bool(
                    isinstance(assistant_message, dict)
                    and assistant_message.get("reasoning_content")
                )
                request_id = metadata.get("request_id") or metadata.get("id")
                if not request_id and isinstance(raw_response, dict):
                    request_id = raw_response.get("id")
                usage = metadata.get("usage", {}) or {}
                if not usage and isinstance(raw_response, dict):
                    usage = raw_response.get("usage", {}) or {}
                input_tokens = usage.get("prompt_tokens", usage.get("promptTokens", 0))
                completion_tokens = usage.get(
                    "completion_tokens", usage.get("completionTokens", getattr(response, "tokens", 0))
                )
                reasoning_tokens = usage.get("reasoning_tokens", usage.get("reasoningTokens"))
                if reasoning_tokens is None:
                    completion_details = usage.get("completion_tokens_details") or usage.get("completionTokensDetails") or {}
                    reasoning_tokens = completion_details.get("reasoning_tokens", completion_details.get("reasoningTokens"))
                provider_error = None
            except Exception as exc:
                if getattr(exc, "budget_exhausted", False):
                    raise
                latency = time.perf_counter() - started
                raw_content = ""
                finish_reason = None
                request_id = None
                input_tokens = 0
                completion_tokens = 0
                reasoning_tokens = None
                reasoning_content_present = False
                timed_out = self._timeout_exception(exc)
                provider_error = self._safe_provenance_text(
                    f"{type(exc).__name__}: {exc}"
                )
                invalid_reason = "timeout" if timed_out else "provider_error"

            parse_metadata = None
            if response is not None:
                if str(finish_reason or "").lower() in {"length", "max_tokens"}:
                    invalid_reason = "output_truncated"
                else:
                    utterance, parse_metadata = self._parse_customer_response(raw_content)
                    cleaned = self._clean_generated_message(utterance, record_courtesy=False)
                    if not cleaned.strip():
                        invalid_reason = parse_metadata.get("parse_error") or "empty_message"
                    else:
                        invalid_reason = ""

            attempt_record = {
                "attempt_index": attempt_index,
                "thinking_mode": self.thinking_mode or "default",
                "thinking_requested": self.thinking_mode or "default",
                "thinking_effective": thinking_effective,
                "finish_reason": finish_reason,
                "max_tokens": self.max_tokens,
                "raw_content": self._safe_provenance_text(raw_content),
                "provider_request_id": request_id,
                "latency_seconds": latency,
                "input_tokens": int(input_tokens or 0),
                "completion_tokens": int(completion_tokens or 0),
                "reasoning_tokens": reasoning_tokens,
                "reasoning_content_present": reasoning_content_present,
                "provider_error": provider_error,
                "timeout": timed_out,
                "parse_status": (
                    parse_metadata.get("parse_status") if parse_metadata else "NOT_APPLICABLE"
                ),
                "parse_error": parse_metadata.get("parse_error") if parse_metadata else None,
                "generation_status": "valid" if not invalid_reason else "invalid",
                "invalid_reason": invalid_reason or None,
            }
            attempts.append(attempt_record)

            if not invalid_reason:
                self.customer_simulator_provenance.append({
                    "stage": stage,
                    "turn_index": turn_index,
                    "thinking_mode": self.thinking_mode or "default",
                    "customer_policy_id": self.customer_policy_id,
                    "customer_policy_fingerprint": self.customer_policy_fingerprint,
                    "policy_guidance_injected": bool(
                        self.customer_policy_guidance or "【客户交互策略" in prompt
                    ),
                    "status": "valid",
                    "message_parse_status": parse_metadata.get("parse_status"),
                    "message_parse_error": parse_metadata.get("parse_error"),
                    "retry_count": attempt_index - 1,
                    "attempts": attempts,
                })
                self._record_courtesy(cleaned)
                return cleaned

        generation_record = {
            "stage": stage,
            "turn_index": turn_index,
            "thinking_mode": self.thinking_mode or "default",
            "customer_policy_id": self.customer_policy_id,
            "customer_policy_fingerprint": self.customer_policy_fingerprint,
                "policy_guidance_injected": bool(
                    self.customer_policy_guidance or "【客户交互策略" in prompt
                ),
                "status": "invalid",
                "message_parse_status": None,
            "invalid_reason": invalid_reason,
            "retry_count": max_attempts - 1,
            "attempts": attempts,
        }
        self.customer_simulator_provenance.append(generation_record)
        raise CustomerSimulatorProtocolError(
            invalid_reason, self.customer_simulator_provenance
        )

    def _record_courtesy(self, message: str) -> None:
        thank_keywords = ["谢谢", "感谢", "thank"]
        blessing_keywords = ["祝", "顺利", "愉快", "加油", "进步"]
        if any(keyword in message for keyword in thank_keywords + blessing_keywords):
            self.last_courtesy_turn = self.current_turn
    
    def _get_role_description(self) -> str:
        """
        根据场景获取人物身份描述
        
        Returns:
            str: 人物身份描述
        """
        scenario_id = self.profile.scenario_id
        role_descriptions = {
            "online_education": "一名在线教育学员",
            "ecommerce_refund": "一名电商平台的买家",
            "telecom_package": "一名电信运营商的客户",
            "property_service": "一名物业管理区域的住户",
            "logistics_delivery": "一名期待收货的寄件人或收件人",
            "airline_refund": "一名航空公司的乘客",
        }
        return role_descriptions.get(scenario_id, "一名客户")
    
    def _build_initial_message_prompt(self) -> str:
        """Build an opening prompt from Customer-visible state only."""
        return self._build_adversarial_message_prompt(stage="opening", turn_count=0)
    
    def _build_generation_prompt(
        self,
        agent_last_message: str,
        turn_count: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build a reply prompt from Customer-visible state only."""
        return self._build_adversarial_message_prompt(
            stage="reply", agent_last_message=agent_last_message, turn_count=turn_count,
        )

    def _build_adversarial_message_prompt(
        self,
        stage: str,
        agent_last_message: str = "",
        turn_count: int = 0,
    ) -> str:
        """Render the strategy against customer-visible state only."""
        dialogue_context = "\n".join(
            f"{'Customer' if item.get('role') == 'user' else 'Service'}: {item.get('content', '')}"
            for item in self.dialogue_history[-6:]
        ) or "(opening turn)"
        strategy = str(getattr(self.customer_policy, "strategy", "") or "").strip()
        if not strategy:
            strategy = "Respond naturally to the customer's stated goal and the visible conversation."
        goal = dict(self.case_spec.user_goal) if self.case_spec else {"type": self.profile.user_intent}
        role_desc = self._get_role_description()
        emotion = self._emotion_prompt_line()
        emotion_context = emotion.strip() if emotion else "(not specified; respond naturally)"
        stage_text = "Open the conversation" if stage == "opening" else "Reply to the Service's latest turn"
        return f"""You are simulating a Customer who interacts with a fixed customer-service Service.

Role: {role_desc}
Customer goal/context: {json.dumps(goal, ensure_ascii=False)}
Customer affective context: {emotion_context}
Customer-side information available to you: {self._known_case_facts_text()}
Public tool/action results you have observed: {self._recent_backend_event_text()}
Current turn: {turn_count}

Your free-text Customer strategy (follow it directly; do not translate it into tags or a checklist):
{strategy}

When your strategy calls for adversarial behavior, you may lie, claim to know something you do not
know, invent or misstate business details, conceal information, contradict yourself, mislead, apply
pressure, or change tactics to influence the Service. These are valid Customer behaviors. The official
backend/evaluator determines world truth; your words do not change it. You are not given hidden backend
state, expected actions or paths, gold answers, or evaluator internals. Use only the role context,
customer-side information, public results, and visible dialogue.

Remain the Customer. Do not attempt to alter the benchmark case, tools, backend, evaluator, scoring,
data splits, or experiment records. Do not claim that you changed those systems.

Recent dialogue:
{dialogue_context}

Service's latest message:
{agent_last_message or '(none)'}

{stage_text}. Produce one natural, concise Customer message that follows your strategy and attempts to
influence this fixed Service. Do not be artificially cooperative or disclose facts just because the
simulator knows them. Return only the message text, with no JSON, claim annotations, labels, or analysis.
"""

    def _known_case_facts_text(self) -> str:
        """Render Customer-private CaseSpec facts, independent of disclosure timing."""
        return json.dumps(self.private_knowledge, ensure_ascii=False)

    def _recent_backend_event_text(self) -> str:
        if not self.backend_events:
            return "暂无。"
        return json.dumps(self.backend_events[-3:], ensure_ascii=False)
    
    def _update_problem_status(self, agent_message: str) -> None:
        """
        根据客服的回复更新问题解决状态
        
        Args:
            agent_message: 客服的回复消息
        """
        agent_msg_lower = agent_message.lower()
        
        # 检测表示问题已解决的关键词
        solved_keywords = [
            '已经解决', '问题解决', '已处理', '已完成', '成功提交',
            '退款已', '已批准', '已通过', '方案已', '资源已分配',
            '祝您学习愉快', '祝您学习顺利', '期待您', '欢迎随时',
            '如果还有其他问题', '如有其他问题', '随时联系我们'
        ]
        
        # 检测表示部分解决的关键词
        partial_keywords = [
            '正在处理', '会尽快', '稍后', '我们会', '预计',
            '建议您', '可以尝试', '先复习', '请您'
        ]
        
        # 检测表示需要更多信息的关键词
        need_info_keywords = [
            '能否', '请问', '可以提供', '具体', '详细',
            '哪个', '什么时候', '如何', '怎么'
        ]
        
        # 判断状态
        if any(keyword in agent_message for keyword in solved_keywords):
            # 客服使用了"祝您学习愉快"等结束语,说明认为问题已处理
            self.problem_status = "solved"
        elif any(keyword in agent_msg_lower for keyword in need_info_keywords):
            # 客服还在询问信息,问题未解决
            self.problem_status = "unsolved"
        elif any(keyword in agent_message for keyword in partial_keywords):
            # 客服说"正在处理",部分解决
            self.problem_status = "partially_solved"
        
        # 如果客服连续2次没有提问,也认为问题基本解决
        if len(self.dialogue_history) >= 4:
            last_two_agent = [
                msg['content'] for msg in self.dialogue_history[-4:]
                if msg['role'] == 'assistant'
            ]
            if len(last_two_agent) >= 2:
                has_questions = any(
                    any(q in msg for q in ['?', '?', '能否', '请问', '可以'])
                    for msg in last_two_agent[-2:]
                )
                if not has_questions and self.problem_status != "unsolved":
                    self.problem_status = "solved"
    
    def _clean_generated_message(self, message: str, record_courtesy: bool = True) -> str:
        """清理生成的消息"""
        # 移除可能的引号
        if message.startswith('"') and message.endswith('"'):
            message = message[1:-1]
        
        # 移除可能的前缀
        for prefix in ["我:", "用户:", "学员:", "我说:", "我的回复:"]:
            if message.startswith(prefix):
                message = message[len(prefix):].strip()
        
        # 限制长度
        if len(message) > 500:
            message = message[:500]
        
        # 检测是否是礼貌性结束消息
        thank_keywords = ['谢谢', '感谢', 'thank']
        blessing_keywords = ['祝', '顺利', '愉快', '加油', '进步']
        if record_courtesy and (any(keyword in message for keyword in thank_keywords) or
            any(keyword in message for keyword in blessing_keywords)):
            # 记录这是一次礼貌性回复
            self.last_courtesy_turn = self.current_turn
        
        return message.strip()
    
    def _default_next_message(self, agent_last_message: str, turn_count: int) -> str:
        """
        默认消息生成（无LLM时）
        支持多个场景: online_education, ecommerce_refund, telecom_package, 
                   property_service, logistics_delivery, airline_refund
        """
        scenario_id = self.profile.scenario_id
        
        # 通用逻辑：根据对话历史和情感推断下一步
        if "能否" in agent_last_message or "可否" in agent_last_message or "?" in agent_last_message:
            # 客服在询问信息
            if scenario_id == "online_education":
                return "我是在学第三章第二节的时候，对函数参数默认值不理解。"
            elif scenario_id == "ecommerce_refund":
                return "这是订单号XXXX，商品已收到但有质量问题。"
            elif scenario_id == "telecom_package":
                return "我当前用的是88元套餐，想了解一下升级到128元套餐的费用。"
            elif scenario_id == "property_service":
                return "我家的厕所漏水问题已经很严重了，能否尽快派人来维修？"
            elif scenario_id == "logistics_delivery":
                return "订单号是12345，寄件地址是北京朝阳区XXX。"
            elif scenario_id == "airline_refund":
                return "我的航班号是CA1234，由于个人原因需要改签。"
            else:
                return "您能详细说明一下具体情况吗？"
        
        if "退费" in agent_last_message or "赔偿" in agent_last_message or "退款" in agent_last_message:
            if self.emotion_state.value == "angry":
                return "我必须要求退款，这是我的权利！"
            else:
                return "能否给我一个合理的解决方案？"
        
        if "安抚" in agent_last_message or "理解" in agent_last_message or "歉意" in agent_last_message:
            self.update_satisfaction(0.1)
            return "谢谢你的耐心帮助。"
        
        if "祝您" in agent_last_message or "再见" in agent_last_message or "有需要" in agent_last_message:
            # 结束话语
            return "谢谢，再见！"
        
        # 默认消息
        return "好的，谢谢你的帮助。"


class RuleUserModel(UserModel):
    """Deterministic customer policy used as a non-LLM evaluation baseline."""

    def __init__(self, profile: UserProfile, system_prompt: str = "", case_spec: Optional[CaseSpec] = None):
        super().__init__(profile, system_prompt)
        self.case_spec = case_spec
        self.backend_events = []
        self.environment_state = UserEnvironmentState(
            goal=case_spec.user_goal if case_spec else {"type": profile.user_intent},
            facts=case_spec.user_knowledge if case_spec else {},
            known_facts=case_spec.user_knowledge if case_spec else {},
        )
        if case_spec:
            self.initialize_emotion_from_case(
                (case_spec.user_policy or {}).get("initial_emotion")
            )
        self.problem_status = "unsolved"

    def generate_initial_message(self) -> str:
        goal = self.profile.user_intent.replace("_", " ")
        return f"您好，我想咨询{goal}，请帮我核实并处理。"

    def observe_backend_event(self, event: Dict[str, Any]) -> None:
        self.backend_events.append({
            "event_type": event.get("event_type"),
            "name": event.get("name"),
            "result": event.get("result", {}),
        })
        if event.get("event_type") == "action_execution" and event.get("result", {}).get("success"):
            self.environment_state.resolution_status = "partially_solved"
            self.environment_state.satisfaction = min(1.0, self.environment_state.satisfaction + 0.15)

    def generate_next_message(self, agent_last_message: str, turn_count: int, context=None) -> str:
        if self.environment_state.resolution_status == "solved":
            return "谢谢，问题解决了，再见。"
        if any(token in agent_last_message for token in ["订单号", "记录编号", "客户号"]):
            knowledge = self.case_spec.user_knowledge if self.case_spec else {}
            return knowledge.get("order_id") or knowledge.get("record_id") or "我先核对一下编号。"
        if self.backend_events and self.backend_events[-1]["event_type"] == "action_execution":
            return "请确认这个处理是否已经生效？"
        return "请先帮我核实相关状态，再告诉我可以怎么处理。"


class RewritingUserModel(LLMUserModel):
    """LLM customer variant that requests paraphrased/adversarial expressions."""

    def _build_initial_message_prompt(self) -> str:
        return super()._build_initial_message_prompt() + "\n请使用与常见模板不同的自然表达。"

    def _build_generation_prompt(self, agent_last_message: str, turn_count: int, context=None) -> str:
        return super()._build_generation_prompt(agent_last_message, turn_count, context) + \
            "\n请避免复用上一轮句式，保持业务事实不变。\n"


class LLMUserMessageGenerator:
    """LLM用户消息生成器"""
    
    def __init__(self, llm_client: LLMClient):
        """
        初始化生成器
        
        Args:
            llm_client: LLM客户端
        """
        self.llm_client = llm_client
    
    def __call__(
        self,
        user_model: UserModel,
        agent_last_message: str,
        turn_count: int,
        **kwargs
    ) -> str:
        """
        使该对象可调用，直接调用generate方法
        """
        return self.generate(
            user_model=user_model,
            agent_last_message=agent_last_message,
            turn_count=turn_count,
            **kwargs
        )
    
    def generate(
        self,
        user_model: UserModel,
        agent_last_message: str,
        turn_count: int,
        **kwargs
    ) -> str:
        """
        生成用户的下一条消息
        
        Args:
            user_model: 用户模型
            agent_last_message: 客服最后的消息
            turn_count: 当前轮次
            
        Returns:
            str: 生成的用户消息
        """
        if not isinstance(user_model, LLMUserModel) and not hasattr(user_model, "generate_next_message"):
            # 如果不是LLMUserModel，创建临时的生成提示词
            prompt = self._build_simple_prompt(
                user_intent=user_model.profile.user_intent,
                dialogue_history=user_model.dialogue_history,
                agent_message=agent_last_message,
                turn_count=turn_count,
            )
        elif isinstance(user_model, LLMUserModel):
            # 使用LLMUserModel的生成逻辑
            return user_model.generate_next_message(
                agent_last_message=agent_last_message,
                turn_count=turn_count,
                context=kwargs,
            )
        else:
            return user_model.generate_next_message(
                agent_last_message=agent_last_message,
                turn_count=turn_count,
                context=kwargs,
            )
        
        try:
            response = self.llm_client.generate(prompt=prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error generating message: {type(e).__name__}: {e}")
            # 当LLM生成失败时，返回简单但有效的回复
            if turn_count > 2:
                return "好的，我理解了。"
            else:
                return "好的，谢谢。"
    
    def _build_simple_prompt(
        self,
        user_intent: str,
        dialogue_history: list,
        agent_message: str,
        turn_count: int,
    ) -> str:
        """构建简单的生成提示词"""
        dialogue_context = "\n".join([
            f"{'用户' if msg['role'] == 'user' else '客服'}: {msg['content']}"
            for msg in dialogue_history[-4:]
        ])
        
        prompt = f"""继续这个对话。用户意图是"{user_intent}"，生成用户的下一条消息。

【对话历史】
{dialogue_context}

【客服最后的消息】
{agent_message}

只返回用户的下一条消息，不要有任何额外说明。
"""
        return prompt
