#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM集成的用户模型
LLM-powered User Model

使用LLM生成更自然的用户消息
"""

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
from ..core.customer_contract import get_customer_opening_contract

logger = logging.getLogger(__name__)


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
        self.case_spec = case_spec
        self.generation_retry_limit = 1
        self.customer_simulator_provenance: list[dict[str, Any]] = []
        self.backend_events = []
        self.environment_state = UserEnvironmentState(
            goal=case_spec.user_goal if case_spec else {"type": profile.user_intent},
            facts=case_spec.user_knowledge if case_spec else {},
            known_facts=case_spec.user_knowledge if case_spec else {},
        )
        
        # 新增: 追踪问题是否已解决
        self.problem_status = "unsolved"  # unsolved, partially_solved, solved
        self.last_courtesy_turn = -1  # 记录上次礼貌性结束的轮次
        
        if llm_client is None:
            logger.warning("No LLM client provided. User message generation may be limited.")

    def attach_case_spec(self, case_spec: CaseSpec) -> None:
        """Bind this customer simulator to one deterministic benchmark case."""
        self.case_spec = case_spec

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
                desired = (self.case_spec.expected_outcome if self.case_spec else {})
                action_name = result.get("action_name")
                if desired and action_name:
                    self.problem_status = "partially_solved"
                    self.environment_state.resolution_status = "partially_solved"
                else:
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
            required_order_id=self._mandatory_opening_order_id(),
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

    def _mandatory_opening_order_id(self) -> Optional[str]:
        """Return the required opening identifier from the shared contract."""
        return next((
            item["value"]
            for item in get_customer_opening_contract(self.case_spec)
            if item["field"] == "order_id"
        ), None)

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
        required_order_id: Optional[str] = None,
    ) -> str:
        attempts: list[dict[str, Any]] = []
        invalid_reason = "empty_message"
        max_attempts = 1 + self.generation_retry_limit

        for attempt_index in range(1, max_attempts + 1):
            started = time.perf_counter()
            response = None
            provider_error = None
            timed_out = False
            try:
                response = self.llm_client.generate(
                    prompt=prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                latency = float(
                    (getattr(response, "metadata", {}) or {}).get("latency_seconds")
                    or (time.perf_counter() - started)
                )
                raw_content = self._raw_response_content(response)
                finish_reason = self._response_finish_reason(response)
                metadata = getattr(response, "metadata", {}) or {}
                raw_response = getattr(response, "raw_response", None)
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
                latency = time.perf_counter() - started
                raw_content = ""
                finish_reason = None
                request_id = None
                input_tokens = 0
                completion_tokens = 0
                reasoning_tokens = None
                timed_out = self._timeout_exception(exc)
                provider_error = self._safe_provenance_text(
                    f"{type(exc).__name__}: {exc}"
                )
                invalid_reason = "timeout" if timed_out else "provider_error"

            if response is not None:
                if str(finish_reason or "").lower() in {"length", "max_tokens"}:
                    invalid_reason = "output_truncated"
                else:
                    cleaned = self._clean_generated_message(raw_content, record_courtesy=False)
                    if not cleaned.strip():
                        invalid_reason = "empty_message"
                    elif required_order_id and required_order_id not in cleaned:
                        invalid_reason = "missing_mandatory_order_id"
                    else:
                        invalid_reason = ""

            attempt_record = {
                "attempt_index": attempt_index,
                "finish_reason": finish_reason,
                "max_tokens": self.max_tokens,
                "raw_content": self._safe_provenance_text(raw_content),
                "provider_request_id": request_id,
                "latency_seconds": latency,
                "input_tokens": int(input_tokens or 0),
                "completion_tokens": int(completion_tokens or 0),
                "reasoning_tokens": reasoning_tokens,
                "provider_error": provider_error,
                "timeout": timed_out,
                "parse_status": "NOT_APPLICABLE",
                "generation_status": "valid" if not invalid_reason else "invalid",
                "invalid_reason": invalid_reason or None,
            }
            attempts.append(attempt_record)

            if not invalid_reason:
                self.customer_simulator_provenance.append({
                    "stage": stage,
                    "turn_index": turn_index,
                    "status": "valid",
                    "retry_count": attempt_index - 1,
                    "attempts": attempts,
                })
                self._record_courtesy(cleaned)
                return cleaned

        generation_record = {
            "stage": stage,
            "turn_index": turn_index,
            "status": "invalid",
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
        """
        构建初始消息生成的提示词
        支持多个场景: online_education, ecommerce_refund, telecom_package, 
                   property_service, logistics_delivery, airline_refund
        
        Returns:
            str: 初始消息生成提示词
        """
        # 根据对抗强度确定语气引导
        intensity_guidance = {
            "zero_conflict": "友好礼貌，但简洁直接",
            "weak_conflict": "礼貌但带有一些疑虑或急切",
            "strong_conflict": "不满或急躁，语气较强硬"
        }
        intensity_desc = intensity_guidance.get(self.profile.adversarial_intensity, "正常交互")
        
        # 根据场景选择人物身份
        scenario_id = self.profile.scenario_id
        role_desc = self._get_role_description()
        opening_contract = get_customer_opening_contract(self.case_spec)
        contract_text = ""
        if opening_contract:
            disclosures = "\n".join(
                f"- {item['field']}: {json.dumps(item['value'], ensure_ascii=False)}"
                for item in opening_contract
            )
            contract_text = f"""
【强制首轮披露约定】
你的第一条消息必须包含以下你已知的信息：
{disclosures}
这是 CaseSpec 的硬约束，优先于 CustomerPolicy 中任何暂缓、隐瞒或延后披露这些字段的指示。CustomerPolicy 仍可调整表达方式、语气和对抗行为，但不得省略上述信息。
不得披露 Customer 不知道的事实或任何后台私有状态。
"""

        prompt = f"""你正在扮演{role_desc}，准备向客服发起对话。

【你的身份】
- 意图: {self.profile.user_intent}
- 对抗强度: {intensity_desc}
- 情感状态: {self.emotion_state.value}
- 场景: {scenario_id}

【你的背景和问题】
{self.system_prompt}

【你实际知道的信息】
{self._known_case_facts_text()}
{contract_text}

【要求】
1. 根据身份、背景、已知信息和以上强制约定，生成第一条开场消息
2. 消息自然、简洁(30-60字)，直接表达问题或诉求，并按对抗强度调整语气: {intensity_desc}
3. 只提供当前诉求和必要事实；不要主动泄露无关信息，也不要为了多轮而隐藏必要事实
4. 只输出下一条 Customer 消息本身，不要提供解释、分析或额外格式

【你的第一条消息】
"""
        return prompt
    
    def _build_generation_prompt(
        self,
        agent_last_message: str,
        turn_count: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """构建用户消息生成提示词"""
        
        # 构建对话上下文
        dialogue_context = "\n".join([
            f"{'用户' if msg['role'] == 'user' else '客服'}: {msg['content']}"
            for msg in self.dialogue_history[-4:]  # 最近4条消息
        ])
        
        # 根据对抗强度调整指导
        intensity = self.profile.adversarial_intensity
        intensity_guidance = {
            "zero_conflict": "友好配合，表现出信任和耐心",
            "weak_conflict": "有些疑虑但愿意配合，提出合理的问题",
            "strong_conflict": "对立态度，需要证据支撑，可能有些挑剔",
        }
        
        intensity_desc = intensity_guidance.get(intensity, "正常交互")
        
        # 判断问题是否已解决
        self._update_problem_status(agent_last_message)
        
        # 根据问题状态调整prompt
        if self.problem_status == "solved" and turn_count >= 3:
            # 问题已解决且对话至少3轮,应该礼貌结束
            ending_guidance = """
【重要】你的问题已经得到解决！请简短地表示感谢并礼貌地结束对话。
不要再提出新的问题或继续讨论,避免无意义的重复感谢。
可以说："谢谢您的帮助，问题解决了，再见！"或类似的话。
"""
        elif turn_count - self.last_courtesy_turn <= 1 and self.last_courtesy_turn > 0:
            # 如果上一轮已经表示过感谢,这一轮应该结束对话
            ending_guidance = """
【重要】你上一轮已经表示过感谢了，现在应该直接说"再见"结束对话，不要继续重复感谢！
"""
        else:
            ending_guidance = """
【对话策略】
- 如果你的问题已经得到满意的解答,简短感谢后说"再见"结束对话
- 如果还有疑问,继续追问,但要聚焦在核心问题上
- 避免空泛的感谢和祝福,要么提问要么结束
"""
        
        # 根据场景选择人物身份
        scenario_id = self.profile.scenario_id
        role_desc = self._get_role_description()
        
        prompt = f"""你正在扮演{role_desc},继续与客服的对话。

【用户身份】
- 意图: {self.profile.user_intent}
- 对抗强度: {intensity_desc}
- 情感状态: {self.emotion_state.value}
- 满意度: {self.satisfaction_score:.1f}/1.0
- 当前轮次: {turn_count}
- 场景: {scenario_id}

【对话上下文】
{dialogue_context}

【你实际知道的信息】
{self._known_case_facts_text()}

【最近的业务事件】
{self._recent_backend_event_text()}

【客服最后的消息】
{agent_last_message}

{ending_guidance}

【要求】
1. 根据对话上下文和客服消息生成你的下一条回复
2. 保持角色一致性,但避免过度礼貌导致对话无法结束
3. 回复应该自然、简洁(20-50字),不要过度感谢
4. 根据满意度和情感状态调整态度
5. 只返回你的消息,不要有额外的说明

【你的下一条消息】
"""
        return prompt

    def _known_case_facts_text(self) -> str:
        """Render only customer-known facts, never the full backend record."""
        if not self.case_spec:
            return "暂无额外结构化信息。"
        knowledge = self.case_spec.user_knowledge or {}
        policy = self.case_spec.user_policy or {}
        visible = {}
        if (
            knowledge.get("knows_order_id")
            and knowledge.get("order_id")
            and policy.get("show_order_id_initially", True)
        ):
            visible["order_id"] = knowledge["order_id"]
        if knowledge.get("knows_record_id") and knowledge.get("record_id"):
            visible["record_id"] = knowledge["record_id"]
        if knowledge.get("knows_customer_id") and knowledge.get("customer_id"):
            visible["customer_id"] = knowledge["customer_id"]
        if knowledge.get("product_name"):
            visible["product_name"] = knowledge["product_name"]
        if knowledge.get("believes_shipping_status"):
            visible["user_belief_about_shipping"] = knowledge["believes_shipping_status"]
        visible["can_reveal_order_id_on_request"] = policy.get(
            "reveal_order_id_on_request", False
        )
        visible["can_reveal_record_id_on_request"] = policy.get(
            "reveal_record_id_on_request", False
        )
        visible["can_reveal_customer_id_on_request"] = policy.get(
            "reveal_customer_id_on_request", False
        )
        return json.dumps(visible, ensure_ascii=False)

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
