#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对话模拟器
Dialogue Simulator

负责：
1. 管理用户模型和客服模型的交互
2. 多轮对话的流程控制
3. 对话结束条件判定
4. 对话数据的收集和保存
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import json

from ..models import UserModel, AgentModel, AgentTurnOutput
from ..backend import BackendEnvironment, CaseSpec


@dataclass
class SimulationTurn:
    """模拟单轮"""
    turn_id: int
    user_message: str
    agent_output: AgentTurnOutput
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Judge 评测结果（投票模式中保存三个 Judge 的原始输出）
    judge_evaluation: Optional[Dict[str, Any]] = None

    # 本轮事件快照。它们由 Simulator/Backend 生成，不由 Agent 自报。
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    action_calls: List[Dict[str, Any]] = field(default_factory=list)
    action_results: List[Dict[str, Any]] = field(default_factory=list)
    backend_state_before: Dict[str, Any] = field(default_factory=dict)
    backend_state_after: Dict[str, Any] = field(default_factory=dict)
    predicted_classification: Dict[str, Any] = field(default_factory=dict)
    predicted_path: List[str] = field(default_factory=list)
    predicted_action: str = ""
    executed_path: List[str] = field(default_factory=list)
    executed_action: str = ""


@dataclass
class SimulationResult:
    """模拟结果"""
    simulation_id: str
    scenario_id: str
    model_name: str
    user_intent: str
    adversarial_intensity: str
    
    # 对话过程
    turns: List[SimulationTurn] = field(default_factory=list)
    
    # Agent 可见的公开上下文；隐藏 backend_record 不放在这里
    context_data: Dict[str, Any] = field(default_factory=dict)

    # 结构化案例与后端审计轨迹。案例可供评估器使用，但不会注入 Agent prompt。
    case_spec: Optional[Dict[str, Any]] = None
    backend_events: List[Dict[str, Any]] = field(default_factory=list)
    backend_final_state: Dict[str, Any] = field(default_factory=dict)
    user_environment_state: Dict[str, Any] = field(default_factory=dict)
    
    # 终止信息
    termination_reason: str = ""
    final_status: str = ""  # success, failed, timeout, etc.
    goal_solved: bool = False
    
    # 指标数据
    dialogue_length: int = 0
    path_taken: List[str] = field(default_factory=list)
    actions_taken: List[str] = field(default_factory=list)
    predicted_path: List[str] = field(default_factory=list)
    predicted_action: str = ""
    executed_path: List[str] = field(default_factory=list)
    executed_action: str = ""
    
    # 时间戳
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: str = ""
    duration_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "simulation_id": self.simulation_id,
            "scenario_id": self.scenario_id,
            "model_name": self.model_name,
            "user_intent": self.user_intent,
            "adversarial_intensity": self.adversarial_intensity,
            "dialogue_length": self.dialogue_length,
            "path_taken": self.path_taken,
            "actions_taken": self.actions_taken,
            "predicted_path": self.predicted_path,
            "predicted_action": self.predicted_action,
            "executed_path": self.executed_path,
            "executed_action": self.executed_action,
            "termination_reason": self.termination_reason,
            "final_status": self.final_status,
            "goal_solved": self.goal_solved,
            "context_data": self.context_data,
            "case_spec": self.case_spec,
            "backend_events": self.backend_events,
            "backend_final_state": self.backend_final_state,
            "user_environment_state": self.user_environment_state,
            "duration_seconds": self.duration_seconds,
            "turns": [
                {
                    "turn_id": turn.turn_id,
                    "user_message": turn.user_message,
                    "agent_output": turn.agent_output.to_dict(),
                    "timestamp": turn.timestamp,
                    "judge_evaluation": turn.judge_evaluation if turn.judge_evaluation else None,
                    "tool_calls": turn.tool_calls,
                    "tool_results": turn.tool_results,
                    "action_calls": turn.action_calls,
                    "action_results": turn.action_results,
                    "backend_state_before": turn.backend_state_before,
                    "backend_state_after": turn.backend_state_after,
                    "predicted_classification": turn.predicted_classification,
                    "predicted_path": turn.predicted_path,
                    "predicted_action": turn.predicted_action,
                    "executed_path": turn.executed_path,
                    "executed_action": turn.executed_action,
                }
                for turn in self.turns
            ],
        }


class DialogueSimulator:
    """
    对话模拟器
    
    模拟用户和客服之间的多轮交互
    """
    
    def __init__(
        self,
        user_model: UserModel,
        agent_model: AgentModel,
        max_turns: int = 10,
        verbose: bool = False,
        backend_environment: Optional[BackendEnvironment] = None,
        case_spec: Optional[CaseSpec] = None,
        max_tool_steps: int = 8,
        legacy_execution: bool = False,
    ):
        """
        初始化对话模拟器
        
        Args:
            user_model: 用户模型
            agent_model: 客服模型
            max_turns: 最大轮次数
            verbose: 是否打印详细日志
        """
        self.user_model = user_model
        self.agent_model = agent_model
        self.max_turns = max_turns
        self.verbose = verbose
        self.backend_environment = backend_environment
        self.case_spec = case_spec
        self.max_tool_steps = max_tool_steps
        self.legacy_execution = legacy_execution
        
        self.simulation_turn = 0
    
    def run(
        self,
        initial_user_message: str,
        context_data: Optional[Dict[str, Any]] = None,
        user_message_generator = None
    ) -> SimulationResult:
        """
        运行对话模拟
        
        Args:
            initial_user_message: 初始用户消息
            context_data: 上下文数据 (学员信息、系统信息等)
            user_message_generator: 用户消息生成器 (可选，用于生成后续用户消息)
            
        Returns:
            SimulationResult: 模拟结果
        """
        import uuid
        import time
        
        simulation_id = f"sim_{uuid.uuid4().hex[:8]}"
        result = SimulationResult(
            simulation_id=simulation_id,
            scenario_id=self.agent_model.scenario_id,
            model_name="unknown",
            user_intent=self.user_model.profile.user_intent,
            adversarial_intensity=self.user_model.profile.adversarial_intensity,
            context_data=context_data or {},
            case_spec=self.case_spec.to_dict() if self.case_spec else None,
        )
        
        start_time = time.time()
        
        try:
            # 第一轮：用户发送初始消息
            if self.verbose:
                print(f"\n[{simulation_id}] 开始对话模拟")
                print(f"用户意图: {self.user_model.profile.user_intent}")
                print(f"对抗强度: {self.user_model.profile.adversarial_intensity}")
            
            # 第一轮对话
            self._execute_turn(
                result, initial_user_message, context_data, 0
            )
            
            # 后续轮次
            turn_count = 1
            while turn_count < self.max_turns:
                # 检查终止条件
                if self._should_terminate(result):
                    result.termination_reason = self._get_termination_reason(result)
                    break
                
                # 生成下一个用户消息
                next_user_message = self._generate_next_user_message(
                    result, user_message_generator
                )
                
                if not next_user_message:
                    result.termination_reason = "no_more_user_messages"
                    break
                
                # 执行对话轮次
                self._execute_turn(
                    result, next_user_message, context_data, turn_count
                )
                
                turn_count += 1
            
            # 完成模拟。API 模式下 AgentModel.path_taken 可能没有被填充，
            # 因此优先使用每轮由分类结果推导出的实际执行路径。
            result.dialogue_length = len(result.turns)
            if result.turns:
                result.path_taken = result.turns[-1].agent_output.path_taken or self.agent_model.path_taken
                result.predicted_path = list(result.turns[-1].predicted_path)
                result.predicted_action = result.turns[-1].predicted_action
            else:
                result.path_taken = self.agent_model.path_taken

            if self.backend_environment is not None:
                result.backend_events = self.backend_environment.get_event_log()
                result.backend_final_state = self.backend_environment.get_state_snapshot()
                result.executed_path = self._reconstruct_execution_path(result.backend_events)
                result.executed_action = self._last_successful_action(result.backend_events)
                result.actions_taken = [
                    event.get("result", {}).get("action_name")
                    for event in result.backend_events
                    if event.get("event_type") == "action_execution"
                    and event.get("result", {}).get("success")
                    and event.get("result", {}).get("action_name")
                ]
                result.goal_solved = self.backend_environment.goal_satisfied()
            else:
                result.actions_taken = [
                    turn.agent_output.action for turn in result.turns if turn.agent_output.action
                ]
                result.goal_solved = getattr(self.user_model, "problem_status", "") == "solved"
            if result.turns:
                # These remain Agent predictions; executed_* below comes only
                # from Backend event log.
                result.predicted_path = list(result.turns[-1].predicted_path)
                result.predicted_action = result.turns[-1].predicted_action
            if result.goal_solved and not result.termination_reason:
                # The loop may end immediately after the final allowed turn,
                # before the next top-of-loop termination check runs.
                result.termination_reason = "goal_fulfilled"
            if hasattr(self.user_model, "environment_state"):
                state = self.user_model.environment_state
                result.user_environment_state = (
                    state.to_dict() if hasattr(state, "to_dict") else dict(state)
                )
            if result.goal_solved:
                result.final_status = "goal_solved"
            elif not result.termination_reason:
                result.termination_reason = "max_turns"
                result.final_status = "max_turns_reached"
            elif result.termination_reason == "end_step_reached":
                result.final_status = "workflow_end_without_goal_confirmation"
            elif result.termination_reason == "no_more_user_messages":
                result.final_status = "user_simulator_stopped"
            else:
                result.final_status = "terminated"
            
            if self.verbose:
                print(f"\n[对话完成]")
                print(f"  轮次数: {result.dialogue_length}")
                print(f"  走过的步骤: {result.path_taken}")
                print(f"  执行的动作: {result.actions_taken}")
            
        except Exception as e:
            import traceback
            result.final_status = "failed"
            result.termination_reason = str(e)
            if self.verbose:
                print(f"\n[错误] {e}")
                traceback.print_exc()
        
        finally:
            result.end_time = datetime.now().isoformat()
            result.duration_seconds = time.time() - start_time
        
        return result
    
    def _execute_turn(
        self,
        result: SimulationResult,
        user_message: str,
        context_data: Optional[Dict[str, Any]],
        turn_id: int
    ) -> None:
        """
        执行单轮对话
        
        Args:
            result: 模拟结果对象
            user_message: 用户消息
            context_data: 上下文数据
            turn_id: 轮次ID
        """
        # 用户发送消息
        self.user_model.add_user_message(user_message)
        
        if self.verbose:
            print(f"\n[轮次 {turn_id}]")
            print(f"用户: {user_message}")
        
        backend_state_before = (
            self.backend_environment.get_state_snapshot()
            if self.backend_environment is not None else {}
        )

        # 客服处理
        agent_output = self.agent_model.process_turn(
            user_message=user_message,
            context_data=context_data,
            backend_environment=self.backend_environment,
            max_tool_steps=self.max_tool_steps,
        )
        
        # 检查agent_output是否为None
        if agent_output is None:
            raise RuntimeError(
                f"Agent模型返回None (轮次{turn_id}). "
                f"用户消息: {user_message[:100]}... "
                f"可能是LLM生成或JSON解析失败,请检查日志中的详细错误信息。"
            )
        
        # 结构化动作执行：只有模型明确给出最终动作时，才触发真实后端状态转移。
        if self.legacy_execution and self.backend_environment is not None and agent_output.action:
            action_result = self.backend_environment.execute_action(
                agent_output.action,
                arguments=agent_output.action_parameters,
                turn_index=turn_id,
            )
            agent_output.action_result = action_result.to_dict()
            self.agent_model.context_data.setdefault("backend_action_results", []).append(
                action_result.to_dict()
            )

        # 客服回复
        self.user_model.add_assistant_message(agent_output.chat)

        # 用户模拟器只看到公开的工具/动作结果，不看到完整后台状态。
        new_events = self.backend_environment.get_event_log() if self.backend_environment else []
        if self.backend_environment:
            result.backend_events = new_events
            result.backend_final_state = self.backend_environment.get_state_snapshot()
        if self.backend_environment and hasattr(self.user_model, "observe_backend_event"):
            for event in new_events:
                if event.get("turn_index") == turn_id:
                    self.user_model.observe_backend_event(event)
        
        if self.verbose:
            print(f"客服: {agent_output.chat}")
            print(f"当前步骤: {agent_output.current_step}")
            print(f"动作: {agent_output.action}")
        
        turn_events = [
            event for event in new_events
            if event.get("turn_index") == turn_id
        ]
        action_tool_names = set(
            self.backend_environment.get_action_tool_map()
            if self.backend_environment is not None else {}
        )
        action_calls = [
            call for call in agent_output.tool_calls
            if call.get("name") in action_tool_names
        ]
        action_results = [
            tool_result for tool_result in agent_output.tool_results
            if tool_result.get("tool_name") in action_tool_names
        ]
        predicted_classification = (
            agent_output.classification_output.to_dict()
            if getattr(agent_output, "classification_output", None)
            and hasattr(agent_output.classification_output, "to_dict")
            else (agent_output.classification_output or {})
        )
        predicted_path = list(
            getattr(agent_output, "predicted_path", None)
            or getattr(agent_output, "expected_path", None)
            or []
        )
        predicted_action = (
            getattr(agent_output, "predicted_action", "")
            or (agent_output.final_output.Action if agent_output.final_output else "")
            or getattr(agent_output, "action", "")
        )
        executed_path = self._reconstruct_turn_execution_path(
            agent_output, turn_events, action_tool_names
        )
        executed_action = self._last_successful_action(turn_events)
        agent_output.predicted_path = predicted_path
        agent_output.predicted_action = predicted_action
        agent_output.executed_path = executed_path
        agent_output.executed_action = executed_action

        # 记录轮次。执行字段全部由事件日志重建，不能由模型输出覆盖。
        turn = SimulationTurn(
            turn_id=turn_id,
            user_message=user_message,
            agent_output=agent_output,
            tool_calls=list(agent_output.tool_calls),
            tool_results=list(agent_output.tool_results),
            action_calls=action_calls,
            action_results=action_results,
            backend_state_before=backend_state_before,
            backend_state_after=(
                self.backend_environment.get_state_snapshot()
                if self.backend_environment is not None else {}
            ),
            predicted_classification=predicted_classification,
            predicted_path=predicted_path,
            predicted_action=predicted_action,
            executed_path=executed_path,
            executed_action=executed_action,
        )
        result.turns.append(turn)
        
        # 更新用户满意度 (简单规则)
        if agent_output.action in ["COMFORT", "REFUND", "PLAN"]:
            self.user_model.update_satisfaction(0.1)
        elif agent_output.action in ["GUIDE", "REVIEW"]:
            self.user_model.update_satisfaction(0.05)

    @staticmethod
    def _last_successful_action(events: List[Dict[str, Any]]) -> str:
        for event in reversed(events or []):
            if (
                event.get("event_type") == "action_execution"
                and event.get("result", {}).get("success")
            ):
                return event.get("result", {}).get("action_name", "")
        return ""

    @classmethod
    def _reconstruct_turn_execution_path(
        cls,
        agent_output: AgentTurnOutput,
        events: List[Dict[str, Any]],
        action_tool_names: set,
    ) -> List[str]:
        """Build an execution trace from formal calls and Backend events."""
        trace = []
        action_events = [
            event for event in events
            if event.get("event_type") == "action_execution"
        ]
        action_index = 0
        for call in agent_output.tool_calls:
            name = call.get("name", "")
            if not name:
                continue
            trace.append(f"tool:{name}")
            if name in action_tool_names and action_index < len(action_events):
                action = action_events[action_index]
                action_name = action.get("result", {}).get("action_name") or action.get("name", "")
                trace.append(f"action:{action_name}")
                action_index += 1
        # Include direct legacy executions, if explicitly enabled.
        for action in action_events[action_index:]:
            action_name = action.get("result", {}).get("action_name") or action.get("name", "")
            trace.append(f"action:{action_name}")
        return trace

    @classmethod
    def _reconstruct_execution_path(cls, events: List[Dict[str, Any]]) -> List[str]:
        trace = []
        for event in events or []:
            if event.get("event_type") == "tool_call":
                trace.append(f"tool:{event.get('name', '')}")
            elif event.get("event_type") == "action_execution":
                result = event.get("result", {})
                action_name = result.get("action_name") or event.get("name", "")
                trace.append(f"action:{action_name}")
        return trace
    
    def _should_terminate(self, result: SimulationResult) -> bool:
        """
        判断是否应该终止对话
        
        Args:
            result: 模拟结果
            
        Returns:
            bool: 是否应该终止
        """
        if len(result.turns) >= self.max_turns:
            return True

        if self.backend_environment is not None and self.backend_environment.goal_satisfied():
            return True

        if result.turns:
            last_output = result.turns[-1].agent_output
            if last_output.metadata.get("tool_loop_limit"):
                return True
            if self.backend_environment is not None:
                turn_events = [
                    event for event in self.backend_environment.get_event_log()
                    if event.get("turn_index") == result.turns[-1].turn_id
                ]
                if any(
                    event.get("event_type") in {"tool_call", "action_execution"}
                    and not event.get("result", {}).get("success")
                    for event in turn_events
                ):
                    return True
        
        # 检查最后的步骤是否为END
        if result.turns:
            last_agent_output = result.turns[-1].agent_output
            executed_path = getattr(last_agent_output, "path_taken", []) or []
            action = getattr(last_agent_output, "action", "")
            # 不再单独信任模型根据 predicted_path 填的 next_step；只有实际
            # 执行路径到达 action 节点或明确输出 END 才结束流程。
            reached_action = bool(executed_path and executed_path[-1].startswith("action_"))
            if last_agent_output.next_step is None and (
                reached_action or action in ["END", "end"] or not executed_path
            ):
                return True
        
        # 检测双方都在礼貌性结束对话 - 避免无意义的重复感谢/再见
        if len(result.turns) >= 2:
            last_turn = result.turns[-1]
            prev_turn = result.turns[-2]
            
            last_user_msg = last_turn.user_message.lower()
            last_agent_msg = last_turn.agent_output.chat.lower()
            prev_user_msg = prev_turn.user_message.lower()
            prev_agent_msg = prev_turn.agent_output.chat.lower()
            
            # 定义结束对话的关键词
            farewell_keywords = ['再见', '拜拜', 'bye', 'goodbye', '88']
            thank_keywords = ['谢谢', '感谢', 'thank']
            blessing_keywords = ['祝', '顺利', '愉快', '加油', '进步']
            
            # 检查是否包含这些关键词
            def has_farewell(msg):
                return any(keyword in msg for keyword in farewell_keywords)
            
            def has_thank_or_blessing(msg):
                return (any(keyword in msg for keyword in thank_keywords) or 
                       any(keyword in msg for keyword in blessing_keywords))
            
            # 情况1: 双方都说再见
            user_says_bye = has_farewell(last_user_msg)
            agent_says_bye = has_farewell(last_agent_msg)
            
            if user_says_bye and agent_says_bye:
                if self.verbose:
                    print(f"\n[检测到双方都说再见，提前终止对话]")
                return True
            
            # 情况2: 检测连续2轮都是互相感谢/祝福(没有实质内容)
            # 用户连续两轮都在感谢/祝福
            user_repeating_thanks = (has_thank_or_blessing(last_user_msg) and 
                                    has_thank_or_blessing(prev_user_msg))
            # 客服连续两轮都在感谢/祝福
            agent_repeating_thanks = (has_thank_or_blessing(last_agent_msg) and 
                                     has_thank_or_blessing(prev_agent_msg))
            
            # 判断消息是否主要是感谢/祝福(长度较短且没有实质问题)
            def is_mostly_courtesy(msg):
                # 如果消息很长(>100字符),可能包含实质内容
                if len(msg) > 100:
                    return False
                # 检查是否包含问题关键词
                question_keywords = ['?', '？', '怎么', '如何', '为什么', '什么', '哪', '能否', '可以']
                has_question = any(keyword in msg for keyword in question_keywords)
                if has_question:
                    return False
                # 主要是感谢祝福
                return has_thank_or_blessing(msg)
            
            if (user_repeating_thanks and agent_repeating_thanks and
                is_mostly_courtesy(last_user_msg) and is_mostly_courtesy(last_agent_msg)):
                if self.verbose:
                    print(f"\n[检测到双方连续互相感谢祝福，提前终止对话]")
                return True
        
        # 情况3: 检测用户重复相同问题3次或客服重复相同回答3次
        if len(result.turns) >= 3:
            # 提取最近的消息用于相似度比较
            recent_user_msgs = [turn.user_message for turn in result.turns[-3:]]
            recent_agent_msgs = [turn.agent_output.chat for turn in result.turns[-3:]]
            
            # 辅助函数: 计算两个字符串的相似度(简单的重叠比例)
            def calculate_similarity(msg1: str, msg2: str) -> float:
                """计算两个消息的相似度(0-1)"""
                if not msg1 or not msg2:
                    return 0.0
                
                # 转为小写并去除标点符号进行比较
                import re
                clean1 = re.sub(r'[^\w\s]', '', msg1.lower())
                clean2 = re.sub(r'[^\w\s]', '', msg2.lower())
                
                # 简单方法: 计算字符集合的Jaccard相似度
                set1 = set(clean1.split())
                set2 = set(clean2.split())
                
                if not set1 or not set2:
                    return 0.0
                
                intersection = len(set1 & set2)
                union = len(set1 | set2)
                
                return intersection / union if union > 0 else 0.0
            
            # 检查用户是否重复相同问题3次
            if len(recent_user_msgs) == 3:
                sim_01 = calculate_similarity(recent_user_msgs[0], recent_user_msgs[1])
                sim_12 = calculate_similarity(recent_user_msgs[1], recent_user_msgs[2])
                sim_02 = calculate_similarity(recent_user_msgs[0], recent_user_msgs[2])
                
                # 如果三条消息两两相似度都>0.7,认为是重复问题
                if sim_01 > 0.7 or sim_12 > 0.7 or sim_02 > 0.7:
                    if self.verbose:
                        print(f"\n[检测到用户连续3次重复相同问题，提前终止对话]")
                        print(f"  相似度: {sim_01:.2f}, {sim_12:.2f}, {sim_02:.2f}")
                    return True
            
            # 检查客服是否重复相同回答3次
            if len(recent_agent_msgs) == 3:
                sim_01 = calculate_similarity(recent_agent_msgs[0], recent_agent_msgs[1])
                sim_12 = calculate_similarity(recent_agent_msgs[1], recent_agent_msgs[2])
                sim_02 = calculate_similarity(recent_agent_msgs[0], recent_agent_msgs[2])
                
                # 如果三条消息两两相似度都>0.7,认为是重复回答
                if sim_01 > 0.7 or sim_12 > 0.7 or sim_02 > 0.7:
                    if self.verbose:
                        print(f"\n[检测到客服连续3次重复相同回答，提前终止对话]")
                        print(f"  相似度: {sim_01:.2f}, {sim_12:.2f}, {sim_02:.2f}")
                    return True
        
        return False
    
    def _get_termination_reason(self, result: SimulationResult) -> str:
        """获取终止原因"""
        if self.backend_environment is not None and self.backend_environment.goal_satisfied():
            return "goal_fulfilled"
        
        if result.turns:
            last_turn = result.turns[-1]
            last_agent_output = last_turn.agent_output
            
            last_user_msg = last_turn.user_message.lower()
            last_agent_msg = last_agent_output.chat.lower()

            turn_events = [
                event for event in (self.backend_environment.get_event_log()
                                    if self.backend_environment else [])
                if event.get("turn_index") == last_turn.turn_id
            ]
            if any(
                event.get("event_type") == "action_execution"
                and event.get("result", {}).get("action_name") == "TransHuman"
                and event.get("result", {}).get("success")
                for event in turn_events
            ):
                return "transfer_human"
            if last_agent_output.metadata.get("tool_loop_limit"):
                return "max_tool_steps"
            if any(
                event.get("event_type") in {"tool_call", "action_execution"}
                and not event.get("result", {}).get("success")
                for event in turn_events
            ):
                return "tool_failure"

        # 检查是否因为双方再见而终止
        if result.turns:
            farewell_keywords = ['再见', '拜拜', 'bye', 'goodbye', '88']
            user_says_bye = any(keyword in last_user_msg for keyword in farewell_keywords)
            agent_says_bye = any(keyword in last_agent_msg for keyword in farewell_keywords)
            
            if user_says_bye and agent_says_bye:
                return "user_ended"
            
            # 检查是否因为重复感谢祝福而终止
            if len(result.turns) >= 2:
                prev_turn = result.turns[-2]
                prev_user_msg = prev_turn.user_message.lower()
                prev_agent_msg = prev_turn.agent_output.chat.lower()
                
                thank_keywords = ['谢谢', '感谢', 'thank']
                blessing_keywords = ['祝', '顺利', '愉快', '加油', '进步']
                
                def has_thank_or_blessing(msg):
                    return (any(keyword in msg for keyword in thank_keywords) or 
                           any(keyword in msg for keyword in blessing_keywords))
                
                user_repeating = (has_thank_or_blessing(last_user_msg) and 
                                 has_thank_or_blessing(prev_user_msg))
                agent_repeating = (has_thank_or_blessing(last_agent_msg) and 
                                  has_thank_or_blessing(prev_agent_msg))
                
                if user_repeating and agent_repeating:
                    return "user_ended"
            
            # 检查是否因为重复问题/回答而终止
            if len(result.turns) >= 3:
                import re
                recent_user_msgs = [turn.user_message for turn in result.turns[-3:]]
                recent_agent_msgs = [turn.agent_output.chat for turn in result.turns[-3:]]
                
                def calculate_similarity(msg1: str, msg2: str) -> float:
                    if not msg1 or not msg2:
                        return 0.0
                    clean1 = re.sub(r'[^\w\s]', '', msg1.lower())
                    clean2 = re.sub(r'[^\w\s]', '', msg2.lower())
                    set1 = set(clean1.split())
                    set2 = set(clean2.split())
                    if not set1 or not set2:
                        return 0.0
                    intersection = len(set1 & set2)
                    union = len(set1 | set2)
                    return intersection / union if union > 0 else 0.0
                
                # 检查用户重复问题
                if len(recent_user_msgs) == 3:
                    sim_01 = calculate_similarity(recent_user_msgs[0], recent_user_msgs[1])
                    sim_12 = calculate_similarity(recent_user_msgs[1], recent_user_msgs[2])
                    sim_02 = calculate_similarity(recent_user_msgs[0], recent_user_msgs[2])
                    if sim_01 > 0.7 or sim_12 > 0.7 or sim_02 > 0.7:
                        return "user_repeating_same_question"
                
                # 检查客服重复回答
                if len(recent_agent_msgs) == 3:
                    sim_01 = calculate_similarity(recent_agent_msgs[0], recent_agent_msgs[1])
                    sim_12 = calculate_similarity(recent_agent_msgs[1], recent_agent_msgs[2])
                    sim_02 = calculate_similarity(recent_agent_msgs[0], recent_agent_msgs[2])
                    if sim_01 > 0.7 or sim_12 > 0.7 or sim_02 > 0.7:
                        return "agent_repeating_same_answer"
            
            if last_agent_output.next_step is None:
                return "end_step_reached"

        if len(result.turns) >= self.max_turns:
            return "max_turns"
        
        return "unknown"
    
    def _generate_next_user_message(
        self,
        result: SimulationResult,
        user_message_generator = None
    ) -> Optional[str]:
        """
        生成下一个用户消息
        
        Args:
            result: 模拟结果
            user_message_generator: 消息生成器 (可选)
            
        Returns:
            Optional[str]: 下一个消息，如果无法生成则返回None
        """
        if user_message_generator:
            return user_message_generator(
                user_model=self.user_model,
                agent_last_message=result.turns[-1].agent_output.chat if result.turns else "",
                turn_count=len(result.turns),
                backend_events=result.backend_events[-10:],
            )
        
        # 默认行为：如果客服要求更多信息，用户提供
        if result.turns:
            last_agent_output = result.turns[-1].agent_output
            if last_agent_output.action == "GUIDE":
                return "我是在学第三章第二节的时候，对公式推导不理解。"
        
        # 无法生成更多消息
        return None
