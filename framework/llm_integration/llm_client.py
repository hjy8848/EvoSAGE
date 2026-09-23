#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM客户端集成
LLM Client Integration

支持两种方式调用LLM：
1. vLLM本地部署
2. API接口调用 (OpenAI兼容或其他API)
"""

import json
import requests
import time
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging

from ..backend.types import ToolCall

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """LLM响应"""
    text: str                           # 生成的文本
    model: str                          # 使用的模型
    tokens: int = 0                     # token数
    metadata: Dict[str, Any] = None     # 额外元数据
    tool_calls: list = None             # 标准化工具调用列表
    raw_response: Any = None            # provider原始响应

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.tool_calls is None:
            self.tool_calls = []

    @property
    def content(self) -> str:
        """OpenAI-compatible alias used by callers that prefer content."""
        return self.text


def _parse_tool_calls(message: Dict[str, Any]) -> list:
    """Normalize OpenAI-compatible tool calls."""
    parsed = []
    for index, item in enumerate(message.get("tool_calls") or []):
        function = item.get("function", {}) if isinstance(item, dict) else {}
        arguments = function.get("arguments", {})
        arguments_valid = True
        argument_error = None
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                arguments = {}
                arguments_valid = False
                argument_error = str(exc)
        if not isinstance(arguments, dict):
            arguments_valid = False
            argument_error = argument_error or "tool arguments must be a JSON object"
            arguments = {}
        parsed.append(
            ToolCall(
                call_id=(item.get("id") if isinstance(item, dict) else None)
                or f"call_{index}",
                name=function.get("name", ""),
                arguments=arguments,
                arguments_valid=arguments_valid,
                argument_error=argument_error,
            )
        )
    return parsed


def _is_timeout_exception(exc: BaseException) -> bool:
    """Recognize timeout wrappers produced by different LiteLLM transports."""
    return (
        isinstance(exc, (TimeoutError, requests.exceptions.Timeout))
        or "timeout" in exc.__class__.__name__.lower()
        or "timed out" in str(exc).lower()
    )


class LLMClient(ABC):
    """LLM客户端抽象基类"""

    def _init_request_stats(self) -> None:
        self.request_count = 0
        self.retry_count = 0
        self.attempts = 0
        self.successes = 0
        self.failures = 0
        self.timeouts = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.latency_seconds = 0.0
        self.total_attempt_latency = 0.0
        self.max_attempt_latency = 0.0

    def _record_request(self, attempt: int) -> None:
        self.request_count = getattr(self, "request_count", 0) + 1
        self.attempts = getattr(self, "attempts", 0) + 1
        if attempt:
            self.retry_count = getattr(self, "retry_count", 0) + 1

    def _record_attempt_timing(self, latency_seconds: float, success: bool,
                               timeout: bool = False) -> None:
        elapsed = max(0.0, float(latency_seconds or 0.0))
        self.total_attempt_latency = getattr(self, "total_attempt_latency", 0.0) + elapsed
        self.max_attempt_latency = max(getattr(self, "max_attempt_latency", 0.0), elapsed)
        if success:
            self.successes = getattr(self, "successes", 0) + 1
        else:
            self.failures = getattr(self, "failures", 0) + 1
            if timeout:
                self.timeouts = getattr(self, "timeouts", 0) + 1

    def _record_failed_attempt(self, started_at: float, timeout: bool = False) -> None:
        self._record_attempt_timing(time.perf_counter() - started_at, success=False, timeout=timeout)

    def _record_completed(self, input_tokens: int = 0, output_tokens: int = 0,
                          latency_seconds: float = 0.0) -> None:
        self.input_tokens = getattr(self, "input_tokens", 0) + int(input_tokens or 0)
        self.output_tokens = getattr(self, "output_tokens", 0) + int(output_tokens or 0)
        self.latency_seconds = getattr(self, "latency_seconds", 0.0) + float(latency_seconds or 0.0)
        self._record_attempt_timing(latency_seconds, success=True)

    def request_stats(self) -> Dict[str, Any]:
        return {
            "requests": int(getattr(self, "request_count", 0) or 0),
            "retries": int(getattr(self, "retry_count", 0) or 0),
            "attempts": int(getattr(self, "attempts", 0) or 0),
            "successes": int(getattr(self, "successes", 0) or 0),
            "failures": int(getattr(self, "failures", 0) or 0),
            "timeouts": int(getattr(self, "timeouts", 0) or 0),
            "input_tokens": int(getattr(self, "input_tokens", 0) or 0),
            "output_tokens": int(getattr(self, "output_tokens", 0) or 0),
            "latency_seconds": float(getattr(self, "latency_seconds", 0.0) or 0.0),
            "total_attempt_latency": float(getattr(self, "total_attempt_latency", 0.0) or 0.0),
            "max_attempt_latency": float(getattr(self, "max_attempt_latency", 0.0) or 0.0),
        }
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """生成文本"""
        pass


class VLLMLocalClient(LLMClient):
    """
    vLLM本地客户端
    
    使用方式：
    1. 启动vLLM服务：
       python -m vllm.entrypoints.openai.api_server \
           --model /path/to/model \
           --port 8000 \
           --tensor-parallel-size 4
    
    2. 创建客户端：
       client = VLLMLocalClient(
           base_url="http://localhost:8000",
           model_name="your-model"
       )
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model_name: str = "default",
        api_key: str = "EMPTY",
        timeout: int = 300,
        max_retries: int = 3,
    ):
        """
        初始化vLLM本地客户端
        
        Args:
            base_url: vLLM服务器URL
            model_name: 模型名称
            api_key: API密钥 (vLLM通常为EMPTY)
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
        """
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self._init_request_stats()
    
    def _filter_think_tags(self, text: str) -> str:
        """
        过滤 Qwen3 think 模式的 <think> 标签内容
        仅保留实际输出，移除思考过程
        
        Args:
            text: 原始文本
            
        Returns:
            str: 过滤后的文本
        """
        import re
        
        # 移除 <think>...</think> 标签及其内容
        # 同时移除标签前后可能的空白字符，避免留下多余的换行
        filtered_text = re.sub(r'\s*<think[^>]*>.*?</think>\s*', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # 移除其他常见的思考标签（如果有）
        filtered_text = re.sub(r'\s*<user_think[^>]*>.*?</user_think>\s*', '', filtered_text, flags=re.DOTALL | re.IGNORECASE)
        
        # 清理开头和结尾的空白字符
        filtered_text = filtered_text.strip()
        
        return filtered_text
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 0.95,
        **kwargs
    ) -> LLMResponse:
        """
        生成文本
        
        Args:
            prompt: 输入提示词
            temperature: 采样温度
            max_tokens: 最大生成token数
            top_p: nucleus采样参数
            **kwargs: 其他参数
            
        Returns:
            LLMResponse: 生成的响应
        """
        url = f"{self.base_url}/v1/completions"
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "stop": kwargs.get("stop", None),
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        # 重试逻辑
        for attempt in range(self.max_retries):
            try:
                self._record_request(attempt)
                request_started = time.perf_counter()
                response = requests.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                
                data = response.json()
                
                if "choices" in data and len(data["choices"]) > 0:
                    text = data["choices"][0]["text"]
                    tokens = data.get("usage", {}).get("completion_tokens", 0)
                    
                    # 过滤 Qwen3 think 模式的 <think> 标签内容（仅保留实际输出）
                    text = self._filter_think_tags(text)
                    usage = data.get("usage", {}) or {}
                    latency_seconds = time.perf_counter() - request_started
                    self._record_completed(
                        usage.get("prompt_tokens", 0),
                        usage.get("completion_tokens", 0),
                        latency_seconds,
                    )
                    
                    return LLMResponse(
                        text=text,
                        model=self.model_name,
                        tokens=tokens,
                        metadata={
                            "finish_reason": data["choices"][0].get("finish_reason"),
                            "attempts": attempt + 1,
                        }
                    )
                else:
                    raise ValueError("No choices in response")
            
            except requests.exceptions.Timeout:
                self._record_failed_attempt(request_started, timeout=True)
                logger.warning(f"Timeout on attempt {attempt + 1}/{self.max_retries}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
            
            except requests.exceptions.ConnectionError:
                self._record_failed_attempt(request_started)
                logger.warning(f"Connection error on attempt {attempt + 1}/{self.max_retries}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
            
            except Exception as e:
                self._record_failed_attempt(request_started)
                logger.error(f"Error on attempt {attempt + 1}/{self.max_retries}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
        
        raise RuntimeError(f"Failed to generate after {self.max_retries} attempts")


class VLLMChatClient(LLMClient):
    """vLLM聊天格式客户端"""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model_name: str = "default",
        api_key: str = "EMPTY",
        timeout: int = 300,
        max_retries: int = 3,
    ):
        """初始化vLLM聊天客户端"""
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self._init_request_stats()
    
    def _filter_think_tags(self, text: str) -> str:
        """
        过滤 Qwen3 think 模式的 <think> 标签内容
        仅保留实际输出，移除思考过程
        
        Args:
            text: 原始文本
            
        Returns:
            str: 过滤后的文本
        """
        import re
        
        # 移除 <think>...</think> 标签及其内容
        # 同时移除标签前后可能的空白字符，避免留下多余的换行
        filtered_text = re.sub(r'\s*<think[^>]*>.*?</think>\s*', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # 移除其他常见的思考标签（如果有）
        filtered_text = re.sub(r'\s*<user_think[^>]*>.*?</user_think>\s*', '', filtered_text, flags=re.DOTALL | re.IGNORECASE)
        
        # 清理开头和结尾的空白字符
        filtered_text = filtered_text.strip()
        
        return filtered_text
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 0.95,
        messages: Optional[list] = None,
        **kwargs
    ) -> LLMResponse:
        """
        生成文本 (聊天格式)
        
        Args:
            prompt: 输入提示词 (如果messages为None则使用)
            temperature: 采样温度
            max_tokens: 最大生成token数
            top_p: nucleus采样参数
            messages: 消息列表 [{role, content}, ...]
            **kwargs: 其他参数
            
        Returns:
            LLMResponse: 生成的响应
        """
        url = f"{self.base_url}/v1/chat/completions"
        
        if messages is None:
            messages = [{"role": "user", "content": prompt}]
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
        }
        # Let the provider choose its own output budget when callers pass
        # None.  This is important for reasoning models whose visible JSON
        # content can be starved by a small local completion cap.
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if kwargs.get("tools") is not None:
            payload["tools"] = kwargs["tools"]
        if kwargs.get("tool_choice") is not None:
            payload["tool_choice"] = kwargs["tool_choice"]
        # Qwen3模型:thinking模式应该在vLLM服务器启动时通过--chat-template-kwargs参数禁用
        # API调用时通过extra_body传递该参数无效,vLLM不支持运行时动态修改chat_template_kwargs
        # 参考: run.sh中的start_server_single函数会在启动时自动为Qwen3添加--chat-template-kwargs参数
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        for attempt in range(self.max_retries):
            try:
                self._record_request(attempt)
                request_started = time.perf_counter()
                # 调试信息：第一次尝试时打印请求详情
                # if attempt == 0:
                #     logger.debug(f"vLLM请求: URL={url}, Model={self.model_name}, Message数={len(messages)}")
                
                response = requests.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                
                data = response.json()
                
                if "choices" in data and len(data["choices"]) > 0:
                    message = data["choices"][0]["message"]
                    text = message.get("content") or ""
                    tokens = data.get("usage", {}).get("completion_tokens", 0)
                    tool_calls = _parse_tool_calls(message)
                    
                    # 过滤 Qwen3 think 模式的 <think> 标签内容（仅保留实际输出）
                    text = self._filter_think_tags(text)
                    usage = data.get("usage", {}) or {}
                    latency_seconds = time.perf_counter() - request_started
                    self._record_completed(
                        usage.get("prompt_tokens", 0),
                        usage.get("completion_tokens", 0),
                        latency_seconds,
                    )
                    
                    return LLMResponse(
                        text=text,
                        model=self.model_name,
                        tokens=tokens,
                        metadata={
                            "finish_reason": data["choices"][0].get("finish_reason"),
                            "assistant_message": message,
                            "attempts": attempt + 1,
                            "usage": usage,
                            "latency_seconds": latency_seconds,
                        },
                        tool_calls=tool_calls,
                        raw_response=data,
                    )
                else:
                    raise ValueError("No choices in response")
            
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                self._record_failed_attempt(
                    request_started,
                    timeout=isinstance(e, requests.exceptions.Timeout),
                )
                logger.warning(f"Error on attempt {attempt + 1}/{self.max_retries}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
            
            except requests.exceptions.HTTPError as e:
                self._record_failed_attempt(request_started)
                if e.response.status_code == 429:
                    logger.warning(f"Rate limit exceeded (429) on attempt {attempt + 1}/{self.max_retries}. Waiting 120 seconds...")
                    if attempt < self.max_retries - 1:
                        time.sleep(120)
                        continue
                    else:
                        raise
                
                logger.error(f"HTTP error on attempt {attempt + 1}/{self.max_retries}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
            
            except Exception as e:
                self._record_failed_attempt(request_started)
                logger.error(f"Error on attempt {attempt + 1}/{self.max_retries}: URL={url}, Model={self.model_name}, Error={e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
        
        raise RuntimeError(f"Failed to generate after {self.max_retries} attempts")


class OpenAIAPIClient(LLMClient):
    """
    OpenAI兼容API客户端
    
    支持：
    - OpenAI官方API
    - 任何OpenAI兼容的API (如LM Studio, Ollama等)
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model_name: str = "gpt-3.5-turbo",
        timeout: int = 300,
        max_retries: int = 3,
    ):
        """
        初始化OpenAI兼容API客户端
        
        Args:
            api_key: API密钥
            base_url: API基础URL
            model_name: 模型名称
            timeout: 请求超时时间
            max_retries: 最大重试次数
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries
        self._init_request_stats()
    
    def _filter_think_tags(self, text: str) -> str:
        """
        过滤 Qwen3 think 模式的 <think> 标签内容
        仅保留实际输出，移除思考过程
        
        Args:
            text: 原始文本
            
        Returns:
            str: 过滤后的文本
        """
        import re
        
        # 移除 <think>...</think> 标签及其内容
        # 同时移除标签前后可能的空白字符，避免留下多余的换行
        filtered_text = re.sub(r'\s*<think[^>]*>.*?</think>\s*', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # 移除其他常见的思考标签（如果有）
        filtered_text = re.sub(r'\s*<user_think[^>]*>.*?</user_think>\s*', '', filtered_text, flags=re.DOTALL | re.IGNORECASE)
        
        # 清理开头和结尾的空白字符
        filtered_text = filtered_text.strip()
        
        return filtered_text
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.95,
        messages: Optional[list] = None,
        **kwargs
    ) -> LLMResponse:
        """生成文本 (聊天格式)"""
        url = f"{self.base_url}/chat/completions"
        
        if messages is None:
            messages = [{"role": "user", "content": prompt}]
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }
        if kwargs.get("tools") is not None:
            payload["tools"] = kwargs["tools"]
        if kwargs.get("tool_choice") is not None:
            payload["tool_choice"] = kwargs["tool_choice"]
        # Provider-specific optional controls are forwarded only when a role
        # explicitly opts in. In particular, DeepSeek's top-level `thinking`
        # field is absent from historical/default requests.
        if kwargs.get("thinking") is not None:
            payload["thinking"] = kwargs["thinking"]
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        for attempt in range(self.max_retries):
            try:
                self._record_request(attempt)
                request_started = time.perf_counter()
                response = requests.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                
                data = response.json()
                
                if "choices" in data and len(data["choices"]) > 0:
                    message = data["choices"][0]["message"]
                    text = message.get("content") or ""
                    tokens = data.get("usage", {}).get("completion_tokens", 0)
                    tool_calls = _parse_tool_calls(message)
                    
                    # 过滤 Qwen3 think 模式的 <think> 标签内容（仅保留实际输出）
                    text = self._filter_think_tags(text)
                    usage = data.get("usage", {}) or {}
                    latency_seconds = time.perf_counter() - request_started
                    self._record_completed(
                        usage.get("prompt_tokens", 0),
                        usage.get("completion_tokens", 0),
                        latency_seconds,
                    )
                    
                    return LLMResponse(
                        text=text,
                        model=self.model_name,
                        tokens=tokens,
                        metadata={
                            "finish_reason": data["choices"][0].get("finish_reason"),
                            "assistant_message": message,
                            "attempts": attempt + 1,
                            "usage": usage,
                            "latency_seconds": latency_seconds,
                        },
                        tool_calls=tool_calls,
                        raw_response=data,
                    )
                else:
                    raise ValueError("No choices in response")
            
            except requests.exceptions.Timeout:
                self._record_failed_attempt(request_started, timeout=True)
                logger.warning(f"Timeout on attempt {attempt + 1}/{self.max_retries}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
            
            except requests.exceptions.HTTPError as e:
                self._record_failed_attempt(request_started)
                if e.response.status_code == 429:
                    logger.warning(f"Rate limit exceeded (429) on attempt {attempt + 1}/{self.max_retries}. Waiting 120 seconds...")
                    if attempt < self.max_retries - 1:
                        time.sleep(120)
                        continue
                    else:
                        raise
                
                logger.error(f"HTTP error on attempt {attempt + 1}/{self.max_retries}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
            
            except Exception as e:
                self._record_failed_attempt(request_started)
                logger.error(f"Error on attempt {attempt + 1}/{self.max_retries}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
        
        raise RuntimeError(f"Failed to generate after {self.max_retries} attempts")


class LiteLLMClient(LLMClient):
    """LiteLLM-backed OpenAI-compatible client.

    LiteLLM normalizes provider responses while preserving the OpenAI chat
    schema.  InferAI models are addressed through LiteLLM's ``openai/``
    provider with the supplied ``api_base``.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model_name: str = "gpt-3.5-turbo",
        timeout: int = 300,
        max_retries: int = 3,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries
        self._init_request_stats()

    @staticmethod
    def _as_dict(value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            return value
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if hasattr(value, "to_dict"):
            return value.to_dict()
        return {}

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 0.95,
        messages: Optional[list] = None,
        **kwargs
    ) -> LLMResponse:
        try:
            import litellm
        except ImportError as exc:
            raise RuntimeError("LiteLLM client requires the 'litellm' package") from exc

        if messages is None:
            messages = [{"role": "user", "content": prompt}]
        model = self.model_name if "/" in self.model_name else f"openai/{self.model_name}"
        payload = {
            "model": model,
            "api_base": self.base_url,
            "api_key": self.api_key,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "timeout": self.timeout,
            "num_retries": 0,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if kwargs.get("tools") is not None:
            payload["tools"] = kwargs["tools"]
        if kwargs.get("tool_choice") is not None:
            payload["tool_choice"] = kwargs["tool_choice"]

        litellm.suppress_debug_info = True
        for attempt in range(self.max_retries):
            try:
                self._record_request(attempt)
                request_started = time.perf_counter()
                response = litellm.completion(**payload)
                raw_response = self._as_dict(response)
                choices = raw_response.get("choices") or []
                if not choices:
                    raise ValueError("No choices in LiteLLM response")
                choice = self._as_dict(choices[0])
                assistant_message = self._as_dict(choice.get("message"))
                text = assistant_message.get("content") or ""
                tool_calls = _parse_tool_calls(assistant_message)
                usage = self._as_dict(raw_response.get("usage"))
                latency_seconds = time.perf_counter() - request_started
                self._record_completed(
                    usage.get("prompt_tokens", 0),
                    usage.get("completion_tokens", 0),
                    latency_seconds,
                )
                return LLMResponse(
                    text=text.strip() if isinstance(text, str) else str(text),
                    model=self.model_name,
                    tokens=int(usage.get("completion_tokens") or 0),
                    metadata={
                        "finish_reason": choice.get("finish_reason"),
                        "assistant_message": assistant_message,
                        "attempts": attempt + 1,
                        "usage": usage,
                        "latency_seconds": latency_seconds,
                    },
                    tool_calls=tool_calls,
                    raw_response=raw_response,
                )
            except Exception as exc:
                # LiteLLM normalizes provider-specific timeout exceptions into
                # different classes depending on the transport.  Preserve a
                # reliable timeout counter even when the gateway does not
                # expose requests.exceptions.Timeout directly.
                self._record_failed_attempt(request_started, timeout=_is_timeout_exception(exc))
                if attempt >= self.max_retries - 1:
                    raise
                logger.warning(
                    "LiteLLM request failed on attempt %s/%s: %s",
                    attempt + 1, self.max_retries, exc,
                )
                time.sleep(2 ** attempt)
        raise RuntimeError(f"LiteLLM failed after {self.max_retries} attempts")


def get_llm_client(
    client_type: str,
    **config
) -> LLMClient:
    """
    工厂函数 - 根据类型创建LLM客户端
    
    Args:
        client_type: 客户端类型
            - "vllm_local": vLLM本地 (completions格式)
            - "vllm_chat": vLLM本地 (chat格式)
            - "openai_api": OpenAI兼容API
            - "litellm": LiteLLM via an OpenAI-compatible API
            - "openai": OpenAI官方API
        **config: 客户端配置参数
        
    Returns:
        LLMClient: LLM客户端实例
    """
    if client_type == "vllm_local":
        return VLLMLocalClient(**config)
    elif client_type == "vllm_chat":
        return VLLMChatClient(**config)
    elif client_type in ["openai_api", "openai"]:
        return OpenAIAPIClient(**config)
    elif client_type == "litellm":
        return LiteLLMClient(**config)
    else:
        raise ValueError(f"Unknown client type: {client_type}")


# 示例配置
EXAMPLE_CONFIG = {
    # vLLM本地配置示例
    "vllm_local": {
        "base_url": "http://localhost:8000",
        "model_name": "Qwen/Qwen2.5-14B-Instruct",
        "timeout": 300,
    },
    # OpenAI API配置示例
    "openai_api": {
        "api_key": "sk-xxx",
        "base_url": "https://api.openai.com/v1",
        "model_name": "gpt-3.5-turbo",
        "timeout": 300,
    },
}
