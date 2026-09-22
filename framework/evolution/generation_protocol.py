"""Shared diagnostics and bounded retry handling for structured evolution output."""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, Optional, Tuple


PROTOCOL_RETRY_LIMIT = 1


class GenerationProtocolError(RuntimeError):
    """A structured-generation attempt remained invalid after one retry."""

    def __init__(self, reason: str, record: Dict[str, Any]):
        self.reason = reason
        self.record = record
        super().__init__(f"{record.get('role', 'evolver')}_generation_invalid:{reason}")


def _choice_message(raw_response: Any) -> Dict[str, Any]:
    if not isinstance(raw_response, dict):
        return {}
    choices = raw_response.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return {}
    message = choices[0].get("message") or {}
    return message if isinstance(message, dict) else {}


def _choice_finish_reason(raw_response: Any) -> Optional[str]:
    if not isinstance(raw_response, dict):
        return None
    choices = raw_response.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return None
    return choices[0].get("finish_reason")


def _usage(response: Any) -> Dict[str, Any]:
    metadata = getattr(response, "metadata", {}) or {}
    raw_response = getattr(response, "raw_response", None) or {}
    usage = metadata.get("usage") or raw_response.get("usage") or {}
    return usage if isinstance(usage, dict) else {}


def _reasoning_tokens(usage: Dict[str, Any]) -> Optional[int]:
    details = usage.get("completion_tokens_details") or {}
    value = details.get("reasoning_tokens")
    return int(value) if value is not None else None


def _attempt_record(
    role: str,
    attempt_index: int,
    max_tokens: int,
    response: Any = None,
    error: BaseException = None,
) -> Dict[str, Any]:
    raw_response = getattr(response, "raw_response", None) if response is not None else None
    metadata = (getattr(response, "metadata", {}) or {}) if response is not None else {}
    usage = _usage(response) if response is not None else {}
    message = _choice_message(raw_response)
    content = getattr(response, "text", "") if response is not None else ""
    content = content if isinstance(content, str) else str(content or "")
    reasoning_content = message.get("reasoning_content") or message.get("reasoning") or ""
    finish_reason = metadata.get("finish_reason") or _choice_finish_reason(raw_response)
    request_id = (
        (raw_response or {}).get("id")
        if isinstance(raw_response, dict)
        else None
    ) or metadata.get("request_id")
    record = {
        "attempt_index": attempt_index,
        "max_tokens": max_tokens,
        "finish_reason": finish_reason,
        "input_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": _reasoning_tokens(usage),
        "raw_content": content,
        "reasoning_content": reasoning_content,
        "raw_provider_response": raw_response,
        "parse_status": "NOT_RUN",
        "parse_error": None,
        "provider_error": None,
        "timeout": False,
        "latency_seconds": metadata.get("latency_seconds"),
        "request_id": request_id,
    }
    if error is not None:
        record["provider_error"] = f"{type(error).__name__}: {error}"
        record["timeout"] = "timeout" in type(error).__name__.lower() or "timed out" in str(error).lower()
    return record


def _clean_json_text(text: str) -> str:
    return re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.I | re.M).strip()


def request_json_with_retry(
    *,
    client: Any,
    prompt: str,
    role: str,
    max_tokens: int,
    temperature: float,
    retry_limit: int = PROTOCOL_RETRY_LIMIT,
) -> Tuple[Any, Dict[str, Any]]:
    """Request JSON with at most ``retry_limit`` protocol retries.

    The returned record contains only provider response data and diagnostics;
    authentication headers and API keys are never part of it.
    """
    record: Dict[str, Any] = {
        "role": role,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "retry_limit": retry_limit,
        "retry_count": 0,
        "status": "pending",
        "reason": None,
        "attempts": [],
    }
    for attempt_index in range(retry_limit + 1):
        request_started = time.perf_counter()
        try:
            response = client.generate(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            attempt = _attempt_record(role, attempt_index, max_tokens, error=exc)
            attempt["latency_seconds"] = time.perf_counter() - request_started
            record["attempts"].append(attempt)
            reason = "timeout" if attempt["timeout"] else "provider_error"
            if attempt_index < retry_limit:
                record["retry_count"] = attempt_index + 1
                continue
            record["status"] = "inconclusive"
            record["reason"] = f"{role}_generation_invalid:{reason}"
            raise GenerationProtocolError(reason, record) from exc

        attempt = _attempt_record(role, attempt_index, max_tokens, response=response)
        record["attempts"].append(attempt)
        content = attempt["raw_content"]
        finish_reason = attempt["finish_reason"]
        if finish_reason in {"length", "max_tokens"}:
            reason = "output_truncated"
        elif not content:
            reason = "empty_content"
        else:
            try:
                parsed = json.loads(_clean_json_text(content))
                if not isinstance(parsed, (dict, list)):
                    raise ValueError("structured output must be a JSON object or array")
                attempt["parse_status"] = "PASS"
                attempt["parsed_json"] = parsed
                record["status"] = "response_valid"
                record["reason"] = None
                record["retry_count"] = attempt_index
                return parsed, record
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                reason = "malformed_json"
                attempt["parse_status"] = "FAIL"
                attempt["parse_error"] = f"{type(exc).__name__}: {exc}"
        attempt["parse_status"] = "FAIL"
        attempt["parse_error"] = attempt["parse_error"] or reason
        if attempt_index < retry_limit:
            record["retry_count"] = attempt_index + 1
            continue
        record["status"] = "inconclusive"
        record["reason"] = f"{role}_generation_invalid:{reason}"
        raise GenerationProtocolError(reason, record)

    raise AssertionError("unreachable structured-generation retry state")
