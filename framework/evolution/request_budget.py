"""Shared pre-request API budget and phase-level accounting.

The budget counts provider attempts, not benchmark episodes or client calls.
It is intentionally independent of scoring and evaluation semantics.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import json
from pathlib import Path
import threading
from typing import Any, Optional


_REQUEST_CONTEXT: ContextVar[tuple[Optional[int], str, Optional[str]]] = ContextVar(
    "evosage_request_context", default=(None, "unattributed", None)
)


@contextmanager
def request_context(generation: Optional[int] = None, phase: Optional[str] = None,
                    role: Optional[str] = None):
    old_generation, old_phase, old_role = _REQUEST_CONTEXT.get()
    token = _REQUEST_CONTEXT.set((
        old_generation if generation is None else generation,
        old_phase if phase is None else str(phase or "unattributed"),
        old_role if role is None else role,
    ))
    try:
        yield
    finally:
        _REQUEST_CONTEXT.reset(token)


class APIRequestBudgetExceeded(RuntimeError):
    """Raised before an HTTP request when a configured hard cap is exhausted."""

    inconclusive = True
    budget_exhausted = True

    def __init__(self, scope: str, limit: int, used: int):
        self.scope = scope
        self.limit = int(limit)
        self.used = int(used)
        super().__init__("api_request_budget_exceeded")


class APIRequestBudget:
    def __init__(
        self,
        max_per_generation: Optional[int] = None,
        max_per_run: Optional[int] = None,
        persist_path: str | Path | None = None,
        resume: bool = False,
    ):
        self.max_per_generation = max_per_generation
        self.max_per_run = max_per_run
        self.persist_path = Path(persist_path) if persist_path else None
        self._lock = threading.RLock()
        self._data: dict[str, Any] = {
            "schema_version": 1,
            "limits": {
                "per_generation": max_per_generation,
                "per_run": max_per_run,
            },
            "total_provider_attempts": 0,
            "blocked_attempts": 0,
            "generations": {},
        }
        if resume and self.persist_path and self.persist_path.exists():
            try:
                loaded = json.loads(self.persist_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict) and loaded.get("schema_version") == 1:
                    self._data = loaded
                    self._data["limits"] = {
                        "per_generation": max_per_generation,
                        "per_run": max_per_run,
                    }
            except (OSError, ValueError, TypeError):
                # A damaged diagnostics file must not hide or mutate experiment
                # semantics. The hard cap begins afresh if it cannot be read.
                pass

    @staticmethod
    def _context(generation=None, phase=None):
        current_generation, current_phase, current_role = _REQUEST_CONTEXT.get()
        return (
            current_generation if generation is None else generation,
            str(current_phase if phase is None else phase or "unattributed"),
            current_role,
        )

    def _phase_bucket(self, generation, phase):
        gen_key = "unknown" if generation is None else str(generation)
        gen = self._data["generations"].setdefault(gen_key, {"provider_attempts": 0, "phases": {}})
        phase_bucket = gen["phases"].setdefault(phase, {
            "episodes": 0,
            "roles": {},
            "logical_requests": 0,
            "provider_attempts": 0,
            "retries": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_seconds": 0.0,
            "timeouts": 0,
            "provider_failures": 0,
        })
        return gen, phase_bucket

    def before_attempt(self, role: str, attempt_index: int = 0) -> None:
        generation, phase, role_override = self._context()
        role = role_override or role
        with self._lock:
            gen, phase_bucket = self._phase_bucket(generation, phase)
            used_run = int(self._data.get("total_provider_attempts", 0))
            used_generation = int(gen.get("provider_attempts", 0))
            if self.max_per_run is not None and used_run >= self.max_per_run:
                self._data["blocked_attempts"] = int(self._data.get("blocked_attempts", 0)) + 1
                self._persist_locked()
                raise APIRequestBudgetExceeded("run", self.max_per_run, used_run)
            if self.max_per_generation is not None and used_generation >= self.max_per_generation:
                self._data["blocked_attempts"] = int(self._data.get("blocked_attempts", 0)) + 1
                self._persist_locked()
                raise APIRequestBudgetExceeded("generation", self.max_per_generation, used_generation)
            self._data["total_provider_attempts"] = used_run + 1
            gen["provider_attempts"] = used_generation + 1
            phase_bucket["provider_attempts"] += 1
            if attempt_index == 0:
                phase_bucket["logical_requests"] += 1
            else:
                phase_bucket["retries"] += 1
            role_bucket = phase_bucket["roles"].setdefault(role, {
                "logical_requests": 0, "provider_attempts": 0, "retries": 0,
                "input_tokens": 0, "output_tokens": 0, "latency_seconds": 0.0,
                "timeouts": 0, "provider_failures": 0,
            })
            role_bucket["provider_attempts"] += 1
            if attempt_index == 0:
                role_bucket["logical_requests"] += 1
            else:
                role_bucket["retries"] += 1
            self._persist_locked()

    def record_result(self, role: str, *, input_tokens=0, output_tokens=0,
                      latency_seconds=0.0, success=True, timeout=False) -> None:
        generation, phase, role_override = self._context()
        role = role_override or role
        with self._lock:
            _, phase_bucket = self._phase_bucket(generation, phase)
            role_bucket = phase_bucket["roles"].setdefault(role, {
                "logical_requests": 0, "provider_attempts": 0, "retries": 0,
                "input_tokens": 0, "output_tokens": 0, "latency_seconds": 0.0,
                "timeouts": 0, "provider_failures": 0,
            })
            phase_bucket["input_tokens"] += int(input_tokens or 0)
            phase_bucket["output_tokens"] += int(output_tokens or 0)
            phase_bucket["latency_seconds"] += float(latency_seconds or 0.0)
            role_bucket["input_tokens"] += int(input_tokens or 0)
            role_bucket["output_tokens"] += int(output_tokens or 0)
            role_bucket["latency_seconds"] += float(latency_seconds or 0.0)
            if not success:
                phase_bucket["provider_failures"] += 1
                role_bucket["provider_failures"] += 1
            if timeout:
                phase_bucket["timeouts"] += 1
                role_bucket["timeouts"] += 1
            self._persist_locked()

    def record_episodes(self, count: int, generation=None, phase=None) -> None:
        generation, phase, _ = self._context(generation, phase)
        with self._lock:
            _, bucket = self._phase_bucket(generation, phase)
            bucket["episodes"] += max(0, int(count))
            self._persist_locked()

    def _persist_locked(self) -> None:
        if not self.persist_path:
            return
        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            self.persist_path.write_text(
                json.dumps(self._data, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except OSError:
            # The in-memory hard cap remains active if observability storage is
            # unavailable; runner-level checkpoint writing will report it.
            return

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            result = json.loads(json.dumps(self._data))
            result["limits"] = {
                "per_generation": self.max_per_generation,
                "per_run": self.max_per_run,
            }
            return result

    def phase_summary(self, generation: int, phase: str) -> dict[str, Any]:
        snapshot = self.snapshot()
        gen = snapshot.get("generations", {}).get(str(generation), {})
        return dict(gen.get("phases", {}).get(phase, {}))
