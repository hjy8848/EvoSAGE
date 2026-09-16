#!/usr/bin/env python3
"""Reproducible Ecommerce Refund pilot runner.

This runner deliberately reuses the frozen pipeline.  It does not alter the
Backend, Agent, User Simulator, or Evaluator; it only fixes the PathList case
matrix and writes pilot-specific reports.
"""

import argparse
import copy
import csv
import json
import os
import subprocess
import sys
import time
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from framework.sop.ecommerce_refund_PathList import (  # noqa: E402
    generate_path_list,
    get_intent_path_mapping,
)
from run_evaluation_with_llm import LLMEvaluationPipeline  # noqa: E402


LEVELS = ("zero_conflict", "weak_conflict", "strong_conflict")
LEAKAGE_KEYS = (
    "expected_path",
    "expected_action",
    "expected_outcome",
    "backend_record",
    "classification_dict",
    "system_info",
)


class RecordingClient:
    """Record provider requests without recording API keys or headers."""

    def __init__(self, client, role: str, max_output_tokens: Optional[int] = None):
        self.client = client
        self.role = role
        self.max_output_tokens = max_output_tokens
        self.requests: List[Dict[str, Any]] = []

    def generate(self, prompt: str, **kwargs):
        requested_tokens = kwargs.get("max_tokens") or self.max_output_tokens
        if self.max_output_tokens and requested_tokens > self.max_output_tokens:
            kwargs["max_tokens"] = self.max_output_tokens
        request = {
            "request_id": uuid.uuid4().hex[:12],
            "role": self.role,
            "messages": copy.deepcopy(kwargs.get("messages", [])),
            "tools": copy.deepcopy(kwargs.get("tools", [])),
            "max_tokens": kwargs.get("max_tokens"),
        }
        started = time.perf_counter()
        try:
            response = self.client.generate(prompt, **kwargs)
            request.update({
                "status": "success",
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "attempts": (getattr(response, "metadata", {}) or {}).get("attempts", 1),
            })
            self.requests.append(request)
            return response
        except Exception as exc:
            request.update({
                "status": "error",
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "error_type": type(exc).__name__,
                "error": str(exc)[:500],
            })
            self.requests.append(request)
            raise

    def __getattr__(self, name):
        return getattr(self.client, name)


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _csv_write(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _resolve_api_key(args) -> str:
    if args.api_key:
        return args.api_key
    env_key = os.environ.get(args.api_key_env, "")
    if env_key:
        return env_key
    if args.keychain_service:
        command = ["security", "find-generic-password"]
        if args.keychain_account:
            command.extend(["-a", args.keychain_account])
        command.extend(["-s", args.keychain_service, "-w"])
        try:
            return subprocess.check_output(command, stderr=subprocess.DEVNULL, text=True).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    raise SystemExit(
        f"未找到 API Key。请设置 {args.api_key_env}，或传入 --api-key / --keychain-service。"
    )


def _path_tasks(repetitions: int, path_ids: Optional[List[int]] = None) -> List[Tuple[int, str, Dict[str, Any], int]]:
    path_list = generate_path_list()
    selected_path_ids = path_ids or list(range(1, len(path_list) + 1))
    invalid = sorted(set(selected_path_ids) - set(range(1, len(path_list) + 1)))
    if invalid:
        raise RuntimeError(f"PathList 编号无效: {invalid}")
    mapping = get_intent_path_mapping()
    path_to_intent = {}
    for intent, config in mapping.items():
        for path_id in config.get("possible_paths", []):
            path_to_intent.setdefault(path_id, intent)
    missing = sorted(set(range(1, len(path_list) + 1)) - set(path_to_intent))
    if missing:
        raise RuntimeError(f"PathList 没有 intent 映射: {missing}")
    tasks = []
    for repetition in range(1, repetitions + 1):
        for path_id in selected_path_ids:
            path_config = path_list[path_id - 1]
            config = copy.deepcopy(path_config)
            config["pilot_path_id"] = path_id
            tasks.append((path_id, path_to_intent[path_id], config, repetition))
    return tasks


def _case_record(
    path_id: int,
    intent: str,
    repetition: int,
    simulation,
    report,
    request_log: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    case = simulation.case_spec or {}
    metadata = case.get("metadata", {})
    path_config = metadata.get("path_config", {})
    return {
        "case_id": case.get("case_id"),
        "path_id": path_id,
        "intent": intent,
        "repetition": repetition,
        "adversarial_level": simulation.adversarial_intensity,
        "model": simulation.model_name,
        "case_spec_summary": {
            "user_goal": case.get("user_goal", {}),
            "user_policy": case.get("user_policy", {}),
            "initial_backend_state": (case.get("backend_record", {}).get("order", {})),
            "expected_outcome": case.get("expected_outcome", {}),
        },
        "user_messages": [turn.user_message for turn in simulation.turns],
        "agent_messages": [turn.agent_output.chat for turn in simulation.turns],
        "tool_calls": [
            {"name": event.get("name"), "arguments": event.get("arguments", {})}
            for event in simulation.backend_events
            if event.get("event_type") == "tool_call"
        ],
        "tool_results": [
            event.get("result", {}) for event in simulation.backend_events
            if event.get("event_type") == "tool_call"
        ],
        "action_calls": [
            {"name": event.get("name"), "arguments": event.get("arguments", {})}
            for event in simulation.backend_events
            if event.get("event_type") == "action_execution"
        ],
        "action_results": [
            event.get("result", {}) for event in simulation.backend_events
            if event.get("event_type") == "action_execution"
        ],
        "backend_event_log": simulation.backend_events,
        "final_backend_state": simulation.backend_final_state,
        "classification_prediction": (
            simulation.turns[-1].agent_output.classification_output.to_dict()
            if simulation.turns and simulation.turns[-1].agent_output.classification_output
            else {}
        ),
        "predicted_policy_path": report.predicted_path,
        "canonical_expected_path": report.gold_path,
        "predicted_action": report.predicted_action,
        "expected_action": path_config.get("final_output", {}).get("Action", ""),
        "executed_action": report.executed_action,
        "executed_trace": report.executed_trace,
        "required_backend_verifications": report.required_backend_verifications,
        "completed_backend_verifications": [
            item for item in report.required_backend_verifications if item.get("verified")
        ],
        "termination_reason": simulation.termination_reason,
        "api_request_log": request_log or [],
        "decision_metrics": {
            "classification_accuracy": report.classification_accuracy,
            "canonical_path_correctness": report.canonical_path_correctness,
            "predicted_action_correctness": report.predicted_action_correctness,
            "logic_score": report.logic_score,
            "sage_style_score": report.sage_style_score,
            "chat_quality": report.chat_quality,
        },
        "execution_metrics": {
            "required_verification": report.required_verification_score,
            "policy_compliance": report.policy_compliance_score,
            "action_execution": report.action_execution_score,
            "goal_fulfillment": report.goal_fulfillment,
            "execution_score": report.execution_score,
        },
        "task_success": report.task_success,
        "error_categories": report.error_categories,
        "simulation": simulation.to_dict(),
        "evaluation": report.to_dict(),
    }


def _leakage_check(record: Dict[str, Any], requests: List[Dict[str, Any]]) -> List[str]:
    case = record.get("simulation", {}).get("case_spec") or {}
    path_config = case.get("metadata", {}).get("path_config", {})
    hidden_values = [
        _safe_json(path_config.get("system_variables", {})),
        _safe_json(case.get("expected_outcome", {})),
    ]
    problems = []
    agent_requests = [request for request in requests if request.get("role") == "agent"]
    for request_index, request in enumerate(agent_requests):
        if request.get("role") != "agent":
            continue
        text = _safe_json(request.get("messages", []))
        for key in LEAKAGE_KEYS:
            if key in text:
                problems.append(f"agent_prompt_contains:{key}")
        if request_index == 0:
            for value in hidden_values:
                if value != "{}" and value in text:
                    problems.append("agent_prompt_contains_hidden_case_value")
    return sorted(set(problems))


def _simple_trace(record: Dict[str, Any]) -> str:
    lines = [f"Case {record['case_id']} | Path {record['path_id']} | {record['adversarial_level']}"]
    for user, agent in zip(record["user_messages"], record["agent_messages"]):
        lines.append(f"User → {user}")
        lines.append(f"Agent → {agent}")
    for event in record["backend_event_log"]:
        if event.get("event_type") == "tool_call":
            lines.append(f"Tool → {event.get('name')} {event.get('arguments', {})}")
            lines.append(f"Tool Result → {event.get('result', {})}")
        elif event.get("event_type") == "action_execution":
            lines.append(f"Action → {event.get('name')} {event.get('arguments', {})}")
            lines.append(f"Backend transition → {event.get('state_before')} => {event.get('state_after')}")
    lines.append(
        "Evaluation → " + _safe_json({
            "sage_style_score": record["decision_metrics"]["sage_style_score"],
            "execution_score": record["execution_metrics"]["execution_score"],
            "task_success": record["task_success"],
            "errors": record["error_categories"],
        })
    )
    return "\n".join(lines) + "\n"


def _aggregate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    metric_names = [
        "classification_accuracy", "canonical_path_correctness",
        "predicted_action_correctness", "logic_score", "sage_style_score",
        "chat_quality", "required_verification", "policy_compliance",
        "action_execution", "goal_fulfillment", "execution_score",
    ]

    def values(items, metric):
        for record in items:
            source = record["decision_metrics"] if metric in record["decision_metrics"] else record["execution_metrics"]
            yield float(source.get(metric, 0.0))

    def one(items):
        result = {metric: _mean(values(items, metric)) for metric in metric_names}
        result["runs"] = len(items)
        result["task_success_rate"] = _mean([1.0 if item["task_success"] else 0.0 for item in items])
        return result

    return {"all": one(records), **{level: one([r for r in records if r["adversarial_level"] == level]) for level in LEVELS}}


def _write_reports(run_dir: Path, records: List[Dict[str, Any]], config: Dict[str, Any], sanity: Dict[str, Any]) -> None:
    summary = _aggregate(records)
    level_counts = ", ".join(f"{level}={summary[level]['runs']}" for level in LEVELS)
    path_rows = []
    for path_id in sorted({r["path_id"] for r in records}):
        items = [r for r in records if r["path_id"] == path_id]
        first = items[0]
        errors = Counter(error for item in items for error in item["error_categories"])
        path_rows.append({
            "path_id": path_id,
            "intent": first["intent"],
            "expected_path": _safe_json(first["canonical_expected_path"]),
            "expected_action": first["expected_action"],
            "required_verifications": _safe_json(first["required_backend_verifications"]),
            "runs": len(items),
            "task_success_rate": _mean([float(item["task_success"]) for item in items]),
            "execution_score": _mean([item["execution_metrics"]["execution_score"] for item in items]),
            "sage_style_score": _mean([item["decision_metrics"]["sage_style_score"] for item in items]),
            "top_error": errors.most_common(1)[0][0] if errors else "",
        })
    adversarial_rows = []
    for level in (*LEVELS, "all"):
        row = {"adversarial_level": level, **summary[level]}
        adversarial_rows.append(row)
    errors = Counter(error for record in records for error in record["error_categories"])
    error_rows = [{"error_category": name, "count": count, "rate": count / len(records) if records else 0.0}
                  for name, count in sorted(errors.items())]
    metric_rows = [{"scope": "all", "metric": metric, "value": value}
                   for metric, value in summary["all"].items() if metric != "runs"]

    _csv_write(run_dir / "path_breakdown.csv", path_rows, list(path_rows[0]) if path_rows else ["path_id"])
    _csv_write(run_dir / "adversarial_breakdown.csv", adversarial_rows, list(adversarial_rows[0]))
    _csv_write(run_dir / "error_breakdown.csv", error_rows, ["error_category", "count", "rate"])
    _csv_write(run_dir / "summary.csv", metric_rows, ["scope", "metric", "value"])
    (run_dir / "summary.json").write_text(json.dumps({"config": config, "summary": summary, "sanity_checks": sanity}, ensure_ascii=False, indent=2), encoding="utf-8")

    gap_threshold = 0.70
    high_gap = [r for r in records if r["decision_metrics"]["sage_style_score"] >= gap_threshold and not r["task_success"]]
    report = [
        "# Ecommerce Refund Pilot Report",
        "",
        f"- Git commit: `{config['git_commit']}`",
        f"- Run ID: `{config['run_id']}`",
        f"- Model: `{config['model']}`",
        f"- Cases: `{len(records)}`",
        f"- Path coverage: `{len({r['path_id'] for r in records})}/{config['target_path_count']}`",
        f"- Adversarial levels: {level_counts}",
        "",
        "## Core metrics",
        "",
        f"- Task Success Rate: `{summary['all']['task_success_rate']:.4f}`",
        f"- Execution Score: `{summary['all']['execution_score']:.4f}`",
        f"- SAGE-style Score: `{summary['all']['sage_style_score']:.4f}`",
        f"- Logic Score: `{summary['all']['logic_score']:.4f}`",
        f"- Chat Quality: `{summary['all']['chat_quality']:.4f}` (0-1)",
        f"- Verification / Policy / Action / Goal: `{summary['all']['required_verification']:.4f}` / `{summary['all']['policy_compliance']:.4f}` / `{summary['all']['action_execution']:.4f}` / `{summary['all']['goal_fulfillment']:.4f}`",
        "",
        "## Sanity checks",
        "",
        f"- GT leakage: `{sanity['gt_leakage']}`",
        f"- Termination errors: `{sanity['termination_errors']}`",
        f"- Verification errors: `{sanity['verification_errors']}`",
        f"- Action execution errors: `{sanity['action_errors']}`",
        f"- Decision-execution gap threshold: SAGE-style >= `{gap_threshold:.2f}` and Task Success = 0",
        f"- High-decision failed-execution cases: `{len(high_gap)}` / `{len(records)}`",
        "",
        "## Findings",
        "",
        f"- Most common error: `{errors.most_common(1)[0][0] if errors else 'none'}`",
        f"- User claim overtrusted cases: `{sum('user_claim_overtrusted' in r['error_categories'] for r in records)}`",
        f"- Claimed action not executed cases: `{sum('claimed_action_not_executed' in r['error_categories'] for r in records)}`",
        "- Scoring / Backend / termination bug: see `sanity_checks` and individual raw cases.",
        "",
        "Detailed path and error tables are in `path_breakdown.csv`, `adversarial_breakdown.csv`, and `error_breakdown.csv`.",
    ]
    (run_dir / "pilot_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def run(args) -> int:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6]
    run_dir = Path(args.output) / run_id
    raw_dir = run_dir / "raw"
    traces_dir = run_dir / "traces"
    raw_dir.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)
    api_key = _resolve_api_key(args)
    if args.proxy:
        os.environ["HTTP_PROXY"] = args.proxy
        os.environ["HTTPS_PROXY"] = args.proxy

    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    pipeline = LLMEvaluationPipeline(
        scenario_id="ecommerce_refund",
        model_name=args.model,
        output_dir=str(raw_dir),
        eval_mode="api",
        api_key=api_key,
        api_url=args.api_url,
        user_model_name=args.model,
        agent_model_type="api",
        agent_model_name=args.model,
        judge_model_name=args.model,
        max_turns=args.max_turns,
        verbose=args.verbose,
        user_simulator_mode=args.user_simulator_mode,
        user_policy_mode=args.user_policy_mode,
    )
    recorders = []
    for role in ("user", "agent", "judge"):
        attribute = {"user": "user_llm_client", "agent": "agent_llm_client", "judge": "judge_llm_client"}[role]
        client = getattr(pipeline, attribute)
        # A pilot must fail one slow provider request and preserve the case
        # error record; the library default of 300s x 3 retries is too long
        # for a controlled smoke stage.
        if hasattr(client, "timeout"):
            client.timeout = args.request_timeout
        if hasattr(client, "max_retries"):
            client.max_retries = 1
        recorder = RecordingClient(client, role, args.max_output_tokens)
        setattr(pipeline, attribute, recorder)
        recorders.append(recorder)

    default_smoke_paths = [1, 6, 8]
    path_ids = args.path_ids or (default_smoke_paths if args.stage == "smoke" else None)
    tasks = _path_tasks(args.repetitions, path_ids)
    records = []
    sanity = {"gt_leakage": [], "termination_errors": [], "verification_errors": [], "action_errors": [], "case_errors": []}
    for path_id, intent, path_config, repetition in tasks:
        user_id = f"pilot-{run_id}-path-{path_id}-rep-{repetition}"
        request_offsets = {id(recorder): len(recorder.requests) for recorder in recorders}
        try:
            simulation, evaluation = pipeline.run_single_simulation(intent, user_id=user_id, path_config=path_config)
            request_log = [
                request
                for recorder in recorders
                for request in recorder.requests[request_offsets[id(recorder)]:]
            ]
            record = _case_record(path_id, intent, repetition, simulation, evaluation, request_log)
            leakage = _leakage_check(record, request_log)
            if leakage:
                sanity["gt_leakage"].extend(leakage)
                raise RuntimeError("CRITICAL GT leakage: " + ", ".join(leakage))
            if simulation.termination_reason == "goal_fulfilled" and not simulation.goal_solved:
                sanity["termination_errors"].append(record["case_id"])
            for item in evaluation.required_backend_verifications:
                if item.get("verified") is False:
                    sanity["verification_errors"].append(record["case_id"])
            if evaluation.predicted_action and evaluation.predicted_action != evaluation.executed_action and evaluation.task_success:
                sanity["action_errors"].append(record["case_id"])
            records.append(record)
            (raw_dir / f"path_{path_id:02d}_rep_{repetition:02d}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            (traces_dir / f"path_{path_id:02d}_rep_{repetition:02d}.txt").write_text(_simple_trace(record), encoding="utf-8")
            print(f"[{len(records)}/{len(tasks)}] path={path_id} level={simulation.adversarial_intensity} task_success={evaluation.task_success}")
        except Exception as exc:
            sanity["case_errors"].append({"path_id": path_id, "intent": intent, "error": str(exc)})
            (raw_dir / f"path_{path_id:02d}_rep_{repetition:02d}_error.json").write_text(json.dumps({"path_id": path_id, "intent": intent, "error": str(exc)}, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"path={path_id} FAILED: {exc}", file=sys.stderr)
            if "CRITICAL GT leakage" in str(exc):
                break

    target_path_ids = path_ids or list(range(1, 16))
    config = {
        "run_id": run_id,
        "git_commit": git_commit,
        "model": args.model,
        "provider": args.api_url,
        "api_url": args.api_url,
        "temperature": "pipeline defaults",
        "top_p": "provider default",
        "max_tokens": args.max_output_tokens,
        "tool_calling_mode": "OpenAI-compatible tools",
        "seed": None,
        "repetitions": args.repetitions,
        "max_turns": args.max_turns,
        "user_simulator_mode": args.user_simulator_mode,
        "user_policy_mode": args.user_policy_mode,
        "stage": args.stage,
        "target_path_count": len(target_path_ids),
    }
    coverage = {str(path_id): sum(1 for record in records if record["path_id"] == path_id) for path_id in target_path_ids}
    sanity["covered_paths"] = sum(count > 0 for count in coverage.values())
    sanity["coverage"] = coverage
    _write_reports(run_dir, records, config, sanity)
    (run_dir / "coverage_report.json").write_text(json.dumps({"covered_paths": sanity["covered_paths"], "total_paths": len(target_path_ids), "paths": coverage}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Pilot output: {run_dir}")
    print(f"Coverage: {sanity['covered_paths']}/{len(target_path_ids)}")
    if sanity["covered_paths"] != len(target_path_ids):
        return 2
    if sanity["gt_leakage"] or sanity["termination_errors"]:
        return 3
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a reproducible Ecommerce Refund pilot")
    parser.add_argument("--stage", choices=["smoke", "pilot"], default="pilot")
    parser.add_argument("--path-ids", type=int, nargs="+", help="仅运行指定 PathList 编号；Smoke 默认运行 1、6、8")
    parser.add_argument("--api-url", default="http://10.130.138.46:8010/v1")
    parser.add_argument("--model", default="dashscope/qwen3.7-plus")
    parser.add_argument("--api-key")
    parser.add_argument("--api-key-env", default="EVOSAGE_API_KEY")
    parser.add_argument("--keychain-service")
    parser.add_argument("--keychain-account")
    parser.add_argument("--proxy", default="http://127.0.0.1:65533")
    parser.add_argument("--output", default="results/ecommerce_pilot")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--request-timeout", type=int, default=90, help="单次 API 请求超时秒数")
    parser.add_argument("--max-output-tokens", type=int, default=2048, help="Pilot 对每次生成设置的最大 token 上限")
    parser.add_argument("--user-simulator-mode", choices=["llm", "rule", "rewrite"], default="llm")
    parser.add_argument("--user-policy-mode", choices=["truthful", "mistaken", "withholding", "adversarial_false_claim"], default="truthful")
    parser.add_argument("--verbose", action="store_true")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
