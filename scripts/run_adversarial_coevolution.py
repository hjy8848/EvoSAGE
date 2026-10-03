#!/usr/bin/env python3
"""Run EvoSAGE adversarial co-evolution (mock by default, real explicitly)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from framework.evolution.config import load_config
from framework.evolution.customer_evolver import CustomerEvolver, LLMCustomerPolicyGenerator
from framework.evolution.customer_policy import CustomerPolicyValidator
from framework.evolution.evaluator_adapter import EvoSAGEEpisodeEvaluator, MockEpisodeEvaluator
from framework.evolution.runner import EvolutionRunner
from framework.evolution.service_evolver import LLMServicePatchGenerator, ServiceEvolver
from framework.evolution.service_gate import ServiceGate
from framework.evolution.service_policy import ServicePolicySanitizer
from framework.evolution.customer_selector import CustomerSelector
from framework.evolution.request_budget import APIRequestBudget


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/ecommerce_coevolution.yaml"))
    parser.add_argument("--mode", choices=["static", "customer_only", "service_only", "coevolution"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluator", choices=["mock", "real"])
    parser.add_argument("--mock", action="store_true", help="Explicitly use the deterministic offline evaluator")
    parser.add_argument("--real", action="store_true", help="Alias for --evaluator real")
    parser.add_argument("--model", default=os.environ.get("EVOSAGE_MODEL", ""))
    parser.add_argument("--api-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--client", choices=["openai_api", "litellm"], default="openai_api",
                        help="LLM client implementation for real API runs")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.mode:
        config.experiment_mode = args.mode
    if args.resume:
        config.persistence.resume = True
    if args.real and args.mock:
        raise SystemExit("choose exactly one evaluator: --mock or --real")
    evaluator_mode = "real" if args.real else args.evaluator
    if args.mock:
        evaluator_mode = "mock"
    if evaluator_mode is None:
        raise SystemExit("choose --evaluator mock or --evaluator real")
    if evaluator_mode == "real":
        config.model_metadata.update({
            "model": args.model,
            "api_url": args.api_url,
            "client": args.client,
        })
        if "inferaiapi.com" in args.api_url.lower():
            config.model_metadata["provider"] = "InferAI"
    # Resolve the run directory before constructing the evaluator.  The real
    # evaluator creates its persistent episode cache in its constructor; if
    # the runner resolved a fresh directory afterwards, artifacts would be
    # split between two different runs.
    run_dir, _ = EvolutionRunner.resolve_run_dir(config)
    request_budget = APIRequestBudget(
        max_per_generation=config.evaluation.max_api_requests_per_generation,
        max_per_run=config.evaluation.max_api_requests_per_run,
        persist_path=run_dir / "analysis" / "request_budget.json",
        resume=config.persistence.resume,
    )
    evaluator = MockEpisodeEvaluator()
    customer_evolver = None
    service_evolver = None
    if evaluator_mode == "real":
        if not args.model:
            raise SystemExit("--real requires --model or EVOSAGE_MODEL")
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("--real requires OPENAI_API_KEY; load it from Keychain in the calling shell")
        from framework.llm_integration import get_llm_client
        from framework.evolution.real_factory import make_real_evaluator
        evaluator = make_real_evaluator(args.model, args.api_url, api_key,
                                        run_dir, config.evaluation.max_turns,
                                        api_timeout=config.evaluation.api_timeout,
                                        client_type=args.client,
                                        judge_in_evolution=config.evaluation.judge_in_evolution,
                                        resume=config.persistence.resume,
                                        token_budget=config.evaluation.token_budget,
                                        customer_thinking_mode=config.evaluation.customer_thinking_mode,
                                        agent_thinking_mode=config.evaluation.agent_thinking_mode,
                                        evolver_thinking_mode=config.evaluation.evolver_thinking_mode,
                                        tool_contract_config=config.evaluation.tool_contract,
                                        max_tool_steps=config.evaluation.max_tool_steps,
                                        invalid_evaluation_retries=config.evaluation.invalid_evaluation_retries,
                                        request_budget=request_budget,
                                        customer_transport_max_retries=config.evaluation.customer_transport_max_retries,
                                        agent_max_retries=config.evaluation.agent_max_retries,
                                        judge_max_retries=config.evaluation.judge_max_retries,
                                        judge_validation_retries=config.evaluation.judge_validation_retries,
                                        customer_protocol_retries=config.evaluation.customer_protocol_retries,
                                        rate_limit_backoff_seconds=config.evaluation.rate_limit_backoff_seconds)
        evolution_client = get_llm_client(
            args.client, api_key=api_key, base_url=args.api_url, model_name=args.model,
            timeout=config.evaluation.api_timeout,
            max_retries=config.evaluation.evolver_max_retries,
            request_budget=request_budget,
            request_role="evolver",
            rate_limit_backoff_seconds=config.evaluation.rate_limit_backoff_seconds,
        )
        if config.experiment_mode in {"customer_only", "coevolution"}:
            customer_evolver = CustomerEvolver(
                config.seed,
                validator=CustomerPolicyValidator(),
                selector=CustomerSelector(),
                strategy_generator=LLMCustomerPolicyGenerator(
                    evolution_client,
                    max_tokens=config.evaluation.token_budget.customer_evolver,
                    thinking_mode=config.evaluation.evolver_thinking_mode,
                    protocol_retries=config.evaluation.evolver_protocol_retries,
                ),
                require_strategy_generator=True,
            )
        if config.experiment_mode in {"service_only", "coevolution"}:
            service_evolver = ServiceEvolver(
                config.seed,
                sanitizer=ServicePolicySanitizer(config.service.allowed_rule_categories),
                gate=ServiceGate(
                    config.service.min_delta,
                    config.service.normal_regression_tolerance,
                    config.service.gate_min_paired_wins,
                    config.service.max_normal_paired_losses,
                ),
                patch_generator=LLMServicePatchGenerator(
                    evolution_client,
                    max_tokens=config.evaluation.token_budget.service_evolver,
                    summary_limit=config.evaluation.summary_limit,
                    thinking_mode=config.evaluation.evolver_thinking_mode,
                    protocol_retries=config.evaluation.evolver_protocol_retries,
                ),
                require_patch_generator=True,
            )
    result = EvolutionRunner(
        config, evaluator=evaluator, customer_evolver=customer_evolver,
        service_evolver=service_evolver, run_dir=run_dir,
    ).run()
    print(result["report"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
