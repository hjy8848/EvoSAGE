#!/usr/bin/env python3
"""Run open-ended Customer strategy search against fixed Service S0."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from framework.evolution.config import load_customer_search_config
from framework.evolution.customer.evolver import CustomerEvolver, LLMAdversaryStrategyGenerator
from framework.evolution.customer.integrity import AdversaryPolicyValidator
from framework.evolution.customer.runner import CustomerEvolutionRunner
from framework.evolution.evaluator_adapter import MockEpisodeEvaluator
from framework.evolution.persistence import resolve_run_dir
from framework.evolution.request_budget import APIRequestBudget


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/customer_search_smoke.yaml"),
        help="Customer-search JSON/YAML config",
    )
    parser.add_argument("--evaluator", choices=["mock", "real"], required=True)
    parser.add_argument("--model", default=os.environ.get("EVOSAGE_MODEL", ""))
    parser.add_argument(
        "--api-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    parser.add_argument("--client", choices=["openai_api", "litellm"], default="openai_api")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = load_customer_search_config(args.config)
    if args.resume:
        config.persistence.resume = True
    if args.evaluator == "real":
        if not args.model:
            raise SystemExit("--evaluator real requires --model or EVOSAGE_MODEL")
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("--evaluator real requires OPENAI_API_KEY")
        config.model_metadata.update({
            "model": args.model,
            "api_url": args.api_url,
            "client": args.client,
            "provider": "InferAI" if "inferaiapi.com" in args.api_url.lower() else None,
        })

    run_dir, fresh_run_isolated = resolve_run_dir(config)
    evaluation = config.evaluation
    request_budget = APIRequestBudget(
        max_per_generation=evaluation.max_api_requests_per_generation,
        max_per_run=evaluation.max_api_requests_per_run,
        persist_path=run_dir / "analysis" / "request_budget.json",
        resume=config.persistence.resume,
    )
    evaluator = MockEpisodeEvaluator()
    customer_evolver = None
    if args.evaluator == "real":
        from framework.evolution.real_factory import make_real_evaluator
        from framework.llm_integration import get_llm_client

        evaluator = make_real_evaluator(
            args.model,
            args.api_url,
            api_key,
            run_dir,
            scenario_id=config.scenario,
            max_turns=evaluation.max_turns,
            api_timeout=evaluation.api_timeout,
            client_type=args.client,
            judge_validation_enabled=evaluation.judge_validation_enabled,
            resume=config.persistence.resume,
            token_budget=evaluation.token_budget,
            customer_thinking_mode=evaluation.customer_thinking_mode,
            agent_thinking_mode=evaluation.agent_thinking_mode,
            tool_contract_config=evaluation.tool_contract,
            max_tool_steps=evaluation.max_tool_steps,
            invalid_evaluation_retries=evaluation.invalid_evaluation_retries,
            request_budget=request_budget,
            customer_transport_max_retries=evaluation.customer_transport_max_retries,
            agent_max_retries=evaluation.agent_max_retries,
            judge_max_retries=evaluation.judge_max_retries,
            judge_validation_retries=evaluation.judge_validation_retries,
            customer_protocol_retries=evaluation.customer_protocol_retries,
            rate_limit_backoff_seconds=evaluation.rate_limit_backoff_seconds,
        )
        generation_client = get_llm_client(
            args.client,
            api_key=api_key,
            base_url=args.api_url,
            model_name=args.model,
            timeout=evaluation.api_timeout,
            max_retries=evaluation.customer_evolver_max_retries,
            request_budget=request_budget,
            request_role="evolver",
            rate_limit_backoff_seconds=evaluation.rate_limit_backoff_seconds,
        )
        customer_evolver = CustomerEvolver(
            seed=config.seed,
            validator=AdversaryPolicyValidator(),
            strategy_generator=LLMAdversaryStrategyGenerator(
                generation_client,
                max_tokens=evaluation.token_budget.customer_evolver,
                thinking_mode=evaluation.customer_evolver_thinking_mode,
                protocol_retries=evaluation.customer_evolver_protocol_retries,
            ),
            require_strategy_generator=True,
        )

    runner = CustomerEvolutionRunner(
        config,
        evaluator=evaluator,
        customer_evolver=customer_evolver,
        run_dir=run_dir,
    )
    result = runner.run(fresh_run_isolated=fresh_run_isolated)
    print(result["report"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
