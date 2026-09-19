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
                                        config.persistence.output_dir, config.evaluation.max_turns,
                                        api_timeout=config.evaluation.api_timeout)
        evolution_client = get_llm_client(
            "openai_api", api_key=api_key, base_url=args.api_url, model_name=args.model,
            timeout=config.evaluation.api_timeout,
        )
        customer_evolver = CustomerEvolver(
            config.seed,
            validator=CustomerPolicyValidator(config.customer.allowed_strategy_tags),
            selector=CustomerSelector(config.customer.fitness_weights),
            strategy_generator=LLMCustomerPolicyGenerator(evolution_client),
            require_strategy_generator=True,
        )
        service_evolver = ServiceEvolver(
            config.seed,
            sanitizer=ServicePolicySanitizer(config.service.allowed_rule_categories),
            gate=ServiceGate(config.service.min_delta, config.service.normal_regression_tolerance),
            patch_generator=LLMServicePatchGenerator(evolution_client),
            require_patch_generator=True,
        )
    result = EvolutionRunner(
        config, evaluator=evaluator, customer_evolver=customer_evolver,
        service_evolver=service_evolver,
    ).run()
    print(result["report"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
