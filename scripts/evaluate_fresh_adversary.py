#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from framework.evolution.config import EvolutionConfig, PersistenceConfig
from framework.evolution.customer_evolver import CustomerEvolver, LLMCustomerPolicyGenerator
from framework.evolution.customer_policy import CustomerPolicyValidator
from framework.evolution.customer_selector import CustomerSelector
from framework.evolution.evaluator_adapter import MockEpisodeEvaluator
from framework.evolution.runner import EvolutionRunner


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--evaluator", choices=["mock", "real"])
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--real", action="store_true", help="Alias for --evaluator real")
    parser.add_argument("--target", choices=["final", "initial", "all"], default="final")
    parser.add_argument("--service-only-run-dir")
    parser.add_argument("--model", default=os.environ.get("EVOSAGE_MODEL", ""))
    parser.add_argument("--api-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    args = parser.parse_args()
    saved_config = Path(args.run_dir) / "config" / "evolution.json"
    if saved_config.exists():
        config = EvolutionConfig.from_dict(json.loads(saved_config.read_text(encoding="utf-8")))
        config.persistence = PersistenceConfig(output_dir=args.run_dir, resume=True)
    else:
        config = EvolutionConfig(persistence=PersistenceConfig(output_dir=args.run_dir))
    if args.real and args.mock:
        raise SystemExit("choose exactly one evaluator: --mock or --real")
    evaluator_mode = "real" if args.real else args.evaluator
    if args.mock:
        evaluator_mode = "mock"
    if evaluator_mode is None:
        raise SystemExit("choose --evaluator mock or --evaluator real")
    evaluator = MockEpisodeEvaluator()
    customer_evolver = None
    if evaluator_mode == "real":
        if not args.model or not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("--real requires --model/EVOSAGE_MODEL and OPENAI_API_KEY")
        from framework.evolution.real_factory import make_real_evaluator
        from framework.llm_integration import get_llm_client
        evaluator = make_real_evaluator(
            args.model, args.api_url, os.environ["OPENAI_API_KEY"], args.run_dir,
            max_turns=config.evaluation.max_turns, api_timeout=config.evaluation.api_timeout,
        )
        evolution_client = get_llm_client(
            "openai_api", api_key=os.environ["OPENAI_API_KEY"], base_url=args.api_url, model_name=args.model,
            timeout=config.evaluation.api_timeout,
        )
        customer_evolver = CustomerEvolver(
            seed=config.seed + 10_000,
            validator=CustomerPolicyValidator(config.customer.allowed_strategy_tags),
            selector=CustomerSelector(config.customer.fitness_weights),
            strategy_generator=LLMCustomerPolicyGenerator(evolution_client),
            require_strategy_generator=True,
        )
    runner = EvolutionRunner(config, evaluator=evaluator, customer_evolver=customer_evolver)
    targets = [args.target] if args.target != "all" else ["initial", "final_coevolved"]
    if args.service_only_run_dir:
        targets.insert(-1 if "final_coevolved" in targets else len(targets), "final_service_only")
    total = 0
    for target in targets:
        service = None
        if target == "initial":
            from framework.evolution.schemas import ServicePolicy
            service = ServicePolicy.from_dict(json.loads((Path(args.run_dir) / "environment/initial_service_policy.json").read_text(encoding="utf-8")))
        elif target == "final_service_only":
            from framework.evolution.persistence import RunStore
            from framework.evolution.schemas import ServicePolicy
            store = RunStore(args.service_only_run_dir)
            generation = max(store.completed_generations())
            service = ServicePolicy.from_dict(store.read_generation(generation, "service_policy"))
        results = runner.fresh_adversary_evaluation(
            rounds=config.fresh_adversary.rounds,
            candidate_count=config.fresh_adversary.candidate_count,
            target_service=service,
            target_label=target,
        )
        total += len(results)
    print(f"wrote {total} fresh-adversary episodes under {Path(args.run_dir) / 'analysis'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
