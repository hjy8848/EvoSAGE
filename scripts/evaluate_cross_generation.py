#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from framework.evolution.config import EvolutionConfig, PersistenceConfig
from framework.evolution.evaluator_adapter import MockEpisodeEvaluator
from framework.evolution.runner import EvolutionRunner


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--evaluator", choices=["mock", "real"])
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--real", action="store_true", help="Alias for --evaluator real")
    parser.add_argument("--model", default=os.environ.get("EVOSAGE_MODEL", ""))
    parser.add_argument("--api-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    args = parser.parse_args()
    config = EvolutionConfig(persistence=PersistenceConfig(output_dir=args.run_dir))
    if args.real and args.mock:
        raise SystemExit("choose exactly one evaluator: --mock or --real")
    evaluator_mode = "real" if args.real else args.evaluator
    if args.mock:
        evaluator_mode = "mock"
    if evaluator_mode is None:
        raise SystemExit("choose --evaluator mock or --evaluator real")
    evaluator = MockEpisodeEvaluator()
    if evaluator_mode == "real":
        if not args.model or not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("--real requires --model/EVOSAGE_MODEL and OPENAI_API_KEY")
        from framework.evolution.real_factory import make_real_evaluator
        evaluator = make_real_evaluator(args.model, args.api_url, os.environ["OPENAI_API_KEY"], args.run_dir, api_timeout=config.evaluation.api_timeout)
    matrix = EvolutionRunner(config, evaluator=evaluator).cross_generation_evaluation()
    print(f"wrote {len(matrix)} matrix cells to {Path(args.run_dir) / 'analysis/cross_generation_matrix.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
