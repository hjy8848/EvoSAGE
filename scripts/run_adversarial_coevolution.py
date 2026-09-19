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
from framework.evolution.evaluator_adapter import EvoSAGEEpisodeEvaluator, MockEpisodeEvaluator
from framework.evolution.runner import EvolutionRunner


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/ecommerce_coevolution.yaml"))
    parser.add_argument("--mode", choices=["static", "customer_only", "service_only", "coevolution"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--real", action="store_true", help="Use the OpenAI-compatible API; default is offline mock")
    parser.add_argument("--model", default=os.environ.get("EVOSAGE_MODEL", ""))
    parser.add_argument("--api-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    args = parser.parse_args()
    config = load_config(args.config)
    if args.mode:
        config.experiment_mode = args.mode
    if args.resume:
        config.persistence.resume = True
    evaluator = MockEpisodeEvaluator()
    if args.real:
        if not args.model:
            raise SystemExit("--real requires --model or EVOSAGE_MODEL")
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("--real requires OPENAI_API_KEY; load it from Keychain in the calling shell")
        from run_evaluation_with_llm import LLMEvaluationPipeline
        from framework.evolution.evaluator_adapter import EvoSAGEEpisodeEvaluator
        def pipeline_factory():
            return LLMEvaluationPipeline(
                scenario_id=config.scenario,
                model_name=args.model,
                output_dir=str(Path(config.persistence.output_dir) / "real_traces"),
                eval_mode="api",
                api_key=api_key,
                api_url=args.api_url,
                user_model_name=args.model,
                agent_model_type="api",
                agent_model_name=args.model,
                judge_model_name=args.model,
                max_turns=config.evaluation.max_turns,
                verbose=False,
            )
        evaluator = EvoSAGEEpisodeEvaluator(pipeline_factory)
    result = EvolutionRunner(config, evaluator=evaluator).run()
    print(result["report"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
