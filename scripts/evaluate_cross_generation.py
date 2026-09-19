#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
    args = parser.parse_args()
    config = EvolutionConfig(persistence=PersistenceConfig(output_dir=args.run_dir))
    matrix = EvolutionRunner(config, evaluator=MockEpisodeEvaluator()).cross_generation_evaluation()
    print(f"wrote {len(matrix)} matrix cells to {Path(args.run_dir) / 'analysis/cross_generation_matrix.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
