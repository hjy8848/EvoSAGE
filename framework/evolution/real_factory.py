"""Small shared constructor for real OpenAI-compatible evaluation scripts."""

from __future__ import annotations

from pathlib import Path

from .evaluator_adapter import EvoSAGEEpisodeEvaluator


def make_real_evaluator(model: str, api_url: str, api_key: str, output_dir: str | Path,
                        max_turns: int = 10, user_simulator_mode: str = "llm", api_timeout: int = 300):
    from run_evaluation_with_llm import LLMEvaluationPipeline

    def pipeline_factory(customer_policy, service_policy):
        return LLMEvaluationPipeline(
            scenario_id="ecommerce_refund",
            model_name=model,
            output_dir=str(Path(output_dir) / "real_traces"),
            eval_mode="api",
            api_key=api_key,
            api_url=api_url,
            user_model_name=model,
            agent_model_type="api",
            agent_model_name=model,
            judge_model_name=model,
            max_turns=max_turns,
            api_timeout=api_timeout,
            verbose=False,
            user_simulator_mode=user_simulator_mode,
            customer_policy=customer_policy,
            service_policy=service_policy,
        )

    return EvoSAGEEpisodeEvaluator(pipeline_factory)
