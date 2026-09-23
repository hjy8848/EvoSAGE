"""Small shared constructor for real OpenAI-compatible evaluation scripts."""

from __future__ import annotations

from pathlib import Path

from .evaluator_adapter import EvoSAGEEpisodeEvaluator


def make_real_evaluator(model: str, api_url: str, api_key: str, output_dir: str | Path,
                        max_turns: int = 10, user_simulator_mode: str = "llm", api_timeout: int = 300,
                        client_type: str = "openai_api", judge_in_evolution: bool = False,
                        resume: bool = False, token_budget=None,
                        customer_thinking_mode=None):
    from run_evaluation_with_llm import LLMEvaluationPipeline

    def budget(name: str, default: int) -> int:
        if token_budget is None:
            return default
        if isinstance(token_budget, dict):
            return int(token_budget.get(name, default))
        return int(getattr(token_budget, name, default))

    def pipeline_factory(customer_policy, service_policy, judge_enabled=True):
        return LLMEvaluationPipeline(
            scenario_id="ecommerce_refund",
            model_name=model,
            output_dir=str(Path(output_dir) / "real_traces"),
            eval_mode="api",
            api_key=api_key,
            api_url=api_url,
            client_type=client_type,
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
            use_llm_judge=judge_enabled,
            user_max_tokens=budget("user", 512),
            agent_max_tokens=budget("agent", 1536),
            judge_max_tokens=budget("judge", 1024),
            customer_thinking_mode=customer_thinking_mode,
        )

    return EvoSAGEEpisodeEvaluator(
        pipeline_factory,
        judge_in_evolution=judge_in_evolution,
        cache_namespace=(
            f"{client_type}|{model}|{api_url}|turns={max_turns}|timeout={api_timeout}"
            f"|customer-thinking={customer_thinking_mode or 'default'}"
        ),
        cache_path=Path(output_dir) / "environment" / "episode_cache.jsonl",
        reset_cache=not resume,
    )
