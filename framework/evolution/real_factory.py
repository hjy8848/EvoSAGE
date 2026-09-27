"""Small shared constructor for real OpenAI-compatible evaluation scripts."""

from __future__ import annotations

from pathlib import Path

from ..backend.tool_contract import ToolContractConfig
from .evaluator_adapter import EvoSAGEEpisodeEvaluator


def make_real_evaluator(model: str, api_url: str, api_key: str, output_dir: str | Path,
                        max_turns: int = 10, user_simulator_mode: str = "llm", api_timeout: int = 300,
                        client_type: str = "openai_api", judge_in_evolution: bool = False,
                        resume: bool = False, token_budget=None,
                        customer_thinking_mode=None, agent_thinking_mode=None,
                        evolver_thinking_mode=None, tool_contract_config=None,
                        max_tool_steps: int = 8, invalid_evaluation_retries: int = 1,
                        request_budget=None, customer_transport_max_retries: int = 1,
                        agent_max_retries: int = 1, judge_max_retries: int = 1,
                        judge_validation_retries: int = 2,
                        customer_protocol_retries: int = 1,
                        rate_limit_backoff_seconds: float = 0.0):
    from run_evaluation_with_llm import LLMEvaluationPipeline
    tool_contract_config = ToolContractConfig.from_value(tool_contract_config)

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
            agent_thinking_mode=agent_thinking_mode,
            user_transport_max_retries=customer_transport_max_retries,
            agent_transport_max_retries=agent_max_retries,
            judge_transport_max_retries=judge_max_retries,
            judge_validation_retries=judge_validation_retries,
            customer_protocol_retries=customer_protocol_retries,
            rate_limit_backoff_seconds=rate_limit_backoff_seconds,
            max_tool_steps=max_tool_steps,
            request_budget=request_budget,
            tool_contract_config=tool_contract_config,
        )

    return EvoSAGEEpisodeEvaluator(
        pipeline_factory,
        judge_in_evolution=judge_in_evolution,
        cache_namespace=(
            f"{client_type}|{model}|{api_url}|turns={max_turns}|timeout={api_timeout}"
            f"|customer-thinking={customer_thinking_mode or 'default'}"
            f"|agent-thinking={agent_thinking_mode or 'default'}"
            f"|evolver-thinking={evolver_thinking_mode or 'default'}"
            f"|max-tool-steps={max_tool_steps}"
            f"|tokens={budget('user', 512)}:{budget('agent', 1536)}:{budget('judge', 1024)}"
            f"|transport-attempts={customer_transport_max_retries}:{agent_max_retries}:{judge_max_retries}"
            f"|customer-protocol-retries={customer_protocol_retries}"
            f"|invalid-eval-retries={invalid_evaluation_retries}"
            f"|judge-retries={judge_max_retries}:{judge_validation_retries}"
            f"|tool-contract={tool_contract_config.provider_schema_strict}:"
            f"{tool_contract_config.runtime_schema_validation}"
        ),
        cache_path=Path(output_dir) / "environment" / "episode_cache.jsonl",
        reset_cache=not resume,
        invalid_evaluation_retries=invalid_evaluation_retries,
        request_budget=request_budget,
    )
