# EvoSAGE: open-ended Customer search

## Active research question

Under a fixed benchmark environment and fixed Service S0, can black-box
evolution of free-form Customer strategies discover interactions that lower the
Service's official benchmark task-success rate?

This first-stage method does not evolve the Service, perform co-evolution, or
claim that all task failures are attributable to Customer tactics.

## Active loop

```text
AdversaryPolicy (free-text strategy)
        ↓
CustomerEvolver(parent strategy + scalar official fitness)
        ↓
LLM Customer
        ↓
fixed Service S0 ↔ official backend/tools
        ↓
official evaluator
        ↓
complete E-panel fitness; incomplete panel is inconclusive
        ↓
strict elitist selection
```

Customer behavior is unconstrained by truthfulness, cooperation, realism,
assigned-goal preservation, or a tactic taxonomy. The Customer may lie, invent
claims, contradict itself, withhold information, refuse, stall, change or
abandon its request, use false identifiers, apply pressure, or request improper
actions. A Customer utterance never changes backend truth; only official tools
can change the environment.

The integrity boundary is capability-based: the Customer and Evolver do not
receive hidden backend state, expected actions/paths, gold outcomes, held-out
answers, evaluator internals, or private benchmark metadata. They cannot modify
the benchmark, evaluator, tools, splits, or experiment records. Provider,
transport, timeout, parsing, missing-score, and environment failures are
runtime-invalid; they are not attack reward.

## Fitness and selection

For a candidate strategy evaluated on the same E panel with `R` repetitions,
the expected panel size is `|E| × R`. Selection fitness is available only when
the number of runtime-valid episodes exactly equals that expected count:

```text
if valid_episode_count == expected_episode_count:
    official_task_success = mean(official task_success over the complete E panel)
    fitness = 1 - official_task_success
else:
    evaluation_status = inconclusive
    fitness = null
```

Invalid episodes never count as attack success and cannot shrink the fitness
denominator. An incomplete incumbent makes that generation/run inconclusive
before children are proposed; an incomplete child is ineligible for selection.
Novelty, coverage, signatures, attribution, or semantic Customer-validity
judgments do not affect selection. The selected strategy replaces the
incumbent only on a strict complete-panel fitness improvement; ties, worse
scores, and inconclusive children retain the incumbent.

The Evolver sees only the parent strategy, parent scalar fitness, generation,
and requested candidate count. It does not see cases, transcripts, tool
arguments, failure labels, or hidden outcomes. Each candidate is evaluated on
the same E cases against the same fixed `service_policy_s0`.

## Active implementation

- `framework/evolution/customer/policy.py`: free-text strategy and provenance.
- `framework/evolution/customer/evolver.py`: proposal generation; no fixed
  tactic taxonomy.
- `framework/evolution/customer/selector.py`: complete-panel official-score
  fitness and strict elitism.
- `framework/evolution/customer/runner.py`: fixed-S0 generation loop,
  evaluation, selection, persistence, and report-only validation.
- `framework/evolution/evaluator_adapter.py`: official outcome adapter and
  explicitly non-research mock plumbing fixture.
- `framework/evolution/split_manager.py`: deterministic E/V/H case panel.
- `scripts/run_customer_search.py`: the single active EvoSAGE research CLI.

Generation artifacts contain proposals, scored episodes, selection (including
expected panel size and per-policy valid/invalid counts), selected Customer
strategy, the unchanged S0 identity, summary, and completion marker.
The run also stores split manifests, traces/provider provenance, and
request-budget/orchestration metrics. There are no active Service candidate,
repair gate, archive, signature, attribution, weakness-frontier, or fresh
adversary artifacts.

## Run

Deterministic plumbing check:

```bash
PYTHONPATH=. .venv/bin/python scripts/run_customer_search.py \
  --config configs/customer_search_smoke.yaml \
  --evaluator mock
```

Real search (inject credentials from the local environment/keychain; never put
them in config or artifacts):

```bash
PYTHONPATH=. .venv/bin/python scripts/run_customer_search.py \
  --config path/to/customer-search.yaml \
  --evaluator real \
  --model deepseek-v4-flash \
  --api-url https://inferaiapi.com/v1 \
  --client openai_api
```

A mock run only verifies orchestration. It is not benchmark evidence and its
mock evaluator is policy-agnostic; it does not simulate attack success.

## Historical methods and artifacts

Older Service evolution/co-evolution designs, configurations, and experiment
artifacts are historical records only. The current runtime intentionally does
not load or execute them. Use the corresponding historical Git commit when an
old result needs to be reproduced. See `experiments/README.md` for the status of
existing runbooks and results.
