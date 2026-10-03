# EvoSAGE Customer evolution

## Research question

The first stage asks:

> When Customer behavior is open-ended and the benchmark environment is fixed,
> can black-box evolutionary search discover interaction strategies that reduce
> a Service's official benchmark performance?

This stage does **not** claim Service co-evolution or repair.

## Core loop

```text
AdversaryPolicy C_t
       ↓ Customer Evolver (parent strategy + scalar reward only)
K free-form candidate strategies
       ↓
same E cases / fixed Service S0 / same evaluator
       ↓
official task_success on runtime-valid episodes
       ↓ fitness = 1 - mean(task_success)
strict elitist selection
       ↓
C_t+1
```

Customer can lie, fabricate business claims, withhold, contradict, mislead,
pressure, refuse, stall, change or abandon the request, or use other
natural-language behavior. None is rejected for being untruthful,
uncooperative, unrealistic, or goal-abandoning. These choices may later be
described in post-hoc analysis, but do not gate online fitness.

## The only hard boundaries

Runtime-invalid episodes do not contribute fitness: provider/transport errors,
timeouts, truncation, parse/empty-output failures, incomplete benchmark runs,
and episodes without a valid Service decision.

Benchmark integrity remains protected. Customer and Evolver receive no hidden
backend state, gold action/path, held-out answers, evaluator internals, or
private benchmark metadata. Customer strategies cannot mutate tasks, tools,
backend, evaluator, scoring, splits, or experiment records. A Customer's spoken
claim never changes backend truth.

The policy generator receives only the parent strategy, its scalar official
fitness, generation index, and requested candidate count. It does not receive
episode traces, failure categories, signatures, case IDs, or hidden outcomes.

## Objective and selection contract

For candidate `C` with runtime-evaluable episodes `E_C`:

```text
official_task_success(C) = mean(task_success(e) for e in E_C)
fitness(C) = 1 - official_task_success(C)
```

An empty valid set is inconclusive, not zero fitness. Provider/protocol-invalid
episodes are excluded from the denominator. `customer_behavior_valid`,
`service_failure_attributable`, failure signatures, novelty, and SOP diversity
are not read by the Customer selector.

Replacement is strictly elitist: a child replaces its incumbent only when its
fitness is strictly greater. Ties and inconclusive children retain the
incumbent. No candidate is promoted for novelty, strategy length, or a
researcher-defined attack category.

## Customer-only runner

`framework/evolution/customer/runner.py` runs one baseline incumbent plus its
proposed children once each on the same E panel and fixed `service_policy_s0`.
The chosen policy reuses its just-computed evaluation; there is no
`customer_failure_scan`, attribution/signature pass, archive/frontier update, or
selected-customer rerun. V is evaluated only after the final generation and is
report-only. H is never passed to proposal, selection, or evaluation by this
runner.

Core artifacts per generation:

```text
generations/gen_NNN/
  proposals.json
  episodes.jsonl
  selection.json
  customer_policy.json
  service_policy.json
  COMPLETE.json
```

The run also stores split manifests, `analysis/trajectory.json`,
`analysis/orchestration_metrics.json`, report-only validation artifacts, and
real traces/provider provenance. Customer-only runs do not create attack or
defense archives, weakness frontiers, or service-gate records.

## Run a deterministic orchestration fixture

```bash
PYTHONPATH=. .venv/bin/python scripts/run_adversarial_coevolution.py \
  --config configs/ecommerce_coevolution.yaml \
  --mode customer_only --evaluator mock
```

For a real run, use `--evaluator real --model ...` and inject
`OPENAI_API_KEY` from the local Keychain/environment. Never place credentials
in configuration or artifacts. A mock run is an orchestration fixture, not
research evidence.

## Legacy service/co-evolution path

The existing `EvolutionRunner`, service policy/gate/evolver, failure
attribution, `AttackArchive`, `DefenseArchive`, and `WeaknessFrontier` remain in
the repository for legacy combined experiments and later research. They are
not part of the new Customer-only core. Historical artifacts remain readable;
legacy config fields `fitness_weights`, `allowed_strategy_tags`, and
`adversary_access` are ignored with a deprecation warning and are not serialized
into new Customer config.

## Interpretation cautions

If evolution selects silence, refusal, stalling, or goal abandonment because
the official evaluator rewards it, record that behavior rather than filtering
it online. It may expose a benchmark/evaluator weakness or a service-handling
failure; post-hoc analysis should characterize it. Do not claim that the system
only discovers realistic or attributable vulnerabilities unless a separate
analysis establishes that.
