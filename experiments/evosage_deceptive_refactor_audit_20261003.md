# EvoSAGE open-ended Customer search contract — audited 2026-10-04

> **Superseded historical snapshot.** This audit was written during the
> transition and incorrectly describes several old Service/co-evolution
> modules and configs as still present. The active repository was subsequently
> simplified; use [`../docs/CUSTOMER_ADVERSARIAL_SEARCH.md`](../docs/CUSTOMER_ADVERSARIAL_SEARCH.md)
> as the current contract. This file is retained to preserve the audit trail,
> not as an implementation/run guide.

## Current research question

The first stage tests whether freely generated Customer interaction strategies
can reduce a **fixed Service S0's** official benchmark performance. It does not
claim Service evolution or co-evolution.

Customer may lie, fabricate business claims or identifiers, withhold, contradict,
refuse, stall, pressure, change the story, change or abandon the request, or use
other natural-language strategies. Customer speech is not world truth; the
backend, official tools, and official evaluator define outcomes.

## Current Customer-only method

```text
parent AdversaryPolicy.strategy + scalar official fitness
                         ↓
              Customer Evolver proposes K strategies
                         ↓
incumbent + each child on the same E panel against fixed Service S0
                         ↓
       official task_success on runtime-valid episodes
                         ↓
           fitness = 1 - mean(task_success)
                         ↓
     strict selection; ties retain the incumbent
```

The Evolver does not receive case IDs, traces, tool arguments, failure
categories/signatures, backend values, gold actions/paths, held-out answers,
evaluator internals, or structured episode feedback. Its non-outcome inputs are
the generation index and requested candidate count needed to form the proposal
request. Proposal diagnostics do not act as a behavioral taxonomy.

For `K` proposed children, `|E|` evolution cases, and `R` repetitions, each
generation runs:

```text
(K + 1) × |E| × R
```

The `+1` is the incumbent re-evaluated in that generation. Each selected policy
reuses its already-computed episode results; there is no selected-candidate
rerun, Customer failure scan, or attribution-based pass in this runner. V is
report-only after the final generation; H is not used or evaluated here.

## Fitness and validity boundary

For each policy, fitness is exactly:

```text
1 - mean(official task_success)
```

The denominator contains only runtime-evaluable episodes with an official
boolean `task_success`. Provider/transport errors, timeouts, parse/truncation or
empty-output failures, missing official scores, and invalid environment runs
are excluded. An empty valid set is inconclusive, never fitness zero.

Customer truthfulness, goal preservation, realism, tactic labels, failure
attribution, FailureSignature, novelty, and SOP diversity do not gate or
contribute to Customer selection. A false claim or an abandoned request is not
invalid merely because of its wording or semantics. Hidden-state isolation and
benchmark integrity remain protected by capability boundaries: Customer and
Evolver do not receive hidden backend state or evaluation answers, and neither
can mutate the backend, tools, evaluator, scoring, or experiment artifacts.
The small text guard is limited to explicit benchmark-infrastructure tampering;
ordinary requests to change a business record or skip verification remain
eligible Customer behavior.

## Core versus legacy/analysis structure

| Area | Current role |
|---|---|
| `framework/evolution/customer/` | Compact free-text policy, integrity boundary, and Customer-only runner |
| `framework/evolution/customer_evolver.py` / `customer_selector.py` | Proposal from parent strategy + scalar official fitness; official-outcome selection |
| `framework/evolution/config.py::CustomerSearchConfig` | Customer-only config; excludes Service evolution, replay, fresh adversary, and legacy selection fields |
| `framework/evolution/analysis/customer_behavior.py` | Post-hoc/legacy analysis only; not imported by the Customer runner or selector |
| `framework/evolution/legacy/customer_policy.py` | Legacy compiler/validator for combined historical runs |
| `framework/testing/policy_customer_fixture.py` | Deterministic tag-driven fixture for tests only; never the REAL Customer executor |
| Service evolver/gate, archives, signatures, attribution | Retained for legacy/other experiments; not Customer-only selection dependencies |

`customer_feedback.py` was removed because the scalar-only method had no call
site for structured outcome feedback. `source_evidence`/`source_failure_ids` are
ignored at old-artifact import boundaries and are not fields in the compact
`AdversaryPolicy`.

## REAL path and provenance

`LLMCustomerPolicyGenerator` proposes free-text strategies. The LLM Customer
receives that strategy, Customer-visible information, public tool results, and
the visible dialogue. Fixed S0 and official tools execute the interaction; the
official evaluator produces `task_success`; `CustomerSelector` excludes only
runtime-invalid evidence and applies strict incumbent retention.

Customer-only artifacts store the compact config, policy lineage, per-candidate
official scores, split manifests, valid/invalid counts, provider usage, and
traces. Episode JSONL omits behavior-validity, attribution, and signature
analysis fields. It does not create attack/defense archives, weakness frontiers,
or Service-gate records.

Historical `EvolutionConfig` remains available for Service/co-evolution and
accepts old config fields there. The CLI makes an explicit conversion to
`CustomerSearchConfig` for Customer-only runs; `elite_count`,
`cases_per_candidate`, `fitness_weights`, `allowed_strategy_tags`,
`adversary_access`, Service settings, and fresh-adversary settings are not
serialized into the new Customer-only manifest.

## Smoke profile

`configs/ecommerce_deceptive_customer_only_smoke_20261003.yaml` is retained as
the current clean, bounded tiny smoke profile: one generation, one proposed
Customer candidate, at most three deterministic cases (one E, one V, one H),
fixed S0, serial API execution, explicit request/token/turn limits, and a fresh
output directory. It is a mechanism check only—not formal or statistical
evidence. InferAI credentials are injected through `OPENAI_API_KEY` and never
stored in config or result artifacts.

Run deterministic offline orchestration with:

```bash
PYTHONPATH=. .venv/bin/python scripts/run_adversarial_coevolution.py \
  --config configs/ecommerce_deceptive_customer_only_smoke_20261003.yaml \
  --evaluator mock
```

For a real provider check, use `--evaluator real --model deepseek-v4-flash` and
load `OPENAI_API_KEY` from the local Keychain/environment. A mock run is not a
research result.

## Interpretation

If the official evaluator rewards refusal, stalling, or goal abandonment, keep
that result in the data and characterize it post hoc. It may reveal a
benchmark/evaluator or service-handling weakness. Do not relabel it as invalid
based only on Customer semantics, and do not claim Customer evolution is
limited to realistic or attributable business attacks.
