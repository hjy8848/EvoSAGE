# EvoSAGE formal experiment runbook (planning draft)

This document prepares the formal experiment only. It does not change the
frozen EvoSAGE runtime, evaluator, fitness, gate, validator, backend, or
Dashboard.

## Frozen parent

- Commit: `1fc5aadf4e9294ef14d59f8ca096943b6d38eb67`
- Tag: `exp-freeze-2026-09-22-calibrated`
- The tag is historical and must not be moved or rewritten.
- The formal config/analysis files in this planning commit are not a new
  runtime freeze.

## Shared formal protocol

The three modes use the same:

- model: `deepseek-v4-flash`
- provider: InferAI
- API URL: `https://inferaiapi.com/v1`
- client: `openai_api`
- seed set: `[7, 17]`
- split strategy: `instance_holdout`
- max turns: `5`
- repetitions: `1`
- concurrency: `1`
- evaluator/scoring/gate: unchanged from the calibrated parent freeze
- Agent budget: `4096`
- Customer Evolver budget: `8192`
- Service Evolver budget: `8192`
- User budget: `512`
- Judge budget: `1024`
- structured-generation retry limit: `1`

The configuration templates use `max_cases: null` so case count is not frozen
prematurely. `cases_per_candidate: 0` means “evaluate candidates on all
evolution cases” in the current runner. Before a real run, copy the selected
template to a coverage-specific config and set `splits.max_cases` and
`splits.instances_per_path` together according to the 10/20/30-case option in
`formal_experiment_plan_20260923.json`.

## Config templates

- `configs/formal_20260923_static.yaml`
- `configs/formal_20260923_customer_only.yaml`
- `configs/formal_20260923_coevolution.yaml`

The templates use seed `7` and `resume: false`. For seed `17`, make a copied
config with the same settings and only change `seed`, `splits.seed`, and
`persistence.output_dir`. Never reuse a completed run directory.

The three mode semantics are:

| Mode | Customer evolution | Service evolution | Candidate search |
|---|---|---|---|
| static | off | off | no candidates used |
| customer-only | on | off | 3 Customer candidates + 1 elite incumbent |
| coevolution | on | on | 3 Customer candidates + 1 elite; 3 Service candidates + 1 replay |

## Preflight

Run from the repository root after loading the key from the local Keychain.
Do not put the key in a command, document, result, or repository file.

```bash
git diff --check
test "$(git rev-parse 'exp-freeze-2026-09-22-calibrated^{}')" = \
  1fc5aadf4e9294ef14d59f8ca096943b6d38eb67
test -x .venv/bin/python
test -n "$OPENAI_API_KEY"
```

Provider preflight is the same model-list and minimal completion check used
for the calibrated pilot. Abort before starting a formal run on repeated
401/403/429/5xx, timeout, or truncation. Keep the attempt as
`ABORTED/INCONCLUSIVE`; do not count it as a business failure.

## Fresh-run / no-resume protocol

Formal data points are uninterrupted runs:

```text
fresh run_dir → one config/seed → uninterrupted generations
```

`resume` is prohibited for formal data points because rejected Service repair
history is not fully restart-safe in the calibrated runtime. If a provider or
protocol failure interrupts a run:

1. preserve the partial directory and label it `ABORTED` or `INCONCLUSIVE`;
2. do not resume it as the same data point;
3. make a new run directory with the identical config and seed;
4. rerun from generation zero.

The old partial directory remains provenance and is excluded from outcome
rates. Never merge its cache, archive, or episodes into the replacement run.

## Commands after choosing coverage

The commands below are the canonical templates. Replace each config with the
coverage-specific copied config and use a fresh output directory for every
seed/mode/coverage combination.

```bash
PYTHONPATH=. .venv/bin/python scripts/run_adversarial_coevolution.py \
  --config configs/formal_20260923_static.yaml \
  --real --model deepseek-v4-flash \
  --api-url https://inferaiapi.com/v1 --client openai_api

PYTHONPATH=. .venv/bin/python scripts/run_adversarial_coevolution.py \
  --config configs/formal_20260923_customer_only.yaml \
  --real --model deepseek-v4-flash \
  --api-url https://inferaiapi.com/v1 --client openai_api

PYTHONPATH=. .venv/bin/python scripts/run_adversarial_coevolution.py \
  --config configs/formal_20260923_coevolution.yaml \
  --real --model deepseek-v4-flash \
  --api-url https://inferaiapi.com/v1 --client openai_api
```

Run the modes with the same chosen coverage and seed set. Do not start a
second generation or mode in a directory that contains a previous attempt.

## Read-only trajectory analysis

After a run completes, analyze structured artifacts without invoking a model
or recomputing scores:

```bash
PYTHONPATH=. .venv/bin/python scripts/analyze_evolution_trajectory.py \
  --run static=results/formal_20260923_static_seed07 \
  --run customer_only=results/formal_20260923_customer_only_seed07 \
  --run coevolution=results/formal_20260923_coevolution_seed07 \
  --output analysis/formal_20260923_seed07_trajectory.json
```

The analyzer reports customer fitness/best-so-far/novelty/coverage,
incumbent retention/replacement, service proposal and gate trajectories, and
deterministic repair recurrence. A repair fingerprint is only:

```text
lowercase → collapse whitespace → category + normalized rule text
```

No LLM judge, embedding, or runtime feedback is used. Generation is a
trajectory axis; cases, seeds and repetitions are the independent evidence.

## Offline Customer elitism sanity check

The current selector/evolver already supports `candidate_count=3` and
`elite_count=1`. Before formal launch, run the repository tests plus the
one-off offline check described in the experiment report. The check constructs
one high-fitness incumbent and three lower-fitness children and verifies that
the incumbent remains selected. It must not call a provider.

## Cost planning estimate

The following estimates are for **3 generations × 2 seeds**, using full
evolution-case candidate coverage (`cases_per_candidate=0`) and the calibrated
real pilots as the per-episode basis. Coevolution is a range because the
latest-attack filter can stop a candidate before replay/normal evaluation.
These are planning estimates, not measured results. Provider pricing was not
available, so monetary cost is intentionally shown as “unknown”; multiply
input/output token columns by the provider's current rates if needed.

| Cases | Mode | Requests | Input tokens | Output tokens | Wall-clock |
|---:|---|---:|---:|---:|---:|
| 10 | static | 108 | 261,450 | 100,260 | 20.4 min |
| 10 | customer-only | 1,146 | 2,877,582 | 1,287,750 | 3.8 h |
| 10 | coevolution | 3,276–4,092 | 7,858,038–9,821,610 | 3,308,178–4,132,482 | 11.0–13.8 h |
| 20 | static | 216 | 522,900 | 200,520 | 40.8 min |
| 20 | customer-only | 2,286 | 5,753,202 | 2,567,190 | 7.6 h |
| 20 | coevolution | 6,540–8,172 | 15,712,326–19,639,470 | 6,605,394–8,254,002 | 22.0–27.5 h |
| 30 | static | 324 | 784,350 | 300,780 | 1.0 h |
| 30 | customer-only | 3,426 | 8,628,822 | 3,846,630 | 11.4 h |
| 30 | coevolution | 9,804–12,252 | 23,566,614–29,457,330 | 9,902,610–12,375,522 | 33.0–41.2 h |

The estimate makes the tradeoff explicit: 3/1/3 is supported, but full
coverage is expensive. The recommended first formal scale is 10 cases, 3
generations, and two seeds if the provider quota supports the coevolution
range. If it does not, use the pre-declared 2/1/2 fallback with the same
three generations; never use 1/0/1 for the formal evolution claim.

## Analysis outputs required for the paper

At minimum, report per generation and across independent seeds/cases:

- Customer fitness, best-so-far fitness, legitimate attack success, novelty,
  coverage, unique failure signatures, incumbent retained/replaced;
- Service proposals, accepted/rejected/invalid, gate reason, delta,
  latest/replay/normal/robustness metrics and repair recurrence;
- task success, action execution, goal fulfillment and invalid rate from the
  existing structured episode artifacts;
- requests, input/output tokens, latency, retries, timeouts and provider
  failures;
- held-out/fresh-adversary metrics only when their artifacts exist.

Do not call a generation an independent sample. Do not claim that EvoSAGE
eliminates stagnation; the current claim is limited to population-based
candidate proposal, elitist/gated incumbent retention, historical replay and
grounded candidate evaluation.

## Blocking correctness status

No new blocking correctness issue is introduced by these planning files. The
known research-design limitation remains: rejected Service repair memory is
not fully restart-safe. That is why the formal protocol forbids resume; it is
not fixed by this planning change.
