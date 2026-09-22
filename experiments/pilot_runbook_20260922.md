# EvoSAGE real pilot runbook — 2026-09-22

This runbook is for the first real provider pilot. It does not modify the
frozen runtime, evaluator, scoring, or evolution algorithm.

## Freeze and prerequisites

- Code freeze commit: `4dfc56d6c92fff86f8310e7fcf2fda12b41df42c`
- Code freeze tag: `exp-freeze-2026-09-22`
- Repository: `https://github.com/hjy8848/EvoSAGE.git`
- Python: `3.9.6` in `.venv`
- Provider: InferAI
- API URL: `https://inferaiapi.com/v1`
- Client: `openai_api`
- Model: `deepseek-v4-flash`
- Seed: `7`
- Pilot generations: `2`
- Pilot max turns: `5`

The current `main` contains only the pilot configs, manifest and analysis
documents after the freeze tag. No core runtime code was changed after the
freeze.

Before running:

```bash
git fetch evosage --tags
test "$(git rev-parse 'exp-freeze-2026-09-22^{}')" = \
  4dfc56d6c92fff86f8310e7fcf2fda12b41df42c
git diff --check
test -z "$(git status --short)"
test -x .venv/bin/python
```

Load the key without putting its value in the shell history or repository.
Use the actual Keychain service label used when the InferAI key was stored:

```bash
export OPENAI_API_KEY="$(security find-generic-password \
  -a 'openai-api-key' \
  -s 'inferaiapi.com' \
  -w)"
test -n "$OPENAI_API_KEY"
```

If the key was stored under a different service label, change only the
`-s` value locally; never replace it with the key itself in this document.

## Provider preflight

This checks authentication and model availability without printing the key:

```bash
curl --fail --silent --show-error --max-time 20 \
  -H "Authorization: Bearer ${OPENAI_API_KEY}" \
  https://inferaiapi.com/v1/models \
| .venv/bin/python -c '
import json, sys
data=json.load(sys.stdin)
ids=[item.get("id", "") for item in data.get("data", [])]
model="deepseek-v4-flash"
print({"models": len(ids), "model_available": model in ids})
raise SystemExit(0 if model in ids else 2)
'
```

Do not start the pilot if this fails, if the model is unavailable, or if the
completion endpoint returns repeated 401/403/429/5xx responses.

## Exact real pilot commands

Run from the repository root with the key already loaded above. The CLI
currently uses `--real`, `--model`, `--api-url` and `--client`:

### Static baseline

```bash
PYTHONPATH=. .venv/bin/python scripts/run_adversarial_coevolution.py \
  --config configs/ecommerce_pilot_20260922_static.yaml \
  --real \
  --model deepseek-v4-flash \
  --api-url https://inferaiapi.com/v1 \
  --client openai_api
```

Expected configured directory: `results/pilot_20260922_static/`.

### Customer-only

```bash
PYTHONPATH=. .venv/bin/python scripts/run_adversarial_coevolution.py \
  --config configs/ecommerce_pilot_20260922_customer_only.yaml \
  --real \
  --model deepseek-v4-flash \
  --api-url https://inferaiapi.com/v1 \
  --client openai_api
```

Expected configured directory: `results/pilot_20260922_customer_only/`.

### Full coevolution

```bash
PYTHONPATH=. .venv/bin/python scripts/run_adversarial_coevolution.py \
  --config configs/ecommerce_pilot_20260922_coevolution.yaml \
  --real \
  --model deepseek-v4-flash \
  --api-url https://inferaiapi.com/v1 \
  --client openai_api
```

Expected configured directory: `results/pilot_20260922_coevolution/`.

All three configs use `resume=false`. If the configured directory already
contains prior run state, the runner creates a separate `*_fresh_*` directory.
Use the actual resolved directory printed by the result/report path; do not
merge artifacts between runs.

## Pilot success criteria

All three runs must satisfy:

- at least one `evaluation_status=valid` episode;
- invalid/provider rate is present and reportable;
- generation JSONL, split manifest and policy artifacts exist;
- `analysis/orchestration_metrics.json` exists;
- raw trace files exist under `real_traces/` or the configured trace directory;
- no cache, archive or artifact is read from or written into another run;
- provider request, token, retry, timeout and latency statistics are present.

Customer-only additionally requires:

- at least one Customer candidate record;
- selected Customer policy and fitness recoverable from
  `generations/gen_*/customer_candidates.json`.

Full coevolution additionally requires:

- at least one Service candidate record;
- patch, candidate policy and source-failure provenance recoverable from
  `generations/gen_*/service_gate.json`;
- valid, rejected and invalid candidate states distinguishable;
- the gate reason and metrics are interpretable.

An abort caused by provider transport, quota, timeout or repeated truncation
must be kept as a provider-aborted run and excluded from pilot outcome rates.
An abort must not be converted into a business failure or used to claim a
Service/Customer trend.

## Scope of this pilot

`max_cases=3` is deliberately a smoke-sized configuration. With the current
`SplitManager`, it represents one evolution case, one validation case and one
held-out case. It is not a six-path or full-benchmark sample.

Two generations are a trajectory axis, not two independent samples. Independent
evidence comes from cases, seeds and repetitions. Therefore:

> Pilot validates orchestration/provider/cost only; it is not used for
> statistical research claims.

Do not select formal `candidate_count`, generation count or seed count from
pilot results until API cost, latency, invalid rate and candidate behavior are
observed.

## Offline summary after all three runs

The read-only summary tool consumes structured artifacts only; it does not
invoke the model or re-score raw text:

```bash
PYTHONPATH=. .venv/bin/python scripts/summarize_pilot_runs.py \
  --static results/pilot_20260922_static \
  --customer-only results/pilot_20260922_customer_only \
  --coevolution results/pilot_20260922_coevolution
```

If a run resolved to `*_fresh_*`, replace that argument with the actual
resolved directory. Use `--json` for machine-readable output.

The summary reports valid/invalid episodes, invalid rate, structured task and
execution metrics, legitimate failures, signature diversity, candidate/gate
counts, request/token/timeout/provider-failure statistics and latency. It also
checks artifact completeness and raw-trace presence.

## Deferred ablations

After the main pilot and only in separate run directories:

1. baseline vs the manual missing-identifier ServicePolicy rule;
2. baseline tool contract vs `query_order.order_id: {"minLength": 1}`.

Both must reuse the same CaseSpec, CustomerPolicy, model and seed/config. They
must not change evaluator or scoring and must be reported separately from the
main experiment.

## Calibrated protocol pilot

The original pilot configs remain historical and are not overwritten. The
standalone `deepseek-v4-flash` Customer Evolver probe showed that a 4096-token
structured-generation budget can be exhausted by reasoning before JSON is
returned. The calibrated pilot therefore uses:

- Agent: 4096
- Customer Evolver: 8192
- Service Evolver: 8192
- User: 512
- Judge: 1024
- Evolver protocol retry: at most one retry, only for truncation/empty or
  malformed structured output, timeout, or provider error

These are protocol-calibration settings, not settings selected from formal
research results. A protocol-invalid generation is marked inconclusive and
is not treated as fitness zero or a legitimate business failure.

Run the new one-generation pilots in this order:

```bash
PYTHONPATH=. .venv/bin/python scripts/run_adversarial_coevolution.py \
  --config configs/ecommerce_pilot_20260922_calibrated_static.yaml \
  --real --model deepseek-v4-flash --api-url https://inferaiapi.com/v1 --client openai_api

PYTHONPATH=. .venv/bin/python scripts/run_adversarial_coevolution.py \
  --config configs/ecommerce_pilot_20260922_calibrated_customer_only.yaml \
  --real --model deepseek-v4-flash --api-url https://inferaiapi.com/v1 --client openai_api

PYTHONPATH=. .venv/bin/python scripts/run_adversarial_coevolution.py \
  --config configs/ecommerce_pilot_20260922_calibrated_coevolution.yaml \
  --real --model deepseek-v4-flash --api-url https://inferaiapi.com/v1 --client openai_api
```

Expected output directories:

- `results/pilot_20260922_calibrated_static`
- `results/pilot_20260922_calibrated_customer_only`
- `results/pilot_20260922_calibrated_coevolution`

For Customer-only, inspect
`generations/gen_000/customer_generation.json` before proceeding. At least
one candidate must be recoverable with a validator PASS, selected policy,
fitness and source failure IDs. A first invalid attempt followed by a valid
retry is valid with retry provenance; two invalid attempts are INCONCLUSIVE and
block the co-evolution pilot. If Customer-only passes, start co-evolution
immediately with the same calibrated budgets.
