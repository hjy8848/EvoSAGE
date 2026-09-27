# EvoSAGE Performance Pass Runbook (2026-09-28)

This is an execution-cost calibration, not a formal research run. It preserves the
same Ecommerce Backend, CaseSpec truth, SOP, evaluator and attribution rules.
The profiling run is capped at 30 actual provider attempts total.

## Profiles

- `configs/ecommerce_performance_profile_20260928.yaml`: one generation,
  one evolution case + one validation case + one sealed held-out case,
  one Customer candidate, one Service candidate, replay off, Judge off,
  one transport attempt per call, zero invalid-episode retries, 30-attempt
  generation/run hard caps.
- `configs/ecommerce_fast_evolution_smoke_20260928.yaml`: two generations,
  four cases split 2/1/1 (evolution/validation/held-out), two Customer
  candidates, one Service candidate, replay off, Judge off, 50 attempts per
  generation and 120 per run. Run only after the profiling run is interpretable.

The held-out split is still created and persisted. The co-evolution runner does
not perform final held-out evaluation as part of these smoke profiles.

## Real profiling command

Load the existing InferAI credential from macOS Keychain without placing the
secret in the repository or command text:

```sh
export OPENAI_API_KEY="$(security find-generic-password \
  -a 'openai-api-key' \
  -s 'inferaiapi.com' \
  -w)"
```

Then run from the repository root:

```sh
PYTHONPATH=. .venv/bin/python scripts/run_adversarial_coevolution.py \
  --config configs/ecommerce_performance_profile_20260928.yaml \
  --real \
  --model deepseek-v4-flash \
  --api-url https://inferaiapi.com/v1 \
  --client openai_api
```

The configured `results/performance_profile_20260928` path is resumable. Its
`analysis/request_budget.json` tracks actual provider attempts across resumes;
completed valid episodes remain eligible for the existing exact-key cache.
An exhausted cap stops before the next request and writes
`run_status=budget_exhausted`, a generation `CHECKPOINT.json`, request metrics,
and the episode cache path. This is an inconclusive execution stop, not a
business failure.

## Profiling checks

Inspect these artifacts after the command:

- `analysis/request_budget.json`: provider attempts by generation/phase/role,
  retries, tokens, latency, timeout and provider failures.
- `analysis/request_budget_estimate.json`: explicitly rough workload estimate.
- `analysis/orchestration_metrics.json`: aggregate client/cache/run status.
- `environment/episode_cache.jsonl`: completed reusable episodes.
- `real_traces/` and generation candidate/provenance JSON: role outputs and
  evolution candidates.

Stage B is interpretable only if request accounting is within the hard cap,
cache and traces are present, provider/protocol failures remain separate from
business failures, and runtime settings in `environment/provenance.json` match
the profile. If the run cleanly completes and the calls/tokens/latency are
plausible, the fast smoke may be run with the second config. Do not launch any
formal protocol config as part of this pass.
