# EvoSAGE adversarial co-evolution

This layer currently targets only `ecommerce_refund`. It does not change the
existing SAGE evaluator, expected paths, weights, or backend truth. It adds a
reusable customer-policy search and structured service-policy patch loop above
that environment.

In real mode, candidate policies and patches are generated from abstract
failure signatures by the configured LLM, then hard-validated and evaluated.
The deterministic mutation/template path remains as an offline fallback when a
generation is invalid or unavailable; it is explicitly recorded as a fallback,
not presented as a learned research result.

## Reproducible offline run

```bash
./.venv/bin/python scripts/run_adversarial_coevolution.py \
  --config configs/ecommerce_coevolution.yaml --evaluator mock
```

The explicit `--evaluator mock` selects a deterministic mock evaluator, so this
command makes zero API calls. It writes `evolution`, `validation`, and `heldout_test` manifests,
versioned policies, attack/defense archives, weakness-frontier JSON/CSV, and a
report. Held-out cases are never passed to an evolver.

Use `--resume` to continue from completed generation markers. Use
`evaluate_cross_generation.py` after a run to produce the customer-generation
by service-generation matrix, and `evaluate_fresh_adversary.py` to propose
fresh customer strategies and evaluate them on the held-out set after
evolution. Fresh strategies are not added to the attack archive.

`configs/ecommerce_coevolution_pilot.yaml` is a small three-generation pilot
(3 instances per path). `configs/ecommerce_coevolution.yaml` is the larger
150-case template, and `configs/ecommerce_coevolution_research.yaml` is a
larger five-generation/20-instances-per-path template. None launches
automatically.

## What can evolve

`CustomerPolicy` contains only reusable interaction behavior: disclosure
timing, pressure, contradiction, escalation, and response to verification.
Validity checks reject benchmark IDs, expected paths/actions, hidden backend
values, parser attacks, and case mutation. `ServicePolicy` is a list of
sanitized rules. Patches cannot rewrite source code or encode case-specific
answers; the service gate checks adversarial improvement, normal-user
regression, and degenerate query/transfer/reject behavior.

The mock loop is an integration fixture, not a research result. Real API
evaluation uses `EvoSAGEEpisodeEvaluator` with the configured
`LLMEvaluationPipeline`; `--evaluator real` (or `--real`) is also available on
the cross-generation and fresh-adversary scripts. Keep credentials in the existing macOS Keychain
workflow, never in configs or result files.
