# EvoSAGE adversarial co-evolution

This layer currently targets only `ecommerce_refund`. It does not change the
existing SAGE evaluator, expected paths, weights, or backend truth. It adds a
reusable customer-policy search and structured service-policy patch loop above
that environment.

## Reproducible offline run

```bash
./.venv/bin/python scripts/run_adversarial_coevolution.py \
  --config configs/ecommerce_coevolution.yaml
```

The default is a deterministic mock evaluator, so this command makes zero API
calls. It writes `evolution`, `validation`, and `heldout_test` manifests,
versioned policies, attack/defense archives, weakness-frontier JSON/CSV, and a
report. Held-out cases are never passed to an evolver.

Use `--resume` to continue from completed generation markers. Use
`evaluate_cross_generation.py` after a run to produce the customer-generation
by service-generation matrix, and `evaluate_fresh_adversary.py` to propose
fresh customer strategies and evaluate them on the held-out set after
evolution. Fresh strategies are not added to the attack archive.

## What can evolve

`CustomerPolicy` contains only reusable interaction behavior: disclosure
timing, pressure, contradiction, escalation, and response to verification.
Validity checks reject benchmark IDs, expected paths/actions, hidden backend
values, parser attacks, and case mutation. `ServicePolicy` is a list of
sanitized rules. Patches cannot rewrite source code or encode case-specific
answers; the service gate checks adversarial improvement, normal-user
regression, and degenerate query/transfer/reject behavior.

The mock loop is an integration fixture, not a research result. Real API
evaluation should construct `EvoSAGEEpisodeEvaluator` with the configured
`LLMEvaluationPipeline` and keep credentials in the existing macOS Keychain
workflow, never in configs or result files.
