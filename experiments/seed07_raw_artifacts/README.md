# Seed 7 raw conversations and model outputs

This directory contains artifact-derived dialogue and model-output records from the
completed Seed 7 **REAL** runs:

- Static: `formal_20260923_static_seed07`
- Customer-only: `formal_20260923_customer_only_seed07_fresh_20260923_011241_230e88`
- Coevolution: `formal_20260923_coevolution_seed07`

`episodes.jsonl` contains the customer/agent conversation, tool calls and results,
action calls and results, and the provider response for each Agent tool-loop attempt.
`evolution_outputs.jsonl` contains Customer/Service Evolver raw attempts and generated
candidates, plus the artifact-recorded Service gate decisions. Model `reasoning` fields
are retained where the provider recorded them. `manifest.json` records source and export
SHA-256 digests and record counts.

The export is for research inspection, not rescoring. It deliberately excludes evaluator
records, scores, gold paths, full CaseSpecs, hidden backend snapshots, simulator-private
state, full prompts, and request headers/credentials. Tool-visible results and the
environment's final status are retained. The exporter does not call a model or evaluator
and does not modify source run artifacts.

These are sensitive raw research traces and should remain in the private EvoSAGE
repository. They are not a new independent experiment batch; the three runs are the
Seed 7 pilot/formal-run artifacts already described in the project notes.

To regenerate from the local ignored `results/` artifacts:

```bash
python3 scripts/export_seed07_raw_artifacts.py
```

Optional `--static`, `--customer-only`, `--coevolution`, and `--output-dir` arguments
can point to other run directories and an alternate export destination.
