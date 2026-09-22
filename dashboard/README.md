# EvoSAGE Research Dashboard

Read-only, LangSmith-inspired research/evaluation viewer for EvoSAGE. It reads
exported structured artifacts and never starts an experiment, edits a policy,
or recomputes scores from raw dialogue.

The dashboard is a read-only visualization layer. It does not participate in
EvoSAGE evaluation, evolution, scoring, or experiment execution.

## Local development

```bash
cd dashboard
npm install
npm run dev
```

Open the URL printed by Vite, normally `http://localhost:5173`.

The UI first tries to load `public/data/runs.json` (the output of the exporter)
and falls back to `public/data/demo.json` when that file is absent. The demo
dataset is explicitly marked `DEMO` and must not be treated as a real
experiment result.

## Export real run artifacts

Run from the repository root:

```bash
.venv/bin/python scripts/export_dashboard_data.py \
  --run static=results/pilot_20260922_static \
  --run customer-only=results/pilot_20260922_customer_only \
  --run coevolution=results/pilot_20260922_coevolution \
  --model deepseek-v4-flash \
  --provider InferAI \
  --output dashboard/public/data/runs.json
```

For formal artifacts, keep the runtime and protocol provenance explicit when
the run itself does not record it:

```bash
.venv/bin/python scripts/analyze_evolution_trajectory.py \
  --run coevolution=results/formal_20260923_coevolution_seed07 \
  --output /tmp/evosage-trajectory.json

.venv/bin/python scripts/export_dashboard_data.py \
  --run coevolution=results/formal_20260923_coevolution_seed07 \
  --model deepseek-v4-flash \
  --provider InferAI \
  --analysis coevolution=/tmp/evosage-trajectory.json \
  --formal-protocol-commit <formal-protocol-sha> \
  --formal-protocol-tag <formal-protocol-tag> \
  --output dashboard/public/data/runs.json
```

Both tools are read-only artifact analysis. The trajectory analyzer and
exporter do not invoke a model, evaluator, scorer, or experiment runner, and
they do not rescore raw dialogue. The normal workflow is:

```text
completed run artifacts
  -> analyze_evolution_trajectory.py (optional offline analysis)
  -> export_dashboard_data.py
  -> dashboard/public/data/runs.json
  -> npm run dev
```

The exporter only reads structured JSON/JSONL artifacts. Trace files are
copied into normalized episode provenance/timeline data; they are not used to
re-score conversations. If a run is marked mock by its provenance or episode
metadata, the exported run is marked `MOCK`. The demo fixture is marked
`DEMO`; neither label can be mistaken for `REAL` in the UI.

If a run resolved to a fresh directory, pass that actual directory instead.
The dashboard can display missing held-out/fresh-adversary artifacts as
`Not evaluated` rather than zero.

## Pages

- Overview: run metadata, compact metrics, comparison and featured failure.
- Experiments: table-first comparison of selected runs.
- Evolution: generation timeline for Customer, failure surface, and Service lanes.
- Episodes: filterable trace list and episode detail panel.
- Failures: aggregated legitimate-failure signatures; invalid evaluations stay separate.
- Robustness: latest/replay/normal/held-out/fresh artifact-backed checks.
- Diagnostics: request, token, latency, provider, artifact, and reproducibility checks.

The Evolution page is compatible with the formal 3/1/3 search protocol
(Customer candidate count 3, Customer elite count 1, Service candidate count 3,
with replay count read from artifacts). It displays all persisted candidate
provenance and gate outcomes rather than inferring outcomes from text. Missing
held-out or fresh-adversary artifacts are displayed as `Not evaluated`, never
as zero.

Generations are trajectory steps, not independent statistical samples. Any
statistical claim must be made offline using independent cases, seeds, and
repetitions from the experiment plan.

The layout intentionally uses a narrow sidebar, compact tabs, subtle borders,
table-first density and a vertical generation timeline. It borrows the
information architecture of research observability tools such as LangSmith
without copying branding or assets.
