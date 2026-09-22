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

The UI initially loads `public/data/demo.json`. The dataset is explicitly
marked `DEMO` and must not be treated as a real experiment result.

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
- Evolution: generation timeline for Customer and Service lanes.
- Episodes: filterable trace list and episode detail panel.
- Failures: legitimate versus invalid evaluation explorer.
- Diagnostics: request, token, latency, provider and reproducibility checks.

The layout intentionally uses a narrow sidebar, compact tabs, subtle borders,
table-first density and a vertical generation timeline. It borrows the
information architecture of research observability tools such as LangSmith
without copying branding or assets.
