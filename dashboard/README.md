# EvoSAGE Research Dashboard

This dashboard is a read-only visualization layer. It does not participate in
EvoSAGE evaluation, evolution, scoring, or experiment execution.

## Run locally

```bash
cd dashboard
npm install
npm run dev
```

The application loads `public/data/runs.json` first and falls back to the
explicitly labeled `public/data/demo.json` fixture. Demo data is illustrative,
not a research result.

## Export Customer-search artifacts

From the repository root:

```bash
.venv/bin/python scripts/export_dashboard_data.py \
  --run customer-search=results/customer-search-run \
  --output dashboard/public/data/runs.json
```

The exporter reads structured JSON/JSONL files only. It copies saved official
scores and display-only trace events; it does not invoke models or evaluators,
and does not rescore dialogue. `REAL` and `MOCK` are read from run provenance.
If an artifact cannot confirm a runtime freeze or a held-out/fresh evaluation,
the dashboard displays it as not confirmed/not evaluated rather than claiming
a mismatch or a zero score.

## Active research semantics

The current dashboard dataset is for open-ended Customer strategy search
against fixed Service S0. It does not report Service repair proposals,
attribution, failure signatures, co-evolution, or fresh-adversary robustness as
current-method metrics. Existing older experiment artifacts remain historical;
see [`../experiments/README.md`](../experiments/README.md).
