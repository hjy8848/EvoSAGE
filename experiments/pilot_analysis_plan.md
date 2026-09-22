# EvoSAGE pilot analysis plan

This document describes analysis only. It does not change the runtime
pipeline or scoring protocol.

## Run-level comparisons

For each independent run directory, read:

- `generations/gen_*/episodes.jsonl`: episode outcomes and invalid status;
- `generations/gen_*/customer_candidates.json`: customer fitness and candidate counts;
- `generations/gen_*/service_gate.json`: accepted, rejected and invalid candidates;
- `archives/attacks.jsonl`: deduplicated legitimate failure signatures;
- `analysis/weakness_frontier.json`: SOP-node failure frontier;
- `analysis/orchestration_metrics.json`: request, token, retry, timeout and latency cost;
- `analysis/heldout_results.json`: held-out outcomes when evaluated;
- `analysis/fresh_adversary_*.json`: fresh-adversary outcomes when evaluated;
- `real_traces/` and episode `dialogue`: representative trajectories and provider traces.

## Metrics that can be computed offline

| Research quantity | Artifact source | Computation |
|---|---|---|
| task success by generation | `episodes.jsonl` | mean `task_success` over valid episodes |
| action execution | `episodes.jsonl` | mean `action_execution_score` over valid episodes |
| goal fulfillment | `episodes.jsonl` | mean `goal_fulfillment_score` over valid episodes |
| legitimate failures | `episodes.jsonl` | valid episodes with `task_success=false` |
| invalid rate | `episodes.jsonl` | `evaluation_status=invalid` / all episodes |
| signature diversity | `archives/attacks.jsonl`, episode `failure_signature` | unique signature IDs and error/node pairs |
| frontier growth | `weakness_frontier.json` per generation | row count and distinct SOP nodes |
| customer fitness | `customer_candidates.json` | selected and candidate fitness by generation |
| service acceptance | `service_gate.json` | accepted / valid candidate records |
| service rejection reasons | `service_gate.json` | counts by `reason` and `evaluation_status` |
| normal regression | service candidate records | `normal_regression_cases` and normal deltas |
| held-out robustness | `heldout_results.json` | valid task success and execution metrics |
| fresh-adversary robustness | `fresh_adversary_*.json` | valid task success by round and target service |
| API cost | `orchestration_metrics.json` | requests, input/output tokens and latency |
| representative failure | episode `dialogue`, `llm_attempts`, backend events | reconstruct turn/tool/action timeline |

Invalid protocol/provider episodes must be reported separately and excluded
from substantive business-rate denominators. Do not recompute scores from raw
chat text.

## Minimal formal matrix

Run the following independently with the same frozen code, seed, split,
model, max turns and evaluation protocol:

1. Static baseline: fixed Customer and fixed Service.
2. Customer-only: Customer evolves, Service remains fixed.
3. Full coevolution: Customer and Service alternate evolution.
4. Optional service-only: include only if provider budget permits.

Use at least two seeds for the final claim when resources allow. If only one
seed is possible, report the main comparison together with representative
failure trajectories and the causal/tool-contract ablations as limitations,
not as a variance estimate.

## Deferred causal ablations

Keep these out of the main run directories and do not change evaluator or
scoring:

- manual ServicePolicy rule for missing required identifiers;
- strict `query_order.order_id` contract with `minLength: 1`.

Both must reuse the same CaseSpec, CustomerPolicy, model and seed/config, and
must report tool arguments, task success, goal fulfillment and error labels
side by side with baseline.
