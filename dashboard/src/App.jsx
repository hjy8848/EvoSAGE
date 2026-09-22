import { useEffect, useMemo, useState } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

const NAV = [
  { id: 'overview', label: 'Overview', icon: '⌂' },
  { id: 'experiments', label: 'Experiments', icon: '▦' },
  { id: 'evolution', label: 'Evolution', icon: '↗' },
  { id: 'episodes', label: 'Episodes', icon: '◉' },
  { id: 'failures', label: 'Failures', icon: '!' },
  { id: 'robustness', label: 'Robustness', icon: '↔' },
  { id: 'diagnostics', label: 'Diagnostics', icon: '⌁' },
]

const COLORS = { blue: '#2563eb', green: '#16805b', red: '#c2413c', orange: '#c66a16', gray: '#697586' }

function fmtPercent(value) {
  return value == null || Number.isNaN(Number(value)) ? '—' : `${(Number(value) * 100).toFixed(1)}%`
}

function fmtNumber(value) {
  return value == null || Number.isNaN(Number(value)) ? '—' : Number(value).toLocaleString()
}

function fmtLatency(value) {
  return value == null || Number.isNaN(Number(value)) ? '—' : `${Number(value).toFixed(1)}s`
}

function display(value, fallback = 'Not recorded') {
  return value == null || value === '' ? fallback : String(value)
}

function sourceTone(source) {
  return source === 'REAL' ? 'real' : source === 'MOCK' ? 'mock' : source === 'DEMO' ? 'demo' : 'neutral'
}

function statusTone(status) {
  if (status === 'VALID SUCCESS' || status === 'accepted' || status === 'matched') return 'success'
  if (status === 'LEGITIMATE FAILURE' || status === 'rejected' || status === 'mismatch') return 'failure'
  if (status === 'INVALID EVALUATION' || status === 'invalid' || status === 'inconclusive') return 'invalid'
  return 'neutral'
}

function StatusBadge({ kind = 'neutral', children }) {
  return <span className={`status-badge ${kind}`}>{children}</span>
}

function MetricCard({ label, value, detail, tone = 'neutral' }) {
  return <div className={`metric-card ${tone}`}><div className="metric-label">{label}</div><div className="metric-value">{value}</div>{detail && <div className="metric-detail">{detail}</div>}</div>
}

function serviceStats(run) {
  if (run?.repair_analysis?.statistics) return run.repair_analysis.statistics
  const candidates = (run?.service_candidates || []).flatMap((item) => item.candidates || [])
  const accepted = candidates.filter((item) => item.accepted === true).length
  const invalid = candidates.filter((item) => ['invalid', 'inconclusive'].includes(item.evaluation_status)).length
  return { total_service_proposals: candidates.length, accepted_service_proposals: accepted, rejected_service_proposals: candidates.length - accepted - invalid, invalid_service_proposals: invalid, service_acceptance_rate: candidates.length ? accepted / candidates.length : null }
}

function SectionHeader({ eyebrow, title, action }) {
  return <div className="section-header"><div>{eyebrow && <div className="eyebrow">{eyebrow}</div>}<h2>{title}</h2></div>{action}</div>
}

function ExperimentHeader({ run, runs, onChangeRun, onNavigate }) {
  const metadata = run?.metadata || {}
  return <header className="experiment-header">
    <div className="breadcrumbs"><span>EvoSAGE</span><b>/</b><span>Research</span><b>/</b><strong>{display(metadata.mode)}</strong></div>
    <div className="header-row"><div><h1>{display(metadata.mode)} experiment</h1><div className="header-subline"><StatusBadge kind={sourceTone(metadata.source_kind)}>{metadata.source_kind || 'UNKNOWN'}</StatusBadge><span>{display(metadata.model)}</span><span className="dot">·</span><span>{display(metadata.provider)}</span><span className="dot">·</span><span>seed {display(metadata.seed)}</span><span className="dot">·</span><span>{fmtNumber(metadata.case_count)} cases</span><span className="dot">·</span><span>{display(metadata.run_status, 'status not recorded')}</span></div></div><div className="header-controls"><label className="run-select-label">Run<select value={run?.id || ''} onChange={(event) => onChangeRun(event.target.value)}>{runs.map((item) => <option key={item.id} value={item.id}>{item.id}</option>)}</select></label><button className="quiet-button" onClick={() => onNavigate('diagnostics')}>Reproducibility</button></div></div>
    {metadata.source_kind === 'DEMO' && <div className="demo-banner">DEMO DATA · illustrative fixture only; not a real experiment result</div>}
  </header>
}

function ExperimentTabs({ page, onChange }) {
  return <div className="tabs" role="tablist">{NAV.map((item) => <button key={item.id} className={page === item.id ? 'active' : ''} onClick={() => onChange(item.id)}>{item.label}</button>)}</div>
}

function MetricGrid({ run }) {
  const metrics = run?.metrics || {}
  const repair = serviceStats(run)
  return <div className="metric-grid metric-grid-wide">
    <MetricCard label="Task Success" value={fmtPercent(metrics.task_success)} tone="green" />
    <MetricCard label="Action Execution" value={fmtPercent(metrics.action_execution)} />
    <MetricCard label="Goal Fulfillment" value={fmtPercent(metrics.goal_fulfillment)} />
    <MetricCard label="Valid Episodes" value={fmtPercent(metrics.valid_episode_rate)} tone="blue" />
    <MetricCard label="Legitimate Failures" value={fmtPercent(metrics.legitimate_failure_rate)} tone="red" />
    <MetricCard label="Unique Failures" value={fmtNumber(metrics.unique_failure_signatures)} tone="red" />
    <MetricCard label="Customer Candidates" value={fmtNumber((run?.customer_candidates || []).reduce((sum, item) => sum + (item.candidate_count || 0), 0))} tone="blue" />
    <MetricCard label="Service Proposals" value={fmtNumber(repair.total_service_proposals)} />
    <MetricCard label="Accepted Repairs" value={fmtNumber(repair.accepted_service_proposals)} tone="green" />
    <MetricCard label="Acceptance Rate" value={fmtPercent(repair.service_acceptance_rate)} tone="green" />
    <MetricCard label="Requests" value={fmtNumber(metrics.requests)} />
    <MetricCard label="Tokens" value={fmtNumber(metrics.tokens)} detail={`${fmtLatency(metrics.latency_seconds)} total`} />
  </div>
}

function SummaryCard({ label, children, tone = 'neutral' }) {
  return <div className={`summary-card ${tone}`}><div className="eyebrow">{label}</div><div className="summary-card-body">{children}</div></div>
}

function ComparisonChart({ runs }) {
  const data = useMemo(() => {
    const generations = [...new Set(runs.flatMap((run) => (run.generation_metrics || []).map((item) => item.generation)))].sort((a, b) => a - b)
    return generations.map((generation) => {
      const row = { generation: `G${generation}` }
      runs.forEach((run) => { const metric = (run.generation_metrics || []).find((item) => item.generation === generation); row[run.id] = metric?.task_success == null ? null : metric.task_success * 100 })
      return row
    })
  }, [runs])
  const palette = [COLORS.gray, COLORS.blue, COLORS.green, COLORS.orange]
  return <div className="chart-card"><div className="chart-card-header"><div><div className="eyebrow">Comparison</div><h3>Task success by generation</h3></div><span className="chart-note">trajectory, not independent samples</span></div><div className="chart-wrap"><ResponsiveContainer width="100%" height="100%"><LineChart data={data} margin={{ top: 8, right: 12, left: -22, bottom: 0 }}><CartesianGrid stroke="#edf0f3" vertical={false} /><XAxis dataKey="generation" tick={{ fontSize: 12, fill: '#7b8794' }} axisLine={false} tickLine={false} /><YAxis domain={[0, 100]} tickFormatter={(value) => `${value}%`} tick={{ fontSize: 12, fill: '#7b8794' }} axisLine={false} tickLine={false} /><Tooltip formatter={(value) => value == null ? 'Not evaluated' : `${Number(value).toFixed(1)}%`} contentStyle={{ border: '1px solid #e1e6eb', borderRadius: 6, fontSize: 12 }} />{runs.map((run, index) => <Line key={run.id} type="monotone" dataKey={run.id} name={run.id} stroke={palette[index % palette.length]} strokeWidth={2} dot={{ r: 3 }} connectNulls />)}</LineChart></ResponsiveContainer></div><div className="legend-row">{runs.map((run, index) => <span key={run.id}><i style={{ background: palette[index % palette.length] }} />{run.id}</span>)}</div></div>
}

function RobustnessSummary({ run }) {
  const robustness = run?.robustness || {}
  const finalCustomer = run?.customer_candidates?.at(-1)
  const selectedId = finalCustomer?.selected_policy_id || finalCustomer?.selected_policy?.policy_id
  const fitness = finalCustomer?.scores?.find((item) => item.policy_id === selectedId)?.fitness
  const repair = serviceStats(run)
  return <div className="summary-grid"><SummaryCard label="Customer evolution" tone="customer"><strong>{fmtPercent(fitness)}</strong><span>selected fitness</span></SummaryCard><SummaryCard label="Service evolution" tone="service"><strong>{fmtNumber(repair.total_service_proposals)}</strong><span>{fmtNumber(repair.accepted_service_proposals)} accepted</span></SummaryCard><SummaryCard label="Robustness" tone="green"><strong>{fmtPercent(robustness.latest)}</strong><span>latest adversary</span></SummaryCard><SummaryCard label="Protocol health" tone="orange"><strong>{fmtPercent(run?.metrics?.invalid_rate == null ? null : 1 - run.metrics.invalid_rate)}</strong><span>{fmtNumber(run?.metrics?.retries)} retries · {fmtNumber(run?.metrics?.provider_failures)} provider failures</span></SummaryCard></div>
}

function traceToolEvents(episode) {
  return (episode?.trace_events || []).filter((event) => event.kind === 'tool_call')
}

function firstToolResult(episode, toolEvent) {
  const events = episode?.trace_events || []
  const index = events.indexOf(toolEvent)
  return events.find((event, eventIndex) => event.kind === 'backend_result' && eventIndex > index)
}

function FeaturedCase({ run, onSelect }) {
  const failures = run?.episodes?.filter((episode) => episode.status === 'LEGITIMATE FAILURE') || []
  const episode = failures.find((item) => traceToolEvents(item).length && run?.service_candidates?.some((generation) => (generation.candidates || []).some((candidate) => (candidate.source_failure_ids || []).includes(item.failure_signature?.signature_id)))) || failures.find((item) => traceToolEvents(item).length) || failures[0]
  if (!episode) return <div className="empty-state">No legitimate failure available for a featured case.</div>
  const toolEvent = traceToolEvents(episode)[0]
  const resultEvent = firstToolResult(episode, toolEvent)
  const args = toolEvent?.payload && Object.prototype.hasOwnProperty.call(toolEvent.payload, 'arguments') ? toolEvent.payload.arguments : null
  const result = resultEvent?.payload || {}
  const proposal = run?.service_candidates?.flatMap((item) => item.candidates || []).find((candidate) => (candidate.source_failure_ids || []).includes(episode.failure_signature?.signature_id))
  const patch = proposal?.patch || {}
  const rule = patch.rules?.[0] || patch
  const status = proposal ? (proposal.evaluation_status === 'invalid' ? 'INVALID' : proposal.accepted ? 'ACCEPTED' : 'REJECTED') : 'NOT RECORDED'
  const backendOutcome = result.error_code || result.error || result.status || result.message || 'Backend result not recorded'
  return <div className="featured-case"><div className="featured-intro"><StatusBadge kind="failure">LEGITIMATE FAILURE</StatusBadge><span className="muted">{episode.case_id} · G{episode.generation}</span></div><div className="featured-flow featured-flow-wide"><div className="flow-step blue-step"><span>Customer</span><strong>{display(episode.case_metadata?.user_intent, 'Not recorded')}</strong></div><div className="flow-arrow">↓</div><div className="flow-step"><span>Agent</span><strong>{display(episode.predicted_action, 'Not recorded')}</strong></div><div className="flow-arrow">↓</div><div className="flow-step tool-step"><span>Tool call</span><strong>{display(toolEvent?.payload?.name, 'Tool call not recorded')}</strong><code>{toolEvent ? (args == null ? 'Arguments not recorded' : JSON.stringify(args, null, 2)) : 'No tool trace'}</code></div><div className="flow-arrow">↓</div><div className="flow-step backend-step"><span>Backend</span><strong>{backendOutcome}</strong></div><div className="flow-arrow">↓</div><div className="flow-step failure-step"><span>Failure</span><strong>{display(episode.failure_signature?.signature_id || episode.termination_reason)}</strong></div></div>{proposal && <div className="proposal-strip"><span>Service proposal</span><strong>{display(rule.text || rule.rule_text || patch.text || patch.rule_text, 'Repair text not recorded')}</strong><StatusBadge kind={statusTone(status.toLowerCase())}>{status}</StatusBadge><span className="muted">delta {proposal.delta == null ? '—' : Number(proposal.delta).toFixed(3)} · {display(proposal.reason, 'gate reason not recorded')}</span></div>}<button className="text-button" onClick={() => onSelect(episode)}>Open episode detail →</button></div>
}

function FailureList({ episodes, onSelect }) {
  const failures = episodes.filter((item) => item.status !== 'VALID SUCCESS')
  if (!failures.length) return <div className="empty-state">No failure episodes in this run.</div>
  return <div className="failure-list">{failures.slice(0, 7).map((episode) => { const invalid = episode.status === 'INVALID EVALUATION'; return <button className="failure-row" key={episode.episode_id} onClick={() => onSelect(episode)}><div className={`failure-mark ${invalid ? 'orange' : 'red'}`} /><div className="failure-row-main"><div className="failure-row-title">{invalid ? episode.invalid_reason : episode.failure_signature?.signature_id || episode.error_types?.[0] || 'legitimate failure'}</div><div className="muted">{episode.case_id} · G{episode.generation} · {episode.sop_node || 'SOP node unknown'}</div></div><StatusBadge kind={invalid ? 'invalid' : 'failure'}>{invalid ? 'INVALID' : 'LEGITIMATE'}</StatusBadge></button> })}</div>
}

function IntegrityChecklist({ run, compact = false }) {
  const metadata = run?.metadata || {}
  const freezeStatus = metadata.runtime_freeze_match_status
  const freezeValue = freezeStatus === 'matched' ? true : freezeStatus === 'mismatch' ? false : null
  const freezeDetail = [metadata.runtime_freeze_commit || 'commit not recorded', metadata.runtime_freeze_tag || 'tag not recorded', freezeStatus ? `status: ${freezeStatus}` : 'status: not confirmed'].join(' · ')
  const checks = [['Runtime freeze', freezeValue, freezeDetail], ['Formal protocol', metadata.formal_protocol_commit || metadata.formal_protocol_tag ? true : null, [metadata.formal_protocol_commit || 'commit not recorded', metadata.formal_protocol_tag || 'tag not recorded'].join(' · ')], ['Split manifest', run?.completeness?.split_manifest, 'artifact present'], ['Valid / invalid separated', run?.completeness?.valid_invalid_separated, 'from structured status'], ['Candidate provenance', run?.completeness?.candidate_provenance, run?.completeness?.candidate_provenance == null ? 'not applicable' : 'patch + policy records'], ['Tool traces', run?.completeness?.traces, 'display-only provenance'], ['Held-out isolated', run?.completeness?.heldout_not_used_by_evolver, 'not confirmed from artifacts']]
  return <div className={`checklist ${compact ? 'compact' : ''}`}>{checks.map(([label, value, detail]) => <div className="check-row" key={label}><span className={`check-icon ${value === true ? 'ok' : value === false ? 'bad' : 'unknown'}`}>{value === true ? '✓' : value === false ? '!' : '·'}</span><div><strong>{label}</strong><span>{value === true ? 'Confirmed' : value === false ? 'Mismatch / missing' : 'Not confirmed'} · {detail}</span></div></div>)}</div>
}

function Overview({ run, runs, onNavigate, onSelectEpisode }) {
  return <><SectionHeader eyebrow="Project overview" title="Evaluation health" action={<span className="muted">Read-only artifact view</span>} /><MetricGrid run={run} /><RobustnessSummary run={run} /><div className="two-column overview-grid"><ComparisonChart runs={runs} /><div className="panel"><SectionHeader eyebrow="Featured case" title="Failure → repair" /><FeaturedCase run={run} onSelect={onSelectEpisode} /></div></div><div className="two-column overview-grid lower"><div className="panel"><SectionHeader eyebrow="Recent" title="Failures" action={<button className="text-button" onClick={() => onNavigate('failures')}>View analysis →</button>} /><FailureList episodes={run.episodes || []} onSelect={onSelectEpisode} /></div><div className="panel"><SectionHeader eyebrow="Integrity" title="Run completeness" action={<button className="text-button" onClick={() => onNavigate('diagnostics')}>Inspect →</button>} /><IntegrityChecklist run={run} compact /></div></div></>
}

function Experiments({ runs }) {
  const [filters, setFilters] = useState({ mode: 'all', seed: 'all', source: 'all', status: 'all' })
  const filtered = runs.filter((run) => (filters.mode === 'all' || run.metadata?.mode === filters.mode) && (filters.seed === 'all' || String(run.metadata?.seed) === filters.seed) && (filters.source === 'all' || run.metadata?.source_kind === filters.source) && (filters.status === 'all' || run.metadata?.run_status === filters.status))
  const options = (key) => [...new Set(runs.map((run) => key === 'mode' ? run.metadata?.mode : key === 'seed' ? String(run.metadata?.seed) : key === 'source' ? run.metadata?.source_kind : run.metadata?.run_status).filter(Boolean))]
  return <><SectionHeader eyebrow="Experiments / Evaluations" title="Compare runs" action={<span className="muted">Generations are trajectory steps, not independent samples.</span>} /><div className="filter-bar compact-filters">{[['mode', 'Mode'], ['seed', 'Seed'], ['source', 'Source'], ['status', 'Status']].map(([key, label]) => <label key={key}>{label}<select value={filters[key]} onChange={(event) => setFilters({ ...filters, [key]: event.target.value })}><option value="all">All</option>{options(key).map((value) => <option key={value} value={value}>{value}</option>)}</select></label>)}</div><div className="panel compare-panel"><div className="table-wrap"><table><thead><tr><th>Run</th><th>Status</th><th>Task</th><th>Action</th><th>Goal</th><th>Legit failures</th><th>Unique failures</th><th>Latest</th><th>Replay</th><th>Held-out</th><th>Fresh</th><th>Requests</th><th>Tokens</th><th>Latency</th></tr></thead><tbody>{filtered.map((run) => <tr key={run.id}><td><strong>{run.id}</strong><small>{run.metadata?.model} · seed {run.metadata?.seed}</small></td><td><StatusBadge kind={statusTone(run.metadata?.run_status)}>{display(run.metadata?.run_status, 'not recorded')}</StatusBadge></td><td>{fmtPercent(run.metrics?.task_success)}</td><td>{fmtPercent(run.metrics?.action_execution)}</td><td>{fmtPercent(run.metrics?.goal_fulfillment)}</td><td>{fmtPercent(run.metrics?.legitimate_failure_rate)}</td><td>{fmtNumber(run.metrics?.unique_failure_signatures)}</td><td>{fmtPercent(run.robustness?.latest)}</td><td>{fmtPercent(run.robustness?.replay)}</td><td>{run.robustness?.heldout ? fmtPercent(run.robustness.heldout.task_success) : 'Not evaluated'}</td><td>{run.robustness?.fresh_adversary ? fmtPercent(run.robustness.fresh_adversary.task_success) : 'Not evaluated'}</td><td>{fmtNumber(run.metrics?.requests)}</td><td>{fmtNumber(run.metrics?.tokens)}</td><td>{fmtLatency(run.metrics?.latency_seconds)}</td></tr>)}</tbody></table>{!filtered.length && <div className="empty-state">No runs match these filters.</div>}</div></div><div className="two-column lower"><div className="chart-card"><div className="chart-card-header"><div><div className="eyebrow">Performance</div><h3>Task success by run</h3></div></div><div className="chart-wrap tall"><ResponsiveContainer width="100%" height="100%"><BarChart data={filtered.map((run) => ({ name: run.id, task: run.metrics?.task_success == null ? null : run.metrics.task_success * 100, action: run.metrics?.action_execution == null ? null : run.metrics.action_execution * 100, goal: run.metrics?.goal_fulfillment == null ? null : run.metrics.goal_fulfillment * 100 }))} margin={{ top: 8, right: 12, left: -22, bottom: 0 }}><CartesianGrid stroke="#edf0f3" vertical={false} /><XAxis dataKey="name" tick={{ fontSize: 11, fill: '#7b8794' }} axisLine={false} tickLine={false} /><YAxis domain={[0, 100]} tickFormatter={(value) => `${value}%`} tick={{ fontSize: 12, fill: '#7b8794' }} axisLine={false} tickLine={false} /><Tooltip formatter={(value) => value == null ? 'Not evaluated' : `${Number(value).toFixed(1)}%`} /><Bar dataKey="task" name="Task Success" fill={COLORS.green} /><Bar dataKey="action" name="Action Execution" fill={COLORS.blue} /><Bar dataKey="goal" name="Goal Fulfillment" fill="#8ba4d8" /></BarChart></ResponsiveContainer></div></div><div className="panel"><SectionHeader eyebrow="Service evolution" title="Proposal outcomes" /><CandidateCounts runs={filtered} /></div></div><div className="panel lower"><SectionHeader eyebrow="Protocol / cost" title="Operational footprint" /><CostTable runs={filtered} /></div></>
}

function CandidateCounts({ runs }) {
  return <div className="cost-list">{runs.map((run) => { const stats = serviceStats(run); return <div className="cost-row" key={run.id}><div><strong>{run.id}</strong><span>{fmtNumber(stats.total_service_proposals)} proposals · {fmtNumber(stats.accepted_service_proposals)} accepted · {fmtNumber(stats.rejected_service_proposals)} rejected · {fmtNumber(stats.invalid_service_proposals)} invalid</span></div><b>{fmtPercent(stats.service_acceptance_rate)}</b></div> })}</div>
}

function CostTable({ runs }) {
  return <div className="table-wrap"><table><thead><tr><th>Run</th><th>Valid</th><th>Invalid</th><th>Provider failures</th><th>Timeouts</th><th>Retries</th><th>Attempts</th><th>Input tokens</th><th>Output tokens</th><th>Latency</th></tr></thead><tbody>{runs.map((run) => <tr key={run.id}><td><strong>{run.id}</strong></td><td>{fmtPercent(run.metrics?.valid_episode_rate)}</td><td>{fmtPercent(run.metrics?.invalid_rate)}</td><td>{fmtNumber(run.metrics?.provider_failures)}</td><td>{fmtNumber(run.metrics?.timeouts)}</td><td>{fmtNumber(run.metrics?.retries)}</td><td>{fmtNumber(run.metrics?.attempts)}</td><td>{fmtNumber(run.metrics?.input_tokens)}</td><td>{fmtNumber(run.metrics?.output_tokens)}</td><td>{fmtLatency(run.metrics?.latency_seconds)}</td></tr>)}</tbody></table></div>
}

function PolicyCard({ policy, customer = false, selected = false, elite = false }) {
  if (!policy || Object.keys(policy).length === 0) return <div className="empty-inline">No policy record</div>
  const title = policy.name || policy.policy_id || policy.candidate_policy_id || (customer ? 'Customer policy' : 'Service policy')
  const tags = policy.strategy_tags || policy.rule_categories || []
  return <div className={`policy-card ${customer ? 'customer-card' : 'service-card'}`}><div className="policy-card-top"><StatusBadge kind={customer ? 'customer' : 'accepted'}>{customer ? 'CUSTOMER' : 'SERVICE'}</StatusBadge>{selected && <StatusBadge kind="success">SELECTED</StatusBadge>}{elite && <StatusBadge kind="customer">ELITE</StatusBadge>}<strong>{title}</strong></div><div className="policy-id">{policy.policy_id || policy.candidate_policy_id || 'policy id not recorded'}</div>{policy.description && <p>{policy.description}</p>}{policy.rationale && <p>{policy.rationale}</p>}{tags.length > 0 && <div className="tag-row">{tags.map((tag) => <span key={tag}>{tag}</span>)}</div>}</div>
}

function ServiceCandidateCard({ candidate }) {
  const patch = candidate.patch || {}
  const rules = patch.rules || []
  const ruleText = rules.map((rule) => `${rule.category || rule.rule_category || 'RULE'}: ${rule.text || rule.rule_text || 'text not recorded'}`).join('\n') || patch.text || patch.rule_text || 'Repair text not recorded'
  const candidateStatus = candidate.evaluation_status || (candidate.accepted ? 'accepted' : 'rejected')
  const gateStatus = candidateStatus === 'invalid' || candidateStatus === 'inconclusive' ? candidateStatus : candidate.accepted ? 'accepted' : 'rejected'
  return <div className="candidate-card"><div className="candidate-card-header"><strong>{display(candidate.patch_id || patch.patch_id, 'patch id not recorded')}</strong><StatusBadge kind={statusTone(gateStatus)}>{gateStatus.toUpperCase()}</StatusBadge></div><pre className="patch-preview">{ruleText}</pre><div className="candidate-meta"><span>category {display(rules[0]?.category || rules[0]?.rule_category || patch.category || patch.rule_category)}</span><span>delta {candidate.delta == null ? '—' : Number(candidate.delta).toFixed(3)}</span><span>{candidate.latest_filter_rejected ? 'latest-only evaluation' : candidate.evaluation_scope || 'evaluation scope not recorded'}</span></div><div className="candidate-meta"><span>source failures {fmtNumber((candidate.source_failure_ids || patch.source_failure_ids || []).length)}</span><span>{display(candidate.reason, 'gate reason not recorded')}</span></div>{candidate.latest_filter_rejected && <div className="small-note">Rejected before replay/normal evaluation.</div>}</div>
}

function FailureLane({ run, generation }) {
  const item = run.failure_analysis?.trajectory?.find((entry) => entry.generation === generation)
  if (!item) return <div className="lane-empty">No failure signature artifact for this generation.</div>
  return <div className="failure-lane-content"><div className="failure-counts"><span className="new-count">new {item.new_count}</span><span className="repeat-count">repeated {item.repeated_count}</span></div>{item.new_signatures.length > 0 && <div><div className="eyebrow">New signatures</div><div className="signature-list">{item.new_signatures.map((id) => <span key={id}>{id}</span>)}</div></div>}{item.repeated_signatures.length > 0 && <div><div className="eyebrow">Repeated signatures</div><div className="signature-list repeated">{item.repeated_signatures.map((id) => <span key={id}>{id}</span>)}</div></div>}{!item.new_signatures.length && !item.repeated_signatures.length && <div className="muted">No failure signatures recorded.</div>}</div>
}

function GenerationSection({ run, generation }) {
  const customer = run.customer_candidates?.find((item) => item.generation === generation)
  const service = run.service_candidates?.find((item) => item.generation === generation)
  const selected = customer?.selected_policy || {}
  const selectedScore = customer?.scores?.find((item) => item.policy_id === (customer.selected_policy_id || selected.policy_id)) || {}
  const candidateRows = customer?.candidates || []
  const allService = service?.candidates || []
  return <section className="generation-section"><div className="generation-marker"><span>G{generation}</span><i /></div><div className="generation-content"><div className="generation-heading"><h3>Generation {generation}</h3><span className="muted">candidate evaluation snapshot</span></div><div className="evolution-three-lanes"><div className="lane customer-lane"><div className="lane-label">CUSTOMER</div><PolicyCard customer policy={selected} selected /><div className="policy-meta"><span>fitness <b>{fmtPercent(selectedScore.fitness)}</b></span><span>attack <b>{fmtPercent(selectedScore.attack_success)}</b></span><span>novelty <b>{fmtPercent(selectedScore.novelty)}</b></span><span>coverage <b>{fmtPercent(selectedScore.coverage)}</b></span></div><div className="candidate-list">{candidateRows.map((candidate) => <div className="candidate-row" key={candidate.policy_id || candidate.candidate_index}><span>{candidate.selected ? <StatusBadge kind="success">SELECTED</StatusBadge> : candidate.elite ? <StatusBadge kind="customer">ELITE</StatusBadge> : <StatusBadge kind="neutral">CANDIDATE</StatusBadge>}</span><strong>{display(candidate.policy_id, `candidate ${candidate.candidate_index ?? '—'}`)}</strong><span>{fmtPercent(candidate.score?.fitness)}</span></div>)}</div>{customer?.incumbent_policy_id && <div className="incumbent-note">{customer.incumbent_policy_id === customer.selected_policy_id ? 'INCUMBENT RETAINED' : 'INCUMBENT REPLACED'} · {customer.incumbent_policy_id}</div>}</div><div className="lane failure-lane"><div className="lane-label">FAILURE SURFACE</div><FailureLane run={run} generation={generation} /></div><div className="lane service-lane"><div className="lane-label">SERVICE CANDIDATES</div><div className="gate-stack">{allService.length ? allService.map((candidate, index) => <ServiceCandidateCard candidate={candidate} key={candidate.patch_id || index} />) : <div className="lane-empty">No Service candidate artifact for this generation.</div>}</div><div className="lane-footer">{allService.length} candidate record(s) · gate {display(service?.reason, 'decision not recorded')}</div></div></div></div></section>
}

function Evolution({ run }) {
  const generations = [...new Set([...(run?.customer_candidates || []).map((item) => item.generation), ...(run?.service_candidates || []).map((item) => item.generation), ...(run?.failure_analysis?.trajectory || []).map((item) => item.generation)])].sort((a, b) => a - b)
  if (!generations.length) return <EmptyPage title="Evolution" message="No generation artifacts were found for this run." />
  return <><SectionHeader eyebrow="Customer ↔ Failure ↔ Service" title="Evolution timeline" action={<StatusBadge kind={sourceTone(run.metadata?.source_kind)}>{run.metadata?.source_kind}</StatusBadge>} /><div className="timeline-page-note">Customer strategy <span>→</span> exposed failure <span>→</span> Service patch <span>→</span> gate decision. Candidate metrics are read from structured artifacts.</div><div className="generation-timeline">{generations.map((generation) => <GenerationSection key={generation} run={run} generation={generation} />)}</div></>
}

function FilterBar({ filters, setFilters, episodes }) {
  const options = (key) => [...new Set(episodes.map((item) => item[key]).filter(Boolean))].sort()
  const fields = [['generation', 'Generation'], ['split', 'Split'], ['status', 'Status'], ['error', 'Error type'], ['sop_node', 'SOP node'], ['customer_policy_id', 'Customer policy'], ['service_policy_id', 'Service policy']]
  return <div className="filter-bar">{fields.map(([key, label]) => <label key={key}>{label}<select value={filters[key]} onChange={(event) => setFilters({ ...filters, [key]: event.target.value })}><option value="all">All</option>{(key === 'error' ? [...new Set(episodes.flatMap((item) => item.error_types || []))].sort() : options(key)).map((value) => <option key={value} value={value}>{value}</option>)}</select></label>)}</div>
}

function EpisodeTable({ episodes, onSelect }) {
  return <div className="table-wrap"><table className="episodes-table"><thead><tr><th>Status</th><th>Case</th><th>Gen / split</th><th>Customer policy</th><th>Service policy</th><th>Task success</th><th>Error</th><th>Tools</th><th>Latency</th></tr></thead><tbody>{episodes.map((episode) => <tr key={episode.episode_id} onClick={() => onSelect(episode)}><td><StatusBadge kind={statusTone(episode.status)}>{episode.status === 'INVALID EVALUATION' ? 'INVALID' : episode.status === 'LEGITIMATE FAILURE' ? 'FAILURE' : 'SUCCESS'}</StatusBadge></td><td><strong>{episode.case_id}</strong><small>{episode.episode_id}</small></td><td>G{episode.generation} · {episode.split}</td><td>{display(episode.customer_policy_id, '—')}</td><td>{display(episode.service_policy_id, '—')}</td><td>{episode.task_success == null ? '—' : episode.task_success ? 'Yes' : 'No'}</td><td>{episode.invalid_reason || episode.error_types?.[0] || '—'}</td><td>{episode.tool_sequence?.length || 0}</td><td>{fmtLatency(episode.latency_seconds)}</td></tr>)}</tbody></table>{!episodes.length && <div className="empty-state">No episodes match these filters.</div>}</div>
}

function Episodes({ run, onSelectEpisode }) {
  const [filters, setFilters] = useState({ generation: 'all', split: 'all', status: 'all', error: 'all', sop_node: 'all', customer_policy_id: 'all', service_policy_id: 'all' })
  const episodes = run?.episodes || []
  const shown = episodes.filter((episode) => Object.entries(filters).every(([key, value]) => value === 'all' || (key === 'error' ? episode.error_types?.includes(value) : String(episode[key]) === value)))
  return <><SectionHeader eyebrow="Projects / Traces" title="Episodes" action={<span className="muted">{shown.length} of {episodes.length} episodes</span>} /><FilterBar filters={filters} setFilters={setFilters} episodes={episodes} /><div className="panel episodes-panel"><EpisodeTable episodes={shown} onSelect={onSelectEpisode} /></div></>
}

function FailureExplorer({ run, onSelectEpisode }) {
  const [selected, setSelected] = useState(null)
  const analysis = run.failure_analysis || { signatures: [], trajectory: [] }
  const current = selected || analysis.signatures[0]
  return <><SectionHeader eyebrow="Failure analysis" title="Weakness frontier" action={<span className="muted">legitimate failures only; invalid evaluations excluded</span>} /><div className="two-column failure-analysis-grid"><div className="panel"><div className="table-wrap"><table><thead><tr><th>Signature</th><th>Count</th><th>First</th><th>Last</th><th>New / repeated</th></tr></thead><tbody>{analysis.signatures.map((item) => <tr className={current?.signature_id === item.signature_id ? 'selected-row' : ''} key={item.signature_id} onClick={() => setSelected(item)}><td><strong>{item.signature_id}</strong><small>{item.sop_nodes?.join(', ') || 'SOP node not recorded'}</small></td><td>{item.count}</td><td>G{item.first_seen_generation}</td><td>G{item.last_seen_generation}</td><td>{item.repeated ? 'Repeated' : 'New'}</td></tr>)}</tbody></table>{!analysis.signatures.length && <div className="empty-state">No legitimate failure signatures are available.</div>}</div></div><div className="panel">{current ? <><SectionHeader eyebrow="Signature detail" title={current.signature_id} /><div className="detail-meta-grid"><Meta label="Error category" value={current.error_types?.join(', ')} /><Meta label="SOP node" value={current.sop_nodes?.join(', ')} /><Meta label="Tool" value={current.tools?.join(', ')} /><Meta label="Count" value={current.count} /><Meta label="First seen" value={`G${current.first_seen_generation}`} /><Meta label="Last seen" value={`G${current.last_seen_generation}`} /><Meta label="Customer policies" value={current.customer_policy_ids?.join(', ')} /><Meta label="Episodes" value={current.representative_episode_ids?.join(', ')} /></div><div className="detail-section"><div className="eyebrow">Representative episodes</div><div className="signature-list">{current.representative_episode_ids?.map((id) => { const episode = run.episodes.find((item) => item.episode_id === id); return <button className="text-button" key={id} onClick={() => episode && onSelectEpisode(episode)}>{id} →</button> })}</div></div></> : <div className="empty-state">Select a failure signature.</div>}</div></div><div className="panel lower"><SectionHeader eyebrow="Novelty trajectory" title="New versus repeated signatures" /><div className="trajectory-list">{analysis.trajectory?.map((item) => <div className="trajectory-row" key={item.generation}><strong>G{item.generation}</strong><span className="new-count">new {item.new_count}</span><span className="repeat-count">repeated {item.repeated_count}</span><span className="muted">{[...(item.new_signatures || []), ...(item.repeated_signatures || [])].join(' · ') || 'none recorded'}</span></div>)}</div></div></>
}

function Robustness({ runs }) {
  const data = runs.map((run) => ({ name: run.id, latest: run.robustness?.latest == null ? null : run.robustness.latest * 100, replay: run.robustness?.replay == null ? null : run.robustness.replay * 100, normal: run.robustness?.normal == null ? null : run.robustness.normal * 100, heldout: run.robustness?.heldout?.task_success == null ? null : run.robustness.heldout.task_success * 100, fresh: run.robustness?.fresh_adversary?.task_success == null ? null : run.robustness.fresh_adversary.task_success * 100 }))
  return <><SectionHeader eyebrow="Generalization checks" title="Robustness" action={<span className="muted">missing artifact = Not evaluated, not zero</span>} /><div className="panel compare-panel"><div className="table-wrap"><table><thead><tr><th>Run</th><th>Latest adversary</th><th>Historical replay</th><th>Normal user</th><th>Held-out</th><th>Fresh adversary</th></tr></thead><tbody>{runs.map((run) => <tr key={run.id}><td><strong>{run.id}</strong><small>{run.metadata?.mode} · seed {run.metadata?.seed}</small></td><td>{fmtPercent(run.robustness?.latest)}</td><td>{fmtPercent(run.robustness?.replay)}</td><td>{fmtPercent(run.robustness?.normal)}</td><td>{run.robustness?.heldout ? fmtPercent(run.robustness.heldout.task_success) : 'Not evaluated'}</td><td>{run.robustness?.fresh_adversary ? fmtPercent(run.robustness.fresh_adversary.task_success) : 'Not evaluated'}</td></tr>)}</tbody></table></div></div><div className="chart-card lower"><div className="chart-card-header"><div><div className="eyebrow">Available aggregate metrics</div><h3>Robustness by run</h3></div></div><div className="chart-wrap tall"><ResponsiveContainer width="100%" height="100%"><BarChart data={data} margin={{ top: 8, right: 12, left: -22, bottom: 0 }}><CartesianGrid stroke="#edf0f3" vertical={false} /><XAxis dataKey="name" tick={{ fontSize: 11, fill: '#7b8794' }} axisLine={false} tickLine={false} /><YAxis domain={[0, 100]} tickFormatter={(value) => `${value}%`} tick={{ fontSize: 12, fill: '#7b8794' }} axisLine={false} tickLine={false} /><Tooltip formatter={(value) => value == null ? 'Not evaluated' : `${Number(value).toFixed(1)}%`} /><Bar dataKey="latest" name="Latest" fill={COLORS.blue} /><Bar dataKey="replay" name="Replay" fill="#6f8fca" /><Bar dataKey="normal" name="Normal" fill={COLORS.green} /><Bar dataKey="heldout" name="Held-out" fill="#9aa4b2" /><Bar dataKey="fresh" name="Fresh" fill={COLORS.orange} /></BarChart></ResponsiveContainer></div></div><div className="panel lower"><SectionHeader eyebrow="Interpretation guardrail" title="What this page does not claim" /><p className="plain-note">Robustness values are shown only when the structured artifact provides them. Generations are trajectory steps, not independent samples, and this page performs no statistical inference.</p></div></>
}

function TraceEvent({ event }) {
  const payload = event.payload
  const value = typeof payload === 'string' ? payload : payload == null ? '—' : JSON.stringify(payload, null, 2)
  return <div className={`trace-event ${event.kind || 'agent'}`}><div className="trace-dot" /><div className="trace-event-content"><div className="trace-event-header"><strong>{event.label || event.kind?.toUpperCase()}</strong>{event.turn != null && <span>turn {event.turn}</span>}</div><pre>{value}</pre></div></div>
}

function EpisodeDetail({ episode, onClose }) {
  if (!episode) return null
  const invalid = episode.status === 'INVALID EVALUATION'
  const tools = traceToolEvents(episode)
  return <div className="detail-panel"><div className="detail-header"><div><StatusBadge kind={statusTone(episode.status)}>{episode.status}</StatusBadge><h2>{episode.episode_id}</h2></div><button className="close-button" aria-label="Close episode detail" onClick={onClose}>×</button></div><div className="detail-meta-grid"><Meta label="Case" value={episode.case_id} /><Meta label="Generation" value={`G${episode.generation}`} /><Meta label="Split" value={episode.split} /><Meta label="Customer policy" value={episode.customer_policy_id} /><Meta label="Service policy" value={episode.service_policy_id} /><Meta label="Expected action" value={episode.expected_action} /><Meta label="Predicted action" value={episode.predicted_action} /><Meta label="Executed action" value={episode.executed_action} /><Meta label="Termination" value={episode.termination_reason} /><Meta label="Error" value={invalid ? episode.invalid_reason : episode.error_types?.join(', ')} /></div><div className="score-strip">{Object.entries(episode.scores || {}).map(([key, value]) => <div key={key}><span>{key.replaceAll('_', ' ')}</span><b>{value == null ? '—' : Number(value).toFixed(2)}</b></div>)}</div><div className="detail-section"><SectionHeader eyebrow="Trace detail" title="Execution timeline" action={<span className="muted">display-only provenance</span>} />{episode.trace_events?.length ? <div className="trace-timeline">{episode.trace_events.map((event, index) => <TraceEvent key={event.id || index} event={event} />)}</div> : <div className="empty-state">No trace artifact was exported for this episode.</div>}</div><div className="detail-section"><SectionHeader eyebrow="Tool summary" title="Tool sequence" />{tools.length ? <div className="tool-chips">{tools.map((event, index) => <span key={`${event.id || event.payload?.name}-${index}`}><b>{index + 1}</b>{event.payload?.name || 'tool name not recorded'}</span>)}</div> : <div className="muted">No tool calls recorded.</div>}</div><div className="detail-footer"><span>trace ref: {display(episode.trace_ref, 'not recorded')}</span><span>model: {display(episode.provenance?.model, 'not recorded')}</span></div></div>
}

function Meta({ label, value }) { return <div className="meta-cell"><span>{label}</span><strong>{display(value, '—')}</strong></div> }

function InvalidBreakdown({ episodes }) {
  const counts = {}
  episodes.filter((item) => item.status === 'INVALID EVALUATION').forEach((item) => { const reason = item.invalid_reason || 'evaluation_invalid'; counts[reason] = (counts[reason] || 0) + 1 })
  return <div className="signature-list">{Object.entries(counts).map(([reason, count]) => <span key={reason}>{reason} · {count}</span>)}{!Object.keys(counts).length && <span>None recorded</span>}</div>
}

function ArtifactStatuses({ run }) {
  const artifacts = run?.completeness?.artifacts || {}
  return <div className="artifact-grid">{Object.entries(artifacts).map(([name, value]) => <div className="artifact-row" key={name}><span>{name.replaceAll('_', ' ')}</span><StatusBadge kind={value === 'confirmed' ? 'success' : value === 'missing' ? 'failure' : 'neutral'}>{value.replaceAll('_', ' ')}</StatusBadge></div>)}</div>
}

function Diagnostics({ run }) {
  const metrics = run?.metrics || {}
  const metadata = run?.metadata || {}
  return <><SectionHeader eyebrow="Observability" title="Diagnostics & reproducibility" action={<StatusBadge kind={sourceTone(metadata.source_kind)}>{metadata.source_kind}</StatusBadge>} /><div className="metric-grid diagnostics-grid"><MetricCard label="Requests" value={fmtNumber(metrics.requests)} /><MetricCard label="Attempts" value={fmtNumber(metrics.attempts)} /><MetricCard label="Successes" value={fmtNumber(metrics.successes)} /><MetricCard label="Input tokens" value={fmtNumber(metrics.input_tokens)} /><MetricCard label="Output tokens" value={fmtNumber(metrics.output_tokens)} /><MetricCard label="Reasoning tokens" value={fmtNumber(metrics.reasoning_tokens)} /><MetricCard label="Latency" value={fmtLatency(metrics.latency_seconds)} detail={`avg ${fmtLatency(metrics.average_attempt_latency)} · max ${fmtLatency(metrics.max_attempt_latency)}`} /><MetricCard label="Retries" value={fmtNumber(metrics.retries)} /><MetricCard label="Timeouts" value={fmtNumber(metrics.timeouts)} tone={metrics.timeouts ? 'invalid' : 'neutral'} /><MetricCard label="Provider failures" value={fmtNumber(metrics.provider_failures)} tone={metrics.provider_failures ? 'invalid' : 'neutral'} /><MetricCard label="Invalid evaluations" value={fmtNumber(metrics.invalid_episodes)} tone={metrics.invalid_episodes ? 'invalid' : 'neutral'} /><MetricCard label="Run status" value={display(metadata.run_status, 'not recorded')} /></div><div className="two-column lower"><div className="panel"><SectionHeader eyebrow="Run identity" title="Manifest" /><div className="manifest-list"><Meta label="Output dir" value={metadata.run_dir} /><Meta label="Mode" value={metadata.mode} /><Meta label="Seed" value={metadata.seed} /><Meta label="Cases" value={metadata.case_count} /><Meta label="Runtime freeze commit" value={metadata.runtime_freeze_commit} /><Meta label="Runtime freeze tag" value={metadata.runtime_freeze_tag} /><Meta label="Freeze match status" value={metadata.runtime_freeze_match_status || 'Not confirmed'} /><Meta label="Formal protocol commit" value={metadata.formal_protocol_commit} /><Meta label="Formal protocol tag" value={metadata.formal_protocol_tag} /><Meta label="Protocol id" value={metadata.formal_protocol_id} /><Meta label="Model" value={metadata.model} /><Meta label="Provider" value={metadata.provider} /><Meta label="Split" value={metadata.split_strategy} /><Meta label="Split seed" value={metadata.split_seed} /><Meta label="Generations" value={metadata.generations} /><Meta label="Max turns" value={metadata.max_turns} /><Meta label="Judge in evolution" value={metadata.judge_in_evolution == null ? 'Not recorded' : String(metadata.judge_in_evolution)} /></div></div><div className="panel"><SectionHeader eyebrow="Protocol diagnostics" title="Invalid evaluation reasons" /><InvalidBreakdown episodes={run?.episodes || []} /><div className="detail-section"><SectionHeader eyebrow="Artifacts" title="Completeness" /><ArtifactStatuses run={run} /></div><div className="detail-section"><SectionHeader eyebrow="Integrity checklist" title="What artifacts can confirm" /><IntegrityChecklist run={run} /></div></div></div></>
}

function EmptyPage({ title, message }) { return <div className="empty-page"><div className="empty-illustration">⌁</div><h2>{title}</h2><p>{message}</p></div> }

function Sidebar({ page, onChange, run }) {
  return <aside className="sidebar"><div className="brand"><div className="brand-mark">E</div><div><strong>EvoSAGE</strong><span>Research dashboard</span></div></div><div className="sidebar-section-label">Workspace</div><nav>{NAV.map((item) => <button key={item.id} className={page === item.id ? 'active' : ''} onClick={() => onChange(item.id)}><span className="nav-icon">{item.icon}</span>{item.label}</button>)}</nav><div className="sidebar-spacer" /><div className="sidebar-footer"><div><StatusBadge kind={sourceTone(run?.metadata?.source_kind)}>{run?.metadata?.source_kind || 'UNKNOWN'}</StatusBadge></div><div className="freeze-label">{display(run?.metadata?.runtime_freeze_tag || run?.metadata?.freeze_tag, 'freeze not recorded')}</div><div className="muted small">read-only · no runtime controls</div></div></aside>
}

function App() {
  const [dataset, setDataset] = useState(null)
  const [loadError, setLoadError] = useState(null)
  const [page, setPage] = useState('overview')
  const [activeRunId, setActiveRunId] = useState('')
  const [selectedEpisode, setSelectedEpisode] = useState(null)
  useEffect(() => { fetch('./data/runs.json', { cache: 'no-store' }).then((response) => response.ok ? response.json() : Promise.reject(new Error('runs.json not found'))).then(setDataset).catch(() => fetch('./data/demo.json').then((response) => response.json()).then(setDataset).catch((error) => setLoadError(error.message))) }, [])
  const runs = dataset?.runs || []
  const activeRun = runs.find((run) => run.id === activeRunId) || runs.find((run) => run.id === 'coevolution') || runs[0]
  useEffect(() => { if (!activeRunId && activeRun?.id) setActiveRunId(activeRun.id) }, [activeRunId, activeRun])
  const navigate = (nextPage) => { setPage(nextPage); setSelectedEpisode(null) }
  const selectEpisode = (episode) => { setSelectedEpisode(episode); setPage('episodes') }
  if (loadError) return <div className="load-error">Could not load dashboard data: {loadError}</div>
  if (!dataset || !activeRun) return <div className="loading-screen"><div className="brand-mark">E</div><span>Loading artifact dataset…</span></div>
  const content = page === 'overview' ? <Overview run={activeRun} runs={runs} onNavigate={navigate} onSelectEpisode={selectEpisode} /> : page === 'experiments' ? <Experiments runs={runs} /> : page === 'evolution' ? <Evolution run={activeRun} /> : page === 'episodes' ? <Episodes run={activeRun} onSelectEpisode={setSelectedEpisode} /> : page === 'failures' ? <FailureExplorer run={activeRun} onSelectEpisode={setSelectedEpisode} /> : page === 'robustness' ? <Robustness runs={runs} /> : <Diagnostics run={activeRun} />
  return <div className="app-shell"><Sidebar page={page} onChange={navigate} run={activeRun} /><main className="main-shell"><ExperimentHeader run={activeRun} runs={runs} onChangeRun={(id) => { setActiveRunId(id); setSelectedEpisode(null) }} onNavigate={navigate} /><ExperimentTabs page={page} onChange={navigate} /><div className="content-shell">{content}</div>{selectedEpisode && <div className="detail-drawer"><EpisodeDetail episode={selectedEpisode} onClose={() => setSelectedEpisode(null)} /></div>}</main></div>
}

export default App
