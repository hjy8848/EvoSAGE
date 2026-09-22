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
  { id: 'diagnostics', label: 'Diagnostics', icon: '⌁' },
]

const COLORS = {
  blue: '#2563eb',
  green: '#16805b',
  red: '#c2413c',
  orange: '#c66a16',
  gray: '#697586',
}

function fmtPercent(value) {
  return value == null || Number.isNaN(Number(value)) ? '—' : `${(Number(value) * 100).toFixed(1)}%`
}

function fmtNumber(value) {
  return value == null || Number.isNaN(Number(value)) ? '—' : Number(value).toLocaleString()
}

function fmtLatency(value) {
  return value == null ? '—' : `${Number(value).toFixed(1)}s`
}

function display(value, fallback = 'Not recorded') {
  if (value == null || value === '') return fallback
  return String(value)
}

function sourceTone(source) {
  return source === 'REAL' ? 'real' : source === 'MOCK' ? 'mock' : 'demo'
}

function StatusBadge({ kind = 'neutral', children }) {
  return <span className={`status-badge ${kind}`}>{children}</span>
}

function MetricCard({ label, value, detail, tone = 'neutral' }) {
  return (
    <div className={`metric-card ${tone}`}>
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
      {detail && <div className="metric-detail">{detail}</div>}
    </div>
  )
}

function SectionHeader({ eyebrow, title, action }) {
  return (
    <div className="section-header">
      <div>
        {eyebrow && <div className="eyebrow">{eyebrow}</div>}
        <h2>{title}</h2>
      </div>
      {action}
    </div>
  )
}

function ExperimentHeader({ run, runs, onChangeRun, onNavigate }) {
  const metadata = run?.metadata || {}
  return (
    <header className="experiment-header">
      <div className="breadcrumbs"><span>EvoSAGE</span><b>/</b><span>Research</span><b>/</b><strong>{display(metadata.mode)}</strong></div>
      <div className="header-row">
        <div>
          <h1>{display(metadata.mode)} experiment</h1>
          <div className="header-subline">
            <StatusBadge kind={sourceTone(metadata.source_kind)}>{metadata.source_kind || 'UNKNOWN'}</StatusBadge>
            <span>{display(metadata.model)}</span>
            <span className="dot">·</span>
            <span>{display(metadata.provider)}</span>
            <span className="dot">·</span>
            <span>seed {display(metadata.seed)}</span>
          </div>
        </div>
        <div className="header-controls">
          <label className="run-select-label">Run
            <select value={run?.id || ''} onChange={(event) => onChangeRun(event.target.value)}>
              {runs.map((item) => <option key={item.id} value={item.id}>{item.id}</option>)}
            </select>
          </label>
          <button className="quiet-button" onClick={() => onNavigate('diagnostics')}>Reproducibility</button>
        </div>
      </div>
    </header>
  )
}

function ExperimentTabs({ page, onChange }) {
  return (
    <div className="tabs" role="tablist">
      {NAV.map((item) => (
        <button key={item.id} className={page === item.id ? 'active' : ''} onClick={() => onChange(item.id)}>
          {item.label}
        </button>
      ))}
    </div>
  )
}

function MetricGrid({ run }) {
  const metrics = run?.metrics || {}
  return (
    <div className="metric-grid">
      <MetricCard label="Task Success" value={fmtPercent(metrics.task_success)} tone="green" />
      <MetricCard label="Action Execution" value={fmtPercent(metrics.action_execution)} />
      <MetricCard label="Goal Fulfillment" value={fmtPercent(metrics.goal_fulfillment)} />
      <MetricCard label="Valid Rate" value={fmtPercent(metrics.valid_episode_rate)} tone="blue" />
      <MetricCard label="Unique Failures" value={fmtNumber(metrics.unique_failure_signatures)} tone="red" />
      <MetricCard label="Accepted Patches" value={fmtNumber(run?.service_candidates?.flatMap((item) => item.candidates || []).filter((item) => item.accepted).length)} tone="green" />
      <MetricCard label="Requests" value={fmtNumber(metrics.requests)} />
      <MetricCard label="Tokens" value={fmtNumber(metrics.tokens)} detail={`${fmtLatency(metrics.latency_seconds)} total latency`} />
    </div>
  )
}

function compareGenerationData(runs) {
  const allGenerations = new Set()
  runs.forEach((run) => (run.generation_metrics || []).forEach((item) => allGenerations.add(item.generation)))
  return [...allGenerations].sort((a, b) => a - b).map((generation) => {
    const row = { generation: `G${generation}` }
    runs.forEach((run) => {
      const metric = (run.generation_metrics || []).find((item) => item.generation === generation)
      row[`${run.id}_task_success`] = metric?.task_success == null ? null : metric.task_success * 100
    })
    return row
  })
}

function ComparisonChart({ runs }) {
  const data = compareGenerationData(runs)
  const palette = [COLORS.gray, COLORS.blue, COLORS.green, COLORS.orange]
  return (
    <div className="chart-card">
      <div className="chart-card-header"><div><div className="eyebrow">Comparison</div><h3>Task success by generation</h3></div><span className="chart-note">valid episodes only</span></div>
      <div className="chart-wrap">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 8, right: 12, left: -22, bottom: 0 }}>
            <CartesianGrid stroke="#edf0f3" vertical={false} />
            <XAxis dataKey="generation" tick={{ fontSize: 12, fill: '#7b8794' }} axisLine={false} tickLine={false} />
            <YAxis domain={[0, 100]} tickFormatter={(value) => `${value}%`} tick={{ fontSize: 12, fill: '#7b8794' }} axisLine={false} tickLine={false} />
            <Tooltip formatter={(value) => `${Number(value).toFixed(1)}%`} contentStyle={{ border: '1px solid #e1e6eb', borderRadius: 6, fontSize: 12 }} />
            {runs.map((run, index) => <Line key={run.id} type="monotone" dataKey={`${run.id}_task_success`} name={run.id} stroke={palette[index % palette.length]} strokeWidth={2} dot={{ r: 3 }} connectNulls />)}
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="legend-row">{runs.map((run, index) => <span key={run.id}><i style={{ background: palette[index % palette.length] }} />{run.id}</span>)}</div>
    </div>
  )
}

function FailureList({ episodes, onSelect }) {
  const failures = episodes.filter((item) => item.status !== 'VALID SUCCESS')
  if (!failures.length) return <div className="empty-state">No failure episodes in this run.</div>
  return (
    <div className="failure-list">
      {failures.slice(0, 6).map((episode) => {
        const invalid = episode.status === 'INVALID EVALUATION'
        return <button className="failure-row" key={episode.episode_id} onClick={() => onSelect(episode)}>
          <div className={`failure-mark ${invalid ? 'orange' : 'red'}`} />
          <div className="failure-row-main"><div className="failure-row-title">{invalid ? episode.invalid_reason : (episode.failure_signature?.signature_id || episode.error_types?.[0] || 'legitimate failure')}</div><div className="muted">{episode.case_id} · G{episode.generation} · {episode.sop_node || 'SOP node unknown'}</div></div>
          <span className="arrow">→</span>
        </button>
      })}
    </div>
  )
}

function FeaturedCase({ run, onSelect }) {
  const preferred = run?.episodes?.find((episode) => {
    const sequence = (episode.tool_sequence || []).join(' ').toLowerCase()
    return episode.status === 'LEGITIMATE FAILURE' && sequence.includes('query_order')
  })
  const episode = preferred || run?.episodes?.find((item) => item.status === 'LEGITIMATE FAILURE')
  if (!episode) return <div className="empty-state">No legitimate failure available for a featured case.</div>
  return (
    <div className="featured-case">
      <div className="featured-intro"><StatusBadge kind="failure">LEGITIMATE FAILURE</StatusBadge><span className="muted">{episode.case_id} · generation {episode.generation}</span></div>
      <div className="featured-flow">
        <div className="flow-step blue-step"><span>Customer</span><strong>{episode.case_metadata?.user_intent || 'Missing required identifier'}</strong></div>
        <div className="flow-arrow">↓</div>
        <div className="flow-step"><span>Agent decision</span><strong>{episode.predicted_action || 'Query without grounded identifier'}</strong></div>
        <div className="flow-arrow">↓</div>
        <div className="flow-step tool-step"><span>Tool call</span><strong>{episode.tool_sequence?.[0] || 'No tool trace'}</strong><code>{episode.tool_sequence?.[0] ? '{ "order_id": "" }' : '—'}</code></div>
        <div className="flow-arrow">↓</div>
        <div className="flow-step backend-step"><span>Backend result</span><strong>{episode.error_types?.[0] || 'goal_not_fulfilled'}</strong></div>
        <div className="flow-arrow">↓</div>
        <div className="flow-step failure-step"><span>Outcome</span><strong>{episode.termination_reason || 'goal_not_fulfilled'}</strong></div>
      </div>
      <button className="text-button" onClick={() => onSelect(episode)}>Open episode detail →</button>
    </div>
  )
}

function Overview({ run, runs, onNavigate, onSelectEpisode }) {
  return <>
    <SectionHeader eyebrow="Project overview" title="Evaluation health" action={<span className="muted">Read-only artifact view</span>} />
    <MetricGrid run={run} />
    <div className="two-column overview-grid">
      <ComparisonChart runs={runs} />
      <div className="panel"><SectionHeader eyebrow="Featured failure" title="What the system learned" /><FeaturedCase run={run} onSelect={onSelectEpisode} /></div>
    </div>
    <div className="two-column overview-grid lower">
      <div className="panel"><SectionHeader eyebrow="Recent" title="Legitimate failures" action={<button className="text-button" onClick={() => onNavigate('failures')}>View all →</button>} /><FailureList episodes={run.episodes || []} onSelect={onSelectEpisode} /></div>
      <div className="panel"><SectionHeader eyebrow="Integrity" title="Run completeness" action={<button className="text-button" onClick={() => onNavigate('diagnostics')}>Inspect →</button>} /><IntegrityChecklist run={run} compact /></div>
    </div>
  </>
}

function IntegrityChecklist({ run, compact = false }) {
  const checks = [
    ['Frozen code', run?.metadata?.commit_sha === '4dfc56d6c92fff86f8310e7fcf2fda12b41df42c', run?.metadata?.commit_sha ? run.metadata.commit_sha.slice(0, 10) : 'Not recorded'],
    ['Split manifest', run?.completeness?.split_manifest, 'artifact present'],
    ['Valid / invalid separated', run?.completeness?.valid_invalid_separated, 'from structured status'],
    ['Candidate provenance', run?.completeness?.candidate_provenance, run?.completeness?.candidate_provenance == null ? 'not applicable' : 'patch + policy records'],
    ['Tool traces', run?.completeness?.traces, 'display-only provenance'],
    ['Held-out isolated', run?.completeness?.heldout_not_used_by_evolver, 'not confirmed from artifacts'],
  ]
  return <div className={`checklist ${compact ? 'compact' : ''}`}>
    {checks.map(([label, value, detail]) => <div className="check-row" key={label}><span className={`check-icon ${value === true ? 'ok' : value === false ? 'bad' : 'unknown'}`}>{value === true ? '✓' : value === false ? '!' : '·'}</span><div><strong>{label}</strong><span>{value === true ? 'Confirmed' : value === false ? 'Missing' : 'Not confirmed'} · {detail}</span></div></div>)}
  </div>
}

function Experiments({ runs }) {
  const data = runs.map((run) => ({
    name: run.id,
    task: run.metrics?.task_success == null ? null : run.metrics.task_success * 100,
    action: run.metrics?.action_execution == null ? null : run.metrics.action_execution * 100,
    goal: run.metrics?.goal_fulfillment == null ? null : run.metrics.goal_fulfillment * 100,
  }))
  const gateData = runs.map((run) => {
    const candidates = (run.service_candidates || []).flatMap((item) => item.candidates || [])
    return {
      name: run.id,
      accepted: candidates.filter((item) => item.accepted).length,
      rejected: candidates.filter((item) => item.accepted === false && item.evaluation_status !== 'invalid' && item.evaluation_status !== 'inconclusive').length,
      invalid: candidates.filter((item) => item.evaluation_status === 'invalid' || item.evaluation_status === 'inconclusive').length,
    }
  })
  return <>
    <SectionHeader eyebrow="Experiments / Evaluations" title="Compare runs" action={<span className="muted">Structured artifact metrics</span>} />
    <div className="panel compare-panel"><div className="table-wrap"><table><thead><tr><th>Run</th><th>Source</th><th>Task Success</th><th>Action Execution</th><th>Goal Fulfillment</th><th>Valid Rate</th><th>Unique Failures</th><th>Held-out</th><th>Fresh adversary</th><th>Requests</th><th>Tokens</th><th>Latency</th></tr></thead><tbody>{runs.map((run) => <tr key={run.id}><td><strong>{run.id}</strong><small>{run.metadata?.model}</small></td><td><StatusBadge kind={sourceTone(run.metadata?.source_kind)}>{run.metadata?.source_kind}</StatusBadge></td><td>{fmtPercent(run.metrics?.task_success)}</td><td>{fmtPercent(run.metrics?.action_execution)}</td><td>{fmtPercent(run.metrics?.goal_fulfillment)}</td><td>{fmtPercent(run.metrics?.valid_episode_rate)}</td><td>{fmtNumber(run.metrics?.unique_failure_signatures)}</td><td>{run.heldout ? 'Evaluated' : 'Not evaluated'}</td><td>{run.fresh_adversary?.length ? 'Evaluated' : 'Not evaluated'}</td><td>{fmtNumber(run.metrics?.requests)}</td><td>{fmtNumber(run.metrics?.tokens)}</td><td>{fmtLatency(run.metrics?.latency_seconds)}</td></tr>)}</tbody></table></div></div>
    <div className="two-column lower"><div className="chart-card"><div className="chart-card-header"><div><div className="eyebrow">Performance</div><h3>Metric comparison</h3></div></div><div className="chart-wrap tall"><ResponsiveContainer width="100%" height="100%"><BarChart data={data} margin={{ top: 8, right: 12, left: -22, bottom: 0 }}><CartesianGrid stroke="#edf0f3" vertical={false} /><XAxis dataKey="name" tick={{ fontSize: 11, fill: '#7b8794' }} axisLine={false} tickLine={false} /><YAxis domain={[0, 100]} tickFormatter={(v) => `${v}%`} tick={{ fontSize: 12, fill: '#7b8794' }} axisLine={false} tickLine={false} /><Tooltip formatter={(value) => value == null ? 'Not evaluated' : `${Number(value).toFixed(1)}%`} contentStyle={{ border: '1px solid #e1e6eb', borderRadius: 6, fontSize: 12 }} /><Bar dataKey="task" name="Task Success" fill={COLORS.green} radius={[3, 3, 0, 0]} /><Bar dataKey="action" name="Action Execution" fill={COLORS.blue} radius={[3, 3, 0, 0]} /><Bar dataKey="goal" name="Goal Fulfillment" fill="#8ba4d8" radius={[3, 3, 0, 0]} /></BarChart></ResponsiveContainer></div></div><div className="panel"><SectionHeader eyebrow="Cost" title="Operational footprint" /><CostList runs={runs} /></div></div>
    <div className="three-column lower"><SignalChart title="Failure diversity" eyebrow="Research signal" data={runs.map((run) => ({ name: run.id, value: run.metrics?.unique_failure_signatures }))} dataKey="value" color={COLORS.red} formatter={(value) => fmtNumber(value)} /><SignalChart title="Invalid evaluation rate" eyebrow="Protocol health" data={runs.map((run) => ({ name: run.id, value: run.metrics?.invalid_rate == null ? null : run.metrics.invalid_rate * 100 }))} dataKey="value" color={COLORS.orange} formatter={(value) => value == null ? 'Not evaluated' : `${Number(value).toFixed(1)}%`} percent /><div className="chart-card"><div className="chart-card-header"><div><div className="eyebrow">Gate</div><h3>Service candidates</h3></div></div><div className="chart-wrap short"><ResponsiveContainer width="100%" height="100%"><BarChart data={gateData} margin={{ top: 8, right: 8, left: -22, bottom: 0 }}><CartesianGrid stroke="#edf0f3" vertical={false} /><XAxis dataKey="name" tick={{ fontSize: 10, fill: '#7b8794' }} axisLine={false} tickLine={false} /><YAxis allowDecimals={false} tick={{ fontSize: 11, fill: '#7b8794' }} axisLine={false} tickLine={false} /><Tooltip contentStyle={{ border: '1px solid #e1e6eb', borderRadius: 6, fontSize: 12 }} /><Bar dataKey="accepted" name="Accepted" fill={COLORS.green} stackId="gate" /><Bar dataKey="rejected" name="Rejected" fill="#9aa4b2" stackId="gate" /><Bar dataKey="invalid" name="Invalid" fill={COLORS.orange} stackId="gate" /></BarChart></ResponsiveContainer></div></div></div>
  </>
}

function SignalChart({ title, eyebrow, data, dataKey, color, formatter, percent = false }) {
  return <div className="chart-card"><div className="chart-card-header"><div><div className="eyebrow">{eyebrow}</div><h3>{title}</h3></div></div><div className="chart-wrap short"><ResponsiveContainer width="100%" height="100%"><BarChart data={data} margin={{ top: 8, right: 8, left: -22, bottom: 0 }}><CartesianGrid stroke="#edf0f3" vertical={false} /><XAxis dataKey="name" tick={{ fontSize: 10, fill: '#7b8794' }} axisLine={false} tickLine={false} /><YAxis allowDecimals={false} tickFormatter={percent ? (value) => `${value}%` : undefined} tick={{ fontSize: 11, fill: '#7b8794' }} axisLine={false} tickLine={false} /><Tooltip formatter={formatter} contentStyle={{ border: '1px solid #e1e6eb', borderRadius: 6, fontSize: 12 }} /><Bar dataKey={dataKey} fill={color} radius={[3, 3, 0, 0]} /></BarChart></ResponsiveContainer></div></div>
}

function CostList({ runs }) {
  return <div className="cost-list">{runs.map((run) => <div className="cost-row" key={run.id}><div><strong>{run.id}</strong><span>{fmtNumber(run.metrics?.requests)} requests · {fmtNumber(run.metrics?.tokens)} tokens</span></div><b>{fmtLatency(run.metrics?.latency_seconds)}</b></div>)}</div>
}

function Evolution({ run }) {
  const generations = [...new Set([...(run?.customer_candidates || []).map((item) => item.generation), ...(run?.service_candidates || []).map((item) => item.generation)])].sort((a, b) => a - b)
  if (!generations.length) return <EmptyPage title="Evolution" message="No generation artifacts were found for this run." />
  return <>
    <SectionHeader eyebrow="Customer ↔ Service" title="Evolution timeline" action={<StatusBadge kind={sourceTone(run.metadata?.source_kind)}>{run.metadata?.source_kind}</StatusBadge>} />
    <div className="timeline-page-note">Customer strategy <span>→</span> exposed failure <span>→</span> Service patch <span>→</span> gate decision</div>
    <div className="generation-timeline">{generations.map((generation) => <GenerationSection key={generation} generation={generation} customer={(run.customer_candidates || []).find((item) => item.generation === generation)} service={(run.service_candidates || []).find((item) => item.generation === generation)} />)}</div>
  </>
}

function PolicyCard({ policy, customer = false }) {
  if (!policy || Object.keys(policy).length === 0) return <div className="empty-inline">No policy record</div>
  const title = policy.name || policy.policy_id || policy.candidate_policy_id || (customer ? 'Customer policy' : 'Service policy')
  const tags = policy.strategy_tags || policy.rule_categories || []
  return <div className={`policy-card ${customer ? 'customer-card' : 'service-card'}`}><div className="policy-card-top"><StatusBadge kind={customer ? 'customer' : 'accepted'}>{customer ? 'CUSTOMER' : 'SERVICE'}</StatusBadge><strong>{title}</strong></div><div className="policy-id">{policy.policy_id || policy.candidate_policy_id || 'policy id not recorded'}</div>{policy.description && <p>{policy.description}</p>}{policy.rationale && <p>{policy.rationale}</p>}{tags.length > 0 && <div className="tag-row">{tags.map((tag) => <span key={tag}>{tag}</span>)}</div>}<div className="policy-meta">{customer && policy.fitness != null && <span>fitness <b>{Number(policy.fitness).toFixed(2)}</b></span>}{policy.source_failure_ids?.length > 0 && <span>source failures <b>{policy.source_failure_ids.length}</b></span>}</div></div>
}

function GateDecision({ service }) {
  const candidates = service?.candidates || []
  if (!candidates.length) return <div className="gate-box"><div className="gate-title"><StatusBadge kind="neutral">NO CANDIDATE</StatusBadge><strong>{service?.reason || 'No service candidate recorded'}</strong></div><div className="muted">The gate decision is preserved exactly as emitted by the artifact.</div></div>
  return <div className="gate-stack">{candidates.map((candidate, index) => { const invalid = candidate.evaluation_status === 'invalid' || candidate.evaluation_status === 'inconclusive'; return <div className="gate-box" key={`${index}-${candidate.patch_id || candidate.reason}`}><div className="gate-title"><StatusBadge kind={invalid ? 'invalid' : candidate.accepted ? 'accepted' : 'rejected'}>{invalid ? candidate.evaluation_status?.toUpperCase() : candidate.accepted ? 'ACCEPTED' : 'REJECTED'}</StatusBadge><strong>{candidate.reason || 'gate result'}</strong></div><div className="gate-meta"><span>delta <b>{candidate.delta == null ? '—' : Number(candidate.delta).toFixed(3)}</b></span>{candidate.patch_id && <span>patch <b>{candidate.patch_id}</b></span>}{candidate.source_failure_ids?.length > 0 && <span>source failures <b>{candidate.source_failure_ids.length}</b></span>}</div>{candidate.patch && <pre className="patch-preview">{typeof candidate.patch === 'string' ? candidate.patch : JSON.stringify(candidate.patch, null, 2)}</pre>}</div>})}</div>
}

function GenerationSection({ generation, customer, service }) {
  const selected = customer?.selected_policy || {}
  return <section className="generation-section"><div className="generation-marker"><span>G{generation}</span><i /></div><div className="generation-content"><div className="generation-heading"><h3>Generation {generation}</h3><span className="muted">candidate evaluation snapshot</span></div><div className="evolution-lanes"><div className="lane customer-lane"><div className="lane-label">CUSTOMER</div><PolicyCard customer policy={{ ...selected, fitness: customer?.scores?.find((item) => item.policy_id === selected.policy_id)?.fitness }} /><div className="lane-connector">exposed <span>↓</span></div><div className="exposed-failure">{selected.source_failure_ids?.[0] || 'failure frontier / current attack surface'}<small>{customer?.candidate_count || 0} candidates · {selected.strategy_tags?.join(' · ') || 'strategy tags not recorded'}</small></div></div><div className="lane service-lane"><div className="lane-label">SERVICE CANDIDATE</div><GateDecision service={service} /><div className="lane-footer">{service?.candidate_count || 0} candidate record(s) · provenance {service?.provenance_present ? 'complete' : 'partial'}</div></div></div></div></section>
}

function FilterBar({ filters, setFilters, errors, generations }) {
  return <div className="filter-bar"><label>Generation<select value={filters.generation} onChange={(e) => setFilters({ ...filters, generation: e.target.value })}><option value="all">All</option>{generations.map((item) => <option key={item} value={item}>G{item}</option>)}</select></label><label>Status<select value={filters.status} onChange={(e) => setFilters({ ...filters, status: e.target.value })}><option value="all">All</option><option value="VALID SUCCESS">Valid success</option><option value="LEGITIMATE FAILURE">Legitimate failure</option><option value="INVALID EVALUATION">Invalid evaluation</option></select></label><label>Error type<select value={filters.error} onChange={(e) => setFilters({ ...filters, error: e.target.value })}><option value="all">All</option>{errors.map((item) => <option key={item} value={item}>{item}</option>)}</select></label><label>SOP node<select value={filters.sop} onChange={(e) => setFilters({ ...filters, sop: e.target.value })}><option value="all">All</option>{filters.sopOptions?.map((item) => <option key={item} value={item}>{item}</option>)}</select></label></div>
}

function EpisodeTable({ episodes, onSelect }) {
  return <div className="table-wrap"><table className="episodes-table"><thead><tr><th>Status</th><th>Case</th><th>Gen</th><th>Customer policy</th><th>Service policy</th><th>Task success</th><th>Error</th><th>Tools</th></tr></thead><tbody>{episodes.map((episode) => { const invalid = episode.status === 'INVALID EVALUATION'; return <tr key={episode.episode_id} onClick={() => onSelect(episode)}><td><StatusBadge kind={invalid ? 'invalid' : episode.status === 'LEGITIMATE FAILURE' ? 'failure' : 'success'}>{invalid ? 'INVALID' : episode.status === 'LEGITIMATE FAILURE' ? 'FAILURE' : 'SUCCESS'}</StatusBadge></td><td><strong>{episode.case_id}</strong><small>{episode.split} · {episode.sop_node || '—'}</small></td><td>G{episode.generation}</td><td>{display(episode.customer_policy_id, '—')}</td><td>{display(episode.service_policy_id, '—')}</td><td>{episode.task_success == null ? '—' : episode.task_success ? 'Yes' : 'No'}</td><td>{episode.invalid_reason || episode.error_types?.[0] || '—'}</td><td>{episode.tool_sequence?.length || 0}</td></tr>})}</tbody></table>{!episodes.length && <div className="empty-state">No episodes match these filters.</div>}</div>
}

function Episodes({ run, onSelectEpisode, focusFailures = false }) {
  const [filters, setFilters] = useState({ generation: 'all', status: focusFailures ? 'LEGITIMATE FAILURE' : 'all', error: 'all', sop: 'all' })
  const episodes = run?.episodes || []
  const generations = [...new Set(episodes.map((item) => item.generation))].sort((a, b) => a - b)
  const errors = [...new Set(episodes.flatMap((item) => item.error_types || []))].sort()
  const sops = [...new Set(episodes.map((item) => item.sop_node).filter(Boolean))].sort()
  const shown = episodes.filter((episode) => (filters.generation === 'all' || String(episode.generation) === filters.generation) && (filters.status === 'all' || episode.status === filters.status) && (filters.error === 'all' || episode.error_types?.includes(filters.error)) && (filters.sop === 'all' || episode.sop_node === filters.sop))
  const actualFilters = { ...filters, sopOptions: sops }
  return <><SectionHeader eyebrow={focusFailures ? 'Failure Explorer' : 'Projects / Traces'} title={focusFailures ? 'Failures' : 'Episodes'} action={<span className="muted">{shown.length} of {episodes.length} episodes</span>} /><FilterBar filters={actualFilters} setFilters={setFilters} errors={errors} generations={generations} /><div className="panel episodes-panel"><EpisodeTable episodes={shown} onSelect={onSelectEpisode} /></div></>
}

function TraceEvent({ event }) {
  const kind = event.kind || 'agent'
  const payload = event.payload
  const value = typeof payload === 'string' ? payload : payload == null ? '—' : JSON.stringify(payload, null, 2)
  return <div className={`trace-event ${kind}`}><div className="trace-dot" /><div className="trace-event-content"><div className="trace-event-header"><strong>{event.label || kind.toUpperCase()}</strong>{event.turn != null && <span>turn {event.turn}</span>}</div><pre>{value}</pre></div></div>
}

function EpisodeDetail({ episode, onClose }) {
  if (!episode) return <div className="detail-empty"><div className="empty-illustration">◉</div><h3>Select an episode</h3><p>Choose a row from Episodes or Failures to inspect the structured trace.</p></div>
  const invalid = episode.status === 'INVALID EVALUATION'
  return <div className="detail-panel"><div className="detail-header"><div><StatusBadge kind={invalid ? 'invalid' : episode.status === 'LEGITIMATE FAILURE' ? 'failure' : 'success'}>{episode.status}</StatusBadge><h2>{episode.episode_id}</h2></div><button className="close-button" onClick={onClose}>×</button></div><div className="detail-meta-grid"><Meta label="Case" value={episode.case_id} /><Meta label="Generation" value={`G${episode.generation}`} /><Meta label="Split" value={episode.split} /><Meta label="Expected action" value={episode.expected_action} /><Meta label="Executed action" value={episode.executed_action} /><Meta label="Termination" value={episode.termination_reason} /><Meta label="SOP node" value={episode.sop_node} /><Meta label="Error" value={invalid ? episode.invalid_reason : episode.error_types?.join(', ')} /></div><div className="score-strip">{Object.entries(episode.scores || {}).map(([key, value]) => <div key={key}><span>{key.replaceAll('_', ' ')}</span><b>{value == null ? '—' : Number(value).toFixed(2)}</b></div>)}</div><div className="detail-section"><SectionHeader eyebrow="Trace detail" title="Execution timeline" action={<span className="muted">display-only provenance</span>} />{episode.trace_events?.length ? <div className="trace-timeline">{episode.trace_events.map((event, index) => <TraceEvent key={event.id || index} event={event} />)}</div> : <div className="empty-state">No trace artifact was exported for this episode.</div>}</div><div className="detail-section"><SectionHeader eyebrow="Tool summary" title="Tool sequence" />{episode.tool_sequence?.length ? <div className="tool-chips">{episode.tool_sequence.map((tool, index) => <span key={`${tool}-${index}`}><b>{index + 1}</b>{tool}</span>)}</div> : <div className="muted">No tool calls recorded.</div>}</div><div className="detail-footer"><span>trace ref: {display(episode.trace_ref, 'not recorded')}</span><span>model: {display(episode.provenance?.model, 'not recorded')}</span></div></div>
}

function Meta({ label, value }) { return <div className="meta-cell"><span>{label}</span><strong>{display(value, '—')}</strong></div> }

function Diagnostics({ run }) {
  const metrics = run?.metrics || {}
  const metadata = run?.metadata || {}
  return <><SectionHeader eyebrow="Observability" title="Diagnostics & reproducibility" action={<StatusBadge kind={sourceTone(metadata.source_kind)}>{metadata.source_kind}</StatusBadge>} /><div className="metric-grid diagnostics-grid"><MetricCard label="Requests" value={fmtNumber(metrics.requests)} /><MetricCard label="Input tokens" value={fmtNumber(metrics.input_tokens)} /><MetricCard label="Output tokens" value={fmtNumber(metrics.output_tokens)} /><MetricCard label="Latency" value={fmtLatency(metrics.latency_seconds)} /><MetricCard label="Timeouts" value={fmtNumber(metrics.timeouts)} tone={metrics.timeouts ? 'invalid' : 'neutral'} /><MetricCard label="Provider failures" value={fmtNumber(metrics.provider_failures)} tone={metrics.provider_failures ? 'invalid' : 'neutral'} /><MetricCard label="Invalid evaluations" value={fmtNumber(metrics.invalid_episodes)} tone={metrics.invalid_episodes ? 'invalid' : 'neutral'} /><MetricCard label="Retries" value={fmtNumber(metrics.retries)} /></div><div className="two-column lower"><div className="panel"><SectionHeader eyebrow="Reproducibility" title="Run manifest" /><div className="manifest-list"><Meta label="Freeze SHA" value={metadata.commit_sha} /><Meta label="Tag" value={metadata.freeze_tag} /><Meta label="Model" value={metadata.model} /><Meta label="Provider" value={metadata.provider} /><Meta label="Seed" value={metadata.seed} /><Meta label="Split" value={metadata.split_strategy} /><Meta label="Cases" value={metadata.max_cases} /><Meta label="Generations" value={metadata.generations} /><Meta label="Max turns" value={metadata.max_turns} /><Meta label="Judge in evolution" value={metadata.judge_in_evolution == null ? 'Not recorded' : String(metadata.judge_in_evolution)} /></div></div><div className="panel"><SectionHeader eyebrow="Integrity checklist" title="What artifacts can confirm" /><IntegrityChecklist run={run} /></div></div></>
}

function EmptyPage({ title, message }) { return <div className="empty-page"><div className="empty-illustration">⌁</div><h2>{title}</h2><p>{message}</p></div> }

function Sidebar({ page, onChange, run }) {
  return <aside className="sidebar"><div className="brand"><div className="brand-mark">E</div><div><strong>EvoSAGE</strong><span>Research dashboard</span></div></div><div className="sidebar-section-label">Workspace</div><nav>{NAV.map((item) => <button key={item.id} className={page === item.id ? 'active' : ''} onClick={() => onChange(item.id)}><span className="nav-icon">{item.icon}</span>{item.label}</button>)}</nav><div className="sidebar-spacer" /><div className="sidebar-footer"><div><StatusBadge kind={sourceTone(run?.metadata?.source_kind)}>{run?.metadata?.source_kind || 'UNKNOWN'}</StatusBadge></div><div className="freeze-label">{display(run?.metadata?.freeze_tag, 'freeze not recorded')}</div><div className="muted small">read-only · no runtime controls</div></div></aside>
}

function App() {
  const [dataset, setDataset] = useState(null)
  const [loadError, setLoadError] = useState(null)
  const [page, setPage] = useState('overview')
  const [activeRunId, setActiveRunId] = useState('')
  const [selectedEpisode, setSelectedEpisode] = useState(null)

  useEffect(() => {
    fetch('./data/runs.json', { cache: 'no-store' })
      .then((response) => response.ok ? response.json() : Promise.reject(new Error('runs.json not found')))
      .then((value) => setDataset(value))
      .catch(() => fetch('./data/demo.json').then((response) => response.json()).then((value) => setDataset(value)).catch((error) => setLoadError(error.message)))
  }, [])

  const runs = dataset?.runs || []
  const activeRun = runs.find((run) => run.id === activeRunId) || runs.find((run) => run.id === 'coevolution') || runs[0]
  useEffect(() => { if (!activeRunId && activeRun?.id) setActiveRunId(activeRun.id) }, [activeRunId, activeRun])

  const navigate = (nextPage) => { setPage(nextPage); setSelectedEpisode(null) }
  const selectEpisode = (episode) => { setSelectedEpisode(episode); setPage('episodes') }

  if (loadError) return <div className="load-error">Could not load dashboard data: {loadError}</div>
  if (!dataset || !activeRun) return <div className="loading-screen"><div className="brand-mark">E</div><span>Loading artifact dataset…</span></div>

  const content = page === 'overview' ? <Overview run={activeRun} runs={runs} onNavigate={navigate} onSelectEpisode={selectEpisode} />
    : page === 'experiments' ? <Experiments runs={runs} />
      : page === 'evolution' ? <Evolution run={activeRun} />
        : page === 'episodes' ? <Episodes run={activeRun} onSelectEpisode={setSelectedEpisode} />
          : page === 'failures' ? <Episodes run={activeRun} onSelectEpisode={setSelectedEpisode} focusFailures />
            : <Diagnostics run={activeRun} />

  return <div className="app-shell"><Sidebar page={page} onChange={navigate} run={activeRun} /><main className="main-shell"><ExperimentHeader run={activeRun} runs={runs} onChangeRun={(id) => { setActiveRunId(id); setSelectedEpisode(null) }} onNavigate={navigate} /><ExperimentTabs page={page} onChange={navigate} /><div className="content-shell">{content}</div>{selectedEpisode && <div className="detail-drawer"><EpisodeDetail episode={selectedEpisode} onClose={() => setSelectedEpisode(null)} /></div>}</main></div>
}

export default App
