import { useState, type FormEvent } from "react";
import { api, linkTo, type Run, type Result, type Source } from "./api";
import { Attribution } from "./SourceDetails";
import { messageOf } from "./useTask";

export const statuses: Record<string, string> = {
  queued: "排队中", running: "运行中", reviewed: "有限核验通过", needs_clarification: "等待澄清",
  inconclusive: "未得出结论", failed: "运行失败", interrupted: "已中断",
};
const targets: Record<string, string> = { candidate: "候选 SQL", auditor: "独立核算", reference: "参考程序" };
const stages: Record<string, string> = { discovery: "生成与核验", dispute: "调查分歧", publication: "核对交付来源" };

function display(value: unknown): string {
  if (value === null) return "NULL";
  if (value === undefined) return "—";
  if (typeof value === "object") return JSON.stringify(value, null, 2);
  return String(value);
}

function ModelUsage({ run, replay }: { run: Run; replay: boolean }) {
  if (replay) return <><p className="meta">历史模型记录 · 本次查看新增 0 次 API 请求</p>
    {run.result?.usage?.complete && <p className="meta">原任务用量 · {run.result.usage.requests} 次千问请求 · {run.result.usage.total_tokens.toLocaleString("zh-CN")} tokens</p>}</>;
  if (run.mode === "offline") return <p className="meta">离线演示 · 0 次模型 API 请求</p>;
  if (run.mode !== "qwen") return null;
  const usage = run.result?.usage;
  if (!usage) return <p className="meta">千问实时模式 · 尚无可展示的任务用量记录</p>;
  return <p className="meta">千问 API · 本任务累计 {usage.requests} 次请求尝试 · {usage.complete ? "用量" : "已回报用量"} {usage.total_tokens.toLocaleString("zh-CN")} tokens{!usage.complete && "（数据不完整）"}</p>;
}

function ResultTable({ result, download }: { result: Result; download?: string }) {
  const columns = result.columns || [];
  if (!columns.length) return null;
  const rows = result.rows || [];
  return <section><div className="table-heading"><h3>查询结果 <small>{rows.length} 行</small></h3>
    {download && <a href={download} download>下载 CSV</a>}</div>
    <div className="table-wrap"><table><thead><tr>{columns.map((col, i) => <th key={i}>{col}</th>)}</tr></thead>
      <tbody>{rows.map((row, i) => <tr key={i}>{columns.map((col, j) => <td key={j}>{display(Array.isArray(row) ? row[j] : row[col])}</td>)}</tr>)}</tbody>
    </table></div>{!rows.length && <p className="muted">本次查询返回空结果。</p>}</section>;
}

function EvidenceDetails({ result }: { result: Result }) {
  const { evidence = [], revisions = [] } = result;
  return <>
    {result.finish_citations && <section aria-label="交付引用"><h3>交付引用</h3>
      <h4>签核依据</h4>
      <ul>{result.finish_citations.signoff_ids.map(id => <li key={id}>
        {id.includes(":") ? "当前版本的执行检查" : "当前版本的核验回放"} · <code>{id}</code>
      </li>)}</ul>
      <h4>辅助说明</h4>
      {result.finish_citations.supporting_ids.length ?
        <ul>{result.finish_citations.supporting_ids.map(id => <li key={id}>
          {evidence.find(item => item.id === id)?.detail || "只读查询记录"} · <code>{id}</code>
        </li>)}</ul> : <p className="muted">本次交付未引用调查查询。</p>}
    </section>}
    {!!evidence.length && <section><h3>执行证据 <small>{evidence.length} 条</small></h3>
      {evidence.map((item, i) => <details key={i} className="evidence"><summary>{item.title}</summary>
        {item.scope && <p className="meta">{display(item.scope)}</p>}
        <p>{display(item.detail)}</p>{item.sql && <pre>{item.sql}</pre>}
        {item.checks !== undefined && <p>检查项目 · {item.checks}</p>}
        {item.rows && <><h4>执行返回</h4><pre>{display(item.rows)}</pre></>}
        {item.sample && <><h4>同一输入上的结果</h4>{["candidate", "reference", "independent"].filter(role => role in item.sample!).map(role => <div key={role}><small>{targets[role] || "独立核算"}</small><pre>{display(item.sample![role])}</pre></div>)}</>}
        {item.versions && <p className="meta">程序版本 · {Object.entries(item.versions).map(([name, version]) => (targets[name] || name) + " " + display(version)).join("，")}</p>}
        {item.id && <small className="meta">证据编号 · <code>{item.id}</code></small>}</details>)}</section>}
    {!!revisions.length && <section><h3>程序修订 <small>{revisions.length} 条</small></h3>
      {revisions.map((item, i) => <details key={i} className="evidence"><summary>{targets[item.target] || item.target}{item.accepted !== undefined && (item.accepted ? " · 已采用" : " · 未采用")}</summary>
        <p>{display(item.reason)}</p>{item.quote && <p>需求原文 · {item.quote}</p>}
        {!!item.evidence_ids?.length && <small className="meta">引用证据 · {item.evidence_ids.join("，")}</small>}
        <div className="diff"><div><h4>修订前</h4><pre>{display(item.before)}</pre></div>
          <div><h4>修订后</h4><pre>{display(item.after)}</pre></div></div></details>)}</section>}
  </>;
}

export function RunDetail({ run, source, replay, onUpdated }: { run: Run; source?: Source; replay: boolean; onUpdated: (next: Run) => void }) {
  const [answer, setAnswer] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  async function advance(event: FormEvent) {
    event.preventDefault();
    if (busy) return;
    const resume = run.status === "interrupted";
    if (!resume && !answer.trim()) return;
    setBusy(true); setError("");
    try {
      const next = resume ? await api.resume(run.id) : await api.clarify(run.id, answer.trim());
      setAnswer(""); onUpdated(next);
    } catch (failure) { setError(messageOf(failure)); }
    finally { setBusy(false); }
  }
  return <article className="result" aria-label="任务详情">
    <header className="result-head"><div className="result-label">{replay ? "已保存的历史回放" : "任务详情"}</div>
      <span className={`status ${run.status === "reviewed" ? "passed" : ""}`}>{statuses[run.status] || run.status}</span></header>
    <h2 className="question">{run.question}</h2>
    <p className="meta">任务 <code>{run.id}</code> · {run.mode} · {source?.label || run.source_id}</p>
    <Attribution source={source} />
    <ModelUsage run={run} replay={replay} />
    {run.mode === "qwen" && typeof run.allow_queries === "boolean" &&
      <p className="muted">分歧时的额外查询 · {run.allow_queries ? "已开启" : "已关闭"}</p>}
    {run.parent_id && <a href={linkTo({ kind: "run", id: run.parent_id })}>查看上一次任务</a>}
    {run.child_id && <a href={linkTo({ kind: "run", id: run.child_id })}>查看补充口径后的任务</a>}
    {run.definitions && <p>补充口径 · {display(run.definitions)}</p>}
    {run.stage && !replay && <p className="muted" role="status">处理阶段 · {stages[run.stage] || run.stage}</p>}
    {run.result?.summary && <p className="summary">{run.result.summary}</p>}
    {run.result?.scope && <p className="muted">{display(run.result.scope)}</p>}
    {run.error && <p className="error" role="alert">{display(run.error)}</p>}
    {run.clarification && <p className="summary">{display(run.clarification)}</p>}
    {!replay && !run.child_id && run.status === "needs_clarification" && <form onSubmit={advance} className="clarify">
      <label htmlFor="clarification">补充业务定义</label><textarea id="clarification" value={answer} onChange={e => setAnswer(e.target.value)} rows={2} maxLength={2000} required />
      <button disabled={busy || !answer.trim()}>{busy ? "正在提交…" : "提交并继续"}</button></form>}
    {!replay && run.status === "interrupted" && <form onSubmit={advance}><button disabled={busy}>{busy ? "正在恢复…" : "恢复任务"}</button></form>}
    {error && <p className="error" role="alert">{error}</p>}
    {run.result && <><ResultTable result={run.result} download={run.status === "reviewed" ? `/api/workbench/${replay ? "replays" : "runs"}/${encodeURIComponent(run.id)}/result.csv` : undefined} />
      {(run.result.sql || run.result.candidate_sql) && <section><h3>{run.result.sql ? "交付 SQL" : "候选 SQL（未交付）"}</h3><pre className="sql">{run.result.sql || run.result.candidate_sql}</pre></section>}
      {run.status === "inconclusive" && run.result.reason && <p className="muted">退出原因 · {run.result.reason}</p>}
      <EvidenceDetails result={run.result} /></>}
  </article>;
}
