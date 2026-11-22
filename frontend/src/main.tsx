import { useEffect, useRef, useState, type FormEvent } from "react";
import { createRoot } from "react-dom/client";
import { api, linkTo, type Config, type Run } from "./api";
import { RunDetail, statuses } from "./RunDetail";
import { SourceDetails } from "./SourceDetails";
import { messageOf, useLocation, useTask } from "./useTask";
import "./styles.css";

function App() {
  const [config, setConfig] = useState<Config | null>(null);
  const [configError, setConfigError] = useState("");
  const [configTry, setConfigTry] = useState(0);
  const [sourceId, setSourceId] = useState("");
  const [mode, setMode] = useState("");
  const [allowQueries, setAllowQueries] = useState(false);
  const [splitEvidence, setSplitEvidence] = useState(false);
  const [question, setQuestion] = useState("");
  const [history, setHistory] = useState<Run[]>([]);
  const [historyError, setHistoryError] = useState("");
  const [historyTry, setHistoryTry] = useState(0);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState("");
  const requestKey = useRef({ text: "", id: "" });
  const route = useLocation();
  const { task, error, reload } = useTask(route);

  useEffect(() => {
    const controller = new AbortController();
    setConfigError("");
    api.config(controller.signal).then(data => {
      if (controller.signal.aborted) return;
      setConfig(data);
      const requested = new URL(window.location.href).searchParams.get("source");
      setSourceId(data.sources.find(s => s.id === requested)?.id || data.sources[0]?.id || "");
      setMode(data.modes.find(m => m.available)?.id || "");
    }).catch(e => { if (!controller.signal.aborted) setConfigError(messageOf(e)); });
    return () => controller.abort();
  }, [configTry]);

  useEffect(() => {
    const controller = new AbortController();
    api.history(controller.signal).then(data => {
      if (!controller.signal.aborted) { setHistory(data.runs); setHistoryError(""); }
    }).catch(e => { if (!controller.signal.aborted) setHistoryError(messageOf(e)); });
    return () => controller.abort();
  }, [task?.id, task?.status, historyTry]);

  const source = config?.sources.find(s => s.id === sourceId);
  const selectedMode = config?.modes.find(m => m.id === mode);

  function selectSource(id: string) {
    setSourceId(id);
    setQuestion("");
    setSubmitError("");
    requestKey.current = { text: "", id: "" };
    const url = new URL(window.location.href);
    url.searchParams.set("source", id);
    window.history.replaceState(null, "", url);
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!question.trim() || !selectedMode?.available || submitting) return;
    setSubmitting(true); setSubmitError("");
    const queryPolicy = mode === "qwen" ? allowQueries : undefined;
    const signoffPolicy = mode === "qwen" ? splitEvidence : undefined;
    const text = JSON.stringify([sourceId, mode, question.trim(), queryPolicy, signoffPolicy]);
    if (requestKey.current.text !== text) requestKey.current = { text, id: crypto.randomUUID() };
    try {
      const run = await api.submit(question.trim(), sourceId, mode, requestKey.current.id, queryPolicy, signoffPolicy);
      requestKey.current = { text: "", id: "" };
      window.location.hash = linkTo({ kind: "run", id: run.id });
    } catch (e) { setSubmitError(messageOf(e)); }
    finally { setSubmitting(false); }
  }

  const openUpdated = (next: Run) => {
    if (next.id === route?.id) reload();
    else window.location.hash = linkTo({ kind: "run", id: next.id });
    setHistoryTry(value => value + 1);
  };

  return <><header className="mast"><a href="#" className="brand" aria-label="DataVerify 首页"><span className="brand-mark" aria-hidden="true">≡</span>DataVerify</a><span className="mast-label">SQL 核验工作台</span></header>
    <main className="layout"><aside className="sidebar">
      <a className={!route ? "new-task selected" : "new-task"} href="#">＋ 新建任务</a>
      {!!config?.replays.length && <nav aria-label="历史回放"><h2>案例回放</h2>{config.replays.map(item =>
        <a key={item.id} aria-current={route?.kind === "replay" && route.id === item.id ? "page" : undefined}
          href={linkTo({kind:"replay", id:item.id})}>{item.label}</a>)}</nav>}
      <nav aria-label="最近任务"><h2>最近任务</h2>
        {history.slice(0, 20).map(item => <a key={item.id} aria-current={route?.kind === "run" && route.id === item.id ? "page" : undefined}
          href={linkTo({kind:"run", id:item.id})}>{item.question}<small>{statuses[item.status] || item.status}</small></a>)}
        {!!historyError && <div><p className="error">历史读取失败：{historyError}</p><button className="text-button" onClick={() => setHistoryTry(x => x + 1)}>重新读取</button></div>}
        {!historyError && !history.length && <p className="muted">暂无任务记录</p>}</nav>
    </aside>
    <div className="work">
      {configError && <div className="error" role="alert">配置读取失败：{configError} <button className="text-button" onClick={() => setConfigTry(x=>x+1)}>重试</button></div>}
      {!route && <div className="composer"><h1>新建核验任务</h1>
        <form onSubmit={submit}><div className="form-row"><label htmlFor="source">数据源<select id="source" aria-label="数据源" value={sourceId} onChange={e => selectSource(e.target.value)}>
          {config?.sources.map(s => <option key={s.id} value={s.id}>{s.label}</option>)}</select></label>
        <label htmlFor="mode">执行模式<select id="mode" value={mode} onChange={e => setMode(e.target.value)}>
          {config?.modes.map(m => <option key={m.id} value={m.id} disabled={!m.available}>{m.label}{!m.available ? "（不可用）" : ""}</option>)}</select></label></div>
        {source && <SourceDetails key={source.id} source={source} />}
        {config?.modes.filter(m => !m.available && m.reason).map(m => <p key={m.id} className="hint">{m.label} · {m.reason}</p>)}
        {mode === "offline" && <p className="hint">离线模式使用预设模型响应执行下方示例。自定义问题请选千问实时。</p>}
        {mode === "qwen" && <div className="query-policy">
          <label><input type="checkbox" checked={allowQueries} onChange={e => setAllowQueries(e.target.checked)} aria-describedby="query-policy-note" /> 分歧时允许额外查询</label>
          <p id="query-policy-note" className="hint">仅控制分歧调查时的额外只读查询，基础 SQL 核验照常执行。</p>
          <label><input type="checkbox" checked={splitEvidence} onChange={e => setSplitEvidence(e.target.checked)} /> 试用分栏签核</label>
        </div>}
        <label htmlFor="question">统计问题</label><textarea id="question" rows={5} maxLength={2000} value={question} onChange={e => setQuestion(e.target.value)} placeholder="写下统计对象、时间范围，以及你希望采用的业务口径。" required />
        {!!source?.examples.length && <div className="examples"><span>可选示例</span>{source.examples.map(q => <button key={q} type="button" className="example" onClick={() => setQuestion(q)}>{q}</button>)}</div>}
        {submitError && <p className="error" role="alert">{submitError}</p>}
        <div className="actions"><button type="submit" disabled={submitting || !selectedMode?.available || !sourceId || !question.trim() || (mode === "offline" && !source?.examples.includes(question.trim()))}>{submitting ? "正在提交…" : "提交核验"}</button></div></form>
      </div>}
      {route && <>
        <a href="#" className="back">← 新建任务</a>
        {error && <div role="alert" className="error">{error} <button className="text-button" onClick={reload}>重新读取</button></div>}
        {!task && !error && <p role="status">正在读取任务…</p>}
        {task && <RunDetail key={route.kind + "/" + task.id} run={task} source={config?.sources.find(s => s.id === task.source_id)} replay={route.kind === "replay"} onUpdated={openUpdated} />}
      </>}
    </div></main></>;
}

createRoot(document.getElementById("root")!).render(<App />);
