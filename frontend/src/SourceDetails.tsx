import { useEffect, useState } from "react";
import { api, type Preview, type Source } from "./api";
import { messageOf } from "./useTask";

export function Attribution({ source }: { source?: Source }) {
  const credit = source?.attribution;
  if (!credit) return null;
  return <p className="meta">数据来源 · <a href={credit.url} target="_blank" rel="noreferrer">{credit.title}</a> · {credit.license}</p>;
}

export function SourceDetails({ source }: { source: Source }) {
  const [open, setOpen] = useState(false);
  const [preview, setPreview] = useState<Preview | null>(null);
  const [error, setError] = useState("");
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    if (!open || preview) return;
    const controller = new AbortController();
    setError("");
    api.preview(source.id, controller.signal)
      .then(data => { if (!controller.signal.aborted) setPreview(data); })
      .catch(e => { if (!controller.signal.aborted) setError(messageOf(e)); });
    return () => controller.abort();
  }, [open, preview, source.id, attempt]);

  return <>
    {source.description && <p className="hint">{source.description}</p>}
    <Attribution source={source} />
    <details className="schema" onToggle={e => setOpen(e.currentTarget.open)}>
      <summary>查看原始样本、表结构与统计口径</summary>
      {source.definitions && <p>{source.definitions}</p>}
      {open && !preview && !error && <p role="status">正在读取样本…</p>}
      {error && <p role="alert">{error} <button type="button" className="text-button" onClick={() => setAttempt(n => n + 1)}>重试</button></p>}
      {Object.entries(source.schema || {}).map(([table, columns]) => <div key={table}>
        <p><code>{table}</code> · 每张表最多预览 5 行</p>
        <div className="table-wrap"><table aria-label={table + " 原始样本"}>
          <thead><tr>{columns.map(name => <th key={name}>{name}</th>)}</tr></thead>
          <tbody>{preview?.tables[table]?.map((row, i) => <tr key={i}>{columns.map(name =>
            <td key={name}>{row[name] == null ? "NULL" : String(row[name])}</td>)}</tr>)}</tbody>
        </table></div>
      </div>)}
    </details>
  </>;
}
