import { useEffect, useState } from "react";
import { api, type LocationRef, type Run, readLocation } from "./api";

export function messageOf(error: unknown) {
  return error instanceof Error ? error.message : String(error);
}

export function useLocation() {
  const [hash, setHash] = useState(() => window.location.hash);
  useEffect(() => {
    const update = () => setHash(window.location.hash);
    window.addEventListener("hashchange", update);
    return () => window.removeEventListener("hashchange", update);
  }, []);
  return readLocation(hash);
}

export function useTask(ref: LocationRef | null) {
  const id = ref?.id;
  const kind = ref?.kind;
  const [task, setTask] = useState<Run | null>(null);
  const [error, setError] = useState("");
  const [revision, setRevision] = useState(0);

  useEffect(() => {
    setTask(null);
    setError("");
    if (!id || !kind) return;
    const controller = new AbortController();
    let timer: number | undefined;
    async function load() {
      try {
        const next = await api.get({ id: id!, kind: kind! }, controller.signal);
        if (controller.signal.aborted) return;
        setTask(next);
        setError("");
        if (kind === "run" && ["queued", "running"].includes(next.status)) {
          timer = window.setTimeout(load, 1200);
        }
      } catch (failure) {
        if (!controller.signal.aborted) setError(messageOf(failure));
      }
    }
    void load();
    return () => { window.clearTimeout(timer); controller.abort(); };
  }, [id, kind, revision]);
  return { task, error, reload: () => setRevision((value) => value + 1) };
}
