/**
 * The page's side of the analytics worker: sends each AlphaEarth chunk once, asks for the view.
 */
import AnalyticsWorker from "./analytics.worker?worker";

export type ViewStats = {
  pixels: number;
  read_ha?: number;
  likely_ha?: number;
  digs?: { cell: string; ha: number; peak: number }[];
  places?: { name: string; ha: number; of_ha: number }[];
};

export type ChunkIn = { key: string; r: number; c: number; w: number; h: number; first: Int8Array; last: Int8Array };

export function createAnalytics(models: Record<string, unknown>, onStatus: (text: string) => void) {
  const worker = new AnalyticsWorker();
  let ready = false;
  const sent = new Set<string>();
  const waiting = new Map<number, (r: ViewStats | null, err?: string) => void>();
  let seq = 0;
  let onReady: () => void = () => {};
  worker.onmessage = (e: MessageEvent) => {
    const d = e.data;
    if (d.type === "status") onStatus(d.text);
    else if (d.type === "ready") {
      ready = true;
      onStatus("");
      onReady();
    } else if (d.type === "result" || d.type === "error") {
      if (d.id === -1) onStatus(`DuckDB did not start: ${d.error}`);
      const f = waiting.get(d.id);
      waiting.delete(d.id);
      if (f) f(d.type === "result" ? d.result : null, d.error);
    }
  };
  worker.postMessage({ type: "init", models });
  return {
    isReady: () => ready,
    whenReady: (f: () => void) => {
      onReady = f;
    },
    /** The view's stats from the chunks under it (sent to the worker once each). */
    query(opts: {
      chunks: ChunkIn[];
      bbox: [number, number, number, number];
      model: string;
      divCells: BigUint64Array;
      divNames: string[];
    }): Promise<ViewStats | null> {
      for (const ch of opts.chunks) {
        if (sent.has(ch.key)) continue;
        sent.add(ch.key);
        // copies: the map keeps its own
        worker.postMessage({ type: "chunk", key: ch.key, r: ch.r, c: ch.c, w: ch.w, h: ch.h, first: ch.first.slice(), last: ch.last.slice() });
      }
      // the worker keeps 64; forget what it will have let go
      if (sent.size > 64) for (const k of [...sent].slice(0, sent.size - 64)) sent.delete(k);
      const id = ++seq;
      return new Promise((resolve, reject) => {
        waiting.set(id, (r, err) => (err ? reject(new Error(err)) : resolve(r)));
        worker.postMessage({
          type: "query",
          id,
          keys: opts.chunks.map((c) => c.key),
          bbox: opts.bbox,
          model: opts.model,
          divCells: opts.divCells,
          divNames: opts.divNames,
        });
      });
    },
  };
}
