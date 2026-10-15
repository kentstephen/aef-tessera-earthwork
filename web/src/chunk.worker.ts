import * as zarr from "zarrita";

import { type AefArray, NUM_BANDS, openMosaic } from "./aef";
import { aefScores, type WorldModel } from "./earthwork-cpu";
import { tesseraOnChunk } from "./tessera";

/**
 * Every chunk is read here, off the page's thread: AlphaEarth's two years, its own scores (which say
 * where TESSERA is worth reading), then TESSERA on its pixels. Downloads, decoding and resampling never
 * hold up the map, the imagery or the keys; the page only makes the textures.
 */
type Read = { type: "read"; id: number; px0: number; py0: number; w: number; h: number; firstIdx: number; lastIdx: number };
type Msg = { type: "init"; model: WorldModel } | Read | { type: "abort"; id: number };

let model: WorldModel | null = null;
let mosaic: Promise<AefArray> | null = null;
const running = new Map<number, AbortController>();

async function read(m: Read) {
  const ac = new AbortController();
  running.set(m.id, ac);
  const signal = ac.signal;
  try {
    if (!mosaic) mosaic = openMosaic().then((o) => o.arr);
    mosaic.catch(() => (mosaic = null));
    const arr = await mosaic;
    const ys = zarr.slice(m.py0, m.py0 + m.h), xs = zarr.slice(m.px0, m.px0 + m.w);
    const [f, l] = await Promise.all([
      zarr.get(arr, [m.firstIdx, null, ys, xs], { signal }),
      zarr.get(arr, [m.lastIdx, null, ys, xs], { signal }),
    ]);
    for (const c of [f, l])
      if (c.shape.length !== 3 || c.shape[0] !== NUM_BANDS || c.shape[1] !== m.h || c.shape[2] !== m.w)
        throw new Error(`Unexpected chunk shape [${c.shape.join(", ")}] for a ${m.w} x ${m.h} tile`);
    const aefFirst = f.data as Int8Array, aefLast = l.data as Int8Array;
    const scores = model ? aefScores(aefFirst, aefLast, m.w * m.h, model) : undefined;
    const t = await tesseraOnChunk(m.px0, m.py0, m.w, m.h, m.firstIdx, m.lastIdx, signal, scores);
    const out = { id: m.id, aefFirst, aefLast, tesFirst: t.first, tesLast: t.last, tesFlags: t.flags };
    postMessage(out, { transfer: [aefFirst.buffer, aefLast.buffer, t.first.buffer, t.last.buffer, t.flags.buffer] as ArrayBuffer[] });
  } catch (e) {
    const err = e as { name?: string; message?: string };
    postMessage({ id: m.id, error: { name: err?.name || "Error", message: err?.message || String(e) } });
  } finally {
    running.delete(m.id);
  }
}

self.onmessage = (ev: MessageEvent<Msg>) => {
  const m = ev.data;
  if (m.type === "init") model = m.model;
  else if (m.type === "abort") running.get(m.id)?.abort();
  else void read(m);
};
