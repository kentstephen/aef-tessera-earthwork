import * as zarr from "zarrita";

import { type AefArray, NUM_BANDS, openMosaic } from "./aef";
import { aefScores, type WorldModel } from "./earthwork-cpu";
import { tesseraOnChunk } from "./tessera";

/**
 * Every chunk is read here, off the page's thread: AlphaEarth's two years, its own scores (which say
 * where TESSERA is worth reading), then TESSERA on its pixels (not for AlphaEarth's layer alone). Downloads, decoding and resampling never
 * hold up the map, the imagery or the keys; the page only makes the textures.
 */
type Read = { type: "read"; id: number; px0: number; py0: number; w: number; h: number; firstIdx: number; lastIdx: number; aefOnly?: boolean };
type Msg = { type: "init"; model: WorldModel } | Read | { type: "abort"; id: number };

let model: WorldModel | null = null;
let mosaic: Promise<AefArray> | null = null;
const running = new Map<number, AbortController>();

// AlphaEarth's chunks by year, the last used last (each 64 bands x 256 x 256, 4 MB; 48 kept, about 200 MB): the paint
// steps one year at a time, and each pair shares a year with the next (2023-2024, then 2024-2025), so a step reads one
// year, not two
const AEF_KEEP = 48;
// a read is shared by the tiles waiting on it, and stopped only when every one of them has let go
type Kept = { p: Promise<zarr.Chunk<"int8">>; ac: AbortController; waiting: number; done: boolean };
const aefKept = new Map<string, Kept>();
const aborted = () => Object.assign(new Error("aborted"), { name: "AbortError" });
function aefChunk(arr: AefArray, idx: number, py0: number, px0: number, h: number, w: number, signal: AbortSignal) {
  const k = `${idx}:${py0}:${px0}:${h}:${w}`;
  let e = aefKept.get(k);
  if (e) {
    aefKept.delete(k);
    aefKept.set(k, e);
  } else {
    const ac = new AbortController();
    const p = zarr.get(arr, [idx, null, zarr.slice(py0, py0 + h), zarr.slice(px0, px0 + w)], { signal: ac.signal }) as Promise<zarr.Chunk<"int8">>;
    const ent: Kept = { p, ac, waiting: 0, done: false };
    p.then(() => (ent.done = true), () => aefKept.get(k) === ent && aefKept.delete(k));
    aefKept.set(k, ent);
    while (aefKept.size > AEF_KEEP) aefKept.delete(aefKept.keys().next().value as string);
    e = ent;
  }
  if (e.done) return e.p;
  const ent = e;
  ent.waiting++;
  return new Promise<zarr.Chunk<"int8">>((resolve, reject) => {
    const letGo = () => {
      reject(aborted());
      if (--ent.waiting === 0 && !ent.done) {
        ent.ac.abort();
        if (aefKept.get(k) === ent) aefKept.delete(k);
      }
    };
    if (signal.aborted) return letGo();
    signal.addEventListener("abort", letGo, { once: true });
    ent.p.then(
      (v) => { signal.removeEventListener("abort", letGo); ent.waiting--; resolve(v); },
      (err) => { signal.removeEventListener("abort", letGo); ent.waiting--; reject(err); },
    );
  });
}

async function read(m: Read) {
  const ac = new AbortController();
  running.set(m.id, ac);
  const signal = ac.signal;
  try {
    if (!mosaic) mosaic = openMosaic().then((o) => o.arr);
    mosaic.catch(() => (mosaic = null));
    const arr = await mosaic;
    const [f, l] = await Promise.all([
      aefChunk(arr, m.firstIdx, m.py0, m.px0, m.h, m.w, signal),
      aefChunk(arr, m.lastIdx, m.py0, m.px0, m.h, m.w, signal),
    ]);
    for (const c of [f, l])
      if (c.shape.length !== 3 || c.shape[0] !== NUM_BANDS || c.shape[1] !== m.h || c.shape[2] !== m.w)
        throw new Error(`Unexpected chunk shape [${c.shape.join(", ")}] for a ${m.w} x ${m.h} tile`);
    // copies: the kept chunks stay here, the copies go to the page
    const aefFirst = (f.data as Int8Array).slice(), aefLast = (l.data as Int8Array).slice();
    // AlphaEarth's layer alone needs no TESSERA
    if (m.aefOnly) {
      const none = new Int8Array(0), noFlags = new Uint8Array(0);
      postMessage({ id: m.id, aefFirst, aefLast, tesFirst: none, tesLast: none, tesFlags: noFlags }, { transfer: [aefFirst.buffer, aefLast.buffer] as ArrayBuffer[] });
      return;
    }
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
