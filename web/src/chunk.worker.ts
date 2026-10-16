import * as zarr from "zarrita";

import { type AefArray, NUM_BANDS, openMosaic } from "./aef";
import { aefScores, type LayerKind, scorePixel, type WorldModel } from "./earthwork-cpu";
import { tesseraOnChunk } from "./tessera";

/**
 * Every chunk is read here, off the page's thread: AlphaEarth's two years, its own scores (which say
 * where TESSERA is worth reading), then TESSERA on its pixels (not for AlphaEarth's layer alone). Downloads, decoding and resampling never
 * hold up the map, the imagery or the keys; the page only makes the textures.
 */
type Read = { type: "read"; id: number; px0: number; py0: number; w: number; h: number; firstIdx: number; lastIdx: number; aefOnly?: boolean };
/** The paint: a layer's score for one chunk, for every year in `allIdx` (each from the year before it). */
type Score = { type: "score"; id: number; kind: LayerKind; px0: number; py0: number; w: number; h: number; allIdx: number[] };
type Msg = { type: "init"; model: WorldModel } | Read | Score | { type: "abort"; id: number };

let model: WorldModel | null = null;
let mosaic: Promise<AefArray> | null = null;
const running = new Map<number, AbortController>();

// AlphaEarth's chunks by year, the last used last (each 64 bands x 256 x 256, 4 MB; 64 kept, about 250 MB): a chunk's
// years are read once for all the pairs that share them (2022 for 2021-2022 and 2022-2023)
const AEF_KEEP = 64;
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

// ---- the paint's scores: per layer, chunk and year, the layer's score from the year before (TESSERA read for the
// aggregate and TESSERA's own, where AlphaEarth points, as for the map), as 1 + 254 x score (0 where
// a year is missing), 64 KB a chunk and year where its two years are 8 MB. Kept by "kind:lastIdx:py0:px0:h:w", the
// last used last, so a layer switched back to draws from here
const SCORE_KEEP = 1500;
const scoreKept = new Map<string, Promise<Uint8Array>>();
const scoreKey = (kind: LayerKind, lastIdx: number, m: { py0: number; px0: number; h: number; w: number }) => `${kind}:${lastIdx}:${m.py0}:${m.px0}:${m.h}:${m.w}`;
function encodeScores(sc: Float32Array): Uint8Array {
  const out = new Uint8Array(sc.length);
  for (let i = 0; i < sc.length; i++) out[i] = sc[i] < 0 ? 0 : 1 + Math.round(254 * Math.min(1, sc[i]));
  return out;
}
async function pairScore(kind: LayerKind, lastIdx: number, g: { py0: number; px0: number; h: number; w: number }, signal: AbortSignal): Promise<Uint8Array> {
  const k = scoreKey(kind, lastIdx, g);
  const kept = scoreKept.get(k);
  if (kept) {
    scoreKept.delete(k);
    scoreKept.set(k, kept);
    return kept;
  }
  if (!mosaic) mosaic = openMosaic().then((o) => o.arr);
  mosaic.catch(() => (mosaic = null));
  const arr = await mosaic;
  const p = Promise.all([
    aefChunk(arr, lastIdx - 1, g.py0, g.px0, g.h, g.w, signal),
    aefChunk(arr, lastIdx, g.py0, g.px0, g.h, g.w, signal),
  ]).then(async ([f, l]) => {
    const aefFirst = f.data as Int8Array, aefLast = l.data as Int8Array, plane = g.w * g.h;
    const aef = aefScores(aefFirst, aefLast, plane, model!);
    if (kind === "aef") return encodeScores(aef);
    const t = await tesseraOnChunk(g.px0, g.py0, g.w, g.h, lastIdx - 1, lastIdx, signal, aef);
    const out = new Float32Array(plane), px = { aefFirst, aefLast, tesFirst: t.first, tesLast: t.last, tesFlags: t.flags, plane, at: 0 };
    for (let at = 0; at < plane; at++) {
      px.at = at;
      const v = scorePixel(kind, model!, px);
      out[at] = v == null ? -1 : v;
    }
    return encodeScores(out);
  });
  scoreKept.set(k, p);
  p.catch(() => scoreKept.get(k) === p && scoreKept.delete(k));
  while (scoreKept.size > SCORE_KEEP) scoreKept.delete(scoreKept.keys().next().value as string);
  return p;
}
// every year of a chunk at once, packed one year per channel (RGBA: up to four years), so a step between years
// only reads another channel on the GPU
async function score(m: Score) {
  const ac = new AbortController();
  running.set(m.id, ac);
  try {
    const plane = m.w * m.h, n = m.allIdx.length;
    const each = await Promise.all(m.allIdx.map((y) => pairScore(m.kind, y, m, ac.signal)));
    const out = new Uint8Array(4 * plane);
    for (let c = 0; c < n && c < 4; c++) {
      const sc = each[c];
      for (let i = 0; i < plane; i++) out[4 * i + c] = sc[i];
    }
    postMessage({ id: m.id, score: out }, { transfer: [out.buffer] as ArrayBuffer[] });
  } catch (e) {
    const err = e as { name?: string; message?: string };
    postMessage({ id: m.id, error: { name: err?.name || "Error", message: err?.message || String(e) } });
  } finally {
    running.delete(m.id);
  }
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
  else if (m.type === "score") void score(m);
  else void read(m);
};
