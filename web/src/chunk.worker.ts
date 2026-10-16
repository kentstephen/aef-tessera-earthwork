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
/** The paint: a layer's score for one chunk, the year before `lastIdx` to it; `allIdx` the paint's other years. */
type Score = { type: "score"; id: number; kind: LayerKind; px0: number; py0: number; w: number; h: number; lastIdx: number; allIdx: number[] };
type Msg = { type: "init"; model: WorldModel } | Read | Score | { type: "abort"; id: number };

let model: WorldModel | null = null;
let mosaic: Promise<AefArray> | null = null;
const running = new Map<number, AbortController>();

// AlphaEarth's chunks by year, the last used last (each 64 bands x 256 x 256, 4 MB; 128 kept, about 500 MB, two years of a wide view and the next year read ahead): the paint
// steps one year at a time, and each pair shares a year with the next (2023-2024, then 2024-2025), so a step reads one
// year, not two
const AEF_KEEP = 128;
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
// a year is missing), 64 KB a chunk where its two years are 8 MB. Kept by "lastIdx:py0:px0:h:w", the last used last.
// Once a chunk's year is in, its other years are scored in the background (nearest year first, newest chunks first,
// only while nothing in view is waiting), so a step to another year draws from here with nothing to download
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
// the background: one job per chunk and year, taken nearest year first (the next step's for the whole view before the
// one after it), then the newest chunk first; only while nothing in view is waiting
type Ahead = { kind: LayerKind; g: { py0: number; px0: number; h: number; w: number }; y: number; dist: number; seq: number };
const ahead: Ahead[] = [];
const AHEAD_MAX = 600, AHEAD_RUN = 6;
let fgWaiting = 0, aheadRunning = 0, aheadSeq = 0;
const never = new AbortController().signal;
function nextAhead(): Ahead | undefined {
  let bi = -1;
  for (let i = 0; i < ahead.length; i++) {
    const j = ahead[i], b = bi < 0 ? null : ahead[bi];
    if (!b || j.dist < b.dist || (j.dist === b.dist && j.seq > b.seq)) bi = i;
  }
  return bi < 0 ? undefined : ahead.splice(bi, 1)[0];
}
function pumpAhead() {
  while (fgWaiting === 0 && aheadRunning < AHEAD_RUN) {
    const job = nextAhead();
    if (!job) return;
    if (scoreKept.has(scoreKey(job.kind, job.y, job.g))) continue;
    aheadRunning++;
    pairScore(job.kind, job.y, job.g, never)
      .catch(() => {})  // a year that cannot be read: left for a step to it
      .finally(() => { aheadRunning--; pumpAhead(); });
  }
}
async function score(m: Score) {
  const ac = new AbortController();
  running.set(m.id, ac);
  fgWaiting++;
  try {
    const sc = await pairScore(m.kind, m.lastIdx, m, ac.signal);
    const out = sc.slice();
    postMessage({ id: m.id, score: out }, { transfer: [out.buffer] as ArrayBuffer[] });
    // the chunk's other years, nearest first
    const rest = m.allIdx.filter((y) => y !== m.lastIdx && !scoreKept.has(scoreKey(m.kind, y, m)));
    const seq = ++aheadSeq, g = { py0: m.py0, px0: m.px0, h: m.h, w: m.w };
    for (const y of rest) ahead.push({ kind: m.kind, g, y, dist: Math.abs(y - m.lastIdx), seq });
    // the oldest chunks' jobs go first when there are too many (a long pan)
    if (ahead.length > AHEAD_MAX) { ahead.sort((p, q) => p.seq - q.seq); ahead.splice(0, ahead.length - AHEAD_MAX); }
  } catch (e) {
    const err = e as { name?: string; message?: string };
    postMessage({ id: m.id, error: { name: err?.name || "Error", message: err?.message || String(e) } });
  } finally {
    running.delete(m.id);
    fgWaiting--;
    pumpAhead();
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
