import * as zarr from "zarrita";

import { RES, X0, Y0 } from "./aef";
import { toUtm, zoneOf } from "./utm";

/**
 * TESSERA v1.1 (dClimate's global run) on Source Cooperative: one Zarr v3 group
 * per UTM zone, 10 m, 128 int8 bands per year (2017 to 2025), sharded 4096 px
 * with 32 x 32 inner chunks holding all bands.
 */
export const TESSERA_URL = "https://data.source.coop/tessera/tessera/zarr/v1.1-dclimate";
export const TES_BANDS = 128;
const TES_PX = 10;

/** Bits of the flags plane. */
export const TES_OK = 1;
export const TES_FLIP = 2;

/** TESSERA on one AlphaEarth chunk's pixels, both years of the window. */
export type TesseraChunk = {
  /** int8 [band, y, x], the chunk's own size: the first and last year. */
  first: Int8Array;
  last: Int8Array;
  /** uint8 [y, x]: TES_OK where both years are embedded, TES_FLIP where its radar came or went. */
  flags: Uint8Array;
};

type U16 = zarr.Array<"uint16", zarr.Readable>;
type Zone = {
  x0: number;
  y0: number;
  rows: number;
  cols: number;
  emb: zarr.Array<"int8", zarr.Readable>;
  scales: zarr.Array<"float32", zarr.Readable>;
  asc: U16;
  desc: U16;
};

/**
 * TESSERA's store as zarrita reads it, with fewer requests: zarrita asks for
 * each 32 x 32 inner chunk on its own (about 650 requests per AlphaEarth chunk,
 * counting the scales and radar counts, more than a browser keeps in flight).
 * The ranges asked of one shard in one turn of the event loop are sorted and
 * neighbors merged into one request, as the notebook's reader does, and at most
 * MAX_IN_FLIGHT requests are out at once across every zone. A failed request is
 * tried again, up to three times.
 */
const MAX_IN_FLIGHT = 48;
const MERGE_GAP = 1 << 16;
const MERGE_MAX = 1 << 23;
let inFlight = 0;
const waiting: (() => void)[] = [];

async function limited<T>(f: () => Promise<T>): Promise<T> {
  if (inFlight >= MAX_IN_FLIGHT) await new Promise<void>((go) => waiting.push(go));
  inFlight++;
  try {
    return await f();
  } finally {
    inFlight--;
    waiting.shift()?.();
  }
}

async function retried<T>(f: () => Promise<T>, signal?: AbortSignal): Promise<T> {
  for (let k = 0; ; k++) {
    try {
      return await limited(f);
    } catch (e) {
      if (k >= 3 || signal?.aborted) throw e;
      await new Promise((go) => setTimeout(go, 500 * 2 ** k));
    }
  }
}

type Ask = { offset: number; length: number; signal?: AbortSignal; resolve: (v: Uint8Array | undefined) => void; reject: (e: unknown) => void };

class MergingStore implements zarr.AsyncReadable {
  private inner: zarr.FetchStore;
  private asks = new Map<zarr.AbsolutePath, Ask[]>();

  constructor(url: string) {
    this.inner = new zarr.FetchStore(url);
  }

  get(key: zarr.AbsolutePath, opts?: { signal?: AbortSignal }) {
    return retried(() => this.inner.get(key, opts), opts?.signal);
  }

  getRange(key: zarr.AbsolutePath, range: zarr.RangeQuery, opts?: { signal?: AbortSignal }): Promise<Uint8Array | undefined> {
    if ("suffixLength" in range) return retried(() => this.inner.getRange(key, range, opts), opts?.signal);
    return new Promise((resolve, reject) => {
      let list = this.asks.get(key);
      if (!list) {
        list = [];
        this.asks.set(key, list);
        setTimeout(() => this.flush(key), 0);
      }
      list.push({ offset: range.offset, length: range.length, signal: opts?.signal, resolve, reject });
    });
  }

  private flush(key: zarr.AbsolutePath) {
    const list = (this.asks.get(key) || []).sort((a, b) => a.offset - b.offset);
    this.asks.delete(key);
    const groups: { start: number; end: number; asks: Ask[] }[] = [];
    for (const a of list) {
      const g = groups[groups.length - 1];
      if (g && a.offset - g.end <= MERGE_GAP && a.offset + a.length - g.start <= MERGE_MAX) {
        g.end = Math.max(g.end, a.offset + a.length);
        g.asks.push(a);
      } else groups.push({ start: a.offset, end: a.offset + a.length, asks: [a] });
    }
    for (const g of groups) {
      // a request serving several reads is let go only when all of them share one signal
      const sig = g.asks.every((a) => a.signal === g.asks[0].signal) ? g.asks[0].signal : undefined;
      retried(() => this.inner.getRange(key, { offset: g.start, length: g.end - g.start }, { signal: sig }), sig).then(
        (buf) => g.asks.forEach((a) => a.resolve(buf && buf.subarray(a.offset - g.start, a.offset - g.start + a.length))),
        (e) => g.asks.forEach((a) => a.reject(e)),
      );
    }
  }
}

const zones = new Map<number, Promise<Zone | null>>();

function openZone(zone: number): Promise<Zone | null> {
  let p = zones.get(zone);
  if (!p) {
    p = (async () => {
      const store = new MergingStore(`${TESSERA_URL}/utm${String(zone).padStart(2, "0")}`);
      const root = await zarr.open.v3(store, { kind: "group" });
      const tr = root.attrs["spatial:transform"] as number[];
      const arr = (name: string) => zarr.open.v3(root.resolve(name), { kind: "array" });
      const [emb, scales, asc, desc] = await Promise.all(
        ["embeddings", "scales", "s1_asc_obs_count", "s1_desc_obs_count"].map(arr),
      );
      if (!emb.is("int8") || !scales.is("float32") || !asc.is("uint16") || !desc.is("uint16")) {
        throw new Error(`TESSERA zone ${zone}: unexpected data types`);
      }
      const [, , rows, cols] = emb.shape;
      return { x0: tr[2], y0: tr[5], rows, cols, emb, scales, asc, desc };
    })();
    // a failed open is tried again on the next chunk
    p.catch(() => zones.delete(zone));
    zones.set(zone, p);
  }
  return p;
}

/**
 * TESSERA at each pixel center of the AlphaEarth chunk whose top left pixel is
 * (px0, py0) on the mosaic's grid, w x h pixels: the TESSERA pixel holding the
 * center (r = floor((y0 - N) / 10), c = floor((E - x0) / 10)), each pixel in
 * the UTM zone of its own longitude. Not embedded, or outside the store: flags 0.
 */
export async function tesseraOnChunk(
  px0: number,
  py0: number,
  w: number,
  h: number,
  firstIdx: number,
  lastIdx: number,
  signal?: AbortSignal,
): Promise<TesseraChunk> {
  const plane = w * h;
  const first = new Int8Array(TES_BANDS * plane);
  const last = new Int8Array(TES_BANDS * plane);
  const flags = new Uint8Array(plane);
  // the columns of each zone (a chunk on a zone line reads both)
  const byZone = new Map<number, [number, number]>();
  for (let i = 0; i < w; i++) {
    const z = zoneOf(X0 + (px0 + i + 0.5) * RES);
    const span = byZone.get(z);
    if (span) span[1] = i + 1;
    else byZone.set(z, [i, i + 1]);
  }
  await Promise.all(
    [...byZone].map(async ([zone, [i0, i1]]) => {
      const zn = await openZone(zone);
      if (!zn) return;
      // each pixel's TESSERA row and column, and the window holding them all
      const n = (i1 - i0) * h;
      const rr = new Int32Array(n), cc = new Int32Array(n);
      const en = new Float64Array(2);
      let r0 = Infinity, r1 = -Infinity, c0 = Infinity, c1 = -Infinity;
      for (let j = 0, k = 0; j < h; j++) {
        const lat = Y0 - (py0 + j + 0.5) * RES;
        for (let i = i0; i < i1; i++, k++) {
          toUtm(X0 + (px0 + i + 0.5) * RES, lat, zone, en);
          const r = Math.floor((zn.y0 - en[1]) / TES_PX), c = Math.floor((en[0] - zn.x0) / TES_PX);
          rr[k] = r;
          cc[k] = c;
          if (r < r0) r0 = r;
          if (r > r1) r1 = r;
          if (c < c0) c0 = c;
          if (c > c1) c1 = c;
        }
      }
      r0 = Math.max(r0, 0);
      c0 = Math.max(c0, 0);
      r1 = Math.min(r1, zn.rows - 1);
      c1 = Math.min(c1, zn.cols - 1);
      if (r1 < r0 || c1 < c0) return;
      const ys = zarr.slice(r0, r1 + 1), xs = zarr.slice(c0, c1 + 1);
      const ww = c1 - c0 + 1, wplane = (r1 - r0 + 1) * ww;
      const year = (t: number) =>
        Promise.all([
          zarr.get(zn.emb, [t, null, ys, xs], { signal }),
          zarr.get(zn.scales, [t, ys, xs], { signal }),
          zarr.get(zn.asc, [t, ys, xs], { signal }),
          zarr.get(zn.desc, [t, ys, xs], { signal }),
        ]);
      const [[ef, sf, af, df], [el, sl, al, dl]] = await Promise.all([year(firstIdx), year(lastIdx)]);
      const embF = ef.data as Int8Array, embL = el.data as Int8Array;
      const scF = sf.data as Float32Array, scL = sl.data as Float32Array;
      const ascF = af.data as Uint16Array, descF = df.data as Uint16Array;
      const ascL = al.data as Uint16Array, descL = dl.data as Uint16Array;
      // the window's pixel under each chunk pixel (-1: outside the store or not embedded)
      const src = new Int32Array(n).fill(-1), dst = new Int32Array(n);
      for (let j = 0, k = 0; j < h; j++)
        for (let i = i0; i < i1; i++, k++) {
          dst[k] = j * w + i;
          const r = rr[k], c = cc[k];
          if (r < r0 || r > r1 || c < c0 || c > c1) continue;
          const s = (r - r0) * ww + (c - c0);
          const okF = scF[s] > 0, okL = scL[s] > 0; // false for NaN
          const radarF = ascF[s] + descF[s] > 0, radarL = ascL[s] + descL[s] > 0;
          flags[dst[k]] = (okF && okL ? TES_OK : 0) | (radarF !== radarL ? TES_FLIP : 0);
          src[k] = s;
        }
      for (let b = 0; b < TES_BANDS; b++) {
        const ob = b * plane, ib = b * wplane;
        for (let k = 0; k < n; k++) {
          const s = src[k];
          if (s < 0) continue;
          first[ob + dst[k]] = embF[ib + s];
          last[ob + dst[k]] = embL[ib + s];
        }
      }
    }),
  );
  return { first, last, flags };
}
