import {
  COLORMAP_INDEX,
  Colormap,
  createColormapTexture,
  decodeColormapSprite,
} from "@developmentseed/deck.gl-raster/gpu-modules";
import colormapsUrl from "@developmentseed/deck.gl-raster/gpu-modules/colormaps.png?url";
import { ZarrLayer } from "@developmentseed/deck.gl-zarr";
import type { Device, Texture } from "@luma.gl/core";
import * as zarr from "zarrita";

import { type AefArray, makeGetTileData, openMosaic, type PairTileData, YEAR_ORIGIN } from "./aef";
import { EarthworkAlpha, EarthworkScore } from "./earthwork-score";

export type Model = { w: number[]; b: number };

/** The AEF mosaic's grid (EPSG:4326, pixel registered). */
const RES = 8.983111749910169e-5;
const X0 = -180;
const Y0 = 83.68570533713473;
const CHUNK = 256;
const GATE_LO = 0.05;
const GATE_HI = 0.15;

/**
 * The earthwork map layer: AlphaEarth's two years per chunk, scored on the GPU.
 * Also keeps each chunk's raw bytes (the last few dozen) so a point can be
 * scored on the CPU for the tooltip and the card.
 */
export async function createEarthwork(modelUrls: Record<string, string>, beforeId: () => string) {
  const models: Record<string, Model> = {};
  for (const [k, url] of Object.entries(modelUrls)) models[k] = await (await fetch(url)).json();
  const { arr, attrs } = await openMosaic();

  let gpu: { weights: Record<string, Texture>; colormap: Texture } | null = null;
  let gpuPending = false;
  const colormapBytes = fetch(colormapsUrl).then((r) => r.arrayBuffer());
  let onReady: () => void = () => {};

  function onDevice(device: Device) {
    if (gpu || gpuPending) return;
    gpuPending = true;
    const weight = (m: Model) =>
      device.createTexture({
        format: "r32float",
        width: 256,
        height: 1,
        mipLevels: 1,
        data: new Float32Array(m.w),
        sampler: { minFilter: "nearest", magFilter: "nearest" },
      });
    colormapBytes
      .then((buf) => decodeColormapSprite(buf))
      .then((img) => {
        const weights: Record<string, Texture> = {};
        for (const [k, m] of Object.entries(models)) weights[k] = weight(m);
        gpu = { weights, colormap: createColormapTexture(device, img) };
        onReady();
      });
  }

  // the raw chunks by "first-last:row-col", the last used last. A view holds more chunks than are
  // kept (about 70 at zoom 12.7 in a laptop window, hundreds full screen or paired; each is 8 MB),
  // so a point whose chunk was let go reads it again (readAt)
  type Chunk = { first: Int8Array; last: Int8Array; w: number; h: number };
  const chunks = new Map<string, Chunk>();
  const KEEP = 48;
  const keyOf = (first: number, last: number, r: number, c: number) => `${first}-${last}:${r}-${c}`;
  const keepChunk = (k: string, ch: Chunk) => {
    chunks.delete(k);
    chunks.set(k, ch);
    while (chunks.size > KEEP) chunks.delete(chunks.keys().next().value as string);
  };
  const reading = new Map<string, Promise<void>>();

  let tilesRead = 0;

  function layer(opts: { visible: boolean; first: number; last: number; model: string; minZoom: number; onTile?: () => void }) {
    const { visible, first, last, model, minZoom, onTile } = opts;
    const firstIdx = first - YEAR_ORIGIN;
    const lastIdx = last - YEAR_ORIGIN;
    const g = gpu;
    const m = models[model] || Object.values(models)[0];
    const read = makeGetTileData(firstIdx, lastIdx, onDevice);
    return new ZarrLayer<zarr.Readable, "int8", PairTileData>({
      id: `earthwork-${first}-${last}`,
      node: arr as AefArray,
      metadata: attrs,
      selection: { time: firstIdx, band: null },
      getTileData: async (a, o) => {
        const d = await read(a, o);
        const ys = o.sliceSpec[2] as zarr.Slice, xs = o.sliceSpec[3] as zarr.Slice;
        const r = Math.floor((ys.start ?? 0) / CHUNK), c = Math.floor((xs.start ?? 0) / CHUNK);
        keepChunk(keyOf(first, last, r, c), { first: d.first, last: d.last, w: d.width, h: d.height });
        tilesRead++;
        return d;
      },
      renderTile: (data: PairTileData) => ({
        renderPipeline: g
          ? [
              {
                module: EarthworkScore,
                props: {
                  firstTex: data.firstTex,
                  lastTex: data.lastTex,
                  weightTex: g.weights[model] || Object.values(g.weights)[0],
                  bias: m.b,
                  gateLo: GATE_LO,
                  gateHi: GATE_HI,
                },
              },
              { module: Colormap, props: { colormapTexture: g.colormap, colormapIndex: COLORMAP_INDEX.viridis } },
              { module: EarthworkAlpha },
            ]
          : [],
      }),
      visible,
      minZoom,
      // source.coop answers over HTTP/2, so more than the browser's 6 per host
      maxRequests: 12,
      // each tile is two years x 64 bands (8 MB on the GPU)
      maxCacheSize: 24,
      onTileLoad: () => onTile && onTile(),
      updateTriggers: { renderTile: [model, !!g] },
      // @ts-expect-error beforeId is read by the MapLibre overlay, not typed on the layer
      beforeId: beforeId(),
    });
  }

  /**
   * The earthwork score at a point, as the shader computes it, from the chunk
   * already read; null where that chunk is not in hand or a year has no data.
   */
  function scoreAt(lon: number, lat: number, first: number, last: number, model: string): number | null {
    const px = Math.floor((lon - X0) / RES), py = Math.floor((Y0 - lat) / RES);
    const r = Math.floor(py / CHUNK), c = Math.floor(px / CHUNK);
    const k = keyOf(first, last, r, c), ch = chunks.get(k);
    const m = models[model];
    if (!ch || !m) return null;
    keepChunk(k, ch);
    const ix = px - c * CHUNK, iy = py - r * CHUNK;
    if (ix < 0 || iy < 0 || ix >= ch.w || iy >= ch.h) return null;
    const plane = ch.w * ch.h, at = iy * ch.w + ix;
    const b = new Float64Array(64), a = new Float64Array(64);
    let nb = 0, na = 0;
    for (let i = 0; i < 64; i++) {
      const vb = ch.first[i * plane + at], va = ch.last[i * plane + at];
      if (vb === -128 || va === -128) return null;
      const fb = vb / 127.5, fa = va / 127.5;
      b[i] = fb * Math.abs(fb);
      a[i] = fa * Math.abs(fa);
      nb += b[i] * b[i];
      na += a[i] * a[i];
    }
    const ib = 1 / Math.max(Math.sqrt(nb), 1e-9), ia = 1 / Math.max(Math.sqrt(na), 1e-9);
    let z = m.b, cos = 0;
    for (let i = 0; i < 64; i++) {
      const bi = b[i] * ib, ai = a[i] * ia, d = ai - bi;
      cos += ai * bi;
      z += m.w[i] * bi + m.w[64 + i] * ai + m.w[128 + i] * ai * bi + m.w[192 + i] * d * d;
    }
    const gate = Math.min(1, Math.max(0, (1 - cos - GATE_LO) / (GATE_HI - GATE_LO)));
    return gate / (1 + Math.exp(-z));
  }

  /**
   * Read the chunk under a point for this window if it is not in hand (a view holds more chunks
   * than are kept); resolves once scoreAt can answer there, or with nothing where it cannot.
   */
  function readAt(lon: number, lat: number, first: number, last: number): Promise<void> {
    const px = Math.floor((lon - X0) / RES), py = Math.floor((Y0 - lat) / RES);
    const r = Math.floor(py / CHUNK), c = Math.floor(px / CHUNK), k = keyOf(first, last, r, c);
    if (chunks.has(k)) return Promise.resolve();
    let p = reading.get(k);
    if (!p) {
      const [, , H, W] = arr.shape;
      if (px < 0 || py < 0 || px >= W || py >= H) return Promise.resolve();
      const ys = zarr.slice(r * CHUNK, Math.min((r + 1) * CHUNK, H)), xs = zarr.slice(c * CHUNK, Math.min((c + 1) * CHUNK, W));
      p = Promise.all([first, last].map((y) => zarr.get(arr as AefArray, [y - YEAR_ORIGIN, null, ys, xs])))
        .then(([f, l]) => keepChunk(k, { first: f.data as Int8Array, last: l.data as Int8Array, w: f.shape[2], h: f.shape[1] }))
        .catch(() => {})
        .finally(() => reading.delete(k));
      reading.set(k, p);
    }
    return p;
  }

  /** The chunks in hand for this window that lie under a box, with their raw bytes. */
  function chunksIn(bbox: [number, number, number, number], first: number, last: number) {
    const [W, S, E, N] = bbox;
    const c0 = Math.floor((W - X0) / RES / CHUNK), c1 = Math.floor((E - X0) / RES / CHUNK);
    const r0 = Math.floor((Y0 - N) / RES / CHUNK), r1 = Math.floor((Y0 - S) / RES / CHUNK);
    const out = [];
    for (let r = r0; r <= r1; r++)
      for (let c = c0; c <= c1; c++) {
        const k = keyOf(first, last, r, c), ch = chunks.get(k);
        if (ch) out.push({ key: k, r, c, w: ch.w, h: ch.h, first: ch.first, last: ch.last });
      }
    return { chunks: out, wanted: (r1 - r0 + 1) * (c1 - c0 + 1) };
  }

  return {
    layer,
    scoreAt,
    readAt,
    chunksIn,
    models,
    tilesRead: () => tilesRead,
    whenReady: (f: () => void) => {
      onReady = f;
    },
  };
}

export { COLORMAP_INDEX };
