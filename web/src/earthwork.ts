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

import { type AefArray, bandTexture, CHUNK, openMosaic, type PairTileData, RES, X0, Y0, YEAR_ORIGIN } from "./aef";
import ChunkWorker from "./chunk.worker.ts?worker";
import { GATE_HI, GATE_LO, LAYERS, type LayerKind, scorePixel, type WorldModel } from "./earthwork-cpu";
import { EarthworkAggregate, EarthworkAlpha, EarthworkCut, EarthworkFlare, EarthworkScore, EarthworkScoreTex, EarthworkTessera, EarthworkYearTex } from "./earthwork-score";
import { TES_BANDS } from "./tessera";

export type { LayerKind, WorldModel };
/** The layers' chunks, the paint's and the year map's: a layer's score scored ahead, a year at a time. */
export type ChunkKind = LayerKind | `paint-${LayerKind}` | `years-${LayerKind}`;
const paintOf = (kind: ChunkKind): LayerKind | null => (kind.startsWith("paint-") || kind.startsWith("years-") ? (kind.slice(6) as LayerKind) : null);

/** One chunk of the paint: every year's score on the GPU (rgba8unorm, a year per channel, 1 + 254 x score, 0 where a year is missing). */
type PaintTile = { width: number; height: number; byteLength: number; key: string; scoreTex: Texture };
/** One chunk of the year map: the years each pixel was over the layer's cutoff, a bit a year (r8unorm). */
type YearTile = { width: number; height: number; byteLength: number; key: string; yearTex: Texture };

/** One chunk on the GPU: AlphaEarth's two years, TESSERA's two years on its pixels, and the flags (none of TESSERA's for AlphaEarth's layer alone). */
type TileData = PairTileData & {
  /** The chunk's key (keyOf), for the chunks loaded in view. */
  key: string;
  tesFirstTex?: Texture;
  tesLastTex?: Texture;
  tesFlagsTex?: Texture;
  tesFirst: Int8Array;
  tesLast: Int8Array;
  tesFlags: Uint8Array;
};

/**
 * The earthwork map layer: per 256 x 256 chunk of the AlphaEarth mosaic, both
 * years of AlphaEarth and of TESSERA (resampled onto AlphaEarth's pixels),
 * scored on the GPU by the layer chosen (aggregate, AlphaEarth or TESSERA, the
 * world recipe's three). Also keeps the last chunks' raw bytes so a point can be
 * scored on the CPU for the tooltip and the card.
 */
export async function createEarthwork(modelUrl: string, beforeId: () => string) {
  const model: WorldModel = await (await fetch(modelUrl)).json();
  const { arr, attrs } = await openMosaic();

  // the chunks are read in a worker (chunk.worker.ts): AlphaEarth, its scores, TESSERA where they point
  type Read = { aefFirst: Int8Array; aefLast: Int8Array; tesFirst: Int8Array; tesLast: Int8Array; tesFlags: Uint8Array };
  const worker = new ChunkWorker();
  worker.postMessage({ type: "init", model });
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const asked = new Map<number, { resolve: (r: any) => void; reject: (e: unknown) => void }>();
  let askId = 0;
  worker.onmessage = (ev: MessageEvent<(Read | { score: Uint8Array }) & { id: number; error?: { name: string; message: string } }>) => {
    const { id, error, ...r } = ev.data;
    const a = asked.get(id);
    if (!a) return;
    asked.delete(id);
    if (error) a.reject(Object.assign(new Error(error.message), { name: error.name }));
    else a.resolve(r);
  };
  function readChunk(px0: number, py0: number, w: number, h: number, firstIdx: number, lastIdx: number, aefOnly: boolean, signal?: AbortSignal): Promise<Read> {
    if (signal?.aborted) return Promise.reject(Object.assign(new Error("aborted"), { name: "AbortError" }));
    const id = ++askId;
    return new Promise<Read>((resolve, reject) => {
      asked.set(id, { resolve, reject });
      worker.postMessage({ type: "read", id, px0, py0, w, h, firstIdx, lastIdx, aefOnly });
      signal?.addEventListener("abort", () => {
        if (!asked.has(id)) return;
        asked.delete(id);
        worker.postMessage({ type: "abort", id });
        reject(Object.assign(new Error("aborted"), { name: "AbortError" }));
      });
    });
  }

  /** The paint's scores for one chunk and layer, every year in `allIdx` (each from the year before), a year per channel. */
  function scoreChunk(kind: LayerKind, px0: number, py0: number, w: number, h: number, allIdx: number[], signal?: AbortSignal): Promise<Uint8Array> {
    if (signal?.aborted) return Promise.reject(Object.assign(new Error("aborted"), { name: "AbortError" }));
    const id = ++askId;
    return new Promise<{ score: Uint8Array }>((resolve, reject) => {
      asked.set(id, { resolve, reject });
      worker.postMessage({ type: "score", id, kind, px0, py0, w, h, allIdx });
      signal?.addEventListener("abort", () => {
        if (!asked.has(id)) return;
        asked.delete(id);
        worker.postMessage({ type: "abort", id });
        reject(Object.assign(new Error("aborted"), { name: "AbortError" }));
      });
    }).then((r) => r.score);
  }

  let gpu: { weights: Record<LayerKind, Texture>; colormap: Texture } | null = null;
  let gpuPending = false;
  const colormapBytes = fetch(colormapsUrl).then((r) => r.arrayBuffer());
  let onReady: () => void = () => {};

  function onDevice(device: Device) {
    if (gpu || gpuPending) return;
    gpuPending = true;
    const weight = (w: number[]) =>
      device.createTexture({
        format: "r32float",
        width: w.length,
        height: 1,
        mipLevels: 1,
        data: new Float32Array(w),
        sampler: { minFilter: "nearest", magFilter: "nearest" },
      });
    colormapBytes
      .then((buf) => decodeColormapSprite(buf))
      .then((img) => {
        const weights = Object.fromEntries(LAYERS.map((k) => [k, weight(model[k].w)])) as Record<LayerKind, Texture>;
        gpu = { weights, colormap: createColormapTexture(device, img) };
        onReady();
      });
  }

  // the raw chunks by "first-last:row-col" ("first-last-aef:..." for AlphaEarth's layer alone, read without TESSERA),
  // the last used last. A view holds more chunks than are
  // kept (about 70 at zoom 12.7 in a laptop window; each is 24 MB with TESSERA), so a point whose
  // chunk was let go reads it again (readAt)
  type Chunk = { aefFirst: Int8Array; aefLast: Int8Array; tesFirst: Int8Array; tesLast: Int8Array; tesFlags: Uint8Array; w: number; h: number };
  const chunks = new Map<string, Chunk>();
  const KEEP = 16;
  const tag = (kind: ChunkKind) => (kind === "aef" ? "-aef" : paintOf(kind) ? `-${kind}` : "");
  // the paint's chunks hold every year, so their key has none
  const keyOf = (first: number, last: number, kind: ChunkKind, r: number, c: number) => (paintOf(kind) ? `${kind}:${r}-${c}` : `${first}-${last}${tag(kind)}:${r}-${c}`);
  // the paint's scores by keyOf, the last used last (256 KB each, every year): the tooltip's and the card's
  type PaintChunk = { s: Uint8Array; w: number; h: number; years: number[]; stride: number };
  const paintScores = new Map<string, PaintChunk>();
  // the paint's years and the year map's, as their last layers asked for them (for a point read on its own, readAt)
  const yearsFor: { paint: number[]; years: number[] } = { paint: [], years: [] };
  const yearsOf = (kind: ChunkKind) => (kind.startsWith("years-") ? yearsFor.years : yearsFor.paint);
  const strideOf = (n: number) => (n <= 4 ? 4 : 8);
  const keepPaint = (k: string, v: PaintChunk) => {
    paintScores.delete(k);
    paintScores.set(k, v);
    while (paintScores.size > 400) paintScores.delete(paintScores.keys().next().value as string);
  };
  const keepChunk = (k: string, ch: Chunk) => {
    chunks.delete(k);
    chunks.set(k, ch);
    while (chunks.size > KEEP) chunks.delete(chunks.keys().next().value as string);
  };
  const reading = new Map<string, Promise<void>>();

  let tilesRead = 0;
  const NONE = new Int8Array(0), NONE_U8 = new Uint8Array(0);
  // the chunks on the GPU now, by key: whether the view's chunks are in
  const loaded = new Set<string>();

  const flagsTexture = (device: Device, data: Uint8Array, width: number, height: number) =>
    device.createTexture({
      format: "r8uint",
      width,
      height,
      mipLevels: 1,
      data,
      sampler: { minFilter: "nearest", magFilter: "nearest", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge" },
    });

  function renderPipeline(kind: LayerKind, data: TileData, cut: number | null) {
    const g = gpu;
    if (!g) return [];
    const score =
      kind === "aef"
        ? {
            module: EarthworkScore,
            props: { firstTex: data.firstTex, lastTex: data.lastTex, weightTex: g.weights.aef, bias: model.aef.b, gateLo: GATE_LO, gateHi: GATE_HI },
          }
        : kind === "tessera"
          ? {
              module: EarthworkTessera,
              props: { tesFirst: data.tesFirstTex, tesLast: data.tesLastTex, tesFlags: data.tesFlagsTex, weightTex: g.weights.tessera, bias: model.tessera.b },
            }
          : {
              module: EarthworkAggregate,
              props: {
                aefFirst: data.firstTex,
                aefLast: data.lastTex,
                tesFirst: data.tesFirstTex,
                tesLast: data.tesLastTex,
                tesFlags: data.tesFlagsTex,
                weightTex: g.weights.aggregate,
                bias: model.aggregate.b,
                gateLo: GATE_LO,
                gateHi: GATE_HI,
                tLo: model.bounds[0],
                tHi: model.bounds[1],
              },
            };
    // the paint's mask: the lit pixels only (the mask pass keeps where a fragment fell, not its color)
    if (cut != null) return [score, { module: EarthworkCut, props: { cut } }];
    // the score as plain gray, opaque: for the headless checks against numpy (window.__ewGray)
    if ((globalThis as { __ewGray?: boolean }).__ewGray) return [score];
    return [
      score,
      { module: Colormap, props: { colormapTexture: g.colormap, colormapIndex: COLORMAP_INDEX.viridis } },
      { module: EarthworkAlpha },
    ];
  }

  /** The layer's id for the window and the layer chosen: the paint's maskId while it is the mask. */
  const layerId = (first: number, last: number, kind: ChunkKind) => (paintOf(kind) ? `earthwork-${kind}` : `earthwork-${first}-${last}${tag(kind)}`);

  /**
   * The paint's mask for a layer: every one of the paint's years (`years`, each the layer's score from the year before)
   * read at once in the worker and drawn from one small texture a chunk, a year per channel. The year shown (`year`)
   * picks the channel, so a step between years reads nothing and loads nothing.
   */
  function paintLayer(opts: { visible: boolean; year: number; years: number[]; model: LayerKind; minZoom: number; onTile?: () => void; onViewportLoad?: () => void }) {
    const { visible, year, minZoom, onTile, onViewportLoad } = opts;
    const kind: LayerKind = LAYERS.includes(opts.model) ? opts.model : "aef", ck: ChunkKind = `paint-${kind}`;
    const years = opts.years.slice(0, 4), allIdx = years.map((y) => y - YEAR_ORIGIN), lastIdx = allIdx[allIdx.length - 1];
    const channel = Math.max(0, years.indexOf(year));
    yearsFor.paint = years;
    const cut = model.cut[kind];
    return new ZarrLayer<zarr.Readable, "int8", PaintTile>({
      id: layerId(year - 1, year, ck),
      operation: "mask",
      node: arr as AefArray,
      metadata: attrs,
      selection: { time: lastIdx, band: null },
      getTileData: async (_a, o) => {
        const ys = o.sliceSpec[2] as zarr.Slice, xs = o.sliceSpec[3] as zarr.Slice;
        const py0 = ys.start ?? 0, px0 = xs.start ?? 0;
        const { device, width, height } = o;
        const s = await scoreChunk(kind, px0, py0, width, height, allIdx, o.signal);
        const key = keyOf(0, 0, ck, Math.floor(py0 / CHUNK), Math.floor(px0 / CHUNK));
        keepPaint(key, { s, w: width, h: height, years, stride: 4 });
        tilesRead++;
        loaded.add(key);
        const scoreTex = device.createTexture({
          format: "rgba8unorm",
          width,
          height,
          mipLevels: 1,
          data: s,
          sampler: { minFilter: "nearest", magFilter: "nearest", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge" },
        });
        return { width, height, byteLength: s.byteLength, key, scoreTex };
      },
      renderTile: (d: PaintTile) => ({
        renderPipeline: [
          { module: EarthworkScoreTex, props: { scoreTex: d.scoreTex, channel } },
          { module: EarthworkCut, props: { cut } },
        ],
      }),
      visible,
      minZoom,
      maxRequests: 12,
      // 64 KB a tile: many kept, so panning back and forth draws from here
      maxCacheSize: 256,
      onTileLoad: () => onTile && onTile(),
      onTileUnload: (tile: { content?: PaintTile | null }) => {
        const t = tile.content;
        if (!t) return;
        loaded.delete(t.key);
        t.scoreTex?.destroy();
      },
      onViewportLoad: () => onViewportLoad && onViewportLoad(),
      // the year shown is only a channel: a step redraws, it reads nothing
      updateTriggers: { renderTile: [channel] },
      // @ts-expect-error beforeId is read by the MapLibre overlay, not typed on the layer
      beforeId: beforeId(),
    });
  }

  /**
   * The year map: each pixel in the color of the latest year, up to the year shown (`year`), that the layer had it
   * over its cutoff (each year from the year before it). Every year (`years`, up to eight) read at once in the
   * worker into one byte a pixel, a bit a year, so a step between years reads nothing and loads nothing.
   *
   * With `flare` (its strength, 0 to 1) it is the flare instead: the same pixels lifting the imagery under them by a
   * screen blend (EarthworkFlare), for a map of its own (the pair's left). It keeps no books (the chunks loaded, the
   * chunks kept for the card), which are the year map's, and its chunks' scores come from the worker's, read once
   */
  function yearLayer(opts: { visible: boolean; year: number; years: number[]; model: LayerKind; minZoom: number; flare?: number; onTile?: () => void; onViewportLoad?: () => void }) {
    const { visible, year, minZoom, onTile, onViewportLoad } = opts;
    const flare = opts.flare;
    const kind: LayerKind = LAYERS.includes(opts.model) ? opts.model : "aef", ck: ChunkKind = `years-${kind}`;
    const years = opts.years.slice(0, 8), allIdx = years.map((y) => y - YEAR_ORIGIN), lastIdx = allIdx[allIdx.length - 1];
    const shown = years.filter((y) => y <= year).length - 1;
    yearsFor.years = years;
    // over the cutoff as the worker encodes a score (1 + 254 x score), so the bits match EarthworkCut
    const over = 1 + 254 * model.cut[kind], n = years.length, stride = strideOf(n);
    const g = gpu;
    return new ZarrLayer<zarr.Readable, "int8", YearTile>({
      id: layerId(0, 0, ck) + (flare == null ? "" : "-flare"),
      node: arr as AefArray,
      metadata: attrs,
      selection: { time: lastIdx, band: null },
      getTileData: async (_a, o) => {
        const ys = o.sliceSpec[2] as zarr.Slice, xs = o.sliceSpec[3] as zarr.Slice;
        const py0 = ys.start ?? 0, px0 = xs.start ?? 0;
        const { device, width, height } = o;
        // the colormap's texture is made on the first device seen (the main map's: never the flare's)
        if (flare == null) onDevice(device);
        const s = await scoreChunk(kind, px0, py0, width, height, allIdx, o.signal);
        const key = keyOf(0, 0, ck, Math.floor(py0 / CHUNK), Math.floor(px0 / CHUNK));
        if (flare == null) {
          keepPaint(key, { s, w: width, h: height, years, stride });
          tilesRead++;
          loaded.add(key);
        }
        const plane = width * height, bits = new Uint8Array(plane);
        for (let i = 0; i < plane; i++) {
          let b = 0;
          for (let c = 0; c < n; c++) if (s[stride * i + c] >= over) b |= 1 << c;
          bits[i] = b;
        }
        const yearTex = device.createTexture({
          format: "r8unorm",
          width,
          height,
          mipLevels: 1,
          data: bits,
          sampler: { minFilter: "nearest", magFilter: "nearest", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge" },
        });
        return { width, height, byteLength: plane, key, yearTex };
      },
      renderTile: (d: YearTile) => ({
        renderPipeline: flare != null ? [{ module: EarthworkFlare, props: { yearTex: d.yearTex, shown, strength: flare } }] : [
          { module: EarthworkYearTex, props: { yearTex: d.yearTex, shown, n } },
          ...(g ? [{ module: Colormap, props: { colormapTexture: g.colormap, colormapIndex: COLORMAP_INDEX.cividis } }] : []),
        ],
      }),
      visible,
      minZoom,
      // a chunk reads every year (nine AlphaEarth reads): fewer at once keeps the worker's kept chunks enough for
      // the pairs that share a year
      maxRequests: 6,
      maxCacheSize: 256,
      onTileLoad: () => flare == null && onTile && onTile(),
      onTileUnload: (tile: { content?: YearTile | null }) => {
        const t = tile.content;
        if (!t) return;
        if (flare == null) loaded.delete(t.key);
        t.yearTex?.destroy();
      },
      onViewportLoad: () => flare == null && onViewportLoad && onViewportLoad(),
      // the flare's screen blend: out = src (1 - dst) + dst, the imagery lifted toward white by src, its alpha kept
      ...(flare != null ? { parameters: { blend: true, blendColorOperation: "add", blendColorSrcFactor: "one-minus-dst", blendColorDstFactor: "one", blendAlphaOperation: "add", blendAlphaSrcFactor: "zero", blendAlphaDstFactor: "one" } } : {}),
      // the year shown is only which bits count: a step redraws, it reads nothing
      updateTriggers: { renderTile: [shown, !!g, flare ?? -1] },
      // @ts-expect-error beforeId is read by the MapLibre overlay, not typed on the layer
      beforeId: beforeId(),
    });
  }

  /**
   * The layer for the window and the layer chosen. As the paint's mask (mask: true) it is not
   * drawn, only its lit pixels (over the layer's cutoff) marked; the same layer either way, so
   * the chunks read for one serve the other.
   */
  function layer(opts: { visible: boolean; first: number; last: number; model: LayerKind; minZoom: number; mask?: boolean; onTile?: () => void; onViewportLoad?: () => void }) {
    const { visible, first, last, minZoom, onTile, onViewportLoad } = opts;
    const kind: LayerKind = LAYERS.includes(opts.model) ? opts.model : "aggregate";
    const cut = opts.mask ? model.cut[kind] : null;
    const firstIdx = first - YEAR_ORIGIN;
    const lastIdx = last - YEAR_ORIGIN;
    return new ZarrLayer<zarr.Readable, "int8", TileData>({
      id: layerId(first, last, kind),
      operation: opts.mask ? "mask" : "draw",
      node: arr as AefArray,
      metadata: attrs,
      selection: { time: firstIdx, band: null },
      getTileData: async (_a, o) => {
        const ys = o.sliceSpec[2] as zarr.Slice, xs = o.sliceSpec[3] as zarr.Slice;
        const py0 = ys.start ?? 0, px0 = xs.start ?? 0;
        const { device, width, height } = o;
        onDevice(device);
        const aefOnly = kind === "aef";
        const r = await readChunk(px0, py0, width, height, firstIdx, lastIdx, aefOnly, o.signal);
        const d = {
          firstTex: bandTexture(device, r.aefFirst, width, height),
          lastTex: bandTexture(device, r.aefLast, width, height),
          first: r.aefFirst,
          last: r.aefLast,
          width,
          height,
          byteLength: r.aefFirst.byteLength + r.aefLast.byteLength,
        };
        const t = { first: r.tesFirst, last: r.tesLast, flags: r.tesFlags };
        const key = keyOf(first, last, kind, Math.floor(py0 / CHUNK), Math.floor(px0 / CHUNK));
        keepChunk(key, {
          aefFirst: d.first,
          aefLast: d.last,
          tesFirst: t.first,
          tesLast: t.last,
          tesFlags: t.flags,
          w: width,
          h: height,
        });
        tilesRead++;
        loaded.add(key);
        // the raw bytes stay only in the chunks kept for scoreAt (KEEP), not in every tile
        return {
          ...d,
          first: NONE,
          last: NONE,
          key,
          tesFirstTex: aefOnly ? undefined : bandTexture(device, t.first, width, height, TES_BANDS),
          tesLastTex: aefOnly ? undefined : bandTexture(device, t.last, width, height, TES_BANDS),
          tesFlagsTex: aefOnly ? undefined : flagsTexture(device, t.flags, width, height),
          tesFirst: NONE,
          tesLast: NONE,
          tesFlags: NONE_U8,
          byteLength: (d.byteLength ?? 0) + t.first.byteLength + t.last.byteLength + t.flags.byteLength,
        };
      },
      renderTile: (data: TileData) => ({ renderPipeline: renderPipeline(kind, data, cut) }),
      visible,
      minZoom,
      // source.coop answers over HTTP/2, so more than the browser's 6 per host
      maxRequests: 12,
      // each tile is AlphaEarth's two years x 64 bands and TESSERA's two x 128 (24 MB on the GPU)
      maxCacheSize: 8,
      onTileLoad: () => onTile && onTile(),
      // a tile let go frees its textures (about 24 MB on the GPU; nothing else does, and a jump to a new
      // place piled up enough of them to crash the page)
      onTileUnload: (tile: { content?: TileData | null }) => {
        const t = tile.content;
        if (!t) return;
        loaded.delete(t.key);
        for (const tex of [t.firstTex, t.lastTex, t.tesFirstTex, t.tesLastTex, t.tesFlagsTex]) tex?.destroy();
      },
      onViewportLoad: () => onViewportLoad && onViewportLoad(),
      updateTriggers: { renderTile: [kind, !!gpu, cut] },
      // @ts-expect-error beforeId is read by the MapLibre overlay, not typed on the layer
      beforeId: beforeId(),
    });
  }

  const chunkAt = (lon: number, lat: number) => {
    const px = Math.floor((lon - X0) / RES), py = Math.floor((Y0 - lat) / RES);
    return { px, py, r: Math.floor(py / CHUNK), c: Math.floor(px / CHUNK) };
  };

  /**
   * A layer's score at a point, as the shader computes it, from the chunk
   * already read; null where that chunk is not in hand or the layer has none there.
   */
  function scoreAt(lon: number, lat: number, first: number, last: number, kind: ChunkKind): number | null {
    const { px, py, r, c } = chunkAt(lon, lat);
    if (paintOf(kind)) {
      const p = paintScores.get(keyOf(first, last, kind, r, c));
      const ix = px - c * CHUNK, iy = py - r * CHUNK, ch = p ? p.years.indexOf(last) : -1;
      if (!p || ch < 0 || ix < 0 || iy < 0 || ix >= p.w || iy >= p.h) return null;
      const v = p.s[p.stride * (iy * p.w + ix) + ch];
      return v ? (v - 1) / 254 : null;
    }
    const k = keyOf(first, last, kind, r, c), ch = chunks.get(k);
    if (!ch) return null;
    keepChunk(k, ch);
    const ix = px - c * CHUNK, iy = py - r * CHUNK;
    if (ix < 0 || iy < 0 || ix >= ch.w || iy >= ch.h) return null;
    return scorePixel(LAYERS.includes(kind as LayerKind) ? (kind as LayerKind) : "aggregate", model, { ...ch, plane: ch.w * ch.h, at: iy * ch.w + ix });
  }

  /**
   * The year map at a point: the years, up to `upTo`, the layer had it over its cutoff, from the chunk in hand;
   * null where that chunk is not in hand or AlphaEarth has none there.
   */
  function detectedAt(lon: number, lat: number, upTo: number, model_: LayerKind): number[] | null {
    const kind: LayerKind = LAYERS.includes(model_) ? model_ : "aef";
    const { px, py, r, c } = chunkAt(lon, lat);
    const p = paintScores.get(keyOf(0, 0, `years-${kind}`, r, c));
    const ix = px - c * CHUNK, iy = py - r * CHUNK;
    if (!p || ix < 0 || iy < 0 || ix >= p.w || iy >= p.h) return null;
    const over = 1 + 254 * model.cut[kind], at = p.stride * (iy * p.w + ix), out: number[] = [];
    let any = false;
    p.years.forEach((y, ch) => {
      const v = p.s[at + ch];
      if (v) any = true;
      if (y <= upTo && v >= over) out.push(y);
    });
    return any ? out : null;
  }

  /**
   * Read the chunk under a point for this window if it is not in hand (a view holds more chunks
   * than are kept); resolves once scoreAt can answer there, or with nothing where it cannot.
   */
  function readAt(lon: number, lat: number, first: number, last: number, kind: ChunkKind): Promise<void> {
    const { px, py, r, c } = chunkAt(lon, lat);
    const k = keyOf(first, last, kind, r, c);
    const pk = paintOf(kind);
    if (pk) {
      if (paintScores.has(k)) return Promise.resolve();
      const [, , H, W] = arr.shape;
      if (px < 0 || py < 0 || px >= W || py >= H) return Promise.resolve();
      const y0 = r * CHUNK, x0 = c * CHUNK, h = Math.min(CHUNK, H - y0), w = Math.min(CHUNK, W - x0);
      const ys = yearsOf(kind), years = ys.length ? ys : [last];
      return scoreChunk(pk, x0, y0, w, h, years.map((y) => y - YEAR_ORIGIN))
        .then((s) => keepPaint(k, { s, w, h, years, stride: strideOf(years.length) }))
        .catch(() => {});
    }
    if (chunks.has(k)) return Promise.resolve();
    let p = reading.get(k);
    if (!p) {
      const [, , H, W] = arr.shape;
      if (px < 0 || py < 0 || px >= W || py >= H) return Promise.resolve();
      const y0 = r * CHUNK, x0 = c * CHUNK, h = Math.min(CHUNK, H - y0), w = Math.min(CHUNK, W - x0);
      p = readChunk(x0, y0, w, h, first - YEAR_ORIGIN, last - YEAR_ORIGIN, kind === "aef")
        .then((r) => keepChunk(k, { ...r, w, h }))
        .catch(() => {})
        .finally(() => reading.delete(k));
      reading.set(k, p);
    }
    return p;
  }

  /** How many of the chunks under a box are loaded on the GPU for this window. */
  function loadedIn(bbox: [number, number, number, number], first: number, last: number, kind: ChunkKind) {
    const [W, S, E, N] = bbox;
    const c0 = Math.floor((W - X0) / RES / CHUNK), c1 = Math.floor((E - X0) / RES / CHUNK);
    const r0 = Math.floor((Y0 - N) / RES / CHUNK), r1 = Math.floor((Y0 - S) / RES / CHUNK);
    let n = 0;
    for (let r = r0; r <= r1; r++) for (let c = c0; c <= c1; c++) if (loaded.has(keyOf(first, last, kind, r, c))) n++;
    return { loaded: n, wanted: (r1 - r0 + 1) * (c1 - c0 + 1) };
  }

  /** How many of the chunks under a box are in hand for this window. */
  function chunksIn(bbox: [number, number, number, number], first: number, last: number, kind: ChunkKind) {
    const [W, S, E, N] = bbox;
    const c0 = Math.floor((W - X0) / RES / CHUNK), c1 = Math.floor((E - X0) / RES / CHUNK);
    const r0 = Math.floor((Y0 - N) / RES / CHUNK), r1 = Math.floor((Y0 - S) / RES / CHUNK);
    let inHand = 0;
    const held = paintOf(kind) ? paintScores : chunks;
    for (let r = r0; r <= r1; r++) for (let c = c0; c <= c1; c++) if (held.has(keyOf(first, last, kind, r, c))) inHand++;
    return { inHand, wanted: (r1 - r0 + 1) * (c1 - c0 + 1) };
  }

  return {
    layer,
    paintLayer,
    yearLayer,
    layerId,
    scoreAt,
    detectedAt,
    readAt,
    chunksIn,
    loadedIn,
    model,
    tilesRead: () => tilesRead,
    whenReady: (f: () => void) => {
      onReady = f;
    },
  };
}

export { COLORMAP_INDEX };
