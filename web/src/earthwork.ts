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
import { EarthworkAggregate, EarthworkAlpha, EarthworkCheckAlpha, EarthworkCut, EarthworkYearCheck, EarthworkFlare, EarthworkScore, EarthworkScoreTex, EarthworkTessera, EarthworkYearTex } from "./earthwork-score";
import { TES_BANDS } from "./tessera";

export type { LayerKind, WorldModel };
/** The layers' chunks, the paint's and the year map's: a layer's score scored ahead, a year at a time. */
export type ChunkKind = LayerKind | `paint-${LayerKind}` | `years-${LayerKind}`;
const paintOf = (kind: ChunkKind): LayerKind | null => (kind.startsWith("paint-") || kind.startsWith("years-") ? (kind.slice(6) as LayerKind) : null);

/** One chunk of the paint: every year's score on the GPU (rgba8unorm, a year per channel, 1 + 254 x score, 0 where a year is missing). */
type PaintTile = { width: number; height: number; byteLength: number; key: string; scoreTex: Texture };
/** One chunk of the year map: the years each pixel was over the layer's cutoff, a bit a year (r8unorm). */
type YearTile = { width: number; height: number; byteLength: number; key: string; yearTex: Texture; checkTex?: Texture };

/** One chunk on the GPU: AlphaEarth's two years, TESSERA's two years on its pixels, and the flags (none of TESSERA's for AlphaEarth's layer alone). */
type TileData = PairTileData & {
  /** The chunk's key (keyOf), for the chunks loaded in view. */
  key: string;
  /** The chunk's place and the GPU it is on, for its cross-check (asked for only while it is shown). */
  px0: number;
  py0: number;
  device: Device;
  checkTex?: Texture;
  checkAsk?: AbortController;
  gone?: boolean;
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
    return scoreCheckChunk(kind, px0, py0, w, h, allIdx, false, signal).then((r) => r.score);
  }
  /** The same, and with `check` each year's cross-check (crossCheckChunk) for the pixels over the cutoff, a plane a year. */
  function scoreCheckChunk(kind: LayerKind, px0: number, py0: number, w: number, h: number, allIdx: number[], check: boolean, signal?: AbortSignal): Promise<{ score: Uint8Array; check?: Uint8Array }> {
    if (signal?.aborted) return Promise.reject(Object.assign(new Error("aborted"), { name: "AbortError" }));
    const id = ++askId;
    return new Promise<{ score: Uint8Array; check?: Uint8Array }>((resolve, reject) => {
      asked.set(id, { resolve, reject });
      worker.postMessage({ type: "score", id, kind, px0, py0, w, h, allIdx, check });
      signal?.addEventListener("abort", () => {
        if (!asked.has(id)) return;
        asked.delete(id);
        worker.postMessage({ type: "abort", id });
        reject(Object.assign(new Error("aborted"), { name: "AbortError" }));
      });
    });
  }

  /** A window's chunk's cross-check (crossCheckChunk), 0 to 255 a pixel. */
  function checkChunk(px0: number, py0: number, w: number, h: number, firstIdx: number, lastIdx: number, aefOnly: boolean, signal: AbortSignal): Promise<Uint8Array> {
    if (signal.aborted) return Promise.reject(Object.assign(new Error("aborted"), { name: "AbortError" }));
    const id = ++askId;
    return new Promise<{ score: Uint8Array }>((resolve, reject) => {
      asked.set(id, { resolve, reject });
      worker.postMessage({ type: "check", id, px0, py0, w, h, firstIdx, lastIdx, aefOnly });
      signal.addEventListener("abort", () => {
        if (!asked.has(id)) return;
        asked.delete(id);
        worker.postMessage({ type: "abort", id });
        reject(Object.assign(new Error("aborted"), { name: "AbortError" }));
      });
    }).then((r) => r.score);
  }
  // a cross-check come in: the layer drawn again (checkN in its updateTriggers), told by whenCheck
  let checkN = 0,
    onCheck: () => void = () => {};

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
  type PaintChunk = { s: Uint8Array; w: number; h: number; years: number[]; stride: number; check?: Uint8Array };
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

  function renderPipeline(kind: LayerKind, data: TileData, cut: number | null, check = false) {
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
      check && data.checkTex ? { module: EarthworkCheckAlpha, props: { checkTex: data.checkTex } } : { module: EarthworkAlpha },
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
   * With `flare` (its strength, 0 to 1) it is the flare instead: the same pixels scaling up the brightness of the
   * imagery under them by 1 + flare (EarthworkFlare and the layer's blend), for a map of its own (the pair's left). It keeps no books (the chunks loaded, the
   * chunks kept for the card), which are the year map's, and its chunks' scores come from the worker's, read once
   */
  function yearLayer(opts: { visible: boolean; year: number; years: number[]; model: LayerKind; minZoom: number; flare?: number; flareDim?: boolean; check?: boolean; mask?: boolean; ramp?: "plasma" | "cividis" | "mono"; onTile?: () => void; onViewportLoad?: () => void }) {
    const { visible, year, minZoom, onTile, onViewportLoad } = opts;
    const flare = opts.flare, dim = opts.flareDim ? 1 : 0;
    // as a mask (the imagery in its own colors inside the year map's pixels, yearMaskId): a layer of its own, a mask
    // from its first tile (a loaded layer switched to a mask keeps drawing its tiles as they were, and the imagery
    // then finds no mask), its scores the worker's, as the flare's; it keeps no books
    const mask = !!opts.mask && flare == null;
    const books = flare == null && !mask;
    // the cross-check (the year map's own, not the flare's): each year's, made with the scores from the same years (so
    // it reads nothing more) and kept with the tile, so A and a step between years only redraw
    const check = !!opts.check && flare == null && !mask;
    const ramp = opts.ramp ?? "plasma", mono = ramp === "mono";
    const kind: LayerKind = LAYERS.includes(opts.model) ? opts.model : "aef", ck: ChunkKind = `years-${kind}`;
    const years = opts.years.slice(0, 8), allIdx = years.map((y) => y - YEAR_ORIGIN), lastIdx = allIdx[allIdx.length - 1];
    const shown = years.filter((y) => y <= year).length - 1;
    yearsFor.years = years;
    // over the cutoff as the worker encodes a score (1 + 254 x score), so the bits match EarthworkCut
    const over = 1 + 254 * model.cut[kind], n = years.length, stride = strideOf(n);
    const g = gpu;
    return new ZarrLayer<zarr.Readable, "int8", YearTile>({
      id: layerId(0, 0, ck) + (flare != null ? "-flare" : mask ? "-mask" : ""),
      operation: mask ? "mask" : "draw",
      node: arr as AefArray,
      metadata: attrs,
      selection: { time: lastIdx, band: null },
      getTileData: async (_a, o) => {
        const ys = o.sliceSpec[2] as zarr.Slice, xs = o.sliceSpec[3] as zarr.Slice;
        const py0 = ys.start ?? 0, px0 = xs.start ?? 0;
        const { device, width, height } = o;
        // the colormap's texture is made on the first device seen (the main map's: never the flare's)
        if (books) onDevice(device);
        const { score: s, check: cb } = await scoreCheckChunk(kind, px0, py0, width, height, allIdx, books, o.signal);
        const key = keyOf(0, 0, ck, Math.floor(py0 / CHUNK), Math.floor(px0 / CHUNK));
        if (books) {
          keepPaint(key, { s, w: width, h: height, years, stride, check: cb });
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
        const checkTex = cb
          ? device.createTexture({
              format: "r8unorm",
              width,
              height: height * allIdx.length,
              mipLevels: 1,
              data: cb,
              sampler: { minFilter: "nearest", magFilter: "nearest", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge" },
            })
          : undefined;
        return { width, height, byteLength: plane * (1 + (cb ? allIdx.length : 0)), key, yearTex, checkTex };
      },
      renderTile: (d: YearTile) => {
        return {
          renderPipeline: flare != null ? [{ module: EarthworkFlare, props: { yearTex: d.yearTex, shown, strength: flare, dim } }] : [
            // one color (mono): Oranges, pale orange the first year to deep orange the last (short of its brown end, which
            // the dark basemap would swallow)
            { module: EarthworkYearTex, props: { yearTex: d.yearTex, shown, n, hi: mono ? 0.8 : 1 } },
            ...(g && !mask ? [{ module: Colormap, props: { colormapTexture: g.colormap, colormapIndex: mono ? COLORMAP_INDEX.oranges : COLORMAP_INDEX[ramp] } }] : []),
            ...(check && d.checkTex ? [{ module: EarthworkYearCheck, props: { checkTex: d.checkTex, h: d.height } }] : []),
          ],
        };
      },
      visible,
      minZoom,
      // a chunk reads every year (nine AlphaEarth reads): fewer at once keeps the worker's kept chunks enough for
      // the pairs that share a year
      maxRequests: 6,
      maxCacheSize: 256,
      onTileLoad: () => books && onTile && onTile(),
      onTileUnload: (tile: { content?: YearTile | null }) => {
        const t = tile.content;
        if (!t) return;
        if (books) loaded.delete(t.key);
        t.yearTex?.destroy();
        t.checkTex?.destroy();
      },
      onViewportLoad: () => books && onViewportLoad && onViewportLoad(),
      // the flare's blend works on what is under it, so a tile drawn over its parent (refining) would flare twice
      // there: no overlap
      ...(flare != null ? { refinementStrategy: "no-overlap" as const } : {}),
      // the flare's blend: out = src dst + dst = dst (1 + src), the imagery's brightness scaled up by 1 + src (its
      // texture, contrast and colors kept); dimming, out = dst src, scaled down. Its alpha kept
      ...(flare != null ? { parameters: { blend: true, blendColorOperation: "add", blendColorSrcFactor: dim ? "zero" : "dst", blendColorDstFactor: dim ? "src" : "one", blendAlphaOperation: "add", blendAlphaSrcFactor: "zero", blendAlphaDstFactor: "one" } } : {}),
      // the year shown is only which bits count: a step redraws, it reads nothing
      updateTriggers: { renderTile: [shown, !!g, flare ?? -1, dim, check, mask, ramp] },
      // @ts-expect-error beforeId is read by the MapLibre overlay, not typed on the layer
      beforeId: beforeId(),
    });
  }

  /** The year map's id for a layer, the imagery's maskId while the year map is its mask. */
  const yearMaskId = (model: LayerKind) => layerId(0, 0, `years-${LAYERS.includes(model) ? model : "aef"}`) + "-mask";

  /**
   * The layer for the window and the layer chosen. As the paint's mask (mask: true) it is not
   * drawn, only its lit pixels (over the layer's cutoff) marked; the same layer either way, so
   * the chunks read for one serve the other.
   */
  function layer(opts: { visible: boolean; first: number; last: number; model: LayerKind; minZoom: number; mask?: boolean; check?: boolean; onTile?: () => void; onViewportLoad?: () => void }) {
    const { visible, first, last, minZoom, onTile, onViewportLoad } = opts;
    const kind: LayerKind = LAYERS.includes(opts.model) ? opts.model : "aggregate";
    const cut = opts.mask ? model.cut[kind] : null;
    // the cross-check: AlphaEarth's own layer and the aggregate (TESSERA's alone has no AlphaEarth to set it against)
    const check = !!opts.check && !opts.mask && kind !== "tessera";
    // a shown chunk's cross-check, asked for once; it comes in as its own texture and the layer is drawn again
    const askCheck = (d: TileData) => {
      if (d.checkTex || d.checkAsk || d.gone) return;
      const ac = new AbortController();
      d.checkAsk = ac;
      checkChunk(d.px0, d.py0, d.width, d.height, firstIdx, lastIdx, kind === "aef", ac.signal)
        .then((bytes) => {
          if (d.gone) return;
          d.checkTex = d.device.createTexture({
            format: "r8unorm",
            width: d.width,
            height: d.height,
            mipLevels: 1,
            data: bytes,
            sampler: { minFilter: "nearest", magFilter: "nearest", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge" },
          });
          checkN++;
          onCheck();
        })
        .catch(() => {
          d.checkAsk = undefined;
        });
    };
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
          px0,
          py0,
          device,
          tesFirstTex: aefOnly ? undefined : bandTexture(device, t.first, width, height, TES_BANDS),
          tesLastTex: aefOnly ? undefined : bandTexture(device, t.last, width, height, TES_BANDS),
          tesFlagsTex: aefOnly ? undefined : flagsTexture(device, t.flags, width, height),
          tesFirst: NONE,
          tesLast: NONE,
          tesFlags: NONE_U8,
          byteLength: (d.byteLength ?? 0) + t.first.byteLength + t.last.byteLength + t.flags.byteLength,
        };
      },
      renderTile: (data: TileData) => {
        if (check) askCheck(data);
        return { renderPipeline: renderPipeline(kind, data, cut, check) };
      },
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
        t.gone = true;
        t.checkAsk?.abort();
        for (const tex of [t.firstTex, t.lastTex, t.tesFirstTex, t.tesLastTex, t.tesFlagsTex, t.checkTex]) tex?.destroy();
      },
      onViewportLoad: () => onViewportLoad && onViewportLoad(),
      updateTriggers: { renderTile: [kind, !!gpu, cut, check, check ? checkN : 0] },
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
  /** The year map's cross-check at a point for its latest year up to `upTo` (the one it is drawn in), 0 to 1; null where
   * nothing is drawn there or its chunk is not in hand. */
  function checkAt(lon: number, lat: number, upTo: number, model_: LayerKind): number | null {
    const kind: LayerKind = LAYERS.includes(model_) ? model_ : "aef";
    const { px, py, r, c } = chunkAt(lon, lat);
    const p = paintScores.get(keyOf(0, 0, `years-${kind}`, r, c));
    const ix = px - c * CHUNK, iy = py - r * CHUNK;
    if (!p || !p.check || ix < 0 || iy < 0 || ix >= p.w || iy >= p.h) return null;
    const over = 1 + 254 * model.cut[kind], i = iy * p.w + ix, plane = p.w * p.h;
    let latest = -1;
    p.years.forEach((y, ch) => {
      if (y <= upTo && p.s[p.stride * i + ch] >= over) latest = ch;
    });
    return latest < 0 ? null : p.check[latest * plane + i] / 255;
  }

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

  type Run = [number, number];
  /** A row's columns (unsorted) as runs [start, end), row by row. */
  function runsFrom(xsOf: Map<number, number[]>) {
    const runsOf = new Map<number, Run[]>();
    for (const [y, xs] of xsOf) {
      xs.sort((u, v) => u - v);
      const rs: Run[] = [];
      for (const x of xs) {
        const r = rs[rs.length - 1];
        if (r && r[1] === x) r[1] = x + 1;
        else rs.push([x, x + 1]);
      }
      runsOf.set(y, rs);
    }
    return runsOf;
  }

  /**
   * The outer outline of pixels given row by row as runs [start, end) of the whole grid's columns, holes filled: edges
   * [lon, lat, lon, lat] (runs along a row or a column merged), and the rows' runs with their holes filled.
   */
  function traceOutline(runsOf: Map<number, Run[]>) {
    const gapsOf = new Map<number, { s: number; e: number; id: number }[]>();
    const parent = [0]; // gap ids from 1; 0 is the outside
    for (const [y, rs] of runsOf) {
      const gs = [];
      for (let i = 1; i < rs.length; i++) gs.push({ s: rs[i - 1][1], e: rs[i][0], id: parent.push(parent.length) - 1 });
      gapsOf.set(y, gs);
    }
    // a gap is a hole unless it reaches the outside through the gaps above and below it (by side, the segment being
    // joined by corner too); holes are filled, so only the outer edge is drawn
    const find = (i: number): number => (parent[i] === i ? i : (parent[i] = find(parent[i])));
    const join = (i: number, j: number) => void (parent[find(i)] = find(j));
    for (const [y, gs] of gapsOf)
      for (const ny of [y - 1, y + 1]) {
        const rn = runsOf.get(ny), gn = gapsOf.get(ny) ?? [];
        let j = 0;
        for (const g of gs) {
          if (!rn || g.s < rn[0][0] || g.e > rn[rn.length - 1][1]) join(g.id, 0);
          while (j < gn.length && gn[j].e <= g.s) j++;
          for (let k = j; k < gn.length && gn[k].s < g.e; k++) join(g.id, gn[k].id);
        }
      }
    const filled = new Map<number, Run[]>();
    for (const [y, rs] of runsOf) {
      const gs = gapsOf.get(y)!, f: Run[] = [[rs[0][0], rs[0][1]]];
      for (let i = 1; i < rs.length; i++) {
        if (find(gs[i - 1].id) !== find(0)) f[f.length - 1][1] = rs[i][1];
        else f.push([rs[i][0], rs[i][1]]);
      }
      filled.set(y, f);
    }
    // the parts of runs a not covered by runs b
    const minus = (a: Run[], b: Run[] | undefined, f: (s: number, e: number) => void) => {
      let j = 0;
      for (const [s0, e] of a) {
        let s = s0;
        while (b && j < b.length && b[j][1] <= s) j++;
        for (let k = j; b && k < b.length && b[k][0] < e; k++) {
          if (b[k][0] > s) f(s, b[k][0]);
          s = Math.max(s, b[k][1]);
        }
        if (s < e) f(s, e);
      }
    };
    // the outline: a row's top (or bottom) where the row beyond does not cover it, and each run's two ends, a column
    // line's runs merged. A column line l is the left of column l; the side's sign keeps a left and a right on the
    // same line apart where they meet only at a corner
    const cols = new Map<number, number[]>();
    const push = (m: Map<number, number[]>, line: number, at: number) => {
      const a = m.get(line);
      if (a) a.push(at);
      else m.set(line, [at]);
    };
    const lonOf = (gx: number) => X0 + gx * RES, latOf = (gy: number) => Y0 - gy * RES;
    const edges: [number, number, number, number][] = [];
    for (const [y, f] of filled) {
      minus(f, filled.get(y - 1), (a, b) => edges.push([lonOf(a), latOf(y), lonOf(b), latOf(y)]));
      minus(f, filled.get(y + 1), (a, b) => edges.push([lonOf(a), latOf(y + 1), lonOf(b), latOf(y + 1)]));
      for (const [s, e] of f) {
        push(cols, s * 2, y);
        push(cols, e * 2 + 1, y);
      }
    }
    const runs = (m: Map<number, number[]>, f: (line: number, a: number, b: number) => void) => {
      for (const [l, a] of m) {
        a.sort((u, v) => u - v);
        let s = a[0], e = a[0];
        for (let i = 1; i <= a.length; i++) {
          if (i < a.length && a[i] === e + 1) e = a[i];
          else {
            f(l >> 1, s, e + 1);
            if (i < a.length) s = e = a[i];
          }
        }
      }
    };
    runs(cols, (l, a, b) => edges.push([lonOf(l), latOf(a), lonOf(l), latOf(b)]));
    return { edges, filled };
  }

  /**
   * The outlines of everything the year map found (any year up to `upTo`) in the chunks over a box, joined across years
   * and holes filled, as edges [lon, lat, lon, lat]; chunks not in hand count as nothing found. Null over more than
   * `maxPx` pixels.
   */
  function outlinesIn(bbox: [number, number, number, number], upTo: number, model_: LayerKind, maxPx = 12_000_000) {
    const kind: LayerKind = LAYERS.includes(model_) ? model_ : "aef";
    const ck: ChunkKind = `years-${kind}`, over = 1 + 254 * model.cut[kind];
    const [W, S, E, N] = bbox;
    const c0 = Math.floor((W - X0) / RES / CHUNK), c1 = Math.floor((E - X0) / RES / CHUNK);
    const r0 = Math.floor((Y0 - N) / RES / CHUNK), r1 = Math.floor((Y0 - S) / RES / CHUNK);
    if ((r1 - r0 + 1) * (c1 - c0 + 1) * CHUNK * CHUNK > maxPx) return null;
    const runsOf = new Map<number, Run[]>();
    for (let r = r0; r <= r1; r++)
      for (let c = c0; c <= c1; c++) {
        const p = paintScores.get(keyOf(0, 0, ck, r, c));
        if (!p) continue;
        const chs = p.years.flatMap((y, ch) => (y <= upTo ? [ch] : []));
        if (!chs.length) continue;
        for (let iy = 0; iy < p.h; iy++) {
          const gy = r * CHUNK + iy;
          let rs = runsOf.get(gy);
          for (let ix = 0; ix < p.w; ix++) {
            const at = p.stride * (iy * p.w + ix);
            let hit = false;
            for (const ch of chs) if (p.s[at + ch] >= over) { hit = true; break; }
            if (!hit) continue;
            const gx = c * CHUNK + ix;
            if (!rs) runsOf.set(gy, (rs = []));
            const last = rs[rs.length - 1];
            if (last && last[1] === gx) last[1] = gx + 1;
            else rs.push([gx, gx + 1]);
          }
        }
      }
    // a row's runs come chunk by chunk from the west, in order, a run going on across a chunk's edge
    return traceOutline(runsOf).edges;
  }

  /**
   * Found ground this many pixels apart or less still joins into one patch (a building and its graded lot with a road or
   * a curb between them): 1, ten meters, one pixel not found between them.
   */
  const PATCH_GAP = 1;

  /**
   * The patches the year map found over a box: every pixel found in any year up to `upTo` joined to its neighbors (by
   * side or corner, or across a gap of PATCH_GAP pixels) whatever its year, one patch, as the joined segment is. The `limit` largest, largest first, and how
   * many there were. Each: its pixels `n`, hectares, center, box [W, S, E, N], a pixel in it (`at`), `years` (pixels
   * first found each year, by `yearList`), `first` its earliest year, `main` the year most of it was first found, `last` the latest year any of it was.
   * Chunks not in hand count as nothing found.
   */
  function patchesIn(bbox: [number, number, number, number], upTo: number, model_: LayerKind, limit = 40) {
    const kind: LayerKind = LAYERS.includes(model_) ? model_ : "aef";
    const ck: ChunkKind = `years-${kind}`, over = 1 + 254 * model.cut[kind];
    const [W, S, E, N] = bbox;
    const c0 = Math.floor((W - X0) / RES / CHUNK), c1 = Math.floor((E - X0) / RES / CHUNK);
    const r0 = Math.floor((Y0 - N) / RES / CHUNK), r1 = Math.floor((Y0 - S) / RES / CHUNK);
    // the runs found, row by row from the north, each row from the west: row, start, end, its pixels' first years in fy
    const ry: number[] = [], rs: number[] = [], re: number[] = [], rf: number[] = [];
    let fy = new Uint8Array(1 << 20), nf = 0;
    const parent: number[] = [];
    const find = (i: number): number => {
      while (parent[i] !== i) i = parent[i] = parent[parent[i]];
      return i;
    };
    const yearSet = new Set<number>();
    // each row's runs [start, end) in the run arrays, for the rows above PATCH_GAP + 1 back
    const rowRuns = new Map<number, [number, number]>();
    const join = (i: number, j: number) => {
      const x = find(i), y = find(j);
      if (x !== y) parent[x] = y;
    };
    for (let r = r0; r <= r1; r++) {
      const ps: (PaintChunk | null)[] = [];
      const chs: number[][] = [];
      for (let c = c0; c <= c1; c++) {
        const p = paintScores.get(keyOf(0, 0, ck, r, c)) ?? null;
        ps.push(p);
        chs.push(p ? p.years.flatMap((y, ch) => (y <= upTo ? [ch] : [])) : []);
        if (p) for (const y of p.years) if (y <= upTo) yearSet.add(y);
      }
      for (let iy = 0; iy < CHUNK; iy++) {
        const gy = r * CHUNK + iy, cur0 = ry.length;
        let open = -1; // the run going on, -1 none
        for (let j = 0; j < ps.length; j++) {
          const p = ps[j], ch = chs[j];
          if (!p || iy >= p.h || !ch.length) { open = -1; continue; }
          const gx0 = (c0 + j) * CHUNK;
          for (let ix = 0; ix < p.w; ix++) {
            const at = p.stride * (iy * p.w + ix);
            let first = -1;
            for (const k of ch) if (p.s[at + k] >= over) { first = p.years[k]; break; }
            if (first < 0) { open = -1; continue; }
            if (nf === fy.length) { const g = new Uint8Array(fy.length * 2); g.set(fy); fy = g; }
            fy[nf++] = first - 2000;
            const gx = gx0 + ix;
            if (open >= 0 && re[open] === gx) re[open] = gx + 1;
            else {
              open = ry.length;
              ry.push(gy); rs.push(gx); re.push(gx + 1); rf.push(nf - 1); parent.push(open);
              // the run before it in this row, across a gap of PATCH_GAP pixels or less
              if (open > cur0 && gx - re[open - 1] <= PATCH_GAP) join(open, open - 1);
            }
          }
          if (p.w < CHUNK) open = -1;
        }
        // join this row's runs to the runs of the rows above within PATCH_GAP + 1 pixels (by side or corner, or across
        // the gap)
        for (let d = 1; d <= PATCH_GAP + 1; d++) {
          const up = rowRuns.get(gy - d);
          if (!up) continue;
          const [p0, p1] = up;
          let a = p0;
          for (let b = cur0; b < ry.length; b++) {
            while (a < p1 && re[a] < rs[b] - PATCH_GAP) a++;
            for (let k = a; k < p1 && rs[k] <= re[b] + PATCH_GAP; k++) join(b, k);
          }
        }
        if (ry.length > cur0) rowRuns.set(gy, [cur0, ry.length]);
        rowRuns.delete(gy - PATCH_GAP - 2);
      }
    }
    const yearList = [...yearSet].sort((u, v) => u - v), yi = new Map(yearList.map((y, i) => [y - 2000, i]));
    type P = { n: number; sx: number; sy: number; x0: number; x1: number; y0: number; y1: number; ax: number; ay: number; years: number[] };
    const by = new Map<number, P>();
    for (let i = 0; i < ry.length; i++) {
      const root = find(i);
      let q = by.get(root);
      if (!q) by.set(root, (q = { n: 0, sx: 0, sy: 0, x0: rs[i], x1: re[i], y0: ry[i], y1: ry[i], ax: rs[i], ay: ry[i], years: yearList.map(() => 0) }));
      const len = re[i] - rs[i];
      q.n += len;
      q.sx += len * (rs[i] + re[i] - 1) / 2;
      q.sy += len * ry[i];
      q.x0 = Math.min(q.x0, rs[i]);
      q.x1 = Math.max(q.x1, re[i]);
      q.y0 = Math.min(q.y0, ry[i]);
      q.y1 = Math.max(q.y1, ry[i] + 1);
      for (let k = rf[i]; k < rf[i] + len; k++) q.years[yi.get(fy[k])!]++;
    }
    const lonOf = (gx: number) => X0 + gx * RES, latOf = (gy: number) => Y0 - gy * RES;
    const all = [...by.values()].sort((u, v) => v.n - u.n);
    const patches = all.slice(0, limit).map((q) => {
      const lat = latOf(q.sy / q.n + 0.5), m = Math.max(...q.years);
      return {
        n: q.n,
        ha: (q.n * (RES * 111320) ** 2 * Math.cos((lat * Math.PI) / 180)) / 1e4,
        lon: lonOf(q.sx / q.n + 0.5),
        lat,
        box: [lonOf(q.x0), latOf(q.y1), lonOf(q.x1), latOf(q.y0)] as [number, number, number, number],
        at: [lonOf(q.ax + 0.5), latOf(q.ay + 0.5)] as [number, number],
        years: q.years,
        first: yearList[q.years.findIndex((v) => v > 0)],
        main: yearList[q.years.indexOf(m)],
        last: yearList[q.years.length - 1 - [...q.years].reverse().findIndex((v) => v > 0)],
      };
    });
    return { patches, total: all.length, yearList };
  }

  /**
   * The year map's segment under a point: the pixels joined to it (by side or corner) whose latest year over the cutoff,
   * up to `upTo`, is its own, across the chunks in hand. Its outer outline (holes filled) as edges [lon, lat, lon, lat]
   * (runs along a row or a column merged), its year, its pixels, its center and `has` (a point inside it); `cut` when
   * a chunk not in hand or the cap stopped it. Null off a segment. `joined`: every pixel found in any year up to `upTo`
   * joined to it, whatever its year, as one, across a gap of PATCH_GAP pixels as a patch is; `first` the earliest year any of its pixels was found.
   */
  function segmentAt(lon: number, lat: number, upTo: number, model_: LayerKind, joined = false, max = 1_500_000) {
    const kind: LayerKind = LAYERS.includes(model_) ? model_ : "aef";
    const ck: ChunkKind = `years-${kind}`, over = 1 + 254 * model.cut[kind];
    const got = new Map<string, PaintChunk | null>();
    // the latest year's index up to upTo at a pixel of the whole grid; -1 none, -2 its chunk not in hand
    const latestAt = (gx: number, gy: number) => {
      const r = Math.floor(gy / CHUNK), c = Math.floor(gx / CHUNK), k = r + ":" + c;
      let p = got.get(k);
      if (p === undefined) got.set(k, (p = paintScores.get(keyOf(0, 0, ck, r, c)) ?? null));
      if (!p) return -2;
      const ix = gx - c * CHUNK, iy = gy - r * CHUNK;
      if (ix >= p.w || iy >= p.h) return -2;
      const at = p.stride * (iy * p.w + ix);
      let latest = -1;
      p.years.forEach((y, ch) => {
        if (y <= upTo && p.s[at + ch] >= over) latest = ch;
      });
      return latest < 0 ? -1 : p.years[latest];
    };
    // the earliest year found up to upTo at a pixel found (its chunk in hand)
    const firstAt = (gx: number, gy: number) => {
      const r = Math.floor(gy / CHUNK), c = Math.floor(gx / CHUNK), p = got.get(r + ":" + c)!;
      const at = p.stride * ((gy - r * CHUNK) * p.w + gx - c * CHUNK);
      return p.years.find((y, ch) => y <= upTo && p.s[at + ch] >= over)!;
    };
    const { px, py } = chunkAt(lon, lat);
    const year = latestAt(px, py);
    if (year < 0) return null;
    // a pixel's key in the whole grid (4,007,520 columns < 2^22)
    const W = 4194304, key = (gx: number, gy: number) => gy * W + gx;
    const inSeg = new Set<number>([key(px, py)]);
    const qx = [px], qy = [py];
    let cut = false, sx = 0, sy = 0, first = year;
    const reach = joined ? PATCH_GAP + 1 : 1;
    for (let q = 0; q < qx.length; q++) {
      const x = qx[q], y = qy[q];
      sx += x;
      sy += y;
      if (joined) first = Math.min(first, firstAt(x, y));
      for (let dy = -reach; dy <= reach; dy++)
        for (let dx = -reach; dx <= reach; dx++) {
          if (!dx && !dy) continue;
          const nx = x + dx, ny = y + dy, k = key(nx, ny);
          if (inSeg.has(k)) continue;
          const v = latestAt(nx, ny);
          if (v === -2) cut = true;
          if (joined ? v < 0 : v !== year) continue;
          if (inSeg.size >= max) {
            cut = true;
            continue;
          }
          inSeg.add(k);
          qx.push(nx);
          qy.push(ny);
        }
    }
    // the segment's pixels row by row
    const xsOf = new Map<number, number[]>();
    for (let q = 0; q < qx.length; q++) {
      const a = xsOf.get(qy[q]);
      if (a) a.push(qx[q]);
      else xsOf.set(qy[q], [qx[q]]);
    }
    const { edges, filled } = traceOutline(runsFrom(xsOf));
    const lonOf = (gx: number) => X0 + gx * RES, latOf = (gy: number) => Y0 - gy * RES;
    const n = qx.length;
    // a point inside the outline (a hole's too)
    const has = (lon_: number, lat_: number) => {
      const p = chunkAt(lon_, lat_), f = filled.get(p.py);
      if (!f) return false;
      let lo = 0, hi = f.length - 1;
      while (lo <= hi) {
        const m = (lo + hi) >> 1;
        if (f[m][1] <= p.px) lo = m + 1;
        else if (f[m][0] > p.px) hi = m - 1;
        else return true;
      }
      return false;
    };
    return { year, first, n, cut, edges, has, lon: lonOf(sx / n + 0.5), lat: latOf(sy / n + 0.5) };
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
    yearMaskId,
    layerId,
    scoreAt,
    detectedAt,
    segmentAt,
    outlinesIn,
    patchesIn,
    checkAt,
    readAt,
    chunksIn,
    loadedIn,
    model,
    tilesRead: () => tilesRead,
    whenReady: (f: () => void) => {
      onReady = f;
    },
    whenCheck: (f: () => void) => {
      onCheck = f;
    },
    checkIn: () => checkN,
  };
}

export { COLORMAP_INDEX };
