import "maplibre-gl/dist/maplibre-gl.css";
import "./style.css";

import type { Layer } from "@deck.gl/core";
import { MapLibreOverlay } from "@deck.gl/maplibre";
import {
  COLORMAP_INDEX,
  Colormap,
  createColormapTexture,
  decodeColormapSprite,
} from "@developmentseed/deck.gl-raster/gpu-modules";
import colormapsUrl from "@developmentseed/deck.gl-raster/gpu-modules/colormaps.png?url";
import { ZarrLayer } from "@developmentseed/deck.gl-zarr";
import type { Device, Texture } from "@luma.gl/core";
import * as maplibregl from "maplibre-gl";
import maplibreWorkerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url";
import type * as zarr from "zarrita";

import { type AefArray, makeGetTileData, MIN_ZOOM, openMosaic, type PairTileData, YEAR_ORIGIN, YEARS } from "./aef";
import { EarthworkScore } from "./earthwork-score";
import { discover, S2_YEARS, s2Layers } from "./s2";

type ModelKey = "current" | "abroad";
type Model = { w: number[]; b: number };

const MODEL_NOTE: Record<ModelKey, string> = {
  current: "Taught on US repeat lidar only. Stronger on US building sites.",
  abroad: "Taught with truth from outside the 48 states too. Stronger on mines and abroad.",
};

// Huntsville AL, one of the US sites the model learned from
const HOME = { center: [-86.8, 34.69] as [number, number], zoom: 13.5 };
const STYLE = "https://tiles.openfreemap.org/styles/dark";
// deck layers go under the basemap's labels
const BEFORE = "highway_name_other";

const state = {
  first: 2023,
  last: 2025,
  model: "current" as ModelKey,
  /** Space held: the imagery in place of the earthwork. */
  holding: false,
  /** Two maps: the imagery on the left, the earthwork on the right. */
  pair: false,
  imgYear: S2_YEARS[S2_YEARS.length - 1],
  /** The shown year's imagery is in, so the other years load too. */
  warm: false,
};

const models: Record<ModelKey, Model> = {
  current: await (await fetch(`${import.meta.env.BASE_URL}models/current.json`)).json(),
  abroad: await (await fetch(`${import.meta.env.BASE_URL}models/abroad.json`)).json(),
};

// GPU resources made once the layer hands over its device
let gpu: { weights: Record<ModelKey, Texture>; colormap: Texture } | null = null;
let gpuPending = false;
const colormapBytes = fetch(colormapsUrl).then((r) => r.arrayBuffer());

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
      gpu = {
        weights: { current: weight(models.current), abroad: weight(models.abroad) },
        colormap: createColormapTexture(device, img),
      };
      render();
    });
}

// MapLibre 6 is ESM only and finds its worker by URL; Vite bundles the worker with its shared chunk
maplibregl.setWorkerUrl(maplibreWorkerUrl);

const CREDITS = [
  '<a href="https://source.coop/tge-labs/aef-mosaic" target="_blank" rel="noopener">AlphaEarth Foundations</a> by Google and Google DeepMind, ' +
    '<a href="https://source.coop/earthgenome/earthindeximagery" target="_blank" rel="noopener">Sentinel-2 mosaics</a> by Earth Genome ' +
    '(<a href="https://creativecommons.org/licenses/by/4.0/" target="_blank" rel="noopener">CC BY 4.0</a>), via Source Cooperative',
];

const map = new maplibregl.Map({
  container: "map",
  style: STYLE,
  center: HOME.center,
  zoom: HOME.zoom,
  hash: true,
  attributionControl: false,
});
const narrow = window.matchMedia("(max-width: 760px)");
map.addControl(new maplibregl.AttributionControl({ customAttribution: CREDITS, compact: narrow.matches }), "bottom-right");
// compact, the credits start folded behind the info button
map.once("load", () => document.querySelector(".maplibregl-ctrl-attrib")?.classList.remove("maplibregl-compact-show"));
const overlay = new MapLibreOverlay({ interleaved: true, layers: [] });
map.addControl(overlay);

// the pair's left map: the imagery, under the same camera
const map2 = new maplibregl.Map({
  container: "map2",
  style: STYLE,
  center: HOME.center,
  zoom: HOME.zoom,
  attributionControl: false,
});
const overlay2 = new MapLibreOverlay({ interleaved: true, layers: [] });
map2.addControl(overlay2);

let syncing = false;
function follow(from: maplibregl.Map, to: maplibregl.Map) {
  from.on("move", () => {
    if (syncing || !state.pair) return;
    syncing = true;
    to.jumpTo({ center: from.getCenter(), zoom: from.getZoom(), bearing: from.getBearing(), pitch: from.getPitch() });
    syncing = false;
  });
}
follow(map, map2);
follow(map2, map);

const { arr, attrs } = await openMosaic();

const tiles = { loaded: 0 };

function earthworkLayer() {
  const firstIdx = state.first - YEAR_ORIGIN;
  const lastIdx = state.last - YEAR_ORIGIN;
  const g = gpu;
  const m = models[state.model];
  return new ZarrLayer<zarr.Readable, "int8", PairTileData>({
    id: `earthwork-${state.first}-${state.last}`,
    node: arr as AefArray,
    metadata: attrs,
    selection: { time: firstIdx, band: null },
    getTileData: makeGetTileData(firstIdx, lastIdx, onDevice),
    renderTile: (data: PairTileData) => ({
      renderPipeline: g
        ? [
            {
              module: EarthworkScore,
              props: {
                firstTex: data.firstTex,
                lastTex: data.lastTex,
                weightTex: g.weights[state.model],
                bias: m.b,
                gateLo: 0.05,
                gateHi: 0.15,
              },
            },
            { module: Colormap, props: { colormapTexture: g.colormap, colormapIndex: COLORMAP_INDEX.viridis } },
          ]
        : [],
    }),
    visible: !state.holding,
    minZoom: MIN_ZOOM,
    // source.coop answers over HTTP/2, so more than the browser's 6 per host
    maxRequests: 12,
    // each tile is two years x 64 bands (8 MB on the GPU)
    maxCacheSize: 24,
    onTileLoad: () => {
      tiles.loaded++;
      status();
    },
    updateTriggers: { renderTile: [state.model, !!g] },
    // @ts-expect-error beforeId is read by the MapLibre overlay, not typed on the layer
    beforeId: BEFORE,
  });
}

/** The view, and half of it again on every side, over both maps. */
function s2View(): [number, number, number, number] {
  const bs = (state.pair ? [map, map2] : [map]).map((m) => m.getBounds());
  const W = Math.min(...bs.map((b) => b.getWest()));
  const E = Math.max(...bs.map((b) => b.getEast()));
  const S = Math.min(...bs.map((b) => b.getSouth()));
  const N = Math.max(...bs.map((b) => b.getNorth()));
  const dx = (E - W) / 2, dy = (N - S) / 2;
  return [W - dx, Math.max(-85, S - dy), E + dx, Math.min(85, N + dy)];
}

/** The imagery years mounted: the one shown, the rest once it is in (a year step is then instant). */
function s2Years() {
  return state.warm ? [state.imgYear, ...S2_YEARS.filter((y) => y !== state.imgYear)] : [state.imgYear];
}

function s2For(left: boolean): Layer[] {
  const showing = left ? state.pair : state.holding;
  // ahead of a hold, the imagery loads once the earthwork does
  if (!showing && (left || map.getZoom() < MIN_ZOOM)) return [];
  const view = s2View();
  return s2Years().flatMap((year) =>
    s2Layers({ year, shownYear: state.imgYear, visible: showing, view, prefix: left ? "s2l-" : "s2-", beforeId: BEFORE }),
  );
}

function render() {
  const s2 = s2For(false);
  overlay.setProps({ layers: [...s2, earthworkLayer()] });
  overlay2.setProps({ layers: s2For(true) });
  status();
  renderCard();
}

function look() {
  if (!(state.holding || state.pair || map.getZoom() >= MIN_ZOOM)) return;
  discover(s2View(), map.getZoom(), s2Years(), () => {
    render();
    // the shown year's footprints are named: let the others follow shortly
    if (!state.warm) setTimeout(() => {
      state.warm = true;
      look();
      render();
    }, 1500);
  });
}
map.on("moveend", () => {
  look();
  render();
});

// ---- the panel ---------------------------------------------------------------
const $ = <T extends HTMLElement>(sel: string) => document.querySelector(sel) as T;

function fillYears(sel: HTMLSelectElement, value: number) {
  sel.innerHTML = YEARS.map((y) => `<option value="${y}"${y === value ? " selected" : ""}>${y}</option>`).join("");
}
fillYears($("#first"), state.first);
fillYears($("#last"), state.last);

$("#first").addEventListener("change", (e) => {
  state.first = Number((e.target as HTMLSelectElement).value);
  render();
});
$("#last").addEventListener("change", (e) => {
  state.last = Number((e.target as HTMLSelectElement).value);
  render();
});

function setModel(m: ModelKey) {
  state.model = m;
  document.querySelectorAll<HTMLButtonElement>("[data-model]").forEach((b) => {
    b.classList.toggle("on", b.dataset.model === m);
  });
  $("#model-note").textContent = MODEL_NOTE[m];
  render();
}
document.querySelectorAll<HTMLButtonElement>("[data-model]").forEach((b) => {
  b.addEventListener("click", () => setModel(b.dataset.model as ModelKey));
});

function status() {
  const z = map.getZoom();
  const el = $("#status");
  if (state.first >= state.last) el.textContent = "The first year has to come before the last.";
  else if (z < MIN_ZOOM) el.textContent = `Zoom in to see earthwork (from zoom ${MIN_ZOOM}; now ${z.toFixed(1)}).`;
  else el.textContent = `${tiles.loaded} chunks read`;
}

// ---- the imagery: hold space, or pair (P) -------------------------------------
function renderCard() {
  const card = $("#s2card");
  card.hidden = !(state.holding || state.pair);
  $("#s2year").textContent = String(state.imgYear);
  card.classList.toggle("left", state.pair);
  $("#pair").classList.toggle("on", state.pair);
}

function stepImg(d: number) {
  const i = S2_YEARS.indexOf(state.imgYear);
  const n = S2_YEARS[Math.max(0, Math.min(S2_YEARS.length - 1, i + d))];
  if (n !== state.imgYear) {
    state.imgYear = n;
    look();
    render();
  }
}

function setHold(on: boolean) {
  if (on === state.holding || (on && state.pair)) return;
  state.holding = on;
  document.body.classList.toggle("holding", on);
  look();
  render();
}

function setPair(on: boolean) {
  if (on === state.pair) return;
  if (on) setHold(false);
  state.pair = on;
  document.body.classList.toggle("pair", on);
  map.resize();
  map2.resize();
  if (on) map2.jumpTo({ center: map.getCenter(), zoom: map.getZoom(), bearing: map.getBearing(), pitch: map.getPitch() });
  look();
  render();
}
$("#pair").addEventListener("click", () => setPair(!state.pair));

window.addEventListener("keydown", (e) => {
  if (e.target instanceof HTMLSelectElement) return;
  const k = e.key;
  if (k === " ") {
    // the space bar is the hold, never a press of the focused button
    e.preventDefault();
    if (!e.repeat) setHold(true);
  } else if (k === "m" || k === "M") setModel(state.model === "current" ? "abroad" : "current");
  else if (k === "p" || k === "P") setPair(!state.pair);
  else if (k === "[" || k === "]") stepImg(k === "]" ? 1 : -1);
  else if ((k === "b" || k === "B") && (state.holding || state.pair)) {
    // the imagery's first year and its latest, back and forth
    const a = S2_YEARS[0], b = S2_YEARS[S2_YEARS.length - 1];
    state.imgYear = state.imgYear === b ? a : b;
    look();
    render();
  }
});
window.addEventListener("keyup", (e) => {
  if (e.key === " ") setHold(false);
});
window.addEventListener("blur", () => setHold(false));

// while holding, the wheel steps the imagery year instead of zooming
let wheelAcc = 0, lastStep = 0;
window.addEventListener(
  "wheel",
  (e) => {
    if (!state.holding) return;
    e.preventDefault();
    e.stopPropagation();
    wheelAcc += e.deltaY;
    const now = performance.now();
    if (Math.abs(wheelAcc) >= 40 && now - lastStep > 140) {
      stepImg(wheelAcc > 0 ? 1 : -1);
      wheelAcc = 0;
      lastStep = now;
    }
  },
  { capture: true, passive: false },
);

setModel(state.model);
look();
