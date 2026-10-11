import "maplibre-gl/dist/maplibre-gl.css";
import "./style.css";

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

type ModelKey = "current" | "abroad";
type Model = { w: number[]; b: number };

const MODEL_NOTE: Record<ModelKey, string> = {
  current: "Taught on US repeat lidar only. Stronger on US building sites.",
  abroad: "Taught with truth from outside the 48 states too. Stronger on mines and abroad.",
};

// Huntsville AL, one of the US sites the model learned from
const HOME = { center: [-86.8, 34.69] as [number, number], zoom: 13.5 };

const state = {
  first: 2023,
  last: 2025,
  model: "current" as ModelKey,
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

const map = new maplibregl.Map({
  container: "map",
  style: "https://tiles.openfreemap.org/styles/dark",
  center: HOME.center,
  zoom: HOME.zoom,
  hash: true,
  attributionControl: {
    customAttribution:
      "AlphaEarth Foundations Satellite Embedding dataset by Google and Google DeepMind (CC BY 4.0), via Source Cooperative",
  },
});
const overlay = new MapLibreOverlay({ interleaved: true, layers: [] });
map.addControl(overlay);

const { arr, attrs } = await openMosaic();

const tiles = { loading: 0, loaded: 0 };

function layer() {
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
    beforeId: "highway_name_other",
  });
}

function render() {
  overlay.setProps({ layers: [layer()] });
  status();
}

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
window.addEventListener("keydown", (e) => {
  if (e.target instanceof HTMLSelectElement) return;
  if (e.key === "m" || e.key === "M") setModel(state.model === "current" ? "abroad" : "current");
});

function status() {
  const z = map.getZoom();
  const el = $("#status");
  if (state.first >= state.last) el.textContent = "The first year has to come before the last.";
  else if (z < MIN_ZOOM) el.textContent = `Zoom in to see earthwork (from zoom ${MIN_ZOOM}; now ${z.toFixed(1)}).`;
  else el.textContent = `${tiles.loaded} chunks read`;
}
map.on("moveend", status);

setModel(state.model);
