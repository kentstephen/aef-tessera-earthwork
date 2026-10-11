import type { Layer } from "@deck.gl/core";
import { COGLayer, MosaicLayer } from "@developmentseed/deck.gl-geotiff";
import type { GetTileDataOptions } from "@developmentseed/deck.gl-geotiff";
import { CreateTexture } from "@developmentseed/deck.gl-raster/gpu-modules";
import { DecoderPool, GeoTIFF, type Overview, PerOriginSemaphore } from "@developmentseed/geotiff";
import GeoTiffWorker from "@developmentseed/geotiff/pool/worker?worker";
import type { ShaderModule } from "@luma.gl/shadertools";
import type { Texture } from "@luma.gl/core";

/**
 * Sentinel-2 yearly true color mosaics by Earth Genome on Source Cooperative.
 * The STAC search names the footprints under the view; each footprint's TCI
 * COG is read and decoded here (workers), through the data.source.coop proxy.
 */
export const S2_YEARS = [2022, 2023, 2024, 2025];
const STAC = "https://stac.earthgenome.org/search";
const YEARLY = "sentinel2-yearly-mosaics";
// fills what the yearly mosaics leave (cloud gaps), painted under them
const FILL = "sentinel2-temporal-mosaics";

type Footprint = { id: string; url: string; bbox: [number, number, number, number] };
/** The footprints one search answered, for one year: each batch is its own MosaicLayer. */
type Batch = { key: string; yearly: Footprint[]; fill: Footprint[]; bbox: [number, number, number, number] };

type StacFeature = {
  id: string;
  collection: string;
  bbox: [number, number, number, number];
  assets: Record<string, { href: string }>;
};

const searches = new Map<string, Promise<StacFeature[]>>();
const byYear = new Map<number, { ids: Set<string>; batches: Batch[] }>();
const asked = new Set<string>();

function search(bbox: number[]): Promise<StacFeature[]> {
  const k = bbox.map((v) => v.toFixed(3)).join(",");
  let p = searches.get(k);
  if (!p) {
    p = fetch(STAC, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ collections: [YEARLY, FILL], bbox, limit: 200 }),
    })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`STAC ${r.status}`))))
      .then((j) => j.features as StacFeature[]);
    p.catch(() => searches.delete(k));
    searches.set(k, p);
  }
  return p;
}

/** The year's footprints among a search's features. */
function footprints(features: StacFeature[], year: number): { yearly: Footprint[]; fill: Footprint[] } {
  const span = `${year}-01-01_${year + 1}-01-01`;
  const yearly: Footprint[] = [];
  const fill: Footprint[] = [];
  for (const f of features) {
    if (!f.id.endsWith(span) || !f.assets.TCI || !f.bbox) continue;
    const fp = { id: `${f.id}|${f.collection}`, url: f.assets.TCI.href, bbox: f.bbox };
    (f.collection === FILL ? fill : yearly).push(fp);
  }
  if (!yearly.length) {
    // The STAC lags the bucket (2025 is in the bucket for tiles the search does
    // not know yet): the same MGRS tile's path with the year swapped. A tile
    // that is not there reads as empty, not as an error
    const seen = new Set<string>();
    for (const f of features) {
      if (f.collection !== YEARLY || !f.assets.TCI) continue;
      const tile = f.id.split("_")[0];
      if (seen.has(tile)) continue;
      seen.add(tile);
      const id = `${tile}_${span}`;
      const base = f.assets.TCI.href.split("/").slice(0, -2).join("/");
      yearly.push({ id: `${id}|${YEARLY}`, url: `${base}/${id}/TCI.tif`, bbox: f.bbox });
    }
  }
  return { yearly, fill };
}

/**
 * Look for footprints over a web mercator grid at zoom 7 to 9 under the view
 * (and half of it again on every side), once per grid tile.
 */
export function discover(view: [number, number, number, number], zoom: number, years: number[], onNew: () => void) {
  const z = Math.max(7, Math.min(9, Math.floor(zoom)));
  const n = 2 ** z;
  const tx = (lon: number) => Math.min(n - 1, Math.max(0, Math.floor(((lon + 180) / 360) * n)));
  const ty = (lat: number) => {
    const s = Math.sin((lat * Math.PI) / 180);
    return Math.min(n - 1, Math.max(0, Math.floor((0.5 - Math.log((1 + s) / (1 - s)) / (4 * Math.PI)) * n)));
  };
  const tlon = (x: number) => (x / n) * 360 - 180;
  const tlat = (y: number) => (Math.atan(Math.sinh(Math.PI * (1 - (2 * y) / n))) * 180) / Math.PI;
  const [W, S, E, N] = view;
  const x0 = tx(W), x1 = tx(E), y0 = ty(N), y1 = ty(S);
  if ((x1 - x0 + 1) * (y1 - y0 + 1) > 64) return;
  for (let x = x0; x <= x1; x++) {
    for (let y = y0; y <= y1; y++) {
      const bbox = [tlon(x), tlat(y + 1), tlon(x + 1), tlat(y)];
      for (const year of years) {
        const k = `${year}/${z}/${x}/${y}`;
        if (asked.has(k)) continue;
        asked.add(k);
        search(bbox).then(
          (features) => {
            let e = byYear.get(year);
            if (!e) {
              e = { ids: new Set(), batches: [] };
              byYear.set(year, e);
            }
            const { yearly, fill } = footprints(features, year);
            const bt: Batch = { key: k.replace(/\//g, "-"), yearly: [], fill: [], bbox: [180, 90, -180, -90] };
            for (const [part, list] of [["yearly", yearly], ["fill", fill]] as const) {
              for (const fp of list) {
                if (e.ids.has(fp.id)) continue;
                e.ids.add(fp.id);
                bt[part].push(fp);
                bt.bbox = [
                  Math.min(bt.bbox[0], fp.bbox[0]),
                  Math.min(bt.bbox[1], fp.bbox[1]),
                  Math.max(bt.bbox[2], fp.bbox[2]),
                  Math.max(bt.bbox[3], fp.bbox[3]),
                ];
              }
            }
            if (bt.yearly.length || bt.fill.length) {
              e.batches.push(bt);
              onNew();
            }
          },
          () => asked.delete(k),
        );
      }
    }
  }
}

// ---- reading and drawing ------------------------------------------------------
// source.coop answers over HTTP/2, so more than the browser's 6 per host
const limiter = new PerOriginSemaphore({ maxRequests: 16 });
let pool: DecoderPool | null = null;
const tiffs = new Map<string, Promise<GeoTIFF>>();

function tiff(url: string, priority: () => [number, number]): Promise<GeoTIFF | null> {
  let p = tiffs.get(url);
  if (!p) {
    p = GeoTIFF.fromUrl(url, { concurrencyLimiter: limiter, getPriority: priority });
    // a failed header is asked again next time
    p.catch(() => tiffs.get(url) === p && tiffs.delete(url));
    tiffs.set(url, p);
  }
  // a footprint that cannot be read (listed, but not in the bucket) is nothing there
  return p.catch(() => null);
}

type S2TileData = { texture: Texture; width: number; height: number; byteLength: number };

async function tileData(image: GeoTIFF | Overview, { device, x, y, signal, pool }: GetTileDataOptions): Promise<S2TileData> {
  const { array } = await image.fetchTile(x, y, { boundless: false, pool, signal });
  const { width, height } = array;
  const data = (array as { data: Uint8Array }).data;
  const px = width * height;
  // TCI's black (0, 0, 0) is nodata: alpha 0 there, so the linear filter's blend
  // of an edge pixel with its nodata neighbor carries a falling alpha the shader undoes
  let rgba = data;
  if (data.length === 3 * px) {
    rgba = new Uint8Array(4 * px);
    for (let i = 0; i < px; i++) {
      const r = data[3 * i], g = data[3 * i + 1], b = data[3 * i + 2];
      rgba[4 * i] = r;
      rgba[4 * i + 1] = g;
      rgba[4 * i + 2] = b;
      rgba[4 * i + 3] = r || g || b ? 255 : 0;
    }
  }
  const texture = device.createTexture({
    data: rgba,
    format: "rgba8unorm",
    width,
    height,
    sampler: { magFilter: "linear", minFilter: "linear" },
  });
  return { texture, width, height, byteLength: rgba.byteLength };
}

/**
 * Nodata (alpha under a half) is dropped; at a footprint edge the filtered color
 * is the edge pixel's color times the filtered alpha, so dividing by alpha gives
 * the pixel back with no dark fringe.
 */
const S2Look = {
  name: "s2Look",
  inject: {
    "fs:DECKGL_FILTER_COLOR": `
  if (color.a < 0.5) discard;
  color = vec4(color.rgb / color.a, 1.0);
`,
  },
} as const satisfies ShaderModule;

/**
 * The year's imagery under the view as MosaicLayers, fill under yearly. Shown or
 * hidden by `visible` (deck checks every parent's visible on each draw; an
 * opacity change never reaches tiles already loaded). Hidden years still load.
 */
export function s2Layers(opts: {
  year: number;
  shownYear: number;
  visible: boolean;
  view: [number, number, number, number];
  prefix: string;
  beforeId: string;
}): Layer[] {
  const { year, shownYear, visible, view, prefix, beforeId } = opts;
  const e = byYear.get(year);
  if (!e) return [];
  if (!pool) pool = new DecoderPool({ createWorker: () => new GeoTiffWorker() });
  const [W, S, E, N] = view;
  const live = e.batches.filter((b) => b.bbox[0] < E && b.bbox[2] > W && b.bbox[1] < N && b.bbox[3] > S);
  const parts = [
    ...live.filter((b) => b.fill.length).map((b) => [b, "fill"] as const),
    ...live.filter((b) => b.yearly.length).map((b) => [b, "yearly"] as const),
  ];
  const p = pool;
  return parts.map(
    ([bt, part]) =>
      new MosaicLayer<Footprint, GeoTIFF | null>({
        id: `${prefix}${bt.key}-${part}`,
        sources: bt[part],
        maxCacheSize: 0,
        minZoom: 7,
        visible: visible && year === shownYear,
        getSource: (src) => tiff(src.url, () => [year === shownYear ? 0 : 1, part === "fill" ? 1 : 0]),
        renderSource: (src, { data, signal }) =>
          !data
            ? null
            : new COGLayer<S2TileData>({
                id: `${prefix}${year}-${src.id}`,
                geotiff: data,
                pool: p,
                signal,
                refinementStrategy: "best-available",
                maxRequests: 16,
                getTileData: tileData,
                renderTile: (d) => ({
                  renderPipeline: [
                    { module: CreateTexture, props: { textureName: d.texture } },
                    { module: S2Look },
                  ],
                }),
              }),
        // @ts-expect-error beforeId is read by the MapLibre overlay, not typed on the layer
        beforeId,
      }),
  );
}
