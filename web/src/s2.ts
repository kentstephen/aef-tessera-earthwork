/**
 * Sentinel-2 footprints for the map, found in the page: single scenes from the
 * official bucket (sentinel-2-l2a COGs on AWS's open data, sentinel-cogs), found
 * through Element 84's Earth Search. Each MGRS tile's clearest scene of the year
 * is the yearly, and the next two clearest are the fill under its clouds and
 * edges; the year still in progress takes its latest clear scene instead, the
 * ground as it is now. The map reads and draws each footprint's TCI COG itself
 * (ui.js, s2CogLayers).
 */
const EARTH_SEARCH = "https://earth-search.aws.element84.com/v1/search";
const ES_BANDS: Record<string, string> = { blue: "B02", green: "B03", red: "B04", nir08: "B8A", swir16: "B11", swir22: "B12" };

/** One footprint, as the map's s2Discover takes it. */
export type S2Item = {
  id: string;
  url: string;
  bbox: [number, number, number, number];
  fill?: boolean;
  bands: Record<string, string>;
  /** the COGs' CRS: a UTM zone, 326zz north or 327zz south */
  epsg?: number;
};

type EsFeature = {
  id: string;
  bbox: [number, number, number, number];
  properties: Record<string, number | string | undefined>;
  assets: Record<string, { href: string }>;
};
const scenes = new Map<string, Promise<S2Item[]>>();

/** A scene this clear counts as clear for the year in progress, where the newest of them wins. */
const CLEAR_NOW = 5;

/**
 * For each MGRS tile under the box, its clearest scene of the year (fewest clouds, then the least of the tile left
 * empty by the swath's edge), and the next two as the fill. The year in progress ranks its clear scenes newest first.
 */
function sceneItems(year: number, bbox: number[]): Promise<S2Item[]> {
  const k = `${year}:${bbox.map((v) => v.toFixed(3)).join(",")}`;
  let p = scenes.get(k);
  if (!p) {
    const now = year >= new Date().getUTCFullYear();
    p = fetch(EARTH_SEARCH, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        collections: ["sentinel-2-l2a"],
        bbox,
        datetime: `${year}-01-01T00:00:00Z/${year}-12-31T23:59:59Z`,
        // the year in progress asks newest first, so the 200 kept are the latest, not the clearest of the year
        query: { "eo:cloud_cover": { lt: 20 } },
        sortby: [now ? { field: "properties.datetime", direction: "desc" } : { field: "properties.eo:cloud_cover", direction: "asc" }],
        limit: 200,
      }),
    })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`Earth Search ${r.status}`))))
      .then((j) => {
        const byTile = new Map<string, EsFeature[]>();
        for (const f of j.features as EsFeature[]) {
          if (!f.assets.visual || !f.bbox) continue;
          const tile = String(f.properties["grid:code"] ?? f.properties["s2:mgrs_tile"] ?? f.id.split("_")[1]);
          const l = byTile.get(tile) ?? [];
          l.push(f);
          byTile.set(tile, l);
        }
        const cloud = (f: EsFeature) => Number(f.properties["eo:cloud_cover"] ?? 100);
        const empty = (f: EsFeature) => Number(f.properties["s2:nodata_pixel_percentage"] ?? 50);
        const rank = (f: EsFeature) => cloud(f) + 0.5 * empty(f);
        // the year in progress: clear and mostly full first, newest of those first; the rest as any other year
        const days = (f: EsFeature) => Date.parse(String(f.properties.datetime ?? "")) / 864e5 || 0;
        const rankNow = (f: EsFeature) => (cloud(f) < CLEAR_NOW && empty(f) < 20 ? -1e6 - days(f) : rank(f));
        const yearly: S2Item[] = [];
        const fill: S2Item[] = [];
        for (const l of byTile.values()) {
          l.sort((a, b) => (now ? rankNow(a) - rankNow(b) : rank(a) - rank(b)));
          l.slice(0, 3).forEach((f, i) => {
            const bands = Object.fromEntries(Object.entries(ES_BANDS).filter(([a]) => f.assets[a]).map(([a, b]) => [b, f.assets[a].href]));
            const epsg = Number(f.properties["proj:epsg"]) || undefined;
            const it: S2Item = { id: i ? `${f.id}#fill` : f.id, url: f.assets.visual.href, bbox: f.bbox, bands, epsg };
            if (i) fill.push({ ...it, fill: true });
            else yearly.push(it);
          });
        }
        return [...yearly, ...fill];
      });
    p.catch(() => scenes.delete(k));
    scenes.set(k, p);
  }
  return p;
}

/** The year's footprints under one web mercator tile: yearly first, the fill after them. */
export async function s2Items(year: number, { z, x, y }: { z: number; x: number; y: number }): Promise<S2Item[]> {
  const n = 2 ** z;
  const lon = (i: number) => (i / n) * 360 - 180;
  const lat = (j: number) => (Math.atan(Math.sinh(Math.PI * (1 - (2 * j) / n))) * 180) / Math.PI;
  return sceneItems(year, [lon(x), lat(y + 1), lon(x + 1), lat(y)]);
}
