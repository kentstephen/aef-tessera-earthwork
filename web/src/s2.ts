/**
 * Sentinel-2 footprints for the map, found in the page: single scenes from the
 * official bucket (sentinel-2-l2a COGs on AWS's open data, sentinel-cogs), found
 * through Element 84's Earth Search. Each MGRS tile's clearest scene of the year
 * is the yearly, and the next two clearest are the fill under its clouds and
 * edges; the year still in progress weighs age too, so it shows the ground as it
 * is now. The map reads and draws each footprint's TCI COG itself (ui.js,
 * s2CogLayers), with its clouds and their shadows cut out by the scene's SCL; a
 * fourth scene lies under them uncut, the base, so ground the SCL takes for cloud
 * in every scene (bright fresh soil and concrete can be) is never a hole. A new
 * look (N on the map) moves each tile down its list: the next scene as the
 * yearly, with its own fill and base.
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
  /** the scene classification (20 m), its clouds cut out of the picture */
  scl?: string;
  /** the scene under every other, drawn uncut */
  base?: boolean;
};

type EsFeature = {
  id: string;
  bbox: [number, number, number, number];
  properties: Record<string, number | string | undefined>;
  assets: Record<string, { href: string }>;
};
/** Each MGRS tile's scenes of the year under a box, best first, as Earth Search answered. */
const scenes = new Map<string, Promise<EsFeature[][]>>();

/** The year in progress: a month older weighs as a point more of cloud (percent of the tile). */
const MONTH_NOW = 1;

/**
 * For each MGRS tile under the box, its scenes of the year ranked: fewest clouds, then the least of the tile left
 * empty by the swath's edge; the year in progress adds each scene's age to that. Scenes the browser cannot read are
 * left out: a few in 2017 to 2020 point at the old JPEG2000 files (s3://sentinel-s2-l2a, no COG), and one of those
 * as the base left every cloud cut over it a hole.
 */
function sceneLists(year: number, bbox: number[]): Promise<EsFeature[][]> {
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
        // the year in progress asks newest first, so the 200 kept are the latest, not only the clearest
        query: { "eo:cloud_cover": { lt: 20 } },
        sortby: [now ? { field: "properties.datetime", direction: "desc" } : { field: "properties.eo:cloud_cover", direction: "asc" }],
        limit: 200,
      }),
    })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`Earth Search ${r.status}`))))
      .then((j) => {
        const cog = (h?: string) => !h || /^https:.*\.tiff?$/i.test(h);
        const byTile = new Map<string, EsFeature[]>();
        for (const f of j.features as EsFeature[]) {
          if (!f.assets.visual || !f.bbox || !cog(f.assets.visual.href) || !cog(f.assets.scl?.href)) continue;
          const tile = String(f.properties["grid:code"] ?? f.properties["s2:mgrs_tile"] ?? f.id.split("_")[1]);
          const l = byTile.get(tile) ?? [];
          l.push(f);
          byTile.set(tile, l);
        }
        const cloud = (f: EsFeature) => Number(f.properties["eo:cloud_cover"] ?? 100);
        const empty = (f: EsFeature) => Number(f.properties["s2:nodata_pixel_percentage"] ?? 50);
        const rank = (f: EsFeature) => cloud(f) + 0.5 * empty(f);
        const months = (f: EsFeature) => (Date.now() - (Date.parse(String(f.properties.datetime ?? "")) || 0)) / (30 * 864e5);
        const rankNow = (f: EsFeature) => rank(f) + MONTH_NOW * months(f);
        const lists = [...byTile.values()];
        for (const l of lists) l.sort((a, b) => (now ? rankNow(a) - rankNow(b) : rank(a) - rank(b)));
        return lists;
      });
    p.catch(() => scenes.delete(k));
    scenes.set(k, p);
  }
  return p;
}

/**
 * The look: for each MGRS tile, its look-th scene as the yearly (0 the best), the next two as the fill and the
 * fourth as the base, wrapping around the tile's list, so a new look (N) is the next scene down with its own fill.
 */
function sceneItems(year: number, bbox: number[], look = 0): Promise<S2Item[]> {
  return sceneLists(year, bbox).then((lists) => {
    const yearly: S2Item[] = [];
    const fill: S2Item[] = [];
    for (const l of lists) {
      const n = Math.min(4, l.length);
      for (let i = 0; i < n; i++) {
        const f = l[(look + i) % l.length];
        const bands = Object.fromEntries(Object.entries(ES_BANDS).filter(([a]) => f.assets[a]).map(([a, b]) => [b, f.assets[a].href]));
        const epsg = Number(f.properties["proj:epsg"]) || undefined;
        const scl = f.assets.scl?.href;
        const it: S2Item = { id: i ? `${f.id}#fill` : f.id, url: f.assets.visual.href, bbox: f.bbox, bands, epsg, scl };
        if (i) fill.push({ ...it, fill: true, base: i === 3 });
        else yearly.push(it);
      }
    }
    return [...yearly, ...fill];
  });
}

/** The year's footprints under one web mercator tile, in the look asked for: yearly first, the fill after them. */
export async function s2Items(year: number, { z, x, y }: { z: number; x: number; y: number }, look = 0): Promise<S2Item[]> {
  const n = 2 ** z;
  const lon = (i: number) => (i / n) * 360 - 180;
  const lat = (j: number) => (Math.atan(Math.sinh(Math.PI * (1 - (2 * j) / n))) * 180) / Math.PI;
  return sceneItems(year, [lon(x), lat(y + 1), lon(x + 1), lat(y)], look);
}
