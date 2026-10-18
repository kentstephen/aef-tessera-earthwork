/**
 * Sentinel-2 footprints for the map, found in the page: single scenes from the
 * official bucket (sentinel-2-l2a COGs on AWS's open data, sentinel-cogs), found
 * through Element 84's Earth Search. Each MGRS tile's clearest leaf-on scene of the year
 * is the yearly, and the next two clearest are the fill under its clouds and
 * edges; the year still in progress weighs age too, so it shows the ground as it
 * is now. The map reads and draws each footprint's TCI COG itself (ui.js,
 * s2CogLayers), with its clouds and their shadows cut out by the scene's SCL and
 * the holes filled, tile by tile, from the scenes under it (s2Stack): the fill,
 * then the last of up to four scenes uncut, the base, so ground the SCL takes for cloud in every
 * scene (bright fresh soil and concrete can be) is never a hole. A new
 * look (N on the map) moves each tile down its list: the next scene as the
 * yearly, with its own fill and base.
 */

import { cacheGet, cachePut } from "./disk-cache";

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
  /** the scenes under a yearly footprint, on its MGRS tile: its fill, then its base */
  under?: S2Item[];
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
 * The leaf-on season, as days of the year (mid May to mid September north of the tropics, mid November to mid March
 * south of them): a year gone by ranks every scene in it ahead of every scene outside it, so the years compare in the
 * same season and the change between them is the ground's, not the leaves'. A year with none in season falls back to the
 * rest. Inside the tropics the season is left out.
 */
const SEASON_NORTH: [number, number] = [135, 258];
const SEASON_SOUTH: [number, number] = [319, 75];
const SEASON_PENALTY = 1000;
function outOfSeason(datetime: string, lat: number): boolean {
  if (Math.abs(lat) < 23.5) return false;
  const t = Date.parse(datetime);
  if (!t) return true;
  const doy = Math.floor((t - Date.UTC(new Date(t).getUTCFullYear(), 0, 1)) / 864e5) + 1;
  const [a, b] = lat >= 0 ? SEASON_NORTH : SEASON_SOUTH;
  // the south's season runs past the new year
  return a <= b ? doy < a || doy > b : doy < a && doy > b;
}

/**
 * For each MGRS tile under the box, its scenes of the year ranked: fewest clouds, then the least of the tile left
 * empty by the swath's edge; a year gone by puts the leaf-on season's first (outOfSeason), the year in progress adds
 * each scene's age to that. Scenes the browser cannot read are
 * left out: a few in 2017 to 2020 point at the old JPEG2000 files (s3://sentinel-s2-l2a, no COG), and one of those
 * as the base left every cloud cut over it a hole.
 */
function sceneLists(year: number, bbox: number[]): Promise<EsFeature[][]> {
  const k = `${year}:${bbox.map((v) => v.toFixed(3)).join(",")}`;
  let p = scenes.get(k);
  if (!p) {
    const now = year >= new Date().getUTCFullYear();
    const body = JSON.stringify({
        collections: ["sentinel-2-l2a"],
        bbox,
        datetime: `${year}-01-01T00:00:00Z/${year}-12-31T23:59:59Z`,
        // the year in progress asks newest first, so the 200 kept are the latest, not only the clearest
        // up to 60% cloud: the share is the whole 110 km tile's, so a cloudy one can be clear over the place, and a
        // cloudy region (Lagos 2025: one scene under 20%) still has scenes for a new look (N). The clearest stay first
        query: { "eo:cloud_cover": { lt: 60 } },
        sortby: [now ? { field: "properties.datetime", direction: "desc" } : { field: "properties.eo:cloud_cover", direction: "asc" }],
        limit: 200,
        // the footprints and links left out (the bbox is enough): whole, 200 scenes can be more than Earth Search
        // answers with (2019 over Ashburn, sorted by cloud: a 502 every time, which the browser reports as CORS). The
        // properties the ranking and the reads use named, or an exclude alone answers with the datetime only
        fields: {
          include: ["id", "bbox", "assets", "properties.datetime", "properties.eo:cloud_cover", "properties.s2:nodata_pixel_percentage",
            "properties.grid:code", "properties.s2:mgrs_tile", "properties.proj:epsg"],
          exclude: ["geometry", "links"],
        },
      });
    // kept on disk: a year gone by for a week (Earth Search adds little to it), the year in progress for hours. Only
    // an answer with scenes in it: Earth Search sometimes answers a search with none (or few) of a year that has them,
    // and kept, that left the year blank for the week. A kept answer with no scene is asked again. es2: the answers
    // kept before this are set aside
    const ck = `es2|${body}`;
    const usable = (j: { features?: EsFeature[] }) => Array.isArray(j.features) && j.features.some((f) => f.assets?.visual && f.bbox);
    // a search Earth Search fails (a 5xx, or no answer at all) asked again for half as many scenes, the clearest still first
    const fetchOr100 = async () => {
      const go = (b: string) => fetch(EARTH_SEARCH, { method: "POST", headers: { "Content-Type": "application/json" }, body: b });
      const half = () => go(body.replace('"limit":200', '"limit":100'));
      try {
        const r = await go(body);
        return r.ok || r.status < 500 ? r : half();
      } catch {
        return half();
      }
    };
    const ask = async () => {
      const r = await fetchOr100();
      if (!r.ok) throw new Error(`Earth Search ${r.status}`);
      const b = new Uint8Array(await r.arrayBuffer());
      const j = JSON.parse(new TextDecoder().decode(b)) as { features: EsFeature[] };
      if (usable(j)) cachePut(ck, b, now ? 6 * 36e5 : 7 * 864e5);
      return j;
    };
    p = cacheGet(ck)
      .then((have) => {
        if (!have) return ask();
        const j = JSON.parse(new TextDecoder().decode(have)) as { features: EsFeature[] };
        return usable(j) ? j : ask();
      })
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
        // a year gone by: the leaf-on season's scenes first, whatever their clouds
        const lat = (bbox[1] + bbox[3]) / 2;
        const rankPast = (f: EsFeature) => rank(f) + (outOfSeason(String(f.properties.datetime ?? ""), lat) ? SEASON_PENALTY : 0);
        const lists = [...byTile.values()];
        for (const l of lists) l.sort((a, b) => (now ? rankNow(a) - rankNow(b) : rankPast(a) - rankPast(b)));
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
 * A tile with fewer than four (2017 has three in many places) has its last as the base, so the holes every scene
 * cuts (the 2017 SCL takes bright roofs for cloud) are filled all the same; a scene alone is drawn uncut.
 */
function sceneItems(year: number, bbox: number[], look = 0): Promise<S2Item[]> {
  return sceneLists(year, bbox).then((lists) => {
    const yearly: S2Item[] = [];
    for (const l of lists) {
      const n = Math.min(4, l.length);
      const its: S2Item[] = [];
      for (let i = 0; i < n; i++) {
        const f = l[(look + i) % l.length];
        const bands = Object.fromEntries(Object.entries(ES_BANDS).filter(([a]) => f.assets[a]).map(([a, b]) => [b, f.assets[a].href]));
        const epsg = Number(f.properties["proj:epsg"]) || undefined;
        const scl = n > 1 ? f.assets.scl?.href : undefined;
        its.push({ id: i ? `${f.id}#fill` : f.id, url: f.assets.visual.href, bbox: f.bbox, bands, epsg, scl, ...(i ? { fill: true, base: i === n - 1 } : {}) });
      }
      if (its.length) yearly.push({ ...its[0], under: its.slice(1) });
    }
    return yearly;
  });
}

/** The year's footprints under one web mercator tile, in the look asked for, each with its fill and base under it. */
export async function s2Items(year: number, { z, x, y }: { z: number; x: number; y: number }, look = 0): Promise<S2Item[]> {
  const n = 2 ** z;
  const lon = (i: number) => (i / n) * 360 - 180;
  const lat = (j: number) => (Math.atan(Math.sinh(Math.PI * (1 - (2 * j) / n))) * 180) / Math.PI;
  return sceneItems(year, [lon(x), lat(y + 1), lon(x + 1), lat(y)], look);
}
