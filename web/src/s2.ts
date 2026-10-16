/**
 * Sentinel-2 footprints for the map, found in the page: Earth Genome's STAC
 * search, the yearly mosaics and the fill mosaics (cloud gaps) under them. The
 * map reads and draws each footprint's TCI COG itself (ui.js, s2CogLayers).
 * Before Earth Genome's first year, single scenes from Element 84's Earth Search
 * (sentinel-2-l2a, COGs on AWS's open bucket): each MGRS tile's clearest scene of
 * the year, and the next clearest as the fill under its clouds and edges.
 */
const STAC = "https://stac.earthgenome.org/search";
const YEARLY = "sentinel2-yearly-mosaics";
const FILL = "sentinel2-temporal-mosaics";
const BANDS = ["B02", "B03", "B04", "B8A", "B11", "B12"];
/** Earth Genome's mosaics start here; the years before come from Earth Search. */
export const MOSAIC_FIRST = 2022;
const EARTH_SEARCH = "https://earth-search.aws.element84.com/v1/search";
const ES_BANDS: Record<string, string> = { blue: "B02", green: "B03", red: "B04", nir08: "B8A", swir16: "B11", swir22: "B12" };

type StacFeature = {
  id: string;
  collection: string;
  bbox: [number, number, number, number];
  assets: Record<string, { href: string }>;
};

/** One footprint, as the map's s2Discover takes it. */
export type S2Item = {
  id: string;
  url: string;
  bbox: [number, number, number, number];
  fill?: boolean;
  bands: Record<string, string>;
};

const searches = new Map<string, Promise<StacFeature[]>>();

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

const bandsOf = (f: StacFeature) =>
  Object.fromEntries(Object.entries(f.assets).filter(([k]) => BANDS.includes(k)).map(([k, a]) => [k, a.href]));

type EsFeature = {
  id: string;
  bbox: [number, number, number, number];
  properties: Record<string, number | string | undefined>;
  assets: Record<string, { href: string }>;
};
const scenes = new Map<string, Promise<S2Item[]>>();

/**
 * A year before the mosaics: for each MGRS tile under the box, its clearest scene of the year (fewest clouds, then
 * the least of the tile left empty by the swath's edge), and the next two as the fill.
 */
function sceneItems(year: number, bbox: number[]): Promise<S2Item[]> {
  const k = `${year}:${bbox.map((v) => v.toFixed(3)).join(",")}`;
  let p = scenes.get(k);
  if (!p) {
    p = fetch(EARTH_SEARCH, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        collections: ["sentinel-2-l2a"],
        bbox,
        datetime: `${year}-01-01T00:00:00Z/${year}-12-31T23:59:59Z`,
        query: { "eo:cloud_cover": { lt: 20 } },
        sortby: [{ field: "properties.eo:cloud_cover", direction: "asc" }],
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
        const rank = (f: EsFeature) => Number(f.properties["eo:cloud_cover"] ?? 100) + 0.5 * Number(f.properties["s2:nodata_pixel_percentage"] ?? 50);
        const yearly: S2Item[] = [];
        const fill: S2Item[] = [];
        for (const l of byTile.values()) {
          l.sort((a, b) => rank(a) - rank(b));
          l.slice(0, 3).forEach((f, i) => {
            const bands = Object.fromEntries(Object.entries(ES_BANDS).filter(([a]) => f.assets[a]).map(([a, b]) => [b, f.assets[a].href]));
            const it: S2Item = { id: i ? `${f.id}#fill` : f.id, url: f.assets.visual.href, bbox: f.bbox, bands };
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
  if (year < MOSAIC_FIRST) return sceneItems(year, [lon(x), lat(y + 1), lon(x + 1), lat(y)]);
  const features = await search([lon(x), lat(y + 1), lon(x + 1), lat(y)]);
  const span = `${year}-01-01_${year + 1}-01-01`;
  const yearly: S2Item[] = [];
  const fill: S2Item[] = [];
  for (const f of features) {
    if (!f.id.endsWith(span) || !f.assets.TCI || !f.bbox) continue;
    if (f.collection === FILL) fill.push({ id: `${f.id}#fill`, url: f.assets.TCI.href, bbox: f.bbox, fill: true, bands: bandsOf(f) });
    else yearly.push({ id: f.id, url: f.assets.TCI.href, bbox: f.bbox, bands: bandsOf(f) });
  }
  if (!yearly.length) {
    // The STAC lags the bucket (2025 is in the bucket for tiles the search does
    // not know yet): the same MGRS tile's path with the year swapped. A tile that
    // is not there reads as empty, not as an error
    const seen = new Set<string>();
    for (const f of features) {
      if (f.collection !== YEARLY || !f.assets.TCI) continue;
      const tile = f.id.split("_")[0];
      if (seen.has(tile)) continue;
      seen.add(tile);
      const id = `${tile}_${span}`;
      const base = f.assets.TCI.href.split("/").slice(0, -2).join("/");
      yearly.push({
        id,
        url: `${base}/${id}/TCI.tif`,
        bbox: f.bbox,
        bands: Object.fromEntries(BANDS.map((b) => [b, `${base}/${id}/${b}.tif`])),
      });
    }
  }
  return [...yearly, ...fill];
}
