/**
 * Sentinel-2 footprints for the map, found in the page: Earth Genome's STAC
 * search, the yearly mosaics and the fill mosaics (cloud gaps) under them. The
 * map reads and draws each footprint's TCI COG itself (ui.js, s2CogLayers).
 */
const STAC = "https://stac.earthgenome.org/search";
const YEARLY = "sentinel2-yearly-mosaics";
const FILL = "sentinel2-temporal-mosaics";
const BANDS = ["B02", "B03", "B04", "B8A", "B11", "B12"];

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

/** The year's footprints under one web mercator tile: yearly first, the fill after them. */
export async function s2Items(year: number, { z, x, y }: { z: number; x: number; y: number }): Promise<S2Item[]> {
  const n = 2 ** z;
  const lon = (i: number) => (i / n) * 360 - 180;
  const lat = (j: number) => (Math.atan(Math.sinh(Math.PI * (1 - (2 * j) / n))) * 180) / Math.PI;
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
