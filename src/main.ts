import "./ui.css";
import "./page.css";

import { createEarthwork } from "./earthwork";
import { s2Items } from "./s2";
// @ts-expect-error ui.js is the notebook widget's JavaScript, carried over as is
import { render } from "./ui.js";

// Overture's divisions (place names under a click), from the release pinned here; Overture keeps only
// its newest releases, so when the pinned one fails the newest in the bucket is taken instead
const OVERTURE = "https://overturemaps-extras-us-west-2.s3.us-west-2.amazonaws.com";
const OVERTURE_RELEASE = "2026-09-23.1";

async function divisionsUrl(): Promise<string | null> {
  const url = (release: string) => `${OVERTURE}/tiles/${release}/divisions.pmtiles`;
  const ok = async (u: string) => {
    try {
      const r = await fetch(u, { headers: { Range: "bytes=0-6" } });
      return r.ok;
    } catch {
      return false;
    }
  };
  if (await ok(url(OVERTURE_RELEASE))) return url(OVERTURE_RELEASE);
  try {
    const xml = await (await fetch(`${OVERTURE}/?list-type=2&prefix=tiles/&delimiter=/`)).text();
    const releases = [...xml.matchAll(/<Prefix>tiles\/([^/<]+)\/<\/Prefix>/g)].map((m) => m[1]).sort();
    const newest = releases[releases.length - 1];
    if (newest && (await ok(url(newest)))) return url(newest);
  } catch {
    // no place names: the card still has the spot's coordinates and score
  }
  return null;
}

// viridis, 9 stops, for the key's ramp (the map's own comes from deck.gl-raster's colormap sprite)
const VIRIDIS = "440154472d7b3b528b2c728e21918c28ae805ec9626dbb2ab5de2bfde725";

// one line; the About page (?) has them in full
const CREDITS = [
  "AlphaEarth &copy; Google, Google DeepMind",
  'Sentinel-2 &copy; Earth Genome (<a href="https://creativecommons.org/licenses/by/4.0/" target="_blank" rel="noopener">CC BY 4.0</a>), via <a href="https://source.coop" target="_blank" rel="noopener">Source Cooperative</a>',
  "Places: Overture Maps (ODbL)",
  "Search: Photon",
  '&copy; <a href="https://carto.com/attributions" target="_blank" rel="noopener">CARTO</a> &copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap</a>',
].join('<span class="sep"></span>');

const base = import.meta.env.BASE_URL;
const [earthwork, divPm] = await Promise.all([
  createEarthwork({ current: `${base}models/current.json`, abroad: `${base}models/abroad.json` }, () => "watername_ocean"),
  divisionsUrl(),
]);

const cfg = {
  // Wildwood, Florida: the starting area, where The Villages grades new neighborhoods out of farmland
  home: { longitude: -81.989804, latitude: 28.751057, zoom: 12.7 },
  fit: true,
  ew_zoom: 12,
  ew_models: ["current", "abroad"],
  // US only: the model taught on US lidar alone, for a start inside the US
  model: "current",
  aef_from: 2023,
  aef_to: 2025,
  aef_years: [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
  s2_years: [2022, 2023, 2024, 2025],
  s2_min_z: 7,
  viridis: VIRIDIS,
  div_pm: divPm,
};

// the notebook's widget model, standing in for the kernel: the config, and nothing back
const model = {
  get: (k: string) => (k === "config" ? JSON.stringify(cfg) : k === "status" ? "" : null),
  set: () => {},
  save_changes: () => {},
  on: () => {},
  send: () => {},
};

const el = document.getElementById("app")!;
earthwork.whenReady(() => window.dispatchEvent(new Event("earthwork-ready")));
render({ model, el, earthwork, s2Items, credits: CREDITS });
