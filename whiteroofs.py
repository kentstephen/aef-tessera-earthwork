"""White roofs: does a dark roof painted white read as earthwork?

New York has coated roofs white for years (NYC CoolRoofs since 2010; since 2020 Local Law 94 asks for a cool,
green or solar roof whenever a roof is replaced). Nothing moves, but at 10 m a flat roof going from dark to white is
one of the largest brightness changes a pixel can make, and bright new ground is much of what the model lights. The
lidar that taught the model never saw a roof repainted. NYC publishes no dated list of coated roofs (only a 2012
point file, before AlphaEarth's first year), so the roofs that turned white are found from imagery:

  roofs        10 m pixels mostly on one roof: the center and four points 70% of the way to the corners inside one
               NYC building footprint (NYC Open Data 5zhs-2jue) built before 2015, so it stood through every year
               compared (row houses narrower than a pixel drop out)
  whitened     the roof's visible brightness (mean of Sentinel-2 B02, B03, B04, Earth Genome's yearly mosaics,
               2022 to 2024) rose by at least WHITE_RISE, from under DARK_BEFORE to over BRIGHT_AFTER
  darkened     the same the other way (bright to dark), for contrast
  steady       the roof's visible brightness changed by less than STEADY
Each NYC square, both embeddings, the pairs 2022-2023, 2023-2024 and 2022-2024 (TESSERA's radar has the same look
directions in these years at New York: ascending only), scored by the abroad recipe's final models at each rule's
60% cutoff. The share of whitened and of steady roof pixels each layer lights.

Run: uv run python whiteroofs.py   (squares read into data/cities, footprints and imagery into data/whiteroofs,
     writes results/whiteroofs.json)
"""

import json
import os
import time
import urllib.parse
import urllib.request

import numpy as np
import rasterio
import zarr
from obstore.store import HTTPStore
from pyproj import Transformer
from rasterio.windows import from_bounds
from shapely import contains_xy, points
from shapely.geometry import shape
from shapely.strtree import STRtree
from zarr.storage import ObjectStore

import cities
from tessera_sites import AEF_RES, AEF_X0, AEF_Y0, TESSERA

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(HERE, "data", "whiteroofs")
OUT = os.path.join(HERE, "results", "whiteroofs.json")
SQUARES = {  # the three New York squares cities.py reads, and four with more big flat roofs
    "nyc-lower-east-side": None, "nyc-midtown": None, "brooklyn-park-slope": None,
    "brooklyn-sunset-park": (-74.012, 40.655), "bronx-hunts-point": (-73.885, 40.812),
    "queens-long-island-city": (-73.940, 40.745), "brooklyn-east-new-york": (-73.885, 40.668),
}
S2_YEARS = (2022, 2023, 2024)
PAIRS = ((2022, 2023), (2023, 2024), (2022, 2024))
BANDS = ("B02", "B03", "B04")
WHITE_RISE, DARK_BEFORE, BRIGHT_AFTER, STEADY = 0.06, 0.16, 0.20, 0.015
FOOTPRINTS = "https://data.cityofnewyork.us/resource/5zhs-2jue.geojson"
STAC = "https://stac.earthgenome.org/search"


def grid(z):
    """Lon, lat of every AlphaEarth pixel center in the square, flattened as cities.py's arrays are."""
    r0, r1, c0, c1 = (int(v) for v in z["rc"])
    lon, lat = np.meshgrid(AEF_X0 + (np.arange(c0, c1) + 0.5) * AEF_RES, AEF_Y0 - (np.arange(r0, r1) + 0.5) * AEF_RES)
    return lon.ravel(), lat.ravel()


def roofs(name, lon, lat):
    """For each pixel, the index of the old building whose footprint holds its center and four inner corners, else -1."""
    path = f"{DIR}/{name}_footprints.geojson"
    W, E, S, N = lon.min() - AEF_RES, lon.max() + AEF_RES, lat.min() - AEF_RES, lat.max() + AEF_RES
    if not os.path.exists(path):
        q = {"$where": f"within_box(the_geom,{N},{W},{S},{E}) AND construction_year < 2015 AND construction_year > 1700",
             "$select": "the_geom,construction_year,height_roof,bin", "$limit": "100000"}
        os.makedirs(DIR, exist_ok=True)
        urllib.request.urlretrieve(f"{FOOTPRINTS}?{urllib.parse.urlencode(q)}", path)
    feats = json.load(open(path))["features"]
    geoms = [shape(f["geometry"]) for f in feats]
    tree = STRtree(geoms)
    h = 0.35 * AEF_RES
    pts = [(lon, lat), (lon - h, lat - h), (lon - h, lat + h), (lon + h, lat - h), (lon + h, lat + h)]
    out = np.full(len(lon), -1, np.int64)
    pi, gi = tree.query(points(lon, lat), predicate="within")
    for p, g in zip(pi, gi):
        x = [px[p] for px, _ in pts]
        y = [py[p] for _, py in pts]
        if contains_xy(geoms[g], x, y).all():
            out[p] = g
    return out, len(geoms)


def visible(name, lon, lat):
    """Mean visible reflectance (B02, B03, B04) per pixel per year from Earth Genome's yearly mosaics; NaN where none."""
    path = f"{DIR}/{name}_s2.npz"
    if os.path.exists(path):
        return dict(np.load(path))
    W, E, S, N = lon.min(), lon.max(), lat.min(), lat.max()
    body = json.dumps({"collections": ["sentinel2-yearly-mosaics"], "bbox": [W, S, E, N], "limit": 50}).encode()
    items = json.load(urllib.request.urlopen(urllib.request.Request(STAC, body, {"Content-Type": "application/json"})))["features"]
    out = {}
    for y in S2_YEARS:
        acc = np.full((len(BANDS), len(lon)), np.nan, np.float32)
        for it in items:
            if not it["id"].endswith(f"{y}-01-01_{y + 1}-01-01"):
                continue
            for k, b in enumerate(BANDS):
                with rasterio.open(it["assets"][b]["href"]) as s:
                    x, yy = Transformer.from_crs(4326, s.crs, always_xy=True).transform(lon, lat)
                    win = from_bounds(min(x) - 20, min(yy) - 20, max(x) + 20, max(yy) + 20, s.transform).round_offsets().round_lengths()
                    a = s.read(1, window=win, boundless=True, fill_value=0).astype(np.float32)
                    wt = s.window_transform(win)
                    col = np.floor((np.asarray(x) - wt.c) / wt.a).astype(int)
                    row = np.floor((np.asarray(yy) - wt.f) / wt.e).astype(int)
                    inside = (row >= 0) & (row < a.shape[0]) & (col >= 0) & (col < a.shape[1])
                    v = np.where(inside, a[np.clip(row, 0, a.shape[0] - 1), np.clip(col, 0, a.shape[1] - 1)], 0)
                    acc[k] = np.where(np.isnan(acc[k]) & (v > 0), v / 1e4, acc[k])
        out[str(y)] = acc.mean(0)
    np.savez_compressed(path, **out)
    return out


def main():
    from aggregate import base_scores
    from models import final
    from quick import RULES
    zarr.config.set({"async.concurrency": 128})
    g = ObjectStore(HTTPStore.from_url(TESSERA), read_only=True)
    for n, ll in SQUARES.items():
        if ll is not None:
            cities.CITIES[n] = ll
        t = time.time()
        cities.read(g, n)
        print(f"read {n}: {time.time() - t:.0f} s", flush=True)
    rec = final("abroad")
    res = {}
    for n in SQUARES:
        z = dict(np.load(f"{cities.DIR}/{n}.npz"))
        lon, lat = grid(z)
        roof, nb = roofs(n, lon, lat)
        vis = visible(n, lon, lat)
        res[n] = {"buildings": nb, "roof_px": int((roof >= 0).sum()), "pairs": {}}
        for yb, ya in PAIRS:
            d, li, group = cities.pair(z, yb, ya)
            vb, va = vis[str(yb)][li], vis[str(ya)][li]
            on = roof[li] >= 0
            rise = va - vb
            white = on & (rise >= WHITE_RISE) & (vb < DARK_BEFORE) & (va > BRIGHT_AFTER)
            dark = on & (rise <= -WHITE_RISE) & (va < DARK_BEFORE) & (vb > BRIGHT_AFTER)
            steady = on & (np.abs(rise) < STEADY)
            s = base_scores(rec["models"], d, li)
            row = {"whitened_px": int(white.sum()), "whitened_buildings": int(len(np.unique(roof[li][white]))),
                   "darkened_px": int(dark.sum()), "steady_px": int(steady.sum()), "radar_same": float((group[li] == 0).mean()),
                   "aef_change": [float(np.median(d["aplain"][li][m])) if m.any() else None for m in (white, steady)],
                   "tes_change": [float(np.median(d["tplain"][li][m])) if m.any() else None for m in (white, steady)]}
            for w in cities.RULES_SHOWN:
                lit = RULES[w](s, rec["bounds"]) > rec["cutoff"][0.6][w]
                row[w] = {"whitened": float(lit[white].mean()) if white.any() else None,
                          "darkened": float(lit[dark].mean()) if dark.any() else None,
                          "steady": float(lit[steady].mean()) if steady.any() else None,
                          "all_roofs": float(lit[on].mean()), "off_roofs": float(lit[~on].mean())}
            res[n]["pairs"][f"{yb}-{ya}"] = row
            print(f"{n} {yb}-{ya}: whitened {row['whitened_px']} px on {row['whitened_buildings']} buildings, steady "
                  f"{row['steady_px']}; lit whitened/steady: "
                  + "  ".join(f"{w} {row[w]['whitened'] if row[w]['whitened'] is not None else float('nan'):.3f}/{row[w]['steady']:.3f}"
                              for w in cities.RULES_SHOWN), flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    report(res)


def report(res=None):
    res = res or json.load(open(OUT))
    print("\npooled over squares and pairs (pixel-weighted): share lit")
    print(f"  {'layer':14s} {'whitened roofs':>15s} {'darkened roofs':>15s} {'steady roofs':>13s} {'all roofs':>10s} {'off roofs':>10s}")
    for w in cities.RULES_SHOWN:
        tot = {k: [0.0, 0] for k in ("whitened", "darkened", "steady", "all_roofs", "off_roofs")}
        for n, v in res.items():
            for p, r in v["pairs"].items():
                for k, cnt in (("whitened", r["whitened_px"]), ("darkened", r["darkened_px"]), ("steady", r["steady_px"]), ("all_roofs", v["roof_px"]), ("off_roofs", None)):
                    if r[w][k] is not None and (cnt is None or cnt > 0):
                        c = cnt if cnt is not None else 1
                        tot[k][0] += r[w][k] * c
                        tot[k][1] += c
        print(f"  {w:14s} " + " ".join(f"{tot[k][0] / max(tot[k][1], 1):{wd}.3f}" for k, wd in (("whitened", 15), ("darkened", 15), ("steady", 13), ("all_roofs", 10), ("off_roofs", 10))))
    print("  whitened roof pixels per pair: " + ", ".join(f"{p} {sum(v['pairs'][p]['whitened_px'] for v in res.values())}" for p in [f"{a}-{b}" for a, b in PAIRS]))


if __name__ == "__main__":
    main()
