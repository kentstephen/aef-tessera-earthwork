"""More ground that changes without anyone digging it, and ground that moves in ways nobody planned: lookalikes.

ground.py counted false alarms on burned forest, clearcuts and drawn-down reservoirs. This adds the other events
whose two years can look like a building site or a pit, each a 6 km square on AlphaEarth's grid with the years
either side of the event, an event group against ground to compare in the same square:
  slide   landslides: tree cover lost in the event year (Hansen GFC v1.12) where an earthquake or a storm set
          off thousands of slides (Hokkaido 2018, Haiti 2021, Romagna 2023) or one debris flow (Wayanad
          2024), against forest that never lost cover. Real moved ground: lidar would call most slide scars
          and their deposits moved
  river   a fast meandering or braided river: ground that was water in one year and land in the other (the
          bank cut away, a new bar), against land both years. The bank and bar moved; lidar flattens water,
          so it would read as moved too
  solar   a large solar farm built between the years (USPVDB polygons, its year of operation), against
          ground outside every array. Grading for the posts and roads moves some ground; panels are not
          ground
  fill    a lignite pit flooded on purpose (Cottbuser Ostsee) and a valley flooded by a new dam (GERD): land
          in the first year, water in the second, against land both years. Nothing dug between the years
  dredge  gravel pits on the lower Rhine: land turned to water between the years is a pit dredged wet, real
          digging under water
  flood   a river flood that came and went (Sindh 2022, Romagna 2023): ground seen under water in at least a
          quarter of the flood weeks' clear looks and land in both years, against land never seen flooded.
          Silt and scour, a few cm to dm
The water groups come from Sentinel-2 L2A (Element84 earth-search COGs, anonymous): each AlphaEarth pixel's
share of clear looks in a year with MNDWI above zero; water past 0.6, land under 0.4, between left out.

The same water mask splits the six reservoirs ground.py tested (kind lake). Hansen's water is the full lake,
so its "land" group held the bathtub ring: reservoirs() scores each lake square's pixels by what they were in
the two years (water both, land both, water to land, land to water).

Scored by both recipes' final models (data/models/*.pkl, read as kept) at each rule's 60% cutoff. Places are
saved as ground-look-<name> (PLACES_DIR and data/tessera) so compare.load reads them, outside ground.PLACES so
ground.py and fire_teach.py leave them alone; scores go to data/scores-look-<recipe>.

Run: uv run python lookalikes.py [name ... | lakes]   (results/lookalikes.json, results/reservoir_rings.json)
"""

import json
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import rasterio
import requests
import zarr
from obstore.store import HTTPStore
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.windows import Window
from zarr.storage import ObjectStore

import ground
from tessera_sites import AEF_RES, AEF_X0, AEF_Y0, OUT, PLACES_DIR, TESSERA, aef_unit, read_rc

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(HERE, "data", "lookalikes")
WATER_DIR = os.path.join(DIR, "water")
STAC = "https://earth-search.aws.element84.com/v1/search"
HALF = 300
WATER, LAND = 0.6, 0.4  # share of clear looks with water: past WATER is water, under LAND is land
SHOW = ("aef_gated", "tes", "mean", "both", "both_either", "either_radar")

# name: (kind, lon, lat, years). slide: (the event's year, the Hansen loss years that hold it), the square searched
# within SEARCH degrees for the one nearest half that loss; the rest: (before, after), the square on the point
PLACES = {
    "iburi-jp": ("slide", 141.98, 42.76, (2018, (2018, 2019))),  # Hokkaido Eastern Iburi earthquake, September 2018 (Hansen dates most of it 2019)
    "haiti-nippes": ("slide", -73.95, 18.40, (2021, (2021, 2022))),  # Nippes earthquake, August 2021, around Pic Macaya
    "romagna-apennines-it": ("slide", 11.80, 44.12, (2023, (2023, 2024))),  # May 2023 rains, tens of thousands of slides
    "wayanad-in": ("slide", 76.14, 11.49, (2024, (2024,))),  # one debris flow, July 2024 (no search: the square is the flow)
    "ucayali-pe": ("river", -74.55, -8.45, (2018, 2021)),
    "brahmaputra-in": ("river", 94.15, 26.88, (2018, 2021)),
    "jamuna-bd": ("river", 89.68, 24.45, (2018, 2021)),
    "kosi-in": ("river", 86.90, 26.30, (2018, 2021)),
    "cottbuser-ostsee-de": ("fill", 14.42, 51.79, (2018, 2024)),  # flooding began April 2019
    "gerd-et": ("fill", 35.15, 11.15, (2019, 2023)),  # filling 2020 to 2023
    "rhine-gravel-de": ("dredge", 6.47, 51.70, (2018, 2023)),
    "sindh-flood-pk": ("flood", 67.75, 26.45, (2021, 2023)),
    "romagna-flood-it": ("flood", 11.83, 44.52, (2022, 2024)),  # Conselice and Lavezzola, under water for weeks
}
# TESSERA has no Hokkaido pixels in 2017 (under 15 clear Sentinel-2 looks): the year before is the event's own,
# whose September slides the calendar-year embedding holds only weakly
PAIR = {"iburi-jp": (2018, 2020)}
SEARCH = {"iburi-jp": 0.08, "haiti-nippes": 0.15, "romagna-apennines-it": 0.12, "wayanad-in": 0.0}
FLOOD_WINDOW = {"sindh-flood-pk": ("2022-08-25", "2022-11-15"), "romagna-flood-it": ("2023-05-17", "2023-06-15")}
# the largest arrays built 2020 to 2023 across ground types: (USPVDB name, its year), the pair (year - 2, year + 1)
SOLAR = {
    "aktina-tx": "Aktina Solar", "impact-tx": "Impact Solar 1", "arrow-canyon-nv": "Arrow Canyon Solar Hybrid",
    "fox-squirrel-oh": "Fox Squirrel Solar", "prospero-tx": "Prospero Solar", "sonoran-az": "Sonoran Solar Energy",
    "slate-ca": "Slate Hybrid", "sun-mountain-co": "Sun Mountain Solar 1",
}
USPVDB = os.path.join(DIR, "uspvdb_v4_0_20260414.geojson")


def grid(rc):
    """Lon and lat of each AlphaEarth pixel center over mosaic rows and columns rc, flattened."""
    y0, y1, x0, x1 = rc
    LON, LAT = np.meshgrid(AEF_X0 + (np.arange(x0, x1) + 0.5) * AEF_RES, AEF_Y0 - (np.arange(y0, y1) + 0.5) * AEF_RES)
    return LON.ravel(), LAT.ravel()


# ---- the water mask: Sentinel-2 L2A looks per pixel -----------------------------------------------------------

def scenes(bbox, start, end, cloud=50, per_month=3, most=None):
    """Sentinel-2 L2A items over the box: per calendar month the per_month least cloudy, one per date and tile."""
    body = {"collections": ["sentinel-2-l2a"], "bbox": list(bbox), "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
            "query": {"eo:cloud_cover": {"lt": cloud}}, "limit": 200}
    items = []
    while True:
        for t in range(4):
            try:
                j = requests.post(STAC, json=body, timeout=90).json()
                break
            except Exception:
                time.sleep(3 * (t + 1))
        items += j.get("features", [])
        nxt = [l for l in j.get("links", []) if l.get("rel") == "next"]
        if not nxt or not j.get("features"):
            break
        body = nxt[0].get("body", body)
    seen, out = set(), {}
    for it in sorted(items, key=lambda i: i["properties"]["eo:cloud_cover"]):
        p = it["properties"]
        key = (p["datetime"][:10], p["grid:code"])
        if key in seen:
            continue
        seen.add(key)
        out.setdefault(p["datetime"][:7], []).append(it)
    picked = [it for m in sorted(out) for it in out[m][:per_month]]
    return picked[:most] if most else picked


def look(it, lon, lat):
    """One item at the pixel centers: (clear, water) booleans. Clear is SCL vegetation, bare, water or dark; water
    is MNDWI above zero (green against SWIR 1.6 um, each band's own scale and offset)."""
    epsg = it["properties"]["proj:epsg"]
    x, y = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
    got = {}
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY=4, GDAL_HTTP_RETRY_DELAY=2):
        for k in ("green", "swir16", "scl"):
            a = it["assets"][k]
            with rasterio.open(a["href"]) as s:
                tr = s.transform
                c0, c1 = int((x.min() - tr.c) / tr.a) - 2, int((x.max() - tr.c) / tr.a) + 3
                r0, r1 = int((y.max() - tr.f) / tr.e) - 2, int((y.min() - tr.f) / tr.e) + 3
                v = s.read(1, window=Window(c0, r0, c1 - c0, r1 - r0), boundless=True, fill_value=0)
                rr = np.clip(((y - tr.f) / tr.e).astype(int) - r0, 0, v.shape[0] - 1)
                cc = np.clip(((x - tr.c) / tr.a).astype(int) - c0, 0, v.shape[1] - 1)
                dn = v[rr, cc]
                rb = (a.get("raster:bands") or [{}])[0]
                got[k] = (dn, dn * rb.get("scale", 1.0) + rb.get("offset", 0.0))
    g, sw = got["green"][1], got["swir16"][1]
    clear = np.isin(got["scl"][0], (2, 4, 5, 6, 7)) & (got["green"][0] > 0) & (got["swir16"][0] > 0)
    water = (g - sw) / np.maximum(g + sw, 1e-4) > 0
    return clear, water & clear


def water_share(lon, lat, start, end, **kw):
    """Per pixel: the share of clear looks that saw water between start and end, and the number of clear looks."""
    bbox = (lon.min(), lat.min(), lon.max(), lat.max())
    items = scenes(bbox, start, end, **kw)
    nc, nw = np.zeros(len(lon), np.int16), np.zeros(len(lon), np.int16)

    def one(it):
        try:
            return look(it, lon, lat)
        except Exception as e:  # a missing or broken COG: one look fewer
            print(f"    skipped {it['id']}: {type(e).__name__}")
            return None
    with ThreadPoolExecutor(12) as ex:
        for got in ex.map(one, items):
            if got is not None:
                nc += got[0]
                nw += got[1]
    return np.where(nc > 0, nw / np.maximum(nc, 1), np.nan).astype(np.float32), nc, len(items)


def water_years(name, rc, years):
    """The water share of each year (and of a flood's weeks), cached in data/lookalikes/water/<name>.npz."""
    path = f"{WATER_DIR}/{name}.npz"
    have = dict(np.load(path)) if os.path.exists(path) else {}
    lon, lat = grid(rc)
    for y in years:
        if f"w{y}" not in have:
            t = time.time()
            have[f"w{y}"], have[f"n{y}"], k = water_share(lon, lat, f"{y}-01-01", f"{y}-12-31")
            print(f"  water {y}: {k} looks, median clear {np.median(have[f'n{y}']):.0f}, {time.time() - t:.0f} s", flush=True)
    if name in FLOOD_WINDOW and "wflood" not in have:
        a, b = FLOOD_WINDOW[name]
        have["wflood"], have["nflood"], k = water_share(lon, lat, a, b, cloud=90, per_month=40)
        print(f"  water in the flood: {k} looks, median clear {np.median(have['nflood']):.0f}", flush=True)
    os.makedirs(WATER_DIR, exist_ok=True)
    np.savez_compressed(path, **have)
    return have


def is_water(w):
    return w > WATER


def is_land(w):
    return w < LAND


# ---- the places ---------------------------------------------------------------------------------------------

def solar_place(name):
    """A solar farm's square: centered on the USPVDB array, its year, and every array's polygon with its year."""
    fs = json.load(open(USPVDB))["features"]
    f = next(f for f in fs if f["properties"]["p_name"] == SOLAR[name])
    p = f["properties"]
    return p["xlong"], p["ylat"], (p["p_year"] - 2, p["p_year"] + 1), fs


def square(lon, lat):
    r, c = int((AEF_Y0 - lat) / AEF_RES), int((lon - AEF_X0) / AEF_RES)
    return (r - HALF, r + HALF, c - HALF, c + HALF)


def solar_groups(rc, pair, fs):
    """1 an array built after the first year and by the second, 2 ground at least 50 m from any array."""
    y0, y1, x0, x1 = rc
    tr = from_origin(AEF_X0 + x0 * AEF_RES, AEF_Y0 - y0 * AEF_RES, AEF_RES, AEF_RES)
    shape = (y1 - y0, x1 - x0)
    new = [f["geometry"] for f in fs if f["properties"].get("p_year") and pair[0] < f["properties"]["p_year"] <= pair[1]]
    old = [f["geometry"] for f in fs]
    built = rasterize(new, shape, transform=tr, fill=0, default_value=1, dtype="uint8") if new else np.zeros(shape, np.uint8)
    anyp = rasterize(old, shape, transform=tr, fill=0, default_value=1, dtype="uint8")
    from scipy.ndimage import binary_dilation
    near = binary_dilation(anyp > 0, iterations=5)
    g = np.zeros(shape, np.int8)
    g[~near] = 2
    g[built > 0] = 1
    return g.ravel()


def slide_pick(lon, lat, loss_years, search):
    """The square within search degrees whose tree cover lost in the loss years is nearest half (ground.pick's rule
    over more than one Hansen year: a September slide is often dated the next year, once Landsat sees it clear)."""
    from scipy.ndimage import uniform_filter
    a, tr = ground.hansen("lossyear", lon - search, lat - search, lon + search, lat + search)
    f = uniform_filter(np.isin(a, [y - 2000 for y in loss_years]).astype(np.float32), int(HALF * 2 * AEF_RES / 0.00025), mode="constant")
    r, c = np.unravel_index(np.argmin(np.abs(f - 0.5)), f.shape)
    print(f"  loss {loss_years}: {f[r, c]:.0%} of the square")
    x, y = tr * (c + 0.5, r + 0.5)
    return float(x), float(y)


def slide_groups(rc, loss_years):
    """1 tree cover lost in the loss years, 2 forest (cover 50% and up in 2000) that never lost cover."""
    lon, lat = grid(rc)
    box = (lon.min() - 0.001, lat.min() - 0.001, lon.max() + 0.001, lat.max() + 0.001)

    def at(a, tr):
        rr = np.clip(((lat - tr.f) / tr.e).astype(int), 0, a.shape[0] - 1)
        cc = np.clip(((lon - tr.c) / tr.a).astype(int), 0, a.shape[1] - 1)
        return a[rr, cc]
    loss, cover = at(*ground.hansen("lossyear", *box)), at(*ground.hansen("treecover2000", *box))
    g = np.zeros(len(lon), np.int8)
    g[(loss == 0) & (cover >= 50)] = 2
    g[np.isin(loss, [y - 2000 for y in loss_years])] = 1
    return g


def water_groups(name, kind, rc, pair):
    """1 the event's ground, 2 land in both years (and, for a flood, never seen under water)."""
    w = water_years(name, rc, pair)
    b, a = w[f"w{pair[0]}"], w[f"w{pair[1]}"]
    land = is_land(b) & is_land(a)
    g = np.zeros(len(b), np.int8)
    if kind == "river":
        ev = (is_water(b) & is_land(a)) | (is_land(b) & is_water(a))
    elif kind in ("fill", "dredge"):
        ev = is_land(b) & is_water(a)
    else:  # flood: under water in the flood's weeks, land in both years
        f, nf = w["wflood"], w["nflood"]
        ev = land & (nf >= 2) & (f >= 0.25)
        land = land & (nf >= 2) & (f == 0)
    g[land] = 2
    g[ev] = 1
    return g


def read(g, name):
    path = f"{PLACES_DIR}/ground-look-{name}.npz"
    if os.path.exists(path) and os.path.exists(f"{OUT}/ground-look-{name}.npz"):
        return
    print(name, flush=True)
    if name in SOLAR:
        kind = "solar"
        lon, lat, pair, fs = solar_place(name)
        rc = square(lon, lat)
        grp = solar_groups(rc, pair, fs)
    else:
        kind, lon, lat, years = PLACES[name]
        if kind == "slide":
            event, loss_years = years
            if SEARCH[name]:
                lon, lat = slide_pick(lon, lat, loss_years, SEARCH[name])
            pair = PAIR.get(name, (event - 1, event + 1))
            rc = square(lon, lat)
            grp = slide_groups(rc, loss_years)
        else:
            pair = years
            if kind == "river":  # centered where Hansen's water (the channel) is nearest half the square
                ground.SEARCH = 0.15
                lon, lat, _ = ground.pick("lake", lon, lat, years)
            rc = square(lon, lat)
            grp = water_groups(name, kind, rc, pair)
    (b, okb), (a, oka) = aef_unit(pair[0], rc), aef_unit(pair[1], rc)
    meta = {"name": f"ground-look-{name}", "kind": kind, "aef": list(pair), "rc": list(rc), "shape": [2 * HALF, 2 * HALF],
            "point": [float(lon), float(lat)]}
    os.makedirs(PLACES_DIR, exist_ok=True)
    np.savez_compressed(path, X=np.concatenate([b, a], 1).astype(np.float16), grp=grp,
                        y=np.where(okb & oka & (grp > 0), 0, -1).astype(np.int8), meta=json.dumps(meta))
    got = read_rc(g, f"ground-look-{name}", rc, pair)
    zone = got.pop("zone")
    np.savez_compressed(f"{OUT}/ground-look-{name}.npz", meta=json.dumps({**meta, "zone": zone}), **got)
    print(f"  {pair}, groups {np.bincount(grp, minlength=3)[1:]} px", flush=True)


# ---- scoring --------------------------------------------------------------------------------------------------

def recipes():
    """Both recipes' final models as kept, each rule's 60% cutoff (recomputed in memory if a rule is new)."""
    import models
    out = {}
    for r in ("us", "abroad"):
        m = pickle.load(open(f"{models.DIR}/{r}.pkl", "rb"))
        if not set(SHOW) <= set(m["cutoff"][0.6]):
            m["cutoff"] = models.cutoffs(models.SCORE_DIRS[r])
        out[r] = m
    return out


def rule_scores(m, d, li):
    from aggregate import base_scores
    from quick import RULES
    s = base_scores(m["models"], d, li)
    return s, {w: RULES[w](s, m["bounds"]) for w in SHOW}


def score(names, rec):
    import aggregate
    from compare import labeled, load
    res = {}
    print(f"\nshare lit at each recipe's 60% cutoff: event ground | ground to compare")
    print(f"  {'place':22s} {'kind':6s} {'years':12s} recipe      " + "  ".join(f"{w:>12s}" for w in SHOW))
    for n in names:
        d = load(f"ground-look-{n}")
        grp = np.load(f"{PLACES_DIR}/ground-look-{n}.npz")["grp"]
        li = labeled(d)
        if len(li) < 50:
            print(f"  {n}: {len(li)} pixels valid in both embeddings and years, not scored")
            continue
        res[n] = {"kind": d["kind"], "years": d["years"], "px": [int((grp[li] == k).sum()) for k in (1, 2)],
                  "radar": [float(np.median(d["s1b"][li])), float(np.median(d["s1a"][li]))],
                  "flip": round(float(d["radar_flip"][li].mean()), 3)}
        for r, m in rec.items():
            s, S = rule_scores(m, d, li)
            aggregate.SCORES = os.path.join(HERE, "data", f"scores-look-{r}")
            aggregate.save_scores(f"ground-look-{n}", d["kind"], s, np.where(grp[li] == 1, 0, -1).astype(np.int8), "test", m["bounds"])
            cut = m["cutoff"][0.6]
            res[n][r] = {k: {w: round(float((S[w][grp[li] == g_] > cut[w]).mean()), 3) if (grp[li] == g_).any() else None
                             for w in SHOW} for k, g_ in (("event", 1), ("compare", 2))}
            for k in ("event", "compare"):
                tag = f"{n:22s} {d['kind']:6s} {str(d['years']):12s}" if (r, k) == ("us", "event") else " " * 42
                vals = "  ".join(f"{v:12.3f}" if v is not None else f"{'':>12s}" for v in res[n][r][k].values())
                extra = f"  px {res[n]['px']}  radar {res[n]['radar'][0]:.0f}/{res[n]['radar'][1]:.0f}" if (r, k) == ("us", "event") else ""
                print(f"  {tag} {r:6s} {k[:3]}  {vals}{extra}", flush=True)
        del d
    return res


def reservoirs(rec):
    """Each ground.py reservoir split by the water mask: water both years, land both, water to land, land to water."""
    from compare import labeled, load
    parts = ("water both", "land both", "water to land", "land to water")
    res = {}
    print(f"\nreservoirs: share lit at each recipe's 60% cutoff, by what the pixel was in the two years (pixels)")
    for n, (kind, *_rest) in ground.PLACES.items():
        if kind != "lake" or not os.path.exists(f"{OUT}/ground-{n}.npz"):
            continue
        d = load(f"ground-{n}")
        meta = json.loads(str(np.load(f"{PLACES_DIR}/ground-{n}.npz", allow_pickle=True)["meta"]))
        pair = meta["aef"]
        w = water_years(f"lake-{n}", tuple(meta["rc"]), pair)
        b, a = w[f"w{pair[0]}"], w[f"w{pair[1]}"]
        part = np.full(len(b), -1)
        part[is_water(b) & is_water(a)] = 0
        part[is_land(b) & is_land(a)] = 1
        part[is_water(b) & is_land(a)] = 2
        part[is_land(b) & is_water(a)] = 3
        hgrp = np.load(f"{PLACES_DIR}/ground-{n}.npz")["grp"]
        li = labeled(d)
        res[n] = {"years": pair, "px": {p: int((part[li] == i).sum()) for i, p in enumerate(parts)},
                  "hansen water that was water to land": round(float((part[hgrp == 1] == 2).mean()), 3),
                  "hansen land that was water to land": round(float((part[hgrp == 2] == 2).mean()), 3)}
        for r, m in rec.items():
            _, S = rule_scores(m, d, li)
            cut = m["cutoff"][0.6]
            res[n][r] = {p: {wn: round(float((S[wn][part[li] == i] > cut[wn]).mean()), 3) if (part[li] == i).sum() >= 50 else None
                             for wn in SHOW} for i, p in enumerate(parts)}
        print(f"  {n} {pair}: " + ", ".join(f"{p} {res[n]['px'][p]}" for p in parts))
        for r in rec:
            for p in parts:
                v = res[n][r][p]
                print(f"    {r:6s} {p:14s} " + "  ".join(f"{wn} {v[wn]:.3f}" if v[wn] is not None else f"{wn}   -  " for wn in SHOW))
        del d
    return res


def main():
    args = sys.argv[1:]
    rec = recipes()
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    if args in ([], ["lakes"]):
        r = reservoirs(rec)
        json.dump(r, open(os.path.join(HERE, "results", "reservoir_rings.json"), "w"), indent=1)
        if args:
            return
    names = [a for a in args if a != "lakes"] or list(PLACES) + list(SOLAR)
    zarr.config.set({"async.concurrency": 128})
    g = ObjectStore(HTTPStore.from_url(TESSERA), read_only=True)
    for n in names:
        t = time.time()
        try:
            read(g, n)
        except Exception as e:  # a place outside a source's years or reach: say so and go on
            print(f"  {n}: not read ({type(e).__name__}: {e})", flush=True)
            continue
        print(f"  {n}: {time.time() - t:.0f} s", flush=True)
    res = score([n for n in names if os.path.exists(f"{OUT}/ground-look-{n}.npz")], rec)
    path = os.path.join(HERE, "results", "lookalikes.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    json.dump({**old, **res}, open(path, "w"), indent=1)
    for kind in ("slide", "river", "solar", "fill", "dredge", "flood"):
        ks = [n for n in res if res[n]["kind"] == kind]
        if not ks:
            continue
        print(f"\n  mean over {len(ks)} {kind} places")
        for r in rec:
            for k in ("event", "compare"):
                vals = {w: [res[n][r][k][w] for n in ks if res[n][r][k][w] is not None] for w in SHOW}
                print(f"  {kind:6s} {r:6s} {k:7s} " + "  ".join(f"{w} {np.mean(v):.3f}" for w, v in vals.items() if v))


if __name__ == "__main__":
    main()
