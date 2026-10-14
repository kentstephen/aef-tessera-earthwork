"""New ground nobody dug, to count false alarms on: burned forest, clearcuts, drawn-down reservoirs.

No lidar here. Each place is a 6 km square on AlphaEarth's grid, the years either side of the event, and a
grouping from Hansen's Global Forest Change (v1.12, 30 m):
  burn  large fires: tree cover lost in the fire year (forest burned to bare ground, the trap Grizzly Flats
        showed), against forest in the same square that never lost cover. The square is the 6 km in a box
        around the fire whose loss that year is closest to half, so both groups are there
  cut   commercial forestry: tree cover lost in the harvest year (clearcuts), against standing forest; the
        square chosen the same way
  lake  reservoirs drawn down between the two years: Hansen's water (the lake bed when full) against the
        land around it; the square where water is closest to half
Every pixel in a group counts as unchanged ground; skid trails, landings, dozer lines and salvage logging are
real but small. Scored by the final models of both recipes (models.py), each rule at its 60% cutoff.

Run: uv run python ground.py [name ...]   (reads what is missing into data/places and data/tessera, prints the
     share of each group lit, writes results/ground.json)
"""

import json
import os
import sys
import time

import numpy as np
import rasterio
import zarr
from obstore.store import HTTPStore
from rasterio.windows import from_bounds
from scipy.ndimage import uniform_filter
from zarr.storage import ObjectStore

from tessera_sites import AEF_RES, AEF_X0, AEF_Y0, OUT, PLACES_DIR, TESSERA, aef_unit, read_rc

HERE = os.path.dirname(os.path.abspath(__file__))
HANSEN = "/vsicurl/https://storage.googleapis.com/earthenginepartners-hansen/GFC-2024-v1.12/Hansen_GFC-2024-v1.12_{}_{}.tif"
HALF = 300  # AlphaEarth pixels either side of the center: 6 km at the equator (narrower east-west toward the poles)
SEARCH = 0.25  # degrees either side of the given point searched for the square with the most loss

# name: (kind, lon, lat, event years tried); the pair read is the year before and the year after the event
PLACES = {
    # US fires, mostly conifer forest
    "dixie-ca": ("burn", -121.0, 40.15, (2021,)), "creek-ca": ("burn", -119.25, 37.25, (2020,)),
    "castle-ca": ("burn", -118.5, 36.15, (2020,)), "east-troublesome-co": ("burn", -105.9, 40.25, (2020,)),
    "holiday-farm-or": ("burn", -122.45, 44.15, (2020,)), "beachie-creek-or": ("burn", -122.25, 44.8, (2020,)),
    "bootleg-or": ("burn", -121.25, 42.6, (2021,)), "hermits-peak-nm": ("burn", -105.4, 35.8, (2022,)),
    "cedar-creek-or": ("burn", -122.1, 43.75, (2022,)), "mosquito-ca": ("burn", -120.7, 39.0, (2022,)),
    # fires abroad
    "donnie-creek-bc": ("burn", -121.9, 57.3, (2023,)), "quebec-2023": ("burn", -75.0, 49.6, (2023,)),
    "kangaroo-island-au": ("burn", 136.75, -35.85, (2019, 2020)), "blue-mountains-au": ("burn", 150.3, -33.9, (2019, 2020)),
    "evia-gr": ("burn", 23.3, 38.9, (2021,)),
    # clearcuts in commercial forest
    "coast-range-or": ("cut", -123.6, 44.6, (2019,)), "willapa-wa": ("cut", -123.6, 46.5, (2021,)),
    "ware-ga": ("cut", -82.4, 31.2, (2019,)), "pine-ms": ("cut", -89.1, 31.3, (2021,)),
    "north-woods-me": ("cut", -69.5, 45.8, (2019,)), "norrland-se": ("cut", 16.5, 62.5, (2020,)),
    "kainuu-fi": ("cut", 27.0, 63.0, (2019,)), "kaingaroa-nz": ("cut", 176.6, -38.6, (2020,)),
    "tres-lagoas-br": ("cut", -51.9, -20.6, (2020,)), "biobio-cl": ("cut", -72.6, -37.5, (2019,)),
    # reservoirs drawn down (or refilled) between the two years: (kind, lon, lat, (before, after))
    "lake-mead-nv": ("lake", -114.4, 36.35, (2019, 2022)), "lake-oroville-ca": ("lake", -121.42, 39.62, (2021, 2023)),
    "folsom-ca": ("lake", -121.12, 38.74, (2021, 2023)), "lake-powell-ut": ("lake", -110.75, 37.35, (2019, 2022)),
    "great-salt-lake-ut": ("lake", -112.35, 41.15, (2020, 2022)), "sau-es": ("lake", 2.40, 41.98, (2021, 2023)),
}


def hansen(layer, lon0, lat0, lon1, lat1):
    """A Hansen layer over a lon/lat box (inside one 10 degree tile) and its transform."""
    top, left = int(np.ceil(lat1 / 10) * 10), int(np.floor(lon0 / 10) * 10)
    tile = f"{abs(top):02d}{'N' if top >= 0 else 'S'}_{abs(left):03d}{'E' if left >= 0 else 'W'}"
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"), rasterio.open(HANSEN.format(layer, tile)) as s:
        w = from_bounds(lon0, lat0, lon1, lat1, s.transform)
        return s.read(1, window=w), s.window_transform(w)


def pick(kind, lon, lat, years):
    """The square's center and year pair: the 6 km in the search box where the event's ground (tree cover lost
    in the event year, or Hansen's water for a lake) is closest to half the square, so the ground to compare
    is there too."""
    layer = "datamask" if kind == "lake" else "lossyear"
    a, tr = hansen(layer, lon - SEARCH, lat - SEARCH, lon + SEARCH, lat + SEARCH)
    best = None
    side = int(HALF * 2 * AEF_RES / 0.00025)  # the square's side in Hansen pixels (north-south)
    for y in ((None,) if kind == "lake" else years):
        hit = a == 2 if kind == "lake" else a == y - 2000
        f = uniform_filter(hit.astype(np.float32), side, mode="constant")
        r, c = np.unravel_index(np.argmin(np.abs(f - 0.5)), f.shape)
        if best is None or abs(f[r, c] - 0.5) < abs(best[0] - 0.5):
            best = (float(f[r, c]), r, c, y)
    share, r, c, y = best
    x, yy = tr * (c + 0.5, r + 0.5)
    print(f"  {'water' if kind == 'lake' else f'{y} loss'}: {share:.0%} of the square")
    return float(x), float(yy), (years if kind == "lake" else (y - 1, y + 1))


def groups(kind, rc, event):
    """Per AlphaEarth pixel: 1 the event's ground (loss that year, or the lake bed), 2 the ground to compare
    (forest that never lost cover, or land by the lake), 0 neither."""
    y0, y1, x0, x1 = rc
    lat = AEF_Y0 - (np.arange(y0, y1) + 0.5) * AEF_RES
    lon = AEF_X0 + (np.arange(x0, x1) + 0.5) * AEF_RES
    box = (lon[0] - 0.001, lat[-1] - 0.001, lon[-1] + 0.001, lat[0] + 0.001)

    def at(a, tr):  # the Hansen pixel under each AlphaEarth pixel center
        rr = np.clip(((lat - tr.f) / tr.e).astype(int), 0, a.shape[0] - 1)
        cc = np.clip(((lon - tr.c) / tr.a).astype(int), 0, a.shape[1] - 1)
        return a[rr][:, cc]
    g = np.zeros((y1 - y0, x1 - x0), np.int8)
    if kind == "lake":
        m = at(*hansen("datamask", *box))
        g[m == 2], g[m == 1] = 1, 2
    else:
        loss, cover = at(*hansen("lossyear", *box)), at(*hansen("treecover2000", *box))
        g[loss == event - 2000] = 1
        g[(loss == 0) & (cover >= 50)] = 2
    return g.ravel()


def read(g, name):
    path = f"{PLACES_DIR}/ground-{name}.npz"
    if os.path.exists(path) and os.path.exists(f"{OUT}/ground-{name}.npz"):
        return
    kind, lon, lat, years = PLACES[name]
    print(name)
    lon, lat, pair = pick(kind, lon, lat, years)
    r, c = int((AEF_Y0 - lat) / AEF_RES), int((lon - AEF_X0) / AEF_RES)
    rc = (r - HALF, r + HALF, c - HALF, c + HALF)
    (b, okb), (a, oka) = aef_unit(pair[0], rc), aef_unit(pair[1], rc)
    grp = groups(kind, rc, pair[0] + 1)
    meta = {"name": f"ground-{name}", "kind": kind, "aef": list(pair), "rc": list(rc), "shape": [2 * HALF, 2 * HALF],
            "point": [lon, lat]}
    os.makedirs(PLACES_DIR, exist_ok=True)
    np.savez_compressed(path, X=np.concatenate([b, a], 1).astype(np.float16), grp=grp,
                        y=np.where(okb & oka & (grp > 0), 0, -1).astype(np.int8), meta=json.dumps(meta))
    got = read_rc(g, f"ground-{name}", rc, pair)
    zone = got.pop("zone")
    np.savez_compressed(f"{OUT}/ground-{name}.npz", meta=json.dumps({**meta, "zone": zone}), **got)
    print(f"  {pair}, groups {np.bincount(grp, minlength=3)[1:]} px")


SAVE_KIND = {"burn": "fire", "cut": "cut", "lake": "lake"}  # "burn" is taken by the lidar burns


def saved(n, d, li):
    """The kind and labels a ground place's scores are saved under: only the event's ground (loss, lake bed)
    counts as unchanged, so quick.py and checks.py read it as they read the other places."""
    grp = np.load(f"{PLACES_DIR}/ground-{n}.npz")["grp"]
    return SAVE_KIND[d["kind"]], np.where(grp[li] == 1, 0, -1).astype(np.int8)


def score(names):
    from compare import labeled, load
    from aggregate import base_scores
    from models import final
    from quick import RULES
    show = ("aef_gated", "tes", "mean", "both", "both_either", "either_sqrt")
    rec = {r: final(r) for r in ("us", "abroad")}
    res = {}
    print(f"\nshare lit at each recipe's 60% cutoff: event ground (loss / lake bed) | ground to compare | radar b/a, S2 looks b/a")
    print(f"  {'place':22s} {'kind':4s} {'years':11s}  recipe  " + "  ".join(f"{w:>11s}" for w in show))
    for n in names:
        d = load(f"ground-{n}")
        grp = np.load(f"{PLACES_DIR}/ground-{n}.npz")["grp"]
        li = labeled(d)
        res[n] = {"kind": d["kind"], "years": d["years"], "px": [int((grp[li] == k).sum()) for k in (1, 2)],
                  "valid": round(float(d["ok"][grp > 0].mean()), 3),
                  "radar": [float(np.median(d["s1b"][li])), float(np.median(d["s1a"][li]))],
                  "s2": [float(np.median(d["s2b"][li])), float(np.median(d["s2a"][li]))]}
        for r, m in rec.items():
            s = base_scores(m["models"], d, li)
            kind, y = saved(n, d, li)
            _saver(r)(f"ground-{n}", kind, s, y, "test", m["bounds"])
            S = {w: RULES[w](s, m["bounds"]) for w in show}
            cut = m["cutoff"][0.6]
            res[n][r] = {k: {w: round(float((S[w][grp[li] == g_] > cut[w]).mean()), 3) for w in show}
                         for k, g_ in (("event", 1), ("compare", 2))}
            for k in ("event", "compare"):
                tag = f"{n:22s} {d['kind']:4s} {str(d['years']):11s}" if (r, k) == ("us", "event") else f"{'':22s} {'':4s} {'':11s}"
                extra = (f"   radar {res[n]['radar'][0]:.0f}/{res[n]['radar'][1]:.0f}  S2 {res[n]['s2'][0]:.0f}/{res[n]['s2'][1]:.0f}"
                         f"  px {res[n]['px']}  valid {res[n]['valid']:.2f}") if (r, k) == ("us", "event") else ""
                print(f"  {tag}  {r:6s} {k[:3]}  " + "  ".join(f"{res[n][r][k][w]:11.3f}" for w in show) + extra)
        del d
    for kind in ("burn", "cut", "lake"):
        ks = [n for n in res if res[n]["kind"] == kind]
        if not ks:
            continue
        print(f"\n  mean over {len(ks)} {kind} places")
        for r in rec:
            for k in ("event", "compare"):
                print(f"  {kind:4s} {r:6s} {k:7s} " + "  ".join(f"{w} {np.mean([res[n][r][k][w] for n in ks]):.3f}" for w in show))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, "results", "ground.json"), "w"), indent=1)


def _saver(recipe):
    """aggregate.save_scores writing into the recipe's scores directory."""
    import aggregate
    from models import SCORE_DIRS

    def save(*a, **k):
        aggregate.SCORES = SCORE_DIRS[recipe]
        aggregate.save_scores(*a, **k)
    return save


def main():
    names = sys.argv[1:] or list(PLACES)
    zarr.config.set({"async.concurrency": 128})
    g = ObjectStore(HTTPStore.from_url(TESSERA), read_only=True)
    for n in names:
        t = time.time()
        try:
            read(g, n)
        except Exception as e:  # a place outside TESSERA's years or a tile edge: say so and go on
            print(f"  {n}: not read ({type(e).__name__}: {e})")
            continue
        print(f"  {n}: {time.time() - t:.0f} s", flush=True)
    score([n for n in names if os.path.exists(f"{OUT}/ground-{n}.npz")])


if __name__ == "__main__":
    main()
