"""Dense cities: why TESSERA lights whole blocks, and two ways to correct it.

On the map, Manhattan's Lower East Side from 2022 to 2025 lit block after block under TESSERA alone, while AEF and
the aggregate lit a few spots and East River Park. TESSERA's radar there: about 30 ascending looks a year from 2017
to 2025 and no descending looks until 2025 (17), when Sentinel-1C began. Radar over buildings depends on the side
it looks from, so a new look direction could read as change. The radar-safe rule only catches radar coming or
going, not a second direction added. Or TESSERA may simply be noisy in dense cities whatever the years: the
earlier tests had few dense blocks.

Here: 3 km squares in 22 dense cities (14 in the US, 8 abroad), every year 2017 to 2025 in both embeddings, every
pair scored by the abroad recipe's final models at each rule's 60% cutoff. Few city blocks are dug in a few years,
so the share lit is close to a false-alarm rate (new buildings count). Pixels are grouped by their radar in the
two years: the same look directions both years, a direction added or dropped, or radar there one year only.

Corrections, each scored the same way:
  shift      TESSERA's change less the median change over the square (the map would take it over the view): a
             shift every pixel shares is the sensor, digging is local
  shift_both the same for AlphaEarth too
  pcs_k      TESSERA's change less its k main directions on pairs where a look direction was added or dropped,
             learned at the other cities on pixels AlphaEarth calls unchanged (each city held out), after shift
Then the cost on the 14 digging sites (moved found, unchanged lit, AP), each correction applied the same way. The
final models were taught on those sites, so their numbers run high; what matters is the change each correction
makes.

Run: uv run python cities.py          (reads into data/cities, writes results/cities.json)
     uv run python cities.py sites     (the digging sites again, on the saved cities)
     uv run python cities.py report
"""

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations

import numpy as np
import zarr
from obstore.store import HTTPStore
from sklearn.metrics import average_precision_score
from zarr.storage import ObjectStore

from tessera_sites import AEF_RES, AEF_X0, AEF_Y0, TESSERA, aef_unit, read_rc

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(HERE, "data", "cities")
OUT = os.path.join(HERE, "results", "cities.json")
YEARS = tuple(range(2017, 2026))
HALF = 150  # 3 km squares
CITIES = {
    "nyc-lower-east-side": (-73.984, 40.718), "nyc-midtown": (-73.985, 40.754), "brooklyn-park-slope": (-73.980, 40.672),
    "chicago-loop": (-87.630, 41.880), "philadelphia-center": (-75.165, 39.952), "boston-back-bay": (-71.080, 42.350),
    "sf-mission": (-122.418, 37.760), "la-koreatown": (-118.300, 34.062), "houston-montrose": (-95.390, 29.745),
    "dc-capitol-hill": (-76.990, 38.890), "seattle-capitol-hill": (-122.320, 47.620), "atlanta-midtown": (-84.385, 33.780),
    "denver-downtown": (-104.985, 39.740), "phoenix-downtown": (-112.070, 33.450),
    "london-islington": (-0.100, 51.540), "paris-11e": (2.380, 48.860), "berlin-kreuzberg": (13.420, 52.500),
    "tokyo-shinjuku": (139.700, 35.690), "mexico-city-roma": (-99.160, 19.420), "sao-paulo-centro": (-46.650, -23.560),
    "lagos-surulere": (3.355, 6.500), "mumbai-dadar": (72.840, 19.020),
}
US = ("nyc-lower-east-side", "nyc-midtown", "brooklyn-park-slope", "chicago-loop", "philadelphia-center",
      "boston-back-bay", "sf-mission", "la-koreatown", "houston-montrose", "dc-capitol-hill", "seattle-capitol-hill",
      "atlanta-midtown", "denver-downtown", "phoenix-downtown")
RULES_SHOWN = ("aef_gated", "tes", "either_radar")
CORR = ("none", "shift", "shift_both", "pcs_2", "pcs_8")
LOOK = ("same looks", "a direction added or dropped", "radar one year only")


def read(g, name):
    path = f"{DIR}/{name}.npz"
    if os.path.exists(path):
        return path
    lon, lat = CITIES[name]
    r, c = int((AEF_Y0 - lat) / AEF_RES), int((lon - AEF_X0) / AEF_RES)
    rc = (r - HALF, r + HALF, c - HALF, c + HALF)
    out = {}
    for y in YEARS:
        e, ok = aef_unit(y, rc)
        out[f"aef_{y}"], out[f"aok_{y}"] = e.astype(np.float16), ok
    out.update(read_rc(g, name, rc, YEARS, tags=[str(y) for y in YEARS]))
    out["zone"] = np.array(out["zone"])
    os.makedirs(DIR, exist_ok=True)
    np.savez_compressed(path, rc=np.array(rc), **out)
    return path


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)


def looks(z, y):
    """Per pixel: 1 ascending, 2 descending, 3 both, 0 no radar."""
    return (z[f"{y}_s1_asc_obs_count"] > 0).astype(np.int8) + 2 * (z[f"{y}_s1_desc_obs_count"] > 0)


def pair(z, yb, ya):
    """A year pair as aggregate.base_scores reads it, the pixels both embeddings cover, and each pixel's look group
    (0 same looks, 1 a direction added or dropped, 2 radar one year only)."""
    ab, aa = z[f"aef_{yb}"].astype(np.float32), z[f"aef_{ya}"].astype(np.float32)
    sb, sa = z[f"{yb}_scales"], z[f"{ya}_scales"]
    ok = z[f"aok_{yb}"] & z[f"aok_{ya}"] & np.isfinite(sb) & np.isfinite(sa) & (sb > 0) & (sa > 0)
    tb = unit(np.nan_to_num(z[f"{yb}_q"].astype(np.float32) * sb[:, None]))
    ta = unit(np.nan_to_num(z[f"{ya}_q"].astype(np.float32) * sa[:, None]))
    lb, la = looks(z, yb), looks(z, ya)
    flip = (lb > 0) != (la > 0)
    group = np.where(flip, 2, np.where(lb != la, 1, 0))
    d = {"ab": ab, "aa": aa, "tb": tb, "ta": ta, "aplain": 1 - (ab * aa).sum(1), "tplain": 1 - (tb * ta).sum(1),
         "radar_flip": flip}
    return d, np.flatnonzero(ok), group


def correct(d, li, how, pcs=None, med=None):
    """d with the correction applied over the pixels li (in place on copies of the changed arrays). med, if given,
    holds each embedding's shift ("t", "a") taken elsewhere, in place of the median over li."""
    if how == "none":
        return d
    d = dict(d)
    keys = ("t",) if how != "shift_both" else ("t", "a")
    for k in keys:
        b, a = d[f"{k}b"][li].astype(np.float32), d[f"{k}a"][li].astype(np.float32)
        ch = a - b
        ch -= np.median(ch, axis=0) if med is None else med[k]
        if how.startswith("pcs") and k == "t":
            ch -= (ch @ pcs.T) @ pcs
        new = d[f"{k}a"].astype(np.float32).copy()
        new[li] = unit(b + ch)
        d[f"{k}a"] = new
        d[f"{k}plain"] = d[f"{k}plain"].copy()
        d[f"{k}plain"][li] = 1 - (b * new[li]).sum(1)
    return d


def look_rows(n):
    """At one city: TESSERA's change on pixels AlphaEarth calls unchanged where a look direction was added or
    dropped, after the square's median shift (up to 3,000 pixels a pair)."""
    rng = np.random.default_rng(0)
    rows = []
    z = np.load(f"{DIR}/{n}.npz")
    for yb, ya in combinations(YEARS, 2):
        d, li, group = pair(z, yb, ya)
        sel = li[(group[li] == 1) & (d["aplain"][li] < 0.03)]
        if len(sel) < 500:
            continue
        med = np.median(d["ta"][li] - d["tb"][li], axis=0)
        sel = rng.choice(sel, min(len(sel), 3000), replace=False)
        rows.append(d["ta"][sel] - d["tb"][sel] - med)
    return np.concatenate(rows) if rows else np.zeros((0, 128), np.float32)


def look_changes(rows, k):
    """The k main directions of the pooled rows."""
    X = np.concatenate(rows)
    if len(X) < 1000:
        return np.zeros((0, 128), np.float32)
    _, _, vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
    return vt[:k].astype(np.float32)


def score_city(n, rec, pcs_by_k):
    from aggregate import base_scores
    from quick import RULES
    z = np.load(f"{DIR}/{n}.npz")
    out = {"looks": {y: np.bincount(looks(z, y).ravel(), minlength=4).tolist() for y in YEARS}, "pairs": {}}
    for yb, ya in combinations(YEARS, 2):
        d0, li, group = pair(z, yb, ya)
        if len(li) < 1000:
            continue
        row = {"groups": [int((group[li] == g).sum()) for g in range(3)]}
        for how in CORR:
            pcs = pcs_by_k.get(int(how.split("_")[1])) if how.startswith("pcs") else None
            d = correct(d0, li, how, pcs)
            s = base_scores(rec["models"], d, li)
            for w in RULES_SHOWN:
                lit = RULES[w](s, rec["bounds"]) > rec["cutoff"][0.6][w]
                row[f"{how}:{w}"] = [float(lit[group[li] == g].mean()) if (group[li] == g).sum() >= 200 else None for g in range(3)]
                row[f"{how}:{w}:all"] = float(lit.mean())
        out["pairs"][f"{yb}-{ya}"] = row
    return out


def score_site(n, rec, pcs_by_k):
    """Moved found, unchanged lit and AP at one digging site under each correction."""
    from aggregate import base_scores
    from compare import load
    from quick import RULES
    d0 = load(n)
    li = np.flatnonzero(d0["ok"] & (d0["y"] >= 0))
    allok = np.flatnonzero(d0["ok"])
    y = d0["y"][li] == 1
    # the shift is the median over the whole tile, as the map would take it over a view (from 200,000 pixels:
    # the full tile at float32 for each correction outgrew memory)
    samp = np.random.default_rng(0).choice(allok, min(len(allok), 200_000), replace=False)
    med = {k: np.median(d0[f"{k}a"][samp].astype(np.float32) - d0[f"{k}b"][samp].astype(np.float32), axis=0)
           for k in ("t", "a")}
    N = len(d0["y"])
    d0 = {k: v[li] if isinstance(v, np.ndarray) and len(v) == N else v for k, v in d0.items()}
    out = {}
    for how in CORR:
        pcs = pcs_by_k.get(int(how.split("_")[1])) if how.startswith("pcs") else None
        V = {w: [] for w in RULES_SHOWN}
        for i in range(0, len(li), 200_000):  # in pieces, so the features never sit whole in memory
            part = np.arange(i, min(i + 200_000, len(li)))
            dp = {k: v[part] if isinstance(v, np.ndarray) and len(v) == len(li) else v for k, v in d0.items()}
            s = base_scores(rec["models"], correct(dp, np.arange(len(part)), how, pcs, med), np.arange(len(part)))
            for w in RULES_SHOWN:
                V[w].append(RULES[w](s, rec["bounds"]))
        for w in RULES_SHOWN:
            v = np.concatenate(V[w])
            lit = v > rec["cutoff"][0.6][w]
            out[f"{how}:{w}"] = {"found": float(lit[y].mean()) if y.any() else None, "lit": float(lit[~y].mean()),
                                 "ap": float(average_precision_score(y, v)) if y.any() else None}
    return out


def in_child(fn, *args):
    with ProcessPoolExecutor(1, max_tasks_per_child=1) as ex:
        return ex.submit(fn, *args).result()


def main():
    from models import final
    zarr.config.set({"async.concurrency": 128})
    g = ObjectStore(HTTPStore.from_url(TESSERA), read_only=True)
    for n in CITIES:
        t = time.time()
        read(g, n)
        print(f"read {n}: {time.time() - t:.0f} s", flush=True)
    rec = final("abroad")
    res = {"cities": {}, "sites": {}}
    names = list(CITIES)
    LR = {n: look_rows(n) for n in names}
    print("look rows per city: " + ", ".join(f"{n} {len(v)}" for n, v in LR.items()), flush=True)
    for n in names:
        t = time.time()
        pcs_by_k = {k: look_changes([LR[m] for m in names if m != n], k) for k in (2, 8)}
        res["cities"][n] = in_child(score_city, n, rec, pcs_by_k)
        print(f"scored {n}: {time.time() - t:.0f} s", flush=True)
        json.dump(res, open(OUT, "w"))
    sites(res, LR)


def sites(res=None, LR=None):
    """The digging sites alone (cities.py sites reruns them on the saved cities)."""
    from compare import TEACH_KINDS, teaching_names
    from models import final
    rec = final("abroad")
    res = res or json.load(open(OUT))
    res["sites"] = {}
    LR = LR or {n: look_rows(n) for n in CITIES}
    pcs_all = {k: look_changes(list(LR.values()), k) for k in (2, 8)}
    del LR
    for n in teaching_names():
        if json.loads(str(np.load(f"{HERE}/data/tessera/{n}.npz", allow_pickle=True)["meta"]))["kind"] not in TEACH_KINDS:
            continue
        t = time.time()
        res["sites"][n] = in_child(score_site, n, rec, pcs_all)
        print(f"site {n}: {time.time() - t:.0f} s", flush=True)
        json.dump(res, open(OUT, "w"))
    report()


def report():
    res = json.load(open(OUT))
    C = res["cities"]
    us = [n for n in C if n in US]
    print("\nlook directions per year, share of pixels (asc only / desc only / both / none), first US and abroad city:")
    for n in (us[0], [m for m in C if m not in us][0]):
        print(f"  {n}: " + "  ".join(f"{y}: " + "/".join(f"{v / max(sum(c), 1):.2f}" for v in (c[1], c[2], c[3], c[0]))
                                    for y, c in C[n]["looks"].items()))

    def mean_lit(names, key, g, pairs=None):
        per = []
        for n in names:
            v = [r[key][g] for p, r in C[n]["pairs"].items() if r[key][g] is not None and (pairs is None or p in pairs)]
            if v:
                per.append(np.mean(v))
        return (float(np.mean(per)), len(per)) if per else (None, 0)

    print("\nshare of city pixels lit at the 60% cutoff, mean over cities (each city's pairs averaged), by radar looks")
    for label, names in (("US cities", us), ("abroad cities", [n for n in C if n not in us])):
        print(f"  {label}")
        print(f"    {'correction:rule':26s} " + " ".join(f"{g[:22]:>24s}" for g in LOOK))
        for how in CORR:
            for w in RULES_SHOWN:
                cells = []
                for g in range(3):
                    m, k = mean_lit(names, f"{how}:{w}", g)
                    cells.append(f"{m:.3f} ({k:2d})" if m is not None else "-")
                print(f"    {how + ':' + w:26s} " + " ".join(f"{c:>24s}" for c in cells))
    n = "nyc-lower-east-side"
    if n in C:
        print(f"\n{n}, share lit (all pixels):")
        for p in ("2022-2024", "2022-2025", "2023-2024", "2023-2025", "2024-2025", "2019-2021", "2017-2021"):
            r = C[n]["pairs"].get(p)
            if r:
                print(f"  {p}  groups {r['groups']}  " + "  ".join(f"{how}: " + " ".join(f"{r[f'{how}:{w}:all']:.3f}" for w in RULES_SHOWN) for how in CORR))
        print(f"  (each: {' '.join(RULES_SHOWN)})")
    S = res.get("sites", {})
    if S:
        print(f"\ndigging sites ({len(S)}), mean of each site: moved found / unchanged lit / AP")
        for how in CORR:
            print(f"  {how:11s} " + "   ".join(
                f"{w}: " + "/".join(f"{np.mean([S[s][f'{how}:{w}'][m] for s in S if S[s][f'{how}:{w}'][m] is not None]):.3f}" for m in ("found", "lit", "ap"))
                for w in RULES_SHOWN))


if __name__ == "__main__":
    {"report": report, "sites": sites}.get(sys.argv[1] if sys.argv[1:] else "", main)()
