"""Structures: does something stand on the pixel in both years? A gate for the earthwork score.

A roof coated white, a dense block whose radar changed: the ground under them did not move, and something stood
there both years. The flat ground reading of the built-up work (embeddings-built-up, height_ground.py) reads from
AlphaEarth alone whether a structure stands on a pixel, any year: at New Capital, held out, AUC .996, 97% of
structures found and 98% of ground called flat, about the same every year. Here the same reading, taught in the
city squares of cities.py and whiteroofs.py:

  teacher   Overture building footprints (2026-09-23.0, read with DuckDB from its S3 bucket): a pixel center under a
            footprint is a structure, a pixel with no footprint on it or beside it is ground. Overture is the state
            now, so AlphaEarth 2025 teaches; every year is read by the same model.
  model     gradient boosting on the pixel's 64 values and the mean of its 3 x 3, classes balanced (as there), held
            out by fold: the seven New York squares are one fold, so New York is always read by a model that never
            saw it, the other cities four folds.
  gate      hard: a pixel that reads standing (50% or more) in both years of the pair is not lit. soft: the score
            times one less the smaller of the two years' chances.

Scored as before: the share of city pixels lit by radar looks (section 25), the whitened and steady roofs
(section 26), and the cost at the 14 digging sites (moved found, unchanged lit, AP), read by a model taught on every
square (none of the sites).

Run: uv run python structures.py          (footprints cached in data/structures, writes results/structures.json)
     uv run python structures.py sites    (the digging sites again, on the saved cities)
     uv run python structures.py report
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

import cities
import whiteroofs
from tessera_sites import AEF_RES, AEF_X0, AEF_Y0

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(HERE, "data", "structures")
OUT = os.path.join(HERE, "results", "structures.json")
OVERTURE = "s3://overturemaps-us-west-2/release/2026-09-23.0/theme=buildings/type=building/*"
SQUARES = list(cities.CITIES) + [n for n in whiteroofs.SQUARES if n not in cities.CITIES]
NYC = [n for n in whiteroofs.SQUARES]
PER_CLASS = 15_000  # teaching pixels a square, each class
STAND = 0.5
GATES = ("none", "hard", "soft")


def footprints(name, rc):
    """Overture footprints on the square's AlphaEarth grid: (h, w) bool, a pixel center under one."""
    path = f"{DIR}/{name}_on.npy"
    if os.path.exists(path):
        return np.load(path)
    import duckdb
    from rasterio.features import rasterize
    from rasterio.transform import Affine
    from shapely import from_wkb
    r0, r1, c0, c1 = (int(v) for v in rc)
    W, E = AEF_X0 + c0 * AEF_RES, AEF_X0 + c1 * AEF_RES
    N, S = AEF_Y0 - r0 * AEF_RES, AEF_Y0 - r1 * AEF_RES
    c = duckdb.connect()
    c.execute("INSTALL httpfs; LOAD httpfs; SET s3_region='us-west-2'; SET enable_progress_bar=false")
    rows = c.execute(f"""SELECT geometry FROM read_parquet('{OVERTURE}', hive_partitioning=1)
                         WHERE bbox.xmin < {E} AND bbox.xmax > {W} AND bbox.ymin < {N} AND bbox.ymax > {S}""").fetchall()
    geoms = [g for g in from_wkb([r[0] for r in rows]) if g is not None and not g.is_empty]
    tr = Affine(AEF_RES, 0, W, 0, -AEF_RES, N)
    on = rasterize(((g, 1) for g in geoms), out_shape=(r1 - r0, c1 - c0), transform=tr, fill=0, dtype=np.uint8).astype(bool)
    os.makedirs(DIR, exist_ok=True)
    np.save(path, on)
    return on


def feats(E, shape):
    """The pixel's values and the mean of its 3 x 3: (n, 2 * d), from (n, d) on a (h, w) grid."""
    from scipy.ndimage import uniform_filter
    h, w = shape
    A = np.nan_to_num(E.astype(np.float32)).reshape(h, w, -1)
    M = uniform_filter(A, size=(3, 3, 1), mode="nearest")
    return np.hstack([A.reshape(h * w, -1), M.reshape(h * w, -1)])


def fit(X, y):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=63, l2_regularization=1.0,
                                          class_weight="balanced", random_state=0).fit(X, y)


def fold_of(n):
    if n in NYC:
        return 0
    others = [m for m in SQUARES if m not in NYC]
    return 1 + others.index(n) % 4


def teach_rows(n, rng):
    from scipy.ndimage import binary_dilation
    z = np.load(f"{cities.DIR}/{n}.npz")
    on = footprints(n, z["rc"])
    shape = on.shape
    ok = z["aok_2025"].reshape(shape)
    ground = ~binary_dilation(on, iterations=1)
    X = feats(z["aef_2025"], shape)
    pick, lab = [], []
    for k, m in ((1, on & ok), (0, ground & ok)):
        i = np.flatnonzero(m.ravel())
        i = rng.choice(i, min(PER_CLASS, len(i)), replace=False)
        pick.append(i)
        lab.append(np.full(len(i), k, np.int8))
    i = np.concatenate(pick)
    return X[i], np.concatenate(lab), float(on.mean())


def read_years(mdl, z, shape):
    return {y: np.where(z[f"aok_{y}"], mdl.predict_proba(feats(z[f"aef_{y}"], shape))[:, 1], np.nan).astype(np.float32)
            for y in cities.YEARS}


def gated(v, pb, pa, how):
    if how == "none":
        return v
    if how == "hard":
        return np.where((pb >= STAND) & (pa >= STAND), 0.0, v)
    return v * (1 - np.minimum(np.nan_to_num(pb), np.nan_to_num(pa)))


def score_city(n, z, P, rec):
    from aggregate import base_scores
    from quick import RULES
    out = {"stand_share": {y: float(np.nanmean(P[y] >= STAND)) for y in cities.YEARS}, "pairs": {}}
    for yb, ya in combinations(cities.YEARS, 2):
        d, li, group = cities.pair(z, yb, ya)
        if len(li) < 1000:
            continue
        s = base_scores(rec["models"], d, li)
        pb, pa = P[yb][li], P[ya][li]
        both = (pb >= STAND) & (pa >= STAND)
        row = {"groups": [int((group[li] == g).sum()) for g in range(3)], "both_stand": float(both.mean())}
        for w in cities.RULES_SHOWN:
            v = RULES[w](s, rec["bounds"])
            for how in GATES:
                lit = gated(v, pb, pa, how) > rec["cutoff"][0.6][w]
                row[f"{how}:{w}"] = [float(lit[group[li] == g].mean()) if (group[li] == g).sum() >= 200 else None for g in range(3)]
                row[f"{how}:{w}:all"] = float(lit.mean())
        out["pairs"][f"{yb}-{ya}"] = row
    return out


def score_roofs(n, z, P, rec):
    from aggregate import base_scores
    from quick import RULES
    lon, lat = whiteroofs.grid(z)
    roof, _ = whiteroofs.roofs(n, lon, lat)
    vis = whiteroofs.visible(n, lon, lat)
    out = {}
    for yb, ya in whiteroofs.PAIRS:
        d, li, _ = cities.pair(z, yb, ya)
        vb, va = vis[str(yb)][li], vis[str(ya)][li]
        on, rise = roof[li] >= 0, va - vb
        groups = {"whitened": on & (rise >= whiteroofs.WHITE_RISE) & (vb < whiteroofs.DARK_BEFORE) & (va > whiteroofs.BRIGHT_AFTER),
                  "darkened": on & (rise <= -whiteroofs.WHITE_RISE) & (va < whiteroofs.DARK_BEFORE) & (vb > whiteroofs.BRIGHT_AFTER),
                  "steady": on & (np.abs(rise) < whiteroofs.STEADY), "off_roofs": ~on}
        s = base_scores(rec["models"], d, li)
        pb, pa = P[yb][li], P[ya][li]
        row = {"px": {g: int(m.sum()) for g, m in groups.items()},
               "both_stand": {g: float(((pb >= STAND) & (pa >= STAND))[m].mean()) if m.any() else None for g, m in groups.items()}}
        for w in cities.RULES_SHOWN:
            v = RULES[w](s, rec["bounds"])
            for how in GATES:
                lit = gated(v, pb, pa, how) > rec["cutoff"][0.6][w]
                row[f"{how}:{w}"] = {g: float(lit[m].mean()) if m.any() else None for g, m in groups.items()}
        out[f"{yb}-{ya}"] = row
    return out


def score_site(n, mdl, rec):
    """Moved found, unchanged lit and AP at one digging site under each gate, and how much of each stands both years."""
    from aggregate import base_scores
    from compare import load
    from quick import RULES
    d = load(n)
    meta = json.loads(str(np.load(f"{HERE}/data/tessera/{n}.npz", allow_pickle=True)["meta"]))
    shape = tuple(meta["shape"])
    pb = mdl.predict_proba(feats(d["ab"], shape))[:, 1].astype(np.float32)
    pa = mdl.predict_proba(feats(d["aa"], shape))[:, 1].astype(np.float32)
    li = np.flatnonzero(d["ok"] & (d["y"] >= 0))
    y = d["y"][li] == 1
    pb, pa = pb[li], pa[li]
    both = (pb >= STAND) & (pa >= STAND)
    out = {"both_moved": float(both[y].mean()) if y.any() else None, "both_unchanged": float(both[~y].mean()),
           "after_moved": float((pa[y] >= STAND).mean()) if y.any() else None, "before_moved": float((pb[y] >= STAND).mean()) if y.any() else None}
    V = {w: [] for w in cities.RULES_SHOWN}
    for i in range(0, len(li), 200_000):  # in pieces, so the features never sit whole
        s = base_scores(rec["models"], d, li[i:i + 200_000])
        for w in cities.RULES_SHOWN:
            V[w].append(RULES[w](s, rec["bounds"]))
    for w in cities.RULES_SHOWN:
        v0 = np.concatenate(V[w])
        for how in GATES:
            v = gated(v0, pb, pa, how)
            lit = v > rec["cutoff"][0.6][w]
            out[f"{how}:{w}"] = {"found": float(lit[y].mean()) if y.any() else None, "lit": float(lit[~y].mean()),
                                 "ap": float(average_precision_score(y, v)) if y.any() else None}
    return out


def main():
    from models import final
    for n, ll in whiteroofs.SQUARES.items():
        if ll is not None:
            cities.CITIES.setdefault(n, ll)
    t = time.time()
    with ThreadPoolExecutor(6) as ex:  # each query mostly waits on Overture's file footers
        list(ex.map(lambda n: footprints(n, np.load(f"{cities.DIR}/{n}.npz")["rc"]), SQUARES))
    print(f"footprints for {len(SQUARES)} squares: {time.time() - t:.0f} s", flush=True)
    rng = np.random.default_rng(0)
    rows = {n: teach_rows(n, rng) for n in SQUARES}
    res = {"cover": {n: r[2] for n, r in rows.items()}, "folds": {}, "cities": {}, "roofs": {}, "sites": {}}
    rec = final("abroad")
    models = {}
    for k in range(5):
        held = [n for n in SQUARES if fold_of(n) == k]
        mdl = fit(np.concatenate([rows[n][0] for n in SQUARES if n not in held]),
                  np.concatenate([rows[n][1] for n in SQUARES if n not in held]))
        models[k] = mdl
        X, y = np.concatenate([rows[n][0] for n in held]), np.concatenate([rows[n][1] for n in held])
        p = mdl.predict_proba(X)[:, 1]
        res["folds"][k] = {"held": held, "auc": float(roc_auc_score(y, p)), "found": float((p[y == 1] >= STAND).mean()),
                           "flat": float((p[y == 0] < STAND).mean())}
        print(f"fold {k} ({', '.join(held)}): AUC {res['folds'][k]['auc']:.3f}, structures found {res['folds'][k]['found']:.3f}, "
              f"ground flat {res['folds'][k]['flat']:.3f}", flush=True)
    for n in SQUARES:
        t = time.time()
        z = dict(np.load(f"{cities.DIR}/{n}.npz"))
        P = read_years(models[fold_of(n)], z, footprints(n, z["rc"]).shape)
        if n in cities.US or n not in NYC:
            res["cities"][n] = score_city(n, z, P, rec)
        if n in NYC:
            res["roofs"][n] = score_roofs(n, z, P, rec)
        print(f"scored {n}: {time.time() - t:.0f} s", flush=True)
        json.dump(res, open(OUT, "w"), indent=1)
    sites(res, rows, rec)


def sites(res=None, rows=None, rec=None):
    """The 14 digging sites, read by a model taught on every square (structures.py sites reruns them alone)."""
    from compare import TEACH_KINDS, teaching_names
    from models import final
    res = res or json.load(open(OUT))
    rec = rec or final("abroad")
    if rows is None:
        rng = np.random.default_rng(0)
        rows = {n: teach_rows(n, rng) for n in SQUARES}
    allm = fit(np.concatenate([r[0] for r in rows.values()]), np.concatenate([r[1] for r in rows.values()]))
    del rows
    res["sites"] = {}
    for n in teaching_names():
        if json.loads(str(np.load(f"{HERE}/data/tessera/{n}.npz", allow_pickle=True)["meta"]))["kind"] not in TEACH_KINDS:
            continue
        t = time.time()
        res["sites"][n] = cities.in_child(score_site, n, allm, rec)
        print(f"site {n}: {time.time() - t:.0f} s", flush=True)
        json.dump(res, open(OUT, "w"), indent=1)
    report(res)


def report(res=None):
    res = res or json.load(open(OUT))
    print("\nthe structure reading, held out by fold (taught on AlphaEarth 2025 and Overture footprints)")
    for k, f in res["folds"].items():
        print(f"  fold {k}: AUC {f['auc']:.3f}, structures found {f['found']:.3f}, ground flat {f['flat']:.3f}  ({', '.join(f['held'])})")
    C = res["cities"]
    us = [n for n in C if n in cities.US]
    ab = [n for n in C if n not in cities.US and n not in NYC]

    def mean_lit(names, key, g):
        per = [np.mean([r[key][g] for r in C[n]["pairs"].values() if r[key][g] is not None]) for n in names
               if any(r[key][g] is not None for r in C[n]["pairs"].values())]
        return (float(np.mean(per)), len(per)) if per else (None, 0)

    print("\nshare of city pixels lit at the 60% cutoff, mean over cities, by radar looks (same / added or dropped)")
    for label, names in (("US cities", us), ("abroad cities", ab)):
        print(f"  {label}: share standing both years (mean over pairs) {np.mean([np.mean([r['both_stand'] for r in C[n]['pairs'].values()]) for n in names]):.2f}")
        for w in cities.RULES_SHOWN:
            print(f"    {w:13s} " + "   ".join(f"{how}: " + " / ".join(f"{m:.3f}" if m is not None else "-" for m, _ in (mean_lit(names, f'{how}:{w}', g) for g in (0, 1)))
                                           for how in GATES))
    n = "nyc-lower-east-side"
    if n in C:
        r = C[n]["pairs"].get("2022-2025")
        if r:
            print(f"\n  {n} 2022-2025, all pixels: " + "   ".join(f"{how}: " + " ".join(f"{r[f'{how}:{w}:all']:.3f}" for w in cities.RULES_SHOWN) for how in GATES)
                  + f"  (each: {' '.join(cities.RULES_SHOWN)})")
    R = res["roofs"]
    if R:
        print("\nNew York roofs, pooled over squares and pairs (pixel-weighted): share lit")
        for w in cities.RULES_SHOWN:
            cells = []
            for how in GATES:
                parts = []
                for g in ("whitened", "darkened", "steady", "off_roofs"):
                    num = sum(r[f"{how}:{w}"][g] * r["px"][g] for v in R.values() for r in v.values() if r[f"{how}:{w}"][g] is not None)
                    den = sum(r["px"][g] for v in R.values() for r in v.values() if r[f"{how}:{w}"][g] is not None)
                    parts.append(f"{num / max(den, 1):.3f}")
                cells.append(f"{how}: " + "/".join(parts))
            print(f"  {w:13s} " + "   ".join(cells) + "   (whitened/darkened/steady/off roofs)")
        for g in ("whitened", "steady", "off_roofs"):
            num = sum(r["both_stand"][g] * r["px"][g] for v in R.values() for r in v.values() if r["both_stand"][g] is not None)
            den = sum(r["px"][g] for v in R.values() for r in v.values() if r["both_stand"][g] is not None)
            print(f"  standing both years, {g}: {num / max(den, 1):.2f}")
    S = res["sites"]
    if S:
        print(f"\ndigging sites ({len(S)}), mean of each site: moved found / unchanged lit / AP")
        print(f"  standing both years: moved {np.mean([v['both_moved'] for v in S.values() if v['both_moved'] is not None]):.3f}, "
              f"unchanged {np.mean([v['both_unchanged'] for v in S.values()]):.3f}; moved standing before {np.mean([v['before_moved'] for v in S.values() if v['before_moved'] is not None]):.3f}, "
              f"after {np.mean([v['after_moved'] for v in S.values() if v['after_moved'] is not None]):.3f}")
        for how in GATES:
            print(f"  {how:5s} " + "   ".join(f"{w}: " + "/".join(f"{np.mean([S[s][f'{how}:{w}'][m] for s in S if S[s][f'{how}:{w}'][m] is not None]):.3f}" for m in ("found", "lit", "ap"))
                                         for w in cities.RULES_SHOWN))


if __name__ == "__main__":
    {"report": report, "sites": sites}.get(sys.argv[1] if sys.argv[1:] else "", main)()
