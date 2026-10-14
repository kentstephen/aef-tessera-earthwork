"""Zoomed out: can a coarse cell say "something was dug here", and from what?

The map works at 10 m. The first idea for a second embedding was a coarser view to find regions, then AEF
zoomed in. Here, on the lidar sites, square cells of about 320 m and 640 m:
  dug     at least 0.2 ha of moved ground in the cell
  quiet   no moved pixel, and at least half the cell labeled unchanged
  (cells between are left out)
Ways to score a cell:
  from the 10 m scores (every pixel, models taught without the site): the cell's highest score, its 99th
  percentile, its mean, and the share of its pixels over the rule's 60% cutoff; for the map today (aef_gated)
  and the aggregate (both_either)
  from pooled embeddings, a coarse model taught on the other sites' cells:
    overview   AEF mean-pooled and renormalized per year, as AEF's COG overviews store it: [b, a, a * b, (a - b)^2]
    stats      mean, std, min and max over the cell of AEF's change (a - b), as Earth Genome pooled AEF
    stats_tes  the same for TESSERA's change
    stats_both both
Each digging site held out in turn (AP of dug against quiet cells); farms, marshes and the storm coast scored
by models taught on every digging site: the share of their quiet cells lit at a cutoff that finds 60% of dug
cells on the digging sites.

Run: uv run python coarse.py   (writes results/coarse.json; data/coarse keeps each site's 10 m scores)
"""

import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from abroad_teach import fit_rows, rows
from aggregate import base_scores, cutoff
from compare import load
from quick import RULES
from tessera_sites import AEF_RES, AEF_Y0, OUT, SITES

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(HERE, "data", "coarse")
CELL_M = (320, 640)
DUG_HA, PX_M = 0.2, 10.0
SHOW = ("aef_gated", "both_either")
TEST_KINDS = ("farm", "marsh", "storm")


def site_grid(name):
    """The site's shape and the latitude of its center (for the east-west pixel size)."""
    m = json.loads(str(np.load(f"{OUT}/{name}.npz", allow_pickle=True)["meta"]))
    h, w = m["shape"]
    return h, w, AEF_Y0 - (m["rc"][0] + h / 2) * AEF_RES


def fine_scores(models, bounds, d):
    """Every rule's 10 m score at every pixel both embeddings cover (NaN elsewhere)."""
    n = len(d["y"])
    ok = np.flatnonzero(d["ok"])
    out = {w: np.full(n, np.nan, np.float32) for w in SHOW}
    for i in range(0, len(ok), 200000):
        li = ok[i:i + 200000]
        s = base_scores(models, d, li)
        for w in SHOW:
            out[w][li] = RULES[w](s, bounds)
    return out


def cells(name, d, cell_m):
    """Each pixel's cell id, and per cell: moved px, unchanged px, px, and the cell label (1 dug, 0 quiet, -1)."""
    h, w, lat = site_grid(name)
    ch = int(round(cell_m / PX_M))
    cw = int(round(cell_m / (PX_M * np.cos(np.radians(lat)))))
    r, c = np.divmod(np.arange(h * w), w)
    nc = (w + cw - 1) // cw
    cid = (r // ch) * nc + c // cw
    k = cid.max() + 1
    y = d["y"]
    moved = np.bincount(cid, y == 1, k)
    still = np.bincount(cid, (y == 0) & d["ok"], k)
    px = np.bincount(cid, None, k)
    px_ha = PX_M * PX_M * np.cos(np.radians(lat)) / 1e4
    lab = np.where(moved * px_ha >= DUG_HA, 1, np.where((moved == 0) & (still >= 0.5 * px), 0, -1))
    full = px >= 0.9 * ch * cw  # cells cut by the site's edge are left out
    return cid, np.where(full, lab, -1)


def pool_fine(cid, k, s, cut):
    """Per cell: highest, 99th percentile, mean, and share over the cutoff of the 10 m score (NaN pixels left out)."""
    v = np.isfinite(s)
    c, x = cid[v], s[v]
    o = np.argsort(c, kind="stable")
    c, x = c[o], x[o]
    starts = np.searchsorted(c, np.arange(k))
    ends = np.searchsorted(c, np.arange(k), side="right")
    out = np.full((k, 4), np.nan, np.float32)
    for i in range(k):
        seg = x[starts[i]:ends[i]]
        if len(seg) >= 20:
            out[i] = (seg.max(), np.quantile(seg, 0.99), seg.mean(), (seg > cut).mean())
    return out


def pool_emb(cid, k, d):
    """Per cell: AEF overview features, and mean/std/min/max of the change in each embedding."""
    v = d["aok"] & d["ok"]
    c = cid[v]
    cnt = np.maximum(np.bincount(c, None, k), 1)[:, None]

    def stats(x):
        mean = np.stack([np.bincount(c, x[:, j], k) for j in range(x.shape[1])], 1) / cnt
        sq = np.stack([np.bincount(c, x[:, j] ** 2, k) for j in range(x.shape[1])], 1) / cnt
        mn = np.full((k, x.shape[1]), np.inf, np.float32)
        mx = np.full((k, x.shape[1]), -np.inf, np.float32)
        np.minimum.at(mn, c, x)
        np.maximum.at(mx, c, x)
        return np.concatenate([mean, np.sqrt(np.maximum(sq - mean ** 2, 0)), mn, mx], 1).astype(np.float32)
    ab, aa = d["ab"][v].astype(np.float32), d["aa"][v].astype(np.float32)
    tb, ta = d["tb"][v].astype(np.float32), d["ta"][v].astype(np.float32)

    def unit_mean(x):
        m = np.stack([np.bincount(c, x[:, j], k) for j in range(x.shape[1])], 1)
        return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)
    b, a = unit_mean(ab), unit_mean(aa)
    out = {"overview": np.concatenate([b, a, a * b, (a - b) ** 2], 1).astype(np.float32),
           "stats": stats(aa - ab), "stats_tes": stats(ta - tb)}
    empty = np.bincount(c, None, k) < 20  # too few pixels both embeddings cover: no pooled features
    for v in out.values():
        v[empty] = np.nan
    return out


def main():
    os.makedirs(DIR, exist_ok=True)
    T, W, D, teach, quiet, _ = rows(())
    D.clear()
    test = [n for n in SITES if os.path.exists(f"{OUT}/{n}.npz") and n not in teach + quiet]
    from aggregate import tes_gate_bounds
    every = lambda n: np.arange(len(T[n]["y"]))

    # 10 m scores everywhere: digging and quiet sites by models taught without them, test sites by all
    for n in teach + quiet + test:
        path = f"{DIR}/{n}.npz"
        if os.path.exists(path):
            continue
        train = [m for m in T if m != n]
        models = {w: fit_rows(T, W, train, w) for w in ("aef", "tes", "both")}
        us = [m for m in train if m in teach + quiet]
        bounds = tes_gate_bounds(T, {m: every(m) for m in us}, us)
        sc = fine_scores(models, bounds, load(n))
        np.savez_compressed(path, **sc)
        print(f"10 m scores: {n}", flush=True)

    from models import final
    cut = final("us")["cutoff"][0.6]
    res = {}
    for cell_m in CELL_M:
        F, E, L, K = {}, {}, {}, {}
        for n in teach + quiet + test:
            d = load(n)
            cid, lab = cells(n, d, cell_m)
            k = len(lab)
            sc = np.load(f"{DIR}/{n}.npz")
            F[n] = {w: pool_fine(cid, k, sc[w], cut[w]) for w in SHOW}
            E[n] = pool_emb(cid, k, d)
            E[n]["stats_both"] = np.concatenate([E[n]["stats"], E[n]["stats_tes"]], 1)
            L[n], K[n] = lab, d["kind"]
            del d
        print(f"\n===== cells of {cell_m} m: dug / quiet cells per digging site =====")
        print("  " + ", ".join(f"{n} {(L[n] == 1).sum()}/{(L[n] == 0).sum()}" for n in teach))
        out = res[cell_m] = {"cells": {n: [int((L[n] == 1).sum()), int((L[n] == 0).sum())] for n in teach + quiet + test}}
        # coarse heads, each digging site held out
        heads = ("overview", "stats", "stats_tes", "stats_both")
        H = {h: {} for h in heads}
        for h in heads:
            for n in teach + quiet + ["*"]:
                tr = [m for m in teach + quiet if m != n]
                use = {m: (L[m] >= 0) & np.isfinite(E[m][h]).all(1) for m in tr}
                X = np.concatenate([E[m][h][use[m]] for m in tr])
                y = np.concatenate([L[m][use[m]] for m in tr])
                model = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced")).fit(X, y)
                for m in ([n] if n != "*" else test):
                    fin = np.isfinite(E[m][h]).all(1)
                    H[h][m] = np.full(len(fin), np.nan)
                    H[h][m][fin] = model.predict_proba(E[m][h][fin])[:, 1]
        names = [f"{w}:{p}" for w in SHOW for p in ("max", "p99", "mean", "lit")] + list(heads)

        def score(n, nm):
            if ":" in nm:
                w, p = nm.split(":")
                return F[n][w][:, ("max", "p99", "mean", "lit").index(p)]
            return H[nm][n]
        print(f"\n  {'cell score':22s} {'mean AP':>7s}  {'building':>8s} {'mining':>6s}  | quiet cells lit at 60% of dug cells found: " + " ".join(f"{k:>6s}" for k in ("burn", "dunes") + TEST_KINDS))
        for nm in names:
            aps, kinds = [], []
            for n in teach:
                s, lab = score(n, nm), L[n]
                sel = (lab >= 0) & np.isfinite(s)
                if (lab[sel] == 1).sum() >= 3:
                    aps.append(average_precision_score(lab[sel], s[sel]))
                    kinds.append(K[n])
            aps, kinds = np.array(aps), np.array(kinds)
            c = cutoff([np.nan_to_num(score(n, nm)[L[n] == 1], nan=-1) for n in teach if (L[n] == 1).sum() >= 3], 0.6)
            fa = {}
            for n in quiet + test:
                s = np.nan_to_num(score(n, nm), nan=-1)
                if (L[n] == 0).sum() >= 10:
                    fa.setdefault(K[n], []).append(float((s[L[n] == 0] > c).mean()))
            out[nm] = {"ap": round(float(aps.mean()), 3), "ap_building": round(float(aps[kinds == "building"].mean()), 3),
                       "ap_mining": round(float(aps[kinds == "mining"].mean()), 3), "quiet_lit": {k: round(float(np.mean(v)), 3) for k, v in fa.items()}}
            print(f"  {nm:22s} {aps.mean():7.3f}  {out[nm]['ap_building']:8.3f} {out[nm]['ap_mining']:6.3f}  |  "
                  + " ".join(f"{out[nm]['quiet_lit'].get(k, float('nan')):6.3f}" for k in ("burn", "dunes") + TEST_KINDS))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump({str(k): v for k, v in res.items()}, open(os.path.join(HERE, "results", "coarse.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
