"""Wider year pairs on the digging sites: does the aggregate still find the moved ground when the two years
straddle more than the lidar window?

Each US digging site's truth is the ground lidar saw move between two flights, and its pair is AlphaEarth's year
before the first flight (b) and the year of the second (a). A map user picks the years, so the pair is often
wider than the work. Here each site is also scored at (b - 1, a), (b - 2, a), (b, a + 1), (b, a + 2) and
(b - 1, a + 1) where both embeddings have the years (2017 to 2025), with the same lidar labels.

The moved ground stays dug in a wider pair, so the share of it found should hold. The unchanged ground is only
known to be unchanged inside the lidar window: a wider pair can take in digging before the first flight or
after the second, so its share lit is an upper bound on the false alarms, not a measure of them.

Scored by the abroad recipe's models with the site held out (abroad_teach.py's rows, each digging site left out
alone, fitted once and kept in data/spans/models.pkl), each rule at the recipe's 60% cutoff (data/scores-abroad).
The native pair is scored too, as a check against data/scores-abroad.

Run: uv run python spans.py read [site ...]   (AlphaEarth and TESSERA for the extra years into data/spans/)
     uv run python spans.py score             (writes results/spans.json)
"""

import json
import os
import pickle
import sys
import time

import numpy as np
from sklearn.metrics import average_precision_score

from tessera_sites import EARTHWORK, OUT, SITES, TESSERA, aef_unit, read_rc

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(HERE, "data", "spans")
DIG = SITES[:14]
YEARS = range(2017, 2026)
SHIFTS = {"native": (0, 0), "b-1": (-1, 0), "b-2": (-2, 0), "a+1": (0, 1), "a+2": (0, 2), "b-1 a+1": (-1, 1)}
SHOW = ("aef_gated", "both_either", "either_radar")


def meta(n):
    return json.loads(str(np.load(f"{OUT}/{n}.npz", allow_pickle=True)["meta"]))


def pairs(n):
    b, a = meta(n)["aef"]
    return {k: (b + db, a + da) for k, (db, da) in SHIFTS.items() if b + db in YEARS and a + da in YEARS}


def read(g, n):
    m = meta(n)
    rc, (b, a) = tuple(m["rc"]), m["aef"]
    need = sorted({y for p in pairs(n).values() for y in p} - {b, a})
    for y in need:
        path = f"{DIR}/{n}-{y}.npz"
        if os.path.exists(path):
            continue
        t = time.time()
        e, ok = aef_unit(y, rc)
        got = read_rc(g, n, rc, (y,), tags="y")
        got.pop("zone")
        os.makedirs(DIR, exist_ok=True)
        np.savez_compressed(path, aef=e.astype(np.float16), aok=ok, **got)
        print(f"{n} {y}: {time.time() - t:.0f} s", flush=True)


def year_data(n, y):
    """One year at a site: AlphaEarth unit vectors, TESSERA unit vectors, valid mask and radar looks."""
    from compare import unit
    m = meta(n)
    b, a = m["aef"]
    if y in (b, a):
        tag = "b" if y == b else "a"
        X = np.load(f"{EARTHWORK}/{n}.npz")["X"]
        e = X[:, :64] if tag == "b" else X[:, 64:128]
        e = e.astype(np.float32)
        ok = np.linalg.norm(e, axis=1) > 0.5
        z = np.load(f"{OUT}/{n}.npz")
    else:
        z = np.load(f"{DIR}/{n}-{y}.npz")
        e, ok, tag = z["aef"].astype(np.float32), z["aok"], "y"
    sc = z[f"{tag}_scales"]
    ok = ok & np.isfinite(sc) & (sc > 0)
    t = unit(np.nan_to_num(z[f"{tag}_q"].astype(np.float32) * sc[:, None]))
    s1 = z[f"{tag}_s1_asc_obs_count"].astype(int) + z[f"{tag}_s1_desc_obs_count"]
    return e, t, ok, s1


def site_pair(n, yb, ya):
    """The site at a year pair in compare.load's layout, enough for aggregate.base_scores."""
    eb, tb, okb, s1b = year_data(n, yb)
    ea, ta, oka, s1a = year_data(n, ya)
    z = np.load(f"{EARTHWORK}/{n}.npz")
    return {"ab": eb.astype(np.float16), "aa": ea.astype(np.float16), "tb": tb.astype(np.float16),
            "ta": ta.astype(np.float16), "aplain": 1 - (eb * ea).sum(1), "tplain": 1 - (tb * ta).sum(1),
            "radar_flip": (s1b > 0) != (s1a > 0), "ok": okb & oka, "y": z["y"], "patch": z["patch"]}


def held_out_models():
    """The abroad recipe's three models with each digging site left out alone, and the TESSERA gate bounds."""
    path = f"{DIR}/models.pkl"
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    from abroad_teach import fit_rows, rows
    from aggregate import tes_gate_bounds
    T, W, D, teach, quiet, abroad = rows(("islands", "space"))
    D.clear()
    out = {}
    for n in teach:
        train = [m for m in T if m != n]
        us = [m for m in train if m in teach + quiet]
        out[n] = {"models": {w: fit_rows(T, W, train, w) for w in ("aef", "tes", "both")},
                  "bounds": tes_gate_bounds(T, {m: np.arange(len(T[m]["y"])) for m in us}, us)}
        print(f"models without {n}", flush=True)
    os.makedirs(DIR, exist_ok=True)
    pickle.dump(out, open(path, "wb"))
    return out


def score():
    from aggregate import base_scores
    from models import SCORE_DIRS, cutoffs
    from quick import RULES
    M = held_out_models()
    cut = cutoffs(SCORE_DIRS["abroad"])[0.6]
    res = {}
    for n in DIG:
        res[n] = {}
        for k, (yb, ya) in pairs(n).items():
            if any(y not in meta(n)["aef"] and not os.path.exists(f"{DIR}/{n}-{y}.npz") for y in (yb, ya)):
                continue
            d = site_pair(n, yb, ya)
            li = np.flatnonzero(d["ok"] & (d["y"] >= 0))
            y = d["y"][li]
            s = base_scores(M[n]["models"], d, li)
            row = {"years": [yb, ya], "moved_px": int((y == 1).sum()), "radar_flip": round(float(s["flip"].mean()), 3)}
            for w in SHOW:
                v = RULES[w](s, M[n]["bounds"])
                row[w] = {"ap": round(float(average_precision_score(y, v)), 3),
                          "found": round(float((v[y == 1] > cut[w]).mean()), 3),
                          "lit": round(float((v[y == 0] > cut[w]).mean()), 4)}
            if k == "native":  # the same pixels and models as abroad_teach.py: its saved scores should come back
                z = np.load(f"{SCORE_DIRS['abroad']}/{n}.npz")
                row["check_max_diff"] = float(np.abs(z["both"] - s["both"]).max()) if len(z["both"]) == len(li) else None
            res[n][k] = row
            print(f"{n:18s} {k:8s} {yb}-{ya}  " + "  ".join(
                f"{w} AP {row[w]['ap']:.3f} found {row[w]['found']:.2f} lit {row[w]['lit']:.3f}" for w in SHOW), flush=True)
    print("\nmean over the sites that have each pair; paired with the same sites' native pair")
    for k in SHIFTS:
        ns = [n for n in DIG if k in res[n]]
        if not ns or k == "native":
            continue
        line = []
        for w in SHOW:
            f = np.array([res[n][k][w]["found"] for n in ns]) - np.array([res[n]["native"][w]["found"] for n in ns])
            ap = np.array([res[n][k][w]["ap"] for n in ns]) - np.array([res[n]["native"][w]["ap"] for n in ns])
            lit = [res[n][k][w]["lit"] for n in ns]
            lit0 = [res[n]["native"][w]["lit"] for n in ns]
            line.append(f"{w}: found {np.mean([res[n][k][w]['found'] for n in ns]):.2f} ({f.mean():+.2f}), "
                        f"AP {np.mean([res[n][k][w]['ap'] for n in ns]):.3f} ({ap.mean():+.3f}), lit {np.mean(lit):.3f} (native {np.mean(lit0):.3f})")
        print(f"  {k:8s} n={len(ns):2d}  " + " | ".join(line))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, "results", "spans.json"), "w"), indent=1)


def main():
    if sys.argv[1:2] == ["read"]:
        import zarr
        from obstore.store import HTTPStore
        from zarr.storage import ObjectStore
        zarr.config.set({"async.concurrency": 128})
        g = ObjectStore(HTTPStore.from_url(TESSERA), read_only=True)
        for n in sys.argv[2:] or DIG:
            read(g, n)
    elif sys.argv[1:2] == ["models"]:
        held_out_models()
    else:
        score()


if __name__ == "__main__":
    main()
