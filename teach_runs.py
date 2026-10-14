"""Teaching recipes side by side, from the held-out scores each one saved (fire_teach.py, abroad_teach.py).

For each scores directory and rule, at the cutoff that finds CATCH of moved ground over the held-out digging
sites: mean AP there and its gain over the map today at the same sites (bootstrap over sites), unchanged ground
lit by kind (lidar burns, burned forest, clearcuts, storm coast, crops, farms), digging found abroad (satellite
labels, islands), and the sites worth watching one by one: the mines in forest (Kolwezi, Hibbing) and the
leveled farms.

Run: uv run python teach_runs.py <scores dir> ...   (adds each directory to results/teach_runs.json)
"""

import json
import os
import sys

import numpy as np
from sklearn.metrics import average_precision_score

from aggregate import cutoff
from checks import boot
from compare import r3
from quick import RULES

HERE = os.path.dirname(os.path.abspath(__file__))
CATCH = 0.6
SHOW = ("aef_gated", "both", "both_either", "either_radar")
KINDS = ("burn", "fire", "cut", "storm", "crop", "farm")
WATCH = ("space-kolwezi-cd", "hibbing-mn", "four-corners-fl", "mountain-pass-ca")
FARMS = ("fresno-farm-ca", "arkansas-delta-ar", "antelope-farm-ne", "hamilton-farm-in")


def run(sd):
    Z = {f[:-4]: dict(np.load(f"{sd}/{f}", allow_pickle=True)) for f in sorted(os.listdir(sd)) if f.endswith(".npz")}
    teach = [n for n, z in Z.items() if str(z["role"]) == "teach"]
    S = {n: {w: RULES[w](z, tuple(z["bounds"])).astype(np.float32) for w in SHOW} for n, z in Z.items()}
    ap = {w: np.array([average_precision_score(Z[n]["y"], S[n][w]) for n in teach]) for w in SHOW}
    out = {}
    for w in SHOW:
        t = cutoff([S[n][w][Z[n]["y"] == 1] for n in teach], CATCH)
        lit = lambda n: float((S[n][w][Z[n]["y"] == 0] > t).mean())
        found = lambda n: float((S[n][w][Z[n]["y"] == 1] > t).mean())
        r = {"ap": r3(ap[w].mean()), "gain": boot(ap[w] - ap["aef_gated"]) if w != "aef_gated" else None}
        for k in KINDS:
            ns = [n for n, z in Z.items() if n not in teach and str(z["kind"]) == k and (z["y"] == 0).sum() >= 10]
            r[k] = r3(np.mean([lit(n) for n in ns])) if ns else None
            r[f"{k}_roles"] = sorted({str(Z[n]["role"]) for n in ns})
        for k in ("space_dig", "island"):
            ns = [n for n, z in Z.items() if str(z["kind"]) == k and (z["y"] == 1).sum() >= 10]
            r[f"{k}_found"] = r3(np.mean([found(n) for n in ns])) if ns else None
        r["watch"] = {n: {"found": r3(found(n)), "ap": r3(average_precision_score(Z[n]["y"], S[n][w])), "role": str(Z[n]["role"])}
                      for n in WATCH if n in Z}
        r["farms"] = {n: {"found": r3(found(n)), "lit": r3(lit(n)), "role": str(Z[n]["role"])} for n in FARMS if n in Z}
        r["grizzly"] = r3(lit("grizzly-flats-ca")) if "grizzly-flats-ca" in Z else None
        out[w] = r
    return out


def main(dirs):
    res = {os.path.basename(sd.rstrip("/")): run(sd) for sd in dirs}
    for name, R in res.items():
        print(f"\n== {name}")
        for w, r in R.items():
            g = r["gain"]
            print(f"  {w:12s} AP {r['ap']:.3f}" + (f" gain {g[0]:+.3f} [{g[1]:+.3f},{g[2]:+.3f}]" if g else " " * 27)
                  + "  " + " ".join(f"{k} {r[k]:.3f}" if r[k] is not None else f"{k} -" for k in KINDS)
                  + f" | sat found {r['space_dig_found']} island {r['island_found']}")
            print(f"  {'':12s} " + "  ".join(f"{n.removeprefix('space-').split('-')[0]} found {v['found']:.2f} AP {v['ap']:.3f}"
                                            for n, v in r["watch"].items())
                  + "  | farms " + "  ".join(f"{n[:7]} found {v['found']:.2f} lit {v['lit']:.3f}" for n, v in r["farms"].items()))
    path = os.path.join(HERE, "results", "teach_runs.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    old = json.load(open(path)) if os.path.exists(path) else {}
    json.dump({**old, **res}, open(path, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
