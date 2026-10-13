"""Why TESSERA lights up ground that did not move: what its plain change on unchanged ground follows.

For the unchanged ground of every place (the lidar's unchanged pixels; every valid pixel at a crop switch or a
place with no truth), TESSERA's plain change 1 - cos(b, a) against what could drive it: how many clear
Sentinel-2 looks each year had (TESSERA embeds a pixel only with 15 or more; 2017 is thin), radar in each
year, the year pair, and AlphaEarth's own plain change on the same pixels. Per place, then pooled.

Run: uv run python desert.py   (writes results/desert.json)
"""

import json
import os

import numpy as np
from scipy.stats import spearmanr

from compare import load, r3
from tessera_sites import OUT, SITES, abroad_sites, cdl_sets, place_names

HERE = os.path.dirname(os.path.abspath(__file__))
TES_HI = 0.267  # the top of TESSERA's calibrated gate (aggregate.py): change past it counts in full
RNG = np.random.default_rng(0)


def main():
    names = [n for n in SITES + [s[0] for s in cdl_sets()] + abroad_sites() + place_names() if os.path.exists(f"{OUT}/{n}.npz")]
    rows, pool = {}, {k: [] for k in ("tplain", "aplain", "s2b", "s2a", "s2min", "s2ratio", "s1b", "s1a", "yb", "ya")}
    print(f"{'place':34s} {'kind':9s} {'years':11s}  unchanged px  TES plain  AEF plain  TES > gate  S2 looks b / a   S1 b / a")
    for n in names:
        d = load(n)
        still = np.flatnonzero(d["ok"] & (d["y"] == 0))
        if len(still) < 200:
            continue
        t, a = d["tplain"][still], d["aplain"][still]
        s2b, s2a = d["s2b"][still].astype(float), d["s2a"][still].astype(float)
        r = {"kind": d["kind"], "years": d["years"], "n": int(len(still)), "tes_plain": r3(np.median(t)),
             "aef_plain": r3(np.median(a)), "tes_over_gate": r3((t > TES_HI).mean()),
             "s2": [r3(np.median(s2b)), r3(np.median(s2a))],
             "s1": [r3(np.median(d["s1b"][still])), r3(np.median(d["s1a"][still]))]}
        rows[n] = r
        if not n.startswith("cdl-") and not n.startswith("shizuoka-"):
            print(f"{n:34s} {d['kind']:9s} {str(d['years']):11s}  {len(still):12d}  {r['tes_plain']:9.3f}  {r['aef_plain']:9.3f}"
                  f"  {r['tes_over_gate']:10.3f}  {r['s2'][0]:6.0f} / {r['s2'][1]:<6.0f}  {r['s1'][0]:4.0f} / {r['s1'][1]:<4.0f}")
        pick = RNG.choice(len(still), min(len(still), 5000), replace=False)
        for k, v in (("tplain", t), ("aplain", a), ("s2b", s2b), ("s2a", s2a), ("s2min", np.minimum(s2b, s2a)),
                     ("s2ratio", np.abs(np.log((s2a + 1) / (s2b + 1)))), ("s1b", d["s1b"][still].astype(float)),
                     ("s1a", d["s1a"][still].astype(float)), ("yb", np.full(len(still), d["years"][0], float)),
                     ("ya", np.full(len(still), d["years"][1], float))):
            pool[k].append(v[pick])
        del d
    P = {k: np.concatenate(v) for k, v in pool.items()}
    print("\npooled unchanged ground (5,000 px per place): Spearman of TESSERA's plain change with")
    res = {"sites": rows, "pooled_rho": {}}
    for k in ("aplain", "s2b", "s2a", "s2min", "s2ratio", "s1b", "s1a", "yb", "ya"):
        res["pooled_rho"][k] = r3(spearmanr(P["tplain"], P[k])[0])
        print(f"  {k:8s} {res['pooled_rho'][k]:+.3f}")
    print("\nTESSERA plain change on unchanged ground by the before year (median, share past the gate):")
    for yb in sorted(set(P["yb"].astype(int))):
        m = P["yb"] == yb
        print(f"  {yb}: {np.median(P['tplain'][m]):.3f}  {(P['tplain'][m] > TES_HI).mean():.3f}   ({m.sum()} px)")
    print("by the clear Sentinel-2 looks in the thinner year:")
    for lo, hi in ((15, 25), (25, 40), (40, 60), (60, 1000)):
        m = (P["s2min"] >= lo) & (P["s2min"] < hi)
        if m.any():
            print(f"  {lo:3d}-{hi:<4d} {np.median(P['tplain'][m]):.3f}  {(P['tplain'][m] > TES_HI).mean():.3f}   ({m.sum()} px)")
    json.dump(res, open(os.path.join(HERE, "results", "desert.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
