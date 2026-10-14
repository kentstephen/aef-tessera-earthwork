"""How much the combining results can carry: the gains tested across sites, not seeds.

From the held-out scores aggregate.py (data/scores) or abroad_teach.py (data/scores-abroad) saved:

1. Ranking. Each digging site's AP under each rule, paired with the map today's (aef_gated) at the same site:
   the mean gain with a bootstrap interval over sites (the seeds only changed the models' fit, never which
   sites there are), the sign test and the Wilcoxon signed-rank test, and building sites apart from mines.
2. False alarms at the matched cutoff, with bootstrap intervals over places (crop sets, farms, marshes), and
   each rule's difference from the map today paired place by place.
3. One cutoff for the map. Each digging site's cutoff set on the other 13 (60% caught), then at that site:
   moved found and unchanged lit. The spread is what one fixed cutoff does site to site.
4. Digging on farms and in marshes: the moved ground there found at the cutoff, next to the fields' false alarms.

Run: uv run python checks.py [scores dir ...]   (default data/scores data/scores-abroad; writes results/checks.json)
"""

import json
import os
import sys

import numpy as np
from scipy.stats import binomtest, wilcoxon
from sklearn.metrics import average_precision_score

from aggregate import cutoff
from compare import r3
from quick import RULES

HERE = os.path.dirname(os.path.abspath(__file__))
SHOW = ("aef_gated", "mean", "both", "both_either", "either_sqrt")
CATCH = 0.6
RNG = np.random.default_rng(0)
NBOOT = 10000


def boot(v, n=NBOOT):
    """The mean of v and its 95% bootstrap interval, resampling the entries (sites or places)."""
    v = np.asarray(v, float)
    m = v[RNG.integers(0, len(v), (n, len(v)))].mean(1)
    return r3(v.mean()), r3(np.quantile(m, 0.025)), r3(np.quantile(m, 0.975))


def main(dirs):
    res = {}
    for sd in dirs:
        Z = {f[:-4]: dict(np.load(f"{sd}/{f}", allow_pickle=True)) for f in sorted(os.listdir(sd)) if f.endswith(".npz")}
        teach = [n for n, z in Z.items() if str(z["role"]) == "teach"]
        S = {n: {w: RULES[w](z, tuple(z["bounds"])).astype(np.float32) for w in SHOW} for n, z in Z.items()}
        out = res[sd] = {}
        print(f"\n===== {sd}: {len(teach)} digging sites =====")

        # 1. ranking, paired by site
        ap = {w: np.array([average_precision_score(Z[n]["y"], S[n][w]) for n in teach]) for w in SHOW}
        mine = np.array([str(Z[n]["kind"]) == "mining" for n in teach])
        print("\n1. AP gain over the map today, paired by site (bootstrap 95% over sites)")
        print(f"  {'rule':12s} {'mean AP':>7s}  {'gain':>6s} {'95% interval':>16s}  wins  sign p  wilcoxon p | building gain  mining gain (n={mine.sum()})")
        out["ap"] = {"sites": teach, "per_site": {w: [r3(v) for v in ap[w]] for w in SHOW}}
        for w in SHOW[1:]:
            d = ap[w] - ap["aef_gated"]
            g, lo, hi = boot(d)
            wins = int((d > 0).sum())
            ps = binomtest(wins, len(d)).pvalue
            pw = wilcoxon(d).pvalue
            gb, gm = boot(d[~mine]), boot(d[mine])
            out["ap"][w] = {"gain": [g, lo, hi], "wins": wins, "sign_p": r3(ps), "wilcoxon_p": r3(pw),
                            "building": gb, "mining": gm, "mining_sites": {n: r3(v) for n, v in zip(np.array(teach)[mine], d[mine])}}
            print(f"  {w:12s} {ap[w].mean():7.3f}  {g:+.3f} [{lo:+.3f}, {hi:+.3f}]  {wins:2d}/{len(d)}  {ps:6.3f}  {pw:10.3f} | "
                  f"{gb[0]:+.3f} [{gb[1]:+.3f},{gb[2]:+.3f}]  {gm[0]:+.3f} [{gm[1]:+.3f},{gm[2]:+.3f}]")
        print("  mining sites, gain per site: " + ", ".join(
            f"{n} " + "/".join(f"{ap[w][i] - ap['aef_gated'][i]:+.3f}" for w in ("both", "both_either"))
            for i, n in enumerate(teach) if mine[i]) + "  (both / both_either)")
        print("  mining AP, map today: " + ", ".join(f"{n} {ap['aef_gated'][i]:.3f}" for i, n in enumerate(teach) if mine[i]))

        # 2. false alarms at the matched cutoff, intervals over places
        cut = {w: cutoff([S[n][w][Z[n]["y"] == 1] for n in teach], CATCH) for w in SHOW}
        kinds = ("crop", "farm", "marsh", "burn", "storm", "island", "notruth", "space_quiet")
        print(f"\n2. unchanged ground lit at {CATCH:.0%} caught: mean [95% over places]; and paired difference from the map today")
        out["false_alarm"] = {}
        for k in kinds:
            names = [n for n, z in Z.items() if str(z["kind"]) == k and (z["y"] == 0).sum() >= 10]
            if not names:
                continue
            lit = {w: np.array([(S[n][w][Z[n]["y"] == 0] > cut[w]).mean() for n in names]) for w in SHOW}
            out["false_alarm"][k] = {w: {"mean": boot(lit[w]), "vs_map": boot(lit[w] - lit["aef_gated"]),
                                         "fewer_than_map": int((lit[w] < lit["aef_gated"]).sum())} for w in SHOW}
            out["false_alarm"][k]["n"] = len(names)
            row = "  ".join(f"{w} {b['mean'][0]:.3f} [{b['mean'][1]:.3f},{b['mean'][2]:.3f}]"
                            + ("" if w == "aef_gated" else f" d{b['vs_map'][0]:+.3f} [{b['vs_map'][1]:+.3f},{b['vs_map'][2]:+.3f}] {b['fewer_than_map']}/{len(names)}")
                            for w, b in out["false_alarm"][k].items() if w != "n")
            print(f"  {k:11s} n={len(names):2d}  {row}")

        # 3. one cutoff: each site's set on the others
        print(f"\n3. each digging site at a cutoff set on the other {len(teach) - 1} ({CATCH:.0%} caught): moved found, unchanged lit")
        out["loso_cutoff"] = {}
        for w in SHOW:
            f, fa, cs = [], [], []
            for n in teach:
                c = cutoff([S[m][w][Z[m]["y"] == 1] for m in teach if m != n], CATCH)
                cs.append(c)
                f.append((S[n][w][Z[n]["y"] == 1] > c).mean())
                fa.append((S[n][w][Z[n]["y"] == 0] > c).mean())
            f, fa = np.array(f), np.array(fa)
            out["loso_cutoff"][w] = {"cutoff_range": [r3(min(cs)), r3(max(cs))], "found": {n: r3(v) for n, v in zip(teach, f)},
                                     "lit": {n: r3(v) for n, v in zip(teach, fa)}}
            print(f"  {w:12s} cutoff {min(cs):.3f}-{max(cs):.3f}  found min {f.min():.2f} ({teach[f.argmin()]}) median {np.median(f):.2f} "
                  f"max {f.max():.2f} ({teach[f.argmax()]}), sd {f.std():.2f} | lit median {np.median(fa):.3f} max {fa.max():.3f} ({teach[fa.argmax()]})")

        # 4. digging on farms and in marshes
        print(f"\n4. moved ground found on farm and marsh lidar (test only), at {CATCH:.0%} caught")
        out["field_digging"] = {}
        for n, z in Z.items():
            if str(z["kind"]) in ("farm", "marsh") and (z["y"] == 1).sum() >= 50:
                v = {w: r3((S[n][w][z["y"] == 1] > cut[w]).mean()) for w in SHOW}
                u = {w: r3((S[n][w][z["y"] == 0] > cut[w]).mean()) for w in SHOW}
                out["field_digging"][n] = {"found": v, "lit": u, "moved_px": int((z["y"] == 1).sum())}
                print(f"  {n:22s} {str(z['kind']):6s} moved px {int((z['y'] == 1).sum()):6d}  found " + "  ".join(f"{w} {v[w]:.2f}" for w in SHOW)
                      + "  | lit " + " ".join(f"{u[w]:.3f}" for w in SHOW))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, "results", "checks.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:] or ["data/scores", "data/scores-abroad"])
