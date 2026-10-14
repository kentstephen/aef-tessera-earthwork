"""A cutoff set from each place's own scores: does it even out how much digging one cutoff finds?

At one cutoff the share of moved ground found swings from under .2 (Four Corners, Hibbing) to over .9 (Katy)
under every rule (checks.py section 3): the dim sites score their digging low everywhere, the bright ones high.
Here each rule's score is first set against the place it sits in, then one cutoff is set as before (60% of moved
ground found, averaged over the held-out digging sites), and each digging site's own cutoff is set on the other
13. Ways to set a score against its place:

  place-wide   rank       the score's quantile among the place's scores (the cutoff lights the same share
                          of every place)
               z          the logit's z-score among the place's logits
               over q99   the score over the place's 99th percentile, the percentile held at least f
  local        contrast   the score less the mean score within about 1 km (101 px) or 3 km (301 px)
               ratio      the score over that local mean, the mean held at least 0.05

Part A works from the saved held-out scores (data/scores, data/scores-abroad, data/scores-fire-abroad), where a
place's scores are only its labeled pixels, so the place-wide rules see less than a map would. Part B rescores
every valid pixel of every full grid with the abroad recipe's models that did not see the place (each digging and
quiet site left out alone, the abroad sites in abroad_teach.py's folds, the rest by the final models), as a map
would hold them, and adds the local rules. Crop switches and satellite labels are not grids: their place is
their own pixels (sampled fields in a 6 km box; scattered laser labels), and the local mean is over the
neighbors among them, falling back to the place's mean where fewer than 20 are in the window.

A place-wide rule cannot change a site's AP (it keeps the site's order); only the local rules can.

Run: uv run python calib.py prep     (part B's scores into data/calib/, about half an hour)
     uv run python calib.py          (both parts; writes results/calib.json)
"""

import json
import os
import pickle
import sys

import numpy as np
from scipy.ndimage import uniform_filter
from sklearn.metrics import average_precision_score

from aggregate import cutoff
from quick import DESERT, RULES

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(HERE, "data", "calib")
SHOW = ("aef_gated", "both_either", "either_radar")
CATCH = 0.6
KINDS = ("crop", "farm", "marsh", "storm", "burn", "dunes", "fire", "cut", "lake", "island", "tropic", "hongkong",
         "shizuoka", "space_quiet")
WATCH = ("four-corners-fl", "hibbing-mn", "mountain-pass-ca", "katy-tx")


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sig(x):
    return 1 / (1 + np.exp(-x))


def local_mean(v, pos, shape, w, fallback):
    """The mean of v over the place's scored pixels within a w x w window of each one (pos: flat grid index)."""
    s, c = np.zeros(shape[0] * shape[1], np.float32), np.zeros(shape[0] * shape[1], np.float32)
    np.add.at(s, pos, v)
    np.add.at(c, pos, 1)
    s = uniform_filter(s.reshape(shape), w, mode="constant").ravel()[pos]
    c = uniform_filter(c.reshape(shape), w, mode="constant").ravel()[pos]
    n = c * w * w
    return np.where(n >= 20, s / np.maximum(c, 1e-9), fallback)


def norms(v, grid=None):
    """Every way of setting a place's scores v against the place, each in [0, 1]. grid = (pos, shape) adds the
    local ones."""
    L = logit(v)
    q99 = float(np.quantile(v, 0.99))
    out = {"global": v,
           "rank": np.searchsorted(np.sort(v), v, side="left") / len(v),
           "z": sig((L - L.mean()) / max(L.std(), 1e-6))}
    for f in (0.05, 0.2, 0.5):
        out[f"over_q99_f{f}"] = np.clip(v / max(q99, f), 0, 1)
    if grid is not None:
        pos, shape = grid
        for w, tag in ((101, "1km"), (301, "3km")):
            m = local_mean(v, pos, shape, w, float(v.mean()))
            out[f"contrast_{tag}"] = (v - m + 1) / 2
            out[f"ratio_{tag}"] = np.clip(v / np.maximum(m, 0.05) / 20, 0, 1)
    return {k: x.astype(np.float32) for k, x in out.items()}


def header(label, P):
    print(f"\n===== {label}: {sum(p['role'] == 'teach' for p in P.values())} digging sites, {len(P)} places =====")
    print(f"  {'rule':13s} {'norm':16s} {'AP':>5s}  found at own LOSO cutoff: min  med  max   sd  | "
          + " ".join(f"{s[:6]:>6s}" for s in WATCH) + " | " + " ".join(f"{k[:6]:>6s}" for k in KINDS)
          + " desert  isl-f spc-f")


def evaluate(P, w):
    """P[name] = {"kind", "role", "y", "S": {norm: scores}} under rule w: AP, found per site at the LOSO cutoff,
    and unchanged lit by kind at the 60% cutoff."""
    teach = [n for n, p in P.items() if p["role"] == "teach"]
    out = {}
    norm_names = list(P[teach[0]]["S"])
    for k in norm_names:
        if not all(k in P[n]["S"] for n in P):
            continue
        S = {n: P[n]["S"][k] for n in P}
        y = {n: P[n]["y"] for n in P}
        moved = {n: S[n][y[n] == 1] for n in teach}
        c = cutoff([moved[n] for n in teach], CATCH)
        ap = float(np.mean([average_precision_score(y[n][y[n] >= 0], S[n][y[n] >= 0]) for n in teach]))
        f = {}
        for n in teach:
            cn = cutoff([moved[m] for m in teach if m != n], CATCH)
            f[n] = float((moved[n] > cn).mean())
        fv = np.array(list(f.values()))
        lit = {}
        for kind in KINDS:
            v = [float((S[n][y[n] == 0] > c).mean()) for n in P if P[n]["kind"] == kind and (y[n] == 0).sum() >= 10]
            lit[kind] = float(np.mean(v)) if v else float("nan")
        des = float(np.mean([(S[n][y[n] >= 0] > c).mean() for n in DESERT if n in P]))
        fd = {kd: float(np.mean([(S[n][y[n] == 1] > c).mean() for n in P if P[n]["kind"] == kd and (y[n] == 1).sum() >= 10] or [np.nan]))
              for kd in ("island", "space_dig")}
        out[f"{w}:{k}"] = {"ap": round(ap, 3), "found": {n: round(v, 3) for n, v in f.items()},
                           "found_sd": round(float(fv.std()), 3), "lit": {kk: round(v, 4) for kk, v in lit.items()},
                           "desert": round(des, 4), "found_abroad": {kk: round(v, 3) for kk, v in fd.items()}}
        print(f"  {w:13s} {k:16s} {ap:5.3f}  {'':24s}{fv.min():.2f} {np.median(fv):.2f} {fv.max():.2f} {fv.std():.2f} | "
              + " ".join(f"{f.get(s, np.nan):6.2f}" for s in WATCH) + " | "
              + " ".join(f"{lit[kk]:6.3f}" for kk in KINDS) + f"  {des:.3f}  {fd['island']:.2f}  {fd['space_dig']:.2f}", flush=True)
    return out


def run(sd, label, local):
    """Every rule in SHOW under every norm, one rule at a time (a full set of grids under every norm is large)."""
    Z = {f[:-4]: dict(np.load(f"{sd}/{f}", allow_pickle=True)) for f in sorted(os.listdir(sd)) if f.endswith(".npz")}
    header(label, {n: {"role": str(z["role"])} for n, z in Z.items()})
    out = {}
    for w in SHOW:
        P = {n: {"kind": str(z["kind"]), "role": str(z["role"]), "y": z["y"],
                 "S": norms(RULES[w](z, tuple(z["bounds"])).astype(np.float32),
                            (z["pos"], tuple(z["shape"])) if local else None)} for n, z in Z.items()}
        out.update(evaluate(P, w))
    return out


# ---- part B: every valid pixel of every grid, abroad recipe ---------------------------------------------------

def models_for():
    """Models that did not see each place: digging sites (spans.py's), quiet sites and the abroad folds fitted
    here, the final models for the rest."""
    from spans import held_out_models
    path = f"{DIR}/models.pkl"
    if os.path.exists(path):
        out = pickle.load(open(path, "rb"))
        out["held"].update(held_out_models())
        return out
    from abroad_teach import FOLDS, fit_rows, rows
    from aggregate import tes_gate_bounds
    out = {"held": {}}
    T, W, D, teach, quiet, abroad = rows(("islands", "space"))
    D.clear()

    def fit(train):
        us = [m for m in train if m in teach + quiet]
        return {"models": {w: fit_rows(T, W, train, w) for w in ("aef", "tes", "both")},
                "bounds": tes_gate_bounds(T, {m: np.arange(len(T[m]["y"])) for m in us}, us)}
    for n in quiet:
        out["held"][n] = fit([m for m in T if m != n])
        print(f"models without {n}", flush=True)
    fold = {n: i % FOLDS for i, n in enumerate(sorted(abroad))}
    for k in range(FOLDS):
        m = fit([x for x in T if fold.get(x) != k])
        for n in abroad:
            if fold[n] == k:
                out["held"][n] = m
        print(f"models without fold {k}", flush=True)
    os.makedirs(DIR, exist_ok=True)
    pickle.dump(out, open(path, "wb"))
    import time
    while not os.path.exists(f"{HERE}/data/spans/models.pkl"):  # spans.py models, when it runs alongside
        time.sleep(30)
    out["held"].update(held_out_models())
    return out


def grid_of(n):
    """A place's grid shape from the TESSERA read's mosaic rows and columns."""
    from tessera_sites import OUT
    rc = json.loads(str(np.load(f"{OUT}/{n}.npz", allow_pickle=True)["meta"]))["rc"]
    return rc[1] - rc[0], rc[3] - rc[2]


def prep():
    import ground
    from aggregate import base_scores
    from compare import labeled, load
    from models import SCORE_DIRS, final
    M = models_for()
    fin = final("abroad")
    sd = SCORE_DIRS["abroad"]
    os.makedirs(DIR, exist_ok=True)
    for f in sorted(os.listdir(sd)):
        n = f[:-4]
        path = f"{DIR}/{n}.npz"
        if os.path.exists(path) or not f.endswith(".npz"):
            continue
        z = np.load(f"{sd}/{f}", allow_pickle=True)
        m = M["held"].get(n, fin)
        d = load(n)
        if n.startswith(("cdl-", "space-")):  # not grids: the place is its own labeled pixels
            li = labeled(d)
            if n.startswith("cdl-"):
                from tessera_sites import EARTHWORK
                idx = np.load(f"{EARTHWORK}/{n}.npz")["idx"][li]
                shape = grid_of(n)
                pos = idx
            else:
                rc = np.load(f"{HERE}/data/space/{n[len('space-'):]}.npz")["rc"][li]
                r0, c0 = rc[:, 0].min(), rc[:, 1].min()
                shape = (int(rc[:, 0].max() - r0 + 1), int(rc[:, 1].max() - c0 + 1))
                pos = (rc[:, 0] - r0) * shape[1] + (rc[:, 1] - c0)
            vi, y = li, np.asarray(z["y"])
        else:
            vi = np.flatnonzero(d["ok"])
            if n.startswith("ground-"):
                _, yl = ground.saved(n[len("ground-"):], d, vi)
            else:
                yl = d["y"][vi]
                if d["kind"] == "space_quiet":
                    yl = np.where(yl == 1, -1, yl)
            y = np.where(yl >= 0, yl, -1).astype(np.int8)
            shape, pos = grid_of(n), vi
        s = base_scores(m["models"], d, vi)
        # the labeled pixels should score as abroad_teach.py saved them
        lab = np.isin(vi, labeled(d)) if not n.startswith(("cdl-", "space-")) else np.ones(len(vi), bool)
        diff = float(np.abs(s["both"][lab] - z["both"]).max()) if lab.sum() == len(z["both"]) else -1.0
        np.savez(path, y=y, pos=pos.astype(np.int64), shape=np.array(shape), kind=str(z["kind"]), role=str(z["role"]),
                 bounds=np.array(m["bounds"]), check=diff, **{k: s[k] for k in ("aef", "tes", "both", "aplain", "tplain", "flip")})
        print(f"{n}: {len(vi)} px, labeled scores differ from saved by at most {diff:.2g}", flush=True)
        del d


def main():
    if sys.argv[1:2] == ["prep"]:
        prep()
        return
    res = {}
    for sd in ("data/scores", "data/scores-abroad", "data/scores-fire-abroad"):
        res[f"A:{sd}"] = run(os.path.join(HERE, sd), f"part A, {sd}, labeled pixels only", False)
    if os.path.isdir(DIR):
        res["B"] = run(DIR, "part B, abroad recipe, every valid pixel of each grid", True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, "results", "calib.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
