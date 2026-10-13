"""One score from both: ways to combine the AlphaEarth and TESSERA models, compared at the same catch rate.

The two models' scores run on different scales (TESSERA's runs hotter), so a fixed .5 cutoff favors whichever
scores higher. Here every rule gets its own cutoff, set so that on held-out digging sites it finds the same
share of moved ground (averaged over sites, each site counting once). At that cutoff: how much unchanged
ground lights up at the digging sites, on burns and dunes, and on the test-only ground (storm coast, farms,
marshes, crop switches), and how much of the farms' and marshes' own moved ground is still found.

Rules
  aef_gated   the map today: AlphaEarth model, quieted where AlphaEarth barely changed
  tes         TESSERA model
  tes_gated   TESSERA model with its own gate, calibrated so the same share of unchanged teaching ground
              falls under it as under AlphaEarth's
  both        one model on both embeddings; both_gated adds AlphaEarth's gate
  min         the lower of aef_gated and tes ("agree" in compare.py); min_gated uses tes_gated
  mean, geo   the average and the geometric mean of aef_gated and tes
  stack_lr    a logistic regression on the two scores and both plain changes
  stack_mono  boosted trees on the same four numbers, constrained so a higher score or more change
              never lowers the result
  both_geo, both_mean   the joint model held to AlphaEarth's gated score: geometric mean, average
  both_radar  the joint model, but AlphaEarth gated alone where TESSERA had radar in only one of the years
  stack_gated stack_mono times AlphaEarth's gate
  both_either the joint model times a gate that opens when EITHER embedding changed (the larger of AlphaEarth's
              gate and TESSERA's): digging, crops and burns change both and pass; ground where neither changed,
              which a model can mistake for graded ground (bare desert), is muted
  both_tgate  the joint model times TESSERA's gate alone
  stack_either, cover_either   stack_mono and stack_cover times the either gate
  stack_cover boosted trees on the four numbers plus AlphaEarth's before-year embedding (what the ground was),
              so it can learn where to trust each model less; monotone in the four, free in the 64
Also: the share of moved ground found by patch size, how many digging sites each rule beats the map's score on,
and the ground AlphaEarth sees but TESSERA left out (too few clear Sentinel-2 looks), scored by AlphaEarth alone.

The stacks learn only from digging sites (moved and unchanged) and burns and dunes (unchanged); crops,
farms and marshes stay test only. A digging site's stack learns from the other sites' held-out scores; those
scores came from models that saw the site in question, a second-order leak small next to the site itself.
Pixels where TESSERA has no embedding are left out here, as in compare.py.

Run: uv run python aggregate.py   (writes results/aggregate.json; ABROAD=0 leaves out the sites outside the 48 states;
     SPACE=1 adds Earthwork's satellite-laser sites, once tessera_sites.py space has read them)
"""

import json
import os

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

import compare
from compare import QUIET_KINDS, TEACH_KINDS, feats, gate, labeled, load, r3, sample
from tessera_sites import OUT, SITES, abroad_sites, cdl_sets, place_names, space_names

HERE = os.path.dirname(os.path.abspath(__file__))
SCORES = os.path.join(HERE, "data", "scores")  # SAVE_SCORES=1: every place's base scores, for quick.py
RULES = ("aef_gated", "tes", "tes_gated", "both", "both_gated", "min", "min_gated", "mean", "geo", "stack_lr", "stack_mono",
         "both_geo", "both_mean", "both_radar", "stack_gated", "stack_cover", "both_either", "both_tgate",
         "stack_either", "cover_either")
SHOW = ("aef_gated", "tes", "both", "mean", "both_mean", "stack_cover", "both_either", "both_tgate", "cover_either")
PATCH_HA = (0, 0.1, 1, 10, np.inf)
CATCH = (0.5, 0.6, 0.7)
SEED = int(os.environ.get("SEED", "0"))
RNG = np.random.default_rng(SEED + 1)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def tes_gate_bounds(D, idx, train):
    """TESSERA plain-change bounds that leave the same share of unchanged teaching ground under them as 0.05 and
    0.15 do for AlphaEarth's."""
    a = np.concatenate([D[n]["aplain"][idx[n][D[n]["y"][idx[n]] == 0]] for n in train])
    t = np.concatenate([D[n]["tplain"][idx[n][D[n]["y"][idx[n]] == 0]] for n in train])
    return tuple(float(np.quantile(t, (a < c).mean())) for c in (0.05, 0.15))


def tgate(p, plain, lo, hi):
    return p * np.clip((plain - lo) / max(hi - lo, 1e-6), 0, 1)


def stack_x(s, cover=False):
    x = np.stack([logit(s["aef"]), logit(s["tes"]), s["aplain"], s["tplain"]], 1)
    return np.concatenate([x, s["cover"].astype(np.float32)], 1) if cover else x


def base_scores(models, d, li):
    s = {w: models[w].predict_proba(feats(d, li, w))[:, 1].astype(np.float32) for w in models}
    s["aplain"], s["tplain"], s["flip"] = d["aplain"][li], d["tplain"][li], d["radar_flip"][li]
    s["cover"] = d["ab"][li]
    return s


def rules(s, bounds, stacks):
    ag = gate(s["aef"], s["aplain"])
    tg = tgate(s["tes"], s["tplain"], *bounds)
    r = {"aef_gated": ag, "tes": s["tes"], "tes_gated": tg, "min": np.minimum(ag, s["tes"]),
         "min_gated": np.minimum(ag, tg), "mean": (ag + s["tes"]) / 2, "geo": np.sqrt(ag * s["tes"])}
    if "both" in s:
        r["both"], r["both_gated"] = s["both"], gate(s["both"], s["aplain"])
        r["both_geo"], r["both_mean"] = np.sqrt(s["both"] * ag), (s["both"] + ag) / 2
        r["both_radar"] = np.where(s["flip"], ag, s["both"])
        ge = np.maximum(np.clip((s["aplain"] - 0.05) / 0.10, 0, 1), np.clip((s["tplain"] - bounds[0]) / max(bounds[1] - bounds[0], 1e-6), 0, 1))
        r["both_either"], r["both_tgate"] = s["both"] * ge, tgate(s["both"], s["tplain"], *bounds)
    if stacks:
        X = stack_x(s)
        r["stack_lr"], r["stack_mono"] = (stacks[k].predict_proba(X)[:, 1] for k in ("lr", "mono"))
        r["stack_gated"] = gate(r["stack_mono"], s["aplain"])
        r["stack_cover"] = np.concatenate([stacks["cover"].predict_proba(stack_x({k: v[i:i + 200000] for k, v in s.items()}, True))[:, 1]
                                           for i in range(0, len(s["aef"]), 200000)])
        if "both" in s:
            r["stack_either"], r["cover_either"] = r["stack_mono"] * ge, r["stack_cover"] * ge
    return {k: v.astype(np.float32) for k, v in r.items()}


def fit_stacks(H, names):
    """Stacks taught on held-out scores H[name] = (scores, y) at the given sites, balanced per site."""
    X, XC, y = [], [], []
    for n in names:
        s, yy = H[n]
        pos, neg = np.flatnonzero(yy == 1), np.flatnonzero(yy == 0)
        pick = np.concatenate([RNG.choice(pos, min(len(pos), 8000), replace=False),
                               RNG.choice(neg, min(len(neg), 40000), replace=False)])
        X.append(stack_x({k: s[k][pick] for k in ("aef", "tes", "aplain", "tplain")}))
        XC.append(s["cover"][pick])
        y.append((yy[pick] == 1).astype(int))
    X, y = np.concatenate(X), np.concatenate(y)
    XC = np.concatenate([X, np.concatenate(XC).astype(np.float32)], 1)
    return {"cover": HistGradientBoostingClassifier(max_iter=300, max_leaf_nodes=15, learning_rate=0.05,
                                                    monotonic_cst=[1, 1, 1, 1] + [0] * 64, class_weight="balanced",
                                                    l2_regularization=1.0, random_state=0).fit(XC, y),"lr": LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(X, y),
            "mono": HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=15, learning_rate=0.05,
                                                   monotonic_cst=[1, 1, 1, 1], class_weight="balanced",
                                                   random_state=0).fit(X, y)}


def cutoff(site_moved, target):
    """The cutoff at which the mean over sites of the share of moved ground scored above it is target."""
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if np.mean([(m > mid).mean() for m in site_moved]) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def save_scores(n, kind, s, y, role, bounds, patch=None):
    os.makedirs(SCORES, exist_ok=True)
    keep = ("aef", "tes", "both", "aplain", "tplain", "flip")
    np.savez(f"{SCORES}/{n}.npz", y=y, kind=kind, role=role, bounds=np.array(bounds),
             patch=patch if patch is not None else np.zeros(0, np.float32), **{k: s[k] for k in keep})


def extra(res, out, R, H, P, M, T, K, FLIP, teach, test):
    """At one catch rate: found by patch size, per-site wins over the map's score, and ground TESSERA left out."""
    print("  moved ground found, by patch size (held-out digging sites, pooled)")
    bins = list(zip(PATCH_HA[:-1], PATCH_HA[1:]))
    print(f"  {'rule':11s}" + "".join(f"{f'{a:g}-{b:g} ha':>12s}" for a, b in bins))
    res["patch"] = {}
    for w in SHOW:
        row = []
        for a, b in bins:
            hit = np.concatenate([R[n][w][(H[n][1] == 1) & (P[n] >= a) & (P[n] < b)] > out[w]["cutoff"] for n in teach])
            row.append(r3(hit.mean()) if len(hit) else float("nan"))
        res["patch"][w] = row
        print(f"  {w:11s}" + "".join(f"{v:12.3f}" for v in row))
    n_bin = [int(sum(((H[n][1] == 1) & (P[n] >= a) & (P[n] < b)).sum() for n in teach)) for a, b in bins]
    print(f"  {'moved px':11s}" + "".join(f"{v:12d}" for v in n_bin) + "\n")

    print("  digging sites (of 14) where the rule ranks better than the map's score (AP), and its worst loss")
    res["wins"] = {}
    base = {n: average_precision_score(H[n][1], R[n]["aef_gated"]) for n in teach}
    for w in SHOW[1:]:
        d = {n: average_precision_score(H[n][1], R[n][w]) - base[n] for n in teach}
        worst = min(d, key=d.get)
        res["wins"][w] = {"wins": sum(v > 0 for v in d.values()), "worst": [worst, r3(d[worst])]}
        print(f"  {w:11s} {res['wins'][w]['wins']:2d}   worst {worst} {d[worst]:+.3f}")

    t = out["aef_gated"]["cutoff"]
    print("\n  ground TESSERA left out (too few clear looks), scored by AlphaEarth alone at the map's cutoff")
    res["tes_missing"] = {}
    for n in teach:
        s, y, n_all = M[n]
        if len(y) == 0:
            continue
        share = len(y) / n_all
        f = (s[y == 1] > t).mean() if (y == 1).sum() >= 50 else float("nan")
        fa = (s[y == 0] > t).mean() if (y == 0).sum() >= 50 else float("nan")
        res["tes_missing"][n] = {"share": r3(share), "found": r3(f), "false_alarm": r3(fa)}
        if share > 0.01:
            print(f"  {n:20s} {share:5.1%} of labeled ground   moved found {f:.2f}   unchanged lit {fa:.3f}")
    print()


def main():
    abroad = abroad_sites() if os.environ.get("ABROAD", "1") == "1" else []
    space = space_names() if os.environ.get("SPACE", "0") == "1" else []
    names = [n for n in SITES + [s[0] for s in cdl_sets()] + abroad + place_names() + space if os.path.exists(f"{OUT}/{n}.npz")]
    # the teaching sites stay loaded; each test place is loaded, scored and let go (the islands are large)
    D = {n: load(n) for n in names if n in SITES}
    lidar = [n for n in names if n in SITES]
    teach = [n for n in lidar if D[n]["kind"] in TEACH_KINDS]
    quiet = [n for n in lidar if D[n]["kind"] in QUIET_KINDS]
    test = [n for n in names if n not in teach + quiet]
    compare.RNG = np.random.default_rng(SEED)
    idx = {n: sample(D[n], quiet=n in quiet) for n in teach + quiet}

    def fit(train, which):
        X = np.concatenate([feats(D[n], idx[n], which) for n in train])
        y = np.concatenate([(D[n]["y"][idx[n]] == 1).astype(int) for n in train])
        return LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced").fit(X, y)

    # held-out base scores at every teaching site
    H, B, P, M = {}, {}, {}, {}
    for n in teach + quiet:
        train = [m for m in teach + quiet if m != n]
        models = {w: fit(train, w) for w in ("aef", "tes", "both")}
        B[n] = tes_gate_bounds(D, idx, train)
        li = labeled(D[n])
        H[n], P[n] = (base_scores(models, D[n], li), D[n]["y"][li]), D[n]["patch"][li]
        # labeled ground AlphaEarth sees and TESSERA left out
        lm = np.flatnonzero(D[n]["aok"] & ~D[n]["ok"] & (D[n]["y"] >= 0))
        mv = models["aef"].predict_proba(feats(D[n], lm, "aef"))[:, 1] if len(lm) else np.zeros(0)
        M[n] = (gate(mv, 1 - (D[n]["ab"][lm].astype(np.float32) * D[n]["aa"][lm].astype(np.float32)).sum(1)),
                D[n]["y"][lm], int((D[n]["aok"] & (D[n]["y"] >= 0)).sum()))
        print(f"held out {n}: TESSERA gate {B[n][0]:.3f} to {B[n][1]:.3f}", flush=True)

    if os.environ.get("SAVE_SCORES") == "1":
        for n in teach + quiet:
            save_scores(n, D[n]["kind"], H[n][0], H[n][1], "teach" if n in teach else "quiet", B[n], P[n])

    # every rule at every teaching site, the stacks taught without the site
    R = {}
    for n in teach + quiet:
        stacks = fit_stacks(H, [m for m in teach + quiet if m != n])
        R[n] = rules(H[n][0], B[n], stacks)

    # test-only ground, scored by models and stacks taught on every teaching site
    models = {w: fit(teach + quiet, w) for w in ("aef", "tes", "both")}
    bounds = tes_gate_bounds(D, idx, teach + quiet)
    stacks = fit_stacks(H, teach + quiet)
    T = {}
    K, FLIP = {}, {}
    for n in test:
        d = D[n] if n in D else load(n)
        li = labeled(d)
        K[n], FLIP[n] = d["kind"], float(d["radar_flip"][d["ok"]].mean())
        bs = base_scores(models, d, li)
        if os.environ.get("SAVE_SCORES") == "1":
            save_scores(n, d["kind"], bs, d["y"][li], "test", bounds)
        T[n] = (rules(bs, bounds, stacks), d["y"][li])
        D.pop(n, None)
        del d
    print(f"test only: TESSERA gate {bounds[0]:.3f} to {bounds[1]:.3f}\n", flush=True)

    res = {"catch": {}, "ap_digging": {}, "tes_gate_test": bounds}
    for w in RULES:
        res["ap_digging"][w] = r3(np.mean([average_precision_score(H[n][1], R[n][w]) for n in teach]))
    kinds = ["dig", "burn", "dunes"] + sorted({K[n] for n in test})
    for c in CATCH:
        out = {}
        for w in RULES:
            t = cutoff([R[n][w][H[n][1] == 1] for n in teach], c)
            fa = {"dig": [(R[n][w][H[n][1] == 0] > t).mean() for n in teach]}
            for n in quiet:
                fa.setdefault(D[n]["kind"], []).append((R[n][w][H[n][1] == 0] > t).mean())
            found = {}
            for n in test:
                s, y = T[n]
                fa.setdefault(K[n], []).append((s[w][y == 0] > t).mean())
                if (y == 1).sum() >= (10 if K[n].startswith("space") else 50):
                    found.setdefault(K[n], []).append((s[w][y == 1] > t).mean())
            out[w] = {"cutoff": r3(t), "false_alarm": {k: r3(np.mean(v)) for k, v in fa.items()},
                      "found_test": {k: r3(np.mean(v)) for k, v in found.items()},
                      "crop_sites": {n: r3((T[n][0][w][T[n][1] == 0] > t).mean()) for n in test if K[n] == "crop"}}
        res["catch"][str(c)] = out
        print(f"finding {c:.0%} of moved ground at held-out digging sites; share of UNCHANGED ground lit up "
              f"(and of farm/marsh MOVED ground found):")
        print(f"  {'rule':11s} {'cutoff':>6s} {'AP':>5s}  " + "  ".join(f"{k:>6s}" for k in kinds) + "  | found: farm  marsh  storm  island  space")
        for w in RULES:
            o = out[w]
            fa = "  ".join(f"{o['false_alarm'].get(k, float('nan')):6.3f}" for k in kinds)
            fd = "  ".join(f"{o['found_test'].get(k, float('nan')):5.2f}" for k in ("farm", "marsh", "storm", "island", "space_dig"))
            print(f"  {w:11s} {o['cutoff']:6.3f} {res['ap_digging'][w]:5.3f}  {fa}  |        {fd}")
        print()
        if c == 0.6:
            extra(res, out, R, H, P, M, T, K, FLIP, teach, test)
        for k, label in (("island", "island moved over 2 m found"), ("space_dig", "satellite-laser digging sites: moved found"),
                         ("space_quiet", "satellite-laser quiet sites: unchanged lit up"),
                         ("notruth", "no truth, AlphaEarth 2023 vs 2025: share lit up")):
            sites = [n for n in test if K[n] == k]
            if not sites:
                continue
            show = SHOW
            print(f"  {label}\n  {'':24s}" + "".join(f"{w:>11s}" for w in show) + "   TESSERA radar one year only")
            for n in sites:
                s_, y = T[n]
                sel = y == (1 if k in ("island", "space_dig") else 0)
                if sel.sum() < (10 if k.startswith("space") else 50):
                    continue
                v = {w: r3((s_[w][sel] > out[w]["cutoff"]).mean()) for w in show}
                out.setdefault(k, {})[n] = v
                flip = FLIP[n]
                print(f"  {n:24s}" + "".join(f"{v[w]:11.3f}" for w in show) + f"   {flip:.2f}")
            print()

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, "results", f"aggregate-seed{SEED}.json" if SEED else "aggregate.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
