"""separate.py's test: can a head on the embeddings tell cleared-and-dug ground from cleared-and-not-dug?

Feature sets, each fed to one logistic regression (scaled, C=0.1, each site weighted the same and its two groups
balanced):
  pair   the map's own features at the label's two years, both embeddings: [b, a, a*b, (a-b)^2]
  ctx    the surroundings at the two years (mean within about 110 m and 510 m): the change there and what it became
  after  the last year read (2025), against the label's later year: what the ground became since
  traj   twelve numbers per pixel over every year: how far it went from b, how far it is from b and a at the
         end, how much of its change it kept, and the year it went farthest
  all    pair, ctx and after together
The aggregate's own score (joint model x radar-safe gate, abroad recipe, held out) is the baseline: how well it
already orders the two groups.

  A  leave one lidar site out: AUC of cleared and dug against cleared and not dug at the held-out site
  B  taught on every lidar site, scored on the clearcut regions and the satellite-laser mines: does it call the
     cuts not dug and the mines dug?
  C  the satellite mines and clearcut regions taught too, each held out in turn: the head a map would use
"""

import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from separate import CUTS, DIR, HERE, LIDAR, NEAR, YEARS, near_name, space_list

SETS = ("pair", "ctx", "after", "traj", "all")
DIG_SITES = {"hibbing-mn", "lakewood-ranch-fl", "four-corners-fl", "huntsville-al", "new-albany-oh"}


def f32(v):
    return v.astype(np.float32)


def cos_d(u, v):
    return 1 - (u * v).sum(-1)


def build(n):
    """Every feature set for one place's kept pixels, the valid mask (both embeddings at b, a and the last year,
    and a later year after a) and the groups."""
    z = np.load(f"{DIR}/{n}.npz")
    E, T = f32(z["aef"]), f32(z["tes"])
    ok = z["aok"] & z["tok"]
    n_ = E.shape[1]
    r = np.arange(n_)
    bi, ai = z["b"].astype(int) - YEARS[0], z["a"].astype(int) - YEARS[0]
    # the last year with both embeddings valid, after a
    later = ok & (np.arange(len(YEARS))[:, None] > ai[None])
    li = np.where(later.any(0), len(YEARS) - 1 - np.argmax(later[::-1], 0), -1)
    valid = ok[bi, r] & ok[ai, r] & (li >= 0)
    li = np.maximum(li, 0)
    out = {}
    pair, ctx, after, traj = [], [], [], []
    for V, C in ((E, f32(z["aef_ctx"])), (T, f32(z["tes_ctx"]))):
        b, a, L = V[bi, r], V[ai, r], V[li, r]
        pair += [b, a, a * b, (a - b) ** 2]
        for k in range(C.shape[1]):
            cb, ca = C[bi, k, r], C[ai, k, r]
            ctx += [ca - cb, ca]
        after += [L, (L - a) ** 2]
        d_ba, d_bL, d_aL = cos_d(b, a), cos_d(b, L), cos_d(a, L)
        dist = cos_d(V, b[None])  # (years, n): distance from b in every year
        span = (np.arange(len(YEARS))[:, None] >= bi[None]) & (np.arange(len(YEARS))[:, None] <= li[None]) & ok
        dmax = np.where(span, dist, -1).max(0)
        ymax = np.where(span, dist, -1).argmax(0) - bi
        # the surroundings: how much the 510 m around changed, and how much of it is still changed at the end
        cb, ca, cL = C[bi, -1, r], C[ai, -1, r], C[li, -1, r]
        traj += [np.stack([d_ba, d_bL, d_aL, d_bL / np.maximum(d_ba, 1e-3), dmax, ymax, li - ai,
                           np.linalg.norm(ca - cb, axis=1), np.linalg.norm(cL - cb, axis=1)], 1)]
    out["pair"], out["ctx"], out["after"] = (np.concatenate(v, 1) for v in (pair, ctx, after))
    out["traj"] = np.concatenate(traj, 1)
    out["all"] = np.concatenate([out["pair"], out["ctx"], out["after"], out["traj"]], 1)
    # the map's features at the two years, for the aggregate's score
    out["_ab"], out["_aa"], out["_tb"], out["_ta"] = E[bi, r], E[ai, r], T[bi, r], T[ai, r]
    s1 = z["s1"].astype(int)
    out["_flip"] = (s1[bi, r] > 0) != (s1[ai, r] > 0)
    out["_row"] = z["idx"]  # the label's row in the place's own arrays
    return {k: v[valid] for k, v in out.items()}, z["grp"][valid], str(z["source"])


def agg_score(F, n):
    """The aggregate (joint model x radar-safe gate, abroad recipe): held out at the five US digging sites
    (spans.py's models), the abroad recipe's final models elsewhere (it never taught the Kentucky lidar, the
    clearcut regions or forest_labels.py's mines; the older satellite sites were taught, so they are marked)."""
    from aggregate import base_scores
    from models import final
    from quick import RULES
    from tessera_sites import space_names
    if n in space_names():  # taught by the abroad recipe: its saved held-out scores (abroad_teach.py's folds)
        from compare import labeled, load
        z = dict(np.load(os.path.join(HERE, "data", "scores-abroad", f"{n}.npz"), allow_pickle=True))
        v = RULES["either_radar"](z, tuple(z["bounds"]))
        li = labeled(load(n))
        j = np.searchsorted(li, F["_row"])
        hit = (j < len(li)) & (li[np.minimum(j, len(li) - 1)] == F["_row"])
        return np.where(hit, v[np.minimum(j, len(li) - 1)], np.nan), final("abroad")["cutoff"][0.6]["either_radar"]
    if n in DIG_SITES:
        from spans import held_out_models
        m = held_out_models()[n]
    else:
        m = final("abroad")
    d = {"ab": F["_ab"], "aa": F["_aa"], "tb": F["_tb"], "ta": F["_ta"], "aplain": cos_d(F["_ab"], F["_aa"]),
         "tplain": cos_d(F["_tb"], F["_ta"]), "radar_flip": F["_flip"]}
    li = np.arange(len(F["_ab"]))
    s = base_scores(m["models"], d, li)
    return RULES["either_radar"](s, m["bounds"]), final("abroad")["cutoff"][0.6]["either_radar"]


RNG = np.random.default_rng(0)
PER_CLASS = 3000  # teaching pixels per place and group (the weights keep each place equal anyway)


def fit(Xs, ys, ws):
    keep = []
    for X_, y_ in zip(Xs, ys):
        i = np.concatenate([RNG.choice(np.flatnonzero(y_ == v), min(PER_CLASS, (y_ == v).sum()), replace=False)
                            for v in (0, 1) if (y_ == v).any()])
        keep.append(i)
    Xs, ys = [X_[i] for X_, i in zip(Xs, keep)], [y_[i] for y_, i in zip(ys, keep)]
    ws = [weights(y_) for y_ in ys]
    X, y, w = np.concatenate(Xs), np.concatenate(ys), np.concatenate(ws)
    sc = StandardScaler().fit(X)
    m = LogisticRegression(C=0.1, max_iter=2000).fit(sc.transform(X), y, sample_weight=w)
    return lambda Z: m.predict_proba(sc.transform(Z))[:, 1]


def weights(y):
    """Each place weighs the same and its two groups balanced."""
    w = np.zeros(len(y))
    for v in (0, 1):
        if (y == v).any():
            w[y == v] = 0.5 / (y == v).sum()
    return w


def auc(y, s):
    return round(float(roc_auc_score(y, s)), 3) if 0 < y.sum() < len(y) else None


def main():
    sections = os.environ.get("SECTIONS", "ABCD")
    names = [n for n in LIDAR + CUTS + space_list() + [near_name(m) for m in NEAR] if os.path.exists(f"{DIR}/{n}.npz")]
    P = {}
    for n in names:
        F, g, src = build(n)
        if len(g) == 0:  # no label with both embeddings valid and a year after its pair
            print(f"{n:24s} no valid labels")
            continue
        sc, cut = agg_score(F, n)
        keep = np.isfinite(sc)  # a saved held-out score exists for the label (older satellite sites)
        F, g = {k: v[keep] for k, v in F.items()}, g[keep]
        P[n] = {"F": F, "g": g, "src": src, "score": sc[keep]}
        print(f"{n:24s} {src:5s} groups " + " ".join(f"{k}:{int((g == k).sum())}" for k in range(1, 8) if (g == k).any()), flush=True)
    near = [n for n in P if P[n]["src"] == "near"]
    names = [n for n in P if n not in near]  # parts A to C as first run, without the clearing near the mines
    path = os.path.join(HERE, "results", "separate.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    for k in sections:
        res[k] = {}
    lid = [n for n in names if P[n]["src"] == "lidar"]
    # the lidar contrast: 1 cleared and dug, 0 cleared and not dug
    L = {n: np.isin(P[n]["g"], (1, 2)) for n in lid}
    Y = {n: (P[n]["g"][L[n]] == 1).astype(int) for n in lid}

    if "A" in sections:
        part_a(P, names, lid, L, Y, res)
    if "B" in sections:
        part_b(P, names, lid, L, Y, res)
    if "C" in sections:
        part_c(P, names, res)
    if "D" in sections:
        part_d(P, names, near, cut, res)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(res, open(path, "w"), indent=1)


def lab(P, n):
    """Taught as dug: cleared and dug (lidar), moved (satellite). As not dug: cleared and not dug, cut ground,
    and the satellite sites' unchanged ground."""
    gg = P[n]["g"]
    m = np.isin(gg, (1, 2, 4, 6, 7))
    return m, np.isin(gg[m], (1, 6)).astype(int)


def part_a(P, names, lid, L, Y, res):
    print("\nA. leave one lidar site out: AUC, cleared and dug against cleared and not dug (pixels dug / not)")
    print(f"  {'site':20s} {'dug':>6s} {'not':>6s} {'score':>6s} " + " ".join(f"{s:>6s}" for s in SETS))
    for n in lid:
        y = Y[n]
        row = {"dug": int(y.sum()), "not": int((1 - y).sum()), "score": auc(y, P[n]["score"][L[n]])}
        for fs in SETS:
            tr = [m for m in lid if m != n and 0 < Y[m].sum() < len(Y[m])]
            f = fit([P[m]["F"][fs][L[m]] for m in tr], [Y[m] for m in tr], [weights(Y[m]) for m in tr])
            row[fs] = auc(y, f(P[n]["F"][fs][L[n]]))
        res["A"][n] = row
        print(f"  {n:20s} {row['dug']:6d} {row['not']:6d} " + " ".join(
            f"{row[k]:6.3f}" if row[k] is not None else f"{'-':>6s}" for k in ("score",) + SETS), flush=True)
    ok = [n for n in lid if res["A"][n]["score"] is not None and min(res["A"][n]["dug"], res["A"][n]["not"]) >= 200]
    print(f"  {'mean (' + str(len(ok)) + ' sites)':20s} {'':13s} " + " ".join(
        f"{np.mean([res['A'][n][k] for n in ok]):6.3f}" for k in ("score",) + SETS))


def part_b(P, names, lid, L, Y, res):
    print("\nB. taught on every lidar site; share called dug at the cutoff where 80% of held-out cleared-and-dug is,")
    print("   on the clearcut regions (cut ground) and the satellite mines (moved / unchanged), and AUC mine moved vs all cuts")
    for fs in ("score",) + SETS:
        if fs == "score":
            g = lambda n: P[n]["score"]
            ref = np.concatenate([P[n]["score"][L[n]][Y[n] == 1] for n in lid])
        else:
            tr = [m for m in lid if 0 < Y[m].sum() < len(Y[m])]
            f = fit([P[m]["F"][fs][L[m]] for m in tr], [Y[m] for m in tr], [weights(Y[m]) for m in tr])
            sc = {n: f(P[n]["F"][fs]) for n in names}
            g = lambda n, sc=sc: sc[n]
            # the cutoff from held-out lidar scores (part A's folds)
            ref = []
            for n in tr:
                tr2 = [m for m in tr if m != n]
                f2 = fit([P[m]["F"][fs][L[m]] for m in tr2], [Y[m] for m in tr2], [weights(Y[m]) for m in tr2])
                ref.append(f2(P[n]["F"][fs][L[n]][Y[n] == 1]))
            ref = np.concatenate(ref)
        t = float(np.quantile(ref, 0.2))
        cuts = np.concatenate([g(n)[P[n]["g"] == 4] for n in names if P[n]["src"] == "cut"])
        row = {"cut_called_dug": round(float((cuts > t).mean()), 3), "per_place": {}}
        for n in names:
            if P[n]["src"] == "lidar":
                continue
            s, gg = g(n), P[n]["g"]
            e = {"moved": round(float((s[gg == 6] > t).mean()), 3) if (gg == 6).any() else None,
                 "unchanged": round(float((s[gg == 7] > t).mean()), 3) if (gg == 7).any() else None,
                 "cut": round(float((s[gg == 4] > t).mean()), 3) if (gg == 4).any() else None,
                 "forest": round(float((s[gg == 5] > t).mean()), 3) if (gg == 5).any() else None}
            if (gg == 6).sum() >= 10:
                yy = np.r_[np.ones((gg == 6).sum()), np.zeros(len(cuts))]
                e["auc_vs_cuts"] = auc(yy, np.r_[s[gg == 6], cuts])
            row["per_place"][n] = e
        mv = [v for v in row["per_place"].values() if v["moved"] is not None]
        res["B"][fs] = row
        print(f"  {fs:6s} cuts called dug {row['cut_called_dug']:.3f} | " + "  ".join(
            f"{n.replace('space-', '')[:12]} {v['moved']:.2f}/{v['unchanged'] if v['unchanged'] is not None else float('nan'):.2f}"
            f" ({v.get('auc_vs_cuts') or float('nan'):.2f})"
            for n, v in row["per_place"].items() if v["moved"] is not None and (P[n]["g"] == 6).sum() >= 10), flush=True)


def part_c(P, names, res):
    print("\nC. lidar, clearcut regions and satellite mines all taught (cut ground and unchanged as not dug, moved as dug),")
    print("   each place held out: AUC at the place, and its dug pixels against every clearcut region's cut ground")
    lab_ = lambda n: lab(P, n)
    for fs in ("all", "pair", "after", "ctx"):
        out, held = {}, {}
        for n in names:
            m, y = lab_(n)
            if not m.any():
                continue
            tr = [k for k in names if k != n and lab_(k)[0].any()]
            f = fit([P[k]["F"][fs][lab_(k)[0]] for k in tr], [lab_(k)[1] for k in tr], [weights(lab_(k)[1]) for k in tr])
            s = f(P[n]["F"][fs][m])
            out[n] = {"auc": auc(y, s), "mean_dug": round(float(s[y == 1].mean()), 3) if y.any() else None,
                      "mean_not": round(float(s[y == 0].mean()), 3) if (1 - y).any() else None}
            held[n] = (y, s)
        allcuts = np.concatenate([held[n][1] for n in held if P[n]["src"] == "cut"])
        for n in held:
            y, s = held[n]
            # a dug place's dug pixels against every clearcut region's cut ground, each scored held out
            if P[n]["src"] != "cut" and y.sum() >= 10:
                out[n]["auc_vs_cuts"] = auc(np.r_[np.ones(int(y.sum())), np.zeros(len(allcuts))], np.r_[s[y == 1], allcuts])
        res["C"][fs] = out
        print(f"  {fs}")
        for n, v in out.items():
            print(f"    {n:24s} AUC {v['auc'] if v['auc'] is not None else '-':>5}  vs cuts {v.get('auc_vs_cuts', '-')!s:>5}  "
                  f"mean dug {v['mean_dug']}  not {v['mean_not']}", flush=True)


def part_d(P, names, near, cut, res):
    """Each mine against the clearing picked near it, neither ever taught: the head taught without any clearing
    near a mine (as part C), then with the other mines' nearby clearing taught too. Also the map's view: the share
    of the mine's moved labels and of the nearby cut ground the aggregate lights at its 60% cutoff, alone and
    where the head also says dug (above one half)."""
    print("\nD. each mine's moved labels against its own nearby clearing (cut ground), mine and square held out")
    print(f"  {'mine':16s} {'km':>3s} {'n':>5s}  aggregate: AUC  found / cut lit | head AUC, and aggregate x head found / cut lit")
    for n in near:
        mine = "space-" + n[len("near-"):]
        if mine not in P or (P[mine]["g"] == 6).sum() < 10:
            continue
        mm, cc = P[mine]["g"] == 6, P[n]["g"] == 4
        sm, sc_ = P[mine]["score"][mm], P[n]["score"][cc]
        yy = np.r_[np.ones(mm.sum()), np.zeros(cc.sum())]
        row = {"km": round(float(np.load(f"{DIR}/{n}.npz")["km"]), 1), "moved": int(mm.sum()), "cut": int(cc.sum()),
               "score": {"auc": auc(yy, np.r_[sm, sc_]), "found": round(float((sm > cut).mean()), 3),
                         "cut_lit": round(float((sc_ > cut).mean()), 3)}}
        for fs in ("all", "pair", "ctx", "after"):
            for tag, extra in (("without", []), ("with", [k for k in near if k != n])):
                tr = [k for k in names + extra if k != mine and lab(P, k)[0].any()]
                f = fit([P[k]["F"][fs][lab(P, k)[0]] for k in tr], [lab(P, k)[1] for k in tr], [weights(lab(P, k)[1]) for k in tr])
                hm, hc = f(P[mine]["F"][fs][mm]), f(P[n]["F"][fs][cc])
                row[f"{fs}_{tag}"] = {"auc": auc(yy, np.r_[hm, hc]),
                                      "found": round(float(((sm > cut) & (hm > 0.5)).mean()), 3),
                                      "cut_lit": round(float(((sc_ > cut) & (hc > 0.5)).mean()), 3)}
        res["D"][n] = row
        sc0 = row["score"]
        print(f"  {mine[6:22]:16s} {row['km']:3.0f} {row['moved']:5d}  {sc0['auc']:.3f}  {sc0['found']:.2f} / {sc0['cut_lit']:.2f} | " +
              "  ".join(f"{k}: {row[k]['auc']:.2f} {row[k]['found']:.2f}/{row[k]['cut_lit']:.2f}"
                        for k in ("all_without", "all_with", "ctx_without", "ctx_with", "pair_with", "after_with")), flush=True)


if __name__ == "__main__":
    main()
