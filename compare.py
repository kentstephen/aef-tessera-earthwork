"""AlphaEarth vs TESSERA on the Earthwork lidar sites: do they see the same moved ground?

For every place read by tessera_sites.py, with the lidar's labels (moved > 0.5 m, unchanged < 0.1 m) on
AlphaEarth's 10 m grid:

1. Plain change. 1 - cos(b, a) in each embedding: how closely the two agree pixel by pixel (Spearman), and
   how well each finds moved ground on its own (AUC, AP).
2. The Earthwork model in each. The same logistic regression on [b, a, a * b, (a - b)^2] from AlphaEarth (256)
   and from TESSERA (512), and from both together. Digging sites (building, mining) teach moved and
   unchanged; burns and dunes teach only unchanged, as in Earthwork; each digging site is scored by a model
   taught on the others. Farmland, marshes, a storm coast and crop switches (Cropland Data Layer) only test:
   fields so often become building sites that teaching them as quiet could hide real digging. Then the two
   used together: a pixel counts only where both models call it moved.
3. The vectors broken down. How much of AlphaEarth's change vector (a - b, 64 numbers) a linear map from
   TESSERA's (128) recovers on a site it was not fit on, number by number, and the canonical correlations
   between the two change vectors: how many directions of change they share.
4. Radar. TESSERA's plain change on unchanged ground where Sentinel-1 was there one year and missing the other.

Run: uv run python compare.py   (writes results/compare.json)
"""

import json
import os

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score

from tessera_sites import EARTHWORK, OUT, SITES, cdl_sets

HERE = os.path.dirname(os.path.abspath(__file__))
TEACH_KINDS = ("building", "mining")
QUIET_KINDS = ("burn", "dunes")  # taught only as not digging, as in Earthwork
MODELS = ("aef", "tes", "both")
RNG = np.random.default_rng(0)


def f32(v):
    return v.astype(np.float32)


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)


def load(name):
    z = np.load(f"{EARTHWORK}/{name}.npz", allow_pickle=True)
    t = np.load(f"{OUT}/{name}.npz", allow_pickle=True)
    meta = json.loads(str(t["meta"]))
    X = z["X"]
    ab, aa = X[:, :64].astype(np.float32), X[:, 64:128].astype(np.float32)
    tb = t["b_q"].astype(np.float32) * t["b_scales"][:, None]
    ta = t["a_q"].astype(np.float32) * t["a_scales"][:, None]
    ok = (np.linalg.norm(ab, axis=1) > 0.5) & (np.linalg.norm(aa, axis=1) > 0.5)
    ok &= np.isfinite(t["b_scales"]) & np.isfinite(t["a_scales"]) & (t["b_scales"] > 0) & (t["a_scales"] > 0)
    s1b = t["b_s1_asc_obs_count"].astype(int) + t["b_s1_desc_obs_count"]
    s1a = t["a_s1_asc_obs_count"].astype(int) + t["a_s1_desc_obs_count"]
    tb, ta = unit(np.nan_to_num(tb)), unit(np.nan_to_num(ta))
    # a crop switch is ground nobody dug: every pixel unchanged
    y = z["y"] if "y" in z.files else np.zeros(len(X), np.int8)
    aplain, tplain = 1 - (ab * aa).sum(1), 1 - (tb * ta).sum(1)
    # held at half precision (as the Earthwork cache is): 54 places at full precision outgrow memory
    return {
        "name": name, "kind": meta["kind"], "years": meta["aef"], "y": y, "ok": ok,
        "ab": ab.astype(np.float16), "aa": aa.astype(np.float16), "tb": tb.astype(np.float16), "ta": ta.astype(np.float16),
        "aplain": aplain, "tplain": tplain,
        "radar_flip": (s1b > 0) != (s1a > 0), "s1": (float(np.median(s1b)), float(np.median(s1a))),
    }


def feats(d, idx, which):
    out = []
    if which in ("aef", "both"):
        b, a = d["ab"][idx].astype(np.float32), d["aa"][idx].astype(np.float32)
        out += [b, a, a * b, (a - b) ** 2]
    if which in ("tes", "both"):
        b, a = d["tb"][idx].astype(np.float32), d["ta"][idx].astype(np.float32)
        out += [b, a, a * b, (a - b) ** 2]
    return np.concatenate(out, 1)


def sample(d, n_pos=8000, n_neg=40000, quiet=False):
    lab = np.where(d["ok"], d["y"], -1)
    pos, neg = np.flatnonzero(lab == 1), np.flatnonzero(lab == 0)
    pos = pos[:0] if quiet else RNG.choice(pos, min(len(pos), n_pos), replace=False)
    return np.concatenate([pos, RNG.choice(neg, min(len(neg), n_neg), replace=False)])


def labeled(d):
    return np.flatnonzero(d["ok"] & (d["y"] >= 0))


def sub(idx, n=200000):
    return idx if len(idx) <= n else RNG.choice(idx, n, replace=False)


def metrics(y, s):
    return {"auc": round(float(roc_auc_score(y, s)), 3), "ap": round(float(average_precision_score(y, s)), 3)}


def gate(p, plain):
    """The Earthwork map's gate: the score, quieted where AlphaEarth barely changed."""
    return p * np.clip((plain - 0.05) / 0.10, 0, 1)


def teaching_names():
    """The read sites that teach: digging (moved and unchanged) and quiet ground (unchanged only)."""
    out = []
    for n in SITES:
        p = f"{OUT}/{n}.npz"
        if os.path.exists(p) and json.loads(str(np.load(p, allow_pickle=True)["meta"]))["kind"] in TEACH_KINDS + QUIET_KINDS:
            out.append(n)
    return out


def fit_models(D, held_out=None, which=("aef", "tes")):
    """The Earthwork model in each embedding, taught on every teaching site but held_out (D holds them loaded)."""
    train = [n for n in teaching_names() if n != held_out]
    idx = {n: sample(D[n], quiet=D[n]["kind"] in QUIET_KINDS) for n in train}
    y = np.concatenate([(D[n]["y"][idx[n]] == 1).astype(int) for n in train])
    return {w: LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced").fit(
        np.concatenate([feats(D[n], idx[n], w) for n in train]), y) for w in which}


def cca(X, Y, k=10, eps=1e-3):
    """The top k canonical correlations between X and Y (rows paired)."""
    X, Y = X - X.mean(0), Y - Y.mean(0)

    def whiten(M):
        U, S, _ = np.linalg.svd(M, full_matrices=False)
        return U[:, S > eps * S[0]]
    s = np.linalg.svd(whiten(X).T @ whiten(Y), compute_uv=False)
    return [round(float(v), 3) for v in s[:k]]


def r3(v):
    return round(float(v), 3)


def main():
    names = [n for n in SITES + [s[0] for s in cdl_sets()] if os.path.exists(f"{OUT}/{n}.npz")]
    D = {n: load(n) for n in names}
    lidar = [n for n in names if not n.startswith("cdl-")]
    res = {"sites": {}}
    print(f"{len(lidar)} lidar sites, {len(names) - len(lidar)} crop-switch sets\n")

    # 1. plain change, and the radar check
    print(f"{'place':34s} {'kind':9s} {'years':11s} {'valid':>5s} {'radar':>5s} {'rho':>5s}  AEF auc/ap   TES auc/ap   TES unchanged: radar same / flipped")
    for n, d in D.items():
        li = labeled(d)
        r = {"kind": d["kind"], "years": d["years"], "valid": r3(d["ok"].mean()),
             "radar_flip": r3(d["radar_flip"][d["ok"]].mean()), "s1_median_obs": d["s1"]}
        si = sub(np.flatnonzero(d["ok"]))
        r["rho_plain"] = r3(spearmanr(d["aplain"][si], d["tplain"][si])[0])
        still = li[d["y"][li] == 0]
        fl = d["radar_flip"][still]
        r["tes_still_plain"] = [r3(np.median(d["tplain"][still[~fl]])) if (~fl).any() else None,
                                r3(np.median(d["tplain"][still[fl]])) if fl.any() else None]
        r["aef_still_plain"] = r3(np.median(d["aplain"][still]))
        if (d["y"][li] == 1).sum() >= 50:
            r["aef_plain"], r["tes_plain"] = metrics(d["y"][li], d["aplain"][li]), metrics(d["y"][li], d["tplain"][li])
        res["sites"][n] = r
        ap = f"{r['aef_plain']['auc']:.3f}/{r['aef_plain']['ap']:.3f}" if "aef_plain" in r else "     -     "
        tp = f"{r['tes_plain']['auc']:.3f}/{r['tes_plain']['ap']:.3f}" if "tes_plain" in r else "     -     "
        print(f"{n:34s} {d['kind']:9s} {str(d['years']):11s} {r['valid']:5.2f} {r['radar_flip']:5.2f} {r['rho_plain']:5.2f}  {ap}  {tp}  {r['tes_still_plain']}")

    # 2. the Earthwork model in each embedding
    teach = [n for n in lidar if D[n]["kind"] in TEACH_KINDS]
    quiet = [n for n in lidar if D[n]["kind"] in QUIET_KINDS]
    test = [n for n in names if n not in teach + quiet]
    idx = {n: sample(D[n], quiet=n in quiet) for n in teach + quiet}

    def fit(train, which):
        X = np.concatenate([feats(D[n], idx[n], which) for n in train])
        y = np.concatenate([(D[n]["y"][idx[n]] == 1).astype(int) for n in train])
        return LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced").fit(X, y)

    def scores(models, d, li):
        sc = {w: models[w].predict_proba(feats(d, li, w))[:, 1] for w in MODELS}
        sc["aef_gated"] = gate(sc["aef"], d["aplain"][li])
        sc["agree"] = np.minimum(sc["aef_gated"], sc["tes"])  # moved only where both say so
        return sc

    print(f"\n{'held out':20s}  AP: AEF   TES  both agree plainAEF  rho  | moved found > .5: AEF(gated) TES  agree | moved: both / AEF only / TES only / neither")
    for n in teach:
        d = D[n]
        li = labeled(d)
        models = {w: fit([m for m in teach + quiet if m != n], w) for w in MODELS}
        sc, y = scores(models, d, li), d["y"][li]
        r, mv = res["sites"][n], y == 1
        r["rho_models"] = r3(spearmanr(sc["aef"], sc["tes"])[0])
        r["ap"] = {w: r3(average_precision_score(y, sc[w])) for w in (*MODELS, "aef_gated", "agree")}
        r["found"] = {w: r3((sc[w][mv] > 0.5).mean()) for w in (*MODELS, "aef_gated", "agree")}
        r["false_alarm"] = {w: r3((sc[w][~mv] > 0.5).mean()) for w in (*MODELS, "aef_gated", "agree")}
        A, T = sc["aef_gated"][mv] > 0.5, sc["tes"][mv] > 0.5
        r["moved_split"] = {"both": r3((A & T).mean()), "aef_only": r3((A & ~T).mean()),
                            "tes_only": r3((~A & T).mean()), "neither": r3((~A & ~T).mean())}
        a, f, s = r["ap"], r["found"], r["moved_split"]
        print(f"{n:20s}  {a['aef']:.3f} {a['tes']:.3f} {a['both']:.3f} {a['agree']:.3f}  {r['aef_plain']['ap']:.3f}  {r['rho_models']:5.2f} |"
              f"   {f['aef_gated']:.2f}  {f['tes']:.2f}  {f['agree']:.2f} | {s['both']:.2f} / {s['aef_only']:.2f} / {s['tes_only']:.2f} / {s['neither']:.2f}")
    for k, keys in (("mean_ap", (*MODELS, "aef_gated", "agree")), ("mean_found", (*MODELS, "aef_gated", "agree"))):
        src = "ap" if k == "mean_ap" else "found"
        for group, members in (("digging", teach), ("building", [n for n in teach if D[n]["kind"] == "building"])):
            res[f"{k}_{group}"] = {w: r3(np.mean([res["sites"][n][src][w] for n in members])) for w in keys}
    print(f"\nmean AP, digging sites held out: {res['mean_ap_digging']}")
    print(f"mean AP, building sites:         {res['mean_ap_building']}")
    print(f"mean share of moved ground found > .5, digging: {res['mean_found_digging']}")

    # test-only ground, scored by models taught on every digging and quiet site
    models = {w: fit(teach + quiet, w) for w in MODELS}
    print(f"\n{'test only':34s} {'kind':6s}  share of unchanged ground scored > .5:  AEF  AEF(gated)  TES  both  agree   | moved found: AEF(gated) TES agree")
    fa_by_kind = {}
    for n in test:
        d = D[n]
        li = labeled(d)
        sc, y = scores(models, d, li), d["y"][li]
        r = res["sites"][n]
        r["rho_models"] = r3(spearmanr(sc["aef"], sc["tes"])[0])
        r["false_alarm"] = {w: r3((sc[w][y == 0] > 0.5).mean()) for w in (*MODELS, "aef_gated", "agree")}
        fa = r["false_alarm"]
        line = f"{n:34s} {d['kind']:6s}  {'':38s}{fa['aef']:.3f}  {fa['aef_gated']:.3f}     {fa['tes']:.3f} {fa['both']:.3f} {fa['agree']:.3f}"
        if (y == 1).sum() >= 50:
            r["found"] = {w: r3((sc[w][y == 1] > 0.5).mean()) for w in (*MODELS, "aef_gated", "agree")}
            line += f"   |   {r['found']['aef_gated']:.2f}  {r['found']['tes']:.2f}  {r['found']['agree']:.2f}"
        print(line)
        fa_by_kind.setdefault(d["kind"], []).append(fa)
    res["false_alarm_by_kind"] = {k: {w: r3(np.mean([f[w] for f in v])) for w in v[0]} for k, v in fa_by_kind.items()}
    print("\nmean share of unchanged ground scored > .5, by kind:")
    for k, v in res["false_alarm_by_kind"].items():
        print(f"  {k:8s} {v}")

    # 3. the vectors broken down: TESSERA's change vector -> AlphaEarth's, fit on the other sites
    print("\nlinear map, TESSERA -> AlphaEarth, R2 on the held-out site (change vectors a - b; single year a for reference)")
    si = {m: sub(np.flatnonzero(D[m]["ok"]), 30000) for m in lidar}
    r2c, r2y, per_dim = [], [], []
    for n in lidar:
        tr = [m for m in lidar if m != n]
        Xc = np.concatenate([f32(D[m]["ta"][si[m]]) - f32(D[m]["tb"][si[m]]) for m in tr])
        Yc = np.concatenate([f32(D[m]["aa"][si[m]]) - f32(D[m]["ab"][si[m]]) for m in tr])
        Xy = np.concatenate([f32(D[m]["ta"][si[m]]) for m in tr])
        Yy = np.concatenate([f32(D[m]["aa"][si[m]]) for m in tr])
        d = D[n]
        pc = Ridge(alpha=1.0).fit(Xc, Yc).predict(f32(d["ta"][si[n]]) - f32(d["tb"][si[n]]))
        py = Ridge(alpha=1.0).fit(Xy, Yy).predict(f32(d["ta"][si[n]]))
        tc, ty = f32(d["aa"][si[n]]) - f32(d["ab"][si[n]]), f32(d["aa"][si[n]])
        per_dim.append(1 - ((tc - pc) ** 2).sum(0) / np.maximum(((tc - tc.mean(0)) ** 2).sum(0), 1e-12))
        r2c.append(1 - ((tc - pc) ** 2).sum() / ((tc - tc.mean(0)) ** 2).sum())
        r2y.append(1 - ((ty - py) ** 2).sum() / ((ty - ty.mean(0)) ** 2).sum())
        res["sites"][n]["r2_change"], res["sites"][n]["r2_year"] = r3(r2c[-1]), r3(r2y[-1])
        print(f"  {n:20s} change R2 {r2c[-1]:6.3f}   year R2 {r2y[-1]:6.3f}")
    pd_ = np.mean(per_dim, 0)
    res["r2_change_mean"], res["r2_year_mean"] = r3(np.mean(r2c)), r3(np.mean(r2y))
    res["r2_change_per_aef_dim"] = [r3(v) for v in pd_]
    print(f"  mean: change {res['r2_change_mean']}, year {res['r2_year_mean']}")
    print(f"  AlphaEarth change numbers best recovered: {[(int(i), round(float(pd_[i]), 2)) for i in np.argsort(-pd_)[:8]]}")
    print(f"  worst: {[(int(i), round(float(pd_[i]), 2)) for i in np.argsort(pd_)[:5]]}")

    allc = [sub(np.flatnonzero(D[m]["ok"]), 15000) for m in lidar]
    mov = [sub(np.flatnonzero(D[m]["ok"] & (D[m]["y"] == 1)), 15000) for m in lidar]
    for key, sel in (("cca_change_all", allc), ("cca_change_moved", mov)):
        X = np.concatenate([f32(D[m]["ta"][i]) - f32(D[m]["tb"][i]) for m, i in zip(lidar, sel)])
        Y = np.concatenate([f32(D[m]["aa"][i]) - f32(D[m]["ab"][i]) for m, i in zip(lidar, sel)])
        res[key] = cca(X, Y)
    print(f"\ncanonical correlations of the change vectors, all pixels: {res['cca_change_all']}")
    print(f"                                       moved pixels only: {res['cca_change_moved']}")

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "compare.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
