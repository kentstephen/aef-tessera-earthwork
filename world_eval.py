"""Does the score mean the same thing abroad? The map's models on the world lidar, then taught with it.

world_sites.py built repeat lidar outside the US into Earthwork's site format (Netherlands, New Zealand, Norway,
Switzerland, England, Scotland, Lyon, Canada, Poland, Estonia, Kanagawa, Sao Paulo). Three steps here:

  baseline  the map's models as shipped (models.final("abroad"): US lidar, island lidar and satellite-laser labels)
            on every world site: moved ground found and unchanged ground lit at the map's cutoff (60% of moved
            ground found at held-out US digging sites), AP, and the median score on moved ground, per country,
            with and without the structure gate (FINDINGS 28). None of these sites taught the models.
  teach     two recipes taught on the same rows, every number held out: "abroad" (the shipped recipe's sites) and
            "world" (those plus the world lidar). US teaching sites in four folds, the islands and satellite labels
            in four folds, the world sites a country at a time; test-only ground (farms, marshes, crop switches,
            fields and towns abroad, deserts, dense cities) by each recipe's model taught on everything. Each
            recipe's cutoff is set on its own held-out US digging scores, as the map's is. Only the AlphaEarth
            model (the map zoomed out) and the joint model (the aggregate) are taught; TESSERA's own layer is not.
  calib     from the world recipe's held-out scores: the aggregate's score mapped to the chance the ground moved
            (isotonic, each site weighted equally), per region, to see whether one fixed chance means the same
            everywhere.

Every site is scored in a child process (a US site holds about 1 GB loaded). Scores go to data/scores-world/<recipe>/,
results to results/world.json.

Run: uv run python world_eval.py baseline | teach | calib | report   (teach takes about two hours)
"""

import json
import os
import pickle
import sys
import time

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

import compare
from cities import in_child
from compare import QUIET_KINDS, TEACH_KINDS, feats, labeled, load, sample
from quick import either, gate_a
from tessera_sites import OUT, SITES, abroad_sites, cdl_sets, place_names, space_names

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(HERE, "data", "scores-world")
RES = os.path.join(HERE, "results", "world.json")
CATCH = 0.6
N_POS, N_NEG = 4000, 20000  # taught pixels a site, every group alike
W_SPACE = 5.0
FOLDS = 4
WHICH = ("aef", "both")
STRUCT = os.path.join(HERE, "models", "structure_hgb_small.pkl")
t0 = time.time()
say = lambda *a: print(f"[{time.time() - t0:6.0f} s]", *a, flush=True)


def tmeta(n):
    return json.loads(str(np.load(f"{OUT}/{n}.npz", allow_pickle=True)["meta"]))


def have(n):
    return os.path.exists(f"{OUT}/{n}.npz")


def world():
    """Every world site with TESSERA read: name -> (country, kind)."""
    from world_sites import names
    return {n: (m["country"], m["kind"]) for n in names() if have(n) for m in [tmeta(n)]}


def rules(s, bounds):
    """The map's two layers here: AlphaEarth gated by its change, and the aggregate (the joint model opened where
    either embedding changed, by AlphaEarth alone where TESSERA's radar came or went)."""
    out = {}
    if "aef" in s:
        out["aef_gated"] = s["aef"] * gate_a(s)
    if "both" in s:
        out["aggregate"] = s["both"] * np.where(s["flip"], gate_a(s), either(s, bounds))
    if "tes" in s:
        out["tes"] = s["tes"]
    return out


def score_site(n, models, bounds, path, role, group, standing=True):
    """Base scores of a site's labeled pixels (chunked), and whether a structure stands there both years."""
    d = load(n)
    if d["kind"] == "space_quiet" or (n.startswith("world-") and tmeta(n).get("quiet_only")):
        d["y"] = np.where(d["y"] == 1, -1, d["y"]).astype(np.int8)
    li = labeled(d)
    s = {w: np.concatenate([m.predict_proba(feats(d, li[i:i + 200_000], w))[:, 1] for i in range(0, max(len(li), 1), 200_000)])
         .astype(np.float32) if len(li) else np.zeros(0, np.float32) for w, m in models.items()}
    stand = np.zeros(len(li), bool)
    if standing and not n.startswith(("space-", "cdl-")) and len(li):
        from structures import STAND
        from structures import feats as sfeats
        meta = tmeta(n)
        shape = tuple(meta["shape"]) if "shape" in meta else None
        if shape and shape[0] * shape[1] == len(d["y"]):
            mdl = pickle.load(open(STRUCT, "rb"))["model"]
            P = []
            for k in ("ab", "aa"):
                p = np.concatenate([mdl.predict_proba(x)[:, 1] for x in np.array_split(sfeats(d[k], shape), max(1, len(d["y"]) // 300_000))])
                P.append(p[li] >= STAND)
            stand = P[0] & P[1]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, y=d["y"][li], patch=d["patch"][li], role=role, group=group, kind=d["kind"], bounds=np.array(bounds),
             aplain=d["aplain"][li], tplain=d["tplain"][li], flip=d["radar_flip"][li], stand=stand, **s)
    return len(li)


# ---------------------------------------------------------------- baseline


def baseline():
    from models import final
    rec = final("abroad")
    W = world()
    say(f"baseline: {len(W)} world sites")
    for n, (c, k) in W.items():
        p = f"{DIR}/shipped/{n}.npz"
        if os.path.exists(p):
            continue
        m = in_child(score_site, n, rec["models"], rec["bounds"], p, "world", c)
        say(f"  {n}: {m:,} labeled px")
    report()


# ---------------------------------------------------------------- teach


def subset(d, idx):
    keep = ("ab", "aa", "tb", "ta", "y", "aplain", "tplain", "radar_flip")
    return {**{k: d[k][idx] for k in keep}, "kind": d["kind"], "ok": np.ones(len(idx), bool)}


def rows():
    """Every taught site's rows and weights, and each site's group and role."""
    compare.RNG = np.random.default_rng(0)
    T, Wt, info = {}, {}, {}
    for n in SITES:
        if not have(n):
            continue
        d = load(n)
        if d["kind"] in TEACH_KINDS:
            T[n], info[n] = subset(d, sample(d, N_POS, N_NEG)), ("us", "teach")
        elif d["kind"] in QUIET_KINDS:
            T[n], info[n] = subset(d, sample(d, N_POS, N_NEG, quiet=True)), ("us", "quiet")
        else:
            info[n] = ("us", "test")
        del d
    for n in [m for m in abroad_sites() if m.startswith(("pr-", "vi-"))] + space_names():
        if not have(n):
            continue
        d = load(n)
        if d["kind"] == "space_quiet":
            d["y"] = np.where(d["y"] == 1, -1, d["y"]).astype(np.int8)
        lab = labeled(d)
        if (d["y"][lab] == 0).sum() < 10:
            continue
        idx = lab if n.startswith("space-") else sample(d, N_POS, N_NEG)
        T[n], info[n] = subset(d, idx), ("islands" if n.startswith(("pr-", "vi-")) else "space", "abroad")
        del d
    for n, (c, k) in world().items():
        if k == "storm":
            info[n] = (c, "worldtest")
            continue
        d = load(n)
        if tmeta(n).get("quiet_only"):
            d["y"] = np.where(d["y"] == 1, -1, d["y"]).astype(np.int8)
        if (d["y"][labeled(d)] == 0).sum() < 10:
            continue
        T[n], info[n] = subset(d, sample(d, N_POS, N_NEG)), (c, "world")
        del d
    for n in T:
        Wt[n] = np.full(len(T[n]["y"]), W_SPACE if n.startswith("space-") else 1.0)
    return T, Wt, info


def fit(T, Wt, train, which):
    X = np.concatenate([feats(T[n], np.arange(len(T[n]["y"])), which) for n in train])
    y = np.concatenate([(T[n]["y"] == 1).astype(int) for n in train])
    w = np.concatenate([Wt[n] for n in train])
    m = LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced").fit(X, y, sample_weight=w)
    del X
    return m


def bounds_of(T, train, info):
    from aggregate import tes_gate_bounds
    us = [n for n in train if info[n][0] == "us"]
    return tes_gate_bounds(T, {n: np.arange(len(T[n]["y"])) for n in us}, us)


def tests():
    """Ground only ever tested: (name, group)."""
    out = [(n, "us") for n in SITES if have(n) and load_kind(n) not in TEACH_KINDS + QUIET_KINDS]
    out += [(s[0], "crop") for s in cdl_sets() if have(s[0])]
    out += [(n, compare.kind_of(n, tmeta(n))) for n in abroad_sites() if not n.startswith(("pr-", "vi-")) and have(n)]
    out += [(n, "notruth") for n in place_names() if have(n)]
    return out


def load_kind(n):
    return tmeta(n).get("kind")


def teach():
    T, Wt, info = rows()
    say(f"rows: {len(T)} taught sites, {sum(len(t['y']) for t in T.values()):,} rows")
    us = sorted(n for n in T if info[n][0] == "us")
    ab = sorted(n for n in T if info[n][1] == "abroad")
    W = world()
    countries = sorted({c for c, _ in W.values()})
    fold = {n: i % FOLDS for i, n in enumerate(us)} | {n: i % FOLDS for i, n in enumerate(ab)}
    test = tests()
    for recipe in ("abroad", "world"):
        out = f"{DIR}/{recipe}"
        pool = [n for n in T if recipe == "world" or info[n][1] != "world"]

        def run(held, score_names, label):
            todo = [n for n, _ in score_names if not os.path.exists(f"{out}/{n}.npz")]
            if not todo:
                return
            train = [n for n in pool if n not in held]
            models = {w: fit(T, Wt, train, w) for w in WHICH}
            b = bounds_of(T, train, info)
            for n, (g, role) in score_names:
                if n in todo:
                    in_child(score_site, n, models, b, f"{out}/{n}.npz", role, g)
            say(f"  {recipe} {label}: scored {len(todo)}")
            return models, b

        for k in range(FOLDS):
            held = [n for n in us if fold[n] == k]
            run(set(held), [(n, info[n]) for n in held], f"US fold {k}")
            held = [n for n in ab if fold[n] == k]
            run(set(held), [(n, info[n]) for n in held], f"abroad fold {k}")
        if recipe == "world":
            for c in countries:
                held = [n for n, (cc, _) in W.items() if cc == c]
                run(set(held), [(n, (c, info.get(n, (c, "world"))[1])) for n in held], f"country {c}")
        # everything else by the model taught on all of the recipe's sites
        rest = [(n, (g, "test")) for n, g in test]
        if recipe == "abroad":
            rest += [(n, (c, "world" if k != "storm" else "worldtest")) for n, (c, k) in W.items()]
        got = run(set(), rest, "final")
        fp = f"{out}/final.pkl"
        if got or not os.path.exists(fp):
            models, b = got or ({w: fit(T, Wt, pool, w) for w in WHICH}, bounds_of(T, pool, info))
            pickle.dump({"models": models, "bounds": b}, open(fp, "wb"))
        if recipe == "world":
            f = pickle.load(open(fp, "rb"))
            save_map_model(f["models"], f["bounds"], out)
    del T
    cities_check()
    report()


def save_map_model(models, b, out):
    """The world recipe's final models in the map's format (models/model_world.npz), with the aggregate's and
    AlphaEarth's cutoffs from its own held-out US digging scores; TESSERA's layer keeps the shipped model."""
    shipped = np.load(os.path.join(HERE, "models", "model_abroad.npz"))
    cut = cutoffs(out)
    np.savez(os.path.join(HERE, "models", "model_world.npz"),
             aef_w=models["aef"].coef_.ravel().astype(np.float32), aef_b=np.float32(models["aef"].intercept_[0]),
             tes_w=shipped["tes_w"], tes_b=shipped["tes_b"],
             both_w=models["both"].coef_.ravel().astype(np.float32), both_b=np.float32(models["both"].intercept_[0]),
             bounds=np.array(b, np.float32), cut_aggregate=np.float32(cut["aggregate"]), cut_aef=np.float32(cut["aef_gated"]),
             cut_tessera=shipped["cut_tessera"], cutoff=np.float32(cut["aggregate"]))


def cities_check():
    """The share of dense-city pixels lit at each recipe's cutoff (cities.py's 22 squares, a few year pairs)."""
    res = json.load(open(RES)) if os.path.exists(RES) else {}
    res["cities"] = {}
    for recipe in ("abroad", "world"):
        out = f"{DIR}/{recipe}"
        if not os.path.isdir(out):
            continue
        res["cities"][recipe] = in_child(_cities, recipe)
        say(f"  cities {recipe} done")
    json.dump(res, open(RES, "w"), indent=1)


def _cities(recipe):
    import cities
    f = pickle.load(open(f"{DIR}/{recipe}/final.pkl", "rb"))
    models, b = f["models"], f["bounds"]
    cut = cutoffs(f"{DIR}/{recipe}")
    out = {}
    for n in cities.CITIES:
        p = f"{cities.DIR}/{n}.npz"
        if not os.path.exists(p):
            continue
        z = np.load(p)
        row = {}
        for yb, ya in ((2018, 2024), (2022, 2025), (2019, 2023)):
            d, li, _ = cities.pair(z, yb, ya)
            s = {w: models[w].predict_proba(feats(d, li, w))[:, 1] for w in WHICH}
            s.update(aplain=d["aplain"][li], tplain=d["tplain"][li], flip=d["radar_flip"][li])
            r = rules(s, b)
            row[f"{yb}-{ya}"] = {w: float((r[w] > cut[w]).mean()) for w in r}
        out[n] = row
    return out


# ---------------------------------------------------------------- report


def cutoff(site_moved, target):
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if np.mean([(m > mid).mean() for m in site_moved]) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def load_scores(d):
    return {f[:-4]: dict(np.load(f"{d}/{f}", allow_pickle=True)) for f in sorted(os.listdir(d)) if f.endswith(".npz")} if os.path.isdir(d) else {}


def cutoffs(d):
    Z = load_scores(d)
    teach = [z for z in Z.values() if str(z["role"]) == "teach"]
    R = [rules(z, tuple(z["bounds"])) for z in teach]
    return {w: cutoff([r[w][z["y"] == 1] for r, z in zip(R, teach)], CATCH) for w in R[0]}


def summarize(d, cut=None):
    """Per group: moved found and unchanged lit at the cutoff (mean over sites), AP, median score on moved ground,
    and the aggregate with the structure gate."""
    Z = load_scores(d)
    if not Z:
        return None
    if cut is None:
        cut = cutoffs(d)
    rows = {}
    for n, z in Z.items():
        role, g = str(z["role"]), str(z["group"])
        key = "us dig" if role == "teach" else f"us {z['kind']}" if g == "us" else g if role in ("world", "abroad") else f"{g} ({role})"
        if role == "worldtest":
            key = f"{g} storm"
        r = rules(z, tuple(z["bounds"]))
        y = z["y"] == 1
        for w, v in r.items():
            e = rows.setdefault(key, {}).setdefault(w, {"found": [], "lit": [], "ap": [], "med": [], "sites": 0, "moved": 0})
            e["sites"] += 1
            e["moved"] += int(y.sum())
            if y.sum() >= 50:
                e["found"].append(float((v[y] > cut[w]).mean()))
                e["med"].append(float(np.median(v[y])))
                if (~y).sum() >= 50:
                    e["ap"].append(float(average_precision_score(y, v)))
            if (~y).sum() >= 50:
                e["lit"].append(float((v[~y] > cut[w]).mean()))
        if "aggregate" in r and z["stand"].any():
            v = np.where(z["stand"], 0, r["aggregate"])
            e = rows[key].setdefault("aggregate+gate", {"found": [], "lit": [], "ap": [], "med": [], "sites": 0, "moved": 0})
            e["sites"] += 1
            if y.sum() >= 50:
                e["found"].append(float((v[y] > cut["aggregate"]).mean()))
            if (~y).sum() >= 50:
                e["lit"].append(float((v[~y] > cut["aggregate"]).mean()))
    mean = lambda v: round(float(np.mean(v)), 3) if v else None
    return {"cutoff": {w: round(c, 3) for w, c in cut.items()},
            "groups": {k: {w: {"sites": e["sites"], "moved_px": e["moved"], "found": mean(e["found"]), "lit": mean(e["lit"]),
                               "ap": mean(e["ap"]), "median_moved": mean(e["med"])} for w, e in v.items()} for k, v in sorted(rows.items())}}


def site_table(d, cut):
    Z = load_scores(d)
    out = {}
    for n, z in Z.items():
        if not n.startswith("world-"):
            continue
        r = rules(z, tuple(z["bounds"]))
        y = z["y"] == 1
        out[n] = {"moved": int(y.sum()), "unchanged": int((~y).sum()),
                  **{w: {"found": round(float((v[y] > cut[w]).mean()), 3) if y.sum() else None,
                         "lit": round(float((v[~y] > cut[w]).mean()), 3) if (~y).sum() else None,
                         "ap": round(float(average_precision_score(y, v)), 3) if 0 < y.sum() < len(y) else None} for w, v in r.items()}}
    return out


def calib():
    """The aggregate's held-out score as a chance of moved ground, per region (world recipe)."""
    Z = load_scores(f"{DIR}/world")
    region = lambda z: "us" if str(z["role"]) == "teach" else str(z["group"])
    keep = [(n, z) for n, z in Z.items() if str(z["role"]) in ("teach", "world", "abroad") and str(z["group"]) != "space"]
    S, Y, Wt, G = [], [], [], []
    for n, z in keep:
        v = rules(z, tuple(z["bounds"]))["aggregate"]
        if len(v) < 100:
            continue
        S.append(v)
        Y.append((z["y"] == 1).astype(float))
        Wt.append(np.full(len(v), 1.0 / len(v)))
        G.append(np.full(len(v), region(z), object))
    S, Y, Wt, G = map(np.concatenate, (S, Y, Wt, G))
    iso = IsotonicRegression(out_of_bounds="clip").fit(S, Y, sample_weight=Wt)
    bins = np.array([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0001])
    res = {"pooled_score_at_chance": {}, "regions": {}}
    grid = np.linspace(0, 1, 1001)
    pg = iso.predict(grid)
    for c in (0.3, 0.5, 0.7):
        res["pooled_score_at_chance"][str(c)] = round(float(grid[np.argmax(pg >= c)]) if (pg >= c).any() else 1.0, 3)
    for g in sorted(set(G)):
        m = G == g
        ir = IsotonicRegression(out_of_bounds="clip").fit(S[m], Y[m], sample_weight=Wt[m])
        pr = ir.predict(grid)
        rel = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            k = m & (S >= lo) & (S < hi)
            rel.append(None if Wt[k].sum() == 0 or k.sum() < 50 else
                       [round(float(np.average(iso.predict(S[k]), weights=Wt[k])), 3), round(float(np.average(Y[k], weights=Wt[k])), 3), int(k.sum())])
        res["regions"][g] = {"score_at_50": round(float(grid[np.argmax(pr >= 0.5)]) if (pr >= 0.5).any() else 1.0, 3),
                             "pixels": int(m.sum()), "moved_share": round(float(np.average(Y[m], weights=Wt[m])), 3),
                             "bins (pooled chance, observed, n)": rel}
    out = json.load(open(RES)) if os.path.exists(RES) else {}
    out["calib"] = res
    json.dump(out, open(RES, "w"), indent=1)
    pickle.dump(iso, open(os.path.join(HERE, "models", "world_calib_isotonic.pkl"), "wb"))
    print(json.dumps(res, indent=1))


def report():
    out = json.load(open(RES)) if os.path.exists(RES) else {}
    from models import final
    rec = final("abroad")
    shipped_cut = {"aef_gated": rec["cutoff"][CATCH]["aef_gated"], "aggregate": rec["cutoff"][CATCH]["either_radar"],
                   "tes": rec["cutoff"][CATCH]["tes"]}
    us_ref = load_scores(os.path.join(HERE, "data", "scores-abroad"))
    us_med = [float(np.median(rules(z, tuple(z["bounds"]))["aggregate"][z["y"] == 1])) for z in us_ref.values()
              if str(z["role"]) == "teach" and "both" in z]
    out["us_reference_median_moved_aggregate"] = round(float(np.mean(us_med)), 3) if us_med else None
    for name, d, cut in (("shipped", f"{DIR}/shipped", shipped_cut), ("abroad", f"{DIR}/abroad", None), ("world", f"{DIR}/world", None)):
        s = summarize(d, cut)
        if s:
            out[name] = s
            out[f"{name}_sites"] = site_table(d, s["cutoff"])
    os.makedirs(os.path.dirname(RES), exist_ok=True)
    json.dump(out, open(RES, "w"), indent=1)
    for name in ("shipped", "abroad", "world"):
        if name not in out:
            continue
        print(f"\n== {name}: cutoffs {out[name]['cutoff']}")
        print(f"  {'group':28s} {'layer':15s} {'sites':>5s} {'moved':>8s} {'found':>6s} {'lit':>6s} {'AP':>6s} {'med':>6s}")
        for g, v in out[name]["groups"].items():
            for w, e in v.items():
                f = lambda x: f"{x:6.3f}" if x is not None else "     -"
                print(f"  {g:28s} {w:15s} {e['sites']:5d} {e['moved_px']:8d} {f(e['found'])} {f(e['lit'])} {f(e['ap'])} {f(e['median_moved'])}")
    if "cities" in out:
        for recipe, C in out["cities"].items():
            for w in ("aef_gated", "aggregate"):
                v = [r[w] for c in C.values() for r in c.values()]
                print(f"  cities {recipe} {w}: mean share lit {np.mean(v):.3f}")


if __name__ == "__main__":
    {"baseline": baseline, "teach": teach, "calib": calib, "report": report, "cities": cities_check}[sys.argv[1]]()
