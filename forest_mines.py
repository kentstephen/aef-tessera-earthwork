"""Mines in forest taught as digging, and digging where TESSERA's radar came or went.

forest_labels.py built the truth: satellite-laser pairs at mines cut into forest and woodland and at mines where
TESSERA had no radar in 2022 to 2024, and repeat 3DEP lidar over eastern Kentucky's surface coal mines. Here:

  tessera  TESSERA read under those labels (data/tessera, as every other site)
  gap      the abroad and US recipes' final models (never taught these sites) on them, at each recipe's 60%
           cutoff: moved ground found and unchanged ground lit, split by whether TESSERA's radar was there in
           one year only. The satellite-laser sites taught before are read from their held-out scores, so
           their 13 labels with radar in one year stand beside the new ones
  base     the new mines scored, test only, by fire_teach.py's abroad recipe (burns, or burns and cuts) refit on
           every site: the baseline the teach runs are read against
  teach    fire_teach.py's abroad recipe with burned forest taught as quiet (and the clearcuts with CUT=1), plus
           the new forest mines taught as digging (satellite labels weighted as abroad_teach weights them, the
           Kentucky lidar sampled as a US digging site), held out in folds with the abroad sites. The
           radar-gap mines stay test only. Scores go to data/scores-forest-abroad[-cut]
  late     sites whose truth came after a teach run, scored by that run's saved final models
  report   for each recipe: AP at the 14 US digging sites, burned forest, clearcuts and storm coast lit, and
           digging found at Kolwezi, Hibbing and each new mine, all at the 60% cutoff

The question: does clearing that turns into digging, taught from many more mines, let the models take burns
and cuts as quiet without doubting forest mines?

Run: uv run python forest_mines.py tessera | gap | base | teach | late <scores dir ...> | report   (CUT=1 teaches the clearcuts too; TEACH_SET=ky
     or space teaches only the lidar or only the satellite labels; about 40 minutes a recipe)
"""

import json
import os
import sys

import numpy as np
from sklearn.metrics import average_precision_score

import compare
import forest_labels as fl
import ground
import tessera_sites
from abroad_teach import FOLDS, W_SPACE, fit_rows, rows, subset
from aggregate import base_scores, cutoff, save_scores, tes_gate_bounds
from compare import labeled, load, sample
from tessera_sites import OUT, SITES, abroad_sites, cdl_sets, place_names, space_names

HERE = os.path.dirname(os.path.abspath(__file__))
CUT = os.environ.get("CUT", "0") == "1"
FIRE_PX = 20000
TEACH_SET = os.environ.get("TEACH_SET", "all")  # which new mines teach: all, ky (the lidar) or space (the satellite labels)
SCORES = os.environ.get("SCORES_DIR", os.path.join(
    HERE, "data", f"scores-forest-abroad{'-cut' if CUT else ''}{'' if TEACH_SET == 'all' else '-' + TEACH_SET}"))
CATCH = 0.6
SPACE_CAP = 2000
SHOW = ("aef_gated", "both_either", "either_radar")
_site_path = tessera_sites.site_path


def site_path(name):
    """The new sites live apart from Earthwork's cache (forest_labels.py)."""
    if name.startswith("space-") and os.path.exists(f"{fl.DIR}/aef/{name}.npz"):
        return f"{fl.DIR}/aef/{name}.npz"
    if os.path.exists(f"{fl.LIDAR_DIR}/{name}.npz"):
        return f"{fl.LIDAR_DIR}/{name}.npz"
    return _site_path(name)


tessera_sites.site_path = compare.site_path = site_path


def new_load(n):
    """A new site loaded; the Kentucky lidar reports as its own kind so it never joins the 14 US digging sites."""
    d = load(n)
    if n.startswith("ky-"):
        d["kind"] = "ky_mine"
    return d


def tessera():
    import zarr
    from obstore.store import HTTPStore
    from zarr.storage import ObjectStore
    zarr.config.set({"async.concurrency": 128})
    g = ObjectStore(HTTPStore.from_url(tessera_sites.TESSERA), read_only=True)
    tessera_sites.SPACE_DIR = fl.DIR
    for n in fl.names():
        tessera_sites.read_space(g, n)
        print(f"{n}: TESSERA read", flush=True)
    for n in fl.lidar_names():
        tessera_sites.read_site(g, n)
        print(f"{n}: TESSERA read", flush=True)


def have(n):
    return os.path.exists(f"{OUT}/{n}.npz")


def gap():
    from models import SCORE_DIRS, final
    from quick import RULES
    res = {}
    for recipe in ("abroad", "us"):
        m = final(recipe)
        cut = m["cutoff"][CATCH]
        rows_ = []
        # the satellite-laser sites taught before: their held-out scores
        sd = SCORE_DIRS[recipe]
        for n in space_names():
            if os.path.exists(f"{sd}/{n}.npz"):
                z = dict(np.load(f"{sd}/{n}.npz", allow_pickle=True))
                rows_.append((n, "before", z, tuple(z["bounds"])))
        for n in fl.names() + fl.lidar_names():
            if not have(n):
                continue
            d = new_load(n)
            li = labeled(d)
            s = base_scores(m["models"], d, li)
            s["y"] = d["y"][li]
            rows_.append((n, "new", s, m["bounds"]))
        out = res[recipe] = {}
        print(f"\n{recipe} recipe, at its {CATCH:.0%} cutoff: moved found / unchanged lit (labels), radar both years | radar one year only")
        for n, tag, s, b in rows_:
            y, flip = np.asarray(s["y"]), np.asarray(s["flip"]).astype(bool)
            r = {"tag": tag, "group": fl.group_of(n) if tag == "new" and n.startswith("space-") else ("ky" if n.startswith("ky-") else "before")}
            for part, m_ in (("steady", ~flip), ("flip", flip)):
                r[part] = {"moved": int(((y == 1) & m_).sum()), "unchanged": int(((y == 0) & m_).sum())}
                for w in SHOW:
                    v = RULES[w](s, b)
                    r[part][w] = [float((v[(y == 1) & m_] > cut[w]).mean()) if ((y == 1) & m_).any() else None,
                                  float((v[(y == 0) & m_] > cut[w]).mean()) if ((y == 0) & m_).any() else None]
            if (y == 1).any() and (y == 0).any():
                r["ap"] = {w: float(average_precision_score(y == 1, RULES[w](s, b))) for w in SHOW}
            out[n] = r
            f = lambda v: "  -  " if v is None else f"{v:.2f}"
            print(f"  {n:24s} {r['group']:7s} " + " | ".join(
                f"{p}: {r[p]['moved']:5d}/{r[p]['unchanged']:5d} " + " ".join(f"{w} {f(r[p][w][0])}/{f(r[p][w][1])}" for w in SHOW)
                for p in ("steady", "flip")) + (f"  AP {r['ap']['either_radar']:.2f}" if "ap" in r else ""), flush=True)
        for grp in ("before", "forest", "gap", "ky"):
            ns = [n for n in out if n != "_pooled" and out[n]["group"] == grp]
            for part in ("steady", "flip"):
                mv = sum(out[n][part]["moved"] for n in ns)
                un = sum(out[n][part]["unchanged"] for n in ns)
                if not (mv or un):
                    continue
                # pooled over labels, each site's share weighted by its labels
                pool = {w: [sum((out[n][part][w][k] or 0) * out[n][part][("moved", "unchanged")[k]] for n in ns) / max(c, 1)
                            for k, c in ((0, mv), (1, un))] for w in SHOW}
                out.setdefault("_pooled", {}).setdefault(grp, {})[part] = {"moved": mv, "unchanged": un, **pool}
                print(f"  pooled {grp:7s} {part:6s} moved {mv:5d} unchanged {un:6d}  " + "  ".join(
                    f"{w} found {pool[w][0]:.2f} lit {pool[w][1]:.3f}" for w in SHOW))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, "results", "forest_gap.json"), "w"), indent=1)


def teach():
    import aggregate
    aggregate.SCORES = SCORES
    T, W, D, teach_, quiet, abroad = rows(("islands", "space"))
    rng = np.random.default_rng(7)
    kinds = ("burn", "cut") if CUT else ("burn",)
    taught_ground = [f"ground-{n}" for n, v in ground.PLACES.items() if v[0] in kinds and have(f"ground-{n}")]
    for n in taught_ground:
        d = load(n)
        grp = np.load(f"{ground.PLACES_DIR}/{n}.npz")["grp"]
        ev = np.flatnonzero(d["ok"] & (grp == 1))
        T[n] = subset(d, rng.choice(ev, min(len(ev), FIRE_PX), replace=False))
        W[n] = np.ones(len(T[n]["y"]))
        del d
    forest = [n for n in fl.names("forest") if have(n)] if TEACH_SET in ("all", "space") else []
    ky = [n for n in fl.lidar_names() if have(n)] if TEACH_SET in ("all", "ky") else []
    compare.RNG = np.random.default_rng(1)
    for n in forest + ky:
        d = new_load(n)
        lab = labeled(d)
        if (d["y"][lab] == 1).sum() < 10:
            print(f"  {n}: too few moved labels, test only")
            continue
        idx = lab if n.startswith("space-") else sample(d)
        T[n] = subset(d, idx)
        # a satellite site weighs at most as much as SPACE_CAP labels at W_SPACE: Fort Hills alone holds 13,000
        # (10,000 moved), more weight than any US lidar site
        w = W_SPACE * min(1.0, SPACE_CAP / len(idx)) if n.startswith("space-") else 1.0
        W[n] = np.full(len(idx), w)
        del d
    new = [n for n in forest + ky if n in T]
    # the folds fire_teach.py used for the sites it held out, so a site such as Kolwezi is held out beside the same
    # sites as before and only the new mines differ; the new mines dealt into the folds on their own
    fold = {n: i % FOLDS for i, n in enumerate(sorted(abroad + taught_ground))}
    fold.update({n: i % FOLDS for i, n in enumerate(sorted(new))})
    held = abroad + taught_ground + new
    print(f"taught: {len(teach_)} US digging, {len(quiet)} US quiet, {len(abroad)} abroad, {len(taught_ground)} burned or cut, "
          f"{len(new)} new forest mines {new}", flush=True)

    def bounds(train):
        us = [n for n in train if n in teach_ + quiet]
        return tes_gate_bounds(T, {n: np.arange(len(T[n]["y"])) for n in us}, us)

    def score(models, n, d, role, b):
        li = labeled(d)
        kind, y = ground.saved(n[len("ground-"):], d, li) if n.startswith("ground-") else (d["kind"], d["y"][li])
        save_scores(n, kind, base_scores(models, d, li), y, role, b, d["patch"][li])

    for n in teach_ + quiet:
        train = [m for m in T if m != n]
        models = {w: fit_rows(T, W, train, w) for w in ("aef", "tes", "both")}
        score(models, n, D[n], "teach" if n in teach_ else "quiet", bounds(train))
        print(f"held out {n}", flush=True)
    for k in range(FOLDS):
        out = [n for n in held if fold[n] == k]
        train = [m for m in T if m not in out]
        models = {w: fit_rows(T, W, train, w) for w in ("aef", "tes", "both")}
        b = bounds(train)
        for n in out:
            d = new_load(n)
            if d["kind"] == "space_quiet":
                d["y"] = np.where(d["y"] == 1, -1, d["y"]).astype(np.int8)
            score(models, n, d, "abroad", b)
            del d
        print(f"held out fold {k}: {out}", flush=True)
    models = {w: fit_rows(T, W, list(T), w) for w in ("aef", "tes", "both")}
    b = bounds(list(T))
    import pickle
    pickle.dump({"models": models, "bounds": b, "taught": list(T)}, open(f"{SCORES}/models.pkl", "wb"))
    D.clear()
    rest = [n for n in SITES + [s[0] for s in cdl_sets()] + abroad_sites() + place_names() + space_names()
            + [f"ground-{g}" for g in ground.PLACES] + fl.names() + fl.lidar_names() if have(n) and n not in T]
    for n in rest:
        score(models, n, new_load(n), "test", b)
    print(f"test only: {len(rest)} places", flush=True)


def base():
    """The new mines scored as test only by fire_teach.py's abroad recipe (burns, or burns and cuts with CUT=1)
    taught on every site, the final models refit here as fire_teach fits them: the baseline the teach runs are
    read against. Written to data/scores-fire-abroad[-cut]-newsites."""
    import aggregate
    T, W, D, teach_, quiet, abroad = rows(("islands", "space"))
    D.clear()
    rng = np.random.default_rng(7)
    kinds = ("burn", "cut") if CUT else ("burn",)
    for n in [f"ground-{g}" for g, v in ground.PLACES.items() if v[0] in kinds and have(f"ground-{g}")]:
        d = load(n)
        grp = np.load(f"{ground.PLACES_DIR}/{n}.npz")["grp"]
        ev = np.flatnonzero(d["ok"] & (grp == 1))
        T[n] = subset(d, rng.choice(ev, min(len(ev), FIRE_PX), replace=False))
        W[n] = np.ones(len(T[n]["y"]))
        del d
    models = {w: fit_rows(T, W, list(T), w) for w in ("aef", "tes", "both")}
    us = teach_ + quiet
    b = tes_gate_bounds(T, {n: np.arange(len(T[n]["y"])) for n in us}, us)
    aggregate.SCORES = os.path.join(HERE, "data", f"scores-fire-abroad{'-cut' if CUT else ''}-newsites")
    for n in fl.names() + fl.lidar_names():
        if have(n):
            d = new_load(n)
            li = labeled(d)
            save_scores(n, d["kind"], base_scores(models, d, li), d["y"][li], "test", b, d["patch"][li])
            print(f"{n} scored", flush=True)


def late(dirs):
    """Sites whose truth arrived after a teach run (Detour Lake), scored by that run's final models as test only."""
    import pickle
    for sd in dirs:
        m = pickle.load(open(f"{sd}/models.pkl", "rb"))
        for n in fl.names() + fl.lidar_names():
            if have(n) and n not in m["taught"] and not os.path.exists(f"{sd}/{n}.npz"):
                d = new_load(n)
                li = labeled(d)
                import aggregate
                aggregate.SCORES = sd
                save_scores(n, d["kind"], base_scores(m["models"], d, li), d["y"][li], "test", m["bounds"], d["patch"][li])
                print(f"{os.path.basename(sd)}: {n} scored", flush=True)


def report(dirs):
    from quick import RULES
    new = fl.names() + fl.lidar_names()
    res = {}
    for sd in dirs:
        Z = {f[:-4]: dict(np.load(f"{sd}/{f}", allow_pickle=True)) for f in sorted(os.listdir(sd)) if f.endswith(".npz")}
        if os.path.isdir(sd + "-newsites"):  # a fire_teach.py run: the new mines scored by base()
            Z.update({f[:-4]: dict(np.load(f"{sd}-newsites/{f}", allow_pickle=True)) for f in sorted(os.listdir(sd + "-newsites"))
                      if f.endswith(".npz")})
        teach_ = [n for n, z in Z.items() if str(z["role"]) == "teach"]
        S = {n: {w: RULES[w](z, tuple(z["bounds"])).astype(np.float32) for w in SHOW} for n, z in Z.items()}
        out = res[os.path.basename(sd)] = {}
        print(f"\n{os.path.basename(sd)}: {len(teach_)} US digging sites")
        for w in SHOW:
            t = cutoff([S[n][w][Z[n]["y"] == 1] for n in teach_], CATCH)
            ap = float(np.mean([average_precision_score(Z[n]["y"], S[n][w]) for n in teach_]))
            lit = {}
            for k in ("burn", "fire", "cut", "storm", "crop", "farm"):
                v = [float((S[n][w][Z[n]["y"] == 0] > t).mean()) for n, z in Z.items() if str(z["kind"]) == k and (z["y"] == 0).sum() >= 10]
                lit[k] = float(np.mean(v)) if v else None
            found = {n: float((S[n][w][Z[n]["y"] == 1] > t).mean()) for n in ["space-kolwezi-cd", "hibbing-mn"] + new
                     if n in Z and (Z[n]["y"] == 1).sum() >= 5}
            roles = {n: str(Z[n]["role"]) for n in found}
            site_ap = {n: float(average_precision_score(Z[n]["y"] == 1, S[n][w])) for n in found if (Z[n]["y"] == 0).sum() >= 5}
            out[w] = {"ap": ap, "lit": lit, "found": found, "roles": roles, "site_ap": site_ap}
            print(f"  {w:13s} AP {ap:.3f}  lit " + " ".join(f"{k} {v:.3f}" if v is not None else f"{k} -" for k, v in lit.items()))
            print("    found (AP) " + ", ".join(f"{n.replace('space-', '')} {v:.2f}{'*' if roles[n] == 'test' else ''}"
                                         + (f" ({site_ap[n]:.2f})" if n in site_ap else "") for n, v in found.items()))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    path = os.path.join(HERE, "results", "forest_teach.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    json.dump({**old, **res}, open(path, "w"), indent=1)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "report"
    if what == "report":
        report(sys.argv[2:] or [os.path.join(HERE, "data", d) for d in
                                ("scores-fire-abroad", "scores-fire-abroad-cut", "scores-forest-abroad-ky", "scores-forest-abroad-cut-ky",
                                 "scores-forest-abroad", "scores-forest-abroad-cut")
                                if os.path.isdir(os.path.join(HERE, "data", d))])
    else:
        {"tessera": tessera, "gap": gap, "teach": teach, "base": base, "late": lambda: late(sys.argv[2:])}[what]()
