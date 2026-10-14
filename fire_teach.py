"""Teaching with burned forest as quiet ground: does it take away the burn cost, and what does it cost?

Abroad teaching made Grizzly Flats (forest burned to bare ground) light up: to the models it looks like land
cleared for a mine. Only two lidar burns teach quiet. ground.py found burned forest by Hansen's tree cover loss
in the fire year in 16 fires; here a sample of that ground (FIRE_PX pixels per fire) is taught as unchanged,
alongside a recipe's usual sites (BASE=us or abroad). CUT=1 teaches the clearcuts the same way.

Teaching every clearcut as quiet quiets clearcuts but costs mines in forest (Kolwezi): clearing is a mine's first
stage. Three ways between burns only and burns with every cut:
  CUT_PX     fewer cut pixels per region (default FIRE_PX)
  CUT_W      a lower weight on each cut pixel (default 1)
  CUT_FAR=1  only the cut pixels least like the digging taught: each cut pixel's change (AlphaEarth's a - b
             beside TESSERA's) against its nearest moved pixel at every taught digging site, and the half of
             each region's cut pixels farthest from all of them taught (at most CUT_PX). The nearest is taken
             over the sites each model is taught on, so a held-out mine never decides which cuts are taught
LEVEL=1 teaches the farm lidar's moved ground (land leveling) as digging, LEVEL=2 its unchanged ground too; the
farms are then held out in the folds with the abroad sites.

Held out as before: each US digging site and lidar burn left out alone, the abroad sites and the fires in FOLDS
folds; everything else (crops, farms, marshes, lakes, clearcuts unless taught, places with no truth) scored by
models taught on every site. Scores go to data/scores-fire-<base>[-cut][-knobs][-level<n>] in aggregate.py's
layout, for quick.py and checks.py.

Run: uv run python fire_teach.py   (BASE=abroad by default; about half an hour)
"""

import os

import numpy as np

import ground
from abroad_teach import FOLDS, fit_rows, rows, subset
from aggregate import base_scores, save_scores, tes_gate_bounds
from compare import labeled, load, sample
from tessera_sites import OUT, SITES, abroad_sites, cdl_sets, place_names, space_names

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("BASE", "abroad")
CUT = os.environ.get("CUT", "0") == "1"
FIRE_PX = 20000
CUT_PX = int(os.environ.get("CUT_PX", FIRE_PX))
CUT_W = float(os.environ.get("CUT_W", "1"))
CUT_FAR = os.environ.get("CUT_FAR", "0") == "1"
LEVEL = int(os.environ.get("LEVEL", "0"))
FARMS = ("arkansas-delta-ar", "fresno-farm-ca", "hamilton-farm-in", "antelope-farm-ne", "box-butte-farm-ne", "othello-pivots-wa")
KNOBS = ((f"-px{CUT_PX}" if CUT_PX != FIRE_PX else "") + (f"-w{CUT_W:g}" if CUT_W != 1 else "")
         + ("-far" if CUT_FAR else "")) if CUT else ""
SCORES = os.path.join(HERE, "data", f"scores-fire-{BASE}{'-cut' if CUT else ''}{KNOBS}{f'-level{LEVEL}' if LEVEL else ''}")
RNG = np.random.default_rng(7)
FAR_MOVED = 3000  # moved pixels per digging site the cut pixels are measured against


def change(d, idx):
    """A pixel's change in both embeddings: AlphaEarth's a - b beside TESSERA's."""
    f = lambda k: d[k][idx].astype(np.float32)
    return np.concatenate([f("aa") - f("ab"), f("ta") - f("tb")], 1)


def nearest(C, M, step=20000):
    """Each row of C's distance to its nearest row of M."""
    m2 = (M ** 2).sum(1)
    out = np.empty(len(C), np.float32)
    for i in range(0, len(C), step):
        c = C[i:i + step]
        out[i:i + step] = np.sqrt(np.maximum((c ** 2).sum(1)[:, None] + m2[None] - 2 * c @ M.T, 0).min(1))
    return out


def main():
    import aggregate
    aggregate.SCORES = SCORES
    have = lambda n: os.path.exists(f"{OUT}/{n}.npz")
    T, W, D, teach, quiet, abroad = rows(("islands", "space") if BASE == "abroad" else ())
    kinds = ("burn", "cut") if CUT else ("burn",)
    taught_ground = [f"ground-{n}" for n, v in ground.PLACES.items() if v[0] in kinds and have(f"ground-{n}")]
    cuts = [n for n in taught_ground if ground.PLACES[n[len("ground-"):]][0] == "cut"]
    for n in taught_ground:
        d = load(n)
        grp = np.load(f"{ground.PLACES_DIR}/{n}.npz")["grp"]
        ev = np.flatnonzero(d["ok"] & (grp == 1))
        # with CUT_FAR every cut pixel is a candidate (up to twice CUT_PX), the farthest kept per model below
        px = FIRE_PX if n not in cuts else 2 * CUT_PX if CUT_FAR else CUT_PX
        T[n] = subset(d, RNG.choice(ev, min(len(ev), px), replace=False))
        if n in cuts and CUT_FAR:
            T[n]["chg"] = change(T[n], np.arange(len(T[n]["y"])))
        W[n] = np.full(len(T[n]["y"]), CUT_W if n in cuts else 1.0)
        del d
    farms = []
    for n in FARMS if LEVEL else ():
        if not have(n):
            continue
        d = load(n)
        idx = sample(d, n_neg=0 if LEVEL == 1 else 40000)
        if (d["y"][idx] == 1).sum() < 50:  # too little moved ground to teach (Othello)
            continue
        T[n], W[n] = subset(d, idx), np.ones(len(idx))
        farms.append(n)
        del d
    held = abroad + taught_ground + farms
    fold = {n: i % FOLDS for i, n in enumerate(sorted(held))}
    print(f"taught: {len(teach)} US digging, {len(quiet)} US quiet, {len(abroad)} abroad, {len(taught_ground)} burned or cut"
          f"{f', {len(farms)} farms leveled' if farms else ''} -> {SCORES}", flush=True)
    for n in cuts + farms:
        print(f"  {n}: {len(T[n]['y'])} pixels ({(T[n]['y'] == 1).sum()} moved), weight {W[n][0]:g}", flush=True)

    # each cut candidate's distance to the nearest moved pixel at each digging site, so the farthest can be picked
    # from the sites a model is taught on
    dist = {}
    if CUT and CUT_FAR:
        movers = [n for n in T if n not in cuts and (T[n]["y"] == 1).sum() > 0]
        for m in movers:
            pos = np.flatnonzero(T[m]["y"] == 1)
            M = change(T[m], RNG.choice(pos, min(len(pos), FAR_MOVED), replace=False))
            dist[m] = {n: nearest(T[n]["chg"], M) for n in cuts}
        print(f"  cut distances to moved ground at {len(movers)} sites", flush=True)

    def view(train):
        """The taught rows for a training set: with CUT_FAR, each cut region down to its pixels farthest from the
        moved ground at the training set's digging sites."""
        if not dist:
            return T, W
        T2, W2 = dict(T), dict(W)
        ms = [m for m in dist if m in train]
        for n in cuts:
            if n in train:
                near = np.min([dist[m][n] for m in ms], 0)
                keep = np.argsort(-near)[:min(CUT_PX, len(near) // 2)]
                T2[n] = {k: (v[keep] if isinstance(v, np.ndarray) else v) for k, v in T[n].items()}
                W2[n] = W[n][keep]
        return T2, W2

    def fit(train, w):
        T2, W2 = view(train)
        return fit_rows(T2, W2, train, w)

    def bounds(train):
        us = [n for n in train if n in teach + quiet]
        return tes_gate_bounds(T, {n: np.arange(len(T[n]["y"])) for n in us}, us)

    def score(models, n, d, role, b):
        li = labeled(d)
        kind, y = ground.saved(n[len("ground-"):], d, li) if n.startswith("ground-") else (d["kind"], d["y"][li])
        save_scores(n, kind, base_scores(models, d, li), y, role, b, d["patch"][li])

    for n in teach + quiet:
        train = [m for m in T if m != n]
        models = {w: fit(train, w) for w in ("aef", "tes", "both")}
        score(models, n, D[n], "teach" if n in teach else "quiet", bounds(train))
        print(f"held out {n}", flush=True)
    for k in range(FOLDS):
        out = [n for n in held if fold[n] == k]
        train = [m for m in T if m not in out]
        models = {w: fit(train, w) for w in ("aef", "tes", "both")}
        b = bounds(train)
        for n in out:
            d = load(n)
            if d["kind"] == "space_quiet":
                d["y"] = np.where(d["y"] == 1, -1, d["y"]).astype(np.int8)
            score(models, n, d, "abroad", b)
            del d
        print(f"held out fold {k}: {out}", flush=True)
    models = {w: fit(list(T), w) for w in ("aef", "tes", "both")}
    b = bounds(list(T))
    D.clear()
    rest = [n for n in SITES + [s[0] for s in cdl_sets()] + abroad_sites() + place_names() + space_names()
            + [f"ground-{g}" for g in ground.PLACES] if have(n) and n not in T]
    for n in rest:
        score(models, n, load(n), "test", b)
    print(f"test only: {len(rest)} places", flush=True)


if __name__ == "__main__":
    main()
