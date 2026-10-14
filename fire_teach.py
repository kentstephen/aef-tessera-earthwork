"""Teaching with burned forest as quiet ground: does it take away the burn cost, and what does it cost?

Abroad teaching made Grizzly Flats (forest burned to bare ground) light up: to the models it looks like land
cleared for a mine. Only two lidar burns teach quiet. ground.py found burned forest by Hansen's tree cover loss
in the fire year in 16 fires; here a sample of that ground (FIRE_PX pixels per fire) is taught as unchanged,
alongside a recipe's usual sites (BASE=us or abroad). CUT=1 teaches the clearcuts the same way.

Held out as before: each US digging site and lidar burn left out alone, the abroad sites and the fires in FOLDS
folds; everything else (crops, farms, marshes, lakes, clearcuts unless taught, places with no truth) scored by
models taught on every site. Scores go to data/scores-fire-<base>[-cut] in aggregate.py's layout, for quick.py
and checks.py.

Run: uv run python fire_teach.py   (BASE=abroad by default; about half an hour)
"""

import os

import numpy as np

import ground
from abroad_teach import FOLDS, fit_rows, rows, subset
from aggregate import base_scores, save_scores, tes_gate_bounds
from compare import labeled, load
from tessera_sites import OUT, SITES, abroad_sites, cdl_sets, place_names, space_names

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("BASE", "abroad")
CUT = os.environ.get("CUT", "0") == "1"
FIRE_PX = 20000
SCORES = os.path.join(HERE, "data", f"scores-fire-{BASE}{'-cut' if CUT else ''}")
RNG = np.random.default_rng(7)


def main():
    import aggregate
    aggregate.SCORES = SCORES
    have = lambda n: os.path.exists(f"{OUT}/{n}.npz")
    T, W, D, teach, quiet, abroad = rows(("islands", "space") if BASE == "abroad" else ())
    kinds = ("burn", "cut") if CUT else ("burn",)
    taught_ground = [f"ground-{n}" for n, v in ground.PLACES.items() if v[0] in kinds and have(f"ground-{n}")]
    for n in taught_ground:
        d = load(n)
        grp = np.load(f"{ground.PLACES_DIR}/{n}.npz")["grp"]
        ev = np.flatnonzero(d["ok"] & (grp == 1))
        T[n] = subset(d, RNG.choice(ev, min(len(ev), FIRE_PX), replace=False))
        W[n] = np.ones(len(T[n]["y"]))
        del d
    held = abroad + taught_ground
    fold = {n: i % FOLDS for i, n in enumerate(sorted(held))}
    print(f"taught: {len(teach)} US digging, {len(quiet)} US quiet, {len(abroad)} abroad, {len(taught_ground)} burned or cut", flush=True)

    def bounds(train):
        us = [n for n in train if n in teach + quiet]
        return tes_gate_bounds(T, {n: np.arange(len(T[n]["y"])) for n in us}, us)

    def score(models, n, d, role, b):
        li = labeled(d)
        kind, y = ground.saved(n[len("ground-"):], d, li) if n.startswith("ground-") else (d["kind"], d["y"][li])
        save_scores(n, kind, base_scores(models, d, li), y, role, b, d["patch"][li])

    for n in teach + quiet:
        train = [m for m in T if m != n]
        models = {w: fit_rows(T, W, train, w) for w in ("aef", "tes", "both")}
        score(models, n, D[n], "teach" if n in teach else "quiet", bounds(train))
        print(f"held out {n}", flush=True)
    for k in range(FOLDS):
        out = [n for n in held if fold[n] == k]
        train = [m for m in T if m not in out]
        models = {w: fit_rows(T, W, train, w) for w in ("aef", "tes", "both")}
        b = bounds(train)
        for n in out:
            d = load(n)
            if d["kind"] == "space_quiet":
                d["y"] = np.where(d["y"] == 1, -1, d["y"]).astype(np.int8)
            score(models, n, d, "abroad", b)
            del d
        print(f"held out fold {k}: {out}", flush=True)
    models = {w: fit_rows(T, W, list(T), w) for w in ("aef", "tes", "both")}
    b = bounds(list(T))
    D.clear()
    rest = [n for n in SITES + [s[0] for s in cdl_sets()] + abroad_sites() + place_names() + space_names()
            + [f"ground-{g}" for g in ground.PLACES] if have(n) and n not in T]
    for n in rest:
        score(models, n, load(n), "test", b)
    print(f"test only: {len(rest)} places", flush=True)


if __name__ == "__main__":
    main()
