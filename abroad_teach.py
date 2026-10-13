"""The three models taught with truth from abroad too, scored for quick.py beside the US-only recipe.

Abroad, the models rank moved ground about as well as at home but score it dim: few pixels pass a cutoff set at
home. Earthwork brightened them by teaching with the abroad truth. This adds, to the US digging and quiet
sites:
  islands     Puerto Rico and the US Virgin Islands lidar: moved over 2 m and unchanged ground
  space dig   ICESat-2 and GEDI pairs at building and mining sites, both ways, each label weighted W_SPACE
              (a site holds hundreds of labels where a lidar site holds tens of thousands)
  space quiet the same pairs on savanna and rainforest, unchanged only
Fields stay test only: Reunion's and Guadeloupe's cane and bananas, Hong Kong's farms and the US farm, marsh
and crop-switch ground are never taught.

Every taught site is scored by models that did not see it: each US digging site left out alone, the abroad
sites in FOLDS folds. Test-only ground is scored by models taught on everything. Scores go to
data/scores-abroad/ in aggregate.py's layout; compare the recipes with
  SCORES_DIR=data/scores-abroad uv run python quick.py

Run: uv run python abroad_teach.py   (about half an hour)
"""

import os

import numpy as np
from sklearn.linear_model import LogisticRegression

import compare
from aggregate import base_scores, save_scores, tes_gate_bounds
from compare import QUIET_KINDS, TEACH_KINDS, feats, labeled, load, sample
from tessera_sites import OUT, SITES, abroad_sites, cdl_sets, place_names, space_names

HERE = os.path.dirname(os.path.abspath(__file__))
SCORES = os.path.join(HERE, "data", "scores-abroad")
W_SPACE = float(os.environ.get("W_SPACE", "5"))
FOLDS = 4


def subset(d, idx):
    """The rows of a loaded site that teaching needs, so a large island need not stay loaded."""
    keep = ("ab", "aa", "tb", "ta", "y", "aplain", "tplain", "radar_flip")
    return {**{k: d[k][idx] for k in keep}, "kind": d["kind"], "ok": np.ones(len(idx), bool)}


def main():
    import aggregate
    aggregate.SCORES = SCORES
    compare.RNG = np.random.default_rng(0)
    have = lambda n: os.path.exists(f"{OUT}/{n}.npz")
    us = [n for n in SITES if have(n)]
    D = {n: load(n) for n in us}
    teach = [n for n in us if D[n]["kind"] in TEACH_KINDS]
    quiet = [n for n in us if D[n]["kind"] in QUIET_KINDS]
    for n in us:
        if n not in teach + quiet:
            D.pop(n)
    islands = [n for n in abroad_sites() if n.startswith(("pr-", "vi-")) and have(n)]
    space = [n for n in space_names() if have(n)]

    # the taught rows of every site, with each row's weight
    T, W = {}, {}
    for n in teach + quiet:
        T[n] = subset(D[n], sample(D[n], quiet=n in quiet))
        W[n] = np.ones(len(T[n]["y"]))
    abroad = []
    for n in islands + space:
        d = load(n)
        if d["kind"] == "space_quiet":
            d["y"] = np.where(d["y"] == 1, -1, d["y"]).astype(np.int8)
        lab = labeled(d)
        if (d["y"][lab] == 0).sum() < 10:
            continue
        idx = lab if n.startswith("space-") else sample(d)
        T[n] = subset(d, idx)
        W[n] = np.full(len(idx), W_SPACE if n.startswith("space-") else 1.0)
        abroad.append(n)
        del d
    fold = {n: i % FOLDS for i, n in enumerate(sorted(abroad))}
    print(f"taught: {len(teach)} US digging, {len(quiet)} US quiet, {len(abroad)} abroad ({FOLDS} folds)", flush=True)
    every = np.arange

    def fit(train, which):
        X = np.concatenate([feats(T[n], every(len(T[n]["y"])), which) for n in train])
        y = np.concatenate([(T[n]["y"] == 1).astype(int) for n in train])
        w = np.concatenate([W[n] for n in train])
        return LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced").fit(X, y, sample_weight=w)

    def bounds(train):
        us_train = [n for n in train if n in teach + quiet]
        return tes_gate_bounds(T, {n: every(len(T[n]["y"])) for n in us_train}, us_train)

    def score(models, n, d, role, b):
        li = labeled(d)
        save_scores(n, d["kind"], base_scores(models, d, li), d["y"][li], role, b, d["patch"][li])

    # each US teaching site left out alone
    for n in teach + quiet:
        train = [m for m in T if m != n]
        models = {w: fit(train, w) for w in ("aef", "tes", "both")}
        score(models, n, D[n], "teach" if n in teach else "quiet", bounds(train))
        print(f"held out {n}", flush=True)
    # the abroad sites a fold at a time
    for k in range(FOLDS):
        out = [n for n in abroad if fold[n] == k]
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
    # everything else, by models taught on every site
    models = {w: fit(list(T), w) for w in ("aef", "tes", "both")}
    b = bounds(list(T))
    D.clear()
    rest = [n for n in SITES + [s[0] for s in cdl_sets()] + abroad_sites() + place_names() + space_names()
            if have(n) and n not in T]
    for n in rest:
        score(models, n, load(n), "test", b)
    print(f"test only: {len(rest)} places", flush=True)


if __name__ == "__main__":
    main()
