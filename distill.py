"""Teaching TESSERA to the AlphaEarth model: AlphaEarth-only students, the same tiny model, taught with TESSERA's help.

The map can only read AlphaEarth at every zoom (TESSERA has no coarser product), so the aim is an AlphaEarth model
that has learned what the aggregate knows. Every student is the map's model unchanged in size: a logistic
regression on [b, a, a * b, (a - b)^2] of AlphaEarth's two years (256 weights), gated by AlphaEarth's change as
the map gates it. Only what it is taught on changes. The teacher sees both embeddings and is taught on the same
sites as the student, so a held-out site is held out from both.

Variants (VARIANTS selects some, comma separated):
  base               lidar labels only: the abroad recipe's AlphaEarth model, as aggregate.py teaches it
  soft_both_50       the label is half the lidar label, half the joint model's score (both embeddings)
  soft_both_0        the joint model's score alone
  soft_agg_50        half lidar, half the aggregate (the joint model x the radar-safe either gate)
  soft_agg_0         the aggregate alone
  soft_agg_50_unl    soft_agg_50, plus the aggregate's score on teaching sites' unlabeled ground (the band between
                     unchanged and moved that the lidar labels leave out), UNL_N pixels a site
  clean              lidar labels, less the moved pixels where neither embedding changed (lidar noise, or ground
                     no embedding can see moving)
  clean_soft_agg_50  clean and soft_agg_50 together
Soft labels are taught as two weighted rows (moved with weight p, unchanged with 1 - p), the moved and unchanged
totals balanced as class_weight="balanced" balances hard labels. Fields stay test only: crop, farm and marsh
ground is never taught, by labels or by the teacher.

Held out as abroad_teach.py holds out: each US teaching site alone, the abroad sites in FOLDS folds, test-only
ground by models taught on everything. Scores go to data/scores-distill/<variant>/ (aef: the student; both: the
teacher's joint model, for the aggregate beside it).

Memory: one teacher per fold shared by every student; every site or place is loaded in a child process that
exits once it has handed back its sampled rows or saved its scores (a site let go inside one long process left
its memory held, about 2 GB a site), so the main process holds only the sampled rows. It stops itself past
MEM_CAP_GB (default 20).

Run: uv run python distill.py                 (every variant; VARIANTS=base,clean picks some)
     uv run python distill.py report [catch]  (every variant's table, beside the aggregate)
"""

import json
import os
import resource
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

import compare
from abroad_teach import FOLDS, W_SPACE, subset
from aggregate import cutoff, tes_gate_bounds
from compare import QUIET_KINDS, TEACH_KINDS, feats, kind_of, labeled, load, sample
from tessera_sites import OUT, SITES, abroad_sites, cdl_sets, place_names, space_names

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "data", "scores-distill")
UNL_N = int(os.environ.get("UNL_N", "20000"))
VARIANTS = {
    "base": {"alpha": 1.0},
    "soft_both_50": {"alpha": 0.5, "teacher": "both"},
    "soft_both_0": {"alpha": 0.0, "teacher": "both"},
    "soft_agg_50": {"alpha": 0.5, "teacher": "agg"},
    "soft_agg_0": {"alpha": 0.0, "teacher": "agg"},
    "soft_agg_50_unl": {"alpha": 0.5, "teacher": "agg", "unl": True},
    "clean": {"alpha": 1.0, "clean": True},
    "clean_soft_agg_50": {"alpha": 0.5, "teacher": "agg", "clean": True},
}
DESERT = ("place-sahara-dz", "place-riyadh-edge-sa", "place-cairo-desert-eg")
MEM_CAP_GB = float(os.environ.get("MEM_CAP_GB", "20"))


def rss_gb():
    """The process's peak memory so far (macOS reports bytes, Linux kilobytes)."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e9 if sys.platform == "darwin" else r / 1e6


def check_mem(where):
    g = rss_gb()
    if g > MEM_CAP_GB:
        raise SystemExit(f"stopped at {where}: peak memory {g:.1f} GB is past MEM_CAP_GB {MEM_CAP_GB:g}")
    return g


def in_child(fn, *args):
    """fn(*args) in a fresh process that exits after: its memory goes back to the system."""
    with ProcessPoolExecutor(1, max_tasks_per_child=1) as ex:
        return ex.submit(fn, *args).result()


def read_site(n, how, want_unl):
    """In a child: one site's taught rows (how: "us", "quiet", "abroad"), and with want_unl its unlabeled sample.
    Returns (rows or None, weights, unlabeled rows or None)."""
    compare.RNG = np.random.default_rng(zlib.crc32(n.encode()))
    d = load(n)
    if d["kind"] == "space_quiet":
        d["y"] = np.where(d["y"] == 1, -1, d["y"]).astype(np.int8)
    if how == "abroad":
        lab = labeled(d)
        if (d["y"][lab] == 0).sum() < 10:
            return None, None, None
        idx = lab if n.startswith("space-") else sample(d)
        w = np.full(len(idx), W_SPACE if n.startswith("space-") else 1.0)
    else:
        idx = sample(d, quiet=how == "quiet")
        w = np.ones(len(idx))
    u = None
    if want_unl and how != "abroad":
        un = np.flatnonzero(d["ok"] & (d["y"] == -1))
        if len(un):
            u = subset(d, compare.RNG.choice(un, min(len(un), UNL_N), replace=False))
    return subset(d, idx), w, u


def light_rows(want_unl):
    """abroad_teach.rows()'s taught sites, each read in a child process (seeded by its name); with want_unl, UNL_N
    unlabeled pixels of each US teaching site too. (T, W, U, teach, quiet, abroad)."""
    have = lambda n: os.path.exists(f"{OUT}/{n}.npz")
    us = [n for n in SITES if have(n)]
    kind = {n: kind_of(n, json.loads(str(np.load(f"{OUT}/{n}.npz", allow_pickle=True)["meta"]))) for n in us}
    teach = [n for n in us if kind[n] in TEACH_KINDS]
    quiet = [n for n in us if kind[n] in QUIET_KINDS]
    T, W, U = {}, {}, {}
    for n in teach + quiet:
        T[n], W[n], u = in_child(read_site, n, "quiet" if n in quiet else "us", want_unl)
        if u is not None:
            U[n] = u
        print(f"read {n} (peak {check_mem(n):.1f} GB)", flush=True)
    islands = [n for n in abroad_sites() if n.startswith(("pr-", "vi-")) and have(n)]
    abroad = []
    for n in islands + [n for n in space_names() if have(n)]:
        r, w, _ = in_child(read_site, n, "abroad", False)
        if r is None:
            continue
        T[n], W[n] = r, w
        abroad.append(n)
        check_mem(n)
    return T, W, U, teach, quiet, abroad


def score_place(n, role, students, both, b):
    """In a child: load one place and save every student's scores there."""
    d = load(n)
    if d["kind"] == "space_quiet":
        d["y"] = np.where(d["y"] == 1, -1, d["y"]).astype(np.int8)
    li = labeled(d)
    for v, s in students.items():
        save(f"{OUT_DIR}/{v}", n, d["kind"], role, d, li, s, both, b)


def gate_a(aplain):
    return np.clip((aplain - 0.05) / 0.10, 0, 1)


def either_radar(both, aplain, tplain, flip, b):
    gt = np.clip((tplain - b[0]) / max(b[1] - b[0], 1e-6), 0, 1)
    return both * np.where(flip, gate_a(aplain), np.maximum(gate_a(aplain), gt))


def teacher_scores(both_model, r, b):
    """The joint model's score and the aggregate's on rows r (a subset dict)."""
    p = both_model.predict_proba(feats(r, np.arange(len(r["y"])), "both"))[:, 1]
    return {"both": p, "agg": either_radar(p, r["aplain"], r["tplain"], r["radar_flip"], b)}


def fit_student(parts):
    """parts: (X, p, w) per site, p the chance each row moved (0 or 1 for lidar labels). One logistic regression on
    two weighted rows per row, moved and unchanged totals balanced."""
    X = np.concatenate([q[0] for q in parts])
    p = np.concatenate([q[1] for q in parts]).astype(np.float64)
    w = np.concatenate([q[2] for q in parts]).astype(np.float64)
    wp, wn = w * p, w * (1 - p)
    tot = wp.sum() + wn.sum()
    wp *= tot / (2 * max(wp.sum(), 1e-9))
    wn *= tot / (2 * max(wn.sum(), 1e-9))
    keep_p, keep_n = wp > 0, wn > 0
    X2 = np.concatenate([X[keep_p], X[keep_n]])
    y2 = np.concatenate([np.ones(keep_p.sum(), int), np.zeros(keep_n.sum(), int)])
    w2 = np.concatenate([wp[keep_p], wn[keep_n]])
    return LogisticRegression(C=0.3, max_iter=3000).fit(X2, y2, sample_weight=w2)


def save(vdir, n, kind, role, d, li, student, both_model, b):
    os.makedirs(vdir, exist_ok=True)
    np.savez(f"{vdir}/{n}.npz", y=d["y"][li], kind=kind, role=role, bounds=np.array(b),
             aef=student.predict_proba(feats(d, li, "aef"))[:, 1].astype(np.float32),
             both=both_model.predict_proba(feats(d, li, "both"))[:, 1].astype(np.float32),
             aplain=d["aplain"][li], tplain=d["tplain"][li], flip=d["radar_flip"][li])


def run(names):
    T, W, U, teach, quiet, abroad = light_rows(any(VARIANTS[v].get("unl") for v in names))
    us = teach + quiet
    fold = {n: i % FOLDS for i, n in enumerate(sorted(abroad))}
    print(f"taught: {len(teach)} US digging, {len(quiet)} US quiet, {len(abroad)} abroad; unlabeled at {len(U)} sites; "
          f"variants {names}", flush=True)

    def bounds(train):
        ut = [n for n in train if n in us]
        return tes_gate_bounds(T, {n: np.arange(len(T[n]["y"])) for n in ut}, ut)

    def fit_fold(train):
        """The teacher and every student for one fold."""
        Xb = np.concatenate([feats(T[n], np.arange(len(T[n]["y"])), "both") for n in train])
        yb = np.concatenate([(T[n]["y"] == 1).astype(int) for n in train])
        wb = np.concatenate([W[n] for n in train])
        both = LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced").fit(Xb, yb, sample_weight=wb)
        del Xb
        b = bounds(train)
        ts = {n: teacher_scores(both, T[n], b) for n in train}
        tu = {n: teacher_scores(both, U[n], b) for n in train if n in U}
        students = {}
        for v in names:
            cfg = VARIANTS[v]
            a, tk = cfg["alpha"], cfg.get("teacher")
            parts = []
            for n in train:
                r = T[n]
                y = (r["y"] == 1).astype(np.float64)
                p = y if tk is None else a * y + (1 - a) * ts[n][tk]
                keep = np.ones(len(y), bool)
                if cfg.get("clean"):  # moved, but neither embedding changed
                    keep &= ~((r["y"] == 1) & (r["aplain"] < 0.05) & (r["tplain"] < b[0]))
                parts.append((feats(r, np.flatnonzero(keep), "aef"), p[keep], W[n][keep]))
                if cfg.get("unl") and n in tu:
                    parts.append((feats(U[n], np.arange(len(U[n]["y"])), "aef"), tu[n][tk], np.ones(len(U[n]["y"]))))
            students[v] = fit_student(parts)
            del parts
        return both, b, students

    def score(students, both, b, n, role):
        in_child(score_place, n, role, students, both, b)

    t0 = time.time()
    for n in us:
        both, b, students = fit_fold([m for m in T if m != n])
        score(students, both, b, n, "teach" if n in teach else "quiet")
        print(f"held out {n} ({time.time() - t0:.0f} s, peak {check_mem(n):.1f} GB)", flush=True)
    for k in range(FOLDS):
        out = [n for n in abroad if fold[n] == k]
        both, b, students = fit_fold([m for m in T if m not in out])
        for n in out:
            score(students, both, b, n, "abroad")
        print(f"held out fold {k} ({time.time() - t0:.0f} s, peak {check_mem(k):.1f} GB)", flush=True)
    both, b, students = fit_fold(list(T))
    have = lambda n: os.path.exists(f"{OUT}/{n}.npz")
    rest = [n for n in SITES + [s[0] for s in cdl_sets()] + abroad_sites() + place_names() + space_names()
            if have(n) and n not in T]
    for n in rest:
        score(students, both, b, n, "test")
        check_mem(n)
    print(f"test only: {len(rest)} places ({time.time() - t0:.0f} s)", flush=True)


def report(catch=0.6):
    """Each student gated as the map gates it, beside today's AlphaEarth model and the aggregate (base's teacher)."""
    have = [v for v in VARIANTS if os.path.isdir(f"{OUT_DIR}/{v}")]
    rows_ = [(v, "aef_gated") for v in have]
    if "base" in have:
        rows_.append(("base", "aggregate"))
    Z0 = None
    for v, rule in rows_:
        vd = f"{OUT_DIR}/{v}"
        Z = {f[:-4]: dict(np.load(f"{vd}/{f}", allow_pickle=True)) for f in sorted(os.listdir(vd)) if f.endswith(".npz")}
        S = {n: (z["aef"] * gate_a(z["aplain"]) if rule == "aef_gated"
                 else either_radar(z["both"], z["aplain"], z["tplain"], z["flip"], tuple(z["bounds"]))) for n, z in Z.items()}
        teach = [n for n, z in Z.items() if str(z["role"]) == "teach"]
        kinds = sorted({str(z["kind"]) for n, z in Z.items() if n not in teach})
        head = ["dig"] + kinds
        if Z0 is None:
            Z0 = head
            print(f"finding {catch:.0%} of moved ground at held-out digging sites: AP, share of unchanged ground lit up, moved found")
            print(f"  {'student':24s} {'AP':>5s}  " + " ".join(f"{k[:8]:>8s}" for k in head) + "  | found: island  space  | desert")
        t = cutoff([S[n][Z[n]["y"] == 1] for n in teach], catch)
        ap = np.mean([average_precision_score(Z[n]["y"], S[n]) for n in teach])
        fa, fd = {}, {}
        for n, z in Z.items():
            k = "dig" if n in teach else str(z["kind"])
            if (z["y"] == 0).sum() >= 10:
                fa.setdefault(k, []).append((S[n][z["y"] == 0] > t).mean())
            if n not in teach and (z["y"] == 1).sum() >= 10:
                fd.setdefault(k, []).append((S[n][z["y"] == 1] > t).mean())
        des = np.mean([(S[n] > t).mean() for n in DESERT if n in S])
        row = " ".join(f"{np.mean(fa[k]) if k in fa else float('nan'):8.3f}" for k in Z0)
        found = "  ".join(f"{np.mean(fd[k]) if k in fd else float('nan'):5.2f}" for k in ("island", "space_dig"))
        label = v if rule == "aef_gated" else "aggregate (both, teacher)"
        print(f"  {label:24s} {ap:5.3f}  {row}  |        {found}  | {des:.3f}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["report"]:
        report(float(sys.argv[2]) if len(sys.argv) > 2 else 0.6)
    else:
        sel = [v for v in os.environ.get("VARIANTS", ",".join(VARIANTS)).split(",") if v]
        run(sel)
