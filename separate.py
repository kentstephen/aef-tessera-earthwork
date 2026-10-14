"""Clearing that is digging against clearing that is not: signals in the embeddings that tell the two apart.

The aggregate lights clearcuts because forest cut to bare ground looks like the first stage of a mine, and every
clearcut taught as quiet costs forest mines (Kolwezi) in step. Here the question is asked directly: given ground
whose forest was cleared, can the embeddings say whether it was then dug?

Truth, same place and years on both sides:
  lidar   at the lidar sites with forest, Hansen's forest loss (v1.12) in a year of the lidar window, split by the
          lidar: cleared and dug (moved) against cleared and not dug (unchanged). Kentucky's coal, Hibbing's iron,
          and forest cleared for building in Florida, Alabama and Ohio
Then, to see whether what is learned there carries:
  cut     the ten Hansen clearcut regions of ground.py (US, Nordic, New Zealand, Brazil, Chile): cut ground
  space   the satellite-laser mines, moved and unchanged labels (Kolwezi, Fort Hills, Detour Lake, Kalimantan,
          Guinea, Peru and the rest of Earthwork's and forest_labels.py's sites)

Every year 2017 to 2025 of both embeddings is read under the kept pixels, with each pixel's surroundings (the
mean embedding within about 110 m and 510 m), so a signal can be the pair the map scores, what the ground became
in the years after (a cut regrows, a pit stays bare), or what lies around it.

Run: uv run python separate.py read [lidar|cut|space|name ...]   (into data/separate/, about an hour for all)
     uv run python separate.py test                              (writes results/separate.json)
"""

import json
import os
import sys
import time

import numpy as np
from scipy.ndimage import uniform_filter

import forest_labels as fl
import forest_mines  # noqa: F401  (site_path for the Kentucky lidar)
import ground
import tessera_sites as ts
from tessera_sites import AEF_RES, AEF_X0, AEF_Y0, OUT, PLACES_DIR, TESSERA, aef_unit, read_rc

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(HERE, "data", "separate")
YEARS = list(range(2017, 2026))
LIDAR = ["ky-clay", "ky-pike", "ky-perry", "hibbing-mn", "lakewood-ranch-fl", "four-corners-fl", "huntsville-al",
         "new-albany-oh"]
CUTS = [n for n, v in ground.PLACES.items() if v[0] == "cut"]
CTX = (11, 51)  # box sides in pixels for the surroundings: about 110 m and 510 m
MARGIN = CTX[-1] // 2
KEEP = {"dug": 15000, "kept": 15000, "other": 10000, "cut": 15000, "forest": 5000}
RNG = np.random.default_rng(0)


def hansen_at(rc):
    """Hansen's loss year (0 none, else year - 2000) and tree cover in 2000 under each AlphaEarth pixel of rc."""
    y0, y1, x0, x1 = rc
    lat = AEF_Y0 - (np.arange(y0, y1) + 0.5) * AEF_RES
    lon = AEF_X0 + (np.arange(x0, x1) + 0.5) * AEF_RES
    box = (lon[0] - 0.001, lat[-1] - 0.001, lon[-1] + 0.001, lat[0] + 0.001)

    def at(a, tr):
        rr = np.clip(((lat - tr.f) / tr.e).astype(int), 0, a.shape[0] - 1)
        cc = np.clip(((lon - tr.c) / tr.a).astype(int), 0, a.shape[1] - 1)
        return a[rr][:, cc].ravel()
    return at(*ground.hansen("lossyear", *box)), at(*ground.hansen("treecover2000", *box))


def pick(mask, n):
    i = np.flatnonzero(mask)
    return np.sort(RNG.choice(i, min(n, len(i)), replace=False)) if len(i) else i


def ctx_mean(v, ok, shape):
    """The mean of unit vectors v (n, d) over each box size in CTX, valid pixels only; (len(CTX), n, d)."""
    h, w = shape
    m = ok.reshape(h, w).astype(np.float32)
    x = (v * ok[:, None]).T.reshape(-1, h, w)
    out = []
    for k in CTX:
        cnt = np.maximum(uniform_filter(m, k, mode="constant"), 1e-6)
        s = np.stack([uniform_filter(b, k, mode="constant") for b in x])
        out.append((s / cnt).reshape(len(x), -1).T)
    return np.stack(out)


def read_window(g, name, rc, idx):
    """Both embeddings, every year, at the flat pixels idx of window rc, with their surroundings."""
    from compare import unit
    shape = (rc[1] - rc[0], rc[3] - rc[2])
    out = {k: [] for k in ("aef", "tes", "aok", "tok", "s1", "aef_ctx", "tes_ctx")}
    for y in YEARS:
        t = time.time()
        e, aok = aef_unit(y, rc)
        got = read_rc(g, name, rc, (y,), tags="y")
        sc = got["y_scales"]
        tok = np.isfinite(sc) & (sc > 0)
        tv = unit(np.nan_to_num(got["y_q"].astype(np.float32) * np.nan_to_num(sc)[:, None]))
        aok = aok & (np.linalg.norm(e, axis=1) > 0.5)
        out["aef"].append(e[idx].astype(np.float16))
        out["tes"].append(tv[idx].astype(np.float16))
        out["aok"].append(aok[idx])
        out["tok"].append(tok[idx])
        out["s1"].append((got["y_s1_asc_obs_count"].astype(np.int32) + got["y_s1_desc_obs_count"])[idx].astype(np.int16))
        out["aef_ctx"].append(ctx_mean(e, aok, shape)[:, idx].astype(np.float16))
        out["tes_ctx"].append(ctx_mean(tv, tok, shape)[:, idx].astype(np.float16))
        print(f"  {name} {y}: {time.time() - t:.0f} s", flush=True)
        del e, tv, got
    return {k: np.stack(v) for k, v in out.items()}


def read_lidar(g, n):
    t = json.loads(str(np.load(f"{OUT}/{n}.npz", allow_pickle=True)["meta"]))
    rc = tuple(t["rc"])
    y = np.load(ts.site_path(n), allow_pickle=True)["y"]
    loss, cover = hansen_at(rc)
    b, a = t["aef"]
    win = (loss > 0) & (loss >= b - 2000) & (loss <= a - 2000)
    grp = np.zeros(len(y), np.int8)  # 1 cleared and dug, 2 cleared and not dug, 3 other labeled ground
    grp[win & (y == 1)], grp[win & (y == 0)], grp[~win & (y >= 0)] = 1, 2, 3
    idx = np.sort(np.concatenate([pick(grp == 1, KEEP["dug"]), pick(grp == 2, KEEP["kept"]), pick(grp == 3, KEEP["other"])]))
    pad = (rc[0] - MARGIN, rc[1] + MARGIN, rc[2] - MARGIN, rc[3] + MARGIN)
    w = rc[3] - rc[2] + 2 * MARGIN
    pidx = (idx // (rc[3] - rc[2]) + MARGIN) * w + idx % (rc[3] - rc[2]) + MARGIN
    d = read_window(g, n, pad, pidx)
    return {**d, "grp": grp[idx], "y": y[idx], "loss": loss[idx], "cover": cover[idx], "idx": idx,
            "b": np.full(len(idx), b, np.int16), "a": np.full(len(idx), a, np.int16), "source": "lidar"}


def read_cut(g, n):
    p = np.load(f"{PLACES_DIR}/ground-{n}.npz", allow_pickle=True)
    meta = json.loads(str(p["meta"]))
    rc, grp = tuple(meta["rc"]), p["grp"]
    b, a = meta["aef"]
    idx = np.sort(np.concatenate([pick(grp == 1, KEEP["cut"]), pick(grp == 2, KEEP["forest"])]))
    pad = (rc[0] - MARGIN, rc[1] + MARGIN, rc[2] - MARGIN, rc[3] + MARGIN)
    w = rc[3] - rc[2] + 2 * MARGIN
    pidx = (idx // (rc[3] - rc[2]) + MARGIN) * w + idx % (rc[3] - rc[2]) + MARGIN
    d = read_window(g, f"ground-{n}", pad, pidx)
    # 4 cut that year, 5 standing forest
    return {**d, "grp": np.where(grp[idx] == 1, 4, 5).astype(np.int8), "y": np.zeros(len(idx), np.int8),
            "idx": idx, "b": np.full(len(idx), b, np.int16), "a": np.full(len(idx), a, np.int16), "source": "cut"}


# clearing near each mine with digging labels: the same forest, region and years, so the test is digging against
# clearing and not one part of the world against another. A 6 km square 15 to 60 km from the mine's labels
NEAR = ("space-kolwezi-cd", "space-fort-hills-ab", "space-detour-lake-on", "space-adaro-id", "space-sangatta-id",
        "space-madre-de-dios-pe", "space-boke-gn", "space-mobimbi-sl", "space-sembehun-sl", "space-weda-bay-id",
        "space-morowali-id")
NEAR_KM = (15, 60)


def near_name(mine):
    return "near-" + mine[len("space-"):]


def pick_near(mine):
    """The 6 km square 15 to 60 km from the mine whose tree cover lost in one year (2018 to 2023) is closest to
    half the square, kept inside one Hansen tile."""
    pos = space_pos(mine)
    lon = AEF_X0 + (pos[:, 1] + 0.5) * AEF_RES
    lat = AEF_Y0 - (pos[:, 0] + 0.5) * AEF_RES
    lo, la = float(lon.mean()), float(lat.mean())
    r = NEAR_KM[1] / 111.0
    kx = r / np.cos(np.radians(la))
    top, left = np.ceil(la / 10) * 10, np.floor(lo / 10) * 10
    box = (max(lo - kx, left + 1e-4), max(la - r, top - 10 + 1e-4), min(lo + kx, left + 10 - 1e-4), min(la + r, top - 1e-4))
    a, tr = ground.hansen("lossyear", *box)
    side = int(ground.HALF * 2 * AEF_RES / 0.00025)
    rr, cc = np.mgrid[0:a.shape[0], 0:a.shape[1]]
    X, Y = tr * (cc + 0.5, rr + 0.5)
    # km to the nearest label, on a coarse sample of the labels
    sub = np.linspace(0, len(lon) - 1, min(len(lon), 200)).astype(int)
    dk = np.full(a.shape, np.inf, np.float32)
    for i in sub:
        dk = np.minimum(dk, np.hypot((X - lon[i]) * 111 * np.cos(np.radians(la)), (Y - lat[i]) * 111))
    allowed = (dk >= NEAR_KM[0]) & (dk <= NEAR_KM[1] - 4)
    best = None
    for y in range(2018, 2024):
        f = uniform_filter((a == y - 2000).astype(np.float32), side, mode="constant")
        f = np.where(allowed, f, -9)
        i = np.unravel_index(np.argmin(np.abs(f - 0.5)), f.shape)
        if best is None or abs(f[i] - 0.5) < abs(best[0] - 0.5):
            best = (float(f[i]), i, y, float(dk[i]))
    share, (r_, c_), y, km = best
    x, yy = tr * (c_ + 0.5, r_ + 0.5)
    print(f"  {near_name(mine)}: {y} loss {share:.0%} of the square, {km:.0f} km from the mine", flush=True)
    return float(x), float(yy), y, km


def read_near(g, n):
    mine = "space-" + n[len("near-"):]
    lon, lat, y, km = pick_near(mine)
    r, c = int((AEF_Y0 - lat) / AEF_RES), int((lon - AEF_X0) / AEF_RES)
    H = ground.HALF
    rc = (r - H, r + H, c - H, c + H)
    grp = ground.groups("cut", rc, y)
    idx = np.sort(np.concatenate([pick(grp == 1, KEEP["cut"]), pick(grp == 2, KEEP["forest"])]))
    pad = (rc[0] - MARGIN, rc[1] + MARGIN, rc[2] - MARGIN, rc[3] + MARGIN)
    w = rc[3] - rc[2] + 2 * MARGIN
    pidx = (idx // (rc[3] - rc[2]) + MARGIN) * w + idx % (rc[3] - rc[2]) + MARGIN
    d = read_window(g, n, pad, pidx)
    return {**d, "grp": np.where(grp[idx] == 1, 4, 5).astype(np.int8), "y": np.zeros(len(idx), np.int8),
            "idx": idx, "b": np.full(len(idx), y - 1, np.int16), "a": np.full(len(idx), y + 1, np.int16),
            "source": "near", "point": np.array([lon, lat]), "km": km, "mine": mine}


def space_pos(n):
    short = n[len("space-"):]
    for d_ in (fl.DIR, ts.SPACE_DIR):
        if os.path.exists(f"{d_}/{short}.npz"):
            return np.load(f"{d_}/{short}.npz")["rc"]
    raise FileNotFoundError(n)


def read_space(g, n):
    pos = space_pos(n)
    y = np.load(forest_mines.site_path(n), allow_pickle=True)["y"]
    keep = np.flatnonzero(y >= 0)
    pos, y = pos[keep], y[keep]
    r0, r1 = pos[:, 0].min() - MARGIN, pos[:, 0].max() + 1 + MARGIN
    c0, c1 = pos[:, 1].min() - MARGIN, pos[:, 1].max() + 1 + MARGIN
    print(f"  {n}: {len(pos)} labels, window {r1 - r0} x {c1 - c0}", flush=True)
    assert (r1 - r0) * (c1 - c0) < 6e6, n
    pidx = (pos[:, 0] - r0) * (c1 - c0) + (pos[:, 1] - c0)
    d = read_window(g, n, (int(r0), int(r1), int(c0), int(c1)), pidx)
    return {**d, "grp": np.where(y == 1, 6, 7).astype(np.int8), "y": y, "idx": keep,
            "b": pos[:, 2].astype(np.int16), "a": pos[:, 3].astype(np.int16), "source": "space"}


def space_list():
    return [n for n in ts.space_names() + fl.names()]


def read(args):
    import zarr
    from obstore.store import HTTPStore
    from zarr.storage import ObjectStore
    zarr.config.set({"async.concurrency": 128})
    g = ObjectStore(HTTPStore.from_url(TESSERA), read_only=True)
    groups = {"lidar": LIDAR, "cut": CUTS, "space": space_list(), "near": [near_name(m) for m in NEAR]}
    names = [n for a in args for n in groups.get(a, [a])] or LIDAR + CUTS + space_list()
    os.makedirs(DIR, exist_ok=True)
    for n in names:
        path = f"{DIR}/{n}.npz"
        if os.path.exists(path):
            continue
        t = time.time()
        f = read_lidar if n in LIDAR else read_cut if n in CUTS else read_near if n.startswith("near-") else read_space
        d = f(g, n)
        np.savez(path, **d)
        print(f"{n}: {len(d['grp'])} px, groups {np.bincount(d['grp'], minlength=8)[1:].tolist()}, {time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "test"
    if cmd == "read":
        read(sys.argv[2:])
    else:
        import separate_test
        separate_test.main()
