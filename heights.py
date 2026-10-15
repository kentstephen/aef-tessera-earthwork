"""Heights: does the height an embedding implies follow the lidar, and does a white roof move it?

AlphaEarth was taught to reconstruct the Copernicus DEM and GEDI canopy height, so its embedding carries a height,
though it sees only Sentinel-2, Landsat and Sentinel-1. Earthwork's relief notebook reads it back as the ghost: a
ridge regression from each 10 m pixel's embedding (the values and their squares) to the real elevation, each pixel
predicted by a model that never saw its 1 km block. Here the ghost is fitted on every year of a place at once against
one static DEM, then predicted for each year: the ghost's change between two years is the height change the
embedding implies, with no lidar in it. Two DEMs, each its own run:

  mapterhorn  Mapterhorn's terrarium tiles at zoom 13, as the notebook reads them
  3dep        USGS 3DEP from its S3 bucket: at each site the first flight's own 1 m DEM (the one the labels were
              cut from), in New York the seamless 1/3 arc-second DEM (about 10 m)
  copernicus  the Copernicus DEM GLO-30 from its AWS bucket: a surface model (buildings and trees in it) and the
              one AlphaEarth was taught to reconstruct

  sites   the 14 digging sites, against the lidar's own dz: does the ghost's change follow dz (correlation, and the
          sign on moved ground: cut or fill), and does its size separate moved from unchanged (AUC), beside plain
          change (1 - cosine) for comparison
  roofs   the seven New York squares of whiteroofs.py: the ghost's change on whitened, darkened and steady roofs
          and off roofs, 2022 to 2024, in the same meters

TESSERA was taught no height, but its ghost is fitted the same way, for comparison.

The head (heights.py head) asks it the other way round: not whether the embedding's own height moves, but whether
the lidar's meters can be learned from the embedding pair. A ridge regression on the pair's features (as the
Earthwork models take them: before, after, their product and squared difference) to the signed dz, and a boosted
tree on both embeddings joined, each held out by site over the 14 digging sites: does it follow dz, does it get cut
or fill right, and does its size separate moved from unchanged as well as plain change does.

Run: uv run python heights.py [3dep|copernicus]   (DEMs cached in data/heights, writes results/heights[_source].json)
     uv run python heights.py report [3dep|copernicus]
     uv run python heights.py head        (writes results/heights_head.json; head report)
"""

import io
import json
import os
import time
import urllib.request

import numpy as np
import rasterio
from PIL import Image
from pyproj import Transformer
from scipy import ndimage
from sklearn.metrics import roc_auc_score

import cities
import whiteroofs
from compare import site_path, teaching_names, TEACH_KINDS
from tessera_sites import AEF_RES, AEF_X0, AEF_Y0

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(HERE, "data", "heights")
SOURCE = "mapterhorn"
OUT = os.path.join(HERE, "results", "heights.json")
S3 = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation"
DEM_URL = "https://tiles.mapterhorn.com/{z}/{x}/{y}.webp"
DEM_Z = 13  # 512 px tiles, about 7 m a pixel at 40 N
SAMPLE = 200_000  # pixels each fold's ridge is fitted on
FOLDS = 5
BLOCK = 100  # 1 km blocks of 10 m pixels


def dem_tile(z, x, y):
    path = f"{DIR}/dem/{z}_{x}_{y}.npy"
    if os.path.exists(path):
        return np.load(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        req = urllib.request.Request(DEM_URL.format(z=z, x=x, y=y), headers={"User-Agent": "aef-tessera-earthwork"})
        b = urllib.request.urlopen(req, timeout=30).read()  # the default Python agent is refused
        p = np.asarray(Image.open(io.BytesIO(b)).convert("RGB")).astype(np.float32)
        v = p[..., 0] * 256 + p[..., 1] + p[..., 2] / 256 - 32768
    except urllib.error.HTTPError as e:
        if e.code != 404:  # no tile is sea; anything else is an error, not ground
            raise
        v = np.full((512, 512), np.nan, np.float32)
    np.save(path, v)
    return v


def dem_at(lon, lat, z=DEM_Z):
    """Elevation in meters at each lon, lat, bilinear from Mapterhorn's tiles."""
    n = 2**z * 512
    s = np.sin(np.radians(lat))
    px, py = (lon + 180) / 360 * n, (0.5 - np.log((1 + s) / (1 - s)) / (4 * np.pi)) * n
    tx0, tx1, ty0, ty1 = int(px.min() // 512), int(px.max() // 512), int(py.min() // 512), int(py.max() // 512)
    big = np.full(((ty1 - ty0 + 1) * 512, (tx1 - tx0 + 1) * 512), np.nan, np.float32)
    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            big[(ty - ty0) * 512:(ty - ty0 + 1) * 512, (tx - tx0) * 512:(tx - tx0 + 1) * 512] = dem_tile(z, tx, ty)
    return ndimage.map_coordinates(big, [py - ty0 * 512 - 0.5, px - tx0 * 512 - 0.5], order=1, mode="nearest")


def raster_at(url, lon, lat, res_m=5.0):
    """Elevation at each lon, lat from one 3DEP GeoTIFF on S3, read over their box at about res_m (averaged down
    from 1 m where finer) and sampled bilinear; NaN off the raster."""
    with rasterio.open("/vsicurl/" + url) as s:
        x, y = Transformer.from_crs(4326, s.crs, always_xy=True).transform(lon, lat)
        win = rasterio.windows.from_bounds(min(x), min(y), max(x), max(y), s.transform).round_offsets().round_lengths()
        win = rasterio.windows.Window(win.col_off - 2, win.row_off - 2, win.width + 4, win.height + 4)
        px = abs(s.transform.a) if s.crs.is_projected else abs(s.transform.a) * 111_320 * np.cos(np.radians(np.mean(lat)))
        f = max(1, int(res_m / px))
        a = s.read(1, window=win, boundless=True, fill_value=s.nodata if s.nodata is not None else -9999,
                   out_shape=(max(1, int(win.height // f)), max(1, int(win.width // f))),
                   resampling=rasterio.enums.Resampling.average).astype(np.float32)
        a[(a == s.nodata) | (a < -1000)] = np.nan
        wt = s.window_transform(win) * rasterio.Affine.scale(win.width / a.shape[1], win.height / a.shape[0])
    col, row = ~wt * (np.asarray(x), np.asarray(y))
    return ndimage.map_coordinates(a, [row - 0.5, col - 0.5], order=1, mode="constant", cval=np.nan)


def seamless_at(lon, lat):
    """3DEP's seamless 1/3 arc-second DEM, from every one-degree tile under the points."""
    out = np.full(len(lon), np.nan, np.float32)
    for n in range(int(np.ceil(lat.min())), int(np.ceil(lat.max())) + 1):
        for w in range(int(np.ceil(-lon.max())), int(np.ceil(-lon.min())) + 1):
            m = (lat > n - 1) & (lat <= n) & (-lon > w - 1) & (-lon <= w)
            if m.any():
                out[m] = raster_at(f"{S3}/13/TIFF/current/n{n:02d}w{w:03d}/USGS_13_n{n:02d}w{w:03d}.tif", lon[m], lat[m], 10.0)
    return out


def copernicus_at(lon, lat):
    """The Copernicus DEM GLO-30, from every one-degree tile under the points."""
    out = np.full(len(lon), np.nan, np.float32)
    for n in range(int(np.floor(lat.min())), int(np.floor(lat.max())) + 1):
        for e in range(int(np.floor(lon.min())), int(np.floor(lon.max())) + 1):
            m = (lat >= n) & (lat < n + 1) & (lon >= e) & (lon < e + 1)
            if m.any():
                t = f"Copernicus_DSM_COG_10_{'N' if n >= 0 else 'S'}{abs(n):02d}_00_{'E' if e >= 0 else 'W'}{abs(e):03d}_00_DEM"
                out[m] = raster_at(f"https://copernicus-dem-30m.s3.amazonaws.com/{t}/{t}.tif", lon[m], lat[m], 30.0)
    return out


def elevation(name, lon, lat, tiles=None):
    """The DEM SOURCE gives for a place, cached."""
    path = f"{DIR}/{SOURCE}_{name}.npy"
    if os.path.exists(path):
        return np.load(path)
    if SOURCE == "mapterhorn":
        v = dem_at(lon, lat)
    elif SOURCE == "copernicus":
        v = copernicus_at(lon, lat)
    else:
        v = raster_at(tiles[0], lon, lat) if tiles else seamless_at(lon, lat)
    os.makedirs(DIR, exist_ok=True)
    np.save(path, v.astype(np.float32))
    return v


def lonlat(rc):
    r0, r1, c0, c1 = (int(v) for v in rc)
    lon, lat = np.meshgrid(AEF_X0 + (np.arange(c0, c1) + 0.5) * AEF_RES, AEF_Y0 - (np.arange(r0, r1) + 0.5) * AEF_RES)
    return lon.ravel(), lat.ravel(), (r1 - r0, c1 - c0)


def feats(E):
    E = E.astype(np.float32)
    return np.hstack([E, E * E, np.ones((len(E), 1), np.float32)])


def ghost(years, elev, shape):
    """years: a list of (embeddings (n, d), valid (n,)) on one grid. One ridge per fold, fitted on every year pooled
    against the static elev, predicts each year out of fold over 1 km blocks: a list of (n,) heights (NaN where not
    valid) and the fit's R2 over all years."""
    h, w = shape
    br, bc = np.meshgrid(np.arange(h) // BLOCK, np.arange(w) // BLOCK, indexing="ij")
    fold = ((br * 7 + bc * 3) % FOLDS).ravel().astype(np.int8)
    rng = np.random.default_rng(0)
    rows = [(k, np.flatnonzero(ok & np.isfinite(elev))) for k, (_, ok) in enumerate(years)]
    per = max(SAMPLE // len(years), 1)
    pick = [(k, rng.choice(i, min(per, len(i)), replace=False)) for k, i in rows if len(i)]
    X = np.concatenate([feats(years[k][0][i]) for k, i in pick])
    y = np.concatenate([elev[i] for _, i in pick])
    f = np.concatenate([fold[i] for _, i in pick])
    W = []
    for k in range(FOLDS):
        A = X[f != k]
        G = A.T @ A
        G[np.diag_indices_from(G)] += 1e-3 * np.mean(np.diag(G))
        W.append(np.linalg.solve(G, A.T @ y[f != k]))
    W = np.stack(W, 1).astype(np.float32)
    out = []
    for E, ok in years:
        g = np.full(len(E), np.nan, np.float32)
        for i in range(0, len(E), 200_000):  # in pieces, so the features never sit whole
            P = feats(E[i:i + 200_000]) @ W
            g[i:i + 200_000] = P[np.arange(len(P)), fold[i:i + 200_000]]
        g[~ok] = np.nan
        out.append(g)
    m = np.isfinite(elev) & np.all([np.isfinite(g) for g in out], 0)
    r = np.concatenate([g[m] - elev[m] for g in out])
    return out, float(1 - np.var(r) / np.var(np.tile(elev[m], len(out))))


def tes_unit(t, p):
    s = t[f"{p}_scales"]
    ok = np.isfinite(s) & (s > 0)
    return cities.unit(np.nan_to_num(t[f"{p}_q"].astype(np.float32) * s[:, None])).astype(np.float16), ok


def site(n):
    z = np.load(site_path(n), allow_pickle=True)
    t = np.load(f"{HERE}/data/tessera/{n}.npz", allow_pickle=True)
    meta = json.loads(str(t["meta"]))
    lon, lat, shape = lonlat(meta["rc"])
    elev = elevation(n, lon, lat, meta["tiles"])
    X = z["X"]
    ab, aa = X[:, :64], X[:, 64:128]
    okb, oka = np.linalg.norm(ab.astype(np.float32), axis=1) > 0.5, np.linalg.norm(aa.astype(np.float32), axis=1) > 0.5
    y, dz, plain = z["y"], z["dz"], z["plain"]
    del X
    (gb, ga), r2a = ghost([(ab, okb), (aa, oka)], elev, shape)
    del ab, aa
    (tb, tob), (ta, toa) = tes_unit(t, "b"), tes_unit(t, "a")
    (hb, ha), r2t = ghost([(tb, tob), (ta, toa)], elev, shape)
    tplain = 1 - (tb.astype(np.float32) * ta.astype(np.float32)).sum(1)
    del tb, ta
    out = {"kind": meta["kind"], "aef_years": meta["aef"], "r2": {"aef": r2a, "tes": r2t}}
    for k, (b, a, pl) in (("aef", (gb, ga, plain)), ("tes", (hb, ha, tplain))):
        d = a - b
        lab = np.isfinite(d) & (y >= 0) & np.isfinite(dz)
        mv = lab & (y == 1)
        out[k] = {
            "moved": int(mv.sum()), "unchanged": int((lab & (y == 0)).sum()),
            "corr_dz": float(np.corrcoef(d[lab], dz[lab])[0, 1]),
            "sign_moved": float((np.sign(d[mv]) == np.sign(dz[mv])).mean()) if mv.any() else None,
            "auc_ghost": float(roc_auc_score(y[lab], np.abs(d[lab]))) if mv.any() else None,
            "auc_plain": float(roc_auc_score(y[lab], pl[lab])) if mv.any() else None,
            "abs_moved": float(np.median(np.abs(d[mv]))) if mv.any() else None,
            "abs_unchanged": float(np.median(np.abs(d[lab & (y == 0)]))),
            "dz_moved": float(np.median(np.abs(dz[mv]))) if mv.any() else None,
            # the ghost's change on moved ground, regressed on dz: meters of ghost per meter dug
            "slope": float(np.polyfit(dz[mv], d[mv], 1)[0]) if mv.sum() > 100 else None,
        }
    return out


def roofs(n):
    z = dict(np.load(f"{cities.DIR}/{n}.npz"))
    lon, lat, shape = lonlat(z["rc"])
    elev = elevation(n, lon, lat)
    roof, _ = whiteroofs.roofs(n, lon, lat)
    vis = whiteroofs.visible(n, lon, lat)
    gs, r2a = ghost([(z[f"aef_{y}"], z[f"aok_{y}"]) for y in cities.YEARS], elev, shape)
    tes = []
    for y in cities.YEARS:
        s = z[f"{y}_scales"]
        tes.append((cities.unit(np.nan_to_num(z[f"{y}_q"].astype(np.float32) * s[:, None])).astype(np.float16),
                    np.isfinite(s) & (s > 0)))
    hs, r2t = ghost(tes, elev, shape)
    del tes
    G = {"aef": dict(zip(cities.YEARS, gs)), "tes": dict(zip(cities.YEARS, hs))}
    out = {"r2": {"aef": r2a, "tes": r2t}, "pairs": {}}
    on = roof >= 0
    for yb, ya in whiteroofs.PAIRS:
        vb, va = vis[str(yb)], vis[str(ya)]
        rise = va - vb
        groups = {
            "whitened": on & (rise >= whiteroofs.WHITE_RISE) & (vb < whiteroofs.DARK_BEFORE) & (va > whiteroofs.BRIGHT_AFTER),
            "darkened": on & (rise <= -whiteroofs.WHITE_RISE) & (va < whiteroofs.DARK_BEFORE) & (vb > whiteroofs.BRIGHT_AFTER),
            "steady": on & (np.abs(rise) < whiteroofs.STEADY), "off_roofs": ~on}
        row = {}
        for k in ("aef", "tes"):
            d = G[k][ya] - G[k][yb]
            row[k] = {g: {"px": int((m & np.isfinite(d)).sum()), "abs": float(np.median(np.abs(d[m & np.isfinite(d)]))) if (m & np.isfinite(d)).any() else None,
                          "signed": float(np.median(d[m & np.isfinite(d)])) if (m & np.isfinite(d)).any() else None}
                      for g, m in groups.items()}
        out["pairs"][f"{yb}-{ya}"] = row
    return out


def main():
    res = {"sites": {}, "roofs": {}}
    for n in teaching_names():
        if json.loads(str(np.load(f"{HERE}/data/tessera/{n}.npz", allow_pickle=True)["meta"]))["kind"] not in TEACH_KINDS:
            continue
        t = time.time()
        res["sites"][n] = site(n)
        a = res["sites"][n]["aef"]
        print(f"site {n}: R2 {res['sites'][n]['r2']['aef']:.2f}, AUC ghost {a['auc_ghost']:.3f} plain {a['auc_plain']:.3f}, "
              f"corr dz {a['corr_dz']:.2f}, {time.time() - t:.0f} s", flush=True)
        json.dump(res, open(OUT, "w"), indent=1)
    for n in whiteroofs.SQUARES:
        t = time.time()
        res["roofs"][n] = roofs(n)
        print(f"roofs {n}: R2 {res['roofs'][n]['r2']['aef']:.2f}, {time.time() - t:.0f} s", flush=True)
        json.dump(res, open(OUT, "w"), indent=1)
    report(res)


def report(res=None):
    res = res or json.load(open(OUT))
    S = res["sites"]
    print("\ndigging sites: the ghost's change against the lidar (moved |dz| > 0.5 m, unchanged |dz| < 0.1 m)")
    print(f"  {'site':22s} {'R2':>5s}  " + "  ".join(f"{k + ':':4s} {'AUC gh':>6s} {'plain':>6s} {'corr':>5s} {'sign':>5s} {'|d| mv':>6s} {'|d| un':>6s} {'slope':>6s}" for k in ("aef", "tes")) + f"  {'|dz| mv':>7s}")
    for n, v in S.items():
        cells = []
        for k in ("aef", "tes"):
            a = v[k]
            f = lambda x, w=6, p=3: f"{x:{w}.{p}f}" if x is not None else f"{'-':>{w}s}"
            cells.append(f"{'':4s} {f(a['auc_ghost'])} {f(a['auc_plain'])} {f(a['corr_dz'], 5, 2)} {f(a['sign_moved'], 5, 2)} {f(a['abs_moved'], 6, 2)} {f(a['abs_unchanged'], 6, 2)} {f(a['slope'], 6, 2)}")
        print(f"  {n:22s} {v['r2']['aef']:5.2f}  " + "  ".join(cells) + f"  {v['aef']['dz_moved']:7.2f}")
    for k in ("aef", "tes"):
        m = lambda key: np.mean([v[k][key] for v in S.values() if v[k][key] is not None])
        print(f"  mean {k}: AUC ghost {m('auc_ghost'):.3f}  plain {m('auc_plain'):.3f}  corr {m('corr_dz'):.2f}  sign {m('sign_moved'):.2f}  "
              f"|d| moved {m('abs_moved'):.2f} m  unchanged {m('abs_unchanged'):.2f} m")
    R = res["roofs"]
    print("\nNew York roofs: median |ghost change| in meters (median signed), pooled pixel-weighted over squares")
    for p in [f"{a}-{b}" for a, b in whiteroofs.PAIRS]:
        for k in ("aef", "tes"):
            cells = []
            for g in ("whitened", "darkened", "steady", "off_roofs"):
                rows = [v["pairs"][p][k][g] for v in R.values() if v["pairs"][p][k][g]["px"]]
                px = sum(r["px"] for r in rows)
                ab = sum(r["abs"] * r["px"] for r in rows) / max(px, 1)
                sg = sum(r["signed"] * r["px"] for r in rows) / max(px, 1)
                cells.append(f"{g} {ab:5.2f} ({sg:+.2f}, {px} px)")
            print(f"  {p} {k}: " + "  ".join(cells))
    print("  R2 per square (aef/tes): " + ", ".join(f"{n} {v['r2']['aef']:.2f}/{v['r2']['tes']:.2f}" for n, v in R.items()))


HEAD_OUT = os.path.join(HERE, "results", "heights_head.json")
MOVED, CALM, BETWEEN = 15_000, 15_000, 5_000  # pixels sampled a site: moved, unchanged and in between
DZ_CLIP = 15.0  # meters: a deep mine's cut does not swamp the rest


def head_rows(n):
    """One digging site's sampled pixels: the features of each embedding, dz, the class and plain change."""
    from compare import feats as pair_feats, load
    d = load(n)
    dz = np.load(site_path(n), allow_pickle=True)["dz"]
    ok = d["ok"] & np.isfinite(dz)
    rng = np.random.default_rng(0)
    pick = []
    for m, k in ((ok & (np.abs(dz) > 0.5), MOVED), (ok & (np.abs(dz) < 0.1), CALM), (ok & (np.abs(dz) >= 0.1) & (np.abs(dz) <= 0.5), BETWEEN)):
        i = np.flatnonzero(m)
        pick.append(rng.choice(i, min(k, len(i)), replace=False))
    idx = np.sort(np.concatenate(pick))
    return {"X": {w: pair_feats(d, idx, w).astype(np.float32) for w in ("aef", "tes", "both")}, "dz": dz[idx].astype(np.float32),
            "aplain": d["aplain"][idx], "tplain": d["tplain"][idx]}


def head_metrics(pred, dz, plain):
    mv, calm = np.abs(dz) > 0.5, np.abs(dz) < 0.1
    lab = mv | calm
    return {"corr_moved": float(np.corrcoef(pred[mv], dz[mv])[0, 1]), "corr_all": float(np.corrcoef(pred, dz)[0, 1]),
            "sign_moved": float((np.sign(pred[mv]) == np.sign(dz[mv])).mean()),
            "auc_size": float(roc_auc_score(mv[lab], np.abs(pred[lab]))), "auc_plain": float(roc_auc_score(mv[lab], plain[lab])),
            "abs_moved": float(np.median(np.abs(pred[mv]))), "abs_calm": float(np.median(np.abs(pred[calm]))),
            "dz_moved": float(np.median(np.abs(dz[mv]))), "moved": int(mv.sum())}


def head():
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    names = [n for n in teaching_names() if json.loads(str(np.load(f"{HERE}/data/tessera/{n}.npz", allow_pickle=True)["meta"]))["kind"] in TEACH_KINDS]
    R = {}
    for n in names:
        t = time.time()
        R[n] = cities.in_child(head_rows, n)  # a child per site: the full load's memory goes back when it ends
        print(f"rows {n}: {len(R[n]['dz'])} px, {time.time() - t:.0f} s", flush=True)
    fits = [("ridge", w) for w in ("aef", "tes", "both")] + [("hgb", "both")]
    # every site's rows in one array per embedding, the held-out site left out by a mask or a zero weight
    # (no copy of the training rows for each fold)
    site = np.concatenate([np.full(len(R[n]["dz"]), k) for k, n in enumerate(names)])
    Y = np.clip(np.concatenate([R[n]["dz"] for n in names]), -DZ_CLIP, DZ_CLIP)
    XS = {w: np.concatenate([R[n]["X"][w] for n in names]) for w in ("aef", "tes", "both")}
    for n in names:
        R[n].pop("X")
    import gc
    import resource
    res = {}
    for k, n in enumerate(names):
        res[n] = {}
        held = site == k
        for how, w in fits:
            t = time.time()
            if how == "ridge":
                mdl = Ridge(alpha=10.0).fit(XS[w][~held], Y[~held])
            else:
                mdl = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, random_state=0, early_stopping=False)
                mdl.fit(XS[w], Y, sample_weight=(~held).astype(np.float32))
            pred = mdl.predict(XS[w][held]).astype(np.float32)
            del mdl
            gc.collect()
            res[n][f"{how}:{w}"] = head_metrics(pred, R[n]["dz"], R[n]["tplain"] if w == "tes" else R[n]["aplain"])
            r = res[n][f"{how}:{w}"]
            print(f"{n} {how}:{w}: corr moved {r['corr_moved']:.2f}, sign {r['sign_moved']:.2f}, AUC size {r['auc_size']:.3f} "
                  f"(plain {r['auc_plain']:.3f}), {time.time() - t:.0f} s, peak {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.1f} GB", flush=True)
        json.dump(res, open(HEAD_OUT, "w"), indent=1)
    head_report(res)


def head_report(res=None):
    res = res or json.load(open(HEAD_OUT))
    keys = list(next(iter(res.values())))
    print("\nthe lidar's meters learned from the embedding pair, each site held out (moved |dz| > 0.5 m, unchanged < 0.1 m)")
    print(f"  {'site':22s} " + "  ".join(f"{k:>34s}" for k in keys) + f"  {'|dz| mv':>7s}")
    print(f"  {'':22s} " + "  ".join(f"{'corr  sign  AUC  plain  |p|mv |p|un':>34s}" for _ in keys))
    for n, v in res.items():
        print(f"  {n:22s} " + "  ".join(f"{v[k]['corr_moved']:5.2f} {v[k]['sign_moved']:5.2f} {v[k]['auc_size']:5.3f} {v[k]['auc_plain']:5.3f} {v[k]['abs_moved']:5.2f} {v[k]['abs_calm']:5.2f}" for k in keys)
              + f"  {v[keys[0]]['dz_moved']:7.2f}")
    for k in keys:
        m = lambda key: np.mean([v[k][key] for v in res.values()])
        print(f"  mean {k:11s}: corr moved {m('corr_moved'):.2f}, all {m('corr_all'):.2f}, sign {m('sign_moved'):.2f}, "
              f"AUC size {m('auc_size'):.3f} against plain {m('auc_plain'):.3f}, |pred| moved {m('abs_moved'):.2f} m, unchanged {m('abs_calm'):.2f} m")


if __name__ == "__main__":
    import sys
    a = sys.argv[1:]
    for src in ("3dep", "copernicus"):
        if src in a:
            SOURCE, OUT = src, os.path.join(HERE, "results", f"heights_{src}.json")
    if "head" in a:
        head_report() if "report" in a else head()
    else:
        report() if "report" in a else main()
