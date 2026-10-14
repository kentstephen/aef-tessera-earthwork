"""Do the models score the same from AlphaEarth's native UTM COGs (tge-labs/aef) as from the lat/lon mosaic they were
taught on (tge-labs/aef-mosaic)? The browser would read the COGs: they sit on TESSERA's grid pixel for pixel.

At each digging site every mosaic pixel center is placed in the site's UTM zone and the COG pixel under it is read for
both years (TESSERA stays as read, at the same centers). Reported: how close the two AEF vectors are, and the joint
model x radar-safe gate (abroad recipe's final models, the rule of record) scored from each, against the lidar.
The final models were taught on these sites, so the AP is in sample: what counts is the difference between inputs.

Run: uv run python cog_vs_mosaic.py   (writes results/cog_vs_mosaic.json)
"""

import asyncio
import json
import os
import subprocess
import sys

import numpy as np
import pyarrow.parquet as pq
from pyproj import Transformer
from sklearn.metrics import average_precision_score

from compare import TEACH_KINDS, feats, load
from models import final
from quick import RULES
from tessera_sites import AEF_RES, AEF_X0, AEF_Y0, OUT, SITES

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "pyodide_probe"))
import both_reader as R  # noqa: E402

INDEX = os.path.join(HERE, "data", "aef_index.parquet")
TILE_M = 81920  # 8192 px of 10 m


def index():
    if not os.path.exists(INDEX):  # urllib is refused (403); curl is not
        subprocess.run(["curl", "-s", "-o", INDEX, f"{R.AEF}/aef_index.parquet"], check=True)
    return pq.read_table(INDEX, columns=["year", "utm_zone", "utm_west", "utm_south", "path"]).to_pandas()


def dequant_unit(q):
    """(n, 64) int8 -> unit float32, NaN rows where masked (the dequantization Earthwork's cache used)."""
    v = q.astype(np.float32)
    v = np.sign(v) * (v / 127.5) ** 2
    v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
    v[q[:, 0] == -128] = np.nan
    return v


async def cog_at(ix, zone, year, X, Y):
    """The COG's int8 vector (n, 64) under each UTM point, reading each tile's window around the points it holds."""
    out = np.full((len(X), 64), -128, np.int8)
    rows = ix[(ix.utm_zone == zone) & (ix.year == year)]
    for t in rows.itertuples():
        col = np.floor((X - t.utm_west) / 10).astype(np.int64)
        fr = np.floor((Y - t.utm_south) / 10).astype(np.int64)  # file row, counted from the south edge
        m = (col >= 0) & (col < 8192) & (fr >= 0) & (fr < 8192)
        if not m.any():
            continue
        r0, r1, c0, c1 = fr[m].min(), fr[m].max() + 1, col[m].min(), col[m].max() + 1
        url = t.path.replace("s3://us-west-2.opendata.source.coop/tge-labs/aef/v1/annual", R.AEF)
        w = await R.aef_window(url, int(r0), int(c0), int(r1 - r0), int(c1 - c0))  # north-up
        out[m] = w[:, (r1 - 1) - fr[m], col[m] - c0].T
    return out


def score(models, bounds, d, li):
    s = {w: models[w].predict_proba(feats(d, li, w))[:, 1] for w in models}
    s["aplain"], s["tplain"], s["flip"] = d["aplain"][li], d["tplain"][li], d["radar_flip"][li]
    return RULES["either_radar"](s, bounds), RULES["aef_gated"](s, bounds)


def main():
    ix = index()
    m = final("abroad")
    models, bounds = m["models"], tuple(m["bounds"])
    res = {}
    names = sys.argv[1:] or SITES
    for name in names:
        meta = json.loads(str(np.load(f"{OUT}/{name}.npz", allow_pickle=True)["meta"]))
        if meta.get("kind") not in TEACH_KINDS:
            continue
        d = load(name)
        y0, y1, x0, x1 = meta["rc"]
        LON, LAT = np.meshgrid(AEF_X0 + (np.arange(x0, x1) + 0.5) * AEF_RES, AEF_Y0 - (np.arange(y0, y1) + 0.5) * AEF_RES)
        epsg = int(meta["zone"].split(":")[1])
        zone = f"{epsg % 100}{'N' if epsg < 32700 else 'S'}"
        X, Y = Transformer.from_crs(4326, epsg, always_xy=True).transform(LON.ravel(), LAT.ravel())
        R.stats.update(gets=0, bytes=0)
        cb, ca = (dequant_unit(asyncio.run(cog_at(ix, zone, yr, X, Y))) for yr in meta["aef"])
        ok = d["ok"] & np.isfinite(cb).all(1) & np.isfinite(ca).all(1)
        mb, ma = d["ab"].astype(np.float32), d["aa"].astype(np.float32)
        cos_b, cos_a = (mb * cb).sum(1)[ok], (ma * ca).sum(1)[ok]
        li = np.flatnonzero(ok & (d["y"] >= 0))
        lab = d["y"][li] == 1
        s_m, a_m = score(models, bounds, d, li)
        dc = {**d, "ab": cb.astype(np.float16), "aa": ca.astype(np.float16), "aplain": 1 - (cb * ca).sum(1)}
        s_c, a_c = score(models, bounds, dc, li)
        cut = m["cutoff"][0.6]["either_radar"]
        r = {
            "pixels": int(ok.sum()), "mb_read": round(R.stats["bytes"] / 1e6),
            "identical_vectors": round(float(((cos_b > 0.99999) & (cos_a > 0.99999)).mean()), 4),
            "cos_median": round(float(np.median(np.r_[cos_b, cos_a])), 5), "cos_p01": round(float(np.quantile(np.r_[cos_b, cos_a], 0.01)), 4),
            "ap_mosaic": round(average_precision_score(lab, s_m), 4), "ap_cog": round(average_precision_score(lab, s_c), 4),
            "ap_aef_mosaic": round(average_precision_score(lab, a_m), 4), "ap_aef_cog": round(average_precision_score(lab, a_c), 4),
            "score_rho": round(float(np.corrcoef(np.argsort(np.argsort(s_m)), np.argsort(np.argsort(s_c)))[0, 1]), 4),
            "found_mosaic": round(float((s_m[lab] > cut).mean()), 4), "found_cog": round(float((s_c[lab] > cut).mean()), 4),
            "lit_mosaic": round(float((s_m[~lab] > cut).mean()), 4), "lit_cog": round(float((s_c[~lab] > cut).mean()), 4),
        }
        res[name] = r
        print(name, r, flush=True)
    keys = [k for k in next(iter(res.values())) if k not in ("pixels", "mb_read")]
    res["mean"] = {k: round(float(np.mean([v[k] for v in res.values()])), 4) for k in keys}
    print("mean", res["mean"])
    if len(names) == len(SITES):
        json.dump(res, open(os.path.join(HERE, "results", "cog_vs_mosaic.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
