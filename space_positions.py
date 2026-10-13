"""Where each of Earthwork's satellite-laser labels sits: AlphaEarth mosaic row, column and year pair.

Earthwork (earthwork_points_space.py) cached its ICESat-2 and GEDI pair labels as data/earthwork/space-<name>.npz,
one row per AlphaEarth pixel and year pair, without the pixel's position. This rebuilds the positions from the
same raw passes Earthwork cached (data/earthwork/space-raw, nothing downloaded) with site_points' own selection,
checks the labels and heights come out row for row as cached, and writes data/space/<name>.npz (row, col, yb, ya)
so TESSERA can be read under the same pixels.

Run (in Earthwork's environment, which holds its readers):
  uv run --project ../aef-earthwork python space_positions.py [site ...]
"""

import datetime
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EW = os.environ.get("EARTHWORK_REPO", os.path.join(HERE, "..", "aef-earthwork"))
sys.path.insert(0, EW)
import earthwork_model as em  # noqa: E402
import earthwork_points_space as sp  # noqa: E402

OUT = os.path.join(HERE, "data", "space")


def positions(name, kind, point, half):
    """site_points' selection, keeping each kept row's pixel and years (the same order as its cache)."""
    p = sp.site_pairs(name, kind, point, half, sp.SOURCES)
    if p is None or not (p["y_"] >= 0).any():
        return None
    keep = p["y_"] >= 0
    if kind not in sp.TEACH_KINDS:
        keep &= p["y_"] == 0
    p = {k: v[keep] for k, v in p.items()}
    lon = point[0] + p["x"] / (111320 * np.cos(np.radians(point[1])))
    lat = point[1] + p["y"] / 110574
    col = np.floor((lon - em.AEF_X0) / em.AEF_RES).astype(int)
    row = np.floor((em.AEF_Y0 - lat) / em.AEF_RES).astype(int)
    day = lambda d: datetime.date.fromisoformat(str(d))
    yrs = np.array([em.aef_years((day(a), day(a)), (day(b), day(b))) for a, b in zip(p["d1"], p["d2"])])
    ok = yrs[:, 1] > yrs[:, 0]
    rc = (row.min(), row.max() + 1, col.min(), col.max() + 1)
    out, emb = [], {}
    for yb, ya in sorted({tuple(v) for v in yrs[ok]}):
        sel = np.flatnonzero(ok & (yrs[:, 0] == yb) & (yrs[:, 1] == ya))
        _, inv = np.unique(np.c_[row[sel], col[sel]], axis=0, return_inverse=True)
        inv = inv.ravel()
        lo, hi = np.ones(inv.max() + 1), np.zeros(inv.max() + 1)
        np.maximum.at(hi, inv, p["y_"][sel])
        np.minimum.at(lo, inv, p["y_"][sel])
        sel = sel[~(hi != lo)[inv]]
        if not len(sel):
            continue
        order = sel[np.argsort(-np.abs(p["dz"][sel]))]
        _, first = np.unique(np.c_[row[order], col[order]], axis=0, return_index=True)
        sel = order[first]
        # the pixels AlphaEarth holds in both years (site_points drops the rest)
        for y_ in (yb, ya):
            if y_ not in emb:
                emb[y_] = em.aef_unit(y_, rc)[1]
        good = emb[yb][row[sel] - rc[0], col[sel] - rc[2]] & emb[ya][row[sel] - rc[0], col[sel] - rc[2]]
        sel = sel[good]
        out.append(np.c_[row[sel], col[sel], np.full(len(sel), yb), np.full(len(sel), ya), sel])
    out = np.concatenate(out)
    return out[:, :4], p["y_"][out[:, 4]].astype(np.int8), p["dz"][out[:, 4]].astype(np.float32)


def main():
    want = sys.argv[1:]
    os.makedirs(OUT, exist_ok=True)
    for name, kind, point, half in sp.SITES:
        cache = f"{em.DATA}/space-{name}.npz"
        if (want and name not in want) or not os.path.exists(cache):
            continue
        got = positions(name, kind, point, half)
        z = np.load(cache, allow_pickle=True)
        if got is None:
            print(f"{name}: no pairs")
            continue
        pos, y, dz = got
        same = len(y) == len(z["y"]) and (y == z["y"]).all() and np.allclose(dz, z["dz"], atol=1e-4)
        print(f"{name}: {len(y)} rows, cache {len(z['y'])}, labels and heights match: {same}")
        if same:
            np.savez_compressed(f"{OUT}/{name}.npz", rc=pos.astype(np.int64),
                                meta=json.dumps({"name": f"space-{name}", "kind": kind, "point": list(point)}))


if __name__ == "__main__":
    main()
