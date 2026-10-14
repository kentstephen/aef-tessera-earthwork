"""Where TESSERA leaves pixels out, worldwide: the ground the aggregate must leave to AEF alone.

TESSERA embeds a pixel only with 15 or more clear Sentinel-2 looks in the year. Random land points (land where
AEF has a pixel, less points TESSERA leaves out every year: AEF covers shallow seas) between 60S and 70N; at each, TESSERA's 32 x 32 pixel block for a few years: the share left
out, by latitude band.

Run: uv run python coverage.py   (writes results/coverage.json)
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import zarr
from obstore.store import HTTPStore, S3Store
from pyproj import Transformer
from zarr.storage import ObjectStore

from tessera_sites import AEF_RES, AEF_X0, AEF_Y0, TESSERA

HERE = os.path.dirname(os.path.abspath(__file__))
YEARS = (2017, 2020, 2022, 2024)
N = 600  # land points wanted; four times as many drawn, about a third land
BANDS = ((-60, -23.5), (-23.5, 23.5), (23.5, 45), (45, 70))
RNG = np.random.default_rng(0)


def main():
    zarr.config.set({"async.concurrency": 32})
    aef = zarr.open_group(ObjectStore(S3Store("tge-labs", endpoint="https://data.source.coop", region="us-west-2",
                                              virtual_hosted_style_request=False, skip_signature=True, prefix="aef-mosaic"),
                                      read_only=True), mode="r")
    times = aef["time"][:]
    ti = int(np.where(times == 2024)[0][0])
    tes = ObjectStore(HTTPStore.from_url(TESSERA), read_only=True)
    zones = {}

    LAT = np.degrees(np.arcsin(RNG.uniform(np.sin(np.radians(-60)), np.sin(np.radians(70)), N * 4)))
    LON = RNG.uniform(-180, 180, N * 4)

    def one(i):
        lat, lon = float(LAT[i]), float(LON[i])
        r, c = int((AEF_Y0 - lat) / AEF_RES), int((lon - AEF_X0) / AEF_RES)
        if aef["embeddings"][ti, 0, r, c] == -128:
            return None
        zn = f"utm{int((lon + 180) // 6) + 1:02d}"
        if zn not in zones:
            zones[zn] = zarr.open_group(tes, mode="r", path=zn)
        z = zones[zn]
        X, Y = Transformer.from_crs(4326, z.attrs["proj:code"], always_xy=True).transform(lon, lat)
        _, _, tx0, _, _, ty0 = z.attrs["spatial:transform"]
        tr, tc = int((ty0 - Y) / 10) // 32 * 32, int((X - tx0) / 10) // 32 * 32
        tt = list(z["time"][:])
        out = {"lat": lat, "lon": lon}
        for y in YEARS:
            s = z["scales"][tt.index(y), tr:tr + 32, tc:tc + 32]
            out[y] = float((~np.isfinite(s) | (s <= 0)).mean())
        return out

    pts = []
    with ThreadPoolExecutor(16) as ex:
        for p in ex.map(one, range(N * 4)):
            if p is not None:
                pts.append(p)
    # AEF also covers shallow seas; a point TESSERA leaves out in every year is taken as water
    pts = [p for p in pts if not all(p[y] == 1.0 for y in YEARS)]
    print(f"{len(pts)} land points")
    res = {"points": pts, "bands": {}}
    print(f"{'latitude':>14s}  n   " + "  ".join(f"{y}: mean / share of points > 10% missing" for y in YEARS))
    for lo, hi in BANDS + ((-60, 70),):
        sel = [p for p in pts if lo <= p["lat"] < hi]
        row = {y: [round(float(np.mean([p[y] for p in sel])), 3), round(float(np.mean([p[y] > 0.1 for p in sel])), 3)] for y in YEARS}
        res["bands"][f"{lo}..{hi}"] = {"n": len(sel), **row}
        print(f"{lo:6.1f} to {hi:5.1f}  {len(sel):3d}  " + "  ".join(f"{row[y][0]:.3f} / {row[y][1]:.2f}{'':26s}" for y in YEARS))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, "results", "coverage.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
