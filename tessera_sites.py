"""TESSERA under the Earthwork lidar sites, on AlphaEarth's 10 m grid.

Earthwork (github.com/kentstephen/aef-earthwork) cached, for each site where 3DEP flew the same ground twice,
the AlphaEarth embeddings of the year before the first flight (b) and the year of the second (a) and the
lidar's label per AlphaEarth pixel (moved > 0.5 m, unchanged < 0.1 m). This reads TESSERA v1.1 for the same
two years under the same pixels, so the two embeddings can be compared pixel for pixel against the same
truth.

TESSERA is stored per UTM zone on its own 10 m grid; each AlphaEarth pixel takes the TESSERA pixel under its
center. Kept per site: TESSERA as int8 with its per-pixel scale (as stored), and each year's Sentinel-1 and
Sentinel-2 observation counts, since a year with no radar (Sentinel-1B failed in December 2021 and
Sentinel-1C opened in 2025) can read as change in TESSERA when nothing moved.

Run: uv run python tessera_sites.py [site ...]   (EARTHWORK_DATA points at aef-earthwork's data/earthwork)
"""

import json
import os
import sys
import time

import numpy as np
import rasterio
import zarr
from obstore.store import HTTPStore
from pyproj import Transformer
from zarr.storage import ObjectStore

HERE = os.path.dirname(os.path.abspath(__file__))
EARTHWORK = os.environ.get("EARTHWORK_DATA", os.path.join(HERE, "..", "aef-earthwork", "data", "earthwork"))
OUT = os.path.join(HERE, "data", "tessera")
TESSERA = "https://data.source.coop/tessera/tessera/zarr/v1.1-dclimate"
AEF_RES, AEF_Y0, AEF_X0 = 8.983111749910169e-05, 83.68570533713473, -180.0

# the Earthwork sites compared: digging (building, mining), ground taught as not digging (burn, dunes), and
# ground that only tests: a storm coast, farmland and marshes flown twice. Fields and wetlands are never taught
# here, since fields so often become building sites
SITES = [
    "new-albany-oh", "katy-tx", "columbus-ne-oh", "aurora-co", "brighton-co", "lakewood-ranch-fl",
    "huntsville-al", "fishers-in", "las-vegas-se-nv", "las-vegas-nw-nv", "lancaster-ca",
    "mountain-pass-ca", "four-corners-fl", "hibbing-mn",
    "cameron-peak-co", "grizzly-flats-ca", "white-sands-nm", "mexico-beach-fl",
    "antelope-farm-ne", "box-butte-farm-ne", "fresno-farm-ca", "hamilton-farm-in", "arkansas-delta-ar",
    "othello-pivots-wa", "brazoria-tx", "anahuac-tx", "myakka-fl",
]
# crop switches (USDA Cropland Data Layer): pixels inside a field that grew a field crop every year of the
# pair, a different one at each end, sampled by Earthwork in a 6 km box around each point. Nothing was dug
CDL_REGIONS = {
    "tulare-ca": (-119.35, 36.20), "story-ia": (-93.45, 42.15), "haskell-ks-pivots": (-100.90, 37.60),
    "bolivar-ms": (-90.75, 33.45), "lubbock-tx": (-101.90, 34.00), "minidoka-id": (-113.70, 42.60),
    "red-river-mn": (-96.80, 47.50), "antelope-ne-pivots": (-98.05, 42.20), "box-butte-ne-pivots": (-102.95, 42.10),
    "dallam-tx-pivots": (-102.55, 36.20), "san-luis-co-pivots": (-105.95, 37.75), "imperial-ca": (-115.50, 32.90),
    "platte-ne-pivots": (-98.35, 40.85), "quincy-wa-pivots": (-119.60, 47.10), "darke-oh": (-84.60, 40.10),
    "tift-ga-pivots": (-83.60, 31.40),
}


def cdl_sets():
    """Every cached crop-switch set: (file name, region, before year, after year)."""
    out = []
    for f in sorted(os.listdir(EARTHWORK)):
        if f.startswith("cdl-") and f.endswith(".npz"):
            stem = f[4:-4]
            region, yb, ya = stem[:-10], int(stem[-9:-5]), int(stem[-4:])
            if region in CDL_REGIONS:
                out.append((f"cdl-{stem}", region, yb, ya))
    return out


def cdl_rc(point):
    """The 6 km box Earthwork drew crop switches from, as AlphaEarth mosaic rows and columns."""
    lon, lat = point
    dy, dx = 0.03, 0.03 / np.cos(np.radians(lat))
    return (int((AEF_Y0 - lat - dy) / AEF_RES), int((AEF_Y0 - lat + dy) / AEF_RES),
            int((lon - dx - AEF_X0) / AEF_RES), int((lon + dx - AEF_X0) / AEF_RES))


def shared_grid(sa, sb):
    """The 2 m grid over where two open tiles meet: west, south, east, north, width and height."""
    W, S = (np.ceil(max(u, v) / 2) * 2 for u, v in ((sa.bounds.left, sb.bounds.left), (sa.bounds.bottom, sb.bounds.bottom)))
    E, N = (np.floor(min(u, v) / 2) * 2 for u, v in ((sa.bounds.right, sb.bounds.right), (sa.bounds.top, sb.bounds.top)))
    return W, S, E, N, int((E - W) / 2), int((N - S) / 2)


def site_rc(meta):
    """The site's AlphaEarth mosaic rows and columns, from its two lidar tiles' headers (as Earthwork built it)."""
    u1, u2 = meta["tiles"]
    with rasterio.open("/vsicurl/" + u1) as sa, rasterio.open("/vsicurl/" + u2) as sb:
        W, S, E, N, w, h = shared_grid(sa, sb)
        inv = Transformer.from_crs(sa.crs, 4326, always_xy=True)
    c = [inv.transform(x, y) for x, y in ((W, N), (E, N), (W, S), (E, S))]
    w_, e_ = max(c[0][0], c[2][0]), min(c[1][0], c[3][0])
    n_, s_ = min(c[0][1], c[1][1]), max(c[2][1], c[3][1])
    x0, x1 = int(np.ceil((w_ - AEF_X0) / AEF_RES)), int((e_ - AEF_X0) / AEF_RES)
    y0, y1 = int(np.ceil((AEF_Y0 - n_) / AEF_RES)), int((AEF_Y0 - s_) / AEF_RES)
    return y0, y1, x0, x1


def read_rc(g, name, rc, years, idx=None):
    """TESSERA for the two years under AlphaEarth mosaic rows and columns rc (only the pixels idx of that box,
    flattened, when given): int8 (n, 128), scales and observation counts per year."""
    y0, y1, x0, x1 = rc
    LON, LAT = np.meshgrid(AEF_X0 + (np.arange(x0, x1) + 0.5) * AEF_RES, AEF_Y0 - (np.arange(y0, y1) + 0.5) * AEF_RES)
    if idx is not None:
        LON, LAT = LON.ravel()[idx], LAT.ravel()[idx]
    z = zone(g, float(LON.mean()))
    X, Y = Transformer.from_crs(4326, z.attrs["proj:code"], always_xy=True).transform(LON, LAT)
    _, _, tx0, _, _, ty0 = z.attrs["spatial:transform"]
    col = np.floor((X - tx0) / 10).astype(np.int64).ravel()
    row = np.floor((ty0 - Y) / 10).astype(np.int64).ravel()
    r0, r1, c0, c1 = row.min(), row.max() + 1, col.min(), col.max() + 1
    at = (row - r0) * (c1 - c0) + (col - c0)
    times = list(z["time"][:])
    out = {"zone": z.attrs["proj:code"]}
    for tag, year in zip("ba", years):
        t, ti = time.time(), times.index(year)
        q = z["embeddings"][ti, :, r0:r1, c0:c1]  # (128, h, w) int8
        out[f"{tag}_q"] = q.reshape(128, -1)[:, at].T.copy()  # (n, 128) int8
        for v in ("scales", "s2_obs_count", "s1_asc_obs_count", "s1_desc_obs_count"):
            out[f"{tag}_{v}"] = z[v][ti, r0:r1, c0:c1].ravel()[at]
        print(f"  {name} {year}: {q.shape[1]}x{q.shape[2]} TESSERA px, {time.time() - t:.0f} s")
    return out


def zone(g, lon):
    """The UTM zone group holding the longitude, read straight from the Zarr: many small reads at once (through
    xarray and dask the same window took 50 times longer)."""
    return zarr.open_group(g, mode="r", path=f"utm{int((lon + 180) // 6) + 1:02d}")


def read_site(g, name):
    path = f"{OUT}/{name}.npz"
    if os.path.exists(path):
        return path
    if name.startswith("cdl-"):
        _, region, yb, ya = next(s for s in cdl_sets() if s[0] == name)
        rc = cdl_rc(CDL_REGIONS[region])
        idx = np.load(f"{EARTHWORK}/{name}.npz")["idx"]
        meta = {"name": name, "kind": "crop", "aef": [yb, ya], "region": region}
    else:
        meta = json.loads(str(np.load(f"{EARTHWORK}/{name}.npz", allow_pickle=True)["meta"]))
        rc, idx = site_rc(meta), None
        assert [rc[1] - rc[0], rc[3] - rc[2]] == meta["shape"], (name, meta["shape"], rc)
    got = read_rc(g, name, rc, meta["aef"], idx)
    zone = got.pop("zone")
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(path, meta=json.dumps({**meta, "rc": list(rc), "zone": zone}), **got)
    return path


def main():
    zarr.config.set({"async.concurrency": 128})
    g = ObjectStore(HTTPStore.from_url(TESSERA), read_only=True)
    for name in sys.argv[1:] or SITES + [s[0] for s in cdl_sets()]:
        t = time.time()
        read_site(g, name)
        print(f"{name}: done in {time.time() - t:.0f} s")


if __name__ == "__main__":
    main()
