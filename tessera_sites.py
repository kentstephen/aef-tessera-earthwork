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

Run: uv run python tessera_sites.py [site ... | abroad | places]   (EARTHWORK_DATA points at aef-earthwork's
     data/earthwork; "abroad" reads every cached site outside the 48 states, "places" the places with no truth,
     reading their AlphaEarth too, "space" the satellite-laser sites space_positions.py placed)
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


# ground outside the 48 states that Earthwork tested on, all test only here. Puerto Rico and the Virgin Islands:
# 3DEP 2018 against 2024, moved ground over 2 m counted (shallower moves there are mostly lidar ground under
# forest). Reunion and Guadeloupe (IGN France) and Hong Kong (CEDD): the first survey predates AlphaEarth, so only
# their unchanged ground counts. Shizuoka: small survey windows, both ways
ISLAND_DEEP_M = 2.0
INTL_POINTS = {  # Earthwork's earthwork_sites_intl.py: a 4 km square about each point on a 2 m grid
    "reunion-sud-cane": (55.46, -21.29), "reunion-est-forest": (55.62, -21.12), "reunion-ouest-port": (55.30, -20.96),
    "reunion-nord-city": (55.47, -20.90), "guadeloupe-moule-cane": (-61.39, 16.33),
    "guadeloupe-capesterre": (-61.58, 16.05), "guadeloupe-pointe": (-61.52, 16.25),
}
# places with no truth Earthwork looked at, AlphaEarth 2023 against 2025 in a 3 km window (earthwork_abroad.py)
PLACES = {
    "nusantara-id": (116.70, -0.97), "manila-bay-ph": (120.975, 14.52), "lekki-ng": (3.98, 6.43),
    "bujumbura-bi": (29.36, -3.38), "taylor-tx": (-97.48, 30.52), "cairo-desert-eg": (31.75, 30.02),
    "sahara-dz": (2.5, 28.0), "riyadh-edge-sa": (46.85, 24.85), "guadeloupe-bananas": (-61.58, 16.05),
}
PLACE_YEARS = (2023, 2025)
PLACES_DIR = os.path.join(HERE, "data", "places")
AEF_NODATA = -128


def abroad_sites():
    """Every cached Earthwork site outside the 48 states with a full grid, islands first."""
    have = {f[:-4] for f in os.listdir(EARTHWORK) if f.endswith(".npz")}
    pre = ("pr-", "vi-", "reunion-", "guadeloupe-", "hk-", "shizuoka-")
    return [n for p in pre for n in sorted(have) if n.startswith(p)]


SPACE_DIR = os.path.join(HERE, "data", "space")


def space_names():
    """Earthwork's satellite-laser sites whose label positions space_positions.py rebuilt."""
    if not os.path.isdir(SPACE_DIR):
        return []
    return [f"space-{f[:-4]}" for f in sorted(os.listdir(SPACE_DIR)) if f.endswith(".npz")]


def read_space(g, name):
    """TESSERA under a satellite-laser site's labels: each row its own pixel and year pair, every year read once
    over the box of all the site's labels."""
    path = f"{OUT}/{name}.npz"
    if os.path.exists(path):
        return path
    pz = np.load(f"{SPACE_DIR}/{name[len('space-'):]}.npz", allow_pickle=True)
    pos, meta = pz["rc"], json.loads(str(pz["meta"]))
    lon, lat = AEF_X0 + (pos[:, 1] + 0.5) * AEF_RES, AEF_Y0 - (pos[:, 0] + 0.5) * AEF_RES
    z = zone(g, float(lon.mean()))
    X, Y = Transformer.from_crs(4326, z.attrs["proj:code"], always_xy=True).transform(lon, lat)
    _, _, tx0, _, _, ty0 = z.attrs["spatial:transform"]
    col, row = np.floor((X - tx0) / 10).astype(np.int64), np.floor((ty0 - Y) / 10).astype(np.int64)
    r0, r1, c0, c1 = row.min(), row.max() + 1, col.min(), col.max() + 1
    at = (row - r0) * (c1 - c0) + (col - c0)
    times = list(z["time"][:])
    keys = ("q", "scales", "s2_obs_count", "s1_asc_obs_count", "s1_desc_obs_count")
    out = {f"{tag}_{k}": None for tag in "ba" for k in keys}
    for year in sorted(set(pos[:, 2]) | set(pos[:, 3])):
        t, ti = time.time(), times.index(year)
        q = z["embeddings"][ti, :, r0:r1, c0:c1].reshape(128, -1)[:, at].T
        got = {"q": q, **{v: z[v][ti, r0:r1, c0:c1].ravel()[at] for v in keys[1:]}}
        for tag, ycol in (("b", 2), ("a", 3)):
            m = pos[:, ycol] == year
            for k in keys:
                if out[f"{tag}_{k}"] is None:
                    out[f"{tag}_{k}"] = np.zeros((len(pos), *got[k].shape[1:]), got[k].dtype)
                out[f"{tag}_{k}"][m] = got[k][m]
        print(f"  {name} {year}: {r1 - r0}x{c1 - c0} TESSERA px, {time.time() - t:.0f} s")
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(path, meta=json.dumps({**meta, "aef": "per row", "zone": z.attrs["proj:code"]}), **out)
    return path


def place_names():
    return [f"place-{n}" for n in PLACES]


def site_path(name):
    """A site's AlphaEarth side: Earthwork's cache, or the AlphaEarth this repo read for a place with no truth."""
    return f"{PLACES_DIR}/{name}.npz" if name.startswith(("place-", "ground-")) else f"{EARTHWORK}/{name}.npz"


def aef_unit(year, rc):
    """AlphaEarth unit embeddings (h * w, 64) for the year over mosaic rows and columns rc, and the valid mask
    (the mosaic Earthwork reads: tge-labs/aef-mosaic on Source Coop)."""
    from obstore.store import S3Store
    if "arr" not in _aef:
        store = S3Store("tge-labs", endpoint="https://data.source.coop", region="us-west-2",
                        virtual_hosted_style_request=False, skip_signature=True, prefix="aef-mosaic")
        g = zarr.open_group(ObjectStore(store, read_only=True), mode="r")
        _aef["times"], _aef["arr"] = g["time"][:], g["embeddings"]
    y0, y1, x0, x1 = rc
    e = np.asarray(_aef["arr"][int(np.where(_aef["times"] == year)[0][0]), :, y0:y1, x0:x1])
    v = e.astype(np.float32)
    v = np.sign(v) * (v / 127.5) ** 2
    v /= np.maximum(np.linalg.norm(v, axis=0), 1e-9)
    return v.reshape(64, -1).T, (e[0] != AEF_NODATA).ravel()


_aef = {}


def read_place(name):
    """A place with no truth: AlphaEarth b and a in Earthwork's cache layout (X holds [b, a]), every valid pixel
    labeled unchanged so the share scored above a cutoff can be read the way false alarms are."""
    path = site_path(name)
    if os.path.exists(path):
        return path
    lon, lat = PLACES[name[len("place-"):]]
    x0, y0 = int((lon - AEF_X0) / AEF_RES) - 150, int((AEF_Y0 - lat) / AEF_RES) - 150
    rc = (y0, y0 + 300, x0, x0 + 300)
    (b, okb), (a, oka) = aef_unit(PLACE_YEARS[0], rc), aef_unit(PLACE_YEARS[1], rc)
    os.makedirs(PLACES_DIR, exist_ok=True)
    np.savez_compressed(path, X=np.concatenate([b, a], 1).astype(np.float16), y=np.where(okb & oka, 0, -1).astype(np.int8),
                        meta=json.dumps({"name": name, "kind": "notruth", "aef": list(PLACE_YEARS), "rc": list(rc),
                                         "shape": [300, 300], "point": [lon, lat]}))
    return path


def intl_rc(point, crs, half_km=2.0, step=2.0):
    """The mosaic rows and columns of an IGN site's 4 km square (Earthwork's _box, then aef_rc)."""
    x, y = Transformer.from_crs(4326, crs, always_xy=True).transform(*point)
    W, S = np.floor((x - half_km * 1000) / step) * step, np.floor((y - half_km * 1000) / step) * step
    n = int(round(2 * half_km * 1000 / step))
    return grid_rc(W, S + n * step, n, n, crs, step)


def grid_rc(W, N, w, h, crs, step=2.0):
    """The AlphaEarth mosaic's rows and columns inside a grid of w x h cells from its northwest corner."""
    inv = Transformer.from_crs(crs, 4326, always_xy=True)
    c = [inv.transform(W + i * step, N - j * step) for i, j in ((0, 0), (w, 0), (0, h), (w, h))]
    w_, e_ = max(c[0][0], c[2][0]), min(c[1][0], c[3][0])
    n_, s_ = min(c[0][1], c[1][1]), max(c[2][1], c[3][1])
    x0, x1 = int(np.ceil((w_ - AEF_X0) / AEF_RES)), int((e_ - AEF_X0) / AEF_RES)
    y0, y1 = int(np.ceil((AEF_Y0 - n_) / AEF_RES)), int((AEF_Y0 - s_) / AEF_RES)
    return y0, y1, x0, x1


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
    """The 2 m grid over where two open tiles meet: west, south, east, north, width and height (the second tile
    warped into the first's CRS when they differ, as Earthwork read the islands)."""
    from rasterio.vrt import WarpedVRT
    bb = WarpedVRT(sb, crs=sa.crs).bounds if sb.crs != sa.crs else sb.bounds
    W, S = (np.ceil(max(u, v) / 2) * 2 for u, v in ((sa.bounds.left, bb.left), (sa.bounds.bottom, bb.bottom)))
    E, N = (np.floor(min(u, v) / 2) * 2 for u, v in ((sa.bounds.right, bb.right), (sa.bounds.top, bb.top)))
    return W, S, E, N, int((E - W) / 2), int((N - S) / 2)


def site_rc(meta):
    """The site's AlphaEarth mosaic rows and columns, as Earthwork built it: stored with the site (Asia), from an
    IGN site's point, or from its two lidar tiles' headers."""
    if "rc" in meta:
        return tuple(meta["rc"])
    if meta["name"] in INTL_POINTS:
        return intl_rc(INTL_POINTS[meta["name"]], meta["crs"])
    u1, u2 = meta["tiles"]
    with rasterio.open("/vsicurl/" + u1) as sa, rasterio.open("/vsicurl/" + u2) as sb:
        W, S, E, N, w, h = shared_grid(sa, sb)
        return grid_rc(W, N, w, h, sa.crs)


def read_rc(g, name, rc, years, idx=None, tags="ba"):
    """TESSERA for the two years under AlphaEarth mosaic rows and columns rc (only the pixels idx of that box,
    flattened, when given): int8 (n, 128), scales and observation counts per year, keyed by each year's tag."""
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
    for tag, year in zip(tags, years):
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
        if name.startswith("place-"):
            read_place(name)
        meta = json.loads(str(np.load(site_path(name), allow_pickle=True)["meta"]))
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
    which = {"abroad": abroad_sites, "places": place_names, "space": space_names}
    names = [n for a in sys.argv[1:] for n in (which[a]() if a in which else [a])]
    for name in names or SITES + [s[0] for s in cdl_sets()]:
        t = time.time()
        (read_space if name.startswith("space-") else read_site)(g, name)
        print(f"{name}: done in {time.time() - t:.0f} s")


if __name__ == "__main__":
    main()
