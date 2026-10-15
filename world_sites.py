"""Repeat lidar outside the US, built into Earthwork's site format: truth for a worldwide model.

The US sites are 3DEP tiles flown twice; abroad the model ranks moved ground well but scores it dim (FINDINGS 4,
5; London lit nothing). Here the open repeat lidar found abroad (Netherlands AHN4 and AHN5, New Zealand LINZ,
Norway, Switzerland, England, Scotland, Lyon, Canada, Poland, Estonia, Kanagawa, Sao Paulo) is built the way
Earthwork built its sites: the two bare-earth DEMs on one 2 m grid, second minus first with the median removed
(a datum or geoid offset between flights), water lying flat in both flights kept as unchanged, each AlphaEarth
10 m pixel labeled MOVED (mean |dz| > 0.5 m) or UNCHANGED (< 0.1 m and under 5% of its 2 m cells moved), the
rest left out. AlphaEarth's years: the year before the first flight and the year of the second (aef_years).

Each source is a module world_<source>.py with SITES (dicts: name, kind, country, lon, lat and the source's own
keys) and dems(site) -> dict(a, b, transform, crs, first, second, source[, cover]): the two DEMs as float32 on
one 2 m grid (NaN where there is none), the grid's transform and CRS, each flight's (start, end) dates, a line
naming the data, and optionally the mask of grid cells the flights cover.

Sites go to data/world/<name>.npz in Earthwork's cache layout (X: [b, a, a * b, (a - b)^2] float16, y, plain,
dz, patch, wet, meta with the mosaic rows and columns), so tessera_sites.read_site reads TESSERA under them and
compare.load loads them like any other site.

Run: uv run python world_sites.py [source ...] [--tessera]   (every source when none is named)
"""

import datetime
import importlib
import json
import os
import sys
import time
import traceback

import numpy as np
from pyproj import Transformer
from scipy import ndimage

from tessera_sites import AEF_RES, AEF_X0, AEF_Y0, aef_unit

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(HERE, "data", "world")
RAW = os.path.join(HERE, "data", "world-raw")  # downloads that cannot be read by window, kept to rebuild
SOURCES = ("nz", "ahn", "norway", "swiss", "england", "scotland", "lyon", "canada", "poland", "estonia",
           "kanagawa", "saopaulo")
MOVED_M, STILL_M = 0.5, 0.1
MIN_OVERLAP = 0.15
STEP = 2.0


def window(site, crs, half_km=None):
    """The 2 m grid of a square about the site's point in crs: (west, south, east, north, width, height)."""
    half = 1000 * (half_km or site.get("half_km", 2.0))
    x, y = Transformer.from_crs(4326, crs, always_xy=True).transform(site["lon"], site["lat"])
    W, S = np.floor((x - half) / STEP) * STEP, np.floor((y - half) / STEP) * STEP
    n = int(round(2 * half / STEP))
    return W, S, W + n * STEP, S + n * STEP, n, n


def read_cogs(urls, crs, grid, nodata_below=-1000):
    """The rasters at urls (GDAL paths: /vsicurl/..., /vsis3/...) mosaicked onto the 2 m grid (west, south, east,
    north, width, height in crs), each averaged down from its full resolution, warped when its CRS differs:
    float32, NaN where none has data."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.vrt import WarpedVRT
    from rasterio.windows import from_bounds
    W, S, E, N, w, h = grid
    out = np.full((h, w), np.nan, np.float32)
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", AWS_NO_SIGN_REQUEST="YES", GDAL_HTTP_MAX_RETRY="5",
                      GDAL_HTTP_RETRY_DELAY="2", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff,.TIF,.vrt"):
        for u in urls:
            # full resolution averaged down: a COG's own overviews differ from year to year on slopes (5 cm median,
            # 14 cm at the 99th percentile in New Zealand's LERC overviews), which reads as moved ground. Through
            # vrt://...?ovr=NONE, since the OVERVIEW_LEVEL open option let nodata into the averages (AHN4)
            with rasterio.open(u if u.startswith("vrt://") else f"vrt://{u}?ovr=NONE") as s0:
                src = s0 if s0.crs == crs else WarpedVRT(s0, crs=crs, resampling=Resampling.bilinear)
                b = src.bounds
                x0, x1 = max(W, b.left), min(E, b.right)
                y0, y1 = max(S, b.bottom), min(N, b.top)
                if x1 - x0 < STEP or y1 - y0 < STEP:
                    continue
                # snap to the 2 m grid
                c0, c1 = int(np.ceil((x0 - W) / STEP)), int(np.floor((x1 - W) / STEP))
                r0, r1 = int(np.ceil((N - y1) / STEP)), int(np.floor((N - y0) / STEP))
                if c1 <= c0 or r1 <= r0:
                    continue
                win = from_bounds(W + c0 * STEP, N - r1 * STEP, W + c1 * STEP, N - r0 * STEP, src.transform)
                a = src.read(1, window=win, out_shape=(r1 - r0, c1 - c0), resampling=Resampling.average,
                             masked=True).astype(np.float32).filled(np.nan)
                a[a < nodata_below] = np.nan
                cur = out[r0:r1, c0:c1]
                fill = np.isnan(cur) & np.isfinite(a)
                cur[fill] = a[fill]
    return out, from_origin(W, N, STEP, STEP)


def aef_years(first, second):
    """AlphaEarth's year before the first flight (the flight's own year if it began in the second half) and
    its year at the second (the year before, if the flight ended in the first half), inside 2017-2025."""
    b = first[0].year if first[0].month >= 7 else first[0].year - 1
    a = second[1].year if second[1].month >= 7 else second[1].year - 1
    return max(2017, b), min(2025, a)


def change(a, b, cover=None):
    """Second minus first with the median removed, and the water lying flat in both flights. Water in one flight
    only is digging one way round: ground then lower water is a dug pond, water then higher ground a filled one;
    the other two (a drawdown, a flood) are left out. The overlap share is over cover when given."""
    dz = b - a
    share = float(np.isfinite(dz)[cover].mean() if cover is not None else np.isfinite(dz).mean())
    dz -= np.nanmedian(dz)

    def flat(z):
        ok = np.isfinite(z)
        f = np.where(ok, z.astype(np.float64) - np.nanmedian(z), 0.0)
        m = ndimage.uniform_filter(f, 5)
        full = ndimage.uniform_filter(ok.astype(np.float64), 5) > 0.999
        return ok & full & (np.sqrt(np.maximum(ndimage.uniform_filter(f * f, 5) - m * m, 0)) < 0.001)

    fa, fb = flat(a), flat(b)
    water = fa & fb & np.isfinite(dz)
    with np.errstate(invalid="ignore"):
        dz[water | (fa & ~fb & (dz < 0)) | (fb & ~fa & (dz > 0))] = np.nan
    return dz, share, water


def aef_rc(tr, crs, h, w):
    """The AlphaEarth mosaic's rows and columns inside a grid of h x w cells (transform tr, crs)."""
    inv = Transformer.from_crs(crs, 4326, always_xy=True)
    c = [inv.transform(*(tr * (x, y))) for x, y in ((0, 0), (w, 0), (0, h), (w, h))]
    W, E = max(c[0][0], c[2][0]), min(c[1][0], c[3][0])
    N, S = min(c[0][1], c[1][1]), max(c[2][1], c[3][1])
    x0, x1 = int(np.ceil((W - AEF_X0) / AEF_RES)), int((E - AEF_X0) / AEF_RES)
    y0, y1 = int(np.ceil((AEF_Y0 - N) / AEF_RES)), int((AEF_Y0 - S) / AEF_RES)
    return y0, y1, x0, x1


def to_aef_grid(dz, tr, crs, water):
    """Each AlphaEarth 10 m pixel over the grid: mean dz, the share of its 2 m cells that moved > 0.5 m and the
    share that were water in both flights (5 x 5 samples a pixel), with the mosaic's row and column range."""
    h, w = dz.shape
    y0, y1, x0, x1 = aef_rc(tr, crs, h, w)
    sub = (np.arange(5) + 0.5) / 5
    cols = x0 + (np.arange(x1 - x0)[:, None] + sub[None]).ravel()
    rows = y0 + (np.arange(y1 - y0)[:, None] + sub[None]).ravel()
    LON, LAT = np.meshgrid(AEF_X0 + cols * AEF_RES, AEF_Y0 - rows * AEF_RES)
    X, Y = Transformer.from_crs(4326, crs, always_xy=True).transform(LON, LAT)
    cc, rr = (~tr) * (X, Y)
    v = ndimage.map_coordinates(dz, [rr - 0.5, cc - 0.5], order=0, mode="constant", cval=np.nan)
    v = v.reshape(y1 - y0, 5, x1 - x0, 5)
    wv = ndimage.map_coordinates(water.astype(np.float32), [rr - 0.5, cc - 0.5], order=0, mode="constant", cval=0)
    wet = wv.reshape(y1 - y0, 5, x1 - x0, 5).mean(axis=(1, 3))
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(v, axis=(1, 3))
        moved = np.nanmean(np.abs(v) > MOVED_M, axis=(1, 3))
    return mean.astype(np.float32), moved.astype(np.float32), wet.astype(np.float32), (y0, y1, x0, x1)


def features(b, a):
    """[b, a, a * b, (a - b)^2] per pixel from (n, 64) unit rows: (n, 256)."""
    return np.concatenate([b, a, a * b, (a - b) ** 2], 1)


def path(name):
    return os.path.join(DIR, f"{name}.npz")


def build(site, got):
    """A site's cache from its two DEMs (got: what the source's dems returned). None when it does not qualify."""
    name, lat = site["name"], site["lat"]
    dz, share, water = change(got["a"], got["b"], got.get("cover"))
    if share < MIN_OVERLAP:
        print(f"  {name}: skipped (the flights overlap on {share:.0%} of the window)", flush=True)
        return None
    mean, moved, wet, rc = to_aef_grid(dz, got["transform"], got["crs"], water)
    d1, d2 = got["first"], got["second"]
    yb, ya = aef_years(d1, d2)
    if ya <= yb:
        print(f"  {name}: skipped (no AlphaEarth year between the flights {d1} and {d2})", flush=True)
        return None
    bv, okb = aef_unit(yb, rc)
    av, oka = aef_unit(ya, rc)
    mean, moved, wet = mean.ravel(), moved.ravel(), wet.ravel()
    ok = okb & oka & np.isfinite(mean)
    lab = np.full(mean.shape, -1, np.int8)
    lab[ok & (np.abs(mean) > MOVED_M)] = 1
    lab[ok & (np.abs(mean) < STILL_M) & (moved < 0.05)] = 0
    lab[okb & oka & (wet > 0.5) & (lab != 1)] = 0
    shape = (rc[1] - rc[0], rc[3] - rc[2])
    px_ha = (AEF_RES * 111320 * np.cos(np.radians(lat))) * (AEF_RES * 110574) / 1e4
    cc, _ = ndimage.label((lab == 1).reshape(shape), structure=np.ones((3, 3)))
    patch = (np.bincount(cc.ravel())[cc] * px_ha).astype(np.float32).ravel()
    with np.errstate(invalid="ignore"):
        stats = {"dz_p50_abs": float(np.nanmedian(np.abs(dz))), "dz_p99_abs": float(np.nanpercentile(np.abs(dz), 99))}
    meta = {"name": name, "kind": site["kind"], "country": site["country"], "point": [site["lon"], site["lat"]],
            "flights": [f"{d1[0]}..{d1[1]}", f"{d2[0]}..{d2[1]}"], "aef": [int(yb), int(ya)], "overlap": round(share, 3),
            "tiles": [], "rc": [int(v) for v in rc], "shape": list(shape), "source": got["source"],
            "quiet_only": bool(site.get("quiet_only", False)), "note": site.get("note", ""), **stats}
    out = {"X": features(bv, av).astype(np.float16), "y": lab, "plain": (1 - (av * bv).sum(1)).astype(np.float32),
           "dz": mean, "patch": patch, "wet": wet > 0.5, "meta": json.dumps(meta)}
    os.makedirs(DIR, exist_ok=True)
    np.savez_compressed(path(name), **out)
    print(f"  {name}: AlphaEarth {yb} vs {ya}, overlap {share:.0%}, moved {int((lab == 1).sum()):,} px, unchanged "
          f"{int((lab == 0).sum()):,} (water {int((wet > 0.5).sum()):,}), |dz| median {stats['dz_p50_abs']:.3f} m", flush=True)
    return meta


def run(source, only=None):
    mod = importlib.import_module(f"world_{source}")
    for site in mod.SITES:
        if only and site["name"] not in only:
            continue
        if os.path.exists(path(site["name"])):
            print(f"  {site['name']}: cached", flush=True)
            continue
        t = time.time()
        try:
            got = mod.dems(site)
            if got is None:
                print(f"  {site['name']}: skipped (no DEMs)", flush=True)
                continue
            build(site, got)
        except Exception as e:
            print(f"  {site['name']}: FAILED {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
        print(f"  {site['name']}: {time.time() - t:.0f} s", flush=True)


def names():
    """Every built world site."""
    return sorted(f[:-4] for f in os.listdir(DIR) if f.endswith(".npz")) if os.path.isdir(DIR) else []


def read_tessera(which=None):
    """TESSERA under every built world site (tessera_sites.read_site), each once."""
    import zarr
    from obstore.store import HTTPStore
    from zarr.storage import ObjectStore

    from tessera_sites import TESSERA, read_site
    zarr.config.set({"async.concurrency": 128})
    g = ObjectStore(HTTPStore.from_url(TESSERA), read_only=True)
    for n in which or names():
        t = time.time()
        try:
            read_site(g, n)
            print(f"{n}: TESSERA in {time.time() - t:.0f} s", flush=True)
        except Exception as e:
            print(f"{n}: TESSERA FAILED {type(e).__name__}: {e}", flush=True)


def date(s):
    return datetime.date.fromisoformat(s)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    for src in args or SOURCES:
        print(f"== {src}", flush=True)
        run(src)
    if "--tessera" in sys.argv:
        read_tessera()
