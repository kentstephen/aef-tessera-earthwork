"""Japan, Kanagawa: the prefecture's open lidar (s3://kanagawa-pointcloud, CC BY 4.0), ground flown twice.

Each fiscal year's cover is a set of 0.5 m bare-earth grids, one zip per 1,000 x 750 m sheet of the national
1:5,000 grid quartered twice ({year}/{01|02}/LP/Grid/MESH/09/MC/74/09MC7421.zip), on plane rectangular zone IX
(JGD2011, EPSG:6677) with JGD2011 heights in both covers (all predate the 2025 height revision). A sheet's .lem
member is 1,500 rows (north first) of a 10 character label and 2,000 heights in 5 character decimeters, with
-1111 (outside the survey) or -9999 (no ground: water, voids) where there is none. The pairs: FY2019 west
Kanagawa (Odawara, Hakone, Yamakita) and FY2020 Sagamihara Midori, each flown again in FY2024. No flight month
is published (the header gives the survey year only), so a flight's dates are its fiscal year, April 1 to
March 31: FY2024 may have been flown in early 2025, which keeps AlphaEarth's after year at 2024. Sheets are
fetched whole into data/world-raw/kanagawa and averaged to 2 m.
"""

import datetime
import json
import os
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from rasterio.crs import CRS

from world_sites import RAW, STEP, window

BUCKET = "https://kanagawa-pointcloud.s3.ap-northeast-1.amazonaws.com"
CRS_JP = CRS.from_epsg(6677)
_L = "ABCDEFGHIJKLMNOPQRST"
WEST, SAGAMI = (2019, 2024), (2020, 2024)


def S(name, kind, lat, lon, pair, half_km=1.5, note=""):
    return dict(name=f"world-kanagawa-{name}", kind=kind, country="jp", lat=lat, lon=lon, pair=pair,
                half_km=half_km, note=note)


SITES = [
    # FY2019 west Kanagawa vs FY2024 (sites placed on a 10 m dz scan of the whole overlap)
    S("yamakita-west", "building", 35.362, 139.020, WEST, 2.0,
      "Shin-Tomei expressway works at the west end of Yamakita: big fills and cuts"),
    S("shintomei-yamakita-matsuda", "building", 35.374, 139.118, WEST, 2.5,
      "Shin-Tomei expressway corridor through Yamakita and Matsuda: cuts along the line, a fill near Matsuda"),
    S("nakai", "building", 35.322, 139.205, WEST, 1.5, "fills and cuts in the Nakai hills (placed by dz)"),
    S("manazuru", "mining", 35.174, 139.128, WEST, 1.5, "Manazuru stone quarries and the hills above them"),
    S("yugawara-hills", "survey", 35.190, 139.085, WEST, 1.5,
      "orchard and forest hills above Yugawara with cuts of 1 to 2 m (placed by dz); mottled +-0.5 to 1 m "
      "under trees, some labeled moved"),
    S("hakone-forest", "quiet", 35.255, 138.990, WEST, 1.5, "Hakone caldera forest, quiet but for one cut strip; scattered moved pixels are ground under canopy"),
    # FY2020 Sagamihara Midori vs FY2024 (sites placed on a 10 m dz scan of the whole overlap)
    S("sagamihara-toya", "building", 35.560, 139.303, SAGAMI, 1.6,
      "Toya, Midori-ku: big cut and fills, likely the Chuo Shinkansen (maglev) depot works and spoil fills; "
      "forest slopes in the northwest read 0.5 to 1.5 m lower in FY2024 (ground under canopy), some labeled moved"),
    S("sagamihara-aone", "building", 35.565, 139.186, SAGAMI, 2.0,
      "two valley fills in Midori-ku (placed by dz; perhaps maglev tunnel spoil)"),
    S("sagamihara-midori-forest", "quiet", 35.533, 139.130, SAGAMI, 1.5, "Tanzawa forest, quiet; scattered moved pixels (+-0.5 to 1.6 m) are ground under canopy on steep slopes"),
]


def sheet_bounds(code):
    """Zone IX bounds (west, south, east, north) of a sheet code such as 09MC7421: a 1:50,000 sheet (two letters,
    30 x 40 km from (300 km N, 160 km W)), its 1:5,000 sheet (3 x 4 km), then two quarterings numbered 1 to 4
    from the northwest (1,500 x 2,000 m, then 750 x 1,000 m)."""
    q1, q2 = int(code[6]) - 1, int(code[7]) - 1
    n = 300000 - 30000 * _L.index(code[2]) - 3000 * int(code[4]) - 1500 * (q1 // 2) - 750 * (q2 // 2)
    w = -160000 + 40000 * _L.index(code[3]) + 4000 * int(code[5]) + 2000 * (q1 % 2) + 1000 * (q2 % 2)
    return w, n - 750, w + 1000, n


def sheets(year):
    """{sheet code: key} of every Grid/MESH zip in the fiscal year's cover (the bucket listing, cached)."""
    cache = os.path.join(RAW, "kanagawa", f"mesh-{year}.json")
    if not os.path.exists(cache):
        keys = []
        for part in ("01", "02"):
            token = None
            while True:
                params = {"list-type": 2, "prefix": f"{year}/{part}/LP/Grid/MESH/"}
                if token:
                    params["continuation-token"] = token
                r = requests.get(BUCKET, params=params, timeout=120).text
                keys += [k.split("<")[0] for k in r.split("<Key>")[1:]]
                if "<NextContinuationToken>" not in r:
                    break
                token = r.split("<NextContinuationToken>")[1].split("<")[0]
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        json.dump(sorted(k for k in keys if k.endswith(".zip")), open(cache, "w"))
    return {k.rsplit("/", 1)[-1][:-4].upper(): k for k in json.load(open(cache))}


def fetch(key):
    path = os.path.join(RAW, "kanagawa", key)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        for attempt in range(4):
            try:
                r = requests.get(f"{BUCKET}/{key}", timeout=120)
                r.raise_for_status()
                break
            except requests.RequestException:
                if attempt == 3:
                    raise
                time.sleep(5 * (attempt + 1))
        with open(path + ".part", "wb") as f:
            f.write(r.content)
        os.replace(path + ".part", path)
    return path


def sheet_dem(path):
    """One sheet's 0.5 m ground grid averaged to 2 m (375 x 500, north up), NaN where it has no ground."""
    with zipfile.ZipFile(path) as z:
        lem = next(n for n in z.namelist() if n.lower().endswith(".lem"))
        raw = z.read(lem).decode("ascii")
    rows = raw.split("\r\n") if "\r\n" in raw else raw.split("\n")
    rows = [r for r in rows if r.strip()]
    v = np.array([np.frombuffer(r[10:10010].encode(), "S5").astype(np.int32) for r in rows])
    h = np.where((v == -1111) | (v == -9999), np.nan, v / 10.0).astype(np.float32)
    assert h.shape == (1500, 2000), (path, h.shape)
    k = int(STEP / 0.5)
    with np.errstate(invalid="ignore"), np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        return np.nanmean(h.reshape(1500 // k, k, 2000 // k, k), axis=(1, 3))


def cover(year, grid):
    """The fiscal year's 2 m mosaic over the grid (west, south, east, north, width, height): NaN where none."""
    W, S, E, N, w, h = grid
    out = np.full((h, w), np.nan, np.float32)
    keys = sheets(year)
    hit = [(c, k) for c, k in keys.items() if c.startswith("09")]
    hit = [(c, k) for c, k in hit if (b := sheet_bounds(c))[0] < E and b[2] > W and b[1] < N and b[3] > S]
    with ThreadPoolExecutor(8) as ex:
        paths = list(ex.map(lambda ck: fetch(ck[1]), hit))
    for (c, _), p in zip(hit, paths):
        sw, ss, se, sn = sheet_bounds(c)
        d = sheet_dem(p)
        # sheet edges are whole meters in multiples of 750 and 1,000, the window's on the 2 m lattice
        c0, r0 = int(round((sw - W) / STEP)), int(round((N - sn) / STEP))
        a0, b0 = max(0, -r0), max(0, -c0)
        a1, b1 = min(d.shape[0], h - r0), min(d.shape[1], w - c0)
        if a1 <= a0 or b1 <= b0:
            continue
        sub = d[a0:a1, b0:b1]
        cur = out[r0 + a0:r0 + a1, c0 + b0:c0 + b1]
        fill = np.isnan(cur) & np.isfinite(sub)
        cur[fill] = sub[fill]
    return out, len(hit)


def fiscal(year):
    return datetime.date(year, 4, 1), datetime.date(year + 1, 3, 31)


def dems(site):
    from rasterio.transform import from_origin
    grid = window(site, CRS_JP)
    y1, y2 = site["pair"]
    a, n1 = cover(y1, grid)
    b, n2 = cover(y2, grid)
    if n1 == 0 or n2 == 0:
        return None
    return dict(a=a, b=b, transform=from_origin(grid[0], grid[3], STEP, STEP), crs=CRS_JP, first=fiscal(y1),
                second=fiscal(y2), source=f"Kanagawa pointcloud FY{y1} / FY{y2} 0.5 m ground grid (dates: fiscal years)")
