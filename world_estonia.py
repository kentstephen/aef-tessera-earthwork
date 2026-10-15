"""Estonia: Maa- ja Ruumiamet (formerly Maa-amet) lidar point clouds per 1 km sheet per year (open, free with
attribution to the Estonian Land and Spatial Development Board).

Each 1:2000 sheet (number: the last 3 digits of the L-EST97 northing km and the easting km) has one LAZ per flight:
regular spring national flights ("tava", leaf-off, about 2.5 points per m2 in 2018), summer forestry flights
("mets", leaf-on, sparser ground) and yearly low-altitude city flights ("madal", 150 to 300 MB per km2, not used
here). The year is in the file name; the days come from each point's GPS time (adjusted standard GPS time). There
are no dated DTM grids, so the ground class (2) is gridded here: mean z per 2 m cell over the window in L-EST97
(EPSG 3301), holes up to 3 cells filled from the nearest cell. Heights are EH2000 (Estonia's EVRF2007) in every
file read, 2018 to 2024; files are read in chunks keeping only the ground points. LAZ files are kept in
data/world-raw/estonia.
Run with `uv run --with "laspy[lazrs]" python world_sites.py estonia`.
"""

import datetime
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.transform import from_origin
from scipy import ndimage

from world_sites import RAW, STEP, window

CRS_EE = CRS.from_epsg(3301)
DIR = os.path.join(RAW, "estonia")
PORTAL = "https://geoportaal.maaamet.ee/index.php"
GPS0 = datetime.datetime(1980, 1, 6)


def S(name, kind, lat, lon, first, second, half_km=1.5, note=""):
    """first and second: (year, product) of each flight, product one of tava, mets, madal. The point moves to the
    nearest sheet center (plus 1 m, inside the 2 m grid) so a 3 km window covers 3 x 3 sheets, not 4 x 4."""
    t = Transformer.from_crs(4326, 3301, always_xy=True)
    x, y = t.transform(lon, lat)
    x, y = np.floor(x / 1000) * 1000 + 501, np.floor(y / 1000) * 1000 + 501
    lon, lat = t.transform(x, y, direction="INVERSE")
    return dict(name=f"world-estonia-{name}", kind=kind, country="ee", lat=round(lat, 6), lon=round(lon, 6),
                first=first, second=second, half_km=half_km, note=note)


SITES = [
    S("narva-pit-west", "mining", 59.235, 27.725, (2018, "tava"), (2022, "tava"),
      note="Narva oil shale open cast, western faces near Sirgala: strip mining moves the face and spoil ridges"),
    S("narva-pit-south", "mining", 59.222, 27.840, (2018, "tava"), (2022, "tava"),
      note="Narva oil shale open cast, southern faces"),
    S("kivioli-north", "quiet", 59.380, 27.005, (2018, "tava"), (2022, "tava"),
      note="forest, fields and clearcuts north of Kivioli: AlphaEarth changes there but the ground barely moved"),
    S("kivioli-west", "mining", 59.365, 26.845, (2018, "tava"), (2022, "tava"),
      note="Kivioli oil shale pits and the ash and spoil hills west of the town"),
    S("vao-quarry", "mining", 59.428, 24.890, (2020, "tava"), (2024, "tava"),
      note="Vao limestone quarry on the east edge of Tallinn, deepening and spreading"),
    S("ulemiste", "building", 59.420, 24.800, (2020, "tava"), (2024, "tava"),
      note="Ulemiste: Rail Baltica Tallinn terminal works (from 2023), Ulemiste City and airport-side building"),
    S("lahemaa", "quiet", 59.550, 25.850, (2018, "tava"), (2022, "tava"),
      note="Lahemaa national park forest and bog"),
    S("tartu", "quiet", 58.380, 26.720, (2019, "tava"), (2023, "tava"),
      note="Tartu old town and riverside, a settled city (some infill building)"),
]


def sheet_ids(site):
    """The 1 km sheets the site's window touches: (sheet id, west, south) in L-EST97."""
    W, S_, E, N, _, _ = window(site, CRS_EE)
    out = []
    for e in range(int(W // 1000), int((E - 1) // 1000) + 1):
        for n in range(int(S_ // 1000), int((N - 1) // 1000) + 1):
            out.append((f"{n % 1000:03d}{e:03d}", e * 1000, n * 1000))
    return out


def listing(sheet):
    """Every LAZ file name of the sheet (all years and products), from the portal's search page (cached)."""
    cache = os.path.join(DIR, "list", f"{sheet}.json")
    if os.path.exists(cache):
        return json.load(open(cache))
    r = requests.get(PORTAL, params={"lang_id": 2, "plugin_act": "otsing", "page_id": 664, "kaardiruut": sheet,
                                     "andmetyyp": "lidar_laz_tava"}, timeout=60)
    r.raise_for_status()
    names = sorted(set(re.findall(rf"({sheet}_\d{{4}}_[a-z]+\.laz)", r.text)))
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    json.dump(names, open(cache, "w"))
    return names


def fetch(sheet, year, product):
    """The sheet's LAZ for that flight as a local path (downloaded once), or None when the portal lists none."""
    f = f"{sheet}_{year}_{product}.laz"
    p = os.path.join(DIR, f)
    if os.path.exists(p):
        return p
    if f not in listing(sheet):
        return None
    q = {"lang_id": 1, "plugin_act": "otsing", "kaardiruut": sheet, "andmetyyp": f"lidar_laz_{product}", "dl": 1,
         "f": f, "page_id": 614}
    for k in range(4):
        try:
            with requests.get(PORTAL, params=q, stream=True, timeout=600) as r:
                r.raise_for_status()
                with open(p + ".part", "wb") as o:
                    for c in r.iter_content(1 << 20):
                        o.write(c)
            with open(p + ".part", "rb") as o:
                if o.read(4) != b"LASF":
                    raise ValueError(f"{f}: not a LAS file")
            os.replace(p + ".part", p)
            return p
        except (requests.RequestException, ValueError):
            if k == 3:
                raise


def grid_ground(paths, grid):
    """Ground points (class 2) of the LAZ files, mean z per 2 m cell on the grid (holes up to 3 cells filled from
    the nearest cell), and the (first, last) flight day: days holding at least 0.5% of the ground points."""
    import laspy
    W, S_, E, N, w, h = grid
    zsum = np.zeros(h * w, np.float64)
    cnt = np.zeros(h * w, np.int64)
    days = {}
    for p in paths:
        with laspy.open(p) as f:
            for pts in f.chunk_iterator(2_000_000):
                g = np.asarray(pts.classification) == 2
                if not g.any():
                    continue
                x, y, z = np.asarray(pts.x)[g], np.asarray(pts.y)[g], np.asarray(pts.z)[g]
                c = np.floor((x - W) / STEP).astype(np.int64)
                r = np.floor((N - y) / STEP).astype(np.int64)
                ok = (c >= 0) & (c < w) & (r >= 0) & (r < h)
                if not ok.any():
                    continue
                i = r[ok] * w + c[ok]
                zsum += np.bincount(i, z[ok], h * w)
                cnt += np.bincount(i, None, h * w)
                # adjusted standard GPS time: GPS seconds minus 1e9
                d = np.floor((np.asarray(pts.gps_time)[g][ok] + 1e9) / 86400).astype(np.int64)
                u, k = np.unique(d, return_counts=True)
                for a, b in zip(u.tolist(), k.tolist()):
                    days[a] = days.get(a, 0) + b
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (zsum / cnt).reshape(h, w).astype(np.float32)
    z[cnt.reshape(h, w) == 0] = np.nan
    hole = np.isnan(z)
    if hole.any() and (~hole).any():
        dist, (ri, ci) = ndimage.distance_transform_edt(hole, return_indices=True)
        fill = hole & (dist <= 3)
        z[fill] = z[ri[fill], ci[fill]]
    if not days:
        return z, None
    tot = sum(days.values())
    keep = sorted(d for d, k in days.items() if k >= 0.005 * tot)
    to = lambda d: (GPS0 + datetime.timedelta(days=d)).date()
    return z, (to(keep[0]), to(keep[-1]))


def flight(site, which, grid):
    year, product = which
    sheets = sheet_ids(site)
    with ThreadPoolExecutor(4) as ex:
        paths = list(ex.map(lambda s: fetch(s[0], year, product), sheets))
    missing = [s[0] for s, p in zip(sheets, paths) if p is None]
    if missing:
        print(f"  {site['name']}: no {year} {product} LAZ for sheets {missing}", flush=True)
    paths = [p for p in paths if p]
    if not paths:
        return None, None
    return grid_ground(paths, grid)


def dems(site):
    grid = window(site, CRS_EE)
    a, d1 = flight(site, site["first"], grid)
    b, d2 = flight(site, site["second"], grid)
    if a is None or b is None or d1 is None or d2 is None:
        return None
    W, S_, E, N, w, h = grid
    (y1, p1), (y2, p2) = site["first"], site["second"]
    return dict(a=a, b=b, transform=from_origin(W, N, STEP, STEP), crs=CRS_EE, first=d1, second=d2,
                source=f"Maa- ja Ruumiamet lidar {y1} {p1} / {y2} {p2}, ground class gridded to 2 m")
