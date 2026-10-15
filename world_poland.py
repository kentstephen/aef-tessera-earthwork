"""Poland: GUGiK's bare-earth DTM sheets (NMT, opendata.geoportal.gov.pl, open and free under Polish law since 2020).

The sheet index is two WMS services (SkorowidzeUkladEVRF2007 and SkorowidzeUkladKRON86); GetFeatureInfo at a point
returns every DTM version of the sheets under it with its acquisition day (aktualnosc), its source (zrDanych:
"Skaning laserowy" for lidar, "Zdj. lotnicze" for image matching, which is dropped here), resolution, horizontal
and vertical datum and the file's URL. Files are whole ASCII grids with no CRS: 1 m sheets on PL-1992 (EPSG 2180)
and 0.5 m sheets on PL-2000 (EPSG 2176 to 2179 by zone); each is downloaded once into data/world-raw/poland and
kept as a float32 GeoTIFF with its CRS. Heights are PL-KRON86-NH for flights up to 2018 and PL-EVRF2007-NH from
2019; the two differ by a near-constant ~0.28 m in Warsaw, which the median removal takes out over a 4 km window,
but a pair is kept inside one datum where the sheets allow. Each flight's dates are the first and last acquisition
day of the sheets that went into the window.
"""

import datetime
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

os.environ.setdefault("GDAL_CACHEMAX", "256")  # MB; 0.5 m sheets are large and GDAL's default cache is 5% of RAM
from pyproj import Transformer
from rasterio.crs import CRS

from world_sites import RAW, read_cogs, window

INDEX = {"EVRF2007": "https://mapy.geoportal.gov.pl/wss/service/PZGIK/NMT/WMS/SkorowidzeUkladEVRF2007",
         "KRON86": "https://mapy.geoportal.gov.pl/wss/service/PZGIK/NMT/WMS/SkorowidzeUkladKRON86"}
CRS_PL = CRS.from_epsg(2180)
EPSG = {"PL-1992": 2180, "PL-2000:S5": 2176, "PL-2000:S6": 2177, "PL-2000:S7": 2178, "PL-2000:S8": 2179}
DIR = os.path.join(RAW, "poland")
STEP_Q = 400  # index query spacing (m); a 0.5 m sheet is 800 m x 500 m, a 1 m sheet about 2.3 km across


def S(name, kind, lat, lon, first, second, half_km=2.0, note=""):
    """first and second: the acquisition date prefixes ("2022", "2023-09") of each flight's lidar sheets."""
    return dict(name=f"world-poland-{name}", kind=kind, country="pl", lat=lat, lon=lon, first=first,
                second=second, half_km=half_km, note=note)


SITES = [
    S("gdansk-baltic-hub", "building", 54.400, 18.710, ("2022",), ("2024",),
      note="Baltic Hub T3 container terminal reclamation in the Gulf of Gdansk, and port works"),
    S("zelazny-most", "mining", 51.510, 16.190, ("2021",), ("2024",), 3.0,
      "KGHM Zelazny Most copper tailings: dam raising, the southern extension, the beach; the whole basin rose about "
      "4 m, so |dz| median is high (the pond, flat in both flights, is kept as unchanged water)"),
    S("warsaw-s2-wilanow", "building", 52.145, 21.065, ("2018",), ("2023-09",),
      note="S2 south ring road (Ursynow tunnel, Wilanow junction) and Wilanow housing; first flight KRON86, "
           "second EVRF2007 (a ~0.28 m datum step, taken out by the median)"),
    S("krakow-modlnica", "building", 50.100, 19.860, ("2021-10",), ("2023-03",),
      note="S52 north Krakow ring road (Polnocna Obwodnica Krakowa) at the Modlnica junction with S7, mid-build"),
    S("wroclaw-jagodno", "building", 51.060, 17.070, ("2019",), ("2024-04",),
      note="Jagodno and southeast Wroclaw housing growth, the Wschodnia Obwodnica Wroclawia road"),
    S("swinoujscie", "building", 53.915, 14.265, ("2022-03",), ("2024-09",),
      note="Swinoujscie: the Swina road tunnel portals, LNG terminal expansion, port and beach works"),
    S("gdynia-port", "survey", 54.535, 18.500, ("2020-05",), ("2024-09",),
      note="Gdynia port (new ferry terminal, quays) and the city"),
    S("poznan-east", "survey", 52.400, 17.000, ("2019-03",), ("2024-09",),
      note="east Poznan: Malta, Antoninek, housing and industrial growth on the city edge"),
    S("zulawy", "quiet", 54.150, 19.000, ("2019-03",), ("2022-04",),
      note="Vistula delta polder farmland (Zulawy), fields and ditches"),
]

_tr = Transformer.from_crs(4326, 2180, always_xy=True)


def query(x, y, datum):
    """Every DTM sheet version under the point (EPSG 2180 x, y) in one index service: a list of dicts."""
    u = INDEX[datum]
    layers = {"EVRF2007": "SkorowidzeNMT2023iStarsze,SkorowidzeNMT2024,SkorowidzeNMT2025,SkorowidzeNMT2026",
              "KRON86": "SkorowidzeNMT2017iStarsze,SkorowidzeNMT2018,SkorowidzeNMT2019"}[datum]
    p = {"SERVICE": "WMS", "request": "GetFeatureInfo", "version": "1.3.0", "styles": "", "crs": "EPSG:2180",
         "width": 101, "height": 101, "format": "image/png", "transparent": "true", "i": 50, "j": 50,
         "INFO_FORMAT": "text/html", "layers": layers, "query_layers": layers,
         "bbox": f"{y - 50},{x - 50},{y + 50},{x + 50}", "FEATURE_COUNT": 100}
    for k in range(5):
        try:
            r = requests.get(u, params=p, timeout=60)
            r.raise_for_status()
            break
        except requests.RequestException:
            if k == 4:
                raise
    out = []
    for m in re.findall(r"push\(\{(.*?)\}\)", r.text):
        d = dict(re.findall(r'(\w+):"([^"]*)"', m))
        if d.get("url"):
            out.append({"url": d["url"], "date": d.get("aktualnosc", ""), "source": d.get("zrDanych", ""),
                        "res": d.get("charakterystykaPrzestrzenna", ""), "datum": d.get("ukladWspolrzednychPionowych", ""),
                        "hcrs": d.get("ukladWspolrzednychPoziomych", ""), "sheet": d.get("godlo", "")})
    return out


def sheets(site):
    """Every DTM sheet version over the site's window, from both index services (cached in data/world-raw/poland)."""
    cache = os.path.join(DIR, f"index_{site['name']}.json")
    if os.path.exists(cache):
        return json.load(open(cache))
    W, S_, E, N, _, _ = window(site, CRS_PL)
    pts = [(x, y) for x in np.arange(W, E + 1, STEP_Q) for y in np.arange(S_, N + 1, STEP_Q)]
    jobs = [(x, y, d) for x, y in pts for d in INDEX]
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(lambda j: query(*j), jobs))
    seen = {}
    for rows in res:
        for r in rows:
            seen[r["url"]] = r
    out = sorted(seen.values(), key=lambda r: (r["date"], r["url"]))
    os.makedirs(DIR, exist_ok=True)
    json.dump(out, open(cache, "w"), indent=0)
    return out


def fetch(r):
    """The sheet as a local float32 GeoTIFF with its CRS (downloaded once)."""
    import rasterio
    name = r["url"].rsplit("/", 1)[1]
    tif = os.path.join(DIR, name[:-4] + ".tif")
    if os.path.exists(tif):
        return tif
    asc = os.path.join(DIR, name)
    if not os.path.exists(asc):
        with requests.get(r["url"], stream=True, timeout=300) as q:
            q.raise_for_status()
            with open(asc + ".part", "wb") as f:
                for c in q.iter_content(1 << 20):
                    f.write(c)
        os.replace(asc + ".part", asc)
    with rasterio.open(asc) as s:
        a = s.read(1)
        prof = dict(driver="GTiff", width=s.width, height=s.height, count=1, dtype="float32", nodata=-9999.0,
                    transform=s.transform, crs=CRS.from_epsg(EPSG[r["hcrs"]]), compress="deflate", tiled=True,
                    predictor=3)
        nd = s.nodata
    a = a.astype(np.float32)
    if nd is not None:
        a[a == nd] = -9999.0
    with rasterio.open(tif + ".part", "w", **prof) as d:
        d.write(a, 1)
    os.replace(tif + ".part", tif)
    os.remove(asc)  # the GeoTIFF holds the same values
    return tif


def flight(site, rows, years, latest):
    """One flight's DEM on the window's grid from the lidar sheets whose date starts with one of years: 1 m sheets
    before 0.5 m ones, then the latest (second flight) or earliest (first) date first; each fills only empty cells.
    Returns (grid, transform, (first day, last day) of the sheets used, datums used)."""
    grid = window(site, CRS_PL)
    use = [r for r in rows
           if r["source"].startswith("Skaning") and r["date"].startswith(tuple(years)) and r["hcrs"] in EPSG]
    use.sort(key=lambda r: (r["res"] != "1.00 m", -int(r["date"].replace("-", "")) if latest else r["date"]))
    W, S_, E, N, w, h = grid
    out = np.full((h, w), np.nan, np.float32)
    tr, days, datums = None, [], set()
    with ThreadPoolExecutor(2) as ex:
        paths = list(ex.map(fetch, use))
    for r, p in zip(use, paths):
        a, tr = read_cogs([p], CRS_PL, grid)
        fill = np.isnan(out) & np.isfinite(a)
        if fill.sum() > 100:
            out[fill] = a[fill]
            days.append(datetime.date.fromisoformat(r["date"]))
            datums.add(r["datum"])
    if not days:
        return None
    return out, tr, (min(days), max(days)), datums


def dems(site):
    rows = sheets(site)
    f1 = flight(site, rows, site["first"], latest=False)
    f2 = flight(site, rows, site["second"], latest=True)
    if f1 is None or f2 is None:
        return None
    (a, tr, d1, u1), (b, _, d2, u2) = f1, f2
    return dict(a=a, b=b, transform=tr, crs=CRS_PL, first=d1, second=d2,
                source=f"GUGiK NMT lidar sheets {d1[0]}..{d1[1]} ({'+'.join(sorted(u1))}) / "
                       f"{d2[0]}..{d2[1]} ({'+'.join(sorted(u2))})")
