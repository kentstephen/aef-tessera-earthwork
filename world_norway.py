"""Norway: Kartverket's per-project lidar DTMs on hoydedata.no (CC BY 4.0), two projects flown over one place.

The ArcGIS ImageServer Prosjekt_DTM holds every lidar project as its own raster (0.25 m for 5 to 30 points per
m2), in ETRS89 / UTM 32 or 33 (EPSG 25832, 25833) with NN2000 heights (EPSG 5941). exportImage with the mosaic
locked to one project's raster id returns that project alone over a box. Each flight is read at 0.5 m in 1 km
blocks and averaged down to the 2 m grid: asked for 2 m directly, the server samples each project's own pyramid,
and on steep ground (slope over 0.4) the two pyramids disagree by a median 0.37 m, against 0.10 m at 0.5 m.
Outside a project's footprint the server returns the noData value. Over
water the DTM is interpolated, not measured, and two projects interpolate a river differently (the Drammen
river by a median 0.7 m); so ground under 2 m above NN2000 in both flights (sea, harbors, the lower rivers) is
left out, and so are cells drawn smooth (to 2 mm) in both flights at levels more than 0.1 m apart (rivers and
lakes higher up).

The catalog (query) gives each project's last flight day (SISTEFLYDATO), its year, point density
(PUNKTTETTHET) and TYPE ('1' lidar, '2' image matching). There is no first flight day or per-tile date, so each
flight is taken as the 90 days up to its last day (a project is flown inside one season); that pushes the
AlphaEarth year before the first flight one year earlier when the last day falls in July or later. Only lidar
projects (TYPE 1) on NN2000 with at least 5 points per m2 are used; the thinned 0.5 point copies are skipped.
"""

import datetime
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from rasterio.crs import CRS
from rasterio.transform import from_origin
from scipy import ndimage

from world_sites import RAW, STEP, window

SERVICE = "https://hoydedata.no/arcgis/rest/services/Prosjekt_DTM/ImageServer"
SEASON = datetime.timedelta(days=90)
NODATA = -9999.0
FINE, BLOCK = 0.5, 1000.0  # read 0.5 m pixels in 1 km blocks
SMOOTH = 0.002  # m: rougher than this is measured ground
LOW = 2.0  # m above NN2000: under this in both flights is taken as water and left out

# (raster id, project) pairs: first flight, second flight
OSLO = (117461, 227102)  # Oslo kommune laserskanning 2019 (2019-09-14), Oslo 10pkt 2024 (2024-05-18)
BAERUM = (171285, 196299)  # Baerum 2018 (2018-05-14), Viken vest 10pkt del 1 2022 (2022-05-06)
STAVANGER = (249233, 225784)  # Stavanger 10pkt 2019 (2019-10-06), Stavanger 10pkt 2024 (2024-06-24)
TROMSO = (122554, 256057)  # Tromso 8pkt 2019 (2019-07-09), Tromsoya 10pkt 2024 (2024-08-07)
TRONDHEIM = (187096, 198658)  # Trondheim 30pkt 2020 (2020-07-13), NDH Trondheim 30pkt 2022 (2022-07-30)
DRAMMEN = (217393, 222490)  # Viken vest 10pkt del 2 2022 (2022-05-06), Nedre Buskerud 5pkt 2024 (2024-06-24)


def S(name, kind, lat, lon, pair, half_km=2.0, note=""):
    return dict(name=f"world-norway-{name}", kind=kind, country="no", lat=lat, lon=lon, pair=pair, half_km=half_km,
                note=note)


SITES = [
    S("oslo-bjorvika", "building", 59.906, 10.760, OSLO,
      note="Bjorvika and Sorenga waterfront, Oslo central station area; the harbor's DTM is interpolated"),
    S("fornebu", "building", 59.896, 10.616, BAERUM,
      note="Fornebubanen metro station pits and portals, Fornebu housing, E18 Lysaker"),
    S("stavanger-forus", "building", 58.905, 5.728, STAVANGER, note="Forus business park growth and road works"),
    S("oslo-hovinbyen", "building", 59.928, 10.800, OSLO,
      note="Hovinbyen: Okern, Lorenskog road, Ulven and Hasle housing and road works"),
    S("oslo-nordstrand", "quiet", 59.870, 10.800, OSLO,
      note="Nordstrand and Ljan houses and the Ekeberg forest, little dug; the fjord's DTM is interpolated"),
    S("tromso", "building", 69.655, 18.945, TROMSO,
      note="Tromsoya: town growth, university and airport side; the biggest fills (+3 to +5 m) are at the "
           "Breivika and Prostneset harbors, which may include quay decks"),
    S("trondheim", "building", 63.435, 10.410, TRONDHEIM,
      note="Trondheim center, Brattora and Nyhavna fills and city works; 30 points per m2 both times"),
    S("drammen", "building", 59.742, 10.175, DRAMMEN,
      note="Drammen, the new hospital at Brakeroya and river side works; a short pair (2022 to 2024)"),
]

_cat = {}


def project(oid):
    """A project's catalog row (cached in data/world-raw/norway)."""
    if oid in _cat:
        return _cat[oid]
    cache = os.path.join(RAW, "norway", f"project_{oid}.json")
    if os.path.exists(cache):
        _cat[oid] = json.load(open(cache))
        return _cat[oid]
    d = requests.get(f"{SERVICE}/query", params={"where": f"OBJECTID={oid}", "outFields": "*",
                                                 "returnGeometry": "false", "f": "json"}, timeout=120).json()
    a = d["features"][0]["attributes"]
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    json.dump(a, open(cache, "w"))
    _cat[oid] = a
    return a


def flight(a):
    """(start, end) of a project's flight: the season up to its last flight day."""
    last = datetime.datetime.fromtimestamp(a["SISTEFLYDATO"] / 1000, datetime.UTC).date()
    return last - SEASON, last


def block(oid, box, n, crs):
    """One project's DTM over a box (west, south, east, north) as n x n pixels: float32, NaN outside it."""
    import rasterio
    W, S_, E, N = box
    p = {"bbox": f"{W},{S_},{E},{N}", "bboxSR": crs.to_epsg(), "imageSR": crs.to_epsg(), "size": f"{n},{n}",
         "format": "tiff", "pixelType": "F32", "noData": NODATA, "noDataInterpretation": "esriNoDataMatchAny",
         "interpolation": "RSP_BilinearInterpolation",
         "mosaicRule": json.dumps({"mosaicMethod": "esriMosaicLockRaster", "lockRasterIds": [oid]}), "f": "image"}
    for k in range(6):
        try:
            r = requests.get(f"{SERVICE}/exportImage", params=p, timeout=300)
            if r.ok and r.headers.get("content-type", "").startswith("image"):
                break
        except requests.RequestException as e:
            print(f"    exportImage {oid}: {e}", flush=True)
        time.sleep(5 * (k + 1))
    else:
        raise RuntimeError(f"exportImage failed for raster {oid}")
    with rasterio.MemoryFile(io.BytesIO(r.content)) as m, m.open() as s:
        a = s.read(1).astype(np.float32)
        assert a.shape == (n, n) and abs(s.transform.c - W) < 0.01 and abs(s.transform.f - N) < 0.01, s.transform
    a[(a <= NODATA + 1) | (a < -1000)] = np.nan
    return a


def export(oid, grid, crs):
    """One project's DTM over the 2 m grid (west, south, east, north, width, height in crs), read at 0.5 m in
    1 km blocks and averaged down (a 2 m cell with any 0.5 m pixel missing is NaN): float32, NaN outside.
    Cached in data/world-raw/norway."""
    W, S_, E, N, w, h = grid
    cache = os.path.join(RAW, "norway", f"dtm_{oid}_{crs.to_epsg()}_{W:.0f}_{N:.0f}_{w}x{h}.npy")
    if os.path.exists(cache):
        return np.load(cache)
    f = int(STEP / FINE)
    nb = int(BLOCK / STEP)  # 2 m cells a block
    out = np.full((h, w), np.nan, np.float32)
    jobs = [(r0, c0) for r0 in range(0, h, nb) for c0 in range(0, w, nb)]

    def one(rc):
        r0, c0 = rc
        rh, cw = min(nb, h - r0), min(nb, w - c0)
        x0, y1 = W + c0 * STEP, N - r0 * STEP
        # a square request (the server wants one size); crop to the grid after
        side = max(rh, cw)
        a = block(oid, (x0, y1 - side * STEP, x0 + side * STEP, y1), side * f, crs)
        a = a.reshape(side, f, side, f).mean(axis=(1, 3))
        out[r0:r0 + rh, c0:c0 + cw] = a[:rh, :cw]
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(one, jobs))
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.save(cache, out)
    return out


def interpolated(z):
    """Cells on a surface the DTM drew rather than measured (water, mostly): its departure from the 3 x 3 mean,
    averaged over 5 x 5, is under 2 mm. Measured ground scores 1 to 10 cm."""
    rough = ndimage.uniform_filter(np.abs(z - ndimage.uniform_filter(z, 3)), 5)
    with np.errstate(invalid="ignore"):
        return rough < SMOOTH


def dems(site):
    rows = [project(oid) for oid in site["pair"]]
    for a in rows:
        assert str(a["TYPE"]) == "1" and str(a["HOYDESYSTEM"]) == "5941" and (a["PUNKTTETTHET"] or 0) >= 5, a["NAME"]
    crs = CRS.from_epsg(int(rows[0]["KOORDINATSYSTEM"]))
    grid = window(site, crs)
    a, b = (export(oid, grid, crs) for oid in site["pair"])
    with np.errstate(invalid="ignore"):
        low = (a < LOW) & (b < LOW)
        # the 5 x 5 test misses the last cells to each bank: grow 3 cells (6 m)
        drawn = (ndimage.binary_dilation(interpolated(a) & interpolated(b), iterations=3)
                 & (np.abs(b - a - np.nanmedian(b - a)) > 0.1))
    a[low | drawn] = b[low | drawn] = np.nan
    if not (np.isfinite(a).any() and np.isfinite(b).any()):
        return None
    return dict(a=a, b=b, transform=from_origin(grid[0], grid[3], STEP, STEP), crs=crs, first=flight(rows[0]),
                second=flight(rows[1]),
                source=f"hoydedata.no Prosjekt_DTM {rows[0]['NAME']} / {rows[1]['NAME']} (NN2000)")
