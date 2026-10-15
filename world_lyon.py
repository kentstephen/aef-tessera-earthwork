"""Lyon: IGN LiDAR HD (2021, some tiles 2022) against the Metropole de Lyon lidar of 2023, both bare-earth MNT.

First cover: IGN LiDAR HD MNT, 0.5 m, Licence Ouverte 2.0, read through the Geoplateforme WMS-R (GetMap
returns float32 GeoTIFF for any bbox and CRS; the server averages correctly onto a 2 m grid in EPSG:3946, checked
against a 4 x 4 block mean of the 0.5 m data). Dates per 1 km tile from the WFS layer
IGNF_LIDAR-HD_METADONNEE:metadata (date_debut_acquisition, date_fin_acquisition); a site takes the earliest start
and latest end of the tiles under its window. Lyon was flown 2021-04 to 2021-09, with a few eastern tiles in
2022 and 2023-02.

Second cover: Metropole de Lyon MNT 2023, one 0.5 m COG in EPSG:3946 (CC46, although its page says Lambert-93),
Licence Ouverte 2.0, flown July to August 2023 (campaign dates from the dataset page; no per-tile dates). The Mont
Verdun sector (Monts d'Or, around 45.855 N 4.785 E) was flown again in September 2024, so no site sits there.
The wooded hills west of the Saone are left out too: at Dardilly (2022 against 2023) the moved pixels traced the
wooded stream valleys, ground under trees filtered differently, not digging.

Both heights are NGF-IGN69. The COG's overviews are shifted by (f - 1) / 2 source pixels (half an overview pixel
less a quarter meter), so the 0.5 m data is read at full resolution and block-averaged here, not read through
read_cogs. The grid is EPSG:3946.
"""

import datetime
import json
import os
import time
import warnings

import numpy as np
import rasterio
import requests
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.windows import Window

from world_sites import RAW, STEP, window

CRS_LY = CRS.from_epsg(3946)
LYON = "/vsicurl/https://data.grandlyon.com/files/grandlyon/imagerie/mnt2023/relief/MNT2023_Altitude_50cm.tif"
LYON_DATES = (datetime.date(2023, 7, 1), datetime.date(2023, 8, 31))
WMS = "https://data.geopf.fr/wms-r"
WFS = "https://data.geopf.fr/wfs/ows"
LAYER = "IGNF_LIDAR-HD_MNT_ELEVATION.ELEVATIONGRIDCOVERAGE.LAMB93"


def S(name, kind, lat, lon, half_km=2.0, note=""):
    return dict(name=f"world-lyon-{name}", kind=kind, country="fr", lat=lat, lon=lon, half_km=half_km, note=note)


SITES = [
    S("corbas", "mining", 45.665, 4.930, note="Corbas and Mions gravel pits and logistics platforms"),
    S("oullins", "building", 45.705, 4.815, note="Metro B extension to Oullins and Saint-Genis-Laval Hopitaux Sud"),
    S("part-dieu", "building", 45.762, 4.865, note="Part-Dieu and Villette works, dense city; DTM under buildings"),
    S("confluence", "building", 45.743, 4.828, note="Confluence, Perrache and Gerland works; Rhone and Saone in view"),
    S("meyzieu", "mining", 45.795, 4.975, note="Meyzieu, Jonage and Grand Large gravel pits, mostly wet"),
    S("carre-de-soie", "building", 45.765, 4.920, note="Carre de Soie, Vaulx-en-Velin and Villeurbanne east"),
    S("saint-priest", "building", 45.725, 4.950, note="Bron and Saint-Priest, flat east (by its dz)"),
    S("tete-dor", "quiet", 45.7785, 4.854, 0.8, "Parc de la Tete d'Or lawns and lake"),
    S("vernaison", "quiet", 45.661, 4.798, note="farmland and villages southwest of Lyon"),
]


def lhd_dates(site, grid):
    """Earliest start and latest end of the LiDAR HD 1 km tiles under the window (cached in data/world-raw/lyon)."""
    cache = os.path.join(RAW, "lyon", f"{site['name']}_lhd_tiles.json")
    if os.path.exists(cache):
        P = json.load(open(cache))
    else:
        W, S_, E, N = grid[:4]
        inv = Transformer.from_crs(CRS_LY, 4326, always_xy=True)
        lon, lat = inv.transform([W, E, W, E], [S_, S_, N, N])
        bbox = f"{min(lat)},{min(lon)},{max(lat)},{max(lon)},urn:ogc:def:crs:EPSG::4326"
        q = dict(SERVICE="WFS", VERSION="2.0.0", REQUEST="GetFeature", TYPENAMES="IGNF_LIDAR-HD_METADONNEE:metadata",
                 OUTPUTFORMAT="application/json", COUNT=1000, BBOX=bbox,
                 PROPERTYNAME="coordonnees_nw,date_debut_acquisition,date_fin_acquisition")
        r = requests.get(WFS, params=q, timeout=120)
        r.raise_for_status()
        P = [f["properties"] for f in r.json()["features"]]
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        json.dump(P, open(cache, "w"))
    d0 = min(datetime.date.fromisoformat(p["date_debut_acquisition"][:10]) for p in P)
    d1 = max(datetime.date.fromisoformat(p["date_fin_acquisition"][:10]) for p in P)
    return d0, d1


def lhd(site, grid):
    """LiDAR HD MNT on the 2 m grid by one GetMap (the service allows 5010 px a side), cached as a GeoTIFF."""
    W, S_, E, N, w, h = grid
    f = os.path.join(RAW, "lyon", f"{site['name']}_lhd_2m.tif")
    if not os.path.exists(f):
        p = dict(SERVICE="WMS", VERSION="1.3.0", REQUEST="GetMap", LAYERS=LAYER, STYLES="", CRS="EPSG:3946",
                 BBOX=f"{W},{S_},{E},{N}", WIDTH=w, HEIGHT=h, FORMAT="image/geotiff")
        for k in range(5):
            r = requests.get(WMS, params=p, timeout=600)
            if r.status_code == 200 and "tiff" in r.headers.get("content-type", ""):
                break
            time.sleep(2 + 3 * k)  # the service allows about one request a second
        else:
            raise RuntimeError(f"LiDAR HD GetMap failed: {r.status_code} {r.text[:200]}")
        os.makedirs(os.path.dirname(f), exist_ok=True)
        open(f, "wb").write(r.content)
    with rasterio.open(f) as ds:
        a = ds.read(1).astype(np.float32)
    a[a < -1000] = np.nan
    return a


def lyon2023(grid):
    """The Metropole's 2023 MNT on the 2 m grid: 0.5 m windows read at full resolution, averaged 4 x 4."""
    W, S_, E, N, w, h = grid
    out = np.full((h, w), np.nan, np.float32)
    k = int(round(STEP / 0.5))
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="2"):
        with rasterio.open(LYON) as ds:
            b = ds.bounds
            x0, x1 = max(W, b.left), min(E, b.right)
            y0, y1 = max(S_, b.bottom), min(N, b.top)
            c0, c1 = int(np.ceil((x0 - W) / STEP)), int(np.floor((x1 - W) / STEP))
            r0, r1 = int(np.ceil((N - y1) / STEP)), int(np.floor((N - y0) / STEP))
            if c1 <= c0 or r1 <= r0:
                return out
            inv = ~ds.transform
            col, row = inv * (W + c0 * STEP, N - r0 * STEP)
            col, row = int(round(col)), int(round(row))
            for rr in range(r0, r1, 250):  # 500 m strips
                n = min(r1, rr + 250) - rr
                win = Window(col, row + (rr - r0) * k, (c1 - c0) * k, n * k)
                a = ds.read(1, window=win, masked=True).astype(np.float32).filled(np.nan)
                a[a < -1000] = np.nan
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN blocks stay NaN
                    out[rr:rr + n, c0:c1] = np.nanmean(a.reshape(n, k, c1 - c0, k), axis=(1, 3))
    return out


def dems(site):
    grid = window(site, CRS_LY)
    a = lhd(site, grid)
    b = lyon2023(grid)
    return dict(a=a, b=b, transform=from_origin(grid[0], grid[3], STEP, STEP), crs=CRS_LY,
                first=lhd_dates(site, grid), second=LYON_DATES,
                source="IGN LiDAR HD MNT (WMS-R) / Metropole de Lyon MNT 2023 COG")
