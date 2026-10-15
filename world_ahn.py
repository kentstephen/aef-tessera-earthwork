"""Netherlands: AHN4 against AHN5, the national 0.5 m bare-earth DTMs (Actueel Hoogtebestand Nederland, CC0 on PDOK).

AHN4 was flown Nov 2019 to Mar 2022 and AHN5 Dec 2022 to Mar 2024 (AHN5 covers the west, north and south; the
east waits for AHN6). Both are 0.5 m DTM tiles of 5 km x 6.25 km in RD New (EPSG:28992), heights on NAP: AHN4 as
Cloud Optimized GeoTIFFs on PDOK, AHN5 as tiled GeoTIFFs (no overviews, CRS tagged EPSG:7415, RD New + NAP height)
on the AHN object storage. Both are read by window over HTTP and averaged down to 2 m from 0.5 m.

The dates come from the flight swaths (AUX/omhullen, the envelope of each flight line with its date), not the
block names: the "2020" AHN4 blocks began flying in Nov 2019 and the "2023" AHN5 tiles include Dec 2022 lines.
Each flight at a site runs from the first to the last date of the swaths over the window.

The east (block Oost, flown Dec 2021 to Mar 2022) has no AHN5 yet; there AHN3 (flown 2018 and 2019) is the first
cover against AHN4. AHN3 exists only as deflated whole-sheet zips (about 240 MB each, no window reads), so its
sheets are downloaded into data/world-raw/ahn/ahn3, and its dates come from the AHN3 flight lines.

The DTMs have no data under buildings and on water, so towns and harbors keep only their open ground.
"""

import datetime
import os
import re

import numpy as np
import rasterio
import requests
from rasterio.crs import CRS
from rasterio.transform import from_origin

from world_sites import RAW, STEP, read_cogs, window

CRS_RD = CRS.from_epsg(28992)
CRS_RD_NAP = CRS.from_epsg(7415)  # AHN5's tag; same horizontal grid as RD New, so nothing is warped
HOST = "https://fsn1.your-objectstorage.com/hwh-ahn"  # basisdata.nl redirects here; go direct for GDAL
PDOK = "https://service.pdok.nl/rws/actueel-hoogtebestand-nederland/atom/downloads/dtm_05m"
# flight swath envelopes with a date each (the flight-line files, AUX/vlieglijnen, miss the 2024 southeast)
SWATHS = {4: "AUX/omhullen/AHN4_omhullen_valid_date.gpkg", 5: "AUX/omhullen/AHN5_omhullen_clip.gpkg"}
AHN3_LINES = "AUX/vlieglijnen/AHN3_clip.gpkg"  # AHN3 has no dated swaths, only lines
LINE_HALF_M = 500  # about half a swath: lines this far outside a window still cover its edge


def S(name, kind, lat, lon, half_km=2.0, note="", pair="ahn5"):
    return dict(name=f"world-ahn-{name}", kind=kind, country="nl", lat=lat, lon=lon, half_km=half_km, note=note,
                pair=pair)


SITES = [
    S("a16-terbregseplein", "building", 51.958, 4.518,
      note="A16 Rotterdam: new motorway, land tunnel and the Terbregseplein junction"),
    S("blankenburg", "building", 51.912, 4.300,
      note="Blankenburgverbinding: Rozenburg ramp fill, the Scheur tunnel approaches and the Maassluis side"),
    S("rijnlandroute", "building", 52.142, 4.452,
      note="Rijnlandroute: A44 junction Leiden-West cut and the Corbulo tunnel at Voorschoten"),
    S("valkenburg", "building", 52.180, 4.428,
      note="Valkenburg (Katwijk): new district on the old air base, sand fills"),
    S("moerdijk-logistics", "building", 51.684, 4.630,
      note="Logistiek Park Moerdijk site preparation and fill"),
    S("hedwigepolder", "survey", 51.352, 4.228, 1.5,
      note="Hedwigepolder depoldering: dikes removed and creeks dug, flooded by the tide in Feb 2022; "
           "AHN5 has no data on the water, so the flooded polder drops out"),
    S("den-oever", "building", 52.938, 5.050,
      note="Afsluitdijk works at Den Oever: locks and pumping station dug, dike reinforcement; much open water"),
    S("grensmaas-meers", "mining", 50.970, 5.730, note="Grensmaas gravel extraction along the Maas near Meers"),
    S("venlo-logistics", "building", 51.392, 6.132,
      note="Venlo logistics belt (Trade Port and Greenport): warehouse platforms and fills"),
    S("maasvlakte", "building", 51.955, 4.035,
      note="Maasvlakte 2 port: terminal platforms and sand depots; the harbor water is no data"),
    # the east has no AHN5 yet: AHN3 (2018, 2019) against AHN4 (Oost, Dec 2021 to Mar 2022), AHN3 by whole tile
    S("groningen-meerstad", "building", 53.2365, 6.6950,
      note="Meerstad and the east edge of Groningen: new housing districts, sand fills and dug water; "
           "AHN3 (Feb 2019) against AHN4, the window kept inside one AHN3 sheet (07GN1)", pair="ahn3"),
    S("heuvelrug", "quiet", 52.060, 5.330, note="Utrechtse Heuvelrug forest, quiet"),
    S("beemster", "quiet", 52.565, 4.885,
      note="Beemster polder fields (west), quiet, clear of the Middenbeemster housing fill; a few cm level shift "
           "between the covers (removed with the median)"),
]


def aux(rel):
    """A small AHN auxiliary file (tile index, flight lines), downloaded once into data/world-raw/ahn."""
    out = os.path.join(RAW, "ahn", os.path.basename(rel))
    if not os.path.exists(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
        r = requests.get(f"{HOST}/{rel}", timeout=300)
        r.raise_for_status()
        with open(out + ".part", "wb") as f:
            f.write(r.content)
        os.replace(out + ".part", out)
    return out


def tiles(grid):
    """The AHN tiles (sheet code and AHN5 0.5 m DTM URL, None where there is no AHN5) over the grid."""
    import pyogrio
    import shapely
    W, S_, E, N = grid[:4]
    df = pyogrio.read_dataframe(aux("AUX/bladwijzer.gpkg"), columns=["AHN", "AHN5_05M_M"], bbox=(W, S_, E, N))
    df = df[df.intersects(shapely.box(W, S_, E, N))]
    return [(r.AHN, r.AHN5_05M_M if isinstance(r.AHN5_05M_M, str) else None) for r in df.itertuples()]


def flown(version, grid):
    """The first and last date of the AHN<version> flight swaths over the grid, leaving out swaths that only
    graze it (under 1% of the window)."""
    import pyogrio
    import shapely
    W, S_, E, N = grid[:4]
    box = shapely.box(W, S_, E, N)
    df = pyogrio.read_dataframe(aux(SWATHS[version]), bbox=(W, S_, E, N))
    df = df[df.intersection(box).area > 0.01 * box.area]
    d = df[[c for c in df.columns if c.startswith("dat")][0]].dropna()
    if d.empty:
        return None
    d = sorted(datetime.date.fromisoformat(str(v)[:10]) for v in d)
    return d[0], d[-1]


_pdok = set()


def pdok():
    """The sheet codes PDOK serves an AHN4 0.5 m DTM for (its atom feed; border sheets can be missing)."""
    if not _pdok:
        out = os.path.join(RAW, "ahn", "pdok_dtm_05m.xml")
        if not os.path.exists(out):
            r = requests.get("https://service.pdok.nl/rws/ahn/atom/dtm_05m.xml", timeout=120)
            r.raise_for_status()
            os.makedirs(os.path.dirname(out), exist_ok=True)
            open(out, "w").write(r.text)
        _pdok.update(re.findall(r"M_([0-9A-Z]+)\.tif", open(out).read()))
    return _pdok


def full(path):
    """path read at full resolution through a vrt:// wrapper. Opening a GeoTIFF that has overviews with the
    OVERVIEW_LEVEL=NONE open option makes GDAL's average ignore nodata: the 3.4e38 fill leaks into 4.6% of the
    2 m cells at the edges of the DTM's holes. The vrt:// ovr=NONE route keeps nodata out."""
    return f"vrt://{path}?ovr=NONE"


def urls(grid):
    """GDAL paths of the AHN4 and AHN5 0.5 m DTM tiles over the grid, wrapped as vrt://...?ovr=NONE (full())."""
    t = tiles(grid)
    u4 = [full(f"/vsicurl/{PDOK}/M_{code}.tif") for code, _ in t if code in pdok()]
    u5 = [full("/vsicurl/" + u.replace("https://basisdata.nl/hwh-ahn", HOST)) for _, u in t if u]
    return u4, u5


def flown3(grid):
    """The first and last date of the AHN3 flight lines over the grid (padded by about half a swath)."""
    import pyogrio
    import shapely
    W, S_, E, N = grid[:4]
    box = (W - LINE_HALF_M, S_ - LINE_HALF_M, E + LINE_HALF_M, N + LINE_HALF_M)
    df = pyogrio.read_dataframe(aux(AHN3_LINES), columns=["datum"], bbox=box)
    d = df[df.intersects(shapely.box(*box))].datum.dropna()
    return (d.min().date(), d.max().date()) if len(d) else None


def ahn3(code):
    """The AHN3 0.5 m DTM of one sheet as a local GeoTIFF: AHN3 comes only as deflated zips (about 240 MB, no
    reading by window), so the zip is downloaded once into data/world-raw/ahn/ahn3 and its one TIF extracted."""
    import shutil
    import zipfile
    d = os.path.join(RAW, "ahn", "ahn3")
    tif = os.path.join(d, f"M_{code}.TIF")
    if os.path.exists(tif):
        return tif
    os.makedirs(d, exist_ok=True)
    z = os.path.join(d, f"M_{code}.zip")
    if not os.path.exists(z):
        with requests.get(f"{HOST}/AHN3/DTM_50cm/M_{code}.zip", stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(z + ".part", "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        os.replace(z + ".part", z)
    with zipfile.ZipFile(z) as zf:
        member = next(m for m in zf.namelist() if m.upper().endswith(".TIF"))
        with zf.open(member) as src, open(tif + ".part", "wb") as dst:  # written under our own name only
            shutil.copyfileobj(src, dst, 1 << 20)
    os.replace(tif + ".part", tif)
    return tif


def read(urls_crs, grid, parts=4):
    """read_cogs of each (urls, crs) over the grid, the grid cut into parts x parts pieces read side by side
    (PDOK serves one stream slowly; a 4 km window is ~200 MB of AHN4 blocks). Returns the arrays and the transform."""
    from concurrent.futures import ThreadPoolExecutor
    W, S_, E, N, w, h = grid
    cols = np.linspace(0, w, parts + 1).astype(int)
    rows = np.linspace(0, h, parts + 1).astype(int)
    jobs = [(k, r0, r1, c0, c1) for k in range(len(urls_crs)) for r0, r1 in zip(rows, rows[1:])
            for c0, c1 in zip(cols, cols[1:])]

    def one(job):
        k, r0, r1, c0, c1 = job
        sub = (W + c0 * STEP, N - r1 * STEP, W + c1 * STEP, N - r0 * STEP, c1 - c0, r1 - r0)
        # each piece fetches its blocks in parallel too; the block cache is capped to hold memory down
        with rasterio.Env(GDAL_NUM_THREADS="4", GDAL_CACHEMAX=256):
            return read_cogs(urls_crs[k][0], urls_crs[k][1], sub)[0]
    out = [np.full((h, w), np.nan, np.float32) for _ in urls_crs]
    with ThreadPoolExecutor(8) as ex:
        for (k, r0, r1, c0, c1), a in zip(jobs, ex.map(one, jobs)):
            out[k][r0:r1, c0:c1] = a
    return out, from_origin(W, N, STEP, STEP)


def dems(site):
    grid = window(site, CRS_RD)
    if site.get("pair") == "ahn3":
        return dems3(site, grid)
    u4, u5 = urls(grid)
    d4, d5 = flown(4, grid), flown(5, grid)
    if not u5 or d4 is None or d5 is None:
        return None
    (a, b), tr = read([(u4, CRS_RD), (u5, CRS_RD_NAP)], grid)
    return dict(a=a, b=b, transform=tr, crs=CRS_RD, first=d4, second=d5,
                source="AHN4 0.5 m DTM (PDOK COG) / AHN5 0.5 m DTM (AHN object storage), NAP")


def dems3(site, grid):
    """AHN3 (whole sheets, local) against AHN4 (PDOK) over the grid."""
    codes = [code for code, _ in tiles(grid) if code in pdok()]
    d3, d4 = flown3(grid), flown(4, grid)
    if not codes or d3 is None or d4 is None:
        return None
    u3 = [full(ahn3(code)) for code in codes]
    u4 = [full(f"/vsicurl/{PDOK}/M_{code}.tif") for code in codes]
    (a, b), tr = read([(u3, CRS_RD), (u4, CRS_RD)], grid)
    return dict(a=a, b=b, transform=tr, crs=CRS_RD, first=d3, second=d4,
                source="AHN3 0.5 m DTM (whole-sheet zip) / AHN4 0.5 m DTM (PDOK COG), NAP")
