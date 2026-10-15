"""Switzerland: swisstopo's swissALTI3D DTM releases in the northeast, and Geneva's canton DTMs (SITG).

swissALTI3D (free OGD, cite "Source: Federal Office of Topography swisstopo") is a 0.5 m and 2 m DTM on 1 km
tiles in LV95 (EPSG 2056) with LN02 heights, the same datum in every release. The STAC
(data.geo.admin.ch/api/stac/v1, collection ch.swisstopo.swissalti3d) keeps several dated releases per tile, named
by release year: in the northeast (SH, ZH, TG, ZG, SZ, GL, SG, AR, AI) the 2019 and 2020 releases are built from
the 2017/2018 lidar and the 2026 release from the 2024 lidar. The 2 m COGs are read by window. Where a tile has
both, the 2019 release is used, since the 2020 edition patched ground that changed by more than 30 to 50 cm from
2019 aerial photos.

Caveat: swissALTI3D patches changed ground photogrammetrically, so at exactly the dug places the first state may
date from 2019 and the second from after the 2024 flight; the lidar flight days below are a lower bound there.

Flight days come from the swissSURFACE3D point clouds of the same tiles: the 2024 cloud is COPC, whose info VLR
(the first 2 KB) holds the GPS time range of the whole tile; the 2017/2018 cloud is a zipped LAS, whose first
64 KB inflate to the header and the first few thousand points, a sample of the tile's flight day. A site's
flight spans its tiles' days. When a header cannot be read, the cloud's year stands in.

Geneva (SITG, open data): the canton DTMs MNA_TERRAIN_2019_03 (0.5 m, flown 2019-03-05 and 06) and
MNA_TERRAIN_2023_03 (0.2 m, flown 2023-03-16 to 21) are canton-wide COGs in LV95 with LN02 heights, averaged down
to 2 m from their full-resolution pixels. Their dates are the flight nights given by SITG (no per-tile dates).

Geneva's DTMs draw rivers and lakes as smooth surfaces at each flight's water level (the Rhone rose 0.6 m).
Cells smooth to 2 mm in both flights and more than 0.1 m apart are left out (still water stays, as unchanged).
swissALTI3D's rivers are textured, not drawn smooth, so there a river whose level changed still reads as moved
(the Thur at Andelfingen fell 0.7 m over 3,000 AlphaEarth pixels): the sites keep off the big rivers.
"""

import collections
import datetime
import json
import os
import struct
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from pyproj import Transformer
from rasterio.crs import CRS
from scipy import ndimage

from world_sites import RAW, STEP, read_cogs, window

CRS_CH = CRS.from_epsg(2056)
STAC = "https://data.geo.admin.ch/api/stac/v1/search"
FIRST, SECOND = (2019, 2020), (2026,)
GE = "/vsicurl/https://ge.ch/sitg/geodata/SITG/COG-DATA/MNA_TERRAIN_{}.tif"
GE_FLIGHTS = {"2019_03": (datetime.date(2019, 3, 5), datetime.date(2019, 3, 6)),
              "2023_03": (datetime.date(2023, 3, 16), datetime.date(2023, 3, 21))}
SMOOTH = 0.002  # m: rougher than this is measured ground
CAVEAT = "swissALTI3D patches changed ground from aerial photos, so the true dates there may be later"


def S(name, kind, lat, lon, half_km=2.0, note="", src="alti"):
    if src == "alti":
        note = f"{note}; {CAVEAT}" if note else CAVEAT
    return dict(name=f"world-swiss-{name}", kind=kind, country="ch", lat=lat, lon=lon, half_km=half_km, note=note,
                src=src)


SITES = [
    S("huentwangen", "mining", 47.5982, 8.5280, 2.0, "Huentwangen and Rafz gravel pits, cut up to 57 m and backfill"),
    S("weiach", "mining", 47.566, 8.452, 2.0, "Weiach gravel pits by the Rhine; tiles without a 2026 release left out"),
    S("zurich-city", "building", 47.3775, 8.5345, 2.0, "Zurich center to Zurich West: Europaallee and city pits"),
    S("zurich-airport", "building", 47.4450, 8.5560, 2.0, "Zurich airport, The Circle, and a fill south of it"),
    S("weinland", "quiet", 47.627, 8.690, 2.0, "Zurich Weinland fields, vineyards and villages (Truellikon, "
      "Marthalen) north of the Thur, little dug"),
    S("uetliberg", "quiet", 47.349, 8.491, 2.0, "Uetliberg forest and Albis slopes, steep, little dug"),
    S("geneva-vernets", "building", 46.194, 6.140, 2.0, "Les Vernets (former barracks) housing pits, Praille-Acacias",
      src="ge"),
    S("geneva-cherpines", "building", 46.188, 6.100, 2.0, "Les Cherpines new district earthworks, Plan-les-Ouates",
      src="ge"),
    S("geneva-satigny", "survey", 46.210, 6.010, 2.0, "Satigny and Peney vineyards, fields and gravel workings; "
      "little moved between the flights", src="ge"),
]


def stac(collection, bbox):
    """{tile: {release year: [asset hrefs]}} of a STAC collection over a lon/lat bbox."""
    out = collections.defaultdict(dict)
    body = {"collections": [collection], "bbox": list(bbox), "limit": 100}
    url, method = STAC, "POST"
    while url:
        r = requests.post(url, json=body, timeout=120) if method == "POST" else requests.get(url, timeout=120)
        d = json.loads(r.text, strict=False)
        for f in d["features"]:
            _, year, tile = f["id"].split("_")
            out[tile][int(year)] = [a["href"] for a in f["assets"].values()]
        nxt = [l for l in d.get("links", []) if l["rel"] == "next"]
        url = nxt[0]["href"] if nxt else None
        method = nxt[0].get("method", "GET") if nxt else None
        body = nxt[0].get("body", body) if nxt else None
    return out


def gps_day(t, adjusted):
    """A LAS GPS time (adjusted standard: seconds since 1980-01-06 minus 1e9) as a date."""
    return (datetime.datetime(1980, 1, 6) + datetime.timedelta(seconds=t + (1e9 if adjusted else 0) - 18)).date()


def copc_days(url):
    """(first, last) GPS day of a COPC file, from its info VLR in the first 2 KB."""
    b = requests.get(url, headers={"Range": "bytes=0-2047"}, timeout=60).content
    hs = struct.unpack_from("<H", b, 94)[0]
    tmin, tmax = struct.unpack_from("<2d", b, hs + 54 + 8 * 5 + 16)
    return gps_day(tmin, True), gps_day(tmax, True)


def laszip_days(url):
    """(first, last) GPS day of the first points of a zipped LAS (point formats 1 and 3: GPS time at byte 20)."""
    b = requests.get(url, headers={"Range": "bytes=0-65535"}, timeout=60).content
    if b[:4] != b"PK\x03\x04":
        return None
    method = struct.unpack_from("<H", b, 8)[0]
    nl, el = struct.unpack_from("<HH", b, 26)
    data = b[30 + nl + el:]
    raw = zlib.decompressobj(-15).decompress(data) if method == 8 else data
    if raw[:4] != b"LASF":
        return None
    enc = struct.unpack_from("<H", raw, 6)[0]
    off = struct.unpack_from("<I", raw, 96)[0]
    fmt, rl = raw[104], struct.unpack_from("<H", raw, 105)[0]
    if fmt not in (1, 3) or not enc & 1:
        return None
    n = (len(raw) - off) // rl
    t = [struct.unpack_from("<d", raw, off + i * rl + 20)[0] for i in range(n)]
    return (gps_day(min(t), True), gps_day(max(t), True)) if t else None


def year_days(y):
    return datetime.date(y, 1, 1), datetime.date(y, 12, 31)


_days = None


def tile_days(tile, surf, years):
    """(first, last) flight day of the tile's swissSURFACE3D cloud from one of the years (cached)."""
    global _days
    cache = os.path.join(RAW, "swiss", "flight_days.json")
    if _days is None:
        _days = json.load(open(cache)) if os.path.exists(cache) else {}
    have = [y for y in sorted(surf.get(tile, {})) if y in years]
    if not have:
        return None
    y = have[0]
    key = f"{tile}_{y}"
    if key not in _days:
        days = None
        for h in surf[tile][y]:
            try:
                if h.endswith(".copc.laz"):
                    days = copc_days(h)
                elif h.endswith(".las.zip"):
                    days = laszip_days(h)
            except Exception as e:
                print(f"    {key}: no flight day ({type(e).__name__}: {e})", flush=True)
            if days:
                break
        days = days or year_days(y)
        _days[key] = [d.isoformat() for d in days]
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        json.dump(_days, open(cache, "w"), indent=0)
    return tuple(datetime.date.fromisoformat(d) for d in _days[key])


def span(days):
    return min(d[0] for d in days), max(d[1] for d in days)


def dems_alti(site):
    grid = window(site, CRS_CH)
    W, S_, E, N, _, _ = grid
    inv = Transformer.from_crs(CRS_CH, 4326, always_xy=True)
    xs, ys = inv.transform([W, E, W, E], [S_, S_, N, N])
    bbox = (min(xs) - 1e-3, min(ys) - 1e-3, max(xs) + 1e-3, max(ys) + 1e-3)
    alti, surf = stac("ch.swisstopo.swissalti3d", bbox), stac("ch.swisstopo.swisssurface3d", bbox)
    A, B, da, db = [], [], [], []
    for tile, rel in sorted(alti.items()):
        e, n = (int(v) * 1000 for v in tile.split("-"))
        if e + 1000 <= W or e >= E or n + 1000 <= S_ or n >= N:
            continue
        f = [y for y in FIRST if y in rel]
        s = [y for y in SECOND if y in rel]
        if not f or not s:
            continue

        def cog(y):
            return "/vsicurl/" + next(h for h in rel[y] if h.endswith("_2_2056_5728.tif"))
        A.append(cog(f[0]))
        B.append(cog(s[-1]))
        da.append(tile_days(tile, surf, range(2016, 2020)))
        db.append(tile_days(tile, surf, range(2023, 2026)))
    if not A:
        return None
    # tiles with no cloud of the flight (the border, the Rhine) take the span of the others
    da, db = [d for d in da if d] or [year_days(2018)], [d for d in db if d] or [year_days(2024)]
    a, tr = read_cogs(A, CRS_CH, grid)
    b, _ = read_cogs(B, CRS_CH, grid)
    return dict(a=a, b=b, transform=tr, crs=CRS_CH, first=span(da), second=span(db),
                source=f"swisstopo swissALTI3D 2 m, release 2019/2020 vs 2026 ({len(A)} tiles), LN02")


def read_native(url, grid, block=1000.0):
    """A fine COG averaged down onto the 2 m grid from its full-resolution pixels, 1 km at a time (cached in
    data/world-raw/swiss). Reading through GDAL's overviews instead (read_cogs) lets each year's own pyramid
    resample the slopes: at Les Cherpines the steep cells (slope over 0.4) then differ by a median 0.65 m between
    the years, against 0.08 m from full resolution. float32, NaN where any fine pixel is missing."""
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.windows import Window
    W, S_, E, N, w, h = grid
    tr = from_origin(W, N, STEP, STEP)
    cache = os.path.join(RAW, "swiss", f"{os.path.basename(url)[:-4]}_{W:.0f}_{N:.0f}_{w}x{h}.npy")
    if os.path.exists(cache):
        return np.load(cache), tr
    out = np.full((h, w), np.nan, np.float32)
    nb = int(block / STEP)
    env = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="2",
               CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif")

    def one(rc):
        r0, c0 = rc
        rh, cw = min(nb, h - r0), min(nb, w - c0)
        with rasterio.Env(**env), rasterio.open(url) as src:
            res = src.res[0]
            f = int(round(STEP / res))
            assert abs(f * res - STEP) < 1e-6, res
            col, row = ~src.transform * (W + c0 * STEP, N - r0 * STEP)
            a = src.read(1, window=Window(int(round(col)), int(round(row)), cw * f, rh * f), boundless=True,
                         masked=True)
        a = a.astype(np.float32).filled(np.nan)
        a[a < -1000] = np.nan
        out[r0:r0 + rh, c0:c0 + cw] = a.reshape(rh, f, cw, f).mean(axis=(1, 3))
    with ThreadPoolExecutor(3) as ex:
        list(ex.map(one, [(r0, c0) for r0 in range(0, h, nb) for c0 in range(0, w, nb)]))
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.save(cache, out)
    return out, tr


def dems_ge(site):
    grid = window(site, CRS_CH)
    a, tr = read_native(GE.format("2019_03"), grid)
    b, _ = read_native(GE.format("2023_03"), grid)
    return dict(a=a, b=b, transform=tr, crs=CRS_CH, first=GE_FLIGHTS["2019_03"], second=GE_FLIGHTS["2023_03"],
                source="SITG Geneva MNA_TERRAIN 2019_03 (0.5 m) vs 2023_03 (0.2 m), LN02")


def interpolated(z):
    """Cells on a surface the DTM drew rather than measured (a river, a lake, under a building): its departure
    from the 3 x 3 mean, averaged over 5 x 5, is under 2 mm. Measured ground scores 1 to 10 cm."""
    rough = ndimage.uniform_filter(np.abs(z - ndimage.uniform_filter(z, 3)), 5)
    with np.errstate(invalid="ignore"):
        return rough < SMOOTH


def dems(site):
    got = dems_alti(site) if site["src"] == "alti" else dems_ge(site)
    if got is None:
        return None
    # water drawn in both flights at another level is not dug ground: leave it out (still water stays)
    a, b = got["a"], got["b"]
    with np.errstate(invalid="ignore"):
        off = np.abs(b - a - np.nanmedian(b - a)) > 0.1
    # the 5 x 5 test misses the last cells to each bank: grow 3 cells (6 m)
    drawn = ndimage.binary_dilation(interpolated(a) & interpolated(b), iterations=3) & off
    a[drawn] = b[drawn] = np.nan
    return got
