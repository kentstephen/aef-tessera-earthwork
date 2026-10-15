"""Scotland: the Scottish Remote Sensing Portal's public sector lidar, Phase 4 against Phase 5 or Phase 6.

The SRSP open data bucket (s3://srsp-open-data, eu-west-2, Open Government Licence v3.0) holds each phase's 0.5 m
DTM as Cloud Optimized GeoTIFFs on 5 km quarter tiles of the British National Grid (EPSG:27700), e.g.
lidar/phase-4/dtm/27700/gridded/NT07SW_50CM_DTM_PHASE4.tif, read here by window at 2 m over HTTP. Heights are
meters above Ordnance Datum Newlyn (OSGM15 stated for Phase 6; Phase 4 and 5 do not say). Phase 4 flew the
Lothians, Falkirk and Dunfermline in 2018-05/06 and Fife north and Ayrshire in 2017-09/10; Phase 5 flew 2020-06
to 08 and Phase 6 Ayrshire 2021-04 to 06. Between phases there is a block bias of a few cm (P4 to P5) to 12 to
15 cm (P4 to P6); the median removal takes out a constant.

Water is blanked by OpenStreetMap's water polygons (ODbL), since the phases fill lochs differently, and at
sites by the Forth the ground below a set height (the tide's reach) is blanked too.

No index carries flight dates. They come from the point clouds: the GPS time of the first point of each 1 km LAZ
tile in the window (two small Range reads a tile), and a site's flight spans the days found. Where no LAZ answers,
the phase's flight months stand in.
"""

import datetime
import json
import os
import re
import struct
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from pyproj import Transformer
from rasterio.crs import CRS

from world_england import osm_water
from world_sites import RAW, read_cogs, window

BUCKET = "https://srsp-open-data.s3.eu-west-2.amazonaws.com"
DTM = "lidar/phase-{n}/dtm/27700/gridded/{t}_50CM_DTM_PHASE{n}.tif"
LAZ = "lidar/phase-{n}/laz/27700/gridded/{t}_4PPM_LAS_PHASE{n}.laz"
CRS_GB = CRS.from_epsg(27700)
DIR = os.path.join(RAW, "scotland")
GPS0 = datetime.datetime(1980, 1, 6)
# each phase's flight months, used only where no LAZ date is found
MONTHS = {4: (datetime.date(2017, 9, 1), datetime.date(2018, 6, 30)),
          5: (datetime.date(2020, 6, 1), datetime.date(2020, 8, 31)),
          6: (datetime.date(2021, 4, 1), datetime.date(2021, 6, 30))}


def S(name, kind, lat, lon, phases=(4, 5), half_km=2.0, note="", **kw):
    return dict(name=f"world-scotland-{name}", kind=kind, country="gb", lat=lat, lon=lon, phases=phases,
                half_km=half_km, note=note, **kw)


SITES = [
    # Phase 4 (2018-05/06) against Phase 5 (2020-06 to 08): Lothians, Falkirk, west Fife
    S("crossgates", "mining", 56.090, -3.330, note="fill near Crossgates (likely restoration of the Muir Dean "
      "opencast coal site) and a quarry or tip by Hillend, west Fife"),
    S("winchburgh", "building", 55.957, -3.460, note="Winchburgh new town groundworks and the M9 junction"),
    S("calderwood", "building", 55.898, -3.475, note="Calderwood housing at East Calder and the Raw Camps quarry"),
    S("ratho", "mining", 55.918, -3.400, note="Hillwood quarry at Ratho, with Ratho and the Union Canal"),
    S("levenseat", "mining", 55.826, -3.630, note="Levenseat sand quarry and landfill near the village of Forth, "
      "West Lothian"),
    S("polmont", "mining", 55.990, -3.672, sea_below=3.0, note="Avondale landfill at Polmont, east of Falkirk "
      "(a fill of up to 9 m); ground below 3 m ODN (the Forth's Kinneil mudflats) is blanked"),
    S("kinglassie", "mining", 56.205, -3.225, note="borderline: first flight 2017 over part of the window (Phase 4 "
      "flew it 2017-09 and 2018-06). A dig of about 5 m near Kinglassie, Fife (by its dz), and Holl Reservoir"),
    S("burntisland", "mining", 56.080, -3.255, note="a dig of up to 11 m near Burntisland, Fife (by its dz); the "
      "Forth shore lies to the south"),
    S("kirknewton", "quiet", 55.847, -3.479, half_km=1.5, note="West Lothian farmland south of Kirknewton, quiet"),
    S("avonbridge", "quiet", 55.932, -3.802, half_km=1.5, note="farmland and moor between Falkirk and "
      "Avonbridge, quiet"),
    # Phase 4 (2017-09/10) against Phase 6 (2021-04 to 06): north Ayrshire
    S("dalry", "building", 55.700, -4.715, (4, 6), note="borderline: first flight 2017. Dalry, north Ayrshire: "
      "the A737 Dalry bypass (built 2017 to 2019) and farmland; a block bias of about -12 to -15 cm between phases "
      "and slight slope striping (median |dz| about 0.07 m)"),
]


def square(E, N):
    """The 100 km square letters of the British National Grid holding E, N."""
    e, n = int(E // 100000), int(N // 100000)
    l1 = (19 - n) - (19 - n) % 5 + (e + 10) // 5
    l2 = (19 - n) * 5 % 25 + e % 5
    return "".join(chr(65 + c + (c > 7)) for c in (l1, l2))


def quarter(E, N):
    """The 5 km quarter tile name (NT07SW) holding E, N."""
    return (f"{square(E, N)}{int(E % 100000 // 10000)}{int(N % 100000 // 10000)}"
            f"{'N' if N % 10000 >= 5000 else 'S'}{'E' if E % 10000 >= 5000 else 'W'}")


def km(E, N):
    """The 1 km tile name (NT0273) holding E, N."""
    return f"{square(E, N)}{int(E % 100000 // 1000):02d}{int(N % 100000 // 1000):02d}"


def tiles(grid, size, name):
    W, S_, E, N = grid[:4]
    return sorted({name(x + size / 2, y + size / 2) for x in range(int(W // size * size), int(E), size)
                   for y in range(int(S_ // size * size), int(N), size)})


_keys = {}


def keys(n, sq):
    """Every object under a phase's gridded DTM and LAZ prefixes in one 100 km square (cached)."""
    if (n, sq) in _keys:
        return _keys[(n, sq)]
    f = os.path.join(DIR, f"keys_phase{n}_{sq}.json")
    if not os.path.exists(f):
        out = []
        for kind in ("dtm", "laz"):
            tok = None
            while True:
                q = {"list-type": "2", "prefix": f"lidar/phase-{n}/{kind}/27700/gridded/{sq}"}
                if tok:
                    q["continuation-token"] = tok
                t = requests.get(BUCKET + "/", params=q, timeout=60).text
                out += re.findall(r"<Key>([^<]*)</Key>", t)
                m = re.search(r"<NextContinuationToken>([^<]*)<", t)
                if not m:
                    break
                tok = m.group(1)
        os.makedirs(DIR, exist_ok=True)
        json.dump(out, open(f, "w"))
    _keys[(n, sq)] = set(json.load(open(f)))
    return _keys[(n, sq)]


def first_point_day(url):
    """The day of a LAS or LAZ file's first point, from its GPS time (adjusted standard time), via two Range reads."""
    def rng(a, b):
        r = requests.get(url, headers={"Range": f"bytes={a}-{b}"}, timeout=60)
        r.raise_for_status()
        return r.content
    h = rng(0, 400)
    if h[:4] != b"LASF":
        return None
    genc = struct.unpack("<H", h[6:8])[0]
    off = struct.unpack("<I", h[96:100])[0]
    fmt, laz = h[104] & 0x3F, bool(h[104] & 0x80)
    rlen = struct.unpack("<H", h[105:107])[0]
    gpos = {1: 20, 3: 20, 4: 20, 5: 20, 6: 22, 7: 22, 8: 22, 9: 22, 10: 22}.get(fmt)
    if gpos is None or not genc & 1:
        return None
    p = rng(off, off + 8 + rlen - 1)
    pt = p[8:] if laz else p  # a LAZ file starts its points with the 8 byte chunk table offset, then a raw point
    g = struct.unpack("<d", pt[gpos:gpos + 8])[0]
    return (GPS0 + datetime.timedelta(seconds=g + 1e9)).date()


def flown(site, n, grid):
    """(first, last) day the phase flew over the window, from the first point of each 1 km LAZ tile (cached)."""
    f = os.path.join(DIR, f"dates_{site['name']}_phase{n}.json")
    if not os.path.exists(f):
        urls = [f"{BUCKET}/{LAZ.format(n=n, t=t)}" for t in tiles(grid, 1000, km)
                if LAZ.format(n=n, t=t) in keys(n, t[:2])]

        def one(u):
            try:
                return first_point_day(u)
            except Exception:
                return None
        with ThreadPoolExecutor(8) as ex:
            days = sorted(str(d) for d in ex.map(one, urls) if d)
        # a first point outside the phase's years is a bad GPS time, not a flight
        y0, y1 = MONTHS[n][0].year, MONTHS[n][1].year
        days = [d for d in days if y0 <= int(d[:4]) <= y1]
        json.dump(days, open(f, "w"))
    days = json.load(open(f))
    if not days:
        return MONTHS[n], "phase months"
    return (datetime.date.fromisoformat(days[0]), datetime.date.fromisoformat(days[-1])), f"{len(days)} LAZ tiles"


def dems(site):
    grid = window(site, CRS_GB)
    n1, n2 = site["phases"]
    quarters = tiles(grid, 5000, quarter)
    both = [t for t in quarters
            if DTM.format(n=n1, t=t) in keys(n1, t[:2]) and DTM.format(n=n2, t=t) in keys(n2, t[:2])]
    if not both:
        return None
    got = []
    for n in (n1, n2):
        z, tr = read_cogs([f"/vsicurl/{BUCKET}/{DTM.format(n=n, t=t)}" for t in both], CRS_GB, grid)
        d, how = flown(site, n, grid)
        got.append((z, d, how))
    (a, d1, h1), (b, d2, h2) = got
    # lochs and reservoirs: one flight leaves a void filled by interpolation, the other a surface at its own level,
    # so the same water reads a meter or more apart; blank OpenStreetMap's water, widened 6 m
    wet = osm_water(site, grid)
    a[wet] = np.nan
    b[wet] = np.nan
    if site.get("sea_below") is not None:
        # the Forth's water and intertidal ground lie as smooth ramps at tide height, not flat: blank them by level
        low = (a < site["sea_below"]) | (b < site["sea_below"])
        a[low] = np.nan
        b[low] = np.nan
    return dict(a=a, b=b, transform=tr, crs=CRS_GB, first=d1, second=d2,
                source=f"SRSP lidar DTM 0.5 m Phase {n1} / Phase {n2}, tiles {' '.join(both)}; dates from {h1} / {h2}")
