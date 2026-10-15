"""England: the Environment Agency's National LIDAR Programme (NLP) 1 m DTM, blocks flown two or three times.

Phase 1 flew England in 25 km blocks from 2017 to 2023, and many blocks were flown again (West London three
times: 2018-01, 2020-12 to 2021-02, 2023-01 to 03). Open Government Licence v3.0. Heights are meters above
Ordnance Datum Newlyn (OSTN15/OSGM15 for every survey since 2016-08), British National Grid (EPSG:27700).

The dates come per 5 km tile from the WFS National_LIDAR_Programme_Index_Catalogue (sd_flown, ed_flown: the
first and last flight day over that tile); its downloadable GeoPackage stops at 2022, so the WFS is used. The
tiles cannot be read by window: each is a whole zip of about 50 to 70 MB (catalog search POST, then a GET of the
tile per year), kept in data/world-raw/england/ and read through /vsizip/. Per tile the first cover is the
earliest flown from 2018 on and the second the latest flown by the end of 2024; tiles without two such covers
are left out of the window, and a site's dates span the tiles it uses. At tidal sites (the Thames) the river,
its banks and the sea side of the coastline are blanked from OpenStreetMap (ODbL), since the DTM fills water at
the tide's height of the day.
"""

import datetime
import json
import os
import time
import zipfile

import numpy as np
import requests
from rasterio.crs import CRS
from scipy import ndimage

from world_sites import RAW, STEP, read_cogs, window

CRS_GB = CRS.from_epsg(27700)
WFS = ("https://environment.data.gov.uk/spatialdata/survey-index-files/wfs?service=WFS&version=2.0.0"
       "&request=GetFeature&outputFormat=application/json&srsName=EPSG:27700"
       "&typeNames=dataset-9f0fa3fc-a860-4729-adc9-47fe53f658d0:National_LIDAR_Programme_Index_Catalogue")
SEARCH = "https://environment.data.gov.uk/backend/catalog/api/tiles/collections/survey/search"
TILE = "https://environment.data.gov.uk/tiles/collections/survey/national_lidar_programme_dtm/{year}/1/{tile}"
PRODUCT = "national_lidar_programme_dtm"
DIR = os.path.join(RAW, "england")
OVERPASS = ("https://overpass-api.de/api/interpreter", "https://overpass.private.coffee/api/interpreter")
SEAWARD_M = 500  # how far past the coastline the foreshore and sea are blanked
FROM, UNTIL = datetime.date(2018, 1, 1), datetime.date(2024, 12, 31)


def S(name, kind, lat, lon, half_km=2.0, note="", **kw):
    return dict(name=f"world-england-{name}", kind=kind, country="gb", lat=lat, lon=lon, half_km=half_km, note=note,
                **kw)


SITES = [
    # West London, block P_10752 > P_12150 > P_12757: 2018-01-18 against 2023-01 to 03
    S("brent-cross", "building", 51.5705, -0.2215, note="Brent Cross Town regeneration earthworks 2020 to 2023; "
      "the east edge (tile TQ2585, no 2023 cover) is left out"),
    S("earls-court", "building", 51.490, -0.212, tidal=True, note="Earls Court west half (tile TQ2075); the "
      "exhibition halls were cleared by 2018-01, so little ground work may fall between the flights; the east half "
      "(TQ2575, first flown 2017-12) is left out; the tidal Thames and its banks are blanked; dense city, so many "
      "moved patches are a few pixels where the ground under buildings was interpolated differently"),
    S("old-oak", "building", 51.5265, -0.2465, note="Old Oak Common HS2 station box and the Atlas Road and "
      "Victoria Road works"),
    S("colne-valley", "building", 51.567, -0.425, note="HS2 Colne Valley viaduct and West Ruislip portal, with "
      "the lakes of the Colne Valley (gravel pits)"),
    S("wembley", "building", 51.561, -0.279, note="Wembley Park housing plots"),
    S("southall", "building", 51.506, -0.383, note="Southall gasworks housing (The Green Quarter) and the Crossrail "
      "depot area"),
    S("northolt", "building", 51.548, -0.370, note="HS2 Northolt tunnel shafts and the West Ruislip to Old Oak "
      "corridor, otherwise suburbs"),
    S("colindale", "building", 51.600, -0.250, 1.5, "Colindale housing (Peel Centre, Grahame Park) among "
      "Hendon and Mill Hill streets"),
    # Richmond Park block, 2018-03 to 04 against 2023-01
    S("richmond-park", "quiet", 51.443, -0.275, 1.5, "Richmond Park grassland and woods with Roehampton and "
      "Kingston streets at the edges, quiet"),
    # East Midlands, 2018-01 to 04 against 2022-01
    S("mountsorrel", "mining", 52.728, -1.148, note="Mountsorrel granite quarry"),
    S("bardon", "mining", 52.712, -1.322, note="Bardon Hill quarry"),
    S("breedon", "mining", 52.806, -1.398, note="Breedon on the Hill limestone quarry and Cloud Hill quarry"),
    S("dene", "mining", 53.078, -1.566, note="Wirksworth limestone quarries (moved ground in the northwest of "
      "the window); otherwise Peak District farmland"),
    S("east-midlands-gateway", "building", 52.836, -1.305, note="East Midlands Gateway logistics platforms and "
      "rail freight terminal; East Midlands Airport to the west"),
    S("leicestershire", "quiet", 52.680, -1.000, 1.5, "Leicestershire farmland, quiet"),
    # Cambridgeshire, 2019-12 or 2020-01 against 2023-02 or 04
    S("a14-huntingdon", "building", 52.327, -0.177, note="A14 Huntingdon viaduct removal and regrading 2021 to 2022"),
    S("black-cat", "building", 52.175, -0.295, note="A428 Black Cat junction, works from 2022 (part of them may "
      "be after the second flight)"),
    S("northstowe", "building", 52.283, 0.040, note="Northstowe new town groundworks"),
    S("waterbeach", "building", 52.270, 0.190, note="Waterbeach new town groundworks"),
    # Cornwall, 2019-12 against 2023-02
    S("st-austell", "mining", 50.380, -4.830, note="St Austell china clay pits and tips (tile SW9555 only)"),
    # Thurrock, 2018-12 to 2019-01 against 2021-12
    S("tilbury2", "building", 51.462, 0.368, tidal=True, note="Tilbury2 port terminal, built 2018 to 2020; the "
      "tidal Thames banks are blanked"),
]


def _cache(name):
    os.makedirs(DIR, exist_ok=True)
    return os.path.join(DIR, name)


def covers(site, grid):
    """The NLP DTM records (tilename, year, sd_flown, ed_flown) over the window, from the WFS (cached)."""
    W, S_, E, N = grid[:4]
    f = _cache(f"index_{site['name']}.json")
    if not os.path.exists(f):
        r = requests.get(f"{WFS}&bbox={W},{S_},{E},{N},EPSG:27700", timeout=120)
        r.raise_for_status()
        json.dump([{k: p[k] for k in ("tilename", "year", "sd_flown", "ed_flown", "polygon_id", "dtm_fn")}
                   for p in (x["properties"] for x in r.json()["features"])], open(f, "w"))
    return json.load(open(f))


def uris(site):
    """The NLP DTM tile zips the catalog search lists over the site's window, {(tile, year): uri} (cached)."""
    f = _cache(f"search_{site['name']}.json")
    if not os.path.exists(f):
        from pyproj import Transformer
        grid = window(site, CRS_GB)
        to = Transformer.from_crs(CRS_GB, 4326, always_xy=True)
        W, S_, E, N = grid[:4]
        ring = [list(to.transform(x, y)) for x, y in ((W, S_), (E, S_), (E, N), (W, N), (W, S_))]
        r = requests.post(SEARCH, data=json.dumps({"type": "Polygon", "coordinates": [ring]}),
                          headers={"Content-Type": "application/geo+json"}, timeout=120)
        r.raise_for_status()
        json.dump([x["uri"] for x in r.json()["results"] if x["product"]["id"] == PRODUCT], open(f, "w"))
    return {(u.rsplit("/", 1)[1], int(u.split("/")[-3])): u for u in json.load(open(f))}


def fetch(tile, year, uri):
    """The tile's zip in data/world-raw/england/nlp/ (downloaded once) as a /vsizip/ path to the DTM inside, or
    None when the server will not give it (a few listed tiles answer 404)."""
    f = _cache(os.path.join("nlp", f"{year}_{tile}.zip"))
    for attempt in range(3):
        if os.path.exists(f):
            break
        os.makedirs(os.path.dirname(f), exist_ok=True)
        try:
            with requests.get(uri, stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(f + ".part", "wb") as out:
                    for chunk in r.iter_content(1 << 20):
                        out.write(chunk)
            os.replace(f + ".part", f)
        except requests.RequestException as e:
            print(f"    {year} {tile}: {e}", flush=True)
            time.sleep(5 * (attempt + 1))
    if not os.path.exists(f):
        return None
    tif = next(n for n in zipfile.ZipFile(f).namelist() if n.lower().endswith(".tif"))
    return f"/vsizip/{f}/{tif}"


def osm_water(site, grid):
    """Mask of the 2 m grid under OpenStreetMap's water and tidal mud and the sea side of its coastline (ODbL;
    Overpass, cached), widened by 6 m to take in river walls and banks. Used at tidal sites: the NLP fills a river
    with a surface at whatever height the tide stood, so the same water reads meters apart between flights, and
    the exposed foreshore is not flat."""
    from pyproj import Transformer
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    from shapely.geometry import LineString, Polygon
    from shapely.ops import linemerge, polygonize, transform, unary_union
    W, S_, E, N, w, h = grid
    to = Transformer.from_crs(CRS_GB, 4326, always_xy=True)
    lon0, lat0 = to.transform(W, S_)
    lon1, lat1 = to.transform(E, N)
    bb = f"({lat0:.5f},{lon0:.5f},{lat1:.5f},{lon1:.5f})"
    f = _cache(f"osm_water_{site['name']}.json")
    if not os.path.exists(f):
        q = "".join(f'{t}["{k}"~"{v}"]{bb};' for t in ("way", "relation") for k, v in
                    (("natural", "^(water|mud|coastline)$"), ("waterway", "^riverbank$"), ("wetland", "^tidalflat$")))
        for attempt in range(6):
            url = OVERPASS[attempt % len(OVERPASS)]
            try:
                r = requests.post(url, data={"data": f"[out:json][timeout:120];({q});out geom;"},
                                  headers={"User-Agent": "aef-tessera-earthwork/0.1 (research)"}, timeout=180)
                r.raise_for_status()
                json.dump(r.json()["elements"], open(f, "w"))
                break
            except (requests.RequestException, ValueError) as e:
                print(f"    Overpass {url}: {e}", flush=True)
                time.sleep(10 * (attempt + 1))
        else:
            raise RuntimeError("no Overpass server answered")
    polys, coast = [], []
    for e in json.load(open(f)):
        if e.get("tags", {}).get("natural") == "coastline":
            # the coastline is mean high water with the sea on its right: the foreshore lies seaward of it
            coast.append(LineString([(p["lon"], p["lat"]) for p in e["geometry"]]))
        elif e["type"] == "way" and len(e.get("geometry", [])) >= 4 and e["geometry"][0] == e["geometry"][-1]:
            polys.append(Polygon([(p["lon"], p["lat"]) for p in e["geometry"]]))
        elif e["type"] == "relation":
            rings = {"outer": [], "inner": []}
            for m in e.get("members", []):
                if m.get("type") == "way" and len(m.get("geometry", [])) >= 2:
                    rings.setdefault(m.get("role") or "outer", []).append(
                        LineString([(p["lon"], p["lat"]) for p in m["geometry"]]))
            outer = unary_union(list(polygonize(linemerge(rings["outer"])))) if rings["outer"] else None
            if outer is not None and not outer.is_empty:
                if rings["inner"]:
                    outer = outer.difference(unary_union(list(polygonize(linemerge(rings["inner"])))))
                polys.append(outer)
    fwd = Transformer.from_crs(4326, CRS_GB, always_xy=True).transform
    polys = [transform(fwd, p.buffer(0)) for p in polys]
    polys += [transform(fwd, c).buffer(-SEAWARD_M, single_sided=True) for c in coast]
    polys = [p for p in polys if not p.is_empty]
    if not polys:
        return np.zeros((h, w), bool)
    m = rasterize([(p, 1) for p in polys], out_shape=(h, w), transform=from_origin(W, N, STEP, STEP),
                  fill=0, dtype="uint8").astype(bool)
    return ndimage.binary_dilation(m, iterations=3)


def day(s):
    return datetime.date.fromisoformat(s[:10])


def dems(site):
    grid = window(site, CRS_GB)
    by = {}
    for p in covers(site, grid):
        by.setdefault(p["tilename"], []).append(p)
    found = uris(site)
    pick = []
    for tile, cs in sorted(by.items()):
        cs = sorted(cs, key=lambda p: p["sd_flown"])
        ok = [p for p in cs if day(p["sd_flown"]) >= FROM and day(p["ed_flown"]) <= UNTIL]
        if len(ok) >= 2 and day(ok[-1]["sd_flown"]) > day(ok[0]["ed_flown"]):
            pick.append((tile, ok[0], ok[-1]))
    if not pick:
        return None
    paths = {}
    for tile, p1, p2 in pick:
        got = [fetch(tile, p["year"], found.get((tile, p["year"]), TILE.format(year=p["year"], tile=tile)))
               for p in (p1, p2)]
        if None not in got:
            paths[tile] = got
    pick = [x for x in pick if x[0] in paths]
    if not pick:
        return None
    a, tr = read_cogs([paths[t][0] for t, *_ in pick], CRS_GB, grid)
    b, _ = read_cogs([paths[t][1] for t, *_ in pick], CRS_GB, grid)
    d1 = (min(day(p1["sd_flown"]) for _, p1, _ in pick), max(day(p1["ed_flown"]) for _, p1, _ in pick))
    d2 = (min(day(p2["sd_flown"]) for _, _, p2 in pick), max(day(p2["ed_flown"]) for _, _, p2 in pick))
    if site.get("tidal"):
        wet = osm_water(site, grid)
        a[wet] = np.nan
        b[wet] = np.nan
    blocks = sorted({f"{p1['polygon_id']} > {p2['polygon_id']}" for _, p1, p2 in pick})
    return dict(a=a, b=b, transform=tr, crs=CRS_GB, first=d1, second=d2,
                source=f"EA National LIDAR Programme DTM 1 m, tiles {' '.join(t for t, *_ in pick)}, "
                       f"blocks {', '.join(blocks)}")
