"""New Zealand: LINZ's 1 m bare-earth DEMs (s3://nz-elevation, CC BY 4.0), surveys flown twice over one place.

Each survey is one prefix, {region}/{survey}/dem_1m/2193/, of Cloud Optimized GeoTIFFs (LERC) on the 1:10,000
sheet grid in NZTM2000, heights on NZVD2016; the two surveys of a pair share sheet ids and the pixel grid. The
STAC collection gives each survey's start and end (no date per tile), so AlphaEarth's years bracket the whole
survey. Windows are read at 2 m over HTTP.
"""

import datetime
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from rasterio.crs import CRS

from world_sites import RAW, read_cogs, window

BUCKET = "https://nz-elevation.s3-ap-southeast-2.amazonaws.com"
CRS_NZ = CRS.from_epsg(2193)
HAMILTON = ("waikato/hamilton_2019", "waikato/hamilton_2023")
CHCH = ("canterbury/christchurch-and-ashley-river_2018-2019", "canterbury/christchurch_2020-2021")
BANKS = ("canterbury/banks-peninsula_2018-2019", "canterbury/banks-peninsula_2023")
BOP = ("bay-of-plenty/bay-of-plenty_2018-2019", "bay-of-plenty/bay-of-plenty_2024")
NORTH = ("northland/northland_2018-2020", "northland/northland_2024")
HAWKES = ("hawkes-bay/hawkes-bay_2020-2021", "hawkes-bay/hawkes-bay_2023")
OTAGO = ("otago/central-otago_2021", "otago/central-otago_2022-2023")
PALM = ("manawatu-whanganui/palmerston-north_2018", "manawatu-whanganui/manawatu-whanganui_2022-2023")


def S(name, kind, lat, lon, pair, half_km=2.0, note=""):
    return dict(name=f"world-nz-{name}", kind=kind, country="nz", lat=lat, lon=lon, pair=pair, half_km=half_km, note=note)


SITES = [
    S("hamilton-ruakura", "building", -37.7826, 175.3307, HAMILTON, note="Ruakura inland port and logistics platform"),
    S("hamilton-south", "building", -37.845, 175.335, HAMILTON, 2.5, "Peacocke growth area, southeast cut, south fill"),
    S("hamilton-northwest", "building", -37.6948, 175.2089, HAMILTON, note="Rotokauri and Te Rapa fills"),
    S("hamilton-frankton", "quiet", -37.790, 175.255, HAMILTON, 1.5, "residential, quiet"),
    S("christchurch-yaldhurst", "mining", -43.500, 172.475, CHCH, 3.0, "Yaldhurst and Pound Road gravel pits"),
    S("lyttelton", "building", -43.600, 172.745, CHCH, note="Te Awaparahi Bay reclamation and Gollans Bay quarry"),
    S("tauranga-tauriko", "building", -37.763, 176.088, BOP, note="Tauriko business estate and The Lakes earthworks"),
    S("papamoa-east", "building", -37.742, 176.277, BOP, note="Papamoa East cut (Wairakei or a quarry)"),
    S("matata-quarry", "mining", -37.894, 176.750, BOP, 1.5, "a quarry near Matata or Awakeri (by its dz)"),
    S("whangarei", "survey", -35.725, 174.320, NORTH, note="Whangarei, subtropical north"),
    S("esk-valley", "storm", -39.405, 176.825, HAWKES, note="Cyclone Gabrielle silt and slips (natural, test only)"),
    S("otago-roxburgh", "survey", -45.450, 169.380, OTAGO, note="Central Otago between Alexandra and Roxburgh, semi-arid"),
    S("palmerston-north", "survey", -40.360, 175.640, PALM, note="Palmerston North, possibly the Te Ahu a Turanga works"),
]

_items = {}


def get(url, tries=5):
    for k in range(tries):
        try:
            return requests.get(url, timeout=60).json()
        except (requests.RequestException, ValueError):
            if k == tries - 1:
                raise
            time.sleep(2 + 3 * k)


def items(survey):
    """(GDAL path, bbox) of every tile in the survey's DEM, from its STAC collection (cached in data/world-raw)."""
    if survey in _items:
        return _items[survey]
    cache = os.path.join(RAW, "nz", survey.replace("/", "__") + ".json")
    if os.path.exists(cache):
        _items[survey] = json.load(open(cache))
        return _items[survey]
    base = f"{BUCKET}/{survey}/dem_1m/2193"
    col = get(f"{base}/collection.json")
    hrefs = [l["href"] for l in col["links"] if l["rel"] == "item"]

    def one(h):
        it = get(f"{base}/{h.lstrip('./')}")
        tif = next(a["href"] for a in it["assets"].values() if a["href"].endswith((".tiff", ".tif")))
        return [f"/vsis3/nz-elevation/{survey}/dem_1m/2193/{tif.lstrip('./')}", it["bbox"]]
    with ThreadPoolExecutor(16) as ex:
        tiles = list(ex.map(one, hrefs))
    iv = col["extent"]["temporal"]["interval"][0]
    out = {"tiles": tiles, "start": iv[0][:10], "end": iv[1][:10]}
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    json.dump(out, open(cache, "w"))
    _items[survey] = out
    return out


def dems(site):
    grid = window(site, CRS_NZ)
    lon0, lat0 = site["lon"], site["lat"]
    pad = 0.06
    got = []
    for survey in site["pair"]:
        it = items(survey)
        # bbox is lon/lat; keep tiles near the point (a window is under 4 km across)
        urls = [u for u, (w, s, e, n) in it["tiles"] if w - pad < lon0 < e + pad and s - pad < lat0 < n + pad]
        a, tr = read_cogs(urls, CRS_NZ, grid)
        got.append((a, (datetime.date.fromisoformat(it["start"]), datetime.date.fromisoformat(it["end"]))))
    (a, d1), (b, d2) = got
    return dict(a=a, b=b, transform=tr, crs=CRS_NZ, first=d1, second=d2,
                source=f"LINZ nz-elevation {site['pair'][0]} / {site['pair'][1]}")
