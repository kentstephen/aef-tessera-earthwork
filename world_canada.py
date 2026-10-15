"""Canada: NRCan HRDEM 1 m bare-earth DTMs by lidar project (s3://canelevation-dem, Open Government Licence Canada).

Each lidar acquisition project is one Cloud Optimized GeoTIFF, hrdem-lidar/{PROJECT}-1m-dtm.tif (float32, LZW,
nodata -32767, EPSG:3979 Canada Atlas Lambert for every project), listed in the STAC collection hrdem-lidar at
datacube.services.geo.ca. Heights are CGVD2013 (HRDEM product specification). Files run to many GB but read by
window (about 2 MB of header per open); their overviews line up with the 1 m data. The site grid is the site's
WGS 84 UTM zone, not EPSG:3979: that Lambert grid turns 15 to 30 degrees against north in the east, and the
AlphaEarth box inside a turned square loses up to half its area. read_cogs warps each file onto it.

The STAC footprints of many projects are bounding boxes and overstate coverage, so each pair below was checked
on the projects' -extent.geojson (the real data outline) and then by a 16 m read of both covers.

Dates: STAC gives one datetime per project (apparently the end of acquisition, sometimes a year after the year in
the project's name), no date per tile. A flight's range here is January 1 of the earlier of the name's year and
the datetime's year through the datetime, so AlphaEarth's before year falls before the whole campaign.

Left out: Ottawa's city lidar meets the 2022 Eastern Ontario cover only in rural strips (the quiet window here);
the oil sands pair north of Fort McMurray (Oil_Sands_2017_2 against North_Fort_McMurray 2022) overlaps in strips
that fill 6% of a window; Sudbury (2022, 2023) and Kelowna (2018, 2019) pairs fit no AlphaEarth year between.
"""

import datetime

from rasterio.crs import CRS

from world_sites import read_cogs, window

BUCKET = "/vsicurl/https://canelevation-dem.s3.ca-central-1.amazonaws.com/hrdem-lidar"
D = datetime.date

# project id: (start, end) of its flight, from the STAC datetime and the project name
PROJECTS = {
    "NRCAN-York_2019-1m": (D(2019, 1, 1), D(2019, 9, 11)),
    "NRCAN-HALTON-1m": (D(2018, 1, 1), D(2018, 6, 9)),  # no year in the name
    "ON-Lake_Simcoe_2019-1m": (D(2019, 1, 1), D(2019, 9, 20)),
    "TRCA-GTA_2023-1m": (D(2023, 1, 1), D(2023, 5, 11)),
    "QC-DIF_Grand_Montreal_2018-1m": (D(2018, 1, 1), D(2019, 7, 3)),
    "QC-600017-31_Partie_NO-1m": (D(2017, 7, 1), D(2018, 10, 19)),  # program 600017, may be a 2017 flight
    "QC-600023_52_CMM_2023-1m": (D(2023, 1, 1), D(2023, 12, 2)),
    "QC-600023_52_CMM_2024-1m": (D(2024, 1, 1), D(2024, 5, 8)),
    "ON-SPL_ON_Cochrane_UTM17_2018-1m": (D(2018, 1, 1), D(2019, 8, 4)),
    "ON-Timmins_UTM17_2024-1m": (D(2024, 1, 1), D(2024, 5, 9)),
    "HC-Halifax_County_Inland_2018-1m": (D(2018, 1, 1), D(2018, 5, 30)),
    "NS-NSDNR_2019_3-1m": (D(2019, 1, 1), D(2019, 10, 16)),
    "VILLE_OTTAWA-Ottawa_Gatineau_2020-1m": (D(2020, 1, 1), D(2020, 5, 21)),
    "SNCA-Eastern_Ontario_2022-1m": (D(2022, 1, 1), D(2022, 4, 30)),
    "NRCAN-Fort_McMurray_2018-1m": (D(2018, 1, 1), D(2018, 5, 18)),
    "NRCAN-Athabasca_Clearwater_UTM12_2022-1m": (D(2022, 1, 1), D(2022, 8, 21)),
}
YORK = ("NRCAN-York_2019-1m", "TRCA-GTA_2023-1m")
HALTON = ("NRCAN-HALTON-1m", "TRCA-GTA_2023-1m")
SIMCOE = ("ON-Lake_Simcoe_2019-1m", "TRCA-GTA_2023-1m")
MTL = ("QC-DIF_Grand_Montreal_2018-1m", "QC-600023_52_CMM_2023-1m")
MTL_NE = ("QC-600017-31_Partie_NO-1m", "QC-600023_52_CMM_2023-1m")
MTL_S = ("QC-DIF_Grand_Montreal_2018-1m", "QC-600023_52_CMM_2024-1m")
TIMMINS = ("ON-SPL_ON_Cochrane_UTM17_2018-1m", "ON-Timmins_UTM17_2024-1m")
HALIFAX = ("HC-Halifax_County_Inland_2018-1m", "NS-NSDNR_2019_3-1m")
OTTAWA = ("VILLE_OTTAWA-Ottawa_Gatineau_2020-1m", "SNCA-Eastern_Ontario_2022-1m")
FMM = ("NRCAN-Fort_McMurray_2018-1m", "NRCAN-Athabasca_Clearwater_UTM12_2022-1m")


def S(name, kind, lat, lon, pair, half_km=2.0, note=""):
    return dict(name=f"world-canada-{name}", kind=kind, country="ca", lat=lat, lon=lon, pair=pair, half_km=half_km,
                note=note)


SITES = [
    S("pickering-seaton", "building", 43.892, -79.118, YORK, 2.5, "Seaton new community, north Pickering"),
    S("vaughan-427", "building", 43.805, -79.640, YORK, note="Highway 427 extension and Vaughan employment lands"),
    S("toronto-portlands", "building", 43.648, -79.345, YORK, 1.5,
      "Port Lands flood protection, new Don River mouth; lake and harbor in view"),
    S("milton", "building", 43.525, -79.835, HALTON, note="Milton new subdivisions (Boyne)"),
    S("east-gwillimbury", "building", 44.100, -79.415, SIMCOE, note="East Gwillimbury and Newmarket growth"),
    S("york-quiet", "quiet", 43.972, -79.330, YORK,
      note="farmland north of Stouffville; one pit, and a few fields 0.5 m lower in 2023 (crop in the 2019 flight)"),
    S("montreal-north-pits", "mining", 45.770, -73.925, MTL,
      note="Sainte-Sophie landfill cells (up to +30 m) and sand pits (down to -15 m), Mirabel"),
    S("terrebonne-landfill", "mining", 45.750, -73.540, MTL_NE,
      note="Terrebonne (Lachenaie) landfill rising; borderline: first flight 2017 or 2018"),
    S("montreal-south", "survey", 45.355, -73.635, MTL_S,
      note="Saint-Constant and Delson (by its dz), farmland around"),
    S("timmins-dome", "mining", 48.440, -81.212, TIMMINS,
      note="Porcupine (Dome) gold mine tailings, raised across their whole surface; "
           "the 2018 cover misses the west half"),
    S("halifax-bayers-lake", "building", 44.645, -63.680, HALIFAX, note="Bayers Lake and Clayton Park West growth"),
    S("ottawa-south", "quiet", 45.078, -75.614, OTTAWA, note="farmland south of Ottawa (Osgoode)"),
    S("fort-mcmurray", "quiet", 56.784, -111.142, FMM,
      note="boreal forest and muskeg east of Fort McMurray; partial cover"),
]


def utm(site):
    return CRS.from_epsg(32600 + int((site["lon"] + 180) // 6) + 1)


def dems(site):
    crs = utm(site)
    grid = window(site, crs)
    got = []
    for p in site["pair"]:
        a, tr = read_cogs([f"{BUCKET}/{p}-dtm.tif"], crs, grid)
        got.append(a)
    a, b = got
    return dict(a=a, b=b, transform=tr, crs=crs, first=PROJECTS[site["pair"][0]], second=PROJECTS[site["pair"][1]],
                source=f"NRCan HRDEM {site['pair'][0]} / {site['pair'][1]}")
