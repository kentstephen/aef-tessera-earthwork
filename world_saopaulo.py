"""Brazil, Sao Paulo: the city's lidar of 2017 and 2020 (GeoSampa, Prefeitura de Sao Paulo), ground flown twice.

Both flights cover the municipality (about 1,544 km2) on the same 1:1,000 sheets (about 530 x 580 m, named like
3213-424), indexed by the GeoSampa WFS (geoportal:quadricula_folha_mds for 2017, quadricula_folha_mdt_mds_2020
for 2020). The bare-earth product (MDT) is one ground-only LAZ per sheet, fetched whole from GeoSampa's download
page (it wants a browser User-Agent and serves no ranges) into data/world-raw/saopaulo; the points are gridded
here to the mean height per 2 m cell, small holes (up to 3 cells) filled from the nearest cell. Both are SIRGAS
2000 / UTM 23S (EPSG:31983) with orthometric (Imbituba) heights; the 2017 header carries no CRS, but its
heights match 2020's within decimeters on shared ground. The 2017 points are class 2, the 2020 points class 8
(the product holds ground only either way). License: listed as GPL v3 on the AWS registry entry for the 2017
set; check before redistributing derived labels.

Dates: 2017 was flown May to July 2017 (its GPS time is week seconds, so no day per point): borderline, since
AlphaEarth's before year clips to 2017. 2020 was flown May to August 2020, and its adjusted standard GPS time
dates every point, so the second flight's dates at a site are the first and last day of its sheets' points.
LAZ needs laspy with lazrs, which the project does not carry: gridding runs in a subprocess under
`uv run --with "laspy[lazrs]"` when laspy is missing.
"""

import datetime
import json
import os
import subprocess
import sys
import time
import zipfile

import numpy as np
import requests
from rasterio.crs import CRS
from scipy import ndimage

from world_sites import HERE, RAW, STEP, window

CRS_SP = CRS.from_epsg(31983)
WFS = "https://wfs.geosampa.prefeitura.sp.gov.br/geoserver/geoportal/wfs"
LAYERS = {2017: "geoportal:quadricula_folha_mds", 2020: "geoportal:quadricula_folha_mdt_mds_2020"}
DOWNLOAD = ("https://geosampa.prefeitura.sp.gov.br/PaginasPublicas/downloadArquivo.aspx?orig=DownloadMapaArticulacao"
            "&arq=MDT_{year}%5C{code}.zip&arqTipo=MAPA_ARTICULACAO")
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/126.0 Safari/537.36"}
FLIGHT_2017 = (datetime.date(2017, 5, 1), datetime.date(2017, 7, 31))
GPS_EPOCH = datetime.datetime(1980, 1, 6)
DIR = os.path.join(RAW, "saopaulo")
PAUSE = 4.0  # seconds between downloads, one at a time, to stay under the firewall's burst limit
BACKOFF = 600  # seconds to wait when the firewall answers with its challenge page anyway


def S(name, kind, lat, lon, half_km=1.25, note=""):
    return dict(name=f"world-saopaulo-{name}", kind=kind, country="br", lat=lat, lon=lon, half_km=half_km,
                note=("borderline: first flight 2017. " + note).strip())


SITES = [
    S("ctl-landfill", "mining", -23.6263, -46.4247, 1.25,
      "Central de Tratamento Leste (CTL) landfill, active, beside the closed Sao Joao landfill"),
    S("grajau", "mining", -23.790, -46.688, 1.25, "UVR Grajau construction-waste landfill and the quarry beside it"),
    S("perus", "mining", -23.425, -46.750, 1.25,
      "Perus: Riuma construction-waste landfill, the closed Bandeirantes landfill and the quarries nearby"),
    S("pedreira", "mining", -23.4109, -46.5624, 1.25, "CDR Pedreira landfill (Veolia) at the north edge of the city"),
    S("rodoanel-norte", "building", -23.400, -46.600, 1.25, "Rodoanel Norte works north of the city (stalled 2018)"),
    S("ibirapuera", "quiet", -23.587, -46.657, 1.0, "Ibirapuera park and the dense city around it, quiet"),
    S("parelheiros", "quiet", -23.820, -46.701, 1.25, "Parelheiros, rural south of the city, quiet; ground under its forest and fields is mottled +-0.2 to 0.4 m"),
]


def index(year):
    """[(sheet code, (west, south, east, north))] of every sheet in the year's cover (the WFS, cached)."""
    cache = os.path.join(DIR, f"index-{year}.json")
    if not os.path.exists(cache):
        r = requests.get(WFS, params={"service": "WFS", "version": "2.0.0", "request": "GetFeature",
                                      "typeNames": LAYERS[year], "outputFormat": "application/json",
                                      "propertyName": "cd_quadricula,ge_poligono"}, headers=UA, timeout=300)
        r.raise_for_status()
        out = []
        for f in r.json()["features"]:
            xy = np.array(f["geometry"]["coordinates"][0])
            out.append([f["properties"]["cd_quadricula"], [*xy.min(0).tolist(), *xy.max(0).tolist()]])
        os.makedirs(DIR, exist_ok=True)
        json.dump(out, open(cache, "w"))
    return [(c, (b[0], b[1], b[2], b[3])) for c, b in json.load(open(cache))]


def fetch(year, code):
    """The sheet's MDT zip (downloaded once; None when the server has no such file). GeoSampa's firewall answers
    a burst of downloads with a script challenge page; that stops the build rather than reading as a gap."""
    path = os.path.join(DIR, str(year), f"{code}.zip")
    if os.path.exists(path):
        return path
    if os.path.exists(path + ".missing"):
        return None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    challenged = 0
    for attempt in range(8):
        try:
            time.sleep(PAUSE)
            r = requests.get(DOWNLOAD.format(year=year, code=code), headers=UA, timeout=600)
            if r.status_code == 200 and r.content[:2] == b"PK":
                with open(path + ".part", "wb") as f:
                    f.write(r.content)
                os.replace(path + ".part", path)
                return path
            if r.status_code == 200 and b"TSPD" in r.content:
                # the firewall's challenge: back off and try again later, a few times at most
                challenged += 1
                if challenged > 3:
                    raise RuntimeError("GeoSampa keeps answering with its bot challenge page: wait and rerun")
                print(f"  GeoSampa challenge page; waiting {BACKOFF} s", flush=True)
                time.sleep(BACKOFF)
                continue
            if r.status_code == 200:
                open(path + ".missing", "w").close()  # another page: no file for this sheet
                return None
        except requests.RequestException:
            pass
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"GeoSampa download failed: {year} {code}")


def grid_path(zpath):
    return zpath[:-4] + ".grid.npz"


def grid_laz(zpath):
    """Sum and count of the sheet's ground heights per 2 m cell of the global 2 m lattice (cells by floor(x / 2),
    floor(y / 2)), with the GPS time range; cached beside the zip. Needs laspy with lazrs."""
    import laspy
    with zipfile.ZipFile(zpath) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".laz"))
        with z.open(name) as f:
            las = laspy.read(f)
    x, y, zz = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z, np.float64)
    keep = np.isin(np.asarray(las.classification), (2, 8))
    x, y, zz = x[keep], y[keep], zz[keep]
    ix, iy = np.floor(x / STEP).astype(np.int64), np.floor(y / STEP).astype(np.int64)
    c0, r1 = ix.min(), iy.max()
    w, h = int(ix.max() - c0 + 1), int(r1 - iy.min() + 1)
    flat = (r1 - iy) * w + (ix - c0)
    s = np.bincount(flat, zz, minlength=w * h).reshape(h, w).astype(np.float32)
    n = np.bincount(flat, minlength=w * h).reshape(h, w).astype(np.uint16)
    g = np.asarray(las.gps_time)
    adjusted = bool(las.header.global_encoding.gps_time_type)
    np.savez_compressed(grid_path(zpath), s=s, n=n, c0=c0, r1=r1, adjusted=adjusted,
                        t=[float(g.min()), float(g.max())])


def ensure_grids(zpaths):
    todo = [p for p in zpaths if not os.path.exists(grid_path(p))]
    if not todo:
        return
    try:
        import laspy  # noqa: F401
        import lazrs  # noqa: F401
    except ImportError:
        subprocess.run(["uv", "run", "--with", "laspy[lazrs]", "python", os.path.join(HERE, "world_saopaulo.py"),
                        "--grid", *todo], cwd=HERE, check=True)
        return
    for p in todo:
        grid_laz(p)


def gps_date(t):
    # adjusted standard GPS time: GPS seconds less 1e9 (leap seconds left in: a day is all that is kept)
    return (GPS_EPOCH + datetime.timedelta(seconds=t + 1e9)).date()


def cover(year, grid):
    """The year's 2 m mosaic over the grid: mean height (NaN where no ground) and the 2020 flight days, or None
    when no sheet touches the window."""
    W, S, E, N, w, h = grid
    hit = [c for c, (a, b, cc, d) in index(year) if a < E and cc > W and b < N and d > S]
    if not hit:
        return None, None
    zpaths = [p for p in (fetch(year, c) for c in hit) if p]
    ensure_grids(zpaths)
    tot = np.zeros((h, w), np.float64)
    cnt = np.zeros((h, w), np.float64)
    W0, N0 = int(round(W / STEP)), int(round(N / STEP))  # the window's west column, north row edge on the lattice
    t = []
    for p in zpaths:
        g = np.load(grid_path(p))
        s, n = g["s"], g["n"]
        r0, c0 = int(N0 - 1 - g["r1"]), int(g["c0"] - W0)
        a0, b0 = max(0, -r0), max(0, -c0)
        a1, b1 = min(s.shape[0], h - r0), min(s.shape[1], w - c0)
        if a1 <= a0 or b1 <= b0:
            continue
        tot[r0 + a0:r0 + a1, c0 + b0:c0 + b1] += s[a0:a1, b0:b1]
        cnt[r0 + a0:r0 + a1, c0 + b0:c0 + b1] += n[a0:a1, b0:b1]
        if bool(g["adjusted"]):
            t += list(g["t"])
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(cnt > 0, tot / cnt, np.nan).astype(np.float32)
    # fill holes up to 3 cells from the nearest cell with ground
    miss = np.isnan(z)
    if miss.any() and (~miss).any():
        d, (ri, ci) = ndimage.distance_transform_edt(miss, return_indices=True)
        fill = miss & (d <= 3)
        z[fill] = z[ri[fill], ci[fill]]
    days = (gps_date(min(t)), gps_date(max(t))) if t else None
    return z, days


def dems(site):
    from rasterio.transform import from_origin
    grid = window(site, CRS_SP)
    a, _ = cover(2017, grid)
    b, days = cover(2020, grid)
    if a is None or b is None:
        return None
    second = days or (datetime.date(2020, 5, 1), datetime.date(2020, 8, 31))
    return dict(a=a, b=b, transform=from_origin(grid[0], grid[3], STEP, STEP), crs=CRS_SP, first=FLIGHT_2017,
                second=second, source="GeoSampa MDT lidar 2017 / 2020 ground LAZ gridded to 2 m (2020 days from GPS time)")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--grid"]:
        for p in sys.argv[2:]:
            grid_laz(p)
