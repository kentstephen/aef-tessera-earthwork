"""Pictures of the clearing picked near each mine (separate.py near): Sentinel-2 true color the year before the
loss and in 2025, beside Hansen's loss that year, so a square that is itself a mine or a field can be seen and
dropped before it counts as clearing.

Run: uv run python near_panels.py   (writes results/near/<name>.png)
"""

import os

import matplotlib
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.windows import Window

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ground
from lookalikes import scenes
from separate import DIR, NEAR, near_name
from tessera_sites import AEF_RES, AEF_X0, AEF_Y0

HERE = os.path.dirname(os.path.abspath(__file__))
GROUND, INK = "#0E0E10", "#E8E8EA"


def tci(lon, lat, year, half_km=3.0):
    """The least cloudy Sentinel-2 true color over the square in the year's driest-looking scene."""
    d = half_km / 111.0
    bbox = (lon - d / np.cos(np.radians(lat)), lat - d, lon + d / np.cos(np.radians(lat)), lat + d)
    its = scenes(bbox, f"{year}-01-01", f"{year}-12-31", cloud=15, per_month=1)
    if not its:
        return None, None
    it = min(its, key=lambda i: i["properties"]["eo:cloud_cover"])
    epsg = it["properties"]["proj:epsg"]
    x, y = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
    with rasterio.open(it["assets"]["visual"]["href"]) as s:
        tr = s.transform
        c0, r0 = int((x - half_km * 1000 - tr.c) / tr.a), int((y + half_km * 1000 - tr.f) / tr.e)
        n = int(2 * half_km * 1000 / tr.a)
        v = s.read((1, 2, 3), window=Window(c0, r0, n, n), boundless=True, fill_value=0)
    return np.moveaxis(v, 0, -1), it["properties"]["datetime"][:10]


def main():
    os.makedirs(f"{HERE}/results/near", exist_ok=True)
    for mine in NEAR:
        n = near_name(mine)
        if not os.path.exists(f"{DIR}/{n}.npz"):
            continue
        z = np.load(f"{DIR}/{n}.npz")
        lon, lat = z["point"]
        y = int(z["b"][0]) + 1
        r, c = int((AEF_Y0 - lat) / AEF_RES), int((lon - AEF_X0) / AEF_RES)
        H = ground.HALF
        grp = ground.groups("cut", (r - H, r + H, c - H, c + H), y).reshape(2 * H, 2 * H)
        fig, ax = plt.subplots(1, 3, figsize=(15, 5.4), facecolor=GROUND)
        for k, (yy, tag) in enumerate(((y - 1, "before"), (2025, "2025"))):
            im, date = tci(lon, lat, yy)
            if im is not None:
                ax[k].imshow(im)
            ax[k].set_title(f"{tag}: {date}", color=INK)
        ax[2].imshow(np.where(grp == 1, 1.0, np.where(grp == 2, 0.35, 0.0)), cmap="gray", vmin=0, vmax=1)
        ax[2].set_title(f"Hansen loss {y} (light), forest kept (gray)", color=INK)
        for a in ax:
            a.set_xticks([]), a.set_yticks([])
        fig.suptitle(f"{n}: {float(z['km']):.0f} km from {mine}, {lon:.3f}, {lat:.3f}", color=INK)
        fig.savefig(f"{HERE}/results/near/{n}.png", dpi=80, facecolor=GROUND, bbox_inches="tight")
        plt.close(fig)
        print(n, flush=True)


if __name__ == "__main__":
    main()
