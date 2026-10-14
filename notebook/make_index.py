"""The AlphaEarth COG index the notebook looks tiles up in: one row per 81.92 km COG footprint, with each year's file
file name. Built once from tge-labs/aef's own index (data/aef_index.parquet, 78 MB) into notebook/public/aef_cogs.parquet.

Run: uv run python notebook/make_index.py
"""
import pathlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

HERE = pathlib.Path(__file__).parent
t = pq.read_table(HERE.parent / "data/aef_index.parquet", columns=[
    "path", "year", "utm_zone", "utm_west", "utm_south", "wgs84_west", "wgs84_south", "wgs84_east", "wgs84_north"]).to_pandas()
name = t.path.str.rsplit("/", n=1).str[-1]
assert name.str.endswith(".tiff").all() and (t.utm_west % 81920 == 8480).all()
t["id"] = name.str[:-5]  # file name without .tiff
t["zone"] = t.utm_zone.str[:-1].astype(np.int8)
t["south"] = t.utm_zone.str[-1] == "S"
wide = t.pivot_table(index=["zone", "south", "utm_west", "utm_south"], columns="year", values="id", aggfunc="first")
box = t.groupby(["zone", "south", "utm_west", "utm_south"])[["wgs84_west", "wgs84_south", "wgs84_east", "wgs84_north"]].first()
wide = wide.join(box).reset_index()
cols = {
    "zone": pa.array(wide.zone, pa.int8()), "hemi_s": pa.array(wide.south),
    "w": pa.array(wide.utm_west, pa.int32()), "s": pa.array(wide.utm_south, pa.int32()),
    **{k: pa.array(wide[f"wgs84_{v}"], pa.float32()) for k, v in (("lon0", "west"), ("lat0", "south"), ("lon1", "east"), ("lat1", "north"))},
    **{f"y{y}": pa.array(wide[y].where(wide[y].notna(), None), pa.string()) for y in range(2017, 2026)},
}
out = HERE / "public/aef_cogs.parquet"
pq.write_table(pa.table(cols), out, compression="zstd")
print(len(wide), "footprints,", round(out.stat().st_size / 1e6, 2), "MB")
