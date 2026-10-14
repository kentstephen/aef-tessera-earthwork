# both_reader.py against rasterio (AEF) and zarr-python (TESSERA), same window, CPython.
import asyncio, os, sys, time
import numpy as np, rasterio, zarr
from rasterio.windows import Window
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import both_reader as R

P23 = "2023/16N/x4v3mzlodlbp170gg-0000000000-0000000000.tiff"
R0, C0, N = 549888, 36688, 1024

async def main():
    t = time.time()
    a = await R.aef_window(f"{R.AEF}/{P23}", 6144, 3072, N, N)
    print("aef", a.shape, round(time.time() - t, 1), "s", R.stats)
    with rasterio.open(f"/vsicurl/{R.AEF}/{P23}") as s:
        ref = s.read(window=Window(3072, 6144, N, N))[:, ::-1]
        print("aef transform", s.transform)
    print("aef identical", np.array_equal(a, ref))
    R.stats.update(gets=0, bytes=0); t = time.time()
    e = await R.tes_window("utm16", "embeddings", 6, R0, C0, N, N)
    s = await R.tes_window("utm16", "scales", 6, R0, C0, N, N)
    print("tes", e.shape, round(time.time() - t, 1), "s", R.stats)
    g = zarr.open_group(f"{R.TES}/utm16", mode="r")
    print("tes identical", np.array_equal(e, g["embeddings"][6, :, R0:R0 + N, C0:C0 + N]),
          np.array_equal(s, g["scales"][6, R0:R0 + N, C0:C0 + N], equal_nan=True))
    # same ground: AEF pixel (row from north, col) -> UTM; TESSERA's (R0, C0) -> UTM
    print("aef NW corner", 500000 + 3072 * 10, 3768320 + (6144 + N) * 10, " tes NW corner", 163840 + C0 * 10, 9338880 - R0 * 10)

asyncio.run(main())
