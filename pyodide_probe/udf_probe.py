# The aggregate in Pyodide with the model as a vectorized Python (Arrow) function inside DuckDB's SQL, the embeddings
# registered through xarray-sql. Three ways to hand a pixel's bands to the function, each checked against numpy:
#   wide    one column per band (128 for AEF's two years, 256 for TESSERA's): DuckDB passes every column as its own
#           array on each call of 2,048 rows, and that alone costs about 17 s per million pixels (WIDE=1 to run it)
#   packed  each pixel's bands packed into one bytes value (numpy's S dtype, a BLOB in DuckDB): one argument
#   lookup  the function gets the pixel's (y, x) and reads the bands from the arrays in memory
# SQL then joins the two embeddings' halves on the pixel, gates and carries the peak, as sql_probe.py's rule A.
import asyncio, json, os, time
import numpy as np
import pyarrow as pa
import xarray as xr
import duckdb
from duckdb.sqltypes import BLOB, FLOAT, INTEGER, TINYINT
import xarray_sql as xql
import both_reader as R

N = int(os.environ.get("N", "1024"))
R0, C0 = 549888, 36688  # TESSERA utm16 row and column of the NW corner (Huntsville, AL)
AR0, AC0 = 6144 + 1024 - N, 3072
YEARS = (2019, 2023)
AEF_PATH = {2019: "2019/16N/xqq5mh8f5wgh4nzlb-0000000000-0000000000.tiff",
            2023: "2023/16N/x4v3mzlodlbp170gg-0000000000-0000000000.tiff"}
M = np.load("model_abroad.npz")
WJ, BJ, W1, B1 = M["both_w"], float(M["both_b"]), M["aef_w"], float(M["aef_b"])
LO, HI = (float(v) for v in M["bounds"])
# AlphaEarth's dequantization as a 256-entry table (aef-loader's trick): sign(q) * (q / 127.5)^2
LUT = (lambda q: np.sign(q) * (q / 127.5) ** 2)(np.arange(256, dtype=np.int16).astype(np.uint8).view(np.int8).astype(np.float32))
AEF_T = duckdb.struct_type({"z": FLOAT, "z1": FLOAT, "plain": FLOAT})
TES_T = duckdb.struct_type({"z": FLOAT, "plain": FLOAT})
out = {"N": N}


def mem():
    try:
        import js, pyodide_js
        return dict(rss_mb=round(js.process.memoryUsage().rss / 1e6), wasm_heap_mb=round(pyodide_js._module.HEAP8.length / 1e6))
    except Exception:
        return {}


def lap(key, t):
    out[key] = round(time.time() - t, 2)
    print(key, out[key], "s", mem(), flush=True)
    return time.time()


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)


def half(b, a, w):
    d = b.shape[1]
    return b @ w[:d] + a @ w[d:2 * d] + (a * b) @ w[2 * d:3 * d] + ((a - b) ** 2) @ w[3 * d:]


def aef_math(q):
    """(n, 128) int8, first year then last -> [joint model's AEF half, AEF-alone logit, plain change]; NaN where
    AlphaEarth is masked."""
    qb, qa = q[:, :64], q[:, 64:]
    b, a = unit(LUT[qb.view(np.uint8)]), unit(LUT[qa.view(np.uint8)])
    bad = (qb[:, 0] == -128) | (qa[:, 0] == -128)
    z, z1, plain = half(b, a, WJ[:256]), half(b, a, W1) + B1, 1 - (a * b).sum(1)
    return pa.StructArray.from_arrays([pa.array(np.where(bad, np.nan, v), pa.float32()) for v in (z, z1, plain)], ["z", "z1", "plain"])


def tes_math(q):
    """(n, 256) int8 -> [joint model's TESSERA half, plain change]; the per-pixel scale cancels in a unit vector."""
    q = q.astype(np.float32)
    b, a = unit(q[:, :128]), unit(q[:, 128:])
    return pa.StructArray.from_arrays([pa.array(half(b, a, WJ[256:]), pa.float32()), pa.array(1 - (a * b).sum(1), pa.float32())], ["z", "plain"])


# ---- packed: one bytes value per pixel ----------------------------------------------------------------------
def pack(*years):
    """(bands, h, w) int8 per year -> (h, w) of bytes, the years' bands end to end, each value + 128. A zero byte
    ends a value on its way into Arrow, so none may occur: TESSERA keeps to -127 .. 127, and AlphaEarth's -128 marks a
    masked pixel (every band), which goes to -127 here and is carried by its own column."""
    v = np.concatenate([np.moveaxis(y, 0, -1) for y in years], -1)
    v = (np.maximum(v, -127).astype(np.int16) + 128).astype(np.uint8)
    return np.ascontiguousarray(v).view(f"S{v.shape[-1]}")[..., 0]


def unpack(arr, d):
    """A BLOB Arrow array of d-byte values -> (n, d) int8."""
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
    off = np.frombuffer(arr.buffers()[1], np.int32)[arr.offset: arr.offset + len(arr) + 1]
    data = np.frombuffer(arr.buffers()[2], np.uint8)[off[0]: off[-1]]
    assert len(data) == len(arr) * d, (len(data), len(arr), d)
    return (data.reshape(len(arr), d).astype(np.int16) - 128).astype(np.int8)


# ---- lookup: the function reads the bands at (y, x) ---------------------------------------------------------
HELD = {}


def at(name, y, x):
    r, c = y.to_numpy() - R0, x.to_numpy() - C0
    return np.concatenate([v[:, r, c].T for v in HELD[name]], 1)


# ---- wide: one column per band ------------------------------------------------------------------------------
def fixed(f, n):
    """f(*cols) as a function of exactly n named arguments: DuckDB counts the parameters it is given."""
    args = ", ".join(f"c{i}" for i in range(n))
    ns = {}
    exec(f"def g({args}):\n    return f({args})", {"f": f}, ns)
    return ns["g"]


def cols_np(*cols):
    return np.stack([c.to_numpy(zero_copy_only=False) for c in cols], 1)


async def read():
    t = time.time()
    aef = await asyncio.gather(*[R.aef_window(f"{R.AEF}/{AEF_PATH[y]}", AR0, AC0, N, N) for y in YEARS])
    t = lap("read_aef_s", t)
    names = ("embeddings", "scales", "s1_asc_obs_count", "s1_desc_obs_count")
    got = await asyncio.gather(*[R.tes_window("utm16", n, y - 2017, R0, C0, N, N) for y in YEARS for n in names])
    lap("read_tes_s", t)
    return aef, [dict(zip(names, got[:4])), dict(zip(names, got[4:]))]


def reference(aef, tes, ok_t, flip):
    """Block peaks in numpy from the same math, strip by strip."""
    za, z1, ap, zt, tp = (np.full((N, N), np.nan, np.float32) for _ in range(5))
    for r in range(0, N, 64):
        h = aef_math(np.concatenate([np.moveaxis(e[:, r:r + 64], 0, -1).reshape(-1, 64) for e in aef], 1))
        za[r:r + 64], z1[r:r + 64], ap[r:r + 64] = (h.field(k).to_numpy(zero_copy_only=False).reshape(-1, N) for k in ("z", "z1", "plain"))
        h = tes_math(np.concatenate([np.moveaxis(t["embeddings"][:, r:r + 64], 0, -1).reshape(-1, 128) for t in tes], 1))
        zt[r:r + 64], tp[r:r + 64] = (h.field(k).to_numpy(zero_copy_only=False).reshape(-1, N) for k in ("z", "plain"))
    ga, gt = np.clip((ap - 0.05) / 0.10, 0, 1), np.clip((tp - LO) / (HI - LO), 0, 1)
    both = 1 / (1 + np.exp(-(za + zt + BJ))) * np.where(flip, ga, np.maximum(ga, gt))
    s = np.where(ok_t & np.isfinite(za), both, 1 / (1 + np.exp(-z1)) * ga)
    by, bx = np.meshgrid((R0 + np.arange(N)) // 32 - R0 // 32, (C0 + np.arange(N)) // 32 - C0 // 32, indexing="ij")
    peak = np.full((by.max() + 1, bx.max() + 1), np.nan, np.float32)
    np.fmax.at(peak, (by.ravel(), bx.ravel()), s.ravel())
    return peak


def sql(aef_call, tes_call, aef="aef", tes="tes"):
    return f"""
WITH a AS (SELECT y, x, {aef_call} AS h FROM {aef} WHERE ok),
     t AS (SELECT y, x, {tes_call} AS h, flip FROM {tes} WHERE ok)
SELECT y // 32 AS by, x // 32 AS bx, max(score) AS peak
FROM (
  SELECT a.y, a.x,
    CASE WHEN t.h IS NULL THEN 1 / (1 + exp(-a.h.z1)) * ga
         ELSE 1 / (1 + exp(-(a.h.z + t.h.z + {BJ})))
              * CASE WHEN t.flip THEN ga ELSE greatest(ga, least(greatest((t.h.plain - {LO}) / {HI - LO}, 0), 1)) END
    END AS score
  FROM (SELECT *, least(greatest((h.plain - 0.05) / 0.10, 0), 1) AS ga FROM a WHERE NOT isnan(h.z)) a
  LEFT JOIN t ON a.y = t.y AND a.x = t.x
)
GROUP BY 1, 2
"""


def check(key, res, peak):
    got = np.full(peak.shape, np.nan, np.float32)
    got[res["by"] - R0 // 32, res["bx"] - C0 // 32] = res["peak"]
    ok = np.isfinite(peak)
    out[f"{key}_max_abs_diff_vs_numpy"] = float(np.nanmax(np.abs(got[ok] - peak[ok])))


async def main():
    aef, tes = await read()
    t = time.time()
    ok_t = np.isfinite(tes[0]["scales"]) & np.isfinite(tes[1]["scales"]) & (tes[0]["scales"] > 0) & (tes[1]["scales"] > 0)
    s1 = [x["s1_asc_obs_count"].astype(int) + x["s1_desc_obs_count"] for x in tes]
    flip = (s1[0] > 0) != (s1[1] > 0)
    coords = dict(y=np.arange(R0, R0 + N, dtype=np.int32), x=np.arange(C0, C0 + N, dtype=np.int32))
    yx = ("y", "x")
    peak = reference(aef, tes, ok_t, flip)
    t = lap("numpy_reference_s", t)
    con = duckdb.connect()

    ok_a = (aef[0][0] != -128) & (aef[1][0] != -128)
    aef_ds = xr.Dataset({"v": (yx, pack(*aef)), "ok": (yx, ok_a)}, coords=coords)
    tes_ds = xr.Dataset({"v": (yx, pack(tes[0]["embeddings"], tes[1]["embeddings"])), "ok": (yx, ok_t), "flip": (yx, flip)}, coords=coords)
    t = lap("pack_s", t)
    con.create_function("aef_packed", lambda v: aef_math(unpack(v, 128)), [BLOB], AEF_T, type="arrow")
    con.create_function("tes_packed", lambda v: tes_math(unpack(v, 256)), [BLOB], TES_T, type="arrow")
    xql.register(con, "aef", aef_ds)
    xql.register(con, "tes", tes_ds)
    check("packed", con.sql(sql("aef_packed(v)", "tes_packed(v)")).fetchnumpy(), peak)
    t = lap("sql_packed_s", t)
    del aef_ds, tes_ds

    HELD["aef"], HELD["tes"] = aef, [x["embeddings"] for x in tes]
    con.create_function("aef_at", lambda y, x: aef_math(at("aef", y, x)), [INTEGER, INTEGER], AEF_T, type="arrow")
    con.create_function("tes_at", lambda y, x: tes_math(at("tes", y, x)), [INTEGER, INTEGER], TES_T, type="arrow")
    xql.register(con, "aef_px", xr.Dataset({"ok": (yx, ok_a)}, coords=coords))
    xql.register(con, "tes_px", xr.Dataset({"ok": (yx, ok_t), "flip": (yx, flip)}, coords=coords))
    t = time.time()
    check("lookup", con.sql(sql("aef_at(y, x)", "tes_at(y, x)", "aef_px", "tes_px")).fetchnumpy(), peak)
    t = lap("sql_lookup_s", t)

    if os.environ.get("WIDE"):
        aef_ds = xr.Dataset({**{f"b{i:02d}": (yx, aef[0][i]) for i in range(64)}, **{f"a{i:02d}": (yx, aef[1][i]) for i in range(64)},
                             "ok": (yx, ok_a)}, coords=coords)
        tes_ds = xr.Dataset({**{f"b{i:03d}": (yx, tes[0]["embeddings"][i]) for i in range(128)},
                             **{f"a{i:03d}": (yx, tes[1]["embeddings"][i]) for i in range(128)},
                             "ok": (yx, ok_t), "flip": (yx, flip)}, coords=coords)
        con.create_function("aef_wide", fixed(lambda *c: aef_math(cols_np(*c)), 128), [TINYINT] * 128, AEF_T, type="arrow")
        con.create_function("tes_wide", fixed(lambda *c: tes_math(cols_np(*c)), 256), [TINYINT] * 256, TES_T, type="arrow")
        xql.register(con, "aef_w", aef_ds)
        xql.register(con, "tes_w", tes_ds)
        t = time.time()
        check("wide", con.sql(sql("aef_wide(*COLUMNS('^[ba]\\d+$'))", "tes_wide(*COLUMNS('^[ba]\\d+$'))", "aef_w", "tes_w")).fetchnumpy(), peak)
        lap("sql_wide_s", t)
    print("RESULT", json.dumps(out))


try:
    await main()
except BaseException:
    import traceback
    traceback.print_exc()
