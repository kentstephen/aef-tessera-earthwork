# The aggregate in Pyodide through xarray-sql and DuckDB: both embeddings read on one UTM grid, joined on the
# pixel, gated, and each 32 x 32 block taking its highest score (carry the peak) as a GROUP BY.
# A: numpy scores each embedding's own half of the joint model (it is linear, so the logit splits into an AEF part
#    and a TESSERA part); SQL joins the halves on the pixel, adds them, gates and carries the peak.
# B: the model itself in SQL, from each pixel's unit vectors joined to the weights by band: the weight test.
import asyncio, json, os, time
import numpy as np
import xarray as xr
import duckdb
import xarray_sql as xql
import both_reader as R

N = int(os.environ.get("N", "1024"))  # window side in pixels (1024 = 10.24 km)
B_SIDES = [int(s) for s in os.environ.get("B_SIDES", "128,256").split(",") if s]
R0, C0 = 549888, 36688  # TESSERA utm16 row and column of the NW corner (Huntsville, AL)
AR0, AC0 = 6144 + 1024 - N, 3072  # the same corner in the AEF COG: file rows count up from the south edge
YEARS = (2019, 2023)  # the Huntsville site's two years in Earthwork
AEF_PATH = {2019: "2019/16N/xqq5mh8f5wgh4nzlb-0000000000-0000000000.tiff",
            2023: "2023/16N/x4v3mzlodlbp170gg-0000000000-0000000000.tiff"}
M = np.load("model_abroad.npz")
WJ, BJ, W1, B1 = M["both_w"], float(M["both_b"]), M["aef_w"], float(M["aef_b"])
LO, HI = (float(v) for v in M["bounds"])
CUT = float(M["cutoff"])
out = {"N": N, "km2": round((N * 0.01) ** 2, 1)}


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


def unit_aef(q):  # (64, n) int8 -> (n, 64) unit float32, the dequantization tessera_sites.aef_unit uses
    v = q.astype(np.float32)
    v = np.sign(v) * (v / 127.5) ** 2
    return (v / np.maximum(np.linalg.norm(v, axis=0), 1e-9)).T


def unit_tes(q):  # the per-pixel scale cancels in a unit vector
    v = q.astype(np.float32)
    return (v / np.maximum(np.linalg.norm(v, axis=0), 1e-9)).T


def half(b, a, w):  # one embedding's part of a linear model on [b, a, a * b, (a - b)^2]
    d = b.shape[1]
    return b @ w[:d] + a @ w[d:2 * d] + (a * b) @ w[2 * d:3 * d] + ((a - b) ** 2) @ w[3 * d:]


async def read():
    t = time.time()
    aef = await asyncio.gather(*[R.aef_window(f"{R.AEF}/{AEF_PATH[y]}", AR0, AC0, N, N) for y in YEARS])
    out["aef_mb"], out["aef_gets"] = round(R.stats["bytes"] / 1e6, 1), R.stats["gets"]
    t = lap("read_aef_s", t)
    R.stats.update(gets=0, bytes=0)
    names = ("embeddings", "scales", "s1_asc_obs_count", "s1_desc_obs_count")
    got = await asyncio.gather(*[R.tes_window("utm16", n, y - 2017, R0, C0, N, N) for y in YEARS for n in names])
    tes = [dict(zip(names, got[:4])), dict(zip(names, got[4:]))]
    out["tes_mb"], out["tes_gets"] = round(R.stats["bytes"] / 1e6, 1), R.stats["gets"]
    lap("read_tes_s", t)
    return aef, tes


def prep(aef, tes, n=N, strip=64):
    """Per pixel: each half of the joint model, the AEF-alone logit, both plain changes, radar flip, valid masks;
    and (for B) the unit vectors of the n x n NW corner."""
    f = {k: np.full((N, N), np.nan, np.float32) for k in ("za", "z1", "aplain", "zt", "tplain")}
    aok = (aef[0][0] != -128) & (aef[1][0] != -128)
    tok = np.isfinite(tes[0]["scales"]) & np.isfinite(tes[1]["scales"]) & (tes[0]["scales"] > 0) & (tes[1]["scales"] > 0)
    s1 = [t["s1_asc_obs_count"].astype(int) + t["s1_desc_obs_count"] for t in tes]
    flip = (s1[0] > 0) != (s1[1] > 0)
    units = {}
    for r in range(0, N, strip):
        ab, aa = (unit_aef(e[:, r:r + strip].reshape(64, -1)) for e in aef)
        tb, ta = (unit_tes(t["embeddings"][:, r:r + strip].reshape(128, -1)) for t in tes)
        sl = (slice(r, r + strip), slice(None))
        f["za"][sl] = half(ab, aa, WJ[:256]).reshape(-1, N)
        f["z1"][sl] = (half(ab, aa, W1) + B1).reshape(-1, N)
        f["aplain"][sl] = (1 - (ab * aa).sum(1)).reshape(-1, N)
        f["zt"][sl] = half(tb, ta, WJ[256:]).reshape(-1, N)
        f["tplain"][sl] = (1 - (tb * ta).sum(1)).reshape(-1, N)
        if r < n:
            k = min(strip, n - r)
            for name, v, d in (("ab", ab, 64), ("aa", aa, 64), ("tb", tb, 128), ("ta", ta, 128)):
                units.setdefault(name, []).append(v.reshape(-1, N, d)[:k, :n])
    for k in ("za", "z1", "aplain"):
        f[k][~aok] = np.nan
    for k in ("zt", "tplain"):
        f[k][~tok] = np.nan
    units = {k: np.concatenate(v, 0) for k, v in units.items()}
    return f, flip, aok, tok, units


def reference(f, flip, aok, tok):
    """The same score and block peaks in numpy, to check the SQL."""
    ga = np.clip((f["aplain"] - 0.05) / 0.10, 0, 1)
    gt = np.clip((f["tplain"] - LO) / (HI - LO), 0, 1)
    both = 1 / (1 + np.exp(-(f["za"] + f["zt"] + BJ))) * np.where(flip, ga, np.maximum(ga, gt))
    alone = 1 / (1 + np.exp(-f["z1"])) * ga
    s = np.where(aok & tok, both, np.where(aok, alone, np.nan))
    return s, both, peaks(s)


def peaks(s):
    """Highest score per 32 x 32 block of the zone grid (TESSERA's inner chunks), indexed from the window's first."""
    h, w = s.shape
    by, bx = np.meshgrid((R0 + np.arange(h)) // 32 - R0 // 32, (C0 + np.arange(w)) // 32 - C0 // 32, indexing="ij")
    out = np.full((by.max() + 1, bx.max() + 1), np.nan, np.float32)
    np.fmax.at(out, (by.ravel(), bx.ravel()), s.ravel())
    return out


def from_sql(res, shape):
    got = np.full(shape, np.nan, np.float32)
    got[res["by"] - R0 // 32, res["bx"] - C0 // 32] = res["peak"]
    return got


SQL_A = f"""
SELECT y // 32 AS by, x // 32 AS bx, max(score) AS peak, avg((score > {CUT})::INT) AS lit, count(*) AS n
FROM (
  SELECT a.y, a.x,
    CASE WHEN t.zt IS NULL THEN 1 / (1 + exp(-a.z1)) * a.ga
         ELSE 1 / (1 + exp(-(a.za + t.zt + {BJ})))
              * CASE WHEN t.flip THEN a.ga ELSE greatest(a.ga, least(greatest((t.tplain - {LO}) / {HI - LO}, 0), 1)) END
    END AS score
  FROM (SELECT *, least(greatest((aplain - 0.05) / 0.10, 0), 1) AS ga FROM aef WHERE NOT isnan(za)) a
  LEFT JOIN (SELECT * FROM tes WHERE NOT isnan(zt)) t ON a.y = t.y AND a.x = t.x
)
GROUP BY 1, 2
"""


def sql_b(n):
    """The joint model in SQL from the unit vectors: one row per pixel and band, joined to the weights by band."""
    return f"""
WITH za AS (
  SELECT y, x, sum(b * wb + a * wa + a * b * wab + (a - b) * (a - b) * wd) AS za, 1 - sum(a * b) AS aplain
  FROM aef_cube JOIN w_aef USING (band) GROUP BY y, x
), zt AS (
  SELECT y, x, sum(b * wb + a * wa + a * b * wab + (a - b) * (a - b) * wd) AS zt, 1 - sum(a * b) AS tplain
  FROM tes_cube JOIN w_tes USING (band) GROUP BY y, x
)
SELECT y // 32 AS by, x // 32 AS bx,
  max(1 / (1 + exp(-(za + zt + {BJ}))) * greatest(least(greatest((aplain - 0.05) / 0.10, 0), 1),
                                                   least(greatest((tplain - {LO}) / {HI - LO}, 0), 1))) AS peak
FROM za JOIN zt USING (y, x)
GROUP BY 1, 2
"""


async def main():
    aef, tes = await read()
    t = time.time()
    f, flip, aok, tok, units = prep(aef, tes, max(B_SIDES or [0]))
    del aef, tes
    out["valid"] = dict(aef=round(float(aok.mean()), 4), tes=round(float(tok.mean()), 4), flip=round(float(flip.mean()), 4))
    t = lap("numpy_halves_s", t)
    s, both_s, peak = reference(f, flip, aok, tok)
    t = lap("numpy_score_and_peak_s", t)

    coords = dict(y=np.arange(R0, R0 + N, dtype=np.int32), x=np.arange(C0, C0 + N, dtype=np.int32))
    aef_ds = xr.Dataset({k: (("y", "x"), f[k]) for k in ("za", "z1", "aplain")}, coords=coords)
    tes_ds = xr.Dataset({"zt": (("y", "x"), f["zt"]), "tplain": (("y", "x"), f["tplain"]), "flip": (("y", "x"), flip)},
                        coords=coords)
    con = duckdb.connect()
    xql.register(con, "aef", aef_ds)
    xql.register(con, "tes", tes_ds)
    t = lap("register_s", t)
    n = con.sql("SELECT count(*) FROM aef JOIN tes USING (y, x)").fetchone()[0]
    out["pixels_joined"] = n
    t = lap("sql_join_count_s", t)
    res = con.sql(SQL_A).fetchnumpy()
    t = lap("sql_A_s", t)
    got = from_sql(res, peak.shape)
    ok = np.isfinite(peak)
    out["A_blocks"] = int(len(res["by"]))
    out["A_max_abs_diff_vs_numpy"] = float(np.nanmax(np.abs(got[ok] - peak[ok])))
    out["share_lit"] = round(float(np.nanmean(s > CUT)), 4)
    out["blocks_with_peak_over_cut"] = round(float((peak[ok] > CUT).mean()), 4)
    # the same A again, once DuckDB holds the tables (a pan or a new year pair would register again)
    con.sql("CREATE TABLE aef_m AS SELECT * FROM aef; CREATE TABLE tes_m AS SELECT * FROM tes")
    t = lap("sql_copy_to_duckdb_s", t)
    con.sql(SQL_A.replace("FROM aef ", "FROM aef_m ").replace("FROM tes ", "FROM tes_m ")).fetchnumpy()
    t = lap("sql_A_on_duckdb_tables_s", t)

    for side in B_SIDES:
        t = time.time()
        cube = lambda b, a, d: xr.Dataset({"b": (("y", "x", "band"), units[b][:side, :side]),
                                           "a": (("y", "x", "band"), units[a][:side, :side])},
                                          coords=dict(y=coords["y"][:side], x=coords["x"][:side], band=np.arange(d, dtype=np.int32)))
        xql.register(con, f"aef_cube_{side}", cube("ab", "aa", 64))
        xql.register(con, f"tes_cube_{side}", cube("tb", "ta", 128))
        if side == B_SIDES[0]:
            for name, w, d in (("w_aef", WJ[:256], 64), ("w_tes", WJ[256:], 128)):
                con.sql(f"CREATE TABLE {name} (band INT, wb FLOAT, wa FLOAT, wab FLOAT, wd FLOAT)")
                con.executemany(f"INSERT INTO {name} VALUES (?, ?, ?, ?, ?)",
                                [(i, *(float(w[k * d + i]) for k in range(4))) for i in range(d)])
        res = con.sql(sql_b(side).replace("_cube ", f"_cube_{side} ")).fetchnumpy()
        dt = time.time() - t
        # B has no AEF-alone fallback and no radar flip: its reference is the joint score where both embeddings are
        ref = peaks(np.where(aok & tok & ~flip, both_s, np.nan)[:side, :side])
        got = from_sql(res, ref.shape)
        m = np.isfinite(ref) & np.isfinite(got)
        out[f"B_{side}"] = dict(s=round(dt, 2), rows=side * side * 192, rows_per_s=round(side * side * 192 / dt),
                                blocks=int(m.sum()), max_abs_diff=float(np.abs(got[m] - ref[m]).max()), **mem())
        print(f"B_{side}", out[f"B_{side}"], flush=True)
    print("RESULT", json.dumps(out))


try:
    await main()
except BaseException:
    import traceback
    traceback.print_exc()
