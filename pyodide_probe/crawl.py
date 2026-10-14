# A crawl in Pyodide: tile after tile of ground scored by the aggregate, one tile of raw embeddings in memory at a time.
# Each tile's AlphaEarth and TESSERA bands are decoded straight into pixel-major byte buffers (value + 128, the form
# the SQL function reads), scored through xarray-sql and DuckDB, and reduced to what is kept: a byte per pixel for the
# most recent tiles (CACHE of them) and the highest score of each 32 x 32 block in a DuckDB table for all of them.
# The wasm heap is logged after every tile: it only grows, so a flat line means the crawl fits.
# Knobs: TILE (1024 = one AEF COG tile, or 512), TILES ("r0:r1,c0:c1" in 1024 px COG tiles, rows counted from the
# file's south edge), CACHE (tiles of pixel scores kept), REUSE (1 = buffers allocated once and refilled).
import asyncio, json, os, time
from collections import OrderedDict
import numpy as np
import pyarrow as pa
import xarray as xr
import duckdb
from duckdb.sqltypes import BLOB, FLOAT
import xarray_sql as xql
import both_reader as R

T = int(os.environ.get("TILE", "1024"))
(TR0, TR1), (TC0, TC1) = (tuple(int(v) for v in p.split(":")) for p in os.environ.get("TILES", "5:8,2:5").split(","))
CACHE = int(os.environ.get("CACHE", "16"))
REUSE = os.environ.get("REUSE", "1") == "1"
BATCH = 16  # byte-range requests in flight per array: bounds the compressed bytes held at once
YEARS = (2019, 2023)
AEF_PATH = {2019: "2019/16N/xqq5mh8f5wgh4nzlb-0000000000-0000000000.tiff",
            2023: "2023/16N/x4v3mzlodlbp170gg-0000000000-0000000000.tiff"}
COG_W, COG_S = 500000, 3768320  # the two COGs' west and south edges (UTM 16N)
TES_X0, TES_Y0 = 163840, 9338880  # TESSERA utm16's west and north edges
M = np.load("model_abroad.npz")
WJ, BJ, W1, B1 = M["both_w"], float(M["both_b"]), M["aef_w"], float(M["aef_b"])
LO, HI = (float(v) for v in M["bounds"])
CUT = float(M["cutoff"])
LUT = (lambda q: np.sign(q) * (q / 127.5) ** 2)(np.arange(256, dtype=np.int16).astype(np.uint8).view(np.int8).astype(np.float32))
AEF_T = duckdb.struct_type({"z": FLOAT, "z1": FLOAT, "plain": FLOAT})
TES_T = duckdb.struct_type({"z": FLOAT, "plain": FLOAT})


def heap_mb():
    try:
        import pyodide_js
        return round(pyodide_js._module.HEAP8.length / 1e6)
    except Exception:
        return None


# ---- the model, on packed bytes (udf_probe.py's) ------------------------------------------------------------
def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)


def half(b, a, w):
    d = b.shape[1]
    return b @ w[:d] + a @ w[d:2 * d] + (a * b) @ w[2 * d:3 * d] + ((a - b) ** 2) @ w[3 * d:]


def unpack(arr, d):
    """A BLOB Arrow array of d-byte values (each value + 128) -> (n, d) uint8."""
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
    off = np.frombuffer(arr.buffers()[1], np.int32)[arr.offset: arr.offset + len(arr) + 1]
    return np.frombuffer(arr.buffers()[2], np.uint8)[off[0]: off[-1]].reshape(len(arr), d)


def aef_fn(v):
    q = unpack(v, 128) ^ np.uint8(0x80)  # back to int8's bit pattern, as uint8 for the table lookup
    b, a = unit(LUT[q[:, :64]]), unit(LUT[q[:, 64:]])
    z, z1, plain = half(b, a, WJ[:256]), half(b, a, W1) + B1, 1 - (a * b).sum(1)
    return pa.StructArray.from_arrays([pa.array(v, pa.float32()) for v in (z, z1, plain)], ["z", "z1", "plain"])


def tes_fn(v):
    q = (unpack(v, 256).astype(np.int16) - 128).astype(np.float32)
    b, a = unit(q[:, :128]), unit(q[:, 128:])
    return pa.StructArray.from_arrays([pa.array(half(b, a, WJ[256:]), pa.float32()), pa.array(1 - (a * b).sum(1), pa.float32())], ["z", "plain"])


PIXELS = f"""
CREATE OR REPLACE TEMP TABLE px AS
WITH a AS (SELECT y, x, aef_fn(v) AS h FROM aef WHERE ok),
     t AS (SELECT y, x, tes_fn(v) AS h, flip FROM tes WHERE ok)
SELECT a.y, a.x,
  CASE WHEN t.h IS NULL THEN 1 / (1 + exp(-a.h.z1)) * ga
       ELSE 1 / (1 + exp(-(a.h.z + t.h.z + {BJ})))
            * CASE WHEN t.flip THEN ga ELSE greatest(ga, least(greatest((t.h.plain - {LO}) / {HI - LO}, 0), 1)) END
  END AS score
FROM (SELECT *, least(greatest((h.plain - 0.05) / 0.10, 0), 1) AS ga FROM a) a
LEFT JOIN t ON a.y = t.y AND a.x = t.x
"""
# a block on a tile's edge gets a row from each tile; the map reads the larger
PEAKS = "INSERT INTO peaks SELECT y // 32 AS brow, x // 32 AS bcol, max(score) AS peak FROM px GROUP BY ALL"


# ---- reading straight into the buffers ----------------------------------------------------------------------
IFD = {}


async def groups_in_batches(url, groups):
    for k in range(0, len(groups), BATCH):
        part = groups[k:k + BATCH]
        bufs = await asyncio.gather(*[R.get(url, f"bytes={a}-{b - 1}") for a, b, _ in part])
        for (a, _, items), buf in zip(part, bufs):
            for off, n, key in items:
                yield key, buf[off - a: off - a + n]
        del bufs


async def aef_into(url, r0, c0, dst, slot, ok):
    """File rows r0 .. r0 + T (from the south edge), cols c0 .. c0 + T, north-up, into dst[..., slot:slot + 64]."""
    if url not in IFD:
        IFD[url] = await R.tiff_ifd(url)
    t = IFD[url]
    W, H, TW, TH = int(t[256][0]), int(t[257][0]), int(t[322][0]), int(t[323][0])
    across, per = -(-W // TW), -(-W // TW) * -(-H // TH)
    trs, tcs = range(r0 // TH, (r0 + T - 1) // TH + 1), range(c0 // TW, (c0 + T - 1) // TW + 1)
    want = [(int(t[324][k]), int(t[325][k]), (b, i, j)) for b in range(64) for i in trs for j in tcs
            for k in [b * per + i * across + j]]
    async for (b, i, j), buf in groups_in_batches(url, R.merge(want)):
        tile = np.frombuffer(R.unzstd(buf), np.uint8).reshape(TH, TW)
        fa, fb = max(i * TH, r0), min((i + 1) * TH, r0 + T)
        ca, cb = max(j * TW, c0), min((j + 1) * TW, c0 + T)
        sub = tile[fa - i * TH: fb - i * TH, ca - j * TW: cb - j * TW][::-1]
        rows, cols = slice(r0 + T - fb, r0 + T - fa), slice(ca - c0, cb - c0)
        packed = sub ^ np.uint8(0x80)  # value + 128; AlphaEarth's masked -128 becomes 0
        if b == 0:
            ok[rows, cols] &= packed != 0
        dst[rows, cols, slot + b] = np.maximum(packed, 1)  # no zero byte may reach Arrow


async def tes_into(name, ti, r0, c0, dst, slot):
    """TESSERA's array name, year index ti, zone rows r0 .. r0 + T and cols c0 .. c0 + T, into dst (embeddings:
    dst[..., slot:slot + 128] as value + 128; others: dst[...])."""
    dt, nb, fill = R.ARRAYS[name]
    if not nb:
        dst[...] = fill
    rr, cc = range(r0 // R.IN, (r0 + T - 1) // R.IN + 1), range(c0 // R.IN, (c0 + T - 1) // R.IN + 1)
    shards = sorted({(i // R.NIN, j // R.NIN) for i in rr for j in cc})

    async def shard(sr, sc):
        url = f"{R.TES}/utm16/{name}/c/{ti}{'/0' if nb else ''}/{sr}/{sc}"
        ib = await R.get(url, f"bytes=-{R.NIN * R.NIN * 16 + 4}")
        if ib is None:
            return
        idx = np.frombuffer(ib[:-4], "<u8").reshape(R.NIN, R.NIN, 2)
        want = [(int(idx[i % R.NIN, j % R.NIN, 0]), int(idx[i % R.NIN, j % R.NIN, 1]), (i, j)) for i in rr for j in cc
                if i // R.NIN == sr and j // R.NIN == sc and idx[i % R.NIN, j % R.NIN, 0] != 2**64 - 1]
        bl = R.Blosc()
        async for (i, j), buf in groups_in_batches(url, R.merge(want)):
            blk = np.frombuffer(bl.decode(buf), dt).reshape((nb, R.IN, R.IN) if nb else (R.IN, R.IN))
            ys, xs = i * R.IN - r0, j * R.IN - c0
            a, b, c, d = max(ys, 0), min(ys + R.IN, T), max(xs, 0), min(xs + R.IN, T)
            part = blk[..., a - ys:b - ys, c - xs:d - xs]
            if nb:
                dst[a:b, c:d, slot:slot + nb] = np.moveaxis(part, 0, -1).view(np.uint8) ^ np.uint8(0x80)
            else:
                dst[a:b, c:d] = part

    await asyncio.gather(*[shard(*s) for s in shards])


class Buffers:
    def __init__(self):
        self.aef = np.empty((T, T, 128), np.uint8)
        self.tes = np.empty((T, T, 256), np.uint8)
        self.ok_a = np.empty((T, T), bool)
        self.scales = [np.empty((T, T), np.float32) for _ in YEARS]
        self.s1 = [[np.empty((T, T), np.uint16) for _ in range(2)] for _ in YEARS]


async def read_tile(buf, fr0, fc0):
    """One tile: COG file rows fr0 .. fr0 + T from the south edge, cols fc0 .. fc0 + T."""
    tr0 = (TES_Y0 - (COG_S + (fr0 + T) * 10)) // 10
    tc0 = (COG_W + fc0 * 10 - TES_X0) // 10
    buf.ok_a[...] = True
    jobs = [aef_into(f"{R.AEF}/{AEF_PATH[y]}", fr0, fc0, buf.aef, 64 * k, buf.ok_a) for k, y in enumerate(YEARS)]
    for k, y in enumerate(YEARS):
        jobs.append(tes_into("embeddings", y - 2017, tr0, tc0, buf.tes, 128 * k))
        jobs.append(tes_into("scales", y - 2017, tr0, tc0, buf.scales[k], 0))
        jobs.append(tes_into("s1_asc_obs_count", y - 2017, tr0, tc0, buf.s1[k][0], 0))
        jobs.append(tes_into("s1_desc_obs_count", y - 2017, tr0, tc0, buf.s1[k][1], 0))
    await asyncio.gather(*jobs)
    return tr0, tc0


def score_tile(con, buf, tr0, tc0):
    """Pixel scores of the tile (T, T) as bytes (255 = 1.0, 0 = no score) and the tile's block peaks into peaks."""
    sc = [np.isfinite(s) & (s > 0) for s in buf.scales]
    ok_t = sc[0] & sc[1]
    flip = ((buf.s1[0][0].astype(np.int32) + buf.s1[0][1]) > 0) != ((buf.s1[1][0].astype(np.int32) + buf.s1[1][1]) > 0)
    coords = dict(y=np.arange(tr0, tr0 + T, dtype=np.int32), x=np.arange(tc0, tc0 + T, dtype=np.int32))
    yx = ("y", "x")
    for name in ("aef", "tes"):
        try:
            con.unregister(name)
        except Exception:
            pass
    xql.register(con, "aef", xr.Dataset({"v": (yx, buf.aef.view("S128")[..., 0]), "ok": (yx, buf.ok_a)}, coords=coords))
    xql.register(con, "tes", xr.Dataset({"v": (yx, buf.tes.view("S256")[..., 0]), "ok": (yx, ok_t), "flip": (yx, flip)}, coords=coords))
    con.sql(PIXELS)
    con.sql(PEAKS)
    r = con.sql("SELECT y, x, score FROM px").fetchnumpy()
    px = np.zeros((T, T), np.uint8)
    px[r["y"] - tr0, r["x"] - tc0] = np.clip(np.rint(r["score"] * 254) + 1, 1, 255).astype(np.uint8)
    lit = float((r["score"] > CUT).mean())
    con.sql("DROP TABLE px")
    return px, lit


async def main():
    con = duckdb.connect()
    con.sql("SET memory_limit = '512MB'")
    con.create_function("aef_fn", aef_fn, [BLOB], AEF_T, type="arrow")
    con.create_function("tes_fn", tes_fn, [BLOB], TES_T, type="arrow")
    con.sql("CREATE TABLE peaks (brow INTEGER, bcol INTEGER, peak FLOAT)")
    cache = OrderedDict()
    tiles = [(fr, fc) for fr in range(TR0 * 1024, TR1 * 1024, T) for fc in range(TC0 * 1024, TC1 * 1024, T)]
    log, buf, t_all = [], Buffers() if REUSE else None, time.time()
    print(json.dumps(dict(tile=T, tiles=len(tiles), km2=round(len(tiles) * (T / 100) ** 2), cache=CACHE, reuse=REUSE, heap_mb=heap_mb())), flush=True)

    async def visit(fr, fc):
        nonlocal buf
        if (fr, fc) in cache:
            cache.move_to_end((fr, fc))
            return dict(hit=True)
        if not REUSE:
            buf = Buffers()
        R.stats.update(gets=0, bytes=0, retries=0)
        t = time.time()
        tr0, tc0 = await read_tile(buf, fr, fc)
        t_read = time.time() - t
        px, lit = score_tile(con, buf, tr0, tc0)
        if CACHE:
            cache[(fr, fc)] = px
            while len(cache) > CACHE:
                cache.popitem(last=False)
        return dict(read_s=round(t_read, 1), score_s=round(time.time() - t - t_read, 1), mb=round(R.stats["bytes"] / 1e6),
                    gets=R.stats["gets"], retries=R.stats["retries"], lit=round(lit, 4))

    for k, (fr, fc) in enumerate(tiles):
        rec = dict(n=k, fr=fr, fc=fc, **(await visit(fr, fc)), heap_mb=heap_mb())
        log.append(rec)
        print(json.dumps(rec), flush=True)
    crawl_s = time.time() - t_all
    # pan back: the first tile crawled (the oldest), then the last
    back = []
    for fr, fc in (tiles[0], tiles[-1]):
        t = time.time()
        rec = await visit(fr, fc)
        back.append(dict(fr=fr, fc=fc, s=round(time.time() - t, 2), **rec))
    blocks = con.sql("SELECT count(*) FROM (SELECT brow, bcol FROM peaks GROUP BY ALL)").fetchone()[0]
    t = time.time()
    con.sql("COPY (SELECT brow, bcol, max(peak) AS peak FROM peaks GROUP BY ALL) TO '/tmp/peaks.parquet' (FORMAT parquet)")
    pq_s, pq_bytes = time.time() - t, os.path.getsize("/tmp/peaks.parquet")
    reads = [r for r in log if "read_s" in r]
    out = dict(
        tile=T, tiles=len(tiles), km2=round(len(tiles) * (T / 100) ** 2), cache=CACHE, reuse=REUSE,
        crawl_s=round(crawl_s), s_per_km2=round(crawl_s / (len(tiles) * (T / 100) ** 2), 3),
        mb_per_km2=round(sum(r["mb"] for r in reads) / (len(tiles) * (T / 100) ** 2), 2),
        read_s_mean=round(np.mean([r["read_s"] for r in reads]), 1), score_s_mean=round(np.mean([r["score_s"] for r in reads]), 1),
        heap_after_first_mb=log[0]["heap_mb"], heap_max_mb=max(r["heap_mb"] for r in log), heap_end_mb=heap_mb(),
        cache_mb=round(sum(v.nbytes for v in cache.values()) / 1e6, 1), pan_back=back,
        blocks=blocks, peaks_parquet_kb=round(pq_bytes / 1e3, 1), peaks_parquet_s=round(pq_s, 3),
        lit_mean=round(np.mean([r["lit"] for r in reads]), 4),
    )
    print("RESULT", json.dumps(out), flush=True)


try:
    await main()
except BaseException:
    import traceback
    traceback.print_exc()
