import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full", app_title="Earthwork crawl")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Earthwork crawl

    Where ground was dug, filled or graded between two years, scored by the aggregate model (AlphaEarth and TESSERA
    joined pixel by pixel) entirely in this browser: Python runs in WebAssembly (Pyodide), reads both embeddings from
    Source Cooperative with range requests, scores them through xarray-sql and DuckDB, and draws the result. No server.

    The map crawls the ground in view one AlphaEarth COG tile (10.24 km square) at a time, nearest the center first.
    Zoomed out, each 320 m block shows its highest 10 m score; from zoom 12, the 10 m pixels. Block peaks are kept in
    this browser's storage, so a reload keeps what was crawled.

    Keys: **X** the map fills the window or sits in the notebook (Esc too), **F** the browser's full screen,
    **P** pause or resume the crawl, **/** search for a place.
    """)
    return


@app.cell
def _():
    import asyncio
    import io
    import json
    import random
    import struct
    import sys
    import time

    import micropip

    await micropip.install("xarray-sql==0.6.0")

    import anywidget
    import duckdb
    import marimo as mo
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import traitlets
    import xarray as xr
    import xarray_sql as xql
    from compression.zstd import decompress as unzstd  # numcodecs' zstd reads GDAL's frames as 4 GB on wasm32
    from duckdb.sqltypes import BLOB, FLOAT
    from numcodecs import Blosc
    import numcodecs.blosc
    import threading
    # numcodecs' Blosc takes a multiprocessing.Lock on first use, which marimo's WebAssembly mode refuses: hand it a
    # thread lock instead (there is one process and Blosc runs without threads here anyway)
    numcodecs.blosc._MUTEX, numcodecs.blosc._MUTEX_IS_INIT, numcodecs.blosc.use_threads = threading.Lock(), True, False
    from pyodide.http import pyfetch
    from pyproj import Transformer
    return (
        BLOB, Blosc, FLOAT, Transformer, anywidget, asyncio, duckdb, io, json, mo, np, pa, pq, pyfetch, random, struct,
        sys, time, traitlets, unzstd, xql, xr,
    )


@app.cell
def _(io, mo, np, pq, pyfetch):
    YA, YB = 2019, 2023  # the year before and the year after
    T = 1024  # one AlphaEarth COG tile, 10.24 km
    PXZ = 12  # zoom from which the map draws 10 m pixels
    MINZ = 9  # zoom from which the map crawls

    async def public_bytes(name):
        loc = str(mo.notebook_location() / "public" / name)
        if loc.startswith("http"):
            r = await pyfetch(loc)
            return await r.bytes()
        with open(loc, "rb") as f:
            return f.read()

    _t = pq.read_table(io.BytesIO(await public_bytes("aef_cogs.parquet")))
    _both = (_t.column(f"y{YA}").is_valid().to_numpy(zero_copy_only=False)
             & _t.column(f"y{YB}").is_valid().to_numpy(zero_copy_only=False))
    _t = _t.filter(_both)
    COGS = {k: _t.column(k).to_numpy(zero_copy_only=False) for k in _t.column_names}

    _m = np.load(io.BytesIO(await public_bytes("model_abroad.npz")))
    WJ, BJ, W1, B1 = _m["both_w"], float(_m["both_b"]), _m["aef_w"], float(_m["aef_b"])
    LO, HI = (float(v) for v in _m["bounds"])
    CUT = float(_m["cutoff"])
    return B1, BJ, COGS, CUT, HI, LO, MINZ, PXZ, T, W1, WJ, YA, YB


@app.cell
def _(Blosc, T, YA, YB, asyncio, json, np, pyfetch, random, struct, unzstd):
    # ---- range reads from Source Cooperative ------------------------------------------------------------------
    TES = "https://data.source.coop/tessera/tessera/zarr/v1.1-dclimate"
    AEF = "https://data.source.coop/tge-labs/aef/v1/annual"
    BATCH = 16  # byte-range requests in flight per array: bounds the compressed bytes held at once
    stats = dict(gets=0, bytes=0, retries=0)

    # Source Cooperative now and then answers a range request with a 503 and an empty body while the same read
    # succeeds a moment later: retry those (and throttling, other 5xx, a failed fetch) with jittered backoff.
    RETRY = {429, 500, 502, 503, 504}
    TRIES = 6

    def backoff(k, after=None):
        try:
            return min(float(after), 30.0)
        except (TypeError, ValueError):
            return min(0.5 * 2 ** k, 8.0) * random.uniform(0.5, 1.5)

    async def get(url, rng=None):
        for k in range(TRIES):
            last = k == TRIES - 1
            try:
                r = await pyfetch(url, headers={"Range": rng} if rng else {})
                b = await r.bytes()
            except Exception as e:  # the fetch itself failed: no status
                if last:
                    raise OSError(f"{e} for {url} {rng} after {TRIES} tries") from e
                stats["retries"] += 1
                await asyncio.sleep(backoff(k))
                continue
            if r.status == 404:
                return None
            if r.status in (200, 206):
                stats["gets"] += 1
                stats["bytes"] += len(b)
                return b
            if r.status not in RETRY or last:
                raise OSError(f"{r.status} for {url} {rng}: {len(b)} bytes, try {k + 1} of {TRIES}")
            stats["retries"] += 1
            await asyncio.sleep(backoff(k, r.headers.get("retry-after")))

    def merge(want, gap=1 << 16):
        """(offset, length, key) sorted by offset -> groups of near-adjacent ranges [start, end, items]."""
        groups, cur = [], None
        for off, n, key in sorted(want, key=lambda t: t[0]):
            if cur and off - cur[1] <= gap:
                cur[1] = max(cur[1], off + n)
                cur[2].append((off, n, key))
            else:
                cur = [off, off + n, [(off, n, key)]]
                groups.append(cur)
        return groups

    async def groups_in_batches(url, groups):
        for k in range(0, len(groups), BATCH):
            part = groups[k:k + BATCH]
            bufs = await asyncio.gather(*[get(url, f"bytes={a}-{b - 1}") for a, b, _ in part])
            for (a, _, items), buf in zip(part, bufs):
                for off, n, key in items:
                    yield key, buf[off - a: off - a + n]
            del bufs

    # ---- AlphaEarth's COGs: bottom-up, band interleaved, 1024 px zstd tiles --------------------------------------
    TIFF_T = {1: "u1", 2: "u1", 3: "u2", 4: "u4", 5: "u4", 6: "i1", 7: "u1", 8: "i2", 9: "i4", 10: "i4", 11: "f4",
              12: "f8", 16: "u8", 17: "i8", 18: "u8"}
    IFD = {}

    async def tiff_ifd(url, head_n=1 << 16):
        """The first (full resolution) IFD's tags of a classic or Big TIFF."""
        head = await get(url, f"bytes=0-{head_n - 1}")
        bo = "<" if head[:2] == b"II" else ">"
        big = struct.unpack(bo + "H", head[2:4])[0] == 43
        if big:
            off = struct.unpack(bo + "Q", head[8:16])[0]
            n, p, es = struct.unpack(bo + "Q", head[off:off + 8])[0], off + 8, 20
        else:
            off = struct.unpack(bo + "I", head[4:8])[0]
            n, p, es = struct.unpack(bo + "H", head[off:off + 2])[0], off + 2, 12
        tags = {}
        for i in range(n):
            e = head[p + i * es: p + (i + 1) * es]
            tag, typ, cnt = struct.unpack(bo + ("HHQ" if big else "HHI"), e[:12 if big else 8])
            val = e[12:20] if big else e[8:12]
            dt = np.dtype(bo + TIFF_T[typ])
            nbytes = dt.itemsize * cnt * (2 if typ in (5, 10) else 1)
            if nbytes <= len(val):
                data = val[:nbytes]
            else:
                o = struct.unpack(bo + ("Q" if big else "I"), val)[0]
                data = head[o:o + nbytes] if o + nbytes <= len(head) else await get(url, f"bytes={o}-{o + nbytes - 1}")
            tags[tag] = np.frombuffer(data, dt)
        return tags

    async def aef_into(url, r0, c0, dst, slot, ok):
        """File rows r0 .. r0 + T (from the south edge), cols c0 .. c0 + T, north-up, into dst[..., slot:slot + 64]
        as value + 128 (masked -128 becomes 0, which also clears ok)."""
        if url not in IFD:
            IFD[url] = await tiff_ifd(url)
        t = IFD[url]
        W, H, TW, TH = int(t[256][0]), int(t[257][0]), int(t[322][0]), int(t[323][0])
        across, per = -(-W // TW), -(-W // TW) * -(-H // TH)
        trs, tcs = range(r0 // TH, (r0 + T - 1) // TH + 1), range(c0 // TW, (c0 + T - 1) // TW + 1)
        want = [(int(t[324][k]), int(t[325][k]), (b, i, j)) for b in range(64) for i in trs for j in tcs
                for k in [b * per + i * across + j]]
        async for (b, i, j), buf in groups_in_batches(url, merge(want)):
            tile = np.frombuffer(unzstd(buf), np.uint8).reshape(TH, TW)
            fa, fb = max(i * TH, r0), min((i + 1) * TH, r0 + T)
            ca, cb = max(j * TW, c0), min((j + 1) * TW, c0 + T)
            sub = tile[fa - i * TH: fb - i * TH, ca - j * TW: cb - j * TW][::-1]
            rows, cols = slice(r0 + T - fb, r0 + T - fa), slice(ca - c0, cb - c0)
            packed = sub ^ np.uint8(0x80)
            if b == 0:
                ok[rows, cols] &= packed != 0
            dst[rows, cols, slot + b] = np.maximum(packed, 1)  # no zero byte may reach Arrow

    # ---- TESSERA's sharded Zarr, one group per UTM zone (both hemispheres on the northern CRS) ---------------------
    SH, IN = 4096, 32
    NIN = SH // IN
    ARRAYS = {  # name: (dtype, bands, fill)
        "embeddings": (np.int8, 128, 0), "scales": (np.float32, 0, np.nan),
        "s1_asc_obs_count": (np.uint16, 0, 0), "s1_desc_obs_count": (np.uint16, 0, 0),
    }
    ORIGIN = {}

    async def zone_origin(zone):
        """TESSERA's zone group's west and north edges, from its zarr.json."""
        if zone not in ORIGIN:
            meta = json.loads(await get(f"{TES}/utm{zone:02d}/zarr.json"))
            tr = meta["attributes"]["spatial:transform"]
            ORIGIN[zone] = (int(tr[2]), int(tr[5]))
        return ORIGIN[zone]

    async def tes_into(zone, name, ti, r0, c0, dst, slot):
        """TESSERA's array name, year index ti, zone rows r0 .. r0 + T and cols c0 .. c0 + T, into dst (embeddings:
        dst[..., slot:slot + 128] as value + 128; the others: dst[...])."""
        dt, nb, fill = ARRAYS[name]
        if not nb:
            dst[...] = fill
        rr, cc = range(r0 // IN, (r0 + T - 1) // IN + 1), range(c0 // IN, (c0 + T - 1) // IN + 1)
        shards = sorted({(i // NIN, j // NIN) for i in rr for j in cc})

        async def shard(sr, sc):
            url = f"{TES}/utm{zone:02d}/{name}/c/{ti}{'/0' if nb else ''}/{sr}/{sc}"
            ib = await get(url, f"bytes=-{NIN * NIN * 16 + 4}")
            if ib is None:  # no shard: nothing embedded there
                return
            idx = np.frombuffer(ib[:-4], "<u8").reshape(NIN, NIN, 2)
            want = [(int(idx[i % NIN, j % NIN, 0]), int(idx[i % NIN, j % NIN, 1]), (i, j)) for i in rr for j in cc
                    if i // NIN == sr and j // NIN == sc and idx[i % NIN, j % NIN, 0] != 2**64 - 1]
            bl = Blosc()
            async for (i, j), buf in groups_in_batches(url, merge(want)):
                blk = np.frombuffer(bl.decode(buf), dt).reshape((nb, IN, IN) if nb else (IN, IN))
                ys, xs = i * IN - r0, j * IN - c0
                a, b, c, d = max(ys, 0), min(ys + IN, T), max(xs, 0), min(xs + IN, T)
                part = blk[..., a - ys:b - ys, c - xs:d - xs]
                if nb:
                    dst[a:b, c:d, slot:slot + nb] = np.moveaxis(part, 0, -1).view(np.uint8) ^ np.uint8(0x80)
                else:
                    dst[a:b, c:d] = part

        await asyncio.gather(*[shard(*s) for s in shards])

    class Buffers:
        """One tile of raw embeddings, allocated once and refilled: a wasm heap never gives memory back, and fresh
        tile-sized arrays do not always fit the holes the last ones left."""

        def __init__(self):
            self.aef = np.empty((T, T, 128), np.uint8)
            self.tes = np.empty((T, T, 256), np.uint8)
            self.ok_a = np.empty((T, T), bool)
            self.scales = [np.empty((T, T), np.float32) for _ in range(2)]
            self.s1 = [[np.empty((T, T), np.uint16) for _ in range(2)] for _ in range(2)]

    async def read_tile(buf, tile):
        """Both years of both embeddings for one tile (a dict from tiles_in_view) into buf; returns TESSERA's
        (row, col) of the tile's north-west pixel."""
        zone, hemi_s, x0, y0 = tile["zone"], tile["hemi_s"], tile["x0"], tile["y0"]
        tx0, ty0 = await zone_origin(zone)
        north = y0 + T * 10 - (10_000_000 if hemi_s else 0)  # TESSERA keeps the south on the northern CRS
        tr0, tc0 = (ty0 - north) // 10, (x0 - tx0) // 10
        hemi = "S" if hemi_s else "N"
        buf.ok_a[...] = True
        jobs = [aef_into(f"{AEF}/{y}/{zone}{hemi}/{name}.tiff", tile["i"] * T, tile["j"] * T, buf.aef, 64 * k, buf.ok_a)
                for k, (y, name) in enumerate(((YA, tile["a"]), (YB, tile["b"])))]
        for k, y in enumerate((YA, YB)):
            jobs.append(tes_into(zone, "embeddings", y - 2017, tr0, tc0, buf.tes, 128 * k))
            jobs.append(tes_into(zone, "scales", y - 2017, tr0, tc0, buf.scales[k], 0))
            jobs.append(tes_into(zone, "s1_asc_obs_count", y - 2017, tr0, tc0, buf.s1[k][0], 0))
            jobs.append(tes_into(zone, "s1_desc_obs_count", y - 2017, tr0, tc0, buf.s1[k][1], 0))
        tasks = [asyncio.ensure_future(j) for j in jobs]
        try:
            await asyncio.gather(*tasks)
        except BaseException:  # one read failed: stop the others, or they pile up under the next try
            for k in tasks:
                k.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        return tr0, tc0
    return Buffers, read_tile, stats


@app.cell
def _(B1, BJ, BLOB, FLOAT, HI, LO, T, W1, WJ, duckdb, np, pa, xql, xr):
    # ---- the aggregate model inside DuckDB: one packed BLOB per pixel per embedding --------------------------------
    LUT = (lambda q: np.sign(q) * (q / 127.5) ** 2)(
        np.arange(256, dtype=np.int16).astype(np.uint8).view(np.int8).astype(np.float32))

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
        return pa.StructArray.from_arrays(
            [pa.array(half(b, a, WJ[256:]), pa.float32()), pa.array(1 - (a * b).sum(1), pa.float32())], ["z", "plain"])

    # the joint model where both embeddings have the pixel, AlphaEarth's own where TESSERA has none; the "either"
    # gate opens where AlphaEarth or TESSERA changed, AlphaEarth's alone where the radar came or went between years
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

    con = duckdb.connect()
    con.sql("SET memory_limit = '512MB'")
    con.create_function("aef_fn", aef_fn, [BLOB], duckdb.struct_type({"z": FLOAT, "z1": FLOAT, "plain": FLOAT}),
                        type="arrow")
    con.create_function("tes_fn", tes_fn, [BLOB], duckdb.struct_type({"z": FLOAT, "plain": FLOAT}), type="arrow")
    # each 32 x 32 block's highest score, as the map's byte (1 = 0.0 .. 255 = 1.0)
    con.sql("CREATE TABLE IF NOT EXISTS peaks (tile VARCHAR, brow UTINYINT, bcol UTINYINT, peak UTINYINT)")

    def score_tile(buf, key, tr0, tc0):
        """The tile's pixel scores as bytes (T, T), north-up (0 = no score, 1 .. 255 = 0.0 .. 1.0); its block
        peaks go into the peaks table."""
        sc = [np.isfinite(s) & (s > 0) for s in buf.scales]
        flip = (((buf.s1[0][0].astype(np.int32) + buf.s1[0][1]) > 0)
                != ((buf.s1[1][0].astype(np.int32) + buf.s1[1][1]) > 0))
        coords = dict(y=np.arange(tr0, tr0 + T, dtype=np.int32), x=np.arange(tc0, tc0 + T, dtype=np.int32))
        yx = ("y", "x")
        for name in ("aef", "tes"):
            try:
                con.unregister(name)
            except Exception:
                pass
        xql.register(con, "aef", xr.Dataset({"v": (yx, buf.aef.view("S128")[..., 0]), "ok": (yx, buf.ok_a)},
                                            coords=coords))
        xql.register(con, "tes", xr.Dataset({"v": (yx, buf.tes.view("S256")[..., 0]), "ok": (yx, sc[0] & sc[1]),
                                             "flip": (yx, flip)}, coords=coords))
        con.sql(PIXELS)
        con.execute(f"""INSERT INTO peaks SELECT ?, (y - {tr0}) // 32, (x - {tc0}) // 32,
                        CAST(round(max(score) * 254) + 1 AS UTINYINT) FROM px GROUP BY ALL""", [key])
        r = con.sql("SELECT y, x, score FROM px").fetchnumpy()
        con.sql("DROP TABLE px")
        px = np.zeros((T, T), np.uint8)
        px[r["y"] - tr0, r["x"] - tc0] = np.clip(np.rint(r["score"] * 254) + 1, 1, 255).astype(np.uint8)
        return px
    return con, score_tile


@app.cell
def _(CUT, MINZ, PXZ, YA, YB, anywidget, json, mo, traitlets):
    class CrawlMap(anywidget.AnyWidget):
        """The map: MapLibre on a dark basemap. Python sends tiles (a JSON header and a byte payload in one message);
        the map sends its view back and which tiles it holds 10 m pixels for."""
        config = traitlets.Unicode("{}").tag(sync=True)
        tile = traitlets.Bytes(b"").tag(sync=True)
        status = traitlets.Unicode("{}").tag(sync=True)
        view = traitlets.Unicode("").tag(sync=True)
        paused = traitlets.Bool(False).tag(sync=True)

        _css = r"""
        .ew{position:relative;width:100%;height:780px;font:13px/1.4 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#e6e9ec;background:#0e1013}
        .ew.fit{position:fixed;inset:0;height:auto;z-index:2147483000}
        .ew .pane{position:absolute;inset:0;background:#0e1013}
        .ew .map{position:absolute;inset:0}
        .ew .card{position:absolute;top:12px;left:12px;width:300px;padding:12px 14px;border-radius:10px;background:#1a1d21eb;border:1px solid #ffffff1f;box-shadow:0 6px 24px #0008;backdrop-filter:blur(6px)}
        .ew h1{margin:0 0 2px;font-size:15px;font-weight:600;letter-spacing:.01em}
        .ew .sub{color:#9ba5af;font-size:12px;margin-bottom:8px}
        .ew .st{min-height:38px;margin:6px 0 8px;font-variant-numeric:tabular-nums}
        .ew .st b{font-weight:600}
        .ew .muted{color:#9ba5af;font-size:12px}
        .ew .row{display:flex;align-items:center;gap:8px;margin-top:8px}
        .ew .row label{flex:none;color:#9ba5af;font-size:12px;width:52px}
        .ew input[type=range]{flex:1;accent-color:#56b4e9}
        .ew input[type=search]{flex:1;min-width:0;background:#0e1013;border:1px solid #ffffff2e;border-radius:6px;color:#e6e9ec;padding:5px 8px;font:inherit}
        .ew .ramp{height:10px;border-radius:3px;margin-top:8px}
        .ew .ticks{display:flex;justify-content:space-between;color:#9ba5af;font-size:11px;font-variant-numeric:tabular-nums}
        .ew .btns{position:absolute;top:12px;right:12px;display:flex;gap:6px}
        .ew button{background:#1a1d21eb;border:1px solid #ffffff2e;color:#e6e9ec;border-radius:7px;padding:5px 10px;font:inherit;cursor:pointer}
        .ew button:hover{background:#23272c}
        .ew button.on{border-color:#56b4e9;color:#56b4e9}
        .ew kbd{font:11px ui-monospace,monospace;color:#9ba5af;border:1px solid #ffffff2e;border-radius:3px;padding:0 3px;margin-left:4px}
        .ew .maplibregl-ctrl-attrib{background:#1a1d21cc!important;color:#9ba5af}
        .ew .maplibregl-ctrl-attrib a{color:#9ba5af}
        """

        _esm = r"""
        import maplibregl from "https://esm.sh/maplibre-gl@5.24.0";
        const STYLE = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";
        // viridis from its blue a fifth of the way in, so the lowest lit ground still shows on a dark basemap
        const STOPS = ["440154","482878","3e4989","31688e","26828e","1f9e89","35b779","6ece58","b5de2b","fde725"].map((h) => [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16)));
        const LUT = new Uint8Array(256 * 3);
        for (let k = 0; k < 256; k++) {
          const t = (0.2 + 0.8 * k / 255) * (STOPS.length - 1), i = Math.min(Math.floor(t), STOPS.length - 2), f = t - i;
          for (let c = 0; c < 3; c++) LUT[k * 3 + c] = Math.round(STOPS[i][c] + f * (STOPS[i + 1][c] - STOPS[i][c]));
        }
        const rampCss = "linear-gradient(90deg," + [0, 64, 128, 192, 255].map((k) => `rgb(${LUT[k*3]},${LUT[k*3+1]},${LUT[k*3+2]})`).join(",") + ")";

        function u8(v) {
          if (!v) return new Uint8Array(0);
          if (v instanceof Uint8Array) return v;
          if (v instanceof ArrayBuffer) return new Uint8Array(v);
          if (ArrayBuffer.isView(v)) return new Uint8Array(v.buffer, v.byteOffset, v.byteLength);
          if (typeof v === "string") { const s = atob(v), a = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) a[i] = s.charCodeAt(i); return a; }
          return new Uint8Array(0);
        }
        function unpackMsg(b) {
          if (b.length < 4) return [null, null];
          const n = new DataView(b.buffer, b.byteOffset, b.byteLength).getUint32(0, true);
          return [JSON.parse(new TextDecoder().decode(b.subarray(4, 4 + n))), b.subarray(4 + n)];
        }
        const el_ = (tag, cls, txt) => { const e = document.createElement(tag); if (cls) e.className = cls; if (txt != null) e.textContent = txt; return e; };
        const fmt = (n) => n.toLocaleString("en-US");

        export default {
          render({ model, el }) {
            const cfg = JSON.parse(model.get("config") || "{}");
            const css = el_("link"); css.rel = "stylesheet"; css.href = "https://unpkg.com/maplibre-gl@5.24.0/dist/maplibre-gl.css"; el.appendChild(css);
            const root = el_("div", "ew fit"), pane = el_("div", "pane"), mapEl = el_("div", "map");
            pane.appendChild(mapEl); root.appendChild(pane); el.appendChild(root);

            // ---- the card ----------------------------------------------------------------------------
            const card = el_("div", "card");
            card.append(el_("h1", null, "Earthwork crawl"), el_("div", "sub", `dug, filled or graded, ${cfg.ya} to ${cfg.yb}`));
            const st = el_("div", "st"), st2 = el_("div", "muted");
            const cutRow = el_("div", "row"), cutIn = el_("input"), cutVal = el_("span", "muted");
            cutIn.type = "range"; cutIn.min = "0.30"; cutIn.max = "0.95"; cutIn.step = "0.01"; cutIn.value = String(cfg.cut);
            cutIn.title = "scores below the cutoff are not drawn";
            cutRow.append(el_("label", null, "cutoff"), cutIn, cutVal);
            const ramp = el_("div", "ramp"); ramp.style.background = rampCss;
            const ticks = el_("div", "ticks");
            const help = el_("div", "muted", "");
            help.style.marginTop = "8px";
            const findRow = el_("div", "row"), find = el_("input");
            find.type = "search"; find.placeholder = "go to a place  ( / )";
            findRow.append(find);
            card.append(st, st2, cutRow, ramp, ticks, help, findRow);
            pane.appendChild(card);
            const btns = el_("div", "btns");
            const bPause = el_("button"), bFit = el_("button"), bFull = el_("button");
            bPause.innerHTML = "Pause<kbd>P</kbd>"; bFull.innerHTML = "Full screen<kbd>F</kbd>";
            btns.append(bPause, bFit, bFull); pane.appendChild(btns);

            let cut = cfg.cut;
            const drawTicks = () => {
              cutVal.textContent = cut.toFixed(2);
              ticks.replaceChildren(el_("span", null, cut.toFixed(2)), el_("span", null, ((1 + cut) / 2).toFixed(2)), el_("span", null, "1.00"));
            };
            drawTicks();

            // ---- the map -----------------------------------------------------------------------------
            const map = new maplibregl.Map({ container: mapEl, style: STYLE, center: cfg.center, zoom: cfg.zoom, attributionControl: { compact: true } });
            map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "bottom-right");
            const px = new Map();    // tile key -> {id, corners, data}: 10 m scores, the most recent few
            const cogs = new Map();  // cog key -> {id, corners, blocks}: 320 m block peaks, everything crawled
            const outlines = [];     // crawled tiles, faint
            let ready = false, queued = [];
            const MAXPX = cfg.maxpx || 24;
            const pxOn = () => map.getZoom() >= cfg.pxz;

            const colorOf = (v) => {  // a score byte -> LUT index, or -1 below the cutoff
              if (!v) return -1;
              const s = (v - 1) / 254;
              if (s < cut) return -1;
              return Math.min(255, Math.round((s - cut) / Math.max(1e-6, 1 - cut) * 255));
            };
            function paint(n, at, canvas, scale = 1) {
              canvas.width = n; canvas.height = n;
              const ctx = canvas.getContext("2d"), img = ctx.createImageData(n, n), d = img.data;
              for (let p = 0; p < n * n; p++) {
                const k = at(p);
                if (k < 0) continue;
                d[p * 4] = LUT[k * 3]; d[p * 4 + 1] = LUT[k * 3 + 1]; d[p * 4 + 2] = LUT[k * 3 + 2]; d[p * 4 + 3] = 255;
              }
              ctx.putImageData(img, 0, 0);
              if (scale === 1) return canvas.toDataURL("image/png");
              // drawn larger, each value a hard-edged square: MapLibre 5.24's nearest resampling draws transparent
              // pixels black, so the layers smooth, and this keeps the smoothing to each square's rim
              const big = document.createElement("canvas"), bx = big.getContext("2d");
              big.width = big.height = n * scale; bx.imageSmoothingEnabled = false;
              bx.drawImage(canvas, 0, 0, n * scale, n * scale);
              return big.toDataURL("image/png");
            }
            function upsert(id, url, corners, before) {
              const src = map.getSource(id);
              if (src) { src.updateImage({ url, coordinates: corners }); return; }
              map.addSource(id, { type: "image", url, coordinates: corners });
              map.addLayer({ id, type: "raster", source: id, paint: { "raster-fade-duration": 0 } }, before);
            }
            const drop = (id) => { if (map.getLayer(id)) map.removeLayer(id); if (map.getSource(id)) map.removeSource(id); };
            function drawPx(key) {
              const t = px.get(key);
              const url = paint(1024, (p) => colorOf(t.data[p]), document.createElement("canvas"));
              upsert(t.id, url, t.corners, "outline-crawled");
              map.setLayoutProperty(t.id, "visibility", pxOn() ? "visible" : "none");
            }
            function drawCog(ck) {
              const c = cogs.get(ck), hide = pxOn();
              const url = paint(256, (p) => {
                const r = Math.floor(p / 256), q = p % 256;
                if (hide && px.has(`${ck}:${7 - Math.floor(r / 32)}:${Math.floor(q / 32)}`)) return -1;
                return colorOf(c.blocks[p]);
              }, document.createElement("canvas"), 4);
              upsert(c.id, url, c.corners, "floor");
            }
            function addBlocks(h, blocks) {  // blocks: 32 x 32 bytes, rows from the north
              let c = cogs.get(h.cog);
              if (!c) { c = { id: "cog-" + h.cog, corners: h.cog_corners, blocks: new Uint8Array(256 * 256) }; cogs.set(h.cog, c); }
              const r0 = (7 - h.i) * 32, c0 = h.j * 32;
              for (let r = 0; r < 32; r++) for (let q = 0; q < 32; q++) c.blocks[(r0 + r) * 256 + c0 + q] = Math.max(c.blocks[(r0 + r) * 256 + c0 + q], blocks[r * 32 + q]);
              outlines.push({ type: "Feature", properties: {}, geometry: { type: "Polygon", coordinates: [[...h.corners, h.corners[0]]] } });
            }
            function onTile(h, payload) {
              const data = payload.slice();  // the message's buffer is not ours to keep
              const blocks = new Uint8Array(32 * 32);
              for (let r = 0; r < 1024; r++) for (let q = 0; q < 1024; q++) {
                const v = data[r * 1024 + q], b = (r >> 5) * 32 + (q >> 5);
                if (v > blocks[b]) blocks[b] = v;
              }
              addBlocks(h, blocks);
              if (px.has(h.key)) px.delete(h.key);
              px.set(h.key, { id: "px-" + h.key, corners: h.corners, data, cog: h.cog });
              drawPx(h.key);
              while (px.size > MAXPX) {  // the oldest tile's pixels go; its blocks stay
                const [k, t] = px.entries().next().value;
                px.delete(k); drop(t.id);
                if (cogs.has(t.cog)) drawCog(t.cog);
              }
              drawCog(h.cog);
              map.getSource("crawled").setData({ type: "FeatureCollection", features: outlines });
              sendView();
            }
            function onRestore(h, payload) {
              h.tiles.forEach((t, n) => addBlocks(t, payload.subarray(n * 1024, (n + 1) * 1024)));
              for (const ck of cogs.keys()) drawCog(ck);
              map.getSource("crawled").setData({ type: "FeatureCollection", features: outlines });
            }
            function onMsg() {
              const [h, payload] = unpackMsg(u8(model.get("tile")));
              if (!h) return;
              if (!ready) { queued.push([h, payload.slice()]); return; }
              if (h.kind === "tile") onTile(h, payload); else if (h.kind === "restore") onRestore(h, payload);
            }
            model.on("change:tile", onMsg);

            function sendView() {
              const b = map.getBounds(), c = map.getCenter();
              model.set("view", JSON.stringify({ lon0: b.getWest(), lat0: b.getSouth(), lon1: b.getEast(), lat1: b.getNorth(),
                lon: c.lng, lat: c.lat, zoom: map.getZoom(), have: [...px.keys()], t: Date.now() }));
              model.save_changes();
            }

            map.on("load", () => {
              const firstSym = (map.getStyle().layers.find((l) => l.type === "symbol") || {}).id;
              const empty = { type: "FeatureCollection", features: [] };
              map.addSource("floor", { type: "geojson", data: empty });
              map.addLayer({ id: "floor", type: "line", source: "floor" }, firstSym);
              map.addSource("crawled", { type: "geojson", data: empty });
              map.addLayer({ id: "outline-crawled", type: "line", source: "crawled", paint: { "line-color": "#ffffff", "line-opacity": 0.18, "line-width": 0.8 } }, firstSym);
              map.addSource("reading", { type: "geojson", data: empty });
              map.addLayer({ id: "outline-reading", type: "line", source: "reading", paint: { "line-color": "#56b4e9", "line-width": 2, "line-dasharray": [2, 1.5] } }, firstSym);
              ready = true;
              for (const [h, p] of queued) { if (h.kind === "tile") onTile(h, p); else if (h.kind === "restore") onRestore(h, p); }
              queued = [];
              onMsg();
              drawReading();
              sendView();
            });
            let wasPx = null;
            map.on("moveend", () => {
              const on = pxOn();
              if (on !== wasPx) {
                wasPx = on;
                for (const t of px.values()) if (map.getLayer(t.id)) map.setLayoutProperty(t.id, "visibility", on ? "visible" : "none");
                for (const ck of cogs.keys()) drawCog(ck);
              }
              sendView();
              drawHelp();
            });

            // ---- the cutoff ----------------------------------------------------------------------------
            let cutTimer = null;
            cutIn.addEventListener("input", () => {
              cut = parseFloat(cutIn.value); drawTicks();
              clearTimeout(cutTimer);
              cutTimer = setTimeout(() => { for (const ck of cogs.keys()) drawCog(ck); for (const k of px.keys()) drawPx(k); }, 250);
            });

            // ---- status --------------------------------------------------------------------------------
            let S = {}, since = Date.now();
            function drawStatus() {
              const secs = Math.round((Date.now() - since) / 1000);
              const phase = { reading: `Reading the tile outlined in blue: ${secs} s`, scoring: `Scoring the tile outlined in blue: ${secs} s`,
                idle: "Everything in view is crawled.", zoom: `Zoom in to crawl (zoom ${cfg.minz}+).`, paused: "Paused.",
                starting: "Starting Python…", error: "Error: " + (S.error || ""), waiting: `Source Cooperative did not answer; trying again in ${S.wait || 0} s.` }[S.phase] || "";
              st.replaceChildren();
              st.append(el_("b", null, phase));
              if (S.queue != null && (S.phase === "reading" || S.phase === "scoring")) st.append(el_("div", "muted", `${fmt(S.queue)} more tile${S.queue === 1 ? "" : "s"} in view to crawl`));
              const L = S.last;
              st2.textContent = (S.done ? `${fmt(S.done)} tile${S.done === 1 ? "" : "s"} crawled (${fmt(Math.round(S.done * 104.9))} km²)` : "No tiles crawled yet") +
                (L ? ` · last: ${L.read_s} s read, ${L.score_s} s score, ${L.mb} MB, ${(L.lit * 100).toFixed(1)}% lit` + (L.retries ? `, ${L.retries} retries` : "") : "") +
                (S.heap_mb ? ` · Python heap ${fmt(S.heap_mb)} MB` : "");
            }
            function drawHelp() {
              help.textContent = pxOn() ? "Zoom 12+: each 10 m pixel's score." : "Each 320 m block shows its highest 10 m score.";
            }
            drawHelp();
            function drawReading() {
              const src = ready && map.getSource("reading");
              if (src) src.setData(S.corners && (S.phase === "reading" || S.phase === "scoring")
                ? { type: "Feature", properties: {}, geometry: { type: "Polygon", coordinates: [[...S.corners, S.corners[0]]] } }
                : { type: "FeatureCollection", features: [] });
            }
            model.on("change:status", () => {
              const n = JSON.parse(model.get("status") || "{}");
              if (n.phase !== S.phase || n.tile !== S.tile) since = Date.now();
              S = n; drawStatus(); drawReading();
            });
            S = JSON.parse(model.get("status") || "{}"); drawStatus();
            const tick = setInterval(drawStatus, 1000);

            // ---- pause, fit to the window, full screen ----------------------------------------------------
            const setPause = (p) => { model.set("paused", p); model.save_changes(); bPause.classList.toggle("on", p); bPause.innerHTML = (p ? "Resume" : "Pause") + "<kbd>P</kbd>"; };
            bPause.onclick = () => setPause(!model.get("paused"));
            const FIT_CLS = "at-fit-on";
            if (!document.getElementById("at-fit-style")) {
              const s = document.createElement("style"); s.id = "at-fit-style";
              s.textContent = ["notebook-actions-dropdown", "cell-actions-button", "drag-button", "expand-output-button", "fullscreen-output-button", "chrome-sidebar", "chrome-footer", "chrome-controls-top-right", "chrome-controls-bottom-right"].map((t) => "html." + FIT_CLS + " [data-testid='" + t + "']").join(",") + ",html." + FIT_CLS + " div[class*='top-[25vh]']{display:none!important}html." + FIT_CLS + "{overflow:hidden}";
              document.head.appendChild(s);
            }
            let fit = true;
            function sizes() {
              root.classList.toggle("fit", fit);
              document.documentElement.classList.toggle(FIT_CLS, fit);
              bFit.innerHTML = (fit ? "Notebook" : "Fill window") + "<kbd>X</kbd>";
              bFit.title = fit ? "back to the notebook (X or Esc)" : "fill the window (X)";
              setTimeout(() => { try { map.resize(); } catch (e) {} }, 30);
            }
            bFit.onclick = () => { fit = !fit; sizes(); };
            sizes();
            const fsOn = () => !!(document.fullscreenElement || document.webkitFullscreenElement);
            const fullScreen = () => {
              if (fsOn()) { (document.exitFullscreen || document.webkitExitFullscreen).call(document); return; }
              const f = pane.requestFullscreen || pane.webkitRequestFullscreen;
              if (f) { const r = f.call(pane); if (r && r.catch) r.catch(() => {}); }
            };
            const onFs = () => setTimeout(() => { try { map.resize(); } catch (e) {} }, 60);
            document.addEventListener("fullscreenchange", onFs);
            document.addEventListener("webkitfullscreenchange", onFs);
            bFull.onclick = fullScreen;
            window.addEventListener("resize", () => { try { map.resize(); } catch (e) {} });
            new ResizeObserver(() => { try { map.resize(); } catch (e) {} }).observe(mapEl);

            // ---- go to a place (Photon, OpenStreetMap's names) -------------------------------------------
            find.addEventListener("keydown", async (e) => {
              e.stopPropagation();
              if (e.key === "Escape") { find.blur(); return; }
              if (e.key !== "Enter" || !find.value.trim()) return;
              try {
                const r = await fetch("https://photon.komoot.io/api/?limit=1&q=" + encodeURIComponent(find.value.trim()));
                const f = (await r.json()).features[0];
                if (!f) { find.value = ""; find.placeholder = "no match"; return; }
                map.flyTo({ center: f.geometry.coordinates, zoom: Math.max(map.getZoom(), 11), essential: true });
                find.blur();
              } catch (err) { find.placeholder = "search failed"; find.value = ""; }
            });

            const onKey = (e) => {
              const t = e.composedPath()[0];
              if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
              if (e.metaKey || e.ctrlKey || e.altKey) return;
              const k = e.key.toLowerCase();
              if (k === "x") { fit = !fit; sizes(); }
              else if (e.key === "Escape" && fit && !fsOn()) { fit = false; sizes(); }
              else if (k === "f") fullScreen();
              else if (k === "p") setPause(!model.get("paused"));
              else if (e.key === "/") { e.preventDefault(); find.focus(); }
            };
            window.addEventListener("keydown", onKey);
            window._ew = { map, px, cogs, status: () => S };  // for headless checks

            return () => {
              clearInterval(tick);
              window.removeEventListener("keydown", onKey);
              document.removeEventListener("fullscreenchange", onFs);
              document.removeEventListener("webkitfullscreenchange", onFs);
              document.documentElement.classList.remove(FIT_CLS);
              try { map.remove(); } catch (e) {}
            };
          },
        };
        """

    crawl_map = CrawlMap(config=json.dumps(dict(
        center=[-86.6, 34.66], zoom=11, cut=round(CUT, 2), ya=YA, yb=YB, pxz=PXZ, minz=MINZ, maxpx=24)))
    crawl_map.status = json.dumps(dict(phase="starting"))
    mo.ui.anywidget(crawl_map)
    return (crawl_map,)


@app.cell
def _(
    Buffers, COGS, CUT, MINZ, PXZ, T, Transformer, YA, YB, asyncio, con, crawl_map, json, np, read_tile, score_tile,
    stats, sys, time,
):
    # ---- the crawl: the tile nearest the view's center that the map lacks, again and again ------------------------
    from collections import OrderedDict

    SAVE_DIR = "/earthwork"
    SAVE = f"{SAVE_DIR}/peaks_{YA}_{YB}_v1.parquet"
    TR = {}

    def transformers(zone, hemi_s):
        k = (zone, hemi_s)
        if k not in TR:
            epsg = (32700 if hemi_s else 32600) + zone
            TR[k] = (Transformer.from_crs(4326, epsg, always_xy=True), Transformer.from_crs(epsg, 4326, always_xy=True))
        return TR[k]

    def cog_key(n):
        return f"{int(COGS['zone'][n])}{'S' if COGS['hemi_s'][n] else 'N'}:{int(COGS['w'][n])}:{int(COGS['s'][n])}"

    def corners(zone, hemi_s, x0, y0, size):
        """Lon/lat of a UTM square's corners: north-west, north-east, south-east, south-west."""
        inv = transformers(zone, hemi_s)[1]
        lon, lat = inv.transform([x0, x0 + size, x0 + size, x0], [y0 + size, y0 + size, y0, y0])
        return [[round(a, 7), round(b, 7)] for a, b in zip(lon, lat)]

    def tile_of(n, i, j):
        zone, hemi_s, w, s = int(COGS["zone"][n]), bool(COGS["hemi_s"][n]), int(COGS["w"][n]), int(COGS["s"][n])
        ck = cog_key(n)
        return dict(key=f"{ck}:{i}:{j}", cog=ck, i=i, j=j, zone=zone, hemi_s=hemi_s, x0=w + j * T * 10,
                    y0=s + i * T * 10, a=COGS[f"y{YA}"][n], b=COGS[f"y{YB}"][n], n=n)

    def geometry(t):
        n = t["n"]
        zone, hemi_s = t["zone"], t["hemi_s"]
        return dict(key=t["key"], cog=t["cog"], i=t["i"], j=t["j"],
                    corners=corners(zone, hemi_s, t["x0"], t["y0"], T * 10),
                    cog_corners=corners(zone, hemi_s, int(COGS["w"][n]), int(COGS["s"][n]), 81920))

    def tiles_in_view(v):
        """Every tile under the view, nearest its center first."""
        lon0, lat0, lon1, lat1 = max(v["lon0"], -180), max(v["lat0"], -80), min(v["lon1"], 180), min(v["lat1"], 84)
        hit = np.nonzero((COGS["lon1"] > lon0) & (COGS["lon0"] < lon1) & (COGS["lat1"] > lat0) & (COGS["lat0"] < lat1))[0]
        edge = np.linspace(0, 1, 9)
        elon = np.concatenate([lon0 + (lon1 - lon0) * edge, np.full(9, lon1), lon1 - (lon1 - lon0) * edge, np.full(9, lon0)])
        elat = np.concatenate([np.full(9, lat1), lat1 - (lat1 - lat0) * edge, np.full(9, lat0), lat0 + (lat1 - lat0) * edge])
        out = []
        for n in hit:
            zone, hemi_s = int(COGS["zone"][n]), bool(COGS["hemi_s"][n])
            fwd = transformers(zone, hemi_s)[0]
            xs, ys = fwd.transform(elon, elat)
            cx, cy = fwd.transform(v["lon"], v["lat"])
            w, s, size = int(COGS["w"][n]), int(COGS["s"][n]), T * 10
            i0, i1 = max(int((np.min(ys) - s) // size), 0), min(int((np.max(ys) - s) // size), 7)
            j0, j1 = max(int((np.min(xs) - w) // size), 0), min(int((np.max(xs) - w) // size), 7)
            for i in range(i0, i1 + 1):
                for j in range(j0, j1 + 1):
                    d = np.hypot(w + (j + 0.5) * size - cx, s + (i + 0.5) * size - cy)
                    out.append((d, n, i, j))
        out.sort()
        return [tile_of(n, i, j) for _, n, i, j in out]

    def heap_mb():
        try:
            import pyodide_js
            return round(pyodide_js._module.HEAP8.length / 1e6)
        except Exception:
            return None

    def send(header, payload=b""):
        h = json.dumps(header).encode()
        crawl_map.tile = len(h).to_bytes(4, "little") + h + payload

    async def syncfs(populate):
        """Pyodide's IDBFS mount to or from the browser's IndexedDB."""
        import pyodide_js
        from pyodide.ffi import create_once_callable
        fut = asyncio.get_event_loop().create_future()
        pyodide_js.FS.syncfs(populate, create_once_callable(lambda err=None: fut.set_result(err)))
        return await fut

    async def restore():
        """Block peaks saved by an earlier visit, back into the peaks table and onto the map."""
        import os
        import pyodide_js
        os.makedirs(SAVE_DIR, exist_ok=True)
        pyodide_js.FS.mount(pyodide_js.FS.filesystems.IDBFS, {}, SAVE_DIR)
        await syncfs(True)
        if not os.path.exists(SAVE):
            return set()
        con.sql(f"INSERT INTO peaks SELECT * FROM read_parquet('{SAVE}')")
        r = con.sql("SELECT tile, brow, bcol, max(peak) AS peak FROM peaks GROUP BY ALL ORDER BY tile").fetchnumpy()
        keys = sorted(set(r["tile"].tolist()))
        at = {k: n for n, k in enumerate(keys)}
        blocks = np.zeros((len(keys), 32, 32), np.uint8)
        blocks[[at[k] for k in r["tile"]], r["brow"], r["bcol"]] = r["peak"]
        index = {cog_key(n): n for n in range(len(COGS["w"]))}
        tiles = []
        for k in keys:
            ck, i, j = k.rsplit(":", 2)
            tiles.append(geometry(tile_of(index[ck], int(i), int(j))))
        send(dict(kind="restore", tiles=tiles), blocks.tobytes())
        return set(keys)

    async def save():
        if not state["persist"]:
            return
        try:
            con.sql(f"COPY (SELECT tile, brow, bcol, max(peak) AS peak FROM peaks GROUP BY ALL) TO '{SAVE}' (FORMAT parquet)")
            await syncfs(False)
        except Exception as e:  # the map still works; only a reload forgets
            print("save:", e)
            state["persist"] = False

    def status(**kw):
        s = json.dumps(dict(kw, heap_mb=heap_mb(), done=len(crawled), last=state["last"], persist=state["persist"]))
        if s != crawl_map.status:
            crawl_map.status = s

    crawled = set()
    recent = OrderedDict()  # tiles whose pixels were sent, as the map's own most-recent list will have them
    state = dict(last=None, persist=True)
    LIT = int(round(CUT * 254)) + 1  # the model's cutoff as a score byte

    async def crawl():
        buf = None
        while not crawl_map.view:  # the map says where it is once it has loaded
            await asyncio.sleep(0.3)
        try:
            crawled.update(await restore())
        except Exception as e:
            print("restore:", e)
            state["persist"] = False
        while True:
            try:
                v = json.loads(crawl_map.view)
                if crawl_map.paused:
                    status(phase="paused")
                    await asyncio.sleep(0.5)
                    continue
                if v["zoom"] < MINZ:
                    status(phase="zoom")
                    await asyncio.sleep(0.5)
                    continue
                have = set(v.get("have", [])) | set(recent)
                todo = [t for t in tiles_in_view(v)
                        if t["key"] not in (have if v["zoom"] >= PXZ else crawled | have)]
                if not todo:
                    status(phase="idle")
                    await asyncio.sleep(0.5)
                    continue
                t = todo[0]
                g = geometry(t)
                status(phase="reading", tile=t["key"], corners=g["corners"], queue=len(todo) - 1)
                if buf is None:
                    buf = Buffers()
                stats.update(gets=0, bytes=0, retries=0)
                t0 = time.time()
                tr0, tc0 = await read_tile(buf, t)
                t1 = time.time()
                status(phase="scoring", tile=t["key"], corners=g["corners"], queue=len(todo) - 1)
                await asyncio.sleep(0)  # let the status reach the map before the scoring holds the thread
                con.execute("DELETE FROM peaks WHERE tile = ?", [t["key"]])
                px = score_tile(buf, t["key"], tr0, tc0)
                lit = float((px >= LIT).sum() / max((px > 0).sum(), 1))
                state["last"] = dict(read_s=round(t1 - t0, 1), score_s=round(time.time() - t1, 1),
                                     mb=round(stats["bytes"] / 1e6), retries=stats["retries"], lit=round(lit, 4))
                crawled.add(t["key"])
                send(dict(kind="tile", **g), px.tobytes())
                recent[t["key"]] = True
                recent.move_to_end(t["key"])
                while len(recent) > MAXPX:
                    recent.popitem(last=False)
                await save()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                import traceback
                traceback.print_exc()
                status(phase="error", error=str(e)[:200], trace=traceback.format_exc()[-1500:])
                await asyncio.sleep(10)

    MAXPX = json.loads(crawl_map.config)["maxpx"]
    if getattr(sys, "_earthwork_crawl", None):
        sys._earthwork_crawl.cancel()
    sys._earthwork_crawl = asyncio.ensure_future(crawl())
    return


if __name__ == "__main__":
    app.run()
