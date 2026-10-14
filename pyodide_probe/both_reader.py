# Both embeddings on one UTM grid, read with ranged GETs (Pyodide's pyfetch, or obstore in CPython):
# AlphaEarth's native COGs (tge-labs/aef: bottom-up, band interleaved, 1024 px zstd tiles) and TESSERA's sharded
# Zarr (32 x 32 inner chunks). Neighboring byte ranges are merged and every request of a read goes out together.
import asyncio
import random
import struct

import numpy as np
from numcodecs import Blosc

try:  # Python 3.14's own zstd: numcodecs' reads GDAL's frames (no content size) as 4 GB on wasm32
    from compression.zstd import decompress as unzstd
except ImportError:
    from numcodecs import Zstd
    unzstd = Zstd().decode

TES = "https://data.source.coop/tessera/tessera/zarr/v1.1-dclimate"
AEF = "https://data.source.coop/tge-labs/aef/v1/annual"
stats = dict(gets=0, bytes=0, retries=0)

# Source Cooperative now and then answers a range request with a 503 and an empty body while the same read succeeds
# a moment later: retry those (and throttling, other 5xx, a failed fetch) with jittered backoff, about 15 s in all.
RETRY = {429, 500, 502, 503, 504}
TRIES = 6


def backoff(k, after=None):
    try:
        return min(float(after), 30.0)
    except (TypeError, ValueError):
        return min(0.5 * 2 ** k, 8.0) * random.uniform(0.5, 1.5)


try:
    from pyodide.http import pyfetch

    async def get(url, rng):
        for k in range(TRIES):
            last = k == TRIES - 1
            try:
                r = await pyfetch(url, headers={"Range": rng})
                b = await r.bytes()
            except Exception as e:  # the fetch itself failed: no status
                if last:
                    raise OSError(f"{e} for {url} {rng} after {TRIES} tries") from e
                stats["retries"] += 1
                await asyncio.sleep(backoff(k))
                continue
            if r.status == 404:
                return None
            if r.status == 206:
                stats["gets"] += 1
                stats["bytes"] += len(b)
                return b
            if r.status not in RETRY or last:
                raise OSError(f"{r.status} for {url} {rng}: {len(b)} bytes, try {k + 1} of {TRIES}")
            stats["retries"] += 1
            await asyncio.sleep(backoff(k, r.headers.get("retry-after")))
except ImportError:  # CPython, to check the reader against rasterio and zarr (obstore retries on its own)
    import obstore as obs
    from obstore.store import HTTPStore

    _stores = {}

    async def get(url, rng):
        base, path = url.rsplit("/", 1)
        s = _stores.setdefault(base, HTTPStore.from_url(base))
        a, b = rng[6:].split("-")
        if a == "":
            r = await obs.get_async(s, path, options={"range": {"suffix": int(b)}})
            out = bytes(await r.bytes_async())
        else:
            out = bytes(await obs.get_range_async(s, path, start=int(a), end=int(b) + 1))
        stats["gets"] += 1
        stats["bytes"] += len(out)
        return out


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


async def fetch_groups(url, groups):
    bufs = await asyncio.gather(*[get(url, f"bytes={a}-{b - 1}") for a, b, _ in groups])
    for (a, _, items), buf in zip(groups, bufs):
        for off, n, key in items:
            yield key, buf[off - a: off - a + n]


# ---- AlphaEarth COG ---------------------------------------------------------------------------------------
_TT = {1: "u1", 2: "u1", 3: "u2", 4: "u4", 5: "u4", 6: "i1", 7: "u1", 8: "i2", 9: "i4", 10: "i4", 11: "f4",
       12: "f8", 16: "u8", 17: "i8", 18: "u8"}


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
        dt = np.dtype(bo + _TT[typ])
        nbytes = dt.itemsize * cnt * (2 if typ in (5, 10) else 1)
        if nbytes <= len(val):
            data = val[:nbytes]
        else:
            o = struct.unpack(bo + ("Q" if big else "I"), val)[0]
            data = head[o:o + nbytes] if o + nbytes <= len(head) else await get(url, f"bytes={o}-{o + nbytes - 1}")
        tags[tag] = np.frombuffer(data, dt)
    return tags


async def aef_window(url, r0, c0, h, w):
    """(64, h, w) int8 north-up, from file rows r0 .. r0 + h counted from the south edge (the COGs are bottom-up)."""
    t = await tiff_ifd(url)
    W, H = int(t[256][0]), int(t[257][0])
    TW, TH, nb = int(t[322][0]), int(t[323][0]), int(t[277][0])
    assert int(t[259][0]) == 50000 and int(t[284][0]) == 2, "expects zstd, band interleaved"
    pred = int(t[317][0]) if 317 in t else 1
    across, per = -(-W // TW), -(-W // TW) * -(-H // TH)
    trs = range(r0 // TH, (r0 + h - 1) // TH + 1)
    tcs = range(c0 // TW, (c0 + w - 1) // TW + 1)
    want = []
    for b in range(nb):
        for i in trs:
            for j in tcs:
                k = b * per + i * across + j
                want.append((int(t[324][k]), int(t[325][k]), (b, i, j)))
    out = np.empty((nb, len(trs) * TH, len(tcs) * TW), np.int8)
    async for (b, i, j), buf in fetch_groups(url, merge(want)):
        tile = np.frombuffer(unzstd(buf), np.int8).reshape(TH, TW)
        if pred == 2:
            tile = np.cumsum(tile, axis=1, dtype=np.int8)
        out[b, (i - trs[0]) * TH:(i - trs[0] + 1) * TH, (j - tcs[0]) * TW:(j - tcs[0] + 1) * TW] = tile
    y, x = r0 - trs[0] * TH, c0 - tcs[0] * TW
    return out[:, y:y + h, x:x + w][:, ::-1]


# ---- TESSERA Zarr -----------------------------------------------------------------------------------------
SH, IN = 4096, 32
NIN = SH // IN
ARRAYS = {  # name: (dtype, bands, fill)
    "embeddings": (np.int8, 128, 0), "scales": (np.float32, 0, np.nan),
    "s1_asc_obs_count": (np.uint16, 0, 0), "s1_desc_obs_count": (np.uint16, 0, 0),
}


async def tes_window(zone, name, ti, r0, c0, h, w):
    """One TESSERA array for year index ti over zone rows r0 .. r0 + h, cols c0 .. c0 + w: (128, h, w) or (h, w)."""
    dt, nb, fill = ARRAYS[name]
    shape = (nb, h, w) if nb else (h, w)
    out = np.full(shape, fill, dt)
    rr, cc = range(r0 // IN, (r0 + h - 1) // IN + 1), range(c0 // IN, (c0 + w - 1) // IN + 1)
    shards = sorted({(i // NIN, j // NIN) for i in rr for j in cc})

    async def shard(sr, sc):
        url = f"{TES}/{zone}/{name}/c/{ti}{'/0' if nb else ''}/{sr}/{sc}"
        ib = await get(url, f"bytes=-{NIN * NIN * 16 + 4}")
        if ib is None:  # no shard: nothing embedded there
            return
        idx = np.frombuffer(ib[:-4], "<u8").reshape(NIN, NIN, 2)
        want = [(int(idx[i % NIN, j % NIN, 0]), int(idx[i % NIN, j % NIN, 1]), (i, j)) for i in rr for j in cc
                if i // NIN == sr and j // NIN == sc and idx[i % NIN, j % NIN, 0] != 2**64 - 1]
        bl = Blosc()
        async for (i, j), buf in fetch_groups(url, merge(want)):
            blk = np.frombuffer(bl.decode(buf), dt).reshape((nb, IN, IN) if nb else (IN, IN))
            ys, xs = i * IN - r0, j * IN - c0  # chunk origin in the window
            a, b = max(ys, 0), min(ys + IN, h)
            c, d = max(xs, 0), min(xs + IN, w)
            out[..., a:b, c:d] = blk[..., a - ys:b - ys, c - xs:d - xs]

    await asyncio.gather(*[shard(*s) for s in shards])
    return out
