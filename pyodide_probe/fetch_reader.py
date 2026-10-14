# TESSERA read in Pyodide without zarr-python: the shard index, neighboring byte ranges merged, fetched together.
import time, json, asyncio, numpy as np
from numcodecs import Blosc
from pyodide.http import pyfetch
BASE = "https://data.source.coop/tessera/tessera/zarr/v1.1-dclimate/utm16/embeddings"
SH, IN = 4096, 32
NIN = SH // IN
stats = dict(gets=0, bytes=0)
async def rng(url, a, b):
    r = await pyfetch(url, headers={"Range": f"bytes={a}-{b-1}"})
    buf = await r.bytes(); stats["gets"] += 1; stats["bytes"] += len(buf); return buf
async def read(yi, r0, c0, h, w, gap=1 << 16):
    sr, sc = r0 // SH, c0 // SH  # window inside one shard here
    url = f"{BASE}/c/{yi}/0/{sr}/{sc}"
    r = await pyfetch(url, headers={"Range": f"bytes=-{NIN*NIN*16+4}"})
    idx = np.frombuffer((await r.bytes())[:-4], dtype="<u8").reshape(NIN, NIN, 2)
    ir = range((r0 % SH) // IN, (r0 % SH + h - 1) // IN + 1)
    ic = range((c0 % SH) // IN, (c0 % SH + w - 1) // IN + 1)
    want = sorted((int(idx[i, j, 0]), int(idx[i, j, 1]), i, j) for i in ir for j in ic if idx[i, j, 0] != 2**64 - 1)
    groups, cur = [], None  # coalesce near-adjacent byte ranges
    for off, n, i, j in want:
        if cur and off - cur[1] <= gap: cur[1] = max(cur[1], off + n); cur[2].append((off, n, i, j))
        else:
            cur = [off, off + n, [(off, n, i, j)]]; groups.append(cur)
    bufs = await asyncio.gather(*[rng(url, a, b) for a, b, _ in groups])
    t = time.time()
    out = np.zeros((128, len(ir) * IN, len(ic) * IN), np.int8); bl = Blosc()
    for (a, b, items), buf in zip(groups, bufs):
        for off, n, i, j in items:
            blk = np.frombuffer(bl.decode(buf[off - a: off - a + n]), np.int8).reshape(128, IN, IN)
            out[:, (i - ir[0]) * IN:(i - ir[0] + 1) * IN, (j - ic[0]) * IN:(j - ic[0] + 1) * IN] = blk
    return out, len(groups), time.time() - t
res = {}
for yi in (5, 8):
    stats.update(gets=0, bytes=0); t = time.time()
    e, ng, dec = await read(yi, 549888, 37316, 512, 512)
    res[f"y{2017+yi}"] = dict(s=round(time.time() - t, 2), requests=stats["gets"] + 1, mb=round(stats["bytes"] / 1e6, 1), decode_s=round(dec, 2), shape=e.shape)
print(json.dumps(res))
