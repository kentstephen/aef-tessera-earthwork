# fetch_reader.py against zarr-python on the same 64 x 64 window: the bytes must match.
import time, json, numpy as np, zarr
from zarr.abc.store import Store, RangeByteRequest, OffsetByteRequest, SuffixByteRequest
from pyodide.http import pyfetch

class FetchStore(Store):
    supports_writes = False; supports_deletes = False; supports_partial_writes = False; supports_listing = False
    def __init__(self, url):
        super().__init__(read_only=True); self.url = url.rstrip("/"); self.n = 0; self.bytes = 0
    def __eq__(self, o): return isinstance(o, FetchStore) and o.url == self.url
    async def get(self, key, prototype, byte_range=None):
        h = {}
        if isinstance(byte_range, RangeByteRequest): h["Range"] = f"bytes={byte_range.start}-{byte_range.end-1}"
        elif isinstance(byte_range, OffsetByteRequest): h["Range"] = f"bytes={byte_range.offset}-"
        elif isinstance(byte_range, SuffixByteRequest): h["Range"] = f"bytes=-{byte_range.suffix}"
        r = await pyfetch(f"{self.url}/{key}", headers=h)
        if r.status == 404: return None
        b = await r.bytes(); self.n += 1; self.bytes += len(b)
        return prototype.buffer.from_bytes(b)
    async def get_partial_values(self, prototype, key_ranges):
        import asyncio
        return await asyncio.gather(*[self.get(k, prototype, br) for k, br in key_ranges])
    async def exists(self, key): return (await self.get(key, zarr.core.buffer.default_buffer_prototype())) is not None
    async def set(self, k, v): raise NotImplementedError
    async def delete(self, k): raise NotImplementedError
    async def set_partial_values(self, kv): raise NotImplementedError
    def list(self): raise NotImplementedError
    def list_prefix(self, p): raise NotImplementedError
    def list_dir(self, p): raise NotImplementedError

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

s = FetchStore("https://data.source.coop/tessera/tessera/zarr/v1.1-dclimate")
emb = zarr.open_group(s, path="utm16", mode="r")["embeddings"]
r0, c0 = 549888, 37316
ref = await emb._async_array.getitem((8, slice(None), slice(r0, r0+64), slice(c0, c0+64)))
mine, _, _ = await read(8, r0, c0, 64, 64)
off = c0 % 32
print("identical:", bool((mine[:, :64, off:off+64] == ref).all()), ref.shape)
