# The same read through zarr-python's async API, and the decode alone from bytes already fetched.
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


import asyncio
zarr.config.set({"async.concurrency": 64})
out = {}
r0, c0 = 549888, 37316
for yi in (5, 8):
    s = FetchStore("https://data.source.coop/tessera/tessera/zarr/v1.1-dclimate")
    g = zarr.open_group(s, path="utm16", mode="r")
    emb, sc = g["embeddings"], g["scales"]
    t = time.time()
    e, q = await asyncio.gather(emb._async_array.getitem((yi, slice(None), slice(r0, r0+512), slice(c0, c0+512))),
                                sc._async_array.getitem((yi, slice(r0, r0+512), slice(c0, c0+512))))
    t1 = time.time()
    # decode-only timing: same bytes from memory
    mem = {}
    class Mem(FetchStore):
        async def get(self, key, prototype, byte_range=None):
            k = (key, str(byte_range))
            if k not in mem: mem[k] = await FetchStore.get(self, key, prototype, byte_range)
            return mem[k]
    m = Mem("https://data.source.coop/tessera/tessera/zarr/v1.1-dclimate")
    em = zarr.open_group(m, path="utm16", mode="r")["embeddings"]
    await em._async_array.getitem((yi, slice(None), slice(r0, r0+512), slice(c0, c0+512)))
    t2 = time.time()
    await em._async_array.getitem((yi, slice(None), slice(r0, r0+512), slice(c0, c0+512)))
    t3 = time.time()
    x = e.astype(np.float32) * q[None]
    t4 = time.time()
    out[f"y{2017+yi}"] = dict(async_s=round(t1-t, 2), mb=round(s.bytes/1e6,1), gets=s.n, decode_only_s=round(t3-t2,2), dequant_s=round(t4-t3,2), shape=e.shape)
print(json.dumps(out, default=str))
