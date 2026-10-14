"""What the browser would download: bytes per square kilometer for TESSERA and for AEF, one year.

Each reads the same 5 km squares (a few places, several latitudes) through a store that counts every byte
fetched, chunk index reads included, as a browser reading the Zarr by range requests would. The web map's start
view (zoom 12) reads about 325 MB of AEF for two years; TESSERA's cost is that times the ratio found here.

Run: uv run python bytes_read.py   (writes results/bytes_read.json)
"""

import json
import os
import time

import numpy as np
import zarr
from obstore.store import HTTPStore, S3Store
from pyproj import Transformer
from zarr.storage import ObjectStore, WrapperStore

from tessera_sites import AEF_RES, AEF_X0, AEF_Y0, TESSERA

HERE = os.path.dirname(os.path.abspath(__file__))
PLACES = {"katy-tx": (-95.8, 29.8), "denver-co": (-104.8, 39.75), "lagos-ng": (3.9, 6.45), "stockholm-se": (18.1, 59.35)}
KM = 5
YEAR = 2024


class Counting(WrapperStore):
    """A store that adds up the bytes of every get (on the class: zarr may wrap a copy of the store)."""
    bytes, gets = 0, 0

    async def get(self, key, prototype, byte_range=None):
        buf = await self._store.get(key, prototype, byte_range)
        if buf is not None:
            Counting.bytes += len(buf)
            Counting.gets += 1
        return buf

    async def get_partial_values(self, prototype, key_ranges):
        out = await self._store.get_partial_values(prototype, key_ranges)
        for buf in out:
            if buf is not None:
                Counting.bytes += len(buf)
                Counting.gets += 1
        return out

    async def _get_many(self, requests):
        async for key, buf in self._store._get_many(requests):
            if buf is not None:
                Counting.bytes += len(buf)
                Counting.gets += 1
            yield key, buf

    async def get_ranges(self, key, byte_ranges, *, prototype, **kw):  # coalesced range reads inside a shard
        async for group in self._store.get_ranges(key, byte_ranges, prototype=prototype, **kw):
            for _, buf in group:
                if buf is not None:
                    Counting.bytes += len(buf)
                    Counting.gets += 1
            yield group


def main():
    zarr.config.set({"async.concurrency": 128})
    res = {}
    for name, (lon, lat) in PLACES.items():
        r = {}
        # AEF: the mosaic in degrees
        store = Counting(ObjectStore(S3Store("tge-labs", endpoint="https://data.source.coop", region="us-west-2",
                                             virtual_hosted_style_request=False, skip_signature=True, prefix="aef-mosaic"), read_only=True))
        g = zarr.open_group(store, mode="r")
        ti = int(np.where(g["time"][:] == YEAR)[0][0])
        dy = int(KM * 1000 / 10 / 2)
        dx = int(dy / np.cos(np.radians(lat)))
        row, col = int((AEF_Y0 - lat) / AEF_RES), int((lon - AEF_X0) / AEF_RES)
        Counting.bytes = Counting.gets = 0
        t = time.time()
        a = g["embeddings"][ti, :, row - dy:row + dy, col - dx:col + dx]
        r["aef"] = {"bytes": Counting.bytes, "gets": Counting.gets, "s": round(time.time() - t, 1), "shape": list(a.shape),
                    "chunks": list(g["embeddings"].chunks), "shards": list(g["embeddings"].shards or [])}
        # TESSERA: UTM, 10 m; embeddings, scales and the observation counts the map would need to mask
        store = Counting(ObjectStore(HTTPStore.from_url(TESSERA), read_only=True))
        z = zarr.open_group(store, mode="r", path=f"utm{int((lon + 180) // 6) + 1:02d}")
        X, Y = Transformer.from_crs(4326, z.attrs["proj:code"], always_xy=True).transform(lon, lat)
        _, _, tx0, _, _, ty0 = z.attrs["spatial:transform"]
        c, rr, h = int((X - tx0) / 10), int((ty0 - Y) / 10), int(KM * 100 / 2)
        tix = list(z["time"][:]).index(YEAR)
        for v in ("embeddings", "scales"):
            Counting.bytes = Counting.gets = 0
            t = time.time()
            arr = z[v][tix, :, rr - h:rr + h, c - h:c + h] if v == "embeddings" else z[v][tix, rr - h:rr + h, c - h:c + h]
            r[f"tes_{v}"] = {"bytes": Counting.bytes, "gets": Counting.gets, "s": round(time.time() - t, 1), "shape": list(arr.shape),
                             "chunks": list(z[v].chunks), "shards": list(z[v].shards or [])}
        km2 = KM * KM
        r["mb_per_km2"] = {"aef": round(r["aef"]["bytes"] / km2 / 1e6, 3),
                           "tessera": round((r["tes_embeddings"]["bytes"] + r["tes_scales"]["bytes"]) / km2 / 1e6, 3)}
        r["ratio"] = round(r["mb_per_km2"]["tessera"] / r["mb_per_km2"]["aef"], 2)
        res[name] = r
        print(f"{name:14s} AEF {r['mb_per_km2']['aef']:.3f} MB/km2 ({r['aef']['gets']} gets, {r['aef']['s']} s)  "
              f"TESSERA {r['mb_per_km2']['tessera']:.3f} MB/km2 (embeddings {r['tes_embeddings']['gets']} gets {r['tes_embeddings']['s']} s, "
              f"scales {r['tes_scales']['bytes'] / 1e6:.2f} MB)  ratio {r['ratio']}", flush=True)
    print(f"\nAEF chunks {res['katy-tx']['aef']['chunks']} shards {res['katy-tx']['aef']['shards']}; "
          f"TESSERA chunks {res['katy-tx']['tes_embeddings']['chunks']} shards {res['katy-tx']['tes_embeddings']['shards']}")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, "results", "bytes_read.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
