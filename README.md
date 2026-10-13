# earthwork-continued

[Earthwork](https://github.com/kentstephen/aef-earthwork) maps where the ground itself was dug, filled or
graded between two years, from AlphaEarth Foundations (AEF) annual embeddings, with a model taught where
USGS 3DEP lidar flew the same ground twice. This repo asks whether a second embedding,
[TESSERA](https://github.com/ucam-eo/geotessera) v1.1, sees the same moved ground, and how much of its
signal it shares with AEF.

Both are 10 m and annual, but built differently: AEF from Sentinel-2, Landsat and Sentinel-1 (64 numbers per
pixel), TESSERA from Sentinel-2 and Sentinel-1 time series (128 numbers). TESSERA v1.1 is global for 2017 to
2025 on Source Cooperative as a Zarr, which is what makes it a candidate for the browser map later.

## What it does

- `tessera_sites.py` reads TESSERA under each Earthwork lidar site, for the same two years Earthwork used,
  onto AEF's 10 m grid. It keeps each year's Sentinel-1 and Sentinel-2 observation counts, because a year
  without radar can read as change.
- `compare.py` compares the two on the lidar's labels (moved more than 0.5 m, unchanged under 0.1 m):
  plain change in each, the Earthwork model taught in each (every digging site held out), how the two
  models agree, and how much of AEF's change vector a linear map from TESSERA's recovers.

It reads the Earthwork site cache (`aef-earthwork/data/earthwork`, set `EARTHWORK_DATA` if it lives
elsewhere).

```sh
uv sync
uv run python tessera_sites.py   # about 20 seconds and 230 MB per lidar site
uv run python compare.py         # writes results/compare.json
```

## Data and credits

- **TESSERA** v1.1 embeddings, University of Cambridge with dClimate, read from the Zarr on Source
  Cooperative (`tessera/tessera`).
- **AlphaEarth Foundations Satellite Embedding dataset**, Google and Google DeepMind (CC BY 4.0), through
  the Earthwork cache (Source Cooperative `tge-labs/aef-mosaic`).
- **USGS 3D Elevation Program (3DEP)** 1 m DEMs (public domain), the labels, through the Earthwork cache.
