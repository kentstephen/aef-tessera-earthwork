# earthwork-continued

[Earthwork](https://github.com/kentstephen/aef-earthwork) maps where the ground itself was dug, filled or
graded between two years, from AlphaEarth Foundations (AEF) annual embeddings, with a model taught where
USGS 3DEP lidar flew the same ground twice. This repo asks whether a second embedding,
[TESSERA](https://github.com/ucam-eo/geotessera) v1.1, sees the same moved ground, and how much of its
signal it shares with AEF, and whether one score built from both does better than either.

Both are 10 m and annual, but built differently: AEF from Sentinel-2, Landsat and Sentinel-1 (64 numbers per
pixel), TESSERA from Sentinel-2 and Sentinel-1 time series (128 numbers). TESSERA v1.1 is global for 2017 to
2025 on Source Cooperative as a Zarr, which is what makes it a candidate for the browser map later.

## What it does

- `tessera_sites.py` reads TESSERA under each Earthwork lidar site, for the same two years Earthwork used,
  onto AEF's 10 m grid. It keeps each year's Sentinel-1 and Sentinel-2 observation counts, because a year
  without radar can read as change.
  `abroad` and `places` read the ground Earthwork tested outside the 48 states (Puerto Rico, the US Virgin
  Islands, Reunion, Guadeloupe, Hong Kong, Shizuoka) and nine places with no truth (AEF read here too);
  `space` reads Earthwork's ICESat-2 and GEDI labels, placed by `space_positions.py`.
- `compare.py` compares the two on the lidar's labels (moved more than 0.5 m, unchanged under 0.1 m):
  plain change in each, the Earthwork model taught in each (every digging site held out), how the two
  models agree, and how much of AEF's change vector a linear map from TESSERA's recovers.
- `aggregate.py` compares ways of combining the two into one score. Each rule's cutoff is set so it finds the
  same share of moved ground at the held-out digging sites, then unchanged ground lit up is counted on
  ground never taught: crop switches, farm and marsh lidar, a storm coast, the islands and places abroad.
  `quick.py` tries rules that need no model taught, from the scores `aggregate.py` saves.
- `abroad_teach.py` teaches the three models with the islands' lidar and the satellite-laser labels too
  (fields stay test only) and saves held-out scores for `quick.py` (`SCORES_DIR=data/scores-abroad`).
- `desert.py` asks what TESSERA's change on unchanged ground follows (clear looks, radar, years, AEF).
- `panels.py` draws a site: lidar truth, each model, which of the two calls the ground moved, both, and the
  aggregate (the joint model muted where neither embedding changed).

It reads the Earthwork site cache (`aef-earthwork/data/earthwork`, set `EARTHWORK_DATA` if it lives
elsewhere).

```sh
uv sync
uv run python tessera_sites.py   # about 20 seconds and 230 MB per lidar site
uv run python compare.py         # writes results/compare.json
uv run python tessera_sites.py abroad places
SAVE_SCORES=1 uv run python aggregate.py   # about 20 minutes, writes results/aggregate.json
uv run python quick.py 0.5 0.6 0.7
uv run python panels.py katy-tx brazoria-tx
```

## Data and credits

- **TESSERA** v1.1 embeddings, University of Cambridge with dClimate, read from the Zarr on Source
  Cooperative (`tessera/tessera`).
- **AlphaEarth Foundations Satellite Embedding dataset**, Google and Google DeepMind (CC BY 4.0), through
  the Earthwork cache (Source Cooperative `tge-labs/aef-mosaic`).
- **USGS 3D Elevation Program (3DEP)** 1 m DEMs (public domain), the labels, through the Earthwork cache.
- Abroad labels through the Earthwork cache: IGN France RGE ALTI and LiDAR HD (Licence Ouverte / Etalab 2.0),
  Hong Kong CEDD lidar, VIRTUAL SHIZUOKA (CC BY 4.0), NASA ICESat-2 and GEDI.
