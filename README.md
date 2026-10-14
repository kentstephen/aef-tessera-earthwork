# aef-tessera-earthwork

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
- `checks.py` tests the gains across sites rather than seeds (bootstrap over the digging sites, paired by
  site), sets each site's cutoff on the others, and counts the digging on farm and marsh lidar each rule finds.
- `models.py` keeps both recipes' final models and each rule's cutoff (`data/models`), for scoring new ground.
- `ground.py` reads new ground nobody dug, picked from Hansen's Global Forest Change: forest burned to bare
  ground, clearcuts, drawn-down reservoirs; `fire_teach.py` teaches the burned and cut ground as unchanged.
- `years.py` reads quiet natural ground for every year 2017 to 2025 and scores every pair of years: the
  Sentinel-1 gap of 2022 to 2024 and the span between the years.
- `coverage.py` samples where in the world TESSERA leaves pixels out; `bytes_read.py` counts what a browser
  would download per square kilometer of each embedding.
- `coarse.py` scores cells of about 320 m and 640 m: the highest 10 m score in the cell against models taught
  on pooled embeddings (AEF's overviews among them).
- `fire_teach.py` knobs: `CUT_PX`, `CUT_W` and `CUT_FAR` teach less of the clearcuts, `BASE=us` teaches burns on the
  US recipe, `LEVEL=1` or `2` teaches the farm lidar's leveled ground as digging; `teach_runs.py` puts the saved
  runs side by side.
- `forest_labels.py` (in Earthwork's environment) builds ICESat-2 and GEDI truth at more mines in forest and at
  mines where TESSERA's radar came and went, and repeat 3DEP lidar over eastern Kentucky's surface coal;
  `forest_mines.py` reads TESSERA under it, scores it and teaches with it.
- `lookalikes.py` reads ground that moved with no one digging (landslides, river banks, new solar arrays, flooded
  pits and valleys, flood silt) and splits the reservoirs by a per-year Sentinel-2 water mask.
- `spans.py` scores the digging sites at wider pairs of years than their lidar window; `calib.py` tries cutoffs
  set from each place's own scores.
- `separate.py` reads every year of both embeddings, with each pixel's surroundings, under forest that was cleared
  and dug, cleared and not dug (Hansen loss split by the lidar), the clearcut regions, the satellite-laser mines and
  clearing picked near each mine; `separate_test.py` teaches small heads to tell the two kinds of clearing apart,
  held out by place; `near_panels.py` draws the clearing near each mine against Sentinel-2.
- `pyodide_probe/` reads TESSERA inside Pyodide (Node), through zarr-python and through a small fetch reader.
  `both_reader.py` reads AlphaEarth's native UTM COGs and TESSERA on one grid with ranged requests (Pyodide or CPython;
  `check_both_reader.py` checks it against rasterio and zarr-python). `sql_probe.py` and `udf_probe.py` score the
  aggregate in Pyodide through xarray-sql and DuckDB: the two embeddings joined on the pixel, each 32 x 32 block's peak
  as a GROUP BY, the model as a vectorized Python function in the SQL. `crawl.py` scores tile after tile with one
  tile in memory and logs the wasm heap (`crawl_runs.sh` runs its choices); `browser_test.py` measures memory and
  storage in real browsers. `run_sql.mjs` runs a probe under Node; `export_model.py` writes the model it uses.
- `cog_vs_mosaic.py` scores the digging sites from AlphaEarth's native COGs against the mosaic the models were taught
  on.
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
uv run python checks.py          # data/scores and data/scores-abroad
uv run python ground.py          # reads 31 places, then scores them with models.py's final models
CUT=1 uv run python fire_teach.py   # about half an hour, writes data/scores-fire-abroad-cut
uv run python years.py           # 20 places x 9 years, every pair
uv run python coarse.py          # about 40 minutes
uv run python separate.py read   # about 2 hours, 4.2 GB into data/separate
uv run python separate.py test   # SECTIONS=ABCD, writes results/separate.json
```

## Data and credits

- **TESSERA** v1.1 embeddings, University of Cambridge with dClimate, read from the Zarr on Source
  Cooperative (`tessera/tessera`).
- **AlphaEarth Foundations Satellite Embedding dataset**, Google and Google DeepMind (CC BY 4.0), through
  the Earthwork cache (Source Cooperative `tge-labs/aef-mosaic`).
- **USGS 3D Elevation Program (3DEP)** 1 m DEMs (public domain), the labels, through the Earthwork cache.
- **Global Forest Change** v1.12 (Hansen, Potapov et al., University of Maryland; CC BY 4.0): tree cover loss
  year, tree cover in 2000 and the water mask, to place and group the new ground.
- **USPVDB** (US Large-Scale Solar Photovoltaic Database, USGS and LBNL; public domain) for the solar arrays;
  **Sentinel-2 L2A** (Copernicus, through Element84's earth-search) for the water masks and the pictures of the
  clearing near each mine.
- Abroad labels through the Earthwork cache: IGN France RGE ALTI and LiDAR HD (Licence Ouverte / Etalab 2.0),
  Hong Kong CEDD lidar, VIRTUAL SHIZUOKA (CC BY 4.0), NASA ICESat-2 and GEDI.
