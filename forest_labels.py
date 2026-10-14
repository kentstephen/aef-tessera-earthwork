"""Satellite-laser truth at more mines in forest and woodland, and at mines where radar came and went.

Teaching burned and cut forest as quiet costs the forest mines: Kolwezi's digging found falls, since clearing
is the first stage of a mine and the models learn to doubt it. Only a few abroad mines teach that clearing
followed by digging is digging. This builds more of them, with Earthwork's ICESat-2 and GEDI pairs (its
earthwork_points_space: the same passes, pairing, thresholds and labels), at mines cut into forest or
woodland:
  forest  Madre de Dios gold (Peruvian Amazon), Tarkwa gold (Ghana), Sangatta and Adaro coal (Kalimantan),
          Sorowako and Pomalaa nickel (Sulawesi), Boke bauxite (Guinea, woodland), Fort Hills oil sands (Alberta
          boreal forest), Detour Lake gold (Ontario boreal forest)
  gap     mines where TESSERA had no radar in 2022 to 2024 (Sentinel-1B failed in December 2021): Orapa and
          Jwaneng diamonds (Botswana), Ekibastuz coal and Sokolov-Sarbai iron (Kazakhstan). A pair across the
          gap is where TESSERA's radar coming or going can read as change
And repeat 3DEP lidar over eastern Kentucky's surface coal mines (the 2019 and 2023 flights, in forest),
the one place found where two 1 m flights cover working mines cut from forest (West Virginia's and Virginia's
mines have one flight each since AlphaEarth began).
Each box sits where tree cover loss in 2019 to 2023 (Hansen v1.12) or AlphaEarth's change from 2019 to 2024 is
densest within about 20 km of the mine. Mining boxes teach both ways: the unchanged pairs there are mostly
forest cleared or left standing with the ground in place, the contrast the burn teaching needs.

Raw passes are cached under data/space-forest-raw, the AlphaEarth side of each site (Earthwork's site_points
layout) under data/space-forest/aef, and each label's pixel and year pair under data/space-forest (rows as
space_positions.py writes), apart from data/space so no other recipe picks them up.

Run (in Earthwork's environment, which holds its readers):
  uv run --project ../aef-earthwork --with icechunk --with earthaccess --with h5py python forest_labels.py [site ...]
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(HERE, "data", "space-forest")

# (name, kind, a point inside the site, half the box in degrees, group)
SITES = [
    ("madre-de-dios-pe", "mining", (-70.066, -12.916), 0.03, "forest"),  # La Pampa and Guacamayo gold workings
    ("tarkwa-gh", "mining", (-1.913, 5.158), 0.03, "forest"),  # gold pits and river workings south of Tarkwa
    ("sangatta-id", "mining", (117.473, 0.701), 0.03, "forest"),  # Kaltim Prima coal pits
    ("adaro-id", "mining", (115.598, -2.184), 0.03, "forest"),  # Adaro coal pits
    ("sorowako-id", "mining", (121.540, -2.548), 0.03, "forest"),  # nickel laterite mines east of Sorowako
    ("pomalaa-id", "mining", (121.759, -4.165), 0.03, "forest"),  # nickel laterite mines inland of Pomalaa
    ("boke-gn", "mining", (-13.683, 10.978), 0.03, "forest"),  # bauxite plateaus in woodland
    ("fort-hills-ab", "mining", (-111.58, 57.35), 0.03, "forest"),  # oil sands cut from boreal forest
    ("detour-lake-on", "mining", (-79.69, 50.02), 0.03, "forest"),  # gold pit and dumps in boreal forest
    ("orapa-bw", "mining", (25.379, -21.293), 0.03, "gap"),  # diamond pit and dumps
    ("jwaneng-bw", "mining", (24.693, -24.521), 0.03, "gap"),  # diamond pit and dumps
    ("ekibastuz-kz", "mining", (75.391, 51.726), 0.03, "gap"),  # coal pits and dumps
    ("sokolov-kz", "mining", (63.188, 52.986), 0.03, "gap"),  # iron pits and dumps
]

# 3DEP sites as Earthwork's site_data builds them: (name, kind, a point inside the tile, first project, second
# project). Points on the densest tree cover loss of 2018 to 2022 in a 4 km block. Tried and left out: Magoffin
# County (-82.96, 37.55), whose moved pixels are scattered singles (none in a patch over 1 ha, 6% past 2 m), the
# lidar's ground under forest rather than mining
LIDAR_SITES = [
    ("ky-pike", "mining", (-82.32, 37.51), "KY_Eastern_2019_A19", "KY_CentralEast_A23"),
    ("ky-perry", "mining", (-83.32, 37.23), "KY_WestCentral_2017_C19", "KY_CentralEast_A23"),
    ("ky-clay", "mining", (-83.40, 37.07), "KY_WestCentral_2017_C19", "KY_CentralEast_A23"),
]
LIDAR_DIR = os.path.join(DIR, "lidar")


def lidar_names():
    have = {f[:-4] for f in os.listdir(LIDAR_DIR) if f.endswith(".npz")} if os.path.isdir(LIDAR_DIR) else set()
    return [s[0] for s in LIDAR_SITES if s[0] in have]


def names(group=None):
    """The sites whose positions were built (space-<name>), all or one group."""
    have = {f[:-4] for f in os.listdir(DIR) if f.endswith(".npz")} if os.path.isdir(DIR) else set()
    return [f"space-{s[0]}" for s in SITES if s[0] in have and (group is None or s[4] == group)]


def group_of(name):
    return next(s[4] for s in SITES if f"space-{s[0]}" == name)


GRANULE_S = 420  # a granule not read in this long is skipped and its workers started fresh


def granules(found, reader, box, what, workers=3):
    """Earthwork's _granules with a deadline: its reads can stall with no connection open and never return (seen
    at Madre de Dios and Orapa, an hour with every worker idle), so a granule past GRANULE_S is skipped and the
    worker pool started again for the ones left. Fewer workers than Earthwork's five: another reader may run."""
    import earthwork_points_space as sp
    from concurrent.futures import ProcessPoolExecutor, TimeoutError
    from concurrent.futures.process import BrokenProcessPool

    jobs = [(g.data_links()[0], reader, box, what) for g in found]
    k = 0
    while k < len(jobs):
        pool = ProcessPoolExecutor(workers)
        futs = [pool.submit(sp._granule, j) for j in jobs[k:]]
        stalled = False
        for f in futs:
            try:
                got = f.result(timeout=GRANULE_S)
            except (TimeoutError, BrokenProcessPool) as e:
                why = f"stalled past {GRANULE_S} s" if isinstance(e, TimeoutError) else "lost its worker"
                print(f"    a {what} granule {why}: skipped, workers restarted", flush=True)
                got, stalled = [], True
            k += 1
            if k % 20 == 0:
                print(f"    {what}: {k} of {len(jobs)} granules read", flush=True)
            yield got
            if stalled:
                break
        for proc in list((getattr(pool, "_processes", None) or {}).values()):
            proc.kill()
        pool.shutdown(wait=False, cancel_futures=True)


def main():
    sys.path.insert(0, os.path.join(HERE, "..", "aef-earthwork"))
    import earthwork_model as em
    import earthwork_points_space as sp
    import space_positions

    em.DATA = LIDAR_DIR
    os.makedirs(LIDAR_DIR, exist_ok=True)
    want = sys.argv[1:]
    for name, kind, point, p1, p2 in LIDAR_SITES:
        if want and name not in want:
            continue
        d = em.site_data(name, kind, point, p1, p2)
        if d is not None:
            m = json.loads(str(d["meta"]))
            print(f"{name}: {m['aef']}, moved {int((d['y'] == 1).sum()):,}, unchanged {int((d['y'] == 0).sum()):,}", flush=True)
    sp.RAW = os.path.join(HERE, "data", "space-forest-raw")
    sp._granules = granules
    os.makedirs(f"{DIR}/aef", exist_ok=True)
    for name, kind, point, half, _ in SITES:
        if want and name not in want:
            continue
        if os.path.exists(f"{DIR}/{name}.npz"):
            continue
        # GEDI flies only between 51.6 degrees north and south: past it a search still finds hundreds of granules
        # (612 at Ekibastuz) that hold no footprint in the box
        sp.SOURCES = ("is2", "atl08") if abs(point[1]) > 51.5 else ("is2", "atl08", "gedi")
        cache = f"{DIR}/aef/space-{name}.npz"
        if not os.path.exists(cache):
            p = sp.site_pairs(name, kind, point, half, sp.SOURCES)
            if sp.site_points(name, kind, point, half, pairs=p, path=cache) is None:
                continue
        got = space_positions.positions(name, kind, point, half)
        z = np.load(cache, allow_pickle=True)
        pos, y, dz = got
        same = len(y) == len(z["y"]) and (y == z["y"]).all() and np.allclose(dz, z["dz"], atol=1e-4)
        print(f"{name}: {len(y)} rows ({int((y == 1).sum())} moved), labels and heights match: {same}", flush=True)
        if same:
            np.savez_compressed(f"{DIR}/{name}.npz", rc=pos.astype(np.int64),
                                meta=json.dumps({"name": f"space-{name}", "kind": kind, "point": list(point)}))


if __name__ == "__main__":
    main()
