"""Which years the map compares: the radar gap and the span between the years, on quiet ground.

Sentinel-1B failed in December 2021 and Sentinel-1C began in 2025, so in 2022 to 2024 about a quarter of
land had no radar in TESSERA's inputs (filled with a neutral input). The Sahara pair (2023 vs 2025) lit up
everywhere for TESSERA alone. Every test so far used the years either side of a lidar pair; the map will
compare whatever years are asked of it.

Here: 2 km squares of mostly natural ground across the world (deserts, steppe, savanna, rainforest, taiga,
three in the US), every year 2017 to 2025 in both embeddings, and every pair of those years scored by both
recipes' final models (models.py) at each rule's 60% cutoff. Nothing is known to have been dug, so the share
lit is a ceiling on false alarms (a road or a new field would count). Pairs are grouped by the radar each
pixel had in the two years (both, one, neither) and by the span between them.

Run: uv run python years.py   (reads into data/years, writes results/years.json)
"""

import json
import os
import time
from itertools import combinations

import numpy as np
import zarr
from obstore.store import HTTPStore
from zarr.storage import ObjectStore

from tessera_sites import AEF_RES, AEF_X0, AEF_Y0, TESSERA, aef_unit, read_rc

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(HERE, "data", "years")
YEARS = tuple(range(2017, 2026))
HALF = 100  # 2 km squares
PLACES = {
    "sahara-erg-dz": (2.5, 28.0), "rub-al-khali-sa": (50.5, 20.5), "atacama-cl": (-69.6, -23.8),
    "simpson-au": (137.0, -25.5), "kalahari-bw": (22.5, -22.5), "sahel-ml": (-1.5, 15.5),
    "serengeti-tz": (34.8, -2.4), "arnhem-au": (134.0, -13.0), "salonga-cd": (21.0, -2.0),
    "jau-br": (-62.5, -2.0), "kerihun-id": (113.0, 1.2), "mongolia-steppe": (106.5, 46.5),
    "kazakh-steppe": (68.0, 49.5), "changtang-cn": (88.0, 33.0), "patagonia-ar": (-69.5, -46.0),
    "yakutia-taiga-ru": (125.0, 62.0), "ontario-shield-ca": (-85.0, 52.0), "nevada-basin-us": (-116.5, 39.5),
    "sandhills-ne-us": (-101.0, 42.0), "adirondack-ny-us": (-74.3, 44.0),
}
SHOW = ("aef_gated", "tes", "mean", "both", "both_either", "either_sqrt", "either_radar")


def read(g, name):
    path = f"{DIR}/{name}.npz"
    if os.path.exists(path):
        return path
    lon, lat = PLACES[name]
    r, c = int((AEF_Y0 - lat) / AEF_RES), int((lon - AEF_X0) / AEF_RES)
    rc = (r - HALF, r + HALF, c - HALF, c + HALF)
    out = {}
    for y in YEARS:
        e, ok = aef_unit(y, rc)
        out[f"aef_{y}"], out[f"aok_{y}"] = e.astype(np.float16), ok
    out.update(read_rc(g, name, rc, YEARS, tags=[str(y) for y in YEARS]))
    out["zone"] = np.array(out["zone"])
    os.makedirs(DIR, exist_ok=True)
    np.savez_compressed(path, rc=np.array(rc), **out)
    return path


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)


def pair(z, yb, ya):
    """A year pair in the layout aggregate.base_scores reads, and the pixels both embeddings cover."""
    ab, aa = z[f"aef_{yb}"].astype(np.float32), z[f"aef_{ya}"].astype(np.float32)
    sb, sa = z[f"{yb}_scales"], z[f"{ya}_scales"]
    ok = z[f"aok_{yb}"] & z[f"aok_{ya}"] & np.isfinite(sb) & np.isfinite(sa) & (sb > 0) & (sa > 0)
    tb = unit(np.nan_to_num(z[f"{yb}_q"].astype(np.float32) * sb[:, None]))
    ta = unit(np.nan_to_num(z[f"{ya}_q"].astype(np.float32) * sa[:, None]))
    s1b = z[f"{yb}_s1_asc_obs_count"].astype(int) + z[f"{yb}_s1_desc_obs_count"]
    s1a = z[f"{ya}_s1_asc_obs_count"].astype(int) + z[f"{ya}_s1_desc_obs_count"]
    d = {"ab": ab, "aa": aa, "tb": tb, "ta": ta, "aplain": 1 - (ab * aa).sum(1), "tplain": 1 - (tb * ta).sum(1),
         "radar_flip": (s1b > 0) != (s1a > 0)}
    radar = np.where((s1b > 0) & (s1a > 0), 2, np.where((s1b > 0) | (s1a > 0), 1, 0))
    return d, np.flatnonzero(ok), radar


def main():
    from aggregate import base_scores
    from models import final
    from quick import RULES
    zarr.config.set({"async.concurrency": 128})
    g = ObjectStore(HTTPStore.from_url(TESSERA), read_only=True)
    for n in PLACES:
        t = time.time()
        read(g, n)
        print(f"{n}: {time.time() - t:.0f} s", flush=True)
    rec = {r: final(r) for r in ("us", "abroad")}
    res = {}
    for n in PLACES:
        z = dict(np.load(f"{DIR}/{n}.npz"))
        radar_years = [int(np.median(z[f"{y}_s1_asc_obs_count"].astype(int) + z[f"{y}_s1_desc_obs_count"])) for y in YEARS]
        res[n] = {"radar_median": dict(zip(YEARS, radar_years)), "pairs": {}}
        print(f"\n{n}: median radar looks per year {dict(zip(YEARS, radar_years))}")
        for yb, ya in combinations(YEARS, 2):
            d, li, radar = pair(z, yb, ya)
            if len(li) < 1000:
                continue
            row = {"valid": round(len(li) / len(radar), 3), "radar": [round(float((radar[li] == k).mean()), 3) for k in (2, 1, 0)]}
            for r, m in rec.items():
                s = base_scores(m["models"], d, li)
                for w in SHOW:
                    lit = RULES[w](s, m["bounds"]) > m["cutoff"][0.6][w]
                    row[f"{r}:{w}"] = round(float(lit.mean()), 4)
                    row[f"{r}:{w}:by_radar"] = [round(float(lit[radar[li] == k].mean()), 4) if (radar[li] == k).sum() >= 200 else None
                                               for k in (2, 1, 0)]
            res[n]["pairs"][f"{yb}-{ya}"] = row
        show = ("aef_gated", "both", "both_either")
        print(f"  {'pair':9s} valid radar both/one/none  " + "  ".join(f"us:{w:>11s}" for w in show) + "  " + "  ".join(f"ab:{w:>11s}" for w in show))
        for p, row in res[n]["pairs"].items():
            yb, ya = map(int, p.split("-"))
            if ya - yb == 1 or (yb, ya) in ((2017, 2025), (2019, 2023), (2021, 2025), (2018, 2024), (2020, 2022), (2021, 2023)):
                print(f"  {p:9s} {row['valid']:.2f}  {row['radar'][0]:.2f}/{row['radar'][1]:.2f}/{row['radar'][2]:.2f}       "
                      + "  ".join(f"{row.get(f'us:{w}', float('nan')):14.3f}" for w in show) + "  "
                      + "  ".join(f"{row.get(f'abroad:{w}', float('nan')):14.3f}" for w in show))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, "results", "years.json"), "w"), indent=1)




def summary():
    """Share lit over places, pixel by pixel radar status (both years, one, neither) and the span between years."""
    res = json.load(open(os.path.join(HERE, "results", "years.json")))
    out = {}
    print("\nmean share lit over places (pixels grouped by radar in the two years; pairs with 200+ such pixels)")
    print(f"  {'recipe:rule':22s}  {'both':>7s} {'one':>7s} {'neither':>7s}   |  radar both years, by span: " + " ".join(f"{s}y" for s in range(1, 9)))
    for r in ("us", "abroad"):
        for w in SHOW:
            k = f"{r}:{w}"
            by = [[], [], []]
            span = {s: [] for s in range(1, 9)}
            for n, v in res.items():
                per = [[], [], []]
                sp = {s: [] for s in range(1, 9)}
                for p, row in v["pairs"].items():
                    yb, ya = map(int, p.split("-"))
                    for i, x in enumerate(row[f"{k}:by_radar"]):
                        if x is not None:
                            per[i].append(x)
                    if row[f"{k}:by_radar"][0] is not None and row["radar"][0] > 0.9:
                        sp[ya - yb].append(row[f"{k}:by_radar"][0])
                for i in range(3):  # each place counts once
                    if per[i]:
                        by[i].append(np.mean(per[i]))
                for s in sp:
                    if sp[s]:
                        span[s].append(np.mean(sp[s]))
            out[k] = {"radar": [round(float(np.mean(b)), 4) if b else None for b in by],
                      "places": [len(b) for b in by], "span": {s: round(float(np.mean(v)), 4) for s, v in span.items() if v}}
            rad = " ".join(f"{x:7.3f}" if x is not None else "      -" for x in out[k]["radar"])
            print(f"  {k:22s}  {rad}   |  " + " ".join(f"{out[k]['span'].get(s, float('nan')):.3f}" for s in range(1, 9)))
    print(f"  places with pixels in each radar group: {out['us:both']['places']}")
    json.dump(out, open(os.path.join(HERE, "results", "years_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
    summary()
