## More mines in forest, and digging where radar came and went

`forest_labels.py` (Earthwork's environment) builds the truth, `forest_mines.py` reads TESSERA under it, scores it
and teaches with it. Satellite labels are Earthwork's ICESat-2 and GEDI pairs (same passes, pairing and thresholds);
the Kentucky sites are repeat 3DEP lidar, labeled as the US digging sites are. Everything new lives in
`data/space-forest` (apart from `data/space`, so no other recipe picks it up). Abroad recipe, either_radar rule, 60%
cutoff set on the 14 US digging sites, unless a line says otherwise.

### The new truth

| site | where, what | truth | moved | unchanged | role |
|---|---|---|---|---|---|
| ky-clay | Kentucky surface coal, forest | lidar 2017 vs 2023 | 22,914 | 845,068 | taught |
| ky-perry | Kentucky surface coal, forest | lidar 2017 vs 2023 | 4,333 | 177,729 | taught |
| ky-pike | Kentucky surface coal, forest | lidar 2019 vs 2023 | 2,988 | 1,195,040 | taught |
| fort-hills-ab | oil sands cut from boreal forest | ATL03, ATL08 | 10,141 | 2,929 | taught (weight capped at 2,000 labels) |
| boke-gn | bauxite in woodland, Guinea | satellite | 35 | 28 | taught |
| adaro-id | coal, Kalimantan | satellite | 28 | 40 | taught |
| madre-de-dios-pe | gold, Peruvian Amazon | satellite | 21 | 173 | taught |
| sangatta-id | coal, Kalimantan | satellite | 13 | 15 | taught |
| detour-lake-on | gold, Ontario boreal forest | satellite | 154 | 2,689 | test only (arrived after the runs started) |
| pomalaa-id, sorowako-id, tarkwa-gh | nickel, nickel, gold | satellite | 5, 2, 0 | 1, 14, 38 | too few moved: test only |
| orapa-bw, jwaneng-bw | diamonds, Botswana (radar gap) | satellite | 1,609, 2,909 | 8,423, 13,306 | test only |
| ekibastuz-kz, sokolov-kz | coal, iron, Kazakhstan (radar gap) | ATL03, ATL08 | 297, 36 | 3,446, 851 | test only |

- Two 1 m 3DEP flights over working mines cut from forest exist in eastern Kentucky (KY_Eastern_2019 or
  KY_WestCentral_2017, then KY_CentralEast_A23). West Virginia's and Virginia's coal country has one flight since
  AlphaEarth began. Magoffin County was tried and left out: its moved pixels are scattered singles (none in a patch
  over 1 ha), the lidar's ground under forest rather than mining.
- The satellite lasers give tens of moved labels at a tropical forest mine, not hundreds: canopy and cloud thin the
  pairs. Open pits (Fort Hills, Botswana, Kazakhstan) give thousands.
- At Madre de Dios the "unchanged" labels (ground within 0.75 m, mostly cleared forest and ponds) are lit 41% by
  the map's rules: the clearing-without-digging case, inside a mining district.
- Earthwork's granule reader can stall with no connection open (an hour, every worker idle). `forest_labels.py`
  gives each granule 420 s, then skips it and restarts the workers; GEDI is skipped north of 51.5 degrees, where a
  search still returns hundreds of granules with no footprint in the box.

### Digging where TESSERA's radar came or went

Final models of each recipe, never taught on these sites; the earlier satellite sites from their held-out scores.
"Flip" is a label whose pair has Sentinel-1 in one year and none in the other. The earlier sites held 7 moved flip
labels (Shurugwi 6, Unki 1); the four new mines add 1,891.

| at 60% caught: moved found / unchanged lit | labels (moved / unchanged) | map today | joint x either | joint x radar-safe |
|---|---|---|---|---|
| abroad recipe, radar both years | 2,953 / 14,595 | .69 / .053 | .47 / .044 | .47 / .044 |
| abroad recipe, radar one year only | 1,898 / 11,431 | .52 / .134 | .31 / .092 | .31 / .082 |
| US recipe, radar both years | 2,953 / 14,595 | .07 / .006 | .11 / .015 | .11 / .015 |
| US recipe, radar one year only | 1,898 / 11,431 | .08 / .032 | .09 / .057 | .09 / .028 |

- The radar-safe gate costs no digging pooled and lights less where radar came or went: half as much under the US
  recipe (.057 to .028), a tenth less under the abroad recipe (.092 to .082).
- Site by site it is not always free: Ekibastuz under the US recipe .82 to .72 found (lit .30 to .13), Sokolov under
  the abroad recipe .37 to .27 (36 labels). Orapa and Jwaneng are found the same.
- These are the same problem abroad as before: the US recipe ranks the Botswana pits well (AP .60, .48) but scores
  them dim (found under .15). The abroad recipe brightens them; AlphaEarth gated alone finds the most there (.52 to
  .69) and lights the most.

### Teaching the forest mines

`forest_mines.py teach`: fire_teach.py's abroad recipe (burned forest taught as quiet, clearcuts too with CUT=1)
plus the eight taught mines above, held out in folds with the abroad sites (the old sites keep fire_teach.py's
folds). The two fire_teach.py baselines score the new mines with their final models (`base`), so a new mine is
test only there and held out by fold in the teach runs.

| joint x radar-safe, abroad-taught | AP (14 US) | clearcuts lit | storm | crop | Kolwezi found (AP) | Hibbing found (AP) | Detour Lake | KY found: Pike, Perry, Clay |
|---|---|---|---|---|---|---|---|---|
| burns taught | .582 | .150 | .089 | .038 | .44 (.85) | .12 (.54) | .03 | .30, .37, .62 |
| burns + new mines | .583 | .121 | .067 | .042 | .38 (.84) | .26 (.57) | .14 | .31, .55, .67 |
| burns + cuts | .575 | .051 | .029 | .035 | .28 (.85) | .12 (.56) | .03 | .33, .36, .60 |
| burns + cuts + Kentucky only | .578 | .057 | .039 | .037 | .18 (.86) | .15 (.57) | | .33, .55, .67 |
| burns + cuts + new mines | .577 | .058 | .030 | .040 | .23 (.82) | .26 (.58) | .11 | .33, .55, .67 |

| satellite forest mines found (AP) | Madre de Dios | Sangatta | Adaro | Boke | Fort Hills |
|---|---|---|---|---|---|
| burns taught (test only) | .43 (.14) | .38 (.74) | .64 (.83) | .74 (1.00) | .31 (.93) |
| burns + new mines (held out by fold) | .43 (.13) | .15 (.75) | .29 (.84) | .17 (.99) | .18 (.93) |
| burns + cuts (test only) | .43 (.15) | .38 (.75) | .54 (.85) | .49 (1.00) | .38 (.93) |
| burns + cuts + new mines (held out by fold) | .43 (.14) | .31 (.76) | .25 (.89) | .43 (.99) | .14 (.92) |

- More forest mines do not remove the trade. Teaching clearcuts costs Kolwezi about as much with them as without
  (.38 to .23 against .44 to .28), and AP at the 14 US sites moves by under .003.
- They help elsewhere: Hibbing's found doubles (.12 to .26) in both recipes, Detour Lake (a boreal mine never
  taught) goes from .03 to .14 (.11 with cuts taught), Perry from .37 to .55, and clearcuts fall from .150 to .121 with no
  clearcut taught.
- At every forest mine the ranking barely moves (Kolwezi's AP .82 to .86, Fort Hills .92 to .93); what moves is
  the score's level against the US cutoff. The satellite forest mines score dimmer when the others are taught
  (Adaro .64 to .29, Boke .74 to .17) at the same AP: their unchanged labels are mostly cleared forest on ground
  that stayed, so they teach clearing as quiet as much as they teach digging.
- Kentucky alone (with cuts) lifts Hibbing a little (.15) and lowers Kolwezi further (.18): US forest coal teaches
  the US level, not the Congo's.
- One run each, no seeds; Kolwezi's found swings with fold makeup (a first Kentucky run with refolded sites gave
  .11), so differences under about .05 at one site are noise.
