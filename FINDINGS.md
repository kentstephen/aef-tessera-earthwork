# Findings: AlphaEarth and TESSERA, apart and together

What the tests in this repo found, October 2026. Every number is held out: a digging site is scored by models
taught without it, and crops, farms, marshes, the storm coast and every place abroad are never taught unless a
section says so. Scripts: `compare.py`, `aggregate.py`, `quick.py`, `desert.py`, `abroad_teach.py`.

## The question

Earthwork's model finds ground that was dug, filled or graded from two years of AlphaEarth (AEF) embeddings,
taught by repeat 3DEP lidar. Its open problem is false alarms on fields (crop switches, farmland) and, less,
wetlands. TESSERA v1.1 is a second 10 m annual embedding that streams from a Zarr the way AEF does. Do the two
see the same moved ground, and does one score built from both do better than either?

## 1. The two see the same events

- Plain change (1 - cos of the two years) correlates .25 to .91 per site between AEF and TESSERA (building
  sites mostly .5 to .87). The Earthwork model taught in each embedding correlates .24 to .82.
- Mean AP over 14 held-out digging sites: AEF gated .553, TESSERA .549, one model on both embeddings .575.
- Canonical correlations of the change vectors are high when pooled (.94 to .80, an upper bound), but a
  linear map from TESSERA's change to AEF's, fit on other sites, fails on the held-out site (mean R2 -0.28).
  They see the same events with no fixed dictionary between them, so combining scores works better than
  translating one into the other.
- They are fooled on different fields: AEF lights Platte (.47) and Box Butte (.60) crop switches where
  TESSERA is quiet (.01, .16); TESSERA lights San Luis (.51) and Red River (.46) where AEF is quiet.

## 2. Combining, compared at the same catch rate

TESSERA's scores run hotter than AEF's, so a fixed .5 cutoff is unfair. Each rule gets its own cutoff, set so it
finds 60% of moved ground averaged over the held-out digging sites; then the share of unchanged ground above it.
"Desert" is the Sahara place (2023 vs 2025, no truth, open erg). Three seeds agree to within .003 AP; burns
swing the most (about .02 to .03).

| rule (US-taught) | AP | crop | farm | marsh | storm | burn | Sahara | island digging found |
|---|---|---|---|---|---|---|---|---|
| AEF gated, the map today | .553 | .153 | .097 | .043 | .144 | .112 | .091 | .45 |
| TESSERA | .550 | .037 | .025 | .162 | .096 | .013 | 1.00 | .60 |
| lower of the two ("agree") | .570 | .084 | .056 | .053 | .152 | .081 | .151 | .48 |
| mean of the two | .596 | .064 | .044 | .056 | .118 | .053 | .243 | .49 |
| joint model (both embeddings) | .575 | .021 | .029 | .020 | .037 | .030 | .734 | .54 |
| joint x either gate | .583 | .049 | .037 | .056 | .102 | .105 | .044 | .53 |
| joint x sqrt(either gate) | .586 | .042 | .034 | .048 | .086 | .089 | .077 | .54 |

- Every combination ranks digging better than the map today.
- The joint model is by far the quietest on US test ground (crops 15% to 2%) but lights bare desert.
- The either gate multiplies the joint model by the larger of AEF's gate and a TESSERA gate calibrated to
  the same share of unchanged teaching ground (0.115 to 0.267 in TESSERA's plain change). Digging, crops and
  burns changed in both embeddings and pass; ground where neither changed is muted. It beats the map today on
  every column. Its cost: the cutoff drops, so burns rise back to about today's level.
- Also tried and dropped: product, geometric mean, both gated by AEF alone (burns .16), TESSERA's own gate on
  the joint model, a softer either gate (Sahara back to .55), learned stacks on the two scores (good at home,
  light deserts), and a stack that sees AEF's before-year embedding (overfit: marsh .33, AP .53).

## 3. Why the deserts lit up

`desert.py`: TESSERA's own change on unchanged desert is small (median Sahara .15, Riyadh .08, Lancaster .08;
almost none past its gate). The models, not the embedding, mistake bare ground for graded ground: the same trap
AEF fell into before Earthwork's gate (Cairo 50% to 4%). Lancaster showed it inside the US training data
(TESSERA "found" 85% of moved ground there with AP .10). Other notes:
- On unchanged ground TESSERA's change follows AEF's (rho .74): where both call it changed, it mostly did
  (burns, dryland farms, the storm coast).
- 2017's thin Sentinel-2 is not the driver; drier places with more clear looks change more.
- The "Cairo desert" place sits on Egypt's New Administrative Capital, real construction: not a clean desert test.

## 4. Abroad

- Puerto Rico and the Virgin Islands (3DEP 2018 vs 2024, moves over 2 m): the joint model and the either gate
  find .53 to .54 of deep digging vs .45 for the map today, at about 2% of unchanged ground. Largest gains at
  Juana Diaz (.17 to .45), St. Thomas (.22 to .57), St. Croix (.57 to .73).
- Reunion, Guadeloupe, Hong Kong, Shizuoka unchanged ground: near zero for every rule but TESSERA alone (7%).
- Satellite-laser sites (ICESat-2 and GEDI pairs: Kolwezi, Mobimbi, Sembehun, Morowali, Shurugwi, Xiong'an,
  New Capital, Eko Atlantic): every model ranks moved ground about equally well there (AUC .7 to .95), but AEF
  and the AEF-gated rules score it dim (14 to 22% found at the US cutoff). TESSERA alone "finds" 81% and lights
  24% of the unchanged ground around the mines. Abroad the problem is the score's level, not its ranking.

## 5. Teaching with abroad truth

`abroad_teach.py` adds the island lidar (both ways) and the satellite-laser labels (digging sites both ways,
weight 5 per label; savanna and rainforest unchanged only). Fields stay test only. Abroad sites held out in 4
folds.

| at 60% caught | AP | burn | crop | farm | storm | Sahara-Riyadh-Cairo | island found | satellite found |
|---|---|---|---|---|---|---|---|---|
| joint, US-taught | .575 | .030 | .021 | .029 | .037 | .639 | .54 | .22 |
| joint x either, US-taught | .583 | .105 | .049 | .037 | .102 | .067 | .53 | .22 |
| joint, abroad-taught | .574 | .159 | .019 | .012 | .063 | .034 | .51 | .39 |
| joint x either, abroad-taught | .583 | .187 | .041 | .014 | .110 | .017 | .50 | .41 |
| joint x either, satellite labels weight 2 | .585 | .207 | .040 | .015 | .093 | .019 | .49 | .34 |
| joint x either, satellite labels only | .585 | .174 | .043 | .018 | .269 | .018 | .55 | .42 |
| joint x either, islands only | .586 | .175 | .042 | .023 | .045 | .054 | .47 | .19 |

- Abroad teaching nearly doubles satellite-laser digging found and cures the joint model's desert problem
  with no gate (64% to 3%). Farms fall further; US ranking holds.
- Burns rise with any abroad teaching. The whole burn cost is one site: Grizzly Flats (Caldor fire, 2021,
  conifer forest to bare ground), .06 to .32 for the joint model. Cameron Peak stays at 0% under every joint
  recipe. Forest burned to bare ground looks like land cleared for a mine.
- The satellite labels carry the abroad and desert gains; the islands keep the storm coast quiet.

## 6. Smaller findings

- Patch size: no combination finds more small patches (under 0.1 ha: AEF .34, TESSERA .33). Over 10 ha AEF
  is clearly better than TESSERA (.70 vs .57).
- Site by site: the mean and the joint model averaged with AEF beat the map today at 12 of 14 digging sites;
  the either gate at 11 to 12, worst loss -.04 (Aurora).
- TESSERA leaves out pixels with fewer than 15 clear Sentinel-2 looks in a year: 1 to 3% at most sites but 61%
  at Fishers. The aggregate needs AEF alone as the fallback there (it finds 45% of moved ground at 1.6% of
  unchanged).
- Radar: no US site had a year without Sentinel-1. The Sahara pair (2023 vs 2025) did, and TESSERA flagged
  it as change everywhere.
- Fewer TESSERA numbers would save no bytes: its Zarr chunks hold all 128 together.

## 7. How far the gains carry

`checks.py`, on the held-out scores. The seeds only changed the models' fit; here the 14 digging sites themselves
are resampled (bootstrap over sites) and each rule's AP is paired with the map today's at the same site.

| gain in AP over the map today | US-taught | wins | Wilcoxon p | abroad-taught | wins | Wilcoxon p |
|---|---|---|---|---|---|---|
| mean of the two | +.042 [.020, .064] | 12/14 | .004 | +.043 [.019, .066] | 10/14 | .005 |
| joint model | +.021 [-.001, .045] | 9/14 | .14 | +.032 [-.001, .065] | 10/14 | .09 |
| joint x either gate | +.030 [.012, .047] | 11/14 | .009 | +.041 [.018, .064] | 11/14 | .007 |
| joint x sqrt(either gate) | +.033 [.014, .051] | 11/14 | .005 | +.043 [.019, .067] | 11/14 | .005 |

- The gated and averaged rules' gains hold across sites; the joint model alone does not clear zero.
- The three mines all gain under the either gate (Four Corners, Hibbing, Mountain Pass); the joint model alone
  loses at Hibbing (-.04 US-taught, -.08 abroad-taught).
- Crop switches: every combination lights fewer than the map at 23 to 27 of 27 crop sets; farms 6 of 6.
- One cutoff for the map: with each site's cutoff set on the other 13, the share of moved ground found swings
  from under .2 (Four Corners, Hibbing, Lancaster) to over .9 (Katy), sd about .20, under every rule. A fixed
  cutoff finds very different shares at different places; that is the ground, not the rule.
- Digging on farms: the joint model alone finds land leveling the others miss (Fresno .56, Arkansas delta .36,
  against .09 and .08 for the map), but the either gate takes most of that back (.10, .13): leveled fields
  change little in either embedding.

## 8. New ground: burned forest, clearcuts, reservoirs

`ground.py`. No lidar: 6 km squares picked from Hansen's Global Forest Change (v1.12) so the event's ground is
about half the square, the years either side, and the share of that ground lit at each rule's 60% cutoff.
Fifteen fires (ten in the US, five abroad), ten clearcut regions (US, Sweden, Finland, New Zealand, Brazil,
Chile), six reservoirs.

| share lit, US-taught | map today | TESSERA | mean | joint | joint x either |
|---|---|---|---|---|---|
| forest burned to bare (15 fires) | .013 | .061 | .007 | .004 | .025 |
| clearcuts (10 regions) | .266 | .306 | .228 | .216 | .378 |
| standing forest beside them | .001 to .002 | .006 to .009 | .001 to .002 | .002 to .010 | .002 to .003 |

- Burned forest is mostly not a trap: Grizzly Flats was the exception, not the rule. TESSERA alone lights some
  (Mosquito .50).
- Clearcuts are: every rule lights a quarter or more of the cut ground, the either gate most (Kainuu .79,
  Oregon Coast Range .75, Tres Lagoas .39). Abroad teaching quiets the Nordic cuts and brightens the Brazilian
  and southeastern US ones.
- Reservoirs are ambiguous: Hansen's water is the lake when full, so the bathtub ring between the two years'
  levels falls in the "land" group, and both groups light (.1 to .5). A drawn-down shore is also ground whose
  surface moved, as lidar would see it.

## 9. Teaching burns and clearcuts as quiet

`fire_teach.py`: the abroad recipe plus 20,000 pixels of burned or cut ground from each place, taught as
unchanged; fires and cuts held out in folds with the abroad sites.

| joint x either gate, abroad-taught | AP | lidar burns | clearcuts (held out) | storm | crop | satellite digging found |
|---|---|---|---|---|---|---|
| without | .583 | .187 | .283 | .110 | .041 | .41 |
| burns and clearcuts taught | .575 | .000 | .051 | .029 | .035 | .40 |
| burns only taught (clearcuts test only) | .582 | .000 | .150 | .089 | .038 | .41 |

- Grizzly Flats goes dark, clearcuts fall from 28% to 5%, and the storm coast quiets as a side effect.
- The cost is at mines in forest and woodland: Kolwezi's digging found falls from .52 to .28, Hibbing's from
  .20 to .12; mean AP -.008 [-.016, -.001], 11 of 14 sites slightly lower. Clearing is the first stage of a
  mine, and the model learns to doubt it.
- Burns alone take Grizzly Flats to zero at almost no cost (AP -.0015 [-.005, +.002]), halve the clearcuts'
  false alarms though no clearcut was taught (.28 to .15), and cost Kolwezi less (.52 to .44). Hibbing's found
  falls to .12 under both, while its AP rises (.545 to .563): its scores shift against the cutoff, not its ranking.

## 10. Which years: the radar gap

`years.py`: 20 squares of quiet natural ground (deserts, steppe, savanna, rainforest, taiga), every pair of
years 2017 to 2025. Seven had no radar, or almost none, in 2022 to 2024 (Sahara, Simpson, Kalahari, Kazakh
steppe, Yakutia, northern Ontario, Mongolia).

| share lit | radar both years | radar one year only | neither year |
|---|---|---|---|
| map today, US-taught | .004 | .011 | .000 |
| TESSERA, US-taught | .092 | .192 | .003 |
| joint x either, US-taught | .002 | .104 | .000 |
| joint x either, abroad-taught | .001 | .035 | .000 |
| joint x radar-safe gate, US-taught | .002 | .005 | .000 |
| joint x radar-safe gate, abroad-taught | .001 | .000 | .000 |

- A pair where TESSERA had radar in one year and not the other is the hazard; pairs inside the gap are fine.
- The radar-safe gate opens on AEF's change alone where TESSERA's radar came or went (`either_radar` in
  quick.py). On every held-out column it equals the either gate (no US site has such pixels; Shurugwi's 13
  labels are found the same), so it costs nothing measured.
- The span between the years barely matters for the combined rules (.000 to .009 at 1 to 8 years).

## 11. Where TESSERA has no pixel

`coverage.py`, 705 random land points: TESSERA leaves out 19% of land in 2017 (25% in the tropics) and under 1%
in 2020, 2022 and 2024 (tropics 1 to 2%, with whole blocks missing in places such as Chongqing and Sulawesi).
At Fishers (2017, 61% missing) AEF alone on the missing ground lights .016 of unchanged ground against the
aggregate's .015 on the rest, so the fallback should not draw a seam.

## 12. Zoomed out

`coarse.py`: cells of about 320 m on the lidar sites, dug (0.2 ha or more moved) against quiet (none moved),
each digging site held out.

| cell score | AP | quiet farm cells lit at 60% of dug cells found |
|---|---|---|
| highest aggregate score in the cell | .872 | .053 |
| highest map score in the cell | .858 | .117 |
| AEF mean-pooled per year (as AEF's COG overviews), model on cells | .757 | .128 |
| model on pooled AEF change (mean, std, min, max) | .746 | .156 |
| model on pooled TESSERA change | .737 | .146 |
| model on both pooled | .756 | .129 |

- Carrying the highest 10 m score up beats every model taught on pooled embeddings, AEF's overviews included.
  At 640 m the order holds (.95 for the highest score against .90 to .91).

## 13. What the browser would read

`bytes_read.py`, per year: AEF about 0.52 MB per square km (its chunks compress to .4 to .7), TESSERA 1.25 to
1.42 MB (barely compresses), about 2.5 times AEF. The aggregate needs both years of both, so the start view
that reads about 325 MB of AEF would read about 800 MB more.

## 14. Clearcuts between burns only and every cut; burns on the US recipe; teaching land leveling

`fire_teach.py` (new knobs CUT_PX, CUT_W, CUT_FAR, LEVEL; the defaults reproduce the runs already on disk) and
`teach_runs.py` (the recipes side by side from their saved scores; `results/teach_runs.json`). Every number is
held out as before. Rule of record: the joint model with the radar-safe gate (`either_radar`); it equals the
either gate in every run here, since no taught or held-out digging pixel had radar in one year only. AP is the
mean over the 14 US digging sites; "vs burns only" pairs each site's AP with the burns-only run's (bootstrap 95%
over sites). Kolwezi has 392 moved labels.

### Clearcuts, abroad recipe with burns taught

| at 60% caught | cut px per region | AP | vs burns only | clearcuts lit (held out) | storm | crop | satellite digging found | Kolwezi found (AP) | Hibbing found (AP) |
|---|---|---|---|---|---|---|---|---|---|
| no burns, no cuts (abroad) | 0 | .583 | +.001 [-.002, +.005] | .283 (test) | .110 | .041 | .414 | .52 (.840) | .20 (.545) |
| burns only | 0 | .582 | | .150 (test) | .089 | .038 | .414 | .44 (.852) | .12 (.545) |
| cuts, 1,000 px | 1,000 | .579 | -.002 [-.005, .000] | .070 | .052 | .039 | .398 | .35 (.857) | .12 (.549) |
| cuts, weight .25 | 20,000 | .577 | -.005 [-.009, -.000] | .060 | .039 | .036 | .413 | .31 (.851) | .13 (.555) |
| cuts, 5,000 px | 5,000 | .578 | -.004 [-.009, .000] | .057 | .035 | .037 | .410 | .30 (.857) | .12 (.557) |
| cuts, farthest half | about 10,000 | .576 | -.005 [-.011, -.000] | .056 | .030 | .034 | .403 | .27 (.859) | .12 (.558) |
| every cut | 20,000 | .575 | -.007 [-.013, -.001] | .051 | .029 | .035 | .403 | .28 (.854) | .12 (.563) |

- The curve is steep at the start and flat after: 1,000 pixels per region takes clearcuts from .150 to .070, and
  twenty times more (or a quarter weight) only reaches .051 to .060.
- Kolwezi's found falls in step with the clearcuts at every point (.44, .35, .30, .28). No variant quiets the
  cuts and keeps Kolwezi. Its AP does not fall (.852 to .859 throughout): the mine's scores sink against the
  cutoff, its ranking holds, as Hibbing's did before.
- CUT_FAR measured each cut pixel's change (AlphaEarth's a - b beside TESSERA's, 192 numbers) against the
  nearest of 3,000 moved pixels at each taught digging site, taking the nearest over only the sites a model is
  taught on, and taught the farthest half of each region (at most 20,000). It lands with every cut taught
  (Kolwezi .27): the cut pixels nearest digging are not what costs Kolwezi.
- Hibbing's found sits at .12 in every burns run, cuts or not; burns, not cuts, move it (.20 to .12).
- AP drops a little with any cut teaching, from -.002 (1,000 px) to -.007 (every cut); 8 to 11 of 14 sites lower.

### Burns only on the US recipe

| at 60% caught | AP | vs US recipe | lidar burns | burned forest | clearcuts (untaught) | storm | crop | farm | satellite found | Kolwezi | Hibbing found (AP) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| US recipe | .583 | | .105 | .025 | .378 | .102 | .049 | .037 | .222 | .06 | .31 (.533) |
| US recipe, burns taught | .585 | +.002 [-.005, +.010] | .000 | .000 | .041 | .087 | .055 | .034 | .203 | .03 | .19 (.569) |

- On the US recipe teaching burns alone takes the clearcuts from .378 to .041 with no cut taught, further than
  the abroad recipe gets with every cut taught (.051), at no measured AP cost (7 of 14 sites lower).
- Burns go dark at both lidar burns and the 15 fires. Hibbing's found falls (.31 to .19) while its AP rises
  (.533 to .569); satellite digging found slips (.222 to .203).
- Gain over the map today: +.032 [+.010, +.052], 11 of 14 sites.

### Teaching land leveling (abroad recipe with burns, farms held out in the folds)

Five farm lidar sites taught their moved ground as digging (Fresno 5,376 px, Antelope 3,260, Arkansas delta
3,327, Hamilton 2,269, Box Butte 195; Othello's 23 too few). LEVEL=1 teaches only the moved ground, LEVEL=2 its
unchanged ground beside it (40,000 px per farm).

| at 60% caught | AP | vs burns only | crop switches lit | farms' unchanged lit | storm | clearcuts | satellite found | Kolwezi | Hibbing | Mountain Pass |
|---|---|---|---|---|---|---|---|---|---|---|
| burns only (farms test only) | .582 | | .038 | .013 | .089 | .150 | .414 | .44 | .12 | .60 |
| leveling taught | .572 | -.010 [-.017, -.003] | .209 | .069 | .071 | .137 | .462 | .52 | .12 | .45 |
| leveling and farms' unchanged taught | .567 | -.014 [-.026, -.004] | .052 | .018 | .038 | .142 | .441 | .40 | .09 | .56 |

Leveling found at each farm, held out (unchanged lit at that farm):

| farm | burns only, gated | leveling taught, gated | both taught, gated | burns only, joint alone | leveling taught, joint alone | both taught, joint alone |
|---|---|---|---|---|---|---|
| Fresno | .04 (.026) | .07 (.073) | .07 (.044) | .05 (.018) | .12 (.052) | .13 (.034) |
| Arkansas delta | .13 (.017) | .14 (.154) | .15 (.013) | .61 (.030) | .66 (.141) | .85 (.039) |
| Antelope | .53 (.004) | .61 (.023) | .52 (.002) | .53 (.003) | .60 (.014) | .51 (.001) |
| Hamilton | .63 (.014) | .65 (.023) | .66 (.020) | .56 (.009) | .57 (.014) | .57 (.013) |

- Under the either gate leveling found barely moves (Fresno .04 to .07, Arkansas .13 to .15): leveled fields
  change little in either embedding and the gate holds them shut, taught or not.
- Teaching only the moved ground lights crop switches 5 times as often (.038 to .209) and farms' unchanged
  ground (.013 to .069). Teaching the farms' unchanged ground beside it takes most of that back (.052, .018).
- The joint model alone (no gate) finds Arkansas leveling (.61 to .85 with both taught), not Fresno (.13).
- Digging AP falls with either variant (-.010, -.014; 11 to 12 of 14 sites lower). Mountain Pass found falls
  (.60 to .45 with moved ground only); satellite digging found rises (.414 to .44 and .46).
- Farms are held out in the same four folds as the abroad sites and fires, so the folds differ from the burns
  only run's; part of Kolwezi's move (.44 to .52 and .40) may be the change of fold, not the farms.

## 15. More mines in forest, and digging where radar came and went

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

## 16. Lookalikes not tried before, and the reservoirs split by a water mask

`lookalikes.py`. No lidar. Each place is a 6 km square on AlphaEarth's grid with the years either side of the
event, an event group against ground to compare in the same square, scored by both recipes' final models at each
rule's 60% cutoff (the share of each group lit). Groups:
- slide: tree cover lost (Hansen GFC v1.12) in the slide's year and the next (Landsat dates many September
  slides a year late), against forest that never lost cover. Iburi (Hokkaido 2018 earthquake), Nippes (Haiti
  2021), the Romagna Apennines (May 2023 rains), Wayanad (one debris flow, 2024).
- river: water one year, land the other (bank cut away, new bar), against land both years. Ucayali, Brahmaputra
  at Majuli, Jamuna, Kosi; 2018 against 2021, each square centered where Hansen's water is nearest half.
- solar: a USPVDB array that went into service between the years, against ground 50 m or more from any array.
  The eight largest built 2020 to 2023 across ground types (Texas prairie, pasture and Permian scrub, Nevada and
  Arizona desert, Ohio and Central Valley farmland, Colorado grassland); years: in service minus 2, plus 1.
- fill: land in the first year, water in the second, nothing dug between: the Cottbus-Nord lignite pit flooded
  on purpose (2018 against 2024) and the Blue Nile valley behind GERD (2019 against 2023).
- dredge: new water among the lower Rhine gravel pits (2018 against 2023): pits dredged wet, real digging.
- flood: seen under water in at least a quarter of the flood weeks' clear looks and land in both years, against
  land never seen flooded. Sindh (2022) and Conselice in Romagna (May 2023).

The water groups come from Sentinel-2 L2A (Element84 earth-search COGs): per pixel, the share of a year's clear
looks (up to 3 per month, SCL clear classes) with MNDWI above zero; water past 0.6, land under 0.4.

| share lit, joint x either gate | US-taught event | compare | abroad-taught event | compare | what lidar would see |
|---|---|---|---|---|---|
| landslides (4) | .319 | .057 | .408 | .064 | moved: scars and deposits, meters |
| river banks and bars (4) | .025 | .004 | .230 | .019 | moved: bank cut, bar built (water is flattened) |
| solar arrays (8) | .267 | .055 | .192 | .026 | some moved: grading and roads; most array ground under .5 m |
| pit or valley flooded (2) | .500 | .119 | .599 | .139 | nothing dug; a DEM difference reads the new water surface as meters of change |
| gravel pits dredged (1) | .510 | .072 | .353 | .037 | moved: real digging under water |
| flood silt (2) | .053 | .011 | .038 | .003 | under .5 m: unchanged |

| same, other rules (US-taught event / abroad-taught event) | map today | TESSERA | mean |
|---|---|---|---|
| landslides | .247 / .424 | .454 / .459 | .251 / .404 |
| river banks and bars | .025 / .463 | .468 / .306 | .033 / .430 |
| solar arrays | .199 / .193 | .474 / .372 | .279 / .216 |
| pit or valley flooded | .499 / .970 | .384 / .420 | .445 / .849 |
| gravel pits dredged | .514 / .431 | .445 / .563 | .488 / .467 |
| flood silt | .004 / .009 | .525 / .352 | .007 / .019 |

- Landslides light about a third (US-taught) to two fifths (abroad-taught) of the scar ground under the either
  gate. Iburi .66 and .61, Romagna .31 and .19, Nippes .29 and .45; Wayanad's single debris flow .02 US-taught
  but .39 abroad-taught. These are real moved ground: as digging goes, a slide is the event the model is built
  to find, made by no one.
- Rivers: the US-taught aggregate is nearly dark on moved banks and bars (.025); abroad teaching lights a fifth
  to a quarter (Jamuna .30, Ucayali .39 on only 259 pixels, Kosi .05). The map today, abroad-taught, lights
  .46: the either gate holds the joint model under it.
- Solar: a quarter of new array ground lights US-taught (.27), a fifth abroad-taught. It ranges widely: Arrow
  Canyon .67 (desert, graded) and Fox Squirrel .54 (Ohio farmland) against Aktina .001 and Sonoran .04. Slate's
  comparison ground (Central Valley fields) lights .34, more than its array (.22): the field false alarms
  again, not the panels.
- Ground flooded behind a dam or in a pit lights almost wherever it went under: Cottbus-Nord .998 under every
  rule but TESSERA alone. At GERD only abroad teaching lights it (.21 joint x either, .95 map today); US-taught
  stays dark (.003). Nothing was dug in either between the years.
- Wet-dredged gravel pits light about half (US-taught .51), the share building sites find.
- Flood silt stays dark for every combined rule (.04 to .05); TESSERA alone lights half of it (.53) and a fifth
  of the land beside it.
- Radar came and went at none of these squares (flip share 0), so the radar-safe gate equals the either gate
  throughout.
- Caveats: Hansen's loss in the slide years includes some harvest; Iburi's pair is 2018 against 2020 because
  TESSERA has no Hokkaido pixels in 2017 (under 15 clear looks), so its "before" year already holds the
  September slides weakly, and its comparison forest lights .15 (mixed pixels at slide edges, slides on
  ground Hansen did not call forest). Sindh's flood covered almost the whole square (only 2,019 comparison
  pixels). Romagna's flood weeks had a median of 3 clear looks.

### The reservoirs, split by what each pixel was in the two years

`lookalikes.py lakes`: the six reservoir squares from section 8, each pixel called water both years, land both,
water to land (the bathtub ring exposed) or land to water (refilled), by the same Sentinel-2 mask.

| share lit, mean over reservoirs | US-taught: map today | joint x either | abroad-taught: map today | joint x either |
|---|---|---|---|---|
| water both years (6) | .029 | .034 | .020 | .016 |
| land both years (6) | .131 | .089 | .100 | .061 |
| water to land, ring exposed (4) | .804 | .828 | .784 | .780 |
| land to water, ring refilled (3) | .659 | .533 | .621 | .303 |

- The lit ground is the bathtub ring. Exposed ring lights about four fifths under every rule (Mead .995, Sau
  .996, Powell .76, Great Salt Lake .56); the lake that stayed water lights 2 to 3%.
- Section 8's reading is explained: at Oroville and Folsom (2021 low, 2023 refilled) the ring went land to water,
  inside Hansen's "land" group, which is why that group lit .38 there. Abroad teaching quiets refilled ring
  (.53 to .30) but not exposed ring.
- Land both years still lights 6 to 13%: at Mead and Powell, ground exposed before the first year and drying,
  washing and greening since (Powell's land lights .12 under the either gate).
- Like a flooded pit, an exposed lake bed is ground whose surface changed by meters without anyone digging: a
  DEM difference with flattened water would call it moved.

Files: `lookalikes.py`, `results/lookalikes.json`, `results/reservoir_rings.json`, logs
`results_lookalikes.log` and `results_lookalikes_redo.log`; places in `data/places/ground-look-*`,
`data/tessera/ground-look-*`, water masks in `data/lookalikes/water/`, scores in `data/scores-look-{us,abroad}/`.

## 17. Wider year pairs on the digging sites

`spans.py`. Each US digging site scored at wider pairs than its lidar window: the year before the first flight
(b) and the year of the second (a), then one or two years earlier at b, later at a, or both, where both
embeddings have the years (2017 to 2025). Same lidar labels, abroad recipe, each site held out alone (the native
pair reproduces data/scores-abroad exactly), each rule at the recipe's 60% cutoff. No US site had radar in only
one year of any pair, so the radar-safe gate equals the either gate throughout.

The moved ground stays dug in a wider pair, so its share found is the measure here. The unchanged ground is only
known to be unchanged inside the lidar window: a wider pair takes in digging before the first flight or after the
second, so its share lit (and so the AP) is an upper bound on the false alarms, not a measure of them.

| joint x either gate, mean over the sites with the pair | sites | moved found (native) | AP (native) | unchanged lit (native) |
|---|---|---|---|---|
| b - 1 | 7 | .60 (.56) | .541 (.599) | .057 (.041) |
| b - 2 | 3 | .55 (.48) | .582 (.613) | .055 (.024) |
| a + 1 | 13 | .61 (.59) | .493 (.569) | .052 (.042) |
| a + 2 | 10 | .63 (.59) | .362 (.559) | .070 (.026) |
| b - 1 and a + 1 | 6 | .64 (.56) | .480 (.571) | .074 (.047) |

| map today (AEF gated), same pairs | moved found change | AP change | unchanged lit (native) |
|---|---|---|---|
| b - 1 | +.05 | -.051 | .083 (.054) |
| b - 2 | +.06 | -.061 | .059 (.032) |
| a + 1 | +.01 | -.045 | .066 (.058) |
| a + 2 | +.08 | -.131 | .082 (.038) |
| b - 1 and a + 1 | +.10 | -.056 | .102 (.061) |

- The moved ground is found as well or better in every wider pair (+.01 to +.08 under the aggregate). A wider
  window does not wash out the digging.
- Unchanged ground lit rises with the span, most two years after the second flight (.026 to .070), and AP falls
  with it (.559 to .362). Some of that is real digging after the lidar (Columbus NE at 2018 to 2022: lit .052 to
  .257, the suburb still building); the labels cannot say how much.
- Hibbing found doubles one year later (.20 to .41 at 2020 to 2023): its second flight's project ran to May
  2023, so the 2022 embedding missed digging the lidar saw. Columbus NE (flown to April 2021) gains the same way
  (.88 to .94). Where a flight's project runs into the next year, the native pair is too narrow.
- The aggregate loses more AP than the map today two years out (-.197 against -.131) while finding about the
  same moved ground; it may light more of the digging after the window, which these labels count as false.

## 18. A cutoff set per place

`calib.py`. Each rule's score set against its own place before one cutoff is set (60% of moved ground found,
averaged over the held-out digging sites), then each digging site at a cutoff set on the other 13, as in
section 7. Place-wide: rank (the score's quantile in the place), z (the logit's z-score), over q99 (the score over
the place's 99th percentile, held at least f). Local: contrast (the score less the mean within about 1 km or 3
km) and ratio (over that mean). Part A from the saved held-out scores (a place is its labeled pixels); part B
rescores every valid pixel of every grid with models that did not see it (abroad recipe; labeled pixels match
the saved scores exactly), as a map would hold them. Crop switches and satellite labels are not grids: their
place is their own pixels, the local mean over neighbors among them (the place's mean under 20 in the window).

| joint x either gate, abroad recipe, part B | AP | found min / median / max, sd | Four Corners | Hibbing | Katy | crop | farm | storm | burn | cut | Sahara-Riyadh-Cairo | island found | satellite found |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| one cutoff (today) | .583 | .17 / .64 / .92, .24 | .18 | .17 | .92 | .041 | .014 | .110 | .187 | .283 | .017 | .50 | .41 |
| rank | .583 | .14 / .64 / .96, .25 | .14 | .35 | .35 | .070 | .056 | .061 | .071 | .525 | .070 | .71 | .40 |
| z | .583 | .11 / .62 / .95, .26 | .11 | .69 | .37 | .070 | .025 | .011 | .029 | .622 | .061 | .65 | .37 |
| over q99 (f .2) | .574 | .17 / .64 / .92, .23 | .17 | .17 | .92 | .057 | .020 | .109 | .192 | .336 | .031 | .56 | .43 |
| contrast 1 km | .385 | .28 / .65 / .85, .18 | .28 | .35 | .48 | .034 | .058 | .172 | .068 | .389 | .078 | .63 | .46 |
| contrast 3 km | .475 | .21 / .66 / .90, .22 | .26 | .30 | .43 | .039 | .041 | .124 | .040 | .370 | .043 | .59 | .40 |
| ratio 3 km | .376 | .05 / .70 / .91, .25 | .31 | .47 | .20 | .060 | .091 | .066 | .040 | .459 | .116 | .69 | .37 |

- Nothing evens out the sites. The spread (sd .24) stays .22 to .26 under every place-wide rule; it moves which
  sites are dim (Hibbing up, Katy down) without closing the gap. Only the 1 km contrast narrows it (.18, min .17 to
  .28), at a third of the AP (.583 to .385): it dims the middle of every large dug patch.
- Rank and z light about the same share of every place by design, so the quiet ground pays: deserts .017 to
  .06 or .07, clearcuts .28 to .5 or .6, islands' unchanged ground .006 to .08 or .14.
- Over q99 changes almost nothing: at most sites the 99th percentile is already near 1.
- The local contrasts quiet the large uniform events: burns .187 to .04 at 3 km (Grizzly Flats burned as one
  sheet). The cost is AP .11 and deserts up from .017 to .043.
- Why: the swing is mostly order inside the site and how much of it was dug, not a level. At Four Corners 21% of
  the site moved and its moved ground scores a median .20 while the top 1% of its unchanged ground scores .99:
  no setting of the cutoff by place puts the dug ground above the rest. Hibbing is dim throughout (its 99th
  percentile .80), so place-wide rules help it, but it is the only site of that kind.
- Part A (the labeled pixels only) agrees in every direction, with somewhat different numbers (rank: Katy .44,
  Mountain Pass .76), and the same holds for the US recipe and for the abroad recipe taught with burns.

## 19. Clearing that was dug against clearing that was not

`separate.py` reads, `separate_test.py` tests, `near_panels.py` draws the clearing picked near each mine. Every
section before this one taught clearcuts as quiet and watched Kolwezi pay. Here the question is asked head on: given
ground whose forest was cleared, can the embeddings say whether it was then dug? A small head (logistic regression on
scaled features, each place weighted the same and its two groups balanced) is taught on that question alone, and
every number is held out by place.

Truth:
- **Same place, same years:** Hansen's forest loss in a year of the lidar window, split by the lidar into cleared and
  dug (moved) and cleared and not dug (unchanged). Kentucky's coal (Clay 13,073 dug / 4,813 not; Pike 410 / 15,000;
  Perry 42 / 88), Hibbing's iron (1,539 / 3,744), and forest cleared for building at Lakewood Ranch (15,000 /
  14,592), Four Corners (2,927 / 7,861), Huntsville (1,046 / 3,146) and New Albany (457 / 937).
- **Far away:** the ten clearcut regions of section 8 (cut ground) and the 26 satellite-laser sites (moved and
  unchanged labels).
- **Near each mine:** a 6 km square 15 to 60 km from each of eleven abroad mines, picked from Hansen's loss in one
  year 2018 to 2023, cut ground against forest kept. Checked by eye against Sentinel-2 (`results/near/`): Kolwezi's
  is a grid of cleared blocks, Madre de Dios's small farm clearings, Boke, Detour Lake and Morowali clearing; Fort
  Hills's is a boreal burn, not a cut; Mobimbi and Sembehun picked the same square; Sangatta's and Weda Bay's hold
  active mining in the square, and Adaro's loss is mostly degraded forest still standing. Those three are suspect.

Every year 2017 to 2025 of both embeddings is read under the kept pixels, with the mean embedding around each pixel
within about 110 m and 510 m. Feature sets: **pair** (the map's own features at the two years), **ctx** (the
surroundings at the two years: the change there and what it became), **after** (the 2025 embedding against the
later year: what the ground became since), **traj** (nine numbers per embedding over every year: distance from the
first year, distance kept at the end, the year it went farthest, the surroundings' change), **all** together. A label
needs a valid year after its pair, so labels whose later year is 2025 drop out (Salonga and Pomalaa entirely). The
baseline is the aggregate itself (joint model x radar-safe gate, abroad recipe), held out: spans.py's leave-one-out
models at the five US digging sites, abroad_teach.py's saved held-out scores at the older satellite sites, and the
final models where they never taught (Kentucky, the clearcut regions, forest_labels.py's mines, the squares near
mines).

### A. Same place: one lidar site held out at a time

| AUC, cleared and dug vs cleared and not dug | aggregate | pair | ctx | after | traj | all |
|---|---|---|---|---|---|---|
| ky-clay | .915 | .924 | .858 | .934 | .977 | .910 |
| ky-pike | .779 | .781 | .795 | .804 | .634 | .839 |
| ky-perry (130 labels) | .898 | .934 | .904 | .943 | .916 | .936 |
| hibbing-mn | .953 | .946 | .976 | .878 | .917 | .944 |
| lakewood-ranch-fl | .737 | .608 | .424 | .575 | .730 | .573 |
| four-corners-fl | .399 | .247 | .188 | .496 | .478 | .208 |
| huntsville-al | .722 | .658 | .690 | .865 | .655 | .773 |
| new-albany-oh | .732 | .739 | .642 | .583 | .598 | .698 |
| mean (7 sites, Perry left out) | .748 | .700 | .653 | .734 | .713 | .706 |

- At the mines in forest the aggregate already orders the two well (Clay .915, Hibbing .953), and a head taught only
  on this question adds a little at some (the path over the years at Clay .977, the years after at Huntsville .865).
- Taught on these eight sites alone, no head beats the aggregate on average. Four Corners is inverted under every
  rule: its cleared-not-dug ground scores above its dug ground, the in-site order problem of section 18.

### B. Taught on the US lidar alone, scored abroad

| head | clearcut regions called dug | Kolwezi moved / unchanged called dug | Kolwezi moved vs all cut ground, AUC |
|---|---|---|---|
| aggregate (at the 80% point of held-out lidar dug) | .237 | .43 / .00 | .65 |
| pair | .635 | .67 / .04 | .59 |
| ctx | .179 | .85 / .79 | .92 |
| after | .589 | .99 / .63 | .92 |
| all | .155 | .75 / .09 | .89 |

- The surroundings carry from Kentucky and Minnesota to the Congo: the ctx head sets mine digging apart from the
  clearcut regions (Kolwezi .92, Detour Lake .95, Madre de Dios .97, Fort Hills .91) where the aggregate does not
  (.65, .46, .58, .63). It does it by the district: it calls the mines' unchanged ground dug too (Kolwezi .79). It
  says "mining around here", not "this pixel was dug".
- The pair features alone, taught on the US lidar, call two thirds of the clearcut regions dug: what US forest
  clearing looks like before digging does not carry to Nordic or southern hemisphere forestry.

### C. Everything taught, each place held out

With the clearcut regions and the satellite sites taught too (cut ground and unchanged as not dug, moved as dug),
every head sets each mine's digging apart from the held-out clearcut regions almost perfectly (AUC .94 to 1.00 under
all, Sangatta .97, Sokolov .94); the held-out clearcut regions score a mean .01 to .05 under all. That is too clean
to read as mine against cut: the clearcut regions are temperate forestry country and the mines are not, and
AlphaEarth carries place strongly (section 1's note). Part D is the fair test. Inside the lidar sites the all head
lands near the aggregate (Clay .931, Hibbing .956, Pike .795, Huntsville .816, Four Corners .255).

### D. Each mine against clearing near it

Each mine's moved labels against the cut ground of its own nearby square, the mine and its square both held out.
"Head" taught without any square near a mine (as C) or with the other mines' squares too. The map's view: the share
of the mine's moved labels and of the nearby cut ground the aggregate lights at its 60% cutoff, alone and where the
head also says dug (over one half).

| mine (km to square) | moved labels | aggregate AUC | aggregate: found / nearby cut lit | all head AUC (without / with) | aggregate x all head (without) | aggregate x ctx head (without) |
|---|---|---|---|---|---|---|
| Kolwezi (47) | 323 | .676 | .47 / .12 | .97 / .99 | .45 / .00 | .43 / .00 |
| Fort Hills (56, a burn) | 6,482 | .950 | .34 / .00 | .99 / 1.00 | .19 / .00 | .20 / .00 |
| Detour Lake (48) | 144 | .774 | .13 / .00 | .98 / .99 | .12 / .00 | .11 / .00 |
| Mobimbi (21) | 48 | .787 | .48 / .00 | .84 / .98 | .48 / .00 | .48 / .00 |
| Sembehun (34, Mobimbi's square) | 84 | .908 | .11 / .00 | .99 / .99 | .11 / .00 | .08 / .00 |
| Madre de Dios (38) | 11 | .083 | .09 / .89 | .88 / 1.00 | .09 / .05 | .09 / .00 |
| Adaro (35, suspect) | 13 | .557 | .61 / .55 | .99 / 1.00 | .31 / .01 | .46 / .01 |
| Sangatta (38, suspect) | 11 | .408 | .46 / .71 | .89 / .97 | .46 / .00 | .46 / .00 |

- Held out by place, near the mine, in the same forest and years, the heads set digging apart from clearing (AUC .84
  to 1.00 against the aggregate's .08 to .95). At Kolwezi the aggregate times the all head keeps nearly all its found
  (.47 to .45) and its nearby clearing goes from .12 lit to none. That is the trade sections 9, 14 and 15 could not
  make by teaching clearcuts as quiet (every variant took Kolwezi's found down in step).
- Madre de Dios's nearby farm clearings light .89 under the aggregate, the clearing-without-digging case of section
  15, and go dark under every head. The aggregate finds little of the mine there either (.09); the head does not
  change that.
- It is not free everywhere: Fort Hills's found falls .34 to .19 and Adaro's .61 to .31 where the head also has to
  agree. Weda Bay and Morowali had too few moved labels left to score.
- The ctx head needs no later year (only the two years of the pair and their surroundings), so it fits a map whose
  later year is 2025; the after and traj heads need a year after the pair.
- Caveats: one square per mine, picked by Hansen loss and checked by eye, three of them suspect and one a burn; tens
  of moved labels at most tropical mines; the heads are not yet tested on the 14 US digging sites as a map rule
  (they need the surroundings read at every digging pixel), so their cost to digging AP there is unmeasured.

Files: `separate.py`, `separate_test.py`, `near_panels.py`, `results/separate.json`, `results/near/`, logs
`results_separate_*.log`; data in `data/separate/` (4.2 GB).

## 20. Reading TESSERA in the browser (Pyodide)

`pyodide_probe/`: Pyodide 314.0.7 in Node (the same Pyodide line marimo 0.25.1 bundles, 314.0.0, and the abi
xarray-sql 0.6.0's wasm wheel targets). TESSERA's sharded Zarr (blosc zstd with bitshuffle, crc32c index) decodes in
Pyodide with its own packages. zarr-python reads each 32 x 32 inner chunk as its own request, one after another: a
512 x 512 window of one year took 124 s (63 s through the async API, 549 requests). A small reader that reads the
shard index, merges neighboring byte ranges and fetches them together took 4 to 6 s (13 requests, 31 MB, decode 0.1
s), byte-identical to zarr's. The network is then the limit, about 1.25 MB per square km per year of TESSERA.

AlphaEarth's own COGs on Source Cooperative (`tge-labs/aef`, what aef-loader reads) are in UTM at 10 m, band
interleaved, 1024 x 1024 tiles, and their pixel edges fall on TESSERA's grid: the Huntsville tile sits 33,616 and
548,864 whole pixels from TESSERA's zone origin. Read from the COGs instead of the lat/lon mosaic, the two embeddings
pair pixel for pixel, with no resampling. The models here were taught on the mosaic, so moving to the COGs means
reading the sites again.

## 21. The aggregate in Pyodide through xarray-sql and DuckDB

`pyodide_probe/both_reader.py`, `sql_probe.py`, `run_sql.mjs`: Pyodide 314.0.7 in Node, xarray-sql 0.6.0's wasm wheel
(4.3 MB) with Pyodide's own DuckDB 1.5.1, pyarrow 22 and xarray. DataFusion has no wasm build, so xarray-sql's
`XarrayContext` is absent there; `xql.register(duckdb_connection, name, ds)` is the path, and it worked as installed.
One window of 1024 x 1024 pixels (10.24 km, 105 square km) at Huntsville, 2019 against 2023, both embeddings read
from their own stores on one UTM grid: AlphaEarth's native COGs (`tge-labs/aef`) and TESSERA's Zarr. The reader
matches rasterio and zarr-python byte for byte, and the two windows share their corner to the meter.

| step (Pyodide, Node, one thread) | time | note |
|---|---|---|
| Pyodide, numpy, pandas, pyarrow, xarray, DuckDB, xarray-sql loaded | 5 s | about 41 MB of downloads, cached |
| AEF, two years | 7.6 s | 85 MB, 130 requests (one 1024 px tile per band per year) |
| TESSERA, two years, with scales and radar counts | 20.7 s | 252 MB, 76 requests |
| each embedding's half of the joint model, numpy | 3.6 s | |
| register both as tables | 0.04 s | lazy |
| pixel join (1,048,576 pixels) | 0.5 s | |
| A: join, gate, score, peak of each 32 x 32 block (GROUP BY) | 2.4 s | 2.2 s on DuckDB's own tables; numpy 0.06 s |
| B: the joint model itself in SQL, 256 x 256 px | 3.5 s | 12.6M rows (pixel x band), 3.6M rows a second |
| B, 512 x 512 px | 13.6 s | wasm heap 3.4 GB, near wasm32's 4 GB |

- The joint model is linear on each embedding's own features, so its logit splits into an AEF half and a TESSERA half.
  Rule A has numpy score each half per pixel; SQL joins the halves on the pixel, adds the bias, applies the
  radar-safe gate (AEF's alone where TESSERA's radar came or went), falls back to the AEF-alone model where TESSERA has
  no pixel (a LEFT JOIN), and carries the peak as `GROUP BY y // 32, x // 32` with `max(score)`. It matches numpy to
  1e-7.
- The 32 x 32 blocks are TESSERA's inner chunks on the zone grid (320 m, `coarse.py`'s cell). A window not starting on
  a block edge spans 33.
- Rule B (the dot product as a join of each pixel's unit vector to the weights by band, then two GROUP BYs and the
  pixel join) matches numpy to 2e-6 but cannot hold a full window in wasm32 memory at once; it would have to run in
  strips.
- xarray-sql adds little over DuckDB itself: A took 2.4 s through the registered xarray tables against 2.2 s once
  copied into DuckDB. The time is DuckDB's, single threaded in wasm.
- In this window 4.5% of pixels pass the 60% cutoff and 36% of 320 m blocks carry a pixel over it (Huntsville is a
  building site). A block's cutoff for the map wants setting on blocks, as `coarse.py` did, not on pixels.
- numcodecs 0.15.1's zstd fails on GDAL's frames in wasm32 (no content size in the frame header, read as 4 GB);
  Python 3.14's own `compression.zstd` decodes them. TESSERA's blosc decodes through numcodecs fine.


### The model as a function inside the SQL

`pyodide_probe/udf_probe.py`: the same window and rule, with the model as a vectorized Python function DuckDB calls on
each batch (`create_function(..., type="arrow")`), the raw int8 bands registered through xarray-sql. It matches numpy
to 2e-6 every way. What costs is how the bands reach the function, not the math:

| how the bands reach the function | full window |
|---|---|
| one column per band (128 AEF, 256 TESSERA) | 40 s; a function that does nothing with 256 columns takes 17 s by itself |
| the same columns gathered into one list inside DuckDB | 21 s to build the lists alone |
| each pixel's bands packed into one bytes value (a BLOB), one argument | 7.3 s, plus 2.6 s to pack |
| the pixel's (y, x) only, the function reading the bands from memory | 8.4 s |
| numpy alone, no SQL | 3.8 s |

- DuckDB calls the function 512 times per million pixels (2,048 rows a call) and hands over each column as its own
  array each time. Scanning all 256 TESSERA columns through xarray-sql without a function takes 0.65 s.
- Packing: a zero byte cuts a value short on its way into Arrow, so each value goes in as value + 128. TESSERA keeps to
  -127 .. 127; AlphaEarth's -128 marks a masked pixel, carried in its own column.
- AlphaEarth's dequantization as a 256-entry lookup table, as aef-loader does it.

### The COGs score as the mosaic did

`cog_vs_mosaic.py`, `results/cog_vs_mosaic.json`: at every mosaic pixel center of the 14 digging sites, the COG pixel
under it, both years, scored by the abroad recipe's final models (in sample, so only the difference counts). Median
cosine between the two AEF vectors .9995 (lowest site .9979, Mountain Pass); joint x radar-safe gate AP .654 from the
mosaic, .655 from the COGs; found at the 60% cutoff .685 against .686; unchanged lit .026 both; rank correlation of the
scores .99. The models taught on the mosaic carry to the COGs, so the browser can read AlphaEarth on TESSERA's grid.

### gpudb

The DuckDB community extension gpudb (Metal on Apple Silicon, CUDA on Linux) loads in native DuckDB 1.5.6 on an M3
Max. It has no WebAssembly build, so it cannot run in Pyodide. Its functions are GPU sums, minimums, maximums,
group-bys and joins on columns uploaded to the device; nothing for the per-pixel model math.

## 22. A crawl in the browser: keeping under 4 GB

`pyodide_probe/crawl.py` (runs through `crawl_runs.sh`), `browser_test.py`, `browser_test.json`. No kernel: Pyodide
does the reading and the scoring. The design: one AEF COG tile (1024 x 1024 px, 105 square km) of raw embeddings in
memory at a time, decoded straight into pixel-major byte buffers allocated once and refilled (value + 128, the form the
SQL function reads, so there is no second packed copy), scored through xarray-sql and DuckDB, then reduced to what is
kept: a byte per pixel for the most recent tiles (an LRU cache) and the highest score of each 32 x 32 block in a DuckDB
table (`peaks`) for everything crawled. Byte ranges go out 16 at a time per array, to bound the compressed bytes held.

**The crawl, 3 x 3 tiles around Huntsville (944 square km), 2019 against 2023, buffers reused:**

| | |
|---|---|
| wasm heap | 708 MB loaded with buffers, 1,201 after the first tile, 1,325 after the second, then flat to the ninth |
| per tile | 30.7 s reading (about 325 MB, 200 requests), 5.8 s scoring |
| per square km | 0.35 s, 3.2 MB |
| pixel scores kept | 1 MB per tile; panning back to a cached tile is instant |
| peaks | 9,312 blocks, 40 KB as Parquet (about 4.3 KB per tile), written in 0.02 s |

The first tile's share lit (.0449 at the 60% cutoff) matches section 21's window exactly.

**The other choices** (the same ground, one run each):

| run | heap | per tile | per square km |
|---|---|---|---|
| 1024 px, buffers reused (above) | 1,201, then 1,325 MB flat | 30.7 s read, 5.8 s score | 0.35 s, 3.2 MB |
| 1024 px, new buffers each tile | 1,203, 1,444 after the second, 1,596 after the seventh | 31.1 s read, 6.0 s score | 0.35 s, 3.2 MB |
| 512 px over 2 x 2 COG tiles (16 tiles, 419 square km) | 480, then 576 MB flat | 13.4 s read (about 148 MB), 1.4 s score | 0.57 s, 5.7 MB |
| 1024 px, cache of 2 tiles (2 x 2, 419 square km) | 1,201, then 1,322 MB flat | 30.3 s read, 5.9 s score | 0.35 s, 3.2 MB |

- Reusing the buffers keeps the heap flat. Without reuse it steps up (1,203, then 1,444, then 1,596 MB at the seventh
  tile): a wasm heap never gives memory back, and fresh tile-sized arrays do not always fit the holes the last ones
  left, so a long crawl without reuse would keep creeping toward 4 GB. Speed is the same either way.
- 512 px tiles halve the heap (576 MB against 1,325) but cost 1.6 times the time and 1.8 times the bytes per square
  km: AEF's COG is stored in 1024 px internal tiles, so each 512 tile pulls its whole COG tile again (four times
  over). The share lit over the 2 x 2 is the same either way (.0517), so the scores do not depend on the tile size.
- Panning back to a tile the cache has dropped is a full re-read: 35.1 s (29.4 read, 5.7 score, 324 MB), against
  nothing for a cached tile. A cached tile's pixel scores are 1 MB, so a cache of 16 (16 MB) costs little.
- The first no-reuse run stopped on the seventh tile: Source Cooperative answered one range request with a 503 and an empty
  body. The same reads had all succeeded ten minutes earlier, and tile 4 of that run read slowly (53 s against about
  30), so the service was struggling for a moment. `both_reader.get` now retries 429, 5xx and failed fetches up to six
  tries with jittered backoff (about 15 s in all, or the server's Retry-After); `retry_check.py` checks it in
  Pyodide against scripted 503s, a 429, a failed fetch, a 403 (not retried) and a 404. Each tile's line in the crawl
  now counts its retries. The no-reuse rerun with it finished all nine tiles; it needed one retry, on the first tile.

**Browsers** (Playwright, headless, a fresh profile; Pyodide 314.0.7 from the CDN):

| browser | WebAssembly memory | Pyodide heap at its limit | storage quota | 1,000 tiles of peaks saved / read after a reload |
|---|---|---|---|---|
| Chromium 156 | 4,228 MB | 4,239 MB | 10,742 MB | 1.3 s / 0.14 s |
| Brave 155 (Chromium) | 4,228 MB | 4,239 MB | 2,147 MB | 1.3 s / 0.14 s |
| WebKit 27.2 (Safari's engine, on macOS) | 4,228 MB | 4,239 MB | 1,049 MB | 2.2 s / 0.19 s |
| Firefox 157 | 4,228 MB | 4,239 MB | 10,737 MB | 2.5 s / 0.15 s |

- Every engine gives a page the full 4 GB of wasm32, and Pyodide can fill it. The crawl's 1.3 GB leaves about 2.9 GB.
- The peaks persist in the browser's own storage (IndexedDB, through Pyodide's IDBFS mount; DuckDB writes and reads
  Parquet there). 1,000 tiles (about 10,500 square km) of random peaks take 4.6 MB; real ones took 4.3 KB per tile.
  The lower 48 (about 77,000 tiles) would be about 330 MB as now stored; as one byte per block about 80 MB.
- Not tested: Safari on iPhone or iPad (Playwright's WebKit is the desktop build), and Brave with "forget me when I
  close this site" on, which would clear the peaks.

## 23. The notebook, in a browser tab

`notebook/earthwork_crawl.py`, exported with `marimo export html-wasm` and driven headless in Chromium and Brave
(Playwright). The crawl runs as a background asyncio task beside the map: the map sends its view, Python sends each
tile's pixel scores as one 1 MB message, and neither waits on the other.

| | |
|---|---|
| page loaded to first tile read | 11 s (marimo, Pyodide, pandas, pyarrow, xarray, DuckDB, xarray-sql from PyPI) |
| per tile (Huntsville, 2019 to 2023) | 20 to 23 s reading, 5.6 to 6.0 s scoring: faster than under Node (30 s) |
| first tile's share lit | .0451, as under Node (.0449; the notebook counts at the cutoff's byte) |
| wasm heap | 1,276 after the first tile, 1,385 MB after the second, then flat |
| reload | the block peaks come back from IndexedDB, in Chromium and in Brave |

- marimo's WebAssembly mode refuses `multiprocessing.Lock`, which numcodecs' Blosc takes on first use: the notebook
  hands it a thread lock.
- A failed read now cancels its sibling reads; left running, they piled up under the retry (the heap climbed 200 MB
  a try and the browser began failing fetches).
- MapLibre 5.24 draws an image layer's transparent pixels black under `raster-resampling: nearest`: the layers smooth,
  and the 320 m blocks are drawn 4 times larger so each stays a square.
- Zoomed out, the blocks look busy (section 21: a third of blocks are lit at the pixel cutoff). The cutoff slider
  moves both; a cutoff of the blocks' own is still open.

## 24. Back to a kernel: the map with three layers, and where TESSERA can be read

The browser-only notebook (section 23) worked but was too slow to use: it reads every 10 m pixel of both
embeddings, so a 105 square km tile takes about 30 s. The map went back to aef-earthwork's `earthwork.py`
(kernel, H3 hexagons carrying their peak, Earth Genome's Sentinel-2 under a held space bar), copied as
`earthwork_map.py`, with a TESSERA reader (obstore range reads of the shard index and the inner chunks under the
view, decoded chunks kept) and three layers on M: Aggregate (the radar-safe either rule), AEF (gated) and TESSERA.
`models/model_abroad.npz` holds the abroad recipe and each layer's cutoff at 60% of held-out digging found
(aggregate .78, AEF .60, TESSERA .80). A slider hides hexagons under the cutoff, per layer, each layer's own
cutoff on double-click; ramps cividis (aggregate), viridis (AEF), plasma (TESSERA).

| Huntsville, zoom 13.5 (about 40 square km) | |
|---|---|
| AlphaEarth, each year (10 m mosaic) | 4.4 to 4.5 s |
| TESSERA, both years (0.44 Mpx) | 6.4 to 8.1 s, beside AlphaEarth's reads |
| frame (three layers scored, peaks carried) | 0.2 s |
| the same ground again | 0.02 s (chunks kept) |

**TESSERA can only be read at 10 m.** AlphaEarth publishes overviews (each coarser pixel the mean of the 10 m
under it), which is how the map draws zoomed out; TESSERA v1.1 publishes none (only `global_rgb`, a 3-band color
preview the models cannot read). Zoomed out, every 10 m pixel under the view would have to be read: about
1.25 MB per square km a year, so about 400 MB at zoom 12.5, 1.5 GB at 11.5, 12 GB at 10. The map reads TESSERA
from zoom 13.2 (where AlphaEarth's reads turn 10 m too) and opens at 13.3.

**The first version was wrong about this.** Below zoom 13.2 its Aggregate layer drew AlphaEarth alone under the
aggregate's name, and it opened at zoom 12.7, so the first view it showed was not an aggregate. An aggregate
needs both embeddings read at the same level; where TESSERA is not read it now draws nothing. The zoomed-out AlphaEarth view has a milder version of the same issue: the 10 m model is run on
overview means it was never taught on.

**Searched for a zoomed-out TESSERA (October 2026).** None published: v1.1 on Source Cooperative and dClimate's
copy on AWS are 10 m only. TESSERA v2's beta store (`zarr/v2-2B-L~beta2`, UTM zones 21, 22, 32, 33 only) adds
`embeddings_d16` and `embeddings_d4`, the first 16 and 4 dimensions as their own arrays (Matryoshka; the v2 paper
keeps 92% of the 128-d performance at 16), still 10 m but an eighth and a thirty-second of the bytes. No US yet,
and the models would need teaching again on v2.

**Teaching TESSERA to the AlphaEarth model: it does not carry over.** So the map can read one embedding at every zoom:
`distill.py` teaches AlphaEarth-only students, the map's model unchanged in size, with TESSERA's help (the joint
model or the aggregate as a teacher, blended with the lidar labels or alone; the aggregate's score on the
teaching sites' unlabeled band; lidar moved pixels dropped where neither embedding changed). Held out as
abroad_teach.py holds out; fields stay test only.

| at 60% caught | AP | dig sites' unchanged lit | burn | crop | farm | marsh | storm | notruth | island found | space found | desert |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base (today's AEF model) | .543 | .055 | .260 | .212 | .094 | .046 | .113 | .077 | .43 | .31 | .032 |
| soft_both_50 | .537 | .058 | .262 | .211 | .100 | .048 | .143 | .080 | .42 | .33 | .031 |
| soft_both_0 | .533 | .060 | .279 | .217 | .105 | .049 | .155 | .082 | .42 | .32 | .031 |
| soft_agg_50 | .532 | .063 | .255 | .286 | .126 | .054 | .154 | .080 | .43 | .38 | .023 |
| soft_agg_0 | .528 | .068 | .236 | .333 | .135 | .055 | .179 | .084 | .43 | .41 | .018 |
| soft_agg_50_unl | .531 | .064 | .262 | .268 | .121 | .062 | .169 | .079 | .42 | .37 | .021 |
| clean | .543 | .055 | .246 | .196 | .086 | .041 | .107 | .059 | .42 | .33 | .025 |
| clean_soft_agg_50 | .535 | .061 | .248 | .266 | .122 | .050 | .154 | .073 | .43 | .36 | .019 |
| aggregate (the teacher, both embeddings) | .582 | .039 | .190 | .038 | .012 | .023 | .099 | .075 | .50 | .42 | .017 |

- No student comes near the teacher on fields. The aggregate lights .038 of the crop switches and .012 of the
  farms' unchanged ground; every student stays at .19 to .33 and .09 to .14, where base is.
- Every soft label makes it worse: AP falls .006 to .015 below base, and crop, farm and storm ground light more
  (soft_agg_0: crop .212 to .333). Taught to copy the teacher's scores, the student copies how much change it
  saw, not which kind: what lets the aggregate say "field, not dig" is in TESSERA, and AlphaEarth's two years,
  read linearly, do not hold it.
- The soft labels do help where the aggregate is better for reasons AlphaEarth can see: satellite digging found
  .31 to .38 to .41 and desert lit .032 to .018, in step with how much of the teacher is in the label.
- `clean` (moved pixels where neither embedding changed left out) is the only one at no cost: AP .543, every
  quiet set a little lower (crop .212 to .196, burn .260 to .246, ground with no truth .077 to .059, desert
  .032 to .025). Small, and the same at 50% and 70% caught.
- The same order at 50% and 70% caught (`distill.py report 0.5`, `report 0.7`).

Files: `distill.py`, `results_distill.log`, scores in `data/scores-distill/<variant>/`. The run took 43 minutes,
the main process at most 17.2 GB with one child scoring a site beside it.

**Zoomed out, the map moves to AEF.** Past zoom 13.2 (where TESSERA stops being read) the map switches to AEF on its
own and grays the Aggregate and TESSERA out (a click or M there says to zoom in); zoomed back in, the layer it moved
from comes back, unless another was picked meanwhile. Checked headless in Chromium across 13.2 both ways.

## 25. Dense cities

On the map, Manhattan's Lower East Side from 2022 to 2025 lit block after block under TESSERA alone; AEF and the
aggregate lit East River Park (torn up and raised on fill for flood protection from late 2021) and a few spots. TESSERA's
own metadata there: about 30 ascending radar looks a year from 2017 to 2025, no descending looks until 2025 (17,
when Sentinel-1C began), and both in 2020. The radar-safe rule only catches radar coming or going, not a second look
direction added.

`cities.py` takes 3 km squares in 22 dense cities (14 US, 8 abroad), every year pair 2017 to 2025, scored by the
abroad recipe's final models at each rule's 60% cutoff, pixels grouped by their radar in the two years. Few city
blocks are dug in a few years, so the share lit is close to a false-alarm rate. Mean over cities of the share lit:

| | US, same looks | US, a direction added or dropped | abroad, same looks | abroad, added or dropped |
|---|---|---|---|---|
| AEF | .007 | .017 | .000 | .003 |
| TESSERA | .075 | .085 | .018 | .011 |
| aggregate | .008 | .013 | .001 | .001 |

- TESSERA's city noise is mostly dense US blocks whatever the looks: .075 with the same looks both years, .085
  with a direction added. The cities abroad (London, Paris, Berlin, Tokyo and others) have both directions every
  year and light a quarter as much. Whether one look direction is what makes US blocks noisy is a reading, not
  tested here.
- The aggregate mostly does not inherit it (.008 and .013 in US cities, beside AEF's .007 and .017).
- The Lower East Side is where the 2025 radar does show: the aggregate lights .036 for 2022 to 2025 against .010
  for 2022 to 2024.
- One US city had radar in one year only: TESSERA alone lit .97 of it, the aggregate none (the radar-safe rule).

Corrections (each applied the same way at the cities and at the 14 digging sites; the final models were taught on
those sites, so their numbers run high and only the change counts):

| correction | US city lit, same / added (aggregate) | sites: moved found | unchanged lit | AP |
|---|---|---|---|---|
| none | .008 / .013 | .685 | .026 | .654 |
| shift (TESSERA less the square's median change) | .008 / .014 | .695 | .038 | .647 |
| shift_both (the same for AlphaEarth too) | .008 / .013 | .671 | .036 | .638 |
| pcs_2 (two main directions of look change removed) | .007 / .012 | .678 | .035 | .644 |
| pcs_8 (eight) | .004 / .008 | .640 | .029 | .634 |

No correction is free. The median shift makes TESSERA alone noisier in cities (.075 to .097) and lights more
unchanged ground at the sites. Removing eight directions of look change halves the aggregate's city lit, at a cost
at the sites: moved found .685 to .640, AP .654 to .634, and a little more unchanged ground lit. Two directions do
almost nothing. Lower East Side, 2022 to 2025, aggregate: .036 with none, .020 with pcs_8.

## 26. White roofs

New York coats roofs white (CoolRoofs since 2010; Local Law 94 since 2020 asks for a cool, green or solar roof on
replacement). Nothing moves, but dark to white is one of the largest brightness changes a 10 m pixel makes, and the
lidar never taught a repainted roof. No dated list of coated roofs exists after 2012, so `whiteroofs.py` finds roofs
that went dark to white (and white to dark) in Earth Genome's Sentinel-2 yearly mosaics, 2022 to 2024 (the same
radar looks both years at New York), on footprints built before 2015, in seven New York squares, and counts what
each layer lights at the 60% cutoffs.

Pooled over the seven squares and three pairs (2022 to 2023, 2023 to 2024, 2022 to 2024), share of pixels lit:

| layer | whitened roofs | darkened roofs | steady roofs | off roofs |
|---|---|---|---|---|
| AEF | .101 | .007 | .001 | .004 |
| TESSERA | .226 | .166 | .054 | .077 |
| aggregate | .059 | .018 | .001 | .005 |

Whitened roof pixels: 274, 369 and 704 for the three pairs (a few hundred buildings in all), so these are small
counts. A white coat lights all three layers. AEF goes from almost nothing on steady roofs to a tenth of whitened
ones, and the aggregate from .001 to .06. TESSERA lights a twentieth of steady roofs anyway (section 25's city
noise) and about four times that on whitened ones. Darkened roofs barely light AEF or the aggregate. Most of it is
in a few places: Hunts Point's warehouses, 2023 to 2024, lit AEF on 67% of whitened pixels (52 px on 12
buildings) and the aggregate on 23%; East New York 2022 to 2024 lit about a fifth to a quarter under each layer.
Midtown's whitened roofs lit TESSERA (.32 for 2022 to 2024) and neither AEF nor the aggregate.

## 27. The height in the embeddings

AlphaEarth was taught to reconstruct the Copernicus DEM and GEDI canopy height, so its embedding carries a height.
Earthwork's relief notebook reads it back as the ghost: a ridge regression from each pixel's embedding (values and
squares) to the real elevation, each pixel predicted out of fold over 1 km blocks. `heights.py` fits one ghost per
place on every year at once against a static DEM and predicts each year, so the ghost's change between two years is
the height change the embedding implies. Two DEMs: Mapterhorn's tiles (as the notebook), and 3DEP from its S3
bucket (each site's first-flight 1 m DEM, the seamless 1/3 arc-second DEM in New York). They give the same numbers;
3DEP's, mean over the 14 digging sites:

| | AlphaEarth ghost | TESSERA ghost |
|---|---|---|
| fit to the terrain, R2 | .33 to .91 by site | lower |
| AUC of the ghost's change size, moved against unchanged | .643 | .641 |
| AUC of plain change (1 - cosine), for comparison | .852 | .860 |
| correlation of the ghost's change with dz | .03 | .01 |
| sign right on moved ground (cut or fill) | .52 | .51 |
| median ghost change, moved / unchanged | 12.9 m / 7.8 m | 20.1 m / 11.6 m |

The Copernicus DEM GLO-30 (a surface model, buildings and trees in it, and the one AlphaEarth was taught to
reconstruct) gives the same at the sites: AUC .629, correlation .02, sign .52. So the target is not why.

The median moved ground at these sites moved 0.8 to 2 m (8.6 m at Mountain Pass, 4.6 m at Hibbing). The ghost
wobbles by meters to tens of meters a year on ground that did not move, so it does not see meter-scale digging: its
change is larger on moved ground only because the embedding changed there, and its direction is a coin flip. On New
York's roofs it moves about 1 m whether the roof was whitened, darkened or left alone, and as much off the roofs: it
says nothing moved at the white roofs, and says much the same on dug ground. Fitted to Copernicus, whose buildings
stand in the surface, the ghost moves 3.4 m on whitened roofs, 3.3 m on darkened ones and 2.8 m on steady ones
(2022 to 2024): no more separation than against the bare earth.

The other way round (`heights.py head`): the lidar's own meters taught from the embedding pair, as the Earthwork
models take it (before, after, their product, their squared difference), signed dz clipped at 15 m, each of the 14
digging sites held out in turn (35,000 pixels a site: moved, unchanged and in between). Means over the sites:

| head | correlation with dz, moved | sign right, moved (cut or fill) | AUC of its size | plain change's AUC | median size, moved / unchanged |
|---|---|---|---|---|---|
| ridge, AlphaEarth | .18 | .54 | .581 | .859 | 1.39 / 1.08 m |
| ridge, TESSERA | .05 | .53 | .552 | .860 | 1.00 / 0.87 m |
| ridge, both | .14 | .55 | .601 | .859 | 1.48 / 1.11 m |
| boosted trees, both | .14 | .55 | .716 | .859 | 0.51 / 0.19 m |

Taught on the meters themselves, the pair still does not carry them to a new site: cut against fill is near a coin
flip at every site (.43 to .68), and how much is barely there (correlation .14 to .18; Lakewood Ranch .54 with the
trees, its sign .47). Even as a detector, the size of the predicted change separates moved from unchanged worse than
plain change does. The embeddings say the ground changed, not how far or which way.

## 28. A gate for ground something stood on both years

The flat ground reading of the built-up work (embeddings-built-up, height_ground.py: at New Capital, held out, AUC
.996) says from AlphaEarth alone whether a structure stands on a pixel, any year. `structures.py` teaches it here:
Overture building footprints (2026-09-23.0) on AlphaEarth 2025 in the 22 city squares and four more in New York, a
pixel center under a footprint a structure, a pixel with no footprint on it or beside it ground; gradient boosting on
the pixel and its 3 x 3 mean, classes balanced, held out by fold (New York's seven squares one fold). The gate: hard,
a pixel standing (50% or more) both years of the pair is not lit; soft, the score times one less the smaller chance.

Held out, the reading is weaker than at New Capital (one year of teaching, Overture's footprints, dense blocks where
ground is a street beside a building): AUC .81 to .90 by fold, structures found .66 to .92, ground flat .70 to .85.

| | no gate | hard | soft |
|---|---|---|---|
| New York whitened roofs lit, aggregate | .059 | .007 | .000 |
| New York whitened roofs lit, AEF / TESSERA | .101 / .226 | .027 / .022 | .003 / .001 |
| US city pixels lit, same looks / added, aggregate | .008 / .013 | .006 / .010 | .004 / .007 |
| US city pixels lit, same looks / added, TESSERA | .075 / .085 | .035 / .047 | .014 / .021 |
| Lower East Side 2022 to 2025, aggregate | .036 | .029 | .025 |
| 14 digging sites, aggregate: moved found | .685 | .683 | .653 |
| unchanged lit | .026 | .026 | .024 |
| AP | .654 | .650 | .633 |

90% of the whitened roofs read standing both years, as do 88% of steady roofs and 58% of the pixels off roofs in
New York's squares. At the digging sites 1.2% of moved ground reads standing both years (2.1% before, 6.9% after:
the new buildings) and 1.8% of unchanged ground. So the hard gate costs almost nothing at the sites and takes the
white roofs out and halves TESSERA's city noise. The soft gate costs real digging in the desert and at the mines
(Las Vegas northwest found .59 to .42, Mountain Pass .69 to .57): the reading calls some bright desert and pit
ground half standing.

On the map (`G`, on to start): the reading has to run on every 10 m pixel the map reads, so a smaller model (100
trees of 31 leaves, 0.74 s a million pixels against 1.9) was tested the same way: hard gate, aggregate, whitened
roofs lit .011, US city pixels .006 / .011, sites moved found .684, unlit .026, AP .652, about the full model's. A
logistic regression, though as good held out in the cities (AUC .850 against .856), reads 28% of the sites' moved
ground and 32% of their unchanged ground as standing both years (the trees: under 2%), and the hard gate would take
moved found from .685 to .532: it does not carry from city blocks to suburbs, desert and pits. The map reads the
small trees (pickled, scikit-learn pinned) where it reads the 10 m mosaic, from zoom 13.2, and a finer cell that
stands both years scores 0 in every layer. At Wildwood 2% of finer cells stood both years; on the Lower East Side,
47%, and the scattered block hexagons go while East River Park stays. The read adds about a second a year.

## 29. Repeat lidar abroad: does the score mean the same thing everywhere?

London lit nothing on the map (Brent Cross, Earls Court) while Frisco looked right, and the abroad tests so far
(sections 4, 5) said the ranking holds abroad but the score's level drops. This section builds truth to test that
directly: open repeat lidar outside the US, built the way Earthwork built the 3DEP sites (`world_sites.py`), then the
map's model on it, then a model taught with it, each country held out.

**The sites.** Two bare-earth DEMs, the first flown 2017 or later, the second by 2024, read on one 2 m grid, second
minus first with the median removed, each AlphaEarth pixel moved (mean |dz| > 0.5 m) or unchanged (< 0.1 m); a 4 km
square about a dug place (some 2 to 6 km), plus quiet squares. One module per source (`world_<source>.py`):

| Country | Source | Sites | Flights | Moved px |
|---|---|---|---|---|
| Netherlands | AHN4 against AHN5 (one AHN3 against AHN4), PDOK and AHN COGs | 13 | 2019-2021 / 2022-2024 | 105,041 |
| England | EA National LIDAR Programme, 1 m tiles (whole zips) | 21 | 2018-2020 / 2021-2023 | 120,801 with Scotland |
| Scotland | SRSP Phase 4 / 5 and 6, 0.5 m COGs | 11 | 2017-2018 / 2020-2021 | |
| Lyon | IGN LiDAR HD 2021 / Grand Lyon 2023 | 9 | 2021 / 2023 | 36,354 |
| Switzerland | swissALTI3D 2019 / 2026 releases, Geneva SITG 2019 / 2023 | 9 | 2018 / 2024, 2019 / 2023 | 55,734 |
| Norway | hoydedata.no projects, one locked raster each | 8 | 2018-2022 / 2021-2024 | 70,116 |
| Poland | GUGiK NMT lidar sheets | 9 | 2018-2022 / 2021-2024 | 296,098 |
| Estonia | Maa-amet LAZ, ground gridded here | 8 | 2018-2020 / 2020-2024 | 69,136 |
| Canada | NRCan HRDEM projects, 1 m COGs | 13 | 2017-2019 / 2019-2023 | 219,473 |
| New Zealand | LINZ nz-elevation, 1 m COGs | 13 | 2018-2021 / 2020-2024 | 131,722 |
| Kanagawa | prefecture point cloud, 0.5 m ground grids | 9 | FY2019-2020 / FY2024 | 127,958 |
| Sao Paulo | GeoSampa MDT LAZ (two sites stopped by the server's firewall) | 5 | 2017 / 2020 | 3,310 |

128 sites. Lessons from building them: a COG's own overviews differ from year to year by 5 cm (median) to 14 cm
(99th percentile) on slopes, which reads as moved ground, so every DEM is read at full resolution and averaged down
(and through `vrt://...?ovr=NONE`: the OVERVIEW_LEVEL open option let nodata into the averages); rivers, tidal
banks and reservoirs at a different level in each flight need a water mask (OpenStreetMap in England and Scotland,
flat-and-shifted cells in Norway and Geneva); Canada's national grid is rotated enough to lose half a window, so
each site uses its UTM zone. Shallow moves abroad are often the lidar's ground under vegetation (steep forest in
Kanagawa and New Zealand: plain AlphaEarth change ranks all moved ground at AUC .54 to .75 there, moved ground
deeper than 2 m at .80 to .98), so every table also gives moves deeper than 2 m, and the world sites teach only
those, as the islands did.

**The map's model as shipped** (abroad recipe, cutoff .777, the aggregate; AlphaEarth gated in brackets):

| | found | unchanged lit | AP | found, deeper than 2 m |
|---|---|---|---|---|
| US digging (held out, for scale) | .60 (.60) | .04 | .57 | .66 |
| Canada | .61 (.63) | .06 | .62 | .74 |
| Estonia | .68 (.68) | .23 | .57 | .75 |
| Netherlands | .40 (.33) | .04 | .55 | .50 |
| Poland | .33 (.28) | .01 | .43 | .50 |
| Britain | .27 (.16) | .01 | .31 | .41 |
| Sao Paulo | .24 (.23) | .01 | .39 | .45 |
| Norway | .22 (.19) | .01 | .29 | .32 |
| Switzerland | .17 (.08) | .00 | .31 | .23 |
| New Zealand | .06 (.05) | .00 | .37 | .13 |
| Kanagawa | .06 (.04) | .00 | .33 | .24 |
| Lyon | .01 (.01) | .00 | .13 | .02 |

So London was not alone: the level is US-like in Canada and Estonia, half in the Netherlands and Poland, and near
nothing in New Zealand, Kanagawa and Lyon. The ranking is weaker abroad too (AP .13 to .43 against .57).

**Why: the digs are smaller.** Median patch of moved ground deeper than 2 m: US sites 20 ha, Canada 33, Estonia 65;
Britain 6, Netherlands 5, Kanagawa 5, Poland 3.5, Switzerland 1.0, Norway 0.9, Lyon 0.4. AlphaEarth's change on
that ground follows (median 1 - cos: US .23, Canada .26, Estonia .22; Britain .10, Norway .07, Lyon .04), while
unchanged ground abroad is as calm or calmer (.02 to .045 in Europe, Japan and New Zealand, .05 in the US, .05 to
.08 in Canada and Estonia). By patch size the regions line up much better (aggregate, world recipe below, found):

| | < 0.1 ha | 0.1 to 1 | 1 to 10 | > 10 ha |
|---|---|---|---|---|
| US | .30 | .49 | .62 | .68 |
| Netherlands | .22 | .37 | .54 | .78 |
| Poland | .12 | .32 | .56 | .74 |
| Canada | .18 | .34 | .43 | .71 |
| Estonia | .21 | .39 | .48 | .82 |
| Switzerland | .07 | .24 | .40 | .67 (2 sites) |
| Britain | .16 | .42 | .43 | .56 |
| Norway | .03 | .17 | .33 | - |
| Lyon | .05 | .15 | .34 | - |
| New Zealand | .02 | .05 | .19 | .41 |
| Kanagawa | .01 | .04 | .21 | .27 |

Big digs are found about alike in the US, Canada and Europe. New Zealand and Kanagawa stay low at every size: steep,
vegetated ground (noisier truth), and Kanagawa's second flight may be as late as March 2025.

**Taught with the world lidar.** Two recipes on the same rows (3,000 moved and 15,000 unchanged pixels a site),
every number held out: "abroad" (the shipped recipe's sites, refit; it reproduces the shipped numbers) and "world"
(those plus the 127 world sites that are not natural change, each country held out in turn; US sites in four
folds, islands and satellite labels in four). Each recipe's cutoff finds 60% at its held-out US digging sites
(abroad .739, world .809). Aggregate found (unchanged lit):

| | abroad | world |
|---|---|---|
| US digging | .60 (.039), AP .571 | .60 (.043), AP .557 |
| Islands | .53 (.010), AP .40 | .51 (.013), AP .33 |
| Satellite labels | .46 (.050) | .49 (.071) |
| Canada | .61 (.068) | .52 (.028) |
| Estonia | .71 (.242) | .54 (.054) |
| Netherlands | .42 (.037) | .47 (.040) |
| Poland | .32 (.008) | .39 (.013) |
| Britain | .30 (.017) | .35 (.012) |
| Switzerland | .14 (.001) | .29 (.006) |
| Sao Paulo | .25 (.006) | .24 (.006) |
| Norway | .22 (.011) | .15 (.003) |
| New Zealand | .07 (.001) | .15 (.004) |
| Kanagawa | .07 (.004) | .08 (.013) |
| Lyon | .01 (.000) | .14 (.005) |
| Shizuoka (test) | .05 | .09 |
| Crop switches lit (test) | .044 | .056 |
| Fields and towns abroad lit (Reunion, Guadeloupe) | .005 | .028 |
| Deserts and places with no truth lit | .073 | .063 |
| US farm, marsh, storm lit | .011, .017, .124 | .022, .028, .031 |
| Dense cities lit (22 squares, three pairs) | .007 | .015 |
| AlphaEarth layer: crop switches lit | .228 | .320 |

One worldwide model moves the weak places up (Lyon .01 to .14, Switzerland .14 to .29, New Zealand .07 to .15,
Britain .30 to .35) and quiets Estonia's and Canada's unchanged ground (.24 to .05, .07 to .03), at a cost: fields,
dense cities and the islands' fields light about twice as much, and the AlphaEarth layer lights crop switches .23 to
.32. It does not level the score: found still runs .08 (Kanagawa) to .54 (Estonia) at one cutoff.

**A cutoff per region does not fix it either.** Finding 60% held out at each region's own cutoff (world recipe)
would light 12% of Britain's unchanged ground, 17% of Switzerland's, 25% of Norway's, 28% of New Zealand's, 37% of
Sao Paulo's and 52% of Kanagawa's, against 4% in the US: the small digs do not rank far enough above calm ground.
Only Canada (.56, lit 6%), Estonia (.60, 12%) and the Netherlands (.46, 5%) are near the US cutoff.

**A calibrated chance is not portable.** Fit isotonic (each site weighted equally) on the held-out aggregate: the
score where half the ground moved runs from .36 (Sao Paulo) to .999 (Netherlands), because the share of moved
ground in a window is set by where the window was put (1% on the islands, 29% in Sao Paulo). Among the highest
scores (.9 to 1) the share that moved is more alike (.46 Lyon and the islands, .53 Netherlands, .59 US, .65
Britain, .72 to .79 Poland, Switzerland, Norway, Estonia, Kanagawa, .82 Canada, .89 New Zealand): a pixel the map
lights is about as trustworthy abroad; what differs is how much of the digging it reaches.

**The structure gate abroad** (hard, section 28) costs .00 to .02 of found and changes unchanged lit by under .01
in every country: it does not hide European digging.

So: the level gap abroad is mostly the size of the digging (European, Japanese and New Zealand works are small,
urban and mixed with buildings, where the US sites are 20 ha grading pads), not a regional offset a calibration can
remove. The world recipe is saved as `models/model_world.npz` (the map's format; TESSERA's layer keeps the shipped
model) but the map still reads `model_abroad.npz`.

Run: `uv run python world_sites.py [source ...]`, then `world_eval.py baseline | teach | calib | report`
(teach about an hour, peak 27 GB); results in `results/world.json`.

## Open

- Clearcuts against forest mines, a second stage (section 19): a head taught on cleared-and-dug against
  cleared-not-dug, multiplied with the aggregate, keeps Kolwezi's found (.47 to .45) and darkens the clearing near
  it (.12 to .00), held out. Not yet tried as a map rule on the 14 US digging sites (needs the surroundings at every
  digging pixel), nor on more than one square of clearing per mine.
- Clearcuts against forest mines (sections 14, 15): on the abroad recipe any clearcut teaching costs Kolwezi in
  step with the clearcuts it quiets (1,000 px per region already .44 to .35), and more forest mines taught do not
  remove it. Kolwezi's ranking holds throughout (AP .82 to .86): the cost is its level against the US cutoff. The
  US recipe with burns taught takes clearcuts to .04 with none taught, but finds less satellite digging (.20) and
  little at Kolwezi (.03).
- Ground that moved with no one digging (section 16): landslides light a third to two fifths, a flooded pit or
  valley and an exposed lake bed most of their ground, river banks a quarter abroad-taught, new solar arrays a
  fifth to a quarter. Lidar would call most of these moved. Whether each is a find or a false alarm is a scope
  question, not settled here. Land leveling (section 14) is held shut by the gate whether taught or not.
- Clearing without digging inside mining districts: Madre de Dios's unchanged labels light 41%.
- Native year pairs can be too narrow where a lidar project ran into the next year (Hibbing to May 2023, Columbus
  NE to April 2021; section 17): the after year could follow the flight's end date.
- A cutoff set per place does not even out the sites (section 18); Four Corners is order inside the site, not level.
- The radar-safe gate (`either_radar`) is now tested on 1,898 moved labels with radar in one year (section 15):
  no digging lost pooled, less lit, but Ekibastuz and Sokolov lose some.
- Panels draw the raw aggregate on a continuous ramp, so ground below the cutoff still looks lit (the Sahara
  panel looks busier than its 4%). Ways to draw it: start the ramp at the cutoff, two tones, or the cutoff
  marked on the key.
- The map (section 24): whether to offer the US-only recipe beside the abroad one (a switch, by place, or not).
  Zoomed-out AlphaEarth is the model run on overview means; worth labeling as a where-to-look view.
- Teaching TESSERA to the AlphaEarth model (section 24): a linear student on AlphaEarth's two years does not take
  on the aggregate's quiet on fields. Untried: a student with more room (a small two-layer net, or AlphaEarth's
  surroundings as in section 19's ctx head). `clean` costs nothing and helps a little; it could replace base.
- Ideas named, not to-dos: a disturbance layer marking clearcuts (never hiding them); ESD (Landsat-based, 30 m)
  compared at the lidar sites; TESSERA made ourselves from its open code and weights (2025 without the descending
  radar, optical only, the full 192-d); a study of one chosen area with every model on the same inputs and 3DEP.
- Height (section 27): no ghost (Mapterhorn, 3DEP, Copernicus) and no head taught on dz carries meters or cut
  against fill to a new site. Untried: a ghost matched in time (each year against its own flight, held out by 1 km
  blocks, close to the head); building height (NYC footprints' height_roof) as the target for roofs and new
  buildings, where the change is tens of meters, not one or two.
- Worldwide (section 29): one model taught with 127 lidar sites abroad lifts the weak places but does not level the
  score, and a cutoff per region would light 12% to 52% of calm ground where digs are small. Not tried: a model
  that reads the pixel's surroundings (section 19's ctx head) for small urban digs; teaching by patch size;
  weighting the world sites against the US; per-site AlphaEarth years from the flight dates of each tile (some
  windows span several flight days); the US-only and world recipes on the map side by side. Sao Paulo's two
  sites behind GeoSampa's firewall, Spain (captcha), Australia (orders by email), Denmark (token) and Sweden
  (second scans 2025 and 2026) are still unbuilt.
