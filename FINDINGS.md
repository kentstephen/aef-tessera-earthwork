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

## Open

- Clearcuts: teaching them as quiet works on clearcuts and costs forest mines (Kolwezi); teaching burns alone
  gets half the way at no measured cost. Between the two: fewer cut pixels, a lower weight, or cuts taught only
  far from any digging.
- The radar-safe gate (`either_radar`) is free on everything measured but tested on digging at only 13
  satellite labels with radar in one year.
- Reservoir shores need a per-year water mask to say whether the lit ground is the bathtub ring.
- Panels draw the raw aggregate on a continuous ramp, so ground below the cutoff still looks lit (the Sahara
  panel looks busier than its 4%). Ways to draw it: start the ramp at the cutoff, two tones, or the cutoff
  marked on the key.
- The notebook: the joint model with the either gate (radar-safe), US-taught or abroad-taught, with AEF alone
  where TESSERA has no pixel; zoomed out, each cell carries its highest 10 m score.
