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

## Open

- The burn cost (Grizzly Flats): more burned forest as quiet teaching ground (only two burn sites are
  cached), or fire perimeters (MTBS, NIFC) as context on the map.
- Panels draw the raw aggregate on a continuous ramp, so ground below the cutoff still looks lit (the Sahara
  panel looks busier than its 4%). Ways to draw it: start the ramp at the cutoff, two tones, or the cutoff
  marked on the key.
- The notebook: the joint model with the either gate, US-taught or abroad-taught, with AEF alone where
  TESSERA has no pixel.
