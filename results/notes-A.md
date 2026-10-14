## Clearcuts between burns only and every cut; burns on the US recipe; teaching land leveling

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
