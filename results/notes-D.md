## Wider year pairs on the digging sites

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

## A cutoff set per place

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
