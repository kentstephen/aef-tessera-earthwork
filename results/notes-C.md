## Lookalikes not tried before, and the reservoirs split by a water mask

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
