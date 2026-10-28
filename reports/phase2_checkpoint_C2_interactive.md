# Phase 2 checkpoint C2: click-and-correct curve (targets NOT met)

Reproduce: `python tools/phase2_interactive.py --draws 5 --max-clicks 40 --save` (~3 min). Log:
`phase2_interactive_curve.log`. Decimated space (ceilings: overall 98.3, F1 97.9, F2 97.1, F3 86.3, F4 97.0),
identity mapping, with height. Protocol: start from per_type (6 clicks: tree, a post, a loop side, a rail, deck,
a cap); each round add one click at the centre of the largest wrongly labelled connected region (skipping regions
that already hold a correct click), re-solve, to 40 clicks. 5 draws differ in the initial 6 clicks.

## Curve (overall, mean / worst of 5 draws; per-section IoU means)

| clicks | raw overall | raw F1 / F2 / F3 / F4 | with rule overall | rule F1 / F2 / F3 / F4 |
|---|---|---|---|---|
| 6 | 88.6 / 86.7 | 86.8 / 78.9 / 41.8 / 81.0 | 88.0 / 86.1 | 86.0 / 78.6 / 41.1 / 79.5 |
| 10 | 88.9 / 86.4 | 87.3 / 78.0 / 43.2 / 82.1 | 88.3 / 86.0 | 86.7 / 77.7 / 42.5 / 80.7 |
| 16 | 89.3 / 87.4 | 88.1 / 77.7 / 47.0 / 82.5 | 88.7 / 85.7 | 87.4 / 77.4 / 46.1 / 81.1 |
| 20 | 89.8 / 87.9 | 88.5 / 78.5 / 50.0 / 83.3 | 89.3 / 86.3 | 87.8 / 78.4 / 49.0 / 81.9 |
| 30 | 90.9 / 88.7 | 89.6 / 80.1 / 53.2 / 85.2 | 90.4 / 86.8 | 89.0 / 79.7 / 52.3 / 83.9 |
| 40 | 91.7 / 90.0 | 90.3 / 82.4 / 54.7 / 86.2 | 91.2 / 88.1 | 89.7 / 82.0 / 53.9 / 85.0 |

- **Clicks to overall >= 90 (raw):** per draw 6, 21, 37, never, 17 -> mean 20 of the 4 that reach it, worst 37;
  1 of 5 draws does not reach it by 40. With the rule: 6, 23, 39, never, 19.
- **Clicks to overall >= 95: never, in any draw.** Clicks to every section IoU >= 90: never.
- Rails (F3) go from 42 to 55 in 34 corrections and are never above 61 (ceiling 86); the worst draw's weakest
  section is still at 44 after 40 clicks.
- Wrong area left inside already-clicked regions at 40 clicks: 0.0-2.2 % of the model: clicks mostly do fix
  the region they land on, the problem is that each fixes only a little.

So the curve is almost flat: ~+3 points for 34 corrections, ~0.1 point per click. A user would not get to 95 this way.

## Time per click update
- Decimated 100k copy: **0.24 s** (patch + prototype density 0.09 s, warm-started 3-cycle graph cut 0.15 s).
- Full mesh (1,000,954 faces, measured with stand-in features, same graph): prototype density 2.4 s, one
  alpha-expansion cycle 3 s (warm start does not help), graph build 2 s + through-links 1 s once.
  **Estimate 6 s per click with one cycle, ~12 s with three.** The few-seconds target holds on the decimated copy
  with margin; on the full mesh it needs either one cycle per click or interaction on the decimated copy with a
  single full-mesh solve at the end (step F anyway).

## Bottom-surface rule
Detection (`autocolor/rules.py`), **FLAGGED orientation-dependent, not coordinate-dependent**: a face is "bottom
surface" if its normal points down (normal . (0,0,-1) > 0.7, print orientation from the 3MF) and a ray cast
straight down from just outside the face hits nothing. On Top: 1,025 decimated faces, 18.5 % of area, truth there
97 % F4, 2 % F1, 1 % F2. The rule relabels a bottom face with the label of the surface directly above it through
the solid (its inward normal ray's first hit, within 7 mm = the slab's own thickness; 84 % of the bottom area has
one); other bottom faces (feet of legs, overhanging branch undersides, fused rail undersides) are left to the method.
It is applied after labelling, never via clicks; bottom faces stay in the score.
- Cost on the truth itself: **none** (98.27 vs 98.32 ceiling). Two earlier versions were wrong and are on record:
  "nearest non-bottom face" copied the mislabelled deck edge (-7 points), and a 12 mm reach let the underside under
  the bottom rail copy the rail's label (rails -11 IoU on the truth).
- Effect on the method: **-0.5 to -0.7 overall at every click count** (88.6 -> 88.0 at 6; 91.7 -> 91.2 at 40).
  The rule copies the deck top, and the method's deck top is only ~85 % right (plank grooves read as bars/rails),
  so the rule propagates the top's errors to the underside, where the method itself was doing slightly better
  thanks to the through-solid links. The rule is correct in principle and free on perfect labels; it will pay
  off once the deck top is right.

## Reading
- The seeded prior (one prototype per click, max over clicks) can only claim faces that look like something
  clicked. Corrections add look-alikes locally (reach 80 mm, feature gate 1.5 std) and that is all they add;
  rails fused into the loop lattice, plank grooves and the textured underside keep being claimed by the wrong
  prototypes between clicks. The curve measures exactly that: linear, slow.
- Global-reach corrections were tried (`--reach 0`): non-monotonic, a correction on the underside dragged the
  loops to F4 (one draw 92 -> 81 -> 91). Local reach is stable but slow.
- Protocol caveat: the simulated user clicks region centres chosen from the truth, which is optimistic.

## Where this leaves the plan
The interactive protocol does not rescue the current prior; it confirms the C conclusion that the bootstrap
needs structure, not more clicks. Proposed next, still awaiting review: the texture-robust part linking held
earlier (offset-surface / cone through-solid links so top, grooves, edges and underside become one slab; rails
linked across their width, not into the lattice), then re-run this same curve, which is now cheap (3 min).
