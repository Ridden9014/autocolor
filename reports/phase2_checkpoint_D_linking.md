# Phase 2 checkpoint D: part linking round, junction metric, stop rule -> STOP

Reproduce: `python tools/phase2_interactive.py --draws 5 --max-clicks 20 --prims 4` (primitives) and without
`--prims` (baseline). Logs: `phase2_prims_curve_final.log` (final soft primitives), `phase2_prims_curve.log`
(baseline, soft, hard; first version with the primitive-vote bug). Decimated space, with height, identity mapping,
bottom rule applied. "Baseline" = checkpoint C2 method.

## Stop rule at 20 clicks (mean of 5 draws, with rule)

| criterion | required | baseline | primitives (final) | met |
|---|---|---|---|---|
| loops IoU | >= 95 | 78.4 | 82.6 | **no** |
| rails IoU | >= 75 | 49.0 | 58.4 | **no** |
| post-rail junction, wrong area in 3 mm zone | clearly better | 30.4 % | 26.5 % | marginal |
| loop-post junction | clearly better | 32.3 % | 28.5 % | marginal |
| loop-rail junction | clearly better | 23.5 % | 20.5 % | marginal |
| misplaced boundary (post-rail / loop-post / loop-rail, mm) | clearly better | 343 / 562 / 2795 | 350 / 563 / 2859 | **no** |

**Verdict: stop the method search.** Phase 3 proceeds with the best method (primitives soft + bottom rule +
local corrections) plus hand touch-up in the slicer. The deck and tree scores do not enter the decision.

## What the round delivered
### Junction metric (`autocolor/junction.py`)
Parts = connected regions of the hand paint typed by area (tree, post, loop, rail, deck, cap; simulator grouping,
no coordinates). A junction type = the mesh edges where two part types meet. Per junction: wrongly labelled area
within 3 mm, and misplaced boundary length = truth junction length with no predicted boundary within 1 mm +
predicted boundary length in the zone with no truth boundary within 1 mm. Floor (truth vs itself, from part-vs-label
mapping noise on the decimated copy): ~4 mm for post junctions, 376 mm for loop-rail, 139-277 mm for deck junctions.

Baseline junction numbers (per_type 6 clicks, best draw): post-rail 21.7 % wrong / 256 mm misplaced, loop-post
21.9 % / 480 mm, loop-rail 20.9 % / 2466 mm, cap-post 67 %, deck-post 35 %, deck-rail 36 %.

### Texture-robust through-links (cone of inward rays from an offset start, median hit, mutual)
No gain: same-label rate 88-91 % like the single-ray links, rails still 30-41 % (a rail's inward ray exits through
the loop lattice it is fused into, whatever the cone), per_type scores unchanged within 0.3. Closed.

### Primitive fitting (`autocolor/primitives.py`, RANSAC, no coordinates, mm scales flagged as tuned on Top)
- **Slab**: largest plane by inlier area + largest parallel plane 2-15 mm away; claim = between the planes +- 2.5 mm
  texture, inside the 2D hull. Found thickness 4.8 mm; claims 49,600 mm^2, **90 % deck, covers 99 % of the deck**;
  the 10 % impurity is the 5 mm band of posts, rails and tree where they cross the slab's height.
- **Cylinders** (axis = slab normal, RANSAC circles on side-facing faces, >= 15 mm height): 6 posts at 96-100 %
  purity (r 5.3-9.5 mm) and 5 tree branches (r 10-14 mm, 100 % tree). 43 % of post area claimed (bands near the
  deck and the cap excluded by the side-facing test); 2 posts missed. Tree cylinders are harmless here (F1 either
  way) but would be wrong on a model where posts and trunk differ: FLAGGED.
- **Rails** (RANSAC lines parallel to the slab through bar-like faces, claim limited to bar-like faces, >= 350 mm^2):
  8 lines, 4 at 92-95 % purity, 4 at 71-80 %; 57 % of rail area. The loop row fused under each rail is what the
  impure half claims.
- 62 % of the area is claimed. Clicks label primitives (majority of clicks on a primitive's own surface, earliest
  breaks ties; fringe clicks only act locally); unclicked cylinders/rails take the label of clicked ones of the same
  kind; claimed faces get a unary bonus (4 = soft; 50 = hard).

### Curve with primitives (soft), start per_type, 5 draws

| clicks | overall mean / worst | F1 / F2 / F3 / F4 | baseline overall |
|---|---|---|---|
| 6 | 89.4 / 86.8 | 87.1 / 79.9 / 44.1 / 82.6 | 88.6 / 86.7 |
| 10 | 90.5 / 88.3 | 87.5 / 80.9 / 50.7 / 84.8 | 88.9 / 86.4 |
| 14 | 91.4 / 90.6 | 88.9 / 81.0 / 55.7 / 86.4 | 89.2 / 87.1 |
| 20 | 91.9 / 90.8 | 89.3 / 82.4 / 58.6 / 87.0 | 89.8 / 87.9 |

- Clicks to >= 90: 6, 13, 10, 7, 8 -> **mean 8.8, worst 13, all 5 draws** (baseline: 2 of 5 draws, mean 11.5,
  worst 17). 95 still never. Every section >= 90 never.
- Hard primitive seeding (strength 50): 91.4 at 8-10 clicks, then the slab's impurity band and the first
  disagreeing correction collapse it (deck -> 0.8). Not usable as hard constraints; soft is the method.
- Bug fixed on the way: a primitive took the label of its last/majority click, so a correction on a post foot
  inside the slab band relabelled the whole deck. Votes now come only from a primitive's core faces.
- 0.3 s per click update on the 100k copy (primitive fitting 12 s once).

## Why rails and loops stay where they are
Rails sit at 58 (ceiling 86): the 8 rail boxes cover 57 % of rail area and half of them are a quarter loops. The
rail-loop boundary is where rails are fused into the lattice: no crease, same thickness within 3 mm, and the bar
feature decays over the last ~4 mm of each rail. Loops at 82: the rail boxes' impurity and the posts' fringe both
eat loop area at junctions, which is exactly the loop-post and loop-rail zones. These are the junctions the stop
rule asks about, and the fitted primitives do not resolve them; they resolve the big faces (deck, post shafts).

## Carried into Phase 3
Best method = per_type-style clicks + primitives (soft) + bottom rule + local corrections; ~92 overall at 20 clicks
on the decimated copy, deck/tree/posts good, loops ~82, rails ~58, with the slicer's own tools for the rail-loop
junctions. Full-mesh transfer + sub-triangle refinement (step F) and Base (held out) are Phase 3 work.
