# Phase 2 checkpoint A: infrastructure and oracle ceilings

Reproduce: `python tools/phase2_oracle.py` (~13 min, dominated by building depth-2/3 refinements of the
22,403 split triangles in pure Python). All scores: exact point-sampled overlap, identity mapping (F1-F4 as
numbered), invisible faces excluded.

## Mesh facts (VERIFIED)
- **Top is solid, not hollow.** One watertight body, consistent winding, volume 315,561 mm^3. Inward rays from all
  1,000,954 faces: 0 hit a front-facing (cavity) wall. The ~1.5-2 mm walls seen in chat cross-sections must have come
  from the slicer's shell/infill view, not from this mesh. Inward-ray thickness is therefore valid; the feature step
  will still cast through to the final exit so it also works on a hollowed mesh.
- Inward first-hit distance by section, median (10th-90th pct): F1 13.4 mm (7.0-24.0), F2 loops 2.7 mm (1.1-6.2),
  F3 rails 5.8 mm (2.7-12.2), F4 deck 5.8 mm (4.1-15.5). Thickness alone separates loops from rails well, posts/tree
  from everything.
- **Faces never visible from outside:** 1,464 faces, 0.053 % of area (32 hemisphere directions on every face, then
  512 on the 9,691 survivors). Mostly inside the loop rings. Excluded from scoring (`out/top_visible.npy`).
- Section boundaries mostly do not follow creases: 30-37 % of boundary edges have a fold > 30 deg (rails/deck 12 %),
  median fold ~20 deg vs 6.7 % of all mesh edges > 30 deg. Shape at 2-15 mm scale, not creases, will have to
  separate the parts.

## Infrastructure (VERIFIED, 25 tests)
- `paint.py`: split geometry ported from `TriangleSelector::perform_split` (corner rotation by special side,
  midpoints, child vertex order). Leaves tile the parent exactly; sampled fractions match the analytic ones.
  `build_tree(label_fn, max_depth)` splits a triangle along a label boundary (2-side split at the corner shared by
  the two crossing edges, else 3-side). `sample_points(64)` is an exact area-stratified lattice.
- `score.py`: `score_paint` = exact overlap by sampling 64 lattice points inside any split triangle (bulk for
  whole-triangle pairs); mode `identity`. Truth vs itself = 100.00 on every section.
- `transfer.py`: fractional truth on the 100k decimated mesh (16 samples per decimated triangle -> nearest full
  centroid), closest-face transfer of labels back to the full mesh.
- `clicks.py`: regions = connected components of each filament's dominant label >= 20 mm^2; clicks weighted by
  area x (hops from boundary)^2, later clicks farthest-first. Census on Top: **35 regions**: F1 9 (tree 24,065 mm^2 +
  8 posts ~2,000-6,400), F2 8 (4 loop sides ~5,000 + 4 slivers ~60), F3 11 (8 rails 770-1,040 + 3 slivers 28-50),
  F4 7 (deck 45,218 + 6 caps 55-75). Dropped slivers < 20 mm^2: 0.12 % of area. Clicks land a median 9-15 hops
  (~4-6 mm) inside their region.

## Oracle ceilings (full mesh, exact, identity)

| ceiling | overall | F1 IoU | F2 loops | F3 rails | F4 deck |
|---|---|---|---|---|---|
| A: whole triangles, full mesh (majority per triangle) | 99.55 | 99.70 | 99.34 | **94.54** | 99.12 |
| C: splits depth <= 2 on the full mesh | 99.89 | 99.90 | 99.81 | 98.74 | 99.79 |
| C: splits depth <= 3 on the full mesh | 99.94 | 99.95 | 99.90 | 99.39 | 99.90 |
| B: whole triangles on 100k decimated, transferred to full | 99.01 | 99.23 | 98.15 | **88.83** | 98.32 |
| iteration metric: decimated majority scored in decimated space | 98.32 | 97.91 | 97.14 | 86.31 | 96.97 |

Reading:
- Targets (overall >= 95, every IoU >= 90) are reachable on the full mesh with whole triangles; nothing in the
  hand paint caps them. Rails are the tight one: whole-triangle output caps F3 at 94.5, so a method that is perfect
  up to triangle resolution still loses 5.5 IoU points on rails to brush splits. Depth-2 splits recover most of it.
- **The decimated copy cannot reach the rail target by itself** (88.8 after transfer, 86.3 in decimated space).
  Iterating on the decimated mesh is fine for F1/F2/F4 and for method search, but F3 must be confirmed on the full
  mesh with the sub-triangle refinement (step F). While iterating, F3 in decimated space should be read against its
  86.3 ceiling, i.e. 80 in decimated space is roughly 93 on the full mesh.
- Refinement in pure Python is slow (350 s for 22k triangles at depth 2); step F will refine only triangles whose
  corners disagree, which is a similar count, so this is acceptable but worth vectorizing if it becomes the loop.

## Status of the plan
A1-A4 done. Stopping here for review before B (features + separability table). Base untouched.
