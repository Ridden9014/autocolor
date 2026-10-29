# Rooftop (Wizards Arcane Treasury dice tower): candidate parts, no ground truth

Input: `refs/SD_RPG_WizardsArcaneTreasuryDiceTower-Rooftop.stl` (no Bambu project in refs/; export will be a bare
3MF written from the STL). Nothing tuned: Top's features, oversegmentation (16 GMM clusters, Potts cut, lam 2) and
primitive fitter as they are. The only choices made here are the working resolution and the merge size that gives
~20-40 parts. Reproduce: `python tools/rooftop_parts.py res`, `... parts --n 200000 --min-part 800`,
`... views --n 200000`. Logs: `rooftop_res.log`, `rooftop_parts.log`.

## 1. Mesh and working resolution (VERIFIED)

1,396,424 triangles, 698,212 vertices after merging, 135.9 x 135.9 x 198.3 mm, 88,143 mm^2 surface, median edge
0.35 mm. The STL is heavily over-tessellated: the decimated copies all stay within 0.05 mm of the full surface.

| copy | decimate | full -> copy distance (area-weighted) | median / p90 edge | features |
|---|---|---|---|---|
| 200k | 26 s | 100 % within 0.25 mm, p99 0.02, max 0.05 mm | 0.75 / 2.0 mm | 753 s (13 min) |
| 300k | 14 s | 100 % within 0.25 mm, p99 0.01, max 0.04 mm | 0.59 / 1.6 mm | 1,676 s (28 min) |
| 400k | 13 s | 100 % within 0.25 mm, p99 0.01, max 0.03 mm | 0.50 / 1.4 mm | 6,676 s (1 h 51) |

**Working copy: 200k.** It keeps every >= 1 mm feature (nothing on the surface is more than 0.05 mm away from it)
and the finer copies buy nothing measurable at 2x / 9x the feature cost. Feature time scales much worse than
linearly (the 3 mm neighbourhood queries and per-face PCA loops grow with face density); the 100k Top copy took
12 s. All three feature files are cached in `out/rooftop_dec{200,300,400}k_features.npz`.
Invisible faces on the 200k copy: 9.0 % of area (Top 0.05 %, Base 5.6 %): internal faces exist; they are
segmented like everything else here and will take whatever label their part gets.

## 2. Over-segmentation (VERIFIED numbers)

16-cluster GMM + Potts cut gives 13,695 raw segments (the shingle texture fragments everything). Greedy region
merging (smallest segment into the neighbour with the longest shared boundary, union-find) with a floor of
800 mm^2 gives **28 parts** covering 100 % of the area. Sweep (clusters x floor -> parts): 16 x 400 -> 43,
16 x 800 -> 28, 16 x 1500 -> 18; 8 x 800 -> 25; 6 x 400 -> 26; 4 x 400 -> 23. Height feature excluded (flagged
option `--height`; not used).

Primitives (Top's fitter, as is): one slab (2.8 mm thick, 17,142 mm^2: the flat underside plus the rim) and two
cylinders (r 12.8 mm h 31 and r 14.3 mm h 41: finial tiers). 24 % of area claimed. No rails found (correct: there
are none). Per-part table with area, face count, mean nz, thickness, anisotropy, curvature, planarity and primitive
membership: `rooftop_parts_table.md`.

## 3. Views

`rooftop_parts_view1.png` front (az 0, el -20), `view2` back (az 180), `view3` 3/4 from above (az 225, el -45),
`view4` from below (az 45, el -130). Numbers sit on each part's largest visible blob in that view; visibility is
exact (face ids rendered with the same painter's order and read back). Parts hidden in a view are unlabelled there
(19 / 19 / 19 / 20 of 28 labelled). The first version of these views placed numbers by the part's visible
centroid with an approximate z-buffer and put several hidden parts (0, 1, 6, 11, 17, 24) onto the cone courses;
the reading below and the first mapping were made from those wrong views. Corrected reading (position facts
VERIFIED from the table below, names INFERRED):

| part | area mm2 | z range (5-95 %) | mean r | nz | what it is |
|---|---|---|---|---|---|
| 0, 22 | 11497, 1023 | 0-2 | 46-55 | -1.0 | flat underside (0 has a wedge split off as 22) |
| 16 | 1587 | 0-2 | 66 | -0.2 | outer rim edge |
| 11 | 2136 | 2-13 | 22 | 0.05 | vertical wall of the central socket / peg under the base (seen only from below) |
| 6 | 5656 | 14-19 | 37 | 0.97 | up-facing flat annulus inside the skirt (hidden) |
| 4, 8, 15, 19, 25, 26 | 7117 .. 909 | 2-31 | 56-60 | 0.25-0.45 | skirt lattice, one surface in six arbitrary patches |
| 17 | 1481 | 19-22 | 50 | 0.53 | step ring at the cone base, below the thick ring |
| 24 | 982 | 20-27 | 48 | -0.56 | under-lip of that step |
| 3 | 7253 | 23-42 | 51 | 0.28 | thick ring top/outer (z 33-41) + the surface below it down to z 23: 46 % of it is below the ring |
| 1 | 9405 | 23-54 | 41 | -0.41 | down-facing under-lips of the lower course steps; 35 % of it is below the ring |
| 20, 23, 14 | 1179, 984, 1690 | 41-60 | 42-45 | 0.5-0.6 | lowest shingle course (three patches) |
| 9, 5, 18, 27 | 2490, 5772, 1358, 836 | 51-100 | 24-39 | 0.1-0.4 | shingle courses, 27 is speckle on a step |
| 2 | 7639 | 59-99 | 27 | -0.33 | under-lips of the upper course steps |
| 12, 13, 21 | 1750, 1746, 1125 | 101-141 | 14-21 | 0.2-0.25 | top courses (ridge rings A-D sit inside these) |
| 7 | 3461 | 104-155 | 12 | -0.30 | under-lips of the top courses and finial collar |
| 10 | 2371 | 144-191 | 8 | 0.2 | spire |

## 4. Flags: what the method cannot see here

- **Shingle rows inside a course** are shallow relief (grooves ~0.5-1 mm into a surface of the same thickness
  and normal). No course was split along shingle rows, and the shingle grooves are what produced the 13k raw
  fragments. If shingles and their grooves want different colours, that is a slicer fill / height-range job.
- **The skirt lattice** (diamond pattern) is likewise surface relief on a 3-4 mm shell: the method sees one
  shell and splits it by noise. Treat 4/8/25/26/15/19 as one part unless the views show otherwise. Separating
  the raised diamonds from the recessed field is out of reach (same thickness, no crease a cut can follow).
- Any engraved rune / 2D decoration on the finial or courses would be invisible for the same reason as Base's
  inlay; none stands out in the features (no part with high curv_3 and the parent's thickness besides the
  speckle patches above), but that is absence of evidence from 28 coarse parts, not proof.
- Part counts and the 800 mm^2 floor were chosen for this model's size (88k mm^2); the floor scales with
  model area, not coordinates.

## 5. Export (`tools/rooftop_export.py --rings all --below-ring F3`, log `rooftop_export.log`, VERIFIED)

Mapping as given by the user from the first (wrong) views: F1 = 21 13 2 12 18 1 5 6 9 11 20 3 23 14 27 24 17,
F2 = 10 7, F3 = 4 8 25 26 15 19 16 0 22. **Rings -> F4 by MANUAL band rules** (height above the base and radius
about the bounding-box axis, this model's coordinates; a hand selection, not a method result): thick ring z 33.5-41
r >= 49 (top, outer and underside, 2,923 mm^2), ridge rings A z 100.5-103.5 r >= 21 (428), B 112.5-116.5 r >= 18
(478), C 126.5-129 r >= 15 (252), D 139.5-142 r >= 11.5 (190 mm^2). Applied on the full mesh after the crease cut.
Pipeline: decimated labels -> closest-face transfer (3 s) -> full-mesh crease cut (4 s, changed 0.01 % of faces)
-> band rules -> whole-triangle paint strings -> bare 3MF (`out/rooftop_autocolor.3mf`, 23.5 MB, one object, STL
triangle order, same layout as the PoC export). Re-read: 1,396,424 triangles, strings and labels identical.
**4 filaments.** Area: F1 55.9 %, F2 6.6 %, F3 32.7 %, F4 4.9 %. Preview: `rooftop_export_preview.png`.

**Second MANUAL rule (user's decision after the first pass): everything below the thick ring (z < 33.5 mm)
that is not F4 -> F3.** This covers the socket wall 11, the hidden annulus 6, the step ring 17 / under-lip 24 and
the lower portions of parts 1 and 3 (which straddle the ring); 411,926 faces, 16,899 mm^2 moved from F1 to F3.
Everything above the ring stays as mapped. Final area: **F1 36.7 %, F2 6.6 %, F3 51.8 %, F4 4.9 %**; 4 filaments.
The first pass (without this rule) had those 19 % as F1, from a mapping made off the mis-labelled first views.
Both ring bands and the below-ring cut are hand selections in this model's coordinates, not method results.
