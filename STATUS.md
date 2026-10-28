# STATUS

**Phase:** 2, step C run; **stopped for review, targets NOT met** (`reports/phase2_checkpoint_C.md`).
Seeded method (per-click KDE prior + feature-aware Potts graph cut + through-solid links), decimated space:
per_type 88.6 mean / 86.7 worst (rails 42); per_filament 81; per_region_1 72; per_region_2 86; per_region_3 83.
Best single draws 92-94. Rails never above 57. One click does label all posts / loop sides / caps; not all rails.
Bounding facts: GBM on representative labels = 97.0, so features suffice and the click-to-training bootstrap is
the whole problem; the deck underside (18 % of area, textured, thin) is reached by no click and looks like loops.
Steps A and B: `reports/phase2_checkpoint_A.md`, `reports/phase2_checkpoint_B.md`.
Combined features >= 0.99 AUC on every section pair with height, >= 0.92 without; rails/deck is 0.924 without
height (tan_aniso, plan_12, curv_3 carry it), 0.995 with. Posts vs tree is the weak pair (0.92) but same filament.
Caps: ext_12 / sph_12 ~0.98-1.0 vs deck and F1.
Plan: A infra -> B features + separability table -> C seeded classifier + graph cut -> E cap detector ->
F confirm on full mesh with sub-triangle refinement. D (automatic track) skipped by decision. Targets: overall >= 95,
every IoU >= 90, scored identity (no mapping), invisible faces excluded.
**Last score (PoC baseline, best mapping):** overall 83.70 %. Per-section IoU: F1 tree+posts 76.6, F2 loops 72.0,
F3 rails 0.0, F4 deck 77.4. Details in `reports/phase1.md`.
**Oracle ceilings** (`reports/phase2_checkpoint_A.md`): whole-triangle full mesh 99.55 overall, F3 rails 94.5;
depth-2 splits 98.7 on rails; decimated 100k -> full only 88.8 on rails, so rails must be confirmed on the full mesh.
Top is solid (not hollow, VERIFIED); invisible faces 0.053 % of area.
**Env:** Python 3.13.3, `.venv` (trimesh, numpy, scipy, open3d, matplotlib, pytest). `requirements.txt` unpinned.

## Inputs (refs/, gitignored, 111 MB; copy by hand on a new machine)
- `SD_RPG_TreetopDruidHavenDiceTower-Top.obj`: 1,000,954 tris, extents 230.5 x 171.0 x 123.0 mm.
- `TreetopDruidHaven_painted.3mf`: Bambu project, 4 filaments. Object 2 = Top (ground truth, index-aligned with the
  OBJ after adding offset (150, 150, 61.5113)). Object 4 = Base (**held out**: parse only, never tune on it).
  Object 6 = unpainted Base duplicate.
- `TreetopDruidHaven-Top_autocolor_poc.3mf`: chat PoC export, whole-triangle, 4 labels.

## Code
- `autocolor/paint.py`: paint_color decode/encode (tree form), area fractions per state, cached per distinct string.
  Round-trip tested for states 0-40 and random split trees (`tests/test_paint.py`, 21 tests).
  Encoder handles split trees already; geometry for emitting sub-triangle splits (which vertex order / special side
  means what) is NOT implemented, Phase 2+.
- `autocolor/truth.py`: `Project` (filament colours, objects, default extruder, mesh + paint per object),
  `load_paint_3mf` for plain exports, `truth_matrix` -> (N, n_filaments) area matrix.
- `autocolor/score.py`: overlap matrix, label mapping (`best` many-to-one headline, `one2one` Hungarian), per-section
  IoU / recall / precision, overall agreement.
- `autocolor/render.py`: open3d decimation + matplotlib 2-view preview.
- `autocolor/paint.py` also: split geometry (perform_split port), `leaf_states_at`, `build_tree`, lattice samples.
- `autocolor/score.py` also: `score_paint` exact point-sampled scoring, `identity` mode.
- `autocolor/transfer.py`: truth onto decimated mesh, labels back to full mesh (closest face).
- `autocolor/clicks.py`: regions (>= 20 mm^2 components), boundary hops, click sampling. 35 regions on Top.
- `autocolor/features.py`: 16 per-face features (through-cavity SDF, in-solid tangent extents, curvature,
  PCA at 4/12 mm, height [FLAGGED]); 12 s on the 100k copy -> `out/top_dec100k_features.npz`.
- `autocolor/clicks.py` also: `part_types` (simulator-only area grouping, FLAGGED), conditions per_region_1/2/3,
  per_filament (4 clicks), per_type (6 clicks).
- `tools/phase2_features.py`: separability table (`reports/phase2_separability.txt`).
- `autocolor/seeded.py`: `FaceGraph` (adjacency, dihedrals, geodesic, through_links), `seeded_labels` (patch
  method, the one reported), `seeded_segments` (over-segmentation variant, worse), `alpha_expansion` (PyMaxflow,
  verified vs brute force), `_fit_predict` (gbm / logreg / kde / proto priors).
- `tools/phase2_seeded.py`: score table runner (`--conditions`, `--draws`, `--height`, `--save-best`,
  hyper-parameters), writes `out/seeded_<cond>_<height>.npz`.
- `tools/phase2_oracle.py`: oracle ceilings (~13 min). `out/top_visible.npy`, `out/top_regions.npz`,
  `out/top_dec100k_truth.npz` (fractional truth on the decimated copy).
- `tools/phase1_report.py`: end-to-end; `--candidate path.3mf` scores any whole-triangle export.
- Outputs in `out/` (gitignored): `top_truth.npz` (M, areas, V, T), `base_truth.npz`, `top_dec100k.npz`
  (decimated copy + `src` index into full-mesh triangles), `top_truth_preview.png`.

## Sections on Top (see reports/phase1.md for the table)
F1 dark red = tree AND posts (two distinct parts, one colour). F2 grey = railing loops. F3 yellow = rails
(top/bottom bars). F4 tan = deck + cut ends of posts/branches. Unpainted 9 % of area -> default F1 (verified).

## Future items (not now)
- Posts vs tree separation will matter once 8-9 filament machines allow different colours for them; shape
  features at <= 12 mm give only 0.92 AUC. Candidate feature: straightness / constant radius over a ~30 mm span
  (a post is a straight constant-radius cylinder, a trunk tapers and bends).

## Open problems
- PoC never separates rails (F3) from loops (F2): F3 IoU 0. Rails are only 6.2 % of area but 13.7 % of their area
  is in brush-split triangles, so whole-triangle output will cap the rail score; sub-triangle output needed eventually.
- PoC "frame+posts" label mixes posts (F1) with deck frame (F4): costs F1 recall (77) and F4 precision (78).
- Whole-triangle painting makes jagged boundaries on long triangles (from CLAUDE.md).
- Scorer assumes product coupling when both sides are fractional; exact only for whole-triangle candidates.

## Next step
Review of checkpoint C. Proposed (not started): a texture-robust part linking (through-solid links from an
offset surface / cone), area-share-weighted self-training with change limits, or an interactive click protocol
(click the largest wrong region, repeat) with a clicks-to-target curve. E (caps) and F (full mesh) wait.
