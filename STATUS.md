# STATUS

**Phase:** 2, steps A and B done (infrastructure, oracle ceilings, features + separability);
**stopped for review before step C** (seeded classifier + graph cut). Separability: `reports/phase2_checkpoint_B.md`.
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
- `tools/phase2_oracle.py`: oracle ceilings (~13 min). `out/top_visible.npy`, `out/top_regions.npz`,
  `out/top_dec100k_truth.npz` (fractional truth on the decimated copy).
- `tools/phase1_report.py`: end-to-end; `--candidate path.3mf` scores any whole-triangle export.
- Outputs in `out/` (gitignored): `top_truth.npz` (M, areas, V, T), `base_truth.npz`, `top_dec100k.npz`
  (decimated copy + `src` index into full-mesh triangles), `top_truth_preview.png`.

## Sections on Top (see reports/phase1.md for the table)
F1 dark red = tree AND posts (two distinct parts, one colour). F2 grey = railing loops. F3 yellow = rails
(top/bottom bars). F4 tan = deck + cut ends of posts/branches. Unpainted 9 % of area -> default F1 (verified).

## Open problems
- PoC never separates rails (F3) from loops (F2): F3 IoU 0. Rails are only 6.2 % of area but 13.7 % of their area
  is in brush-split triangles, so whole-triangle output will cap the rail score; sub-triangle output needed eventually.
- PoC "frame+posts" label mixes posts (F1) with deck frame (F4): costs F1 recall (77) and F4 precision (78).
- Whole-triangle painting makes jagged boundaries on long triangles (from CLAUDE.md).
- Scorer assumes product coupling when both sides are fractional; exact only for whole-triangle candidates.

## Next step
Phase 2 step C on approval: GBM on the features from click neighbourhoods -> per-face probabilities ->
alpha-expansion graph cut (PyMaxflow) with crease-aware pairwise cost + geodesic seed term. Score on the decimated
copy under per_region_1/2/3, per_filament, per_type, with and without height; then E (caps), F (full mesh +
sub-triangle refinement).
