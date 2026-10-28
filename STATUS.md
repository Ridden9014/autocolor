# STATUS

**Phase:** 0 complete (repo setup + input inventory). Phase 1 not started; awaiting plan approval.
**Last score:** none yet (no scoring code).
**Env:** Python 3.13.3, `.venv` (trimesh 5.2.0, numpy 2.5.4, scipy 1.18.1, open3d 0.20.0). `requirements.txt` unpinned.

## Inputs (refs/)
- `SD_RPG_TreetopDruidHavenDiceTower-Top.obj`: 500,088 verts, 1,000,954 tris. Bounds min (34.768, 64.485, 0) max (265.232, 235.515, 123.023); extents 230.465 x 171.029 x 123.023 mm. (VERIFIED)
- `TreetopDruidHaven_painted.3mf` (BambuStudio 2.2.4), 3 objects, all with identity component transforms:
  | obj (component) | name | tris | extents (local) | paint_color |
  |---|---|---|---|---|
  | 2 (1) | Top.stl | 1,000,954 | 230.465 x 171.029 x 123.023 | YES, 866,993 tris (86.6%) |
  | 4 (3) | Base.stl | 1,000,946 | 187.5 x 181.83 x 160.41 | YES, 446,353 tris (44.6%) |
  | 6 (5) | Base.stl | 1,000,793 | 187.5 x 181.83 x 160.41 | NO (0 tris) |
  Files: `3D/Objects/...Top.stl_1.model`, `...Base.stl_2.model`, `...Base.stl_3.model`.
- **Top-piece ground truth = object 2 / mesh id 1.** Triangle count identical to the OBJ and triangle order identical: same-index centroids match (max 8.7e-6 mm) after adding offset (150, 150, 61.5113) to the 3MF's local coords. So paint_color can be mapped to OBJ triangles by index. (VERIFIED)
- Base: two copies; only object 4 is painted, object 6 is an unpainted duplicate. Base not needed unless we add it as a second test model.
- Not decoded yet: paint_color strings -> sections/filaments (Phase 1).
- `TreetopDruidHaven-Top_autocolor_poc.3mf`: PoC export, not inspected in Phase 0.

## Layout
- `refs/` is gitignored (111 MB, third-party model; 60 MB 3MF is near GitHub limits). Not pushed; copy it manually on a new machine.
- `poc/autocolor/`: PoC scripts, copied verbatim from refs. They read from `/mnt/user-data/uploads/...` (chat paths) and use model-specific seed coordinates; reference only.
- `tools/phase0_inspect.py`: OBJ stats + 3MF object listing.

## Open problems (from CLAUDE.md)
- Rails and loops in the railing must be separate sections.
- Post color bleeds onto rail ends and deck edges.
- Whole-triangle painting gives jagged boundaries on long triangles.

## Next step
Phase 1 plan: decode paint_color from the painted Top -> per-section reference labels and a scorer (area-weighted per section). Plan first, wait for approval.
