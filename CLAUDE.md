# autocolor

Auto-section 3D models into color regions for multi-tool FDM printing
(Snapmaker U1 now, Bambu 7-9 tool machines later). Output: 3MF with
per-triangle paint_color that Bambu Studio / Orca slicers read.

## Usage rules (important)
- Never print mesh file contents (OBJ, STL, 3MF XML) into the
  conversation. Work on them only through scripts; report stats.
- Iterate on a decimated copy (~100k triangles) for speed; confirm
  final results on the full mesh.
- Report scores as text. Render preview images only at checkpoints,
  max 2 views, about 1000 px wide.
- Keep STATUS.md current: phase, last score, open problems, next step.
  New sessions start by reading STATUS.md, not by re-reading the repo.

## Working rules
- Plan first for each phase; wait for approval before coding.
- Tag claims as VERIFIED (ran it, checked output) or INFERRED.
- The score is measured against refs/ hand paint, area-weighted per
  section. Report per-section scores, not just the overall number.
- Don't tune rules to one model's coordinates without flagging it;
  methods must carry over to new models.

## Known problems from the chat proof of concept
- Rails and loops in the railing must be separate sections.
- Post color bleeds onto rail ends and deck edges.
- Whole-triangle painting makes jagged boundaries on long triangles.
