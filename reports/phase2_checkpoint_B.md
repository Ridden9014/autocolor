# Phase 2 checkpoint B: features and separability

Reproduce: `python tools/phase2_features.py` (12 s for features on the 100k copy; cached in
`out/top_dec100k_features.npz`, `--recompute` to rebuild). Full table in `phase2_separability.txt`.

Correction for the record: the earlier "hollow" claim came from a leaky voxel fill in the chat analysis. The mesh
is solid (checkpoint A).

## Features (`autocolor/features.py`, per face, all in mm, no coordinates)
| feature | what |
|---|---|
| nz | normal z |
| thick_sdf, thick_min | shape-diameter: 12 inward rays in a 30 deg half-angle cone, cast **through cavities to the final exit** (so it holds on a hollowed mesh); median and min |
| tan_min / tan_med / tan_max, tan_aniso | **in-solid tangent extents**: from a point 1 mm inside the surface, exit distance along 4 tangent axes (both signs summed). Width of the part along the surface. tan_aniso = max/min |
| curv_3 | normal variation within 3 mm |
| lin/plan/sph_4, lin/plan/sph_12, ext_12 | PCA shape of centroids within 4 and 12 mm; ext_12 = sqrt(lambda1)/12 |
| height | z above the mesh's lowest point. **FLAGGED: assumes print orientation** |

## Separability: area-weighted ROC AUC, single features and combined
Part types come from the click simulator's region grouping (posts = F1 regions other than the tree, caps = F4
regions other than the deck). The combined rows are a gradient-boosted classifier trained on half the *regions*
of each side and tested on the other half (single-region sides split at their x median), both ways averaged, so
they measure generalization across parts, which is what a click on one post has to do for the other posts.

| | loops/rails | **rails/deck** | F1/deck | F1/loops | F1/rails | loops/deck | posts/deck | posts/rails | posts/tree | caps/F1 | caps/deck |
|---|---|---|---|---|---|---|---|---|---|---|---|
| best single, orientation-free | tan_max 0.961 | **tan_aniso 0.936** | thick_sdf 0.906 | curv_3 0.977 | sph_4 0.954 | curv_3 0.969 | sph_12 0.889 | tan_aniso 0.961 | tan_med 0.786 | ext_12 0.982 | sph_12 0.998 |
| height alone | 0.548 | 0.992 | 0.978 | 0.771 | 0.759 | 0.992 | 0.978 | 0.711 | 0.654 | 0.703 | 1.000 |
| all features (GBM) | 0.996 | **0.995** | 0.935 | 0.996 | 0.994 | 1.000 | 0.997 | 1.000 | 0.922 | 0.936 | 1.000 |
| all except height | 0.990 | **0.924** | 0.806 | 0.993 | 0.991 | 0.994 | 0.988 | 0.999 | 0.862 | 0.907 | 0.999 |

### Rails vs deck (both 5.8 mm thick)
Thickness is useless here (thick_sdf 0.61), as expected. What splits them, orientation-free:
- **tan_aniso 0.936**: a rail is a long bar; median tan_max 114 mm vs tan_min 4.7 (aniso 18.7). The deck is a slab:
  37 / 7.0 (aniso 4.4).
- **plan_12 0.934** and curv_3 0.902: within 12 mm a rail's neighbourhood is bar-like (planarity 0.24) while the
  deck is planar (0.58).
- tan_min 0.895: rails are narrower than the deck in their narrowest direction.
- height is 0.992 on its own (rails at 7.7 mm, deck top at 4.4 mm) and lifts the combined model from 0.924 to
  0.995. That is the one place the flagged feature matters a lot; on another model the rails could sit anywhere,
  so the method should not lean on it for rails/deck. The orientation-free combination at 0.92 is good but not
  the 0.99+ the other pairs get; the graph cut's smoothing and the seeds will have to supply the rest. Worth
  a second look in C: the deck *edge* faces (vertical, narrow) are the ones that look rail-like.

### Loops vs rails
tan_max 0.961 (loop ring extent 11 mm vs rail length 114 mm), tan_aniso 0.931, thickness 0.79. Combined 0.996,
0.990 without height. Clean.

### Posts
- posts vs deck: thick_sdf 0.867 and sph_12 0.889 single; combined 0.997 / 0.988 without height.
- posts vs rails: tan_aniso 0.961, tan_max 0.951; combined 1.000.
- **posts vs tree: the weak pair.** Best single 0.786 (tan_med), combined 0.922 / 0.862 without height. Posts and
  trunk are both thick, rounded, low curvature. This does not hurt the score (both are F1) but matters for the
  "one click per type" question: a post click and a tree click carry the same label, so the classifier does not
  need to tell them apart. It would matter if a user wanted posts a different colour.

### Caps (cut ends of posts and branches)
caps vs deck: sph_12 0.998, ext_12 0.997 (a cap is a small flat island at the end of a stump: the 12 mm
neighbourhood is blob-shaped, extent 0.30 vs 0.51). caps vs F1: ext_12 0.982, sph_12 0.964, curv_3 0.913.
Combined 0.936 across caps (6 regions, 969 faces; fold halves are 3 caps each, so this estimate is noisy).
So the cap detector in step E can be feature-based: small ext_12 + high sph_12 + low curv_3 + enclosed by a
convex crease; no hand rule on z or position.

## Click conditions (simulator, `clicks.py`)
- per_region_1/2/3: 35/70/105 clicks.
- per_filament: 4 clicks, one in the largest region of each filament (tree, a loop side, a rail, the deck).
- per_type: 6 clicks: tree, one post, one loop side, one rail, the deck, one cap. Slivers excluded.
Part-type grouping is by area within a filament (FLAGGED, simulator only, not used by the method).

## For step C
Combined features separate every section pair at >= 0.99 AUC across regions with height, >= 0.92 without. The
plan for C stands: GBM on these features from the click neighbourhoods -> per-face probabilities -> alpha-expansion
graph cut with crease-aware pairwise cost + geodesic seed term. Run both with and without height and report both.
Posts/tree and caps have few regions, so per_type is the condition to watch.

Stopping here for review before C.
