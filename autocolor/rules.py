"""Fixed post-labeling rules."""
import numpy as np
from scipy.sparse.csgraph import dijkstra


def bottom_surface(graph, V, T, down=(0, 0, -1), cos_min=0.7):
    """Faces that form the model's bottom surface: facing down (normal . down
    > cos_min) with nothing below them (a ray cast along 'down' from just
    outside the face hits nothing). Uses the print orientation (the 3MF's
    gravity direction), not model coordinates -- FLAGGED as orientation-
    dependent, nothing else."""
    from .features import Caster
    cast = Caster(V, T)
    d = np.asarray(down, np.float32)
    N = cast.N
    facing = N @ d > cos_min
    idx = np.where(facing)[0]
    o = graph.C[idx].astype(np.float32) + 1e-3 * N[idx]
    t, _ = cast.cast(o, np.repeat(d[None], len(idx), 0))
    free = ~np.isfinite(t)
    out = np.zeros(graph.n, bool)
    out[idx[free]] = True
    return out


def above_through_solid(graph, V, T, bottom, max_dist=7.0):
    """For each bottom face, the face its inward normal ray hits within
    max_dist (the surface directly above it through the slab), else -1."""
    from .features import Caster
    cast = Caster(V, T)
    idx = np.where(bottom)[0]
    N = cast.N[idx]
    t, p = cast.cast(graph.C[idx].astype(np.float32) - 1e-3 * N, -N)
    ok = np.isfinite(t) & (t < max_dist) & (p < graph.n)
    out = np.full(graph.n, -1, dtype=np.int64)
    out[idx[ok]] = p[ok]
    return out


def apply_bottom_rule(graph, labels, bottom, above, fallback_nearest=False):
    """Bottom-surface faces that have a surface directly above them through
    the solid within 12 mm (slab undersides: the deck underside takes the deck
    top's label, a rail's underside its own top) take that surface's label.
    Other bottom faces (feet of legs, undersides of branches, undersides of
    rails fused into the lattice) are left to the method, or, with
    fallback_nearest, take the nearest non-bottom face's label."""
    out = labels.copy()
    has = bottom & (above >= 0) & ~bottom[np.maximum(above, 0)]
    out[has] = labels[above[has]]
    if fallback_nearest:
        src = np.where(~bottom)[0]
        g = graph.sparse(graph.cdist)
        _, _, sources = dijkstra(g, directed=False, indices=src, min_only=True, return_predecessors=True)
        rest = bottom & ~has
        out[rest] = labels[sources[rest]]
    return out
