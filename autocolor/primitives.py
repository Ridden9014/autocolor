"""Primitive fitting for man-made parts: a slab (two parallel planes), vertical
cylinders (posts) and long thin boxes (rails), fitted by RANSAC on face
centroids/normals. Faces on a fitted primitive's surface within the texture
depth are claimed by it. Organic parts are left unclaimed.

Nothing uses model coordinates: the slab is found as the largest planar
pair, cylinder axes are the slab normal, rails are lines parallel to the slab.
Scales (mm) are print-physical: texture depth, thickness ranges, minimum
areas; FLAGGED as tuned on Top.
"""
import numpy as np
from scipy.spatial import cKDTree


def _plane_inliers(C, N, A, p0, n, tol, cos_min):
    d = (C - p0) @ n
    return (np.abs(d) < tol) & (np.abs(N @ n) > cos_min)


def fit_slab(C, N, A, rng, tol=1.0, cos_min=0.9, iters=300, thick=(2.0, 15.0), texture=2.5):
    """Largest plane by inlier area, then the largest parallel plane at a
    distance within `thick`. Returns dict(n, top, bottom, claim mask)."""
    best = None
    for _ in range(iters):
        f = rng.choice(len(C), p=A / A.sum())
        inl = _plane_inliers(C, N, A, C[f], N[f], tol, cos_min)
        a = A[inl].sum()
        if best is None or a > best[0]:
            best = (a, f, inl)
    a, f, inl = best
    n = N[inl].mean(0); n /= np.linalg.norm(n)
    p0 = (C[inl] * A[inl, None]).sum(0) / A[inl].sum()
    # second plane: parallel, opposite normal, at distance in `thick`
    d = (C - p0) @ n
    anti = (N @ n < -cos_min)
    best2 = None
    for dist in np.arange(-thick[1], thick[1] + 0.25, 0.25):
        if abs(dist) < thick[0]:
            continue
        m = anti & (np.abs(d - dist) < tol)
        a2 = A[m].sum()
        if best2 is None or a2 > best2[0]:
            best2 = (a2, dist, m)
    a2, dist, inl2 = best2
    lo, hi = sorted((0.0, dist))
    # claim: between the planes (+ texture) and inside the 2D hull of both planes' inliers
    u = np.cross(n, [1, 0, 0] if abs(n[0]) < 0.9 else [0, 1, 0]); u /= np.linalg.norm(u); v = np.cross(n, u)
    P2 = np.c_[C @ u, C @ v]
    from scipy.spatial import Delaunay
    hull_pts = P2[inl | inl2]
    hull = Delaunay(hull_pts[np.random.default_rng(0).choice(len(hull_pts), min(len(hull_pts), 4000), replace=False)])
    inside = hull.find_simplex(P2) >= 0
    band = (d > lo - texture) & (d < hi + texture)
    claim = band & inside
    core = claim & (np.abs(N @ n) > 0.8) & ((np.abs(d - lo) < 1.5) | (np.abs(d - hi) < 1.5))
    return dict(kind='slab', n=n, p0=p0, lo=lo, hi=hi, thickness=hi - lo, claim=claim, core=core, top_area=a, bottom_area=a2)


def fit_cylinders(C, N, A, axis, exclude, rng, r_range=(3.0, 15.0), tol=1.0, cos_rad=0.8, min_area=300.0,
                  min_height=15.0, iters=400, max_n=12, texture=2.5, side_cos=0.35):
    """Greedy RANSAC circles in the plane perpendicular to `axis`, on side-facing
    faces not excluded. Returns list of dict(kind='cyl', c2, r, h0, h1, claim)."""
    u = np.cross(axis, [1, 0, 0] if abs(axis[0]) < 0.9 else [0, 1, 0]); u /= np.linalg.norm(u); v = np.cross(axis, u)
    P2 = np.c_[C @ u, C @ v]; N2 = np.c_[N @ u, N @ v]; h = C @ axis
    cand = (np.abs(N @ axis) < side_cos) & ~exclude
    avail = cand.copy()
    out = []
    tree = cKDTree(P2)
    for _ in range(max_n):
        best = None
        idx = np.where(avail)[0]
        if len(idx) < 50:
            break
        for _ in range(iters):
            f = rng.choice(idx, p=A[idx] / A[idx].sum())
            nb = [i for i in tree.query_ball_point(P2[f], 2 * r_range[1]) if avail[i]]
            if len(nb) < 3:
                continue
            s = rng.choice(nb, 3, replace=False)
            # circle through 3 points
            (x1, y1), (x2, y2), (x3, y3) = P2[s]
            D = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
            if abs(D) < 1e-6:
                continue
            cx = ((x1 ** 2 + y1 ** 2) * (y2 - y3) + (x2 ** 2 + y2 ** 2) * (y3 - y1) + (x3 ** 2 + y3 ** 2) * (y1 - y2)) / D
            cy = ((x1 ** 2 + y1 ** 2) * (x3 - x2) + (x2 ** 2 + y2 ** 2) * (x1 - x3) + (x3 ** 2 + y3 ** 2) * (x2 - x1)) / D
            r = np.hypot(x1 - cx, y1 - cy)
            if not (r_range[0] <= r <= r_range[1]):
                continue
            near = np.array(tree.query_ball_point([cx, cy], r + tol))
            near = near[avail[near]]
            rad = P2[near] - [cx, cy]
            dist = np.linalg.norm(rad, axis=1)
            m = (np.abs(dist - r) < tol) & (np.einsum('ij,ij->i', N2[near], rad / np.maximum(dist, 1e-9)[:, None]) > cos_rad)
            inl = near[m]
            if len(inl) < 10:
                continue
            a = A[inl].sum()
            if best is None or a > best[0]:
                best = (a, cx, cy, r, inl)
        if best is None or best[0] < min_area:
            break
        a, cx, cy, r, inl = best
        h0, h1 = np.percentile(h[inl], [2, 98])
        if h1 - h0 < min_height:
            avail[inl] = False   # a short arc (tree bark), drop and continue
            continue
        # claim: radially within texture of the surface, side-facing, within height
        near = np.array(tree.query_ball_point([cx, cy], r + texture))
        rad = P2[near] - [cx, cy]; dist = np.linalg.norm(rad, axis=1)
        m = (np.abs(dist - r) < texture) & (h[near] > h0 - 2) & (h[near] < h1 + 2) & (np.abs(N[near] @ axis) < 0.7) & ~exclude[near]
        claim = np.zeros(len(C), bool); claim[near[m]] = True
        core = np.zeros(len(C), bool); core[inl] = True
        avail &= ~claim
        out.append(dict(kind='cyl', c2=(cx, cy), r=r, h0=h0, h1=h1, claim=claim, core=core, area=a))
    return out


def fit_rails(C, N, A, axis, cand, exclude, rng, width=4.5, min_area=150.0, min_span=40.0, iters=600,
              max_n=12, dir_cos=0.25, texture=1.0, min_claim=350.0):
    """Greedy RANSAC 3D lines parallel to the slab through bar-like candidate
    faces. Inliers: within `width` of the line, normal perpendicular to it.
    Returns list of dict(kind='rail', p, d, s0, s1, claim)."""
    avail = cand & ~exclude
    out = []
    tree = cKDTree(C)
    for _ in range(max_n):
        idx = np.where(avail)[0]
        if len(idx) < 20:
            break
        best = None
        for _ in range(iters):
            f = rng.choice(idx, p=A[idx] / A[idx].sum())
            nb = np.array([i for i in tree.query_ball_point(C[f], 60.0) if avail[i] and i != f])
            if len(nb) == 0:
                continue
            g = rng.choice(nb)
            dvec = C[g] - C[f]
            L = np.linalg.norm(dvec)
            if L < 10:
                continue
            dvec /= L
            if abs(dvec @ axis) > dir_cos:
                continue
            rel = C[idx] - C[f]
            s = rel @ dvec
            perp = np.linalg.norm(rel - s[:, None] * dvec, axis=1)
            m = (perp < width) & (np.abs(N[idx] @ dvec) < 0.35)
            inl = idx[m]
            if len(inl) < 10:
                continue
            a = A[inl].sum()
            if best is None or a > best[0]:
                best = (a, C[f], dvec, inl)
        if best is None or best[0] < min_area:
            break
        a, p, dvec, inl = best
        s = (C[inl] - p) @ dvec
        s0, s1 = np.percentile(s, [1, 99])
        if s1 - s0 < min_span:
            avail[inl] = False
            continue
        rel = C - p; ss = rel @ dvec; perp = np.linalg.norm(rel - ss[:, None] * dvec, axis=1)
        # claim only bar-like faces (cand) along the line: the loop row fused under a rail is not a rail
        claim = (perp < width + texture) & (ss > s0 - 2) & (ss < s1 + 2) & (np.abs(N @ dvec) < 0.5) & ~exclude & cand
        avail &= ~claim
        if A[claim].sum() < min_claim:
            continue   # a sparse line through loop-row faces, not a bar
        core = claim & (perp < width)
        out.append(dict(kind='rail', p=p, d=dvec, s0=s0, s1=s1, claim=claim, core=core, area=a))
    return out


def fit_all(graph, X, names, seed=0, rail_aniso=10.0):
    """Run slab -> cylinders -> rails. Returns list of primitives and a per-face
    primitive index (-1 = unclaimed)."""
    rng = np.random.default_rng(seed)
    C, N, A = graph.C, graph.N, graph.A
    slab = fit_slab(C, N, A, rng)
    prims = [slab]
    excl = slab['claim'].copy()
    thick = X[:, names.index('thick_sdf')]
    cyls = fit_cylinders(C, N, A, slab['n'], excl | (thick < 5.0) | (thick > 30.0), rng)
    for c in cyls:
        excl |= c['claim']
    prims += cyls
    aniso = X[:, names.index('tan_aniso')]
    cand = (aniso > rail_aniso) & (thick > 3.5) & (thick < 9.0)
    rails = fit_rails(C, N, A, slab['n'], cand, excl, rng)
    prims += rails
    owner = np.full(graph.n, -1)
    core_owner = np.full(graph.n, -1)
    for i, p in enumerate(prims):
        owner[p['claim'] & (owner < 0)] = i
        core_owner[p['core'] & (core_owner < 0)] = i
    for p in prims:
        p['core_owner'] = core_owner
    return prims, owner
