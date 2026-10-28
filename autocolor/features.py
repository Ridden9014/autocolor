"""Per-face shape features for sectioning. All lengths in mm. Nothing here
uses model coordinates except `height` (z above the lowest surface), which
assumes the mesh is in print orientation -- FLAGGED.

Features (columns of the matrix returned by compute_features):
  nz            normal z (upward = 1)
  thick_sdf     shape-diameter: median exit distance of 12 inward rays in a
                60 deg cone about -n, cast through cavities to the final exit
  thick_min     shortest of those inward rays
  tan_min/med/max  in-solid tangent extents: from a point just inside the
                surface, exit distance along 8 tangent directions (how wide the
                part is along the surface; rail: small min, deck: large)
  tan_aniso     tan_max / tan_min (bars are anisotropic, slabs/blobs are not)
  curv_3        normal variation within 3 mm: 1 - |mean normal|
  lin_4 plan_4 sph_4   PCA shape of centroids within 4 mm
  lin_12 plan_12 sph_12  same at 12 mm
  ext_12        sqrt(lambda_1) / 12: how far the local surface extends
  height        z - z_min of the mesh (FLAGGED: orientation)
"""
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

NAMES = ['nz', 'thick_sdf', 'thick_min', 'tan_min', 'tan_med', 'tan_max', 'tan_aniso', 'curv_3',
         'lin_4', 'plan_4', 'sph_4', 'lin_12', 'plan_12', 'sph_12', 'ext_12', 'height']


class Caster:
    def __init__(self, V, T):
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh(
            o3d.core.Tensor(np.asarray(V, np.float32)), o3d.core.Tensor(np.asarray(T, np.int32))))
        a = V[T[:, 1]] - V[T[:, 0]]
        b = V[T[:, 2]] - V[T[:, 0]]
        n = np.cross(a, b)
        self.N = (n / np.maximum(np.linalg.norm(n, axis=1), 1e-12)[:, None]).astype(np.float32)

    def cast(self, o, d):
        r = np.concatenate([o, d], 1).astype(np.float32)
        h = self.scene.cast_rays(o3d.core.Tensor(r))
        return h['t_hit'].numpy(), h['primitive_ids'].numpy()

    def exit_distance(self, o, d, max_bounce=6):
        """Distance along d from o (inside the solid) to the final exit: a hit
        on a back face (normal along d) is an exit; a front-face hit means we
        entered a cavity wall, so continue from just past it."""
        o = o.astype(np.float32).copy()
        d = d.astype(np.float32)
        total = np.zeros(len(o), np.float32)
        alive = np.ones(len(o), bool)
        for _ in range(max_bounce):
            if not alive.any():
                break
            t, p = self.cast(o[alive], d[alive])
            hit = np.isfinite(t)
            idx = np.where(alive)[0]
            total[idx[hit]] += t[hit]
            total[idx[~hit]] = np.nan
            alive[idx[~hit]] = False
            back = np.einsum('ij,ij->i', self.N[p[hit].clip(0)], d[idx[hit]]) > 0
            done = idx[hit][back]
            alive[done] = False
            cont = idx[hit][~back]
            o[cont] += d[cont] * (t[hit][~back] + 1e-3)[:, None]
        total[alive] = np.nan
        return total


def _cone_dirs(n, k, half_angle_deg, rng):
    """k unit directions per row of n, within half_angle of -n."""
    u = np.cross(n, np.where(np.abs(n[:, :1]) < 0.9, [[1, 0, 0]], [[0, 1, 0]]))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    v = np.cross(n, u)
    phi = rng.uniform(0, 2 * np.pi, (len(n), k))
    cos_t = rng.uniform(np.cos(np.radians(half_angle_deg)), 1, (len(n), k))
    sin_t = np.sqrt(1 - cos_t ** 2)
    d = (-n[:, None] * cos_t[..., None] + (u[:, None] * np.cos(phi)[..., None] + v[:, None] * np.sin(phi)[..., None]) * sin_t[..., None])
    return d.astype(np.float32)


def _tangent_dirs(n, k):
    u = np.cross(n, np.where(np.abs(n[:, :1]) < 0.9, [[1, 0, 0]], [[0, 1, 0]]))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    v = np.cross(n, u)
    ang = np.linspace(0, np.pi, k, endpoint=False)  # directions and their opposites covered by both signs
    d = np.stack([u[:, None] * np.cos(a) + v[:, None] * np.sin(a) for a in ang], 1)
    return np.concatenate([d, -d], 1).astype(np.float32)  # (n, 2k, 3)


def _pca_feats(pts, tree, radius):
    """Linearity, planarity, sphericity and extent of points within radius."""
    nb = tree.query_ball_point(pts, radius, workers=-1)
    out = np.zeros((len(pts), 4))
    for i, ids in enumerate(nb):
        if len(ids) < 4:
            out[i] = [0, 0, 1, 0]
            continue
        q = tree.data[ids] - tree.data[ids].mean(0)
        lam = np.sort(np.linalg.eigvalsh(q.T @ q / len(ids)))[::-1]
        l1 = max(lam[0], 1e-12)
        out[i] = [(lam[0] - lam[1]) / l1, (lam[1] - lam[2]) / l1, lam[2] / l1, np.sqrt(l1) / radius]
    return out


def compute_features(V, T, seed=0, verbose=True):
    import time
    rng = np.random.default_rng(seed)
    t0 = time.time()
    C = V[T].mean(1).astype(np.float32)
    cast = Caster(V, T)
    N = cast.N
    n = len(T)
    F = {}
    F['nz'] = N[:, 2].astype(float)

    # shape diameter, 12 rays in 60 deg cone (half angle 30), through cavities
    k = 12
    d = _cone_dirs(N, k, 30, rng).reshape(-1, 3)
    o = np.repeat(C - 1e-3 * N, k, 0)
    ex = cast.exit_distance(o, d).reshape(n, k)
    ex = np.where(np.isnan(ex), np.nanmax(ex), ex)
    F['thick_sdf'] = np.median(ex, 1)
    F['thick_min'] = ex.min(1)
    if verbose:
        print(f'  sdf {time.time() - t0:.0f}s')

    # tangent extents from a point pushed inside by min(1 mm, thick_min/2)
    depth = np.minimum(1.0, F['thick_min'] / 2).astype(np.float32)
    kt = 8
    dt = _tangent_dirs(N, kt // 2).reshape(-1, 3)
    ot = np.repeat(C - depth[:, None] * N, kt, 0)
    tx = cast.exit_distance(ot, dt).reshape(n, kt)
    tx = np.where(np.isnan(tx), np.nanmax(tx), tx)
    # pair opposite directions: extent across = d(+) + d(-)
    across = tx[:, :kt // 2] + tx[:, kt // 2:]
    F['tan_min'] = across.min(1)
    F['tan_med'] = np.median(across, 1)
    F['tan_max'] = across.max(1)
    F['tan_aniso'] = F['tan_max'] / np.maximum(F['tan_min'], 0.1)
    if verbose:
        print(f'  tangent {time.time() - t0:.0f}s')

    tree = cKDTree(C)
    nb = tree.query_ball_point(C, 3.0, workers=-1)
    F['curv_3'] = np.array([1 - np.linalg.norm(N[ids].mean(0)) for ids in nb])
    for r, tag in ((4.0, '4'), (12.0, '12')):
        p = _pca_feats(C, tree, r)
        F['lin_' + tag], F['plan_' + tag], F['sph_' + tag] = p[:, 0], p[:, 1], p[:, 2]
        if tag == '12':
            F['ext_12'] = p[:, 3]
    if verbose:
        print(f'  pca {time.time() - t0:.0f}s')
    F['height'] = C[:, 2] - V[:, 2].min()   # FLAGGED: assumes print orientation
    X = np.column_stack([F[k] for k in NAMES])
    return X
