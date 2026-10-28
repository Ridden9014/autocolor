"""Move labels/truth between the full mesh and a decimated copy."""
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from .paint import sample_points


def truth_on_decimated(V, T, M, V2, T2, n_samples=16):
    """Fractional truth (N2, k) for decimated triangles: sample points on each
    decimated triangle, take the truth fractions of the nearest full-mesh
    triangle centroid, average. Returns areas-weighted matrix like M."""
    frac = M / np.maximum(M.sum(1, keepdims=True), 1e-300)
    tree = cKDTree(V[T].mean(1))
    pts = sample_points(n_samples, seed=11)                    # (s,3) bary
    P = np.einsum('sj,njd->nsd', pts, V2[T2])                   # (N2,s,3)
    idx = tree.query(P.reshape(-1, 3), workers=-1)[1].reshape(len(T2), n_samples)
    F2 = frac[idx].mean(1)
    a = V2[T2[:, 1]] - V2[T2[:, 0]]
    b = V2[T2[:, 2]] - V2[T2[:, 0]]
    A2 = 0.5 * np.linalg.norm(np.cross(a, b), axis=1)
    return F2 * A2[:, None], A2


class SurfaceLookup:
    """Closest-point queries against a mesh (open3d raycasting scene)."""

    def __init__(self, V, T):
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh(
            o3d.core.Tensor(np.asarray(V, np.float32)), o3d.core.Tensor(np.asarray(T, np.int32))))

    def face_at(self, pts):
        r = self.scene.compute_closest_points(o3d.core.Tensor(np.asarray(pts, np.float32)))
        return r['primitive_ids'].numpy().astype(np.int64)


def labels_to_full(V2, T2, labels2, V, T):
    """Per-full-triangle label = label of the decimated face closest to its
    centroid."""
    return labels2[SurfaceLookup(V2, T2).face_at(V[T].mean(1))]


def label_fn_from_decimated(V2, T2, labels2):
    """Return f(P_world (n,3)) -> labels, for sub-triangle refinement."""
    look = SurfaceLookup(V2, T2)
    return lambda pts: labels2[look.face_at(pts)]
