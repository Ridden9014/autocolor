"""Checkpoint previews: 2 orthographic views of a mesh with per-triangle
colours, rendered via matplotlib painter's algorithm on a decimated copy."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from scipy.spatial import cKDTree


def decimate(V, T, target=100_000):
    """Quadric decimation with open3d; returns (V2, T2, src) where src maps each
    decimated triangle to the nearest original triangle (by centroid)."""
    import open3d as o3d
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(T))
    m = m.simplify_quadric_decimation(target_number_of_triangles=int(target))
    V2, T2 = np.asarray(m.vertices), np.asarray(m.triangles)
    src = cKDTree(V[T].mean(1)).query(V2[T2].mean(1))[1]
    return V2, T2, src


def _rot(az, el):
    """Rotation taking world coords to view coords (z toward the viewer)."""
    az, el = np.radians(az), np.radians(el)
    Rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(el), -np.sin(el)], [0, np.sin(el), np.cos(el)]])
    return Rx @ Rz


def draw_view(ax, V, T, rgb, az, el, light=(0.3, 0.5, 0.8)):
    P = V @ _rot(az, el).T
    tri = P[T]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nn = np.linalg.norm(n, axis=1)
    n = n / np.maximum(nn, 1e-12)[:, None]
    front = n[:, 2] > 0
    shade = 0.45 + 0.55 * np.clip(n @ (np.array(light) / np.linalg.norm(light)), 0, 1)
    cols = np.clip(rgb * shade[:, None], 0, 1)
    order = np.argsort(tri[:, :, 2].mean(1))
    order = order[front[order]]
    pc = PolyCollection(tri[order][:, :, :2], facecolors=cols[order], edgecolors='none', antialiased=False)
    ax.add_collection(pc)
    ax.set_xlim(P[:, 0].min(), P[:, 0].max())
    ax.set_ylim(P[:, 1].min(), P[:, 1].max())
    ax.set_aspect('equal')
    ax.axis('off')


def preview(V, T, rgb, path, views=((35, -55), (215, -55)), width_px=1000, titles=None):
    """Two views side by side. rgb: (N,3) in 0..1 per triangle. Views are
    (azimuth, elevation) in degrees; elevation -55 looks down from above."""
    fig, axes = plt.subplots(1, len(views), figsize=(width_px / 100, width_px / 100 * 0.45), dpi=100)
    for ax, (az, el), i in zip(np.atleast_1d(axes), views, range(len(views))):
        draw_view(ax, V, T, rgb, az, el)
        if titles:
            ax.set_title(titles[i], fontsize=9)
    fig.tight_layout(pad=0.2)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def hex_to_rgb(h):
    h = h.lstrip('#')
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)]) / 255.0


def filament_rgb(M, colours):
    """Colour each triangle by its dominant filament in area matrix M (N,k)."""
    pal = np.array([hex_to_rgb(c) for c in colours])
    return pal[M.argmax(1)]
