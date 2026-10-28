"""Full-mesh finishing: transfer decimated labels, snap boundaries with a
full-resolution graph cut, refine boundary triangles into sub-triangles, and
export paint_color strings into a copy of a Bambu project."""
import os
import re
import time
import zipfile
import numpy as np
from scipy.spatial import cKDTree
from .seeded import FaceGraph, alpha_expansion
from .transfer import labels_to_full
from .paint import build_tree, encode_tree, encode_state


def full_cut(graph, labels, k, mu=2.0, lam=3.0, sigma_deg=35.0, concave_bonus=0.8, cycles=2):
    """Potts cut on the full mesh: unary mu for leaving the transferred label,
    crease-aware pairwise (no features needed)."""
    U = np.full((graph.n, k), mu)
    U[np.arange(graph.n), labels - 1] = 0.0
    sharp = np.exp(-(np.degrees(graph.angle) / sigma_deg) ** 2)
    g = sharp + (1 - sharp) * np.where(graph.convex, 1 - concave_bonus * 0.5, 1 - concave_bonus)
    W = lam * graph.elen * g
    return alpha_expansion(U, graph.adj, W, init=labels - 1, cycles=cycles) + 1


def refine_boundaries(graph, V, T, labels, max_depth=2, sigma=0.4, knn=8, verbose=True):
    """paint strings per triangle: whole-triangle codes everywhere, split trees
    on triangles that touch a different label. The label at a point is the
    Gaussian-weighted (sigma mm) area vote of the knn nearest face centroids."""
    i, j = graph.adj[:, 0], graph.adj[:, 1]
    bnd = np.zeros(graph.n, bool)
    diff = labels[i] != labels[j]
    bnd[i[diff]] = True
    bnd[j[diff]] = True
    tree = cKDTree(graph.C)
    A = graph.A

    def label_at(P):
        d, idx = tree.query(P, k=knn)
        w = A[idx] * np.exp(-(d ** 2) / (2 * sigma * sigma))
        votes = np.zeros((len(P), labels.max() + 1))
        for c in range(knn):
            np.add.at(votes, (np.arange(len(P)), labels[idx[:, c]]), w[:, c])
        return votes.argmax(1)

    paint = [encode_state(int(l)) for l in labels]
    bidx = np.where(bnd)[0]
    t0 = time.time()
    n_split = 0
    for n, f in enumerate(bidx):
        Vt = V[T[f]]
        t = build_tree(lambda b: label_at(b @ Vt), max_depth=max_depth)
        if not isinstance(t, int):
            n_split += 1
        paint[f] = encode_tree(t) if not isinstance(t, int) else encode_state(t)
        if verbose and n % 10000 == 0 and n:
            print(f'  refined {n}/{len(bidx)} boundary triangles, {n_split} split, {time.time() - t0:.0f}s')
    if verbose:
        print(f'  boundary triangles {len(bidx)}, split {n_split}, {time.time() - t0:.0f}s')
    return paint


def export_project(src_3mf, out_path, part_file, paint):
    """Copy a Bambu project, replacing the paint_color strings of the given
    object part file (triangle order preserved). out_path must be outside
    refs/."""
    assert not os.path.abspath(out_path).startswith(os.path.abspath('refs')), 'never write into refs/'
    zin = zipfile.ZipFile(src_3mf)
    s = zin.read(part_file).decode()
    it = iter(paint)
    tri_re = re.compile(r'<triangle v1="(\d+)" v2="(\d+)" v3="(\d+)"[^/]*/>')

    def sub(m):
        p = next(it)
        if p:
            return f'<triangle v1="{m.group(1)}" v2="{m.group(2)}" v3="{m.group(3)}" paint_color="{p}"/>'
        return f'<triangle v1="{m.group(1)}" v2="{m.group(2)}" v3="{m.group(3)}"/>'

    s2, n = tri_re.subn(sub, s)
    assert n == len(paint), (n, len(paint))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with zipfile.ZipFile(out_path, 'w', zipfile.ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            data = s2.encode() if info.filename == part_file else zin.read(info.filename)
            zout.writestr(info, data)
    return out_path
