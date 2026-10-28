"""Phase 3 / 1d prep for a model (default base = object 4): truth, visibility,
decimated copy + fractional truth, features, click regions, primitive fit.
Writes out/<model>_*.npz like the Top files. Nothing is tuned here."""
import argparse
import os
import sys
import time
import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autocolor.truth import Project, tri_areas, truth_matrix, paint_stats  # noqa: E402
from autocolor.render import decimate  # noqa: E402
from autocolor.transfer import truth_on_decimated  # noqa: E402
from autocolor.features import compute_features, NAMES  # noqa: E402
from autocolor.clicks import regions, boundary_hops, part_types  # noqa: E402
from autocolor.seeded import FaceGraph  # noqa: E402
from autocolor.primitives import fit_all  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='base')
args = ap.parse_args()
OID = {'top': 2, 'base': 4}[args.model]
P = Project('refs/TreetopDruidHaven_painted.3mf')
k = P.n_filaments
t0 = time.time()
V, T, paint = P.load_mesh(OID)
st = paint_stats(paint)
A = tri_areas(V, T)
M = truth_matrix(paint, A, k, P.default_extruder(OID))
print(f'{args.model}: {len(T)} tris, paint {st}, area % per filament {np.round(100 * M.sum(0) / A.sum(), 2)}, {time.time() - t0:.0f}s')
np.savez_compressed(f'out/{args.model}_truth.npz', M=M, areas=A, V=V, T=T)

# visibility (32 hemisphere dirs, then 512 on survivors)
import trimesh  # noqa: E402
m = trimesh.Trimesh(V, T, process=False)
scene = o3d.t.geometry.RaycastingScene()
scene.add_triangles(o3d.t.geometry.TriangleMesh(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(T.astype(np.int32))))
C = m.triangles_center.astype(np.float32); N = m.face_normals.astype(np.float32)
rng = np.random.default_rng(0); eps = 1e-3


def visible(idx, K, seed):
    r = np.random.default_rng(seed); dirs = r.normal(size=(K, 3)).astype(np.float32); dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    vis = np.zeros(len(idx), bool)
    for kk in range(K):
        dk = np.where((N[idx] @ dirs[kk])[:, None] > 0.02, dirs[kk], -dirs[kk]).astype(np.float32)
        rr = np.concatenate([C[idx] + eps * N[idx], dk], 1).astype(np.float32)
        vis |= ~np.isfinite(scene.cast_rays(o3d.core.Tensor(rr))['t_hit'].numpy())
    return vis


vis = visible(np.arange(len(T)), 32, 0)
cand = np.where(~vis)[0]
vis[cand[visible(cand, 512, 1)]] = True
print(f'invisible: {(~vis).sum()} faces, {100 * A[~vis].sum() / A.sum():.3f}% area')
np.save(f'out/{args.model}_visible.npy', vis)

t0 = time.time()
V2, T2, src = decimate(V, T, 100_000)
np.savez_compressed(f'out/{args.model}_dec100k.npz', V=V2, T=T2, src=src)
M2, A2 = truth_on_decimated(V, T, M, V2, T2)
np.savez_compressed(f'out/{args.model}_dec100k_truth.npz', M=M2, areas=A2)
print(f'decimated {len(T2)} tris, {time.time() - t0:.0f}s; decimated-space ceiling (majority): ', end='')
from autocolor.score import score  # noqa: E402
maj = M2.argmax(1) + 1
res = score(M2, np.eye(k)[maj - 1] * A2[:, None], [f'F{i + 1}' for i in range(k)], mode='identity')
print(f"overall {100 * res['overall']:.1f} IoU " + ' '.join(f"{100 * p['iou']:.1f}" for p in res['per_section']))

t0 = time.time()
X = compute_features(V2, T2, verbose=False)
np.savez_compressed(f'out/{args.model}_dec100k_features.npz', X=X, names=np.array(NAMES))
print(f'features {time.time() - t0:.0f}s')

lab = M.argmax(1) + 1
comp, table, adj = regions(T, lab, A, min_area=20.0)
hops = boundary_hops(adj, lab, len(T))
np.savez_compressed(f'out/{args.model}_regions.npz', comp=comp, hops=hops,
                    table=np.array([(r['id'], r['label'], r['area'], r['n_faces']) for r in table]))
types = part_types(table)
from collections import Counter  # noqa: E402
print(f'regions >= 20 mm2: {len(table)}; part types: {dict(Counter(types.values()))}')
for f in range(1, k + 1):
    ar = sorted([r['area'] for r in table if r['label'] == f], reverse=True)
    print(f'  F{f}: {len(ar)} regions, areas: {np.round(ar[:10]).astype(int).tolist()}{"..." if len(ar) > 10 else ""}')

t0 = time.time()
graph = FaceGraph(V2, T2)
prims, owner = fit_all(graph, X, NAMES)
truth2 = M2.argmax(1) + 1
print(f'primitives {len(prims)} in {time.time() - t0:.0f}s, claimed {100 * A2[owner >= 0].sum() / A2.sum():.0f}% of area')
for i, p in enumerate(prims):
    cl = p['claim']; a = A2[cl].sum()
    comp_s = ' '.join(f'F{f}:{100 * A2[cl & (truth2 == f)].sum() / max(a, 1e-9):.0f}%' for f in range(1, k + 1))
    extra = f"thick {p['thickness']:.1f}" if p['kind'] == 'slab' else (f"r {p['r']:.1f} h {p['h1'] - p['h0']:.0f}" if p['kind'] == 'cyl' else f"span {p['s1'] - p['s0']:.0f}")
    print(f'  {i:2d} {p["kind"]:4s} {extra:18s} {a:7.0f} mm2  truth {comp_s}')
