"""Rooftop (no ground truth): working resolution -> features -> over-segmentation
-> numbered views. Nothing tuned; Top's hyper-parameters as they are.

  python tools/rooftop_parts.py res                 # decimate 200k/300k/400k, deviation + timing
  python tools/rooftop_parts.py parts --n 300000    # visibility, features, oversegment, primitives
  python tools/rooftop_parts.py views --n 300000    # 4 numbered views to reports/
"""
import argparse
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autocolor.meshio import load_stl  # noqa: E402
from autocolor.render import decimate  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('stage', choices=['res', 'parts', 'views'])
ap.add_argument('--stl', default='refs/SD_RPG_WizardsArcaneTreasuryDiceTower-Rooftop.stl')
ap.add_argument('--tag', default='rooftop')
ap.add_argument('--n', type=int, default=300_000)
ap.add_argument('--clusters', type=int, default=16)
ap.add_argument('--seg-lam', type=float, default=2.0)
ap.add_argument('--min-part', type=float, default=30.0, help='mm2; smaller segments merge into a neighbour')
ap.add_argument('--height', action='store_true', help='include the FLAGGED height feature')
ap.add_argument('--sweep', action='store_true', help='print part counts over clusters x min-part, write nothing')
args = ap.parse_args()
os.makedirs('out', exist_ok=True)
os.makedirs('reports', exist_ok=True)

t0 = time.time()
V, T, good = load_stl(args.stl)
A_full = 0.5 * np.linalg.norm(np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]), axis=1)
print(f'{args.tag}: {len(T)} tris ({(~good).sum()} degenerate), {len(V)} merged verts, '
      f'area {A_full.sum():.0f} mm2, extent {np.round(V.max(0) - V.min(0), 1)}, load {time.time() - t0:.1f}s')


def dist_to_mesh(P, V2, T2):
    import open3d as o3d
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.t.geometry.TriangleMesh(o3d.core.Tensor(V2.astype(np.float32)), o3d.core.Tensor(T2.astype(np.int32))))
    return sc.compute_distance(o3d.core.Tensor(P.astype(np.float32))).numpy()


if args.stage == 'res':
    C = V[T].mean(1)
    for n in (200_000, 300_000, 400_000):
        t1 = time.time()
        V2, T2, src = decimate(V, T[good], n)
        td = time.time() - t1
        d = dist_to_mesh(C, V2, T2)
        e = np.linalg.norm(V2[T2[:, 1]] - V2[T2[:, 0]], axis=1)
        w = A_full / A_full.sum()
        print(f'{n // 1000}k: {len(T2)} tris in {td:.0f}s; full->dec distance: area within 0.25 mm '
              f'{100 * w[d <= 0.25].sum():.2f}%, within 0.5 mm {100 * w[d <= 0.5].sum():.2f}%, '
              f'p99 {np.percentile(d, 99):.2f} max {d.max():.2f} mm; dec edge median {np.median(e):.2f} p90 {np.percentile(e, 90):.2f}')
        np.savez_compressed(f'out/{args.tag}_dec{n // 1000}k.npz', V=V2, T=T2, src=src)
        from autocolor.features import compute_features, NAMES
        t1 = time.time()
        X = compute_features(V2, T2, verbose=False)
        print(f'     features {time.time() - t1:.0f}s')
        np.savez_compressed(f'out/{args.tag}_dec{n // 1000}k_features.npz', X=X, names=np.array(NAMES))
    sys.exit(0)

tagn = f'{args.tag}_dec{args.n // 1000}k'
z = np.load(f'out/{tagn}.npz')
V2, T2, src = z['V'], z['T'], z['src']

if args.stage == 'parts':
    import open3d as o3d
    import trimesh
    from autocolor.features import compute_features, NAMES
    from autocolor.seeded import FaceGraph, prep_features, oversegment
    from autocolor.primitives import fit_all
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    m = trimesh.Trimesh(V2, T2, process=False)
    A2 = m.area_faces
    # visibility on the working copy (same test as phase3_base_prep)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh(o3d.core.Tensor(V2.astype(np.float32)), o3d.core.Tensor(T2.astype(np.int32))))
    C = m.triangles_center.astype(np.float32); N = m.face_normals.astype(np.float32)

    def visible(idx, K, seed):
        r = np.random.default_rng(seed); dirs = r.normal(size=(K, 3)).astype(np.float32); dirs /= np.linalg.norm(dirs, axis=1)[:, None]
        vis = np.zeros(len(idx), bool)
        for kk in range(K):
            dk = np.where((N[idx] @ dirs[kk])[:, None] > 0.02, dirs[kk], -dirs[kk]).astype(np.float32)
            rr = np.concatenate([C[idx] + 1e-3 * N[idx], dk], 1).astype(np.float32)
            vis |= ~np.isfinite(scene.cast_rays(o3d.core.Tensor(rr))['t_hit'].numpy())
        return vis

    t1 = time.time()
    vis = visible(np.arange(len(T2)), 32, 0)
    cand = np.where(~vis)[0]
    vis[cand[visible(cand, 512, 1)]] = True
    print(f'invisible: {(~vis).sum()} faces, {100 * A2[~vis].sum() / A2.sum():.2f}% area, {time.time() - t1:.0f}s')
    np.save(f'out/{tagn}_visible.npy', vis)

    fp = f'out/{tagn}_features.npz'
    if os.path.exists(fp):
        X = np.load(fp)['X']
    else:
        t1 = time.time()
        X = compute_features(V2, T2, verbose=False)
        np.savez_compressed(fp, X=X, names=np.array(NAMES))
        print(f'features {time.time() - t1:.0f}s')

    t1 = time.time()
    graph = FaceGraph(V2, T2)
    print(f'graph {time.time() - t1:.0f}s')
    cols = [j for j in range(len(NAMES)) if args.height or NAMES[j] != 'height']
    names = [NAMES[j] for j in cols]
    _, Z = prep_features(X[:, cols], names)
    i, j = graph.adj[:, 0], graph.adj[:, 1]

    def merge_small(seg, min_part):
        """Greedy region merging: smallest segment first, into the neighbour
        sharing the longest boundary (union-find, so results stay connected).
        Stops when every segment with a neighbour is >= min_part mm2. Returns
        (seg sorted by area, 0 = largest; areas)."""
        import heapq
        same = seg[i] == seg[j]
        g = coo_matrix((np.ones(same.sum()), (i[same], j[same])), shape=(graph.n, graph.n))
        seg = connected_components(g, directed=False)[1]
        ns = seg.max() + 1
        area = np.bincount(seg, weights=A2, minlength=ns)
        cross = seg[i] != seg[j]
        a, b, L = seg[i][cross], seg[j][cross], graph.elen[cross]
        nbr = [dict() for _ in range(ns)]
        for x, y, w in zip(a.tolist(), b.tolist(), L.tolist()):
            nbr[x][y] = nbr[x].get(y, 0.0) + w
            nbr[y][x] = nbr[y].get(x, 0.0) + w
        parent = np.arange(ns)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        heap = [(area[s], s) for s in range(ns) if area[s] < min_part]
        heapq.heapify(heap)
        while heap:
            ar, s = heapq.heappop(heap)
            if find(s) != s or area[s] != ar or area[s] >= min_part or not nbr[s]:
                continue
            t = max(nbr[s].items(), key=lambda kv: kv[1])[0]
            # merge s into t
            parent[s] = t
            area[t] += area[s]
            for y, w in nbr[s].items():
                if y == t:
                    continue
                nbr[t][y] = nbr[t].get(y, 0.0) + w
                nbr[y][t] = nbr[y].get(t, 0.0) + w
                del nbr[y][s]
            del nbr[t][s]
            nbr[s] = {}
            if area[t] < min_part:
                heapq.heappush(heap, (area[t], t))
        root = np.array([find(s) for s in range(ns)])
        seg = np.unique(root[seg], return_inverse=True)[1]
        area = np.bincount(seg, weights=A2)
        order = np.argsort(-area)
        return np.argsort(order)[seg], area[order]     # 0 = largest

    if args.sweep:
        for nc in (4, 6, 8, 12, 16):
            raw = oversegment(graph, Z, names, n_clusters=nc, lam=args.seg_lam)
            for mp in (200, 400, 800, 1500, 3000):
                seg, area = merge_small(raw, mp)
                big = int((area >= 20.0).sum())
                print(f'clusters {nc:2d} min-part {mp:4d}: raw {raw.max() + 1:6d} -> {seg.max() + 1:5d} components, '
                      f'{big:4d} >= 20 mm2 covering {100 * area[:big].sum() / A2.sum():.1f}%, largest {area[0]:.0f} mm2')
        sys.exit(0)

    t1 = time.time()
    raw = oversegment(graph, Z, names, n_clusters=args.clusters, lam=args.seg_lam)
    print(f'oversegment: {raw.max() + 1} raw segments in {time.time() - t1:.0f}s')
    seg, area = merge_small(raw, args.min_part)
    big = int((area >= 20.0).sum())
    print(f'parts after merge (min-part {args.min_part}): {seg.max() + 1} components, {big} of them >= 20 mm2; '
          f'covering {100 * area[:big].sum() / A2.sum():.1f}% of area')

    t1 = time.time()
    prims, owner = fit_all(graph, X, NAMES)
    print(f'primitives {len(prims)} in {time.time() - t1:.0f}s, claimed {100 * A2[owner >= 0].sum() / A2.sum():.0f}% of area')
    for pi, p in enumerate(prims):
        cl = p['claim']; a = A2[cl].sum()
        extra = f"thick {p['thickness']:.1f}" if p['kind'] == 'slab' else (f"r {p['r']:.1f} h {p['h1'] - p['h0']:.0f}" if p['kind'] == 'cyl' else f"span {p['s1'] - p['s0']:.0f}")
        parts_in = np.bincount(seg[cl], weights=A2[cl], minlength=seg.max() + 1)
        top = np.argsort(-parts_in)[:3]
        print(f'  {pi:2d} {p["kind"]:4s} {extra:18s} {a:7.0f} mm2  parts ' + ' '.join(f'#{t}:{100 * parts_in[t] / a:.0f}%' for t in top if parts_in[t] > 0))
    np.savez_compressed(f'out/{tagn}_parts.npz', seg=seg, owner=owner, area=area, vis=vis,
                        prim_kind=np.array([p['kind'] for p in prims]))

    # per-part table
    lines = ['| part | area mm2 | faces | nz | thick | tan_aniso | curv_3 | plan_12 | primitive |', '|---|---|---|---|---|---|---|---|---|']
    kinds = [p['kind'] for p in prims]
    for s in range(big):
        f = seg == s; w = A2[f] / A2[f].sum()
        def fm(nm):
            return (X[f, NAMES.index(nm)] * w).sum()
        o = owner[f]; o = o[o >= 0]
        pr = '-' if len(o) == 0 else ', '.join(f'{kinds[k]}{k}' for k in np.unique(o) if (o == k).mean() > 0.2)
        lines.append(f'| {s} | {area[s]:.0f} | {f.sum()} | {fm("nz"):.2f} | {fm("thick_sdf"):.1f} | {fm("tan_aniso"):.1f} | {fm("curv_3"):.2f} | {fm("plan_12"):.2f} | {pr} |')
    with open(f'reports/{args.tag}_parts_table.md', 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    sys.exit(0)

if args.stage == 'views':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from autocolor.render import _rot
    import trimesh

    zp = np.load(f'out/{tagn}_parts.npz')
    seg, area = zp['seg'], zp['area']
    big = int((area >= 20.0).sum())
    rng = np.random.default_rng(3)
    pal = rng.uniform(0.25, 0.95, size=(seg.max() + 1, 3))
    # keep neighbouring part colours apart: re-draw colours that are too close to any earlier part
    rgb = pal[seg]
    m = trimesh.Trimesh(V2, T2, process=False)
    views = [('front az0 el-20', 0, -20), ('back az180 el-20', 180, -20), ('3/4 from above az225 el-45', 225, -45), ('from below az45 el-130', 45, -130)]
    W = 1000
    for vi, (title, az, el) in enumerate(views):
        R = _rot(az, el)
        P = V2 @ R.T
        tri = P[T2]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); nn = np.linalg.norm(n, axis=1); n /= np.maximum(nn, 1e-12)[:, None]
        front = n[:, 2] > 0
        light = np.array([0.3, 0.5, 0.8]); light /= np.linalg.norm(light)
        shade = 0.5 + 0.5 * np.clip(n @ light, 0, 1)
        cols = np.clip(rgb * shade[:, None], 0, 1)
        order = np.argsort(tri[:, :, 2].mean(1)); order = order[front[order]]
        # visibility per face in this view: render face ids as colours with the
        # same painter's order and read the buffer back (exact, matches the drawing)
        xmin, xmax = P[:, 0].min(), P[:, 0].max(); ymin, ymax = P[:, 1].min(), P[:, 1].max()
        res = 800
        fid = np.arange(len(T2)) + 1
        idcol = np.stack([(fid >> 16) & 255, (fid >> 8) & 255, fid & 255], 1) / 255.0
        figi = plt.figure(figsize=(res / 100, res / 100 * (ymax - ymin) / (xmax - xmin)), dpi=100)
        axi = figi.add_axes([0, 0, 1, 1])
        axi.add_collection(PolyCollection(tri[order][:, :, :2], facecolors=idcol[order], edgecolors='none', antialiased=False))
        axi.set_xlim(xmin, xmax); axi.set_ylim(ymin, ymax); axi.set_aspect('auto'); axi.axis('off')
        figi.canvas.draw()
        buf = np.asarray(figi.canvas.buffer_rgba())[:, :, :3].astype(np.int64)
        plt.close(figi)
        idbuf = ((buf[:, :, 0] << 16) | (buf[:, :, 1] << 8) | buf[:, :, 2]) - 1
        idbuf = idbuf[::-1]                      # image row 0 is the top; flip to y-up
        idbuf[(idbuf < 0) | (idbuf >= len(T2))] = -1
        sx = (idbuf.shape[1] - 1) / (xmax - xmin); sy = (idbuf.shape[0] - 1) / (ymax - ymin)
        seg_px = seg[idbuf[idbuf >= 0]]
        fig = plt.figure(figsize=(W / 100, W / 100 * (ymax - ymin) / (xmax - xmin) + 0.3), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.add_collection(PolyCollection(tri[order][:, :, :2], facecolors=cols[order], edgecolors='none', antialiased=False))
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_aspect('equal'); ax.axis('off')
        ax.set_title(f'{args.tag} parts, {title}', fontsize=9)
        from scipy import ndimage
        segbuf = np.where(idbuf >= 0, seg[np.maximum(idbuf, 0)], -1)
        for s in range(big):
            mask = segbuf == s
            if mask.sum() < 6:
                continue
            # label on the part's largest visible blob in this view (a ring seen
            # edge-on has its centroid elsewhere), at the blob pixel nearest its centroid
            lab_img, nlab = ndimage.label(ndimage.binary_dilation(mask, iterations=1) & mask)
            if nlab == 0:
                continue
            blob = lab_img == (np.bincount(lab_img.ravel())[1:].argmax() + 1)
            ys, xs = np.nonzero(blob)
            cx, cy = xs.mean(), ys.mean()
            k = np.argmin((xs - cx) ** 2 + (ys - cy) ** 2)
            lx, ly = xs[k] / sx + xmin, ys[k] / sy + ymin
            ax.text(lx, ly, str(s), fontsize=7, ha='center', va='center', color='black',
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85))
        path = f'reports/{args.tag}_parts_view{vi + 1}.png'
        fig.savefig(path, dpi=100); plt.close(fig)
        print(f'wrote {path}: {len(np.unique(seg_px[seg_px < big]))} of {big} parts visible')
