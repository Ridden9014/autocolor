"""Rooftop step 4: part -> filament mapping + MANUAL ring band rules -> full
mesh -> crease cut -> bare whole-triangle 3MF in out/. The band rules are a
hand selection in this model's coordinates (height above the base, radius
about the bounding-box axis), not a method result.

  python tools/rooftop_export.py [--rings all|D|none] [--below-ring F3|none] [--out out/rooftop_autocolor.3mf]
"""
import argparse
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autocolor.meshio import load_stl  # noqa: E402
from autocolor.transfer import labels_to_full  # noqa: E402
from autocolor.seeded import FaceGraph  # noqa: E402
from autocolor.fullmesh import full_cut, export_bare  # noqa: E402
from autocolor.paint import encode_state, decode_tree  # noqa: E402
from autocolor.truth import load_paint_3mf, tri_areas  # noqa: E402
from autocolor.render import decimate, preview  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--stl', default='refs/SD_RPG_WizardsArcaneTreasuryDiceTower-Rooftop.stl')
ap.add_argument('--n', type=int, default=200_000)
ap.add_argument('--rings', default='all', choices=['all', 'D', 'none'])
ap.add_argument('--out', default='out/rooftop_autocolor.3mf')
ap.add_argument('--below-ring', default='F3', choices=['F3', 'none'],
                help='MANUAL rule: everything below the thick ring (z < 33.5) not already F4 -> F3')
args = ap.parse_args()
assert 'refs' not in os.path.normpath(args.out).split(os.sep)

# user's mapping (part numbers from reports/rooftop_parts_view*.png)
MAP = {1: [21, 13, 2, 12, 18, 1, 5, 6, 9, 11, 20, 3, 23, 14, 27, 24, 17],
       2: [10, 7],
       3: [4, 8, 25, 26, 15, 19, 16, 0, 22]}
# MANUAL band rules (mm): (z0, z1, r_min) -> F4. Thick ring includes its underside.
RINGS = {'thick': (33.5, 41.0, 49.0), 'A': (100.5, 103.5, 21.0), 'B': (112.5, 116.5, 18.0),
         'C': (126.5, 129.0, 15.0), 'D': (139.5, 142.0, 11.5)}
use = {'all': list(RINGS), 'D': ['thick', 'D'], 'none': []}[args.rings]
K = 4 if use else 3

t0 = time.time()
V, T, good = load_stl(args.stl)
assert good.all()
z = np.load(f'out/rooftop_dec{args.n // 1000}k.npz'); V2, T2 = z['V'], z['T']
seg = np.load(f'out/rooftop_dec{args.n // 1000}k_parts.npz')['seg']
lab2 = np.zeros(len(T2), int)
for f, parts in MAP.items():
    lab2[np.isin(seg, parts)] = f
assert (lab2 > 0).all(), f'unmapped parts: {np.unique(seg[lab2 == 0])}'
A2 = tri_areas(V2, T2)
print('decimated area % per filament:', np.round(100 * np.bincount(lab2, weights=A2, minlength=K + 1)[1:] / A2.sum(), 1))

lab_t = labels_to_full(V2, T2, lab2, V, T)
print(f'transfer {time.time() - t0:.0f}s')
t1 = time.time()
graph = FaceGraph(V, T)
lab_c = full_cut(graph, lab_t, K)
print(f'full-mesh cut {time.time() - t1:.0f}s, changed {100 * (lab_c != lab_t).mean():.2f}% of faces')

# band rules on the full mesh, after the cut so the cut cannot erode them
C = V[T].mean(1)
cx, cy = (V[:, 0].min() + V[:, 0].max()) / 2, (V[:, 1].min() + V[:, 1].max()) / 2
r = np.hypot(C[:, 0] - cx, C[:, 1] - cy)
A = tri_areas(V, T)
for name in use:
    z0, z1, rmin = RINGS[name]
    sel = (C[:, 2] >= z0) & (C[:, 2] < z1) & (r >= rmin)
    print(f'  ring {name}: {sel.sum()} faces, {A[sel].sum():.0f} mm2 -> F4')
    lab_c[sel] = 4
if args.below_ring == 'F3':
    sel = (C[:, 2] < RINGS['thick'][0]) & (lab_c != 4) & (lab_c != 3)
    print(f'  below-ring rule (z < {RINGS["thick"][0]}, not F4) -> F3: {sel.sum()} faces, {A[sel].sum():.0f} mm2 changed '
          f'(from F1 {A[sel & (lab_c == 1)].sum():.0f}, F2 {A[sel & (lab_c == 2)].sum():.0f} mm2)')
    lab_c[sel] = 3
print('full-mesh area % per filament:', np.round(100 * np.bincount(lab_c, weights=A, minlength=K + 1)[1:] / A.sum(), 2))
np.save('out/rooftop_full_labels.npy', lab_c)

paint = [encode_state(int(l)) for l in lab_c]
t1 = time.time()
export_bare(args.out, V, T, paint, name='WizardsArcaneTreasury-Rooftop')
print(f'exported {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB) in {time.time() - t1:.0f}s')

# re-read check: same triangles, same strings
Vr, Tr, pr = load_paint_3mf(args.out)
assert len(Tr) == len(T) and len(Vr) == len(V), (len(Tr), len(Vr))
assert np.allclose(Vr, V, atol=1e-5) and (Tr == T).all()
lab_r = np.array([decode_tree(p) if p else 0 for p in pr])
same = (np.array(pr) == np.array(paint)).all()
print(f're-read: {len(Tr)} tris, strings identical: {same}, labels identical: {(lab_r == lab_c).all()}; VERIFIED' if same else 're-read MISMATCH')

# preview (checkpoint, 2 views) on a 100k copy
pal = np.array([[0.55, 0.15, 0.15], [0.85, 0.75, 0.2], [0.35, 0.45, 0.6], [0.9, 0.9, 0.9]])
Vp, Tp, src = decimate(V, T, 100_000)
preview(Vp, Tp, pal[lab_c[src] - 1], 'reports/rooftop_export_preview.png', views=((0, -25), (225, -50)),
        titles=['F1 red shingles, F2 yellow finial, F3 blue skirt/rim, F4 white rings', '3/4 from above'])
print('wrote reports/rooftop_export_preview.png')
