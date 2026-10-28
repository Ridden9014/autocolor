"""Phase 3 / 1a-1c: decimated labels -> full mesh -> cut -> sub-triangle
refinement -> export into a copy of the painted project (out/). Scores exact
at every stage.

usage: python tools/phase3_fullmesh.py [--model top] [--labels out/interactive_draw0.npz] [--depth 2]
"""
import argparse
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autocolor.truth import Project  # noqa: E402
from autocolor.seeded import FaceGraph  # noqa: E402
from autocolor.transfer import labels_to_full  # noqa: E402
from autocolor.fullmesh import full_cut, refine_boundaries, export_project  # noqa: E402
from autocolor.paint import encode_state  # noqa: E402
from autocolor.score import score_paint, format_score  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='top')
ap.add_argument('--labels', default='out/interactive_draw0.npz')
ap.add_argument('--depth', type=int, default=2)
ap.add_argument('--no-refine', action='store_true')
args = ap.parse_args()

OID = {'top': 2, 'base': 4}[args.model]
P = Project('refs/TreetopDruidHaven_painted.3mf')
k = P.n_filaments
names = [f'F{i + 1}' for i in range(k)]
V, T, truth_paint = P.load_mesh(OID)
d = np.load(f'out/{args.model}_truth.npz'); A = d['areas']
vis = np.load(f'out/{args.model}_visible.npy') if os.path.exists(f'out/{args.model}_visible.npy') else np.ones(len(T), bool)
Aeff = A * vis
dd = np.load(f'out/{args.model}_dec100k.npz'); V2, T2 = dd['V'], dd['T']
lab2 = np.load(args.labels)['labels_rule']


def report(tag, paint):
    res = score_paint(truth_paint, paint, Aeff, k, names, mode='identity', ref_default=P.default_extruder(OID))
    print(f'\n=== {tag}\n' + format_score(res))
    return res


t0 = time.time()
graph = FaceGraph(V, T)
lab_t = labels_to_full(V2, T2, lab2, V, T)
print(f'transfer {time.time() - t0:.0f}s')
report('1a transferred (closest decimated face)', [encode_state(int(l)) for l in lab_t])
t0 = time.time()
lab_c = full_cut(graph, lab_t, k)
print(f'full cut {time.time() - t0:.0f}s, changed {100 * (lab_c != lab_t).mean():.2f}% of faces')
report('1a after full-mesh cut', [encode_state(int(l)) for l in lab_c])
np.save(f'out/{args.model}_full_labels.npy', lab_c)
paint = [encode_state(int(l)) for l in lab_c]
if not args.no_refine:
    paint = refine_boundaries(graph, V, T, lab_c, max_depth=args.depth)
    report(f'1b after sub-triangle refinement depth {args.depth}', paint)
out = f'out/{args.model}_autocolor.3mf'
part_file = P.objects[OID]['part_file']
export_project('refs/TreetopDruidHaven_painted.3mf', out, part_file, paint)
print(f'\nexported {out} ({os.path.getsize(out) / 1e6:.1f} MB)')
# verify by re-reading
P2 = Project(out)
_, T3, paint3 = P2.load_mesh(OID)
assert len(T3) == len(T) and paint3 == paint, 'export round-trip mismatch'
report('1c re-read export', paint3)
print('export round-trip: identical strings, VERIFIED')
