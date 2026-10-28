"""Phase 2 / A3: oracle ceilings. How well can a labeling reach the hand paint
when restricted to (a) whole triangles on the full mesh, (b) whole triangles on
the 100k decimated mesh transferred back, (c) sub-triangle splits of limited
depth on the full mesh. Scored exactly, identity mapping, invisible faces
excluded."""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autocolor.truth import Project  # noqa: E402
from autocolor.paint import (decode_tree, encode_tree, encode_state, leaf_states_at,  # noqa: E402
                             build_tree, fractions_matrix)
from autocolor.score import score_paint, format_score  # noqa: E402
from autocolor.transfer import truth_on_decimated, labels_to_full  # noqa: E402

P = Project('refs/TreetopDruidHaven_painted.3mf')
k = P.n_filaments
names = [f'F{i + 1} {c}' for i, c in enumerate(P.filament_colours)]
V, T, paint = P.load_mesh(2)
d = np.load('out/top_truth.npz'); M, A = d['M'], d['areas']
vis = np.load('out/top_visible.npy')
Aeff = A * vis
print(f'excluded invisible: {(~vis).sum()} faces, {100 * A[~vis].sum() / A.sum():.3f}% area')


def report(tag, cand_paint):
    t0 = time.time()
    res = score_paint(paint, cand_paint, Aeff, k, names, mode='identity')
    print(f'\n=== {tag}  ({time.time() - t0:.0f}s)\n' + format_score(res))
    return res


# sanity: truth vs itself
report('truth vs truth (sanity, must be 100)', paint)

# (a) whole-triangle oracle on the full mesh: majority filament per triangle
maj = M.argmax(1) + 1
report('oracle A: whole triangles, full mesh', [encode_state(int(s)) for s in maj])

# (c) sub-triangle oracle at limited depth: refine the truth tree itself
F = fractions_matrix(paint, k + 1)
split_idx = np.where((F[:, 0] < 1) & np.array([len(s) > 2 for s in paint]))[0]
for depth in (2, 3):
    cand = [encode_state(int(s)) for s in maj]
    t0 = time.time()
    for i in split_idx:
        t = decode_tree(paint[i])
        fn = lambda x, t=t: np.where(leaf_states_at(t, x) == 0, P.default_extruder(2), leaf_states_at(t, x))
        cand[i] = encode_tree(build_tree(fn, max_depth=depth))
    print(f'built depth-{depth} refinement of {len(split_idx)} split tris in {time.time() - t0:.0f}s')
    report(f'oracle C: sub-triangle splits depth<={depth}, full mesh', cand)

# (b) decimated oracle: majority on decimated, transferred to full as whole triangles
dd = np.load('out/top_dec100k.npz'); V2, T2 = dd['V'], dd['T']
M2, A2 = truth_on_decimated(V, T, M, V2, T2)
np.savez_compressed('out/top_dec100k_truth.npz', M=M2, areas=A2)
maj2 = M2.argmax(1) + 1
full = labels_to_full(V2, T2, maj2, V, T)
report('oracle B: whole triangles on 100k decimated -> full (nearest face)', [encode_state(int(s)) for s in full])
# decimated-space self score (what we will see while iterating)
from autocolor.score import score  # noqa: E402
res = score(M2, np.eye(k)[maj2 - 1] * A2[:, None], names, mode='identity')
print('\n=== decimated-space score of decimated majority (iteration metric ceiling)\n' + format_score(res))
