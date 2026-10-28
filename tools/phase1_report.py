"""Phase 1: build ground truth from the hand-painted project, print the
section report, render one 2-view preview, and score a candidate 3MF.

usage: python tools/phase1_report.py [--candidate out/x.3mf] [--no-render]
"""
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autocolor.truth import Project, load_paint_3mf, tri_areas, truth_matrix, paint_stats  # noqa: E402
from autocolor.paint import fractions_matrix  # noqa: E402
from autocolor.score import score, format_score  # noqa: E402

PROJECT = 'refs/TreetopDruidHaven_painted.3mf'
TOP_OID, BASE_OID = 2, 4


def build_truth(P, oid, tag):
    V, T, paint = P.load_mesh(oid)
    st = paint_stats(paint)
    A = tri_areas(V, T)
    M = truth_matrix(paint, A, P.n_filaments, P.default_extruder(oid))
    assert st['unparsable'] == 0, st
    assert abs(M.sum() - A.sum()) < 1e-6 * A.sum()
    os.makedirs('out', exist_ok=True)
    np.savez_compressed(f'out/{tag}_truth.npz', M=M, areas=A, V=V, T=T)
    return V, T, paint, A, M, st


def section_table(P, A, M, paint):
    tot = A.sum()
    split = np.array([len(s) > 2 for s in paint])
    lines = [f"{'filament':<10}{'colour':<10}{'area%':>8}{'tris':>10}{'split-area%':>13}"]
    for f in range(P.n_filaments):
        dom = M.argmax(1) == f
        lines.append(f"F{f + 1:<9}{P.filament_colours[f]:<10}{100 * M[:, f].sum() / tot:8.2f}"
                     f"{dom.sum():10d}{100 * M[split, f].sum() / M[:, f].sum():13.2f}")
    return '\n'.join(lines)


def score_candidate(path, A, M, names):
    V, T, paint = load_paint_3mf(path)
    assert len(T) == len(A), f'{path}: {len(T)} tris vs truth {len(A)}'
    n_states = max(P_N, max(len(s) for s in paint)) + 1  # generous; stats below give real max
    F = fractions_matrix(paint, n_states)
    C = F[:, 1:] * A[:, None]
    C = C[:, C.sum(0) > 0] if C.shape[1] > P_N else C
    out = []
    for mode in ('best', 'one2one'):
        res = score(M, C, names=names, mode=mode)
        out.append(f'--- mapping mode: {mode}\n' + format_score(res))
    return '\n'.join(out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidate', default='refs/TreetopDruidHaven-Top_autocolor_poc.3mf')
    ap.add_argument('--no-render', action='store_true')
    args = ap.parse_args()

    P = Project(PROJECT)
    P_N = P.n_filaments
    names = [f'F{i + 1} {c}' for i, c in enumerate(P.filament_colours)]
    print(f'project: {P_N} filaments {P.filament_colours}')

    V, T, paint, A, M, st = build_truth(P, TOP_OID, 'top')
    print('top paint stats:', st)
    print(section_table(P, A, M, paint))

    _, _, bpaint, bA, bM, bst = build_truth(P, BASE_OID, 'base')
    print('base (held out) parses:', bst, 'area%', np.round(100 * bM.sum(0) / bA.sum(), 2))

    if not args.no_render:
        from autocolor.render import decimate, preview, filament_rgb
        V2, T2, src = decimate(V, T, 100_000)
        np.savez_compressed('out/top_dec100k.npz', V=V2, T=T2, src=src)
        preview(V2, T2, filament_rgb(M, P.filament_colours)[src], 'out/top_truth_preview.png',
                titles=['front-left, from above', 'back-right, from above'])
        print('wrote out/top_truth_preview.png')

    if args.candidate:
        print(f'\ncandidate: {args.candidate}')
        print(score_candidate(args.candidate, A, M, names))
