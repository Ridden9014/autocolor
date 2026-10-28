"""Phase 2 / C: run the seeded method on the decimated copy under every click
condition, with and without the height feature, several random draws.

usage: python tools/phase2_seeded.py [--conditions a,b,...] [--draws 5]
       [--height both|with|without] [--lam 3] [--save-best]
"""
import argparse
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autocolor.features import NAMES  # noqa: E402
from autocolor.clicks import sample_clicks_condition, part_types  # noqa: E402
from autocolor.seeded import FaceGraph, clicks_to_mesh, seeded_labels, seeded_segments, oversegment, prep_features  # noqa: E402
from autocolor.score import score, format_score  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--conditions', default='per_filament,per_type,per_region_1,per_region_2,per_region_3')
ap.add_argument('--draws', type=int, default=5)
ap.add_argument('--height', default='both')
ap.add_argument('--lam', type=float, default=3.0)
ap.add_argument('--beta', type=float, default=2.0)
ap.add_argument('--patch', type=float, default=6.0)
ap.add_argument('--rounds', type=int, default=0)
ap.add_argument('--tau', type=float, default=15.0)
ap.add_argument('--kappa', type=float, default=10.0)
ap.add_argument('--sigma-feat', type=float, default=1.0)
ap.add_argument('--save-best', action='store_true')
ap.add_argument('--method', default='patch', choices=['seg', 'patch'])
ap.add_argument('--prior', default='proto')
ap.add_argument('--h', type=float, default=1.0)
ap.add_argument('--balance', action='store_true')
ap.add_argument('--lam-through', type=float, default=3.0)
ap.add_argument('--clusters', type=int, default=16)
ap.add_argument('--seg-lam', type=float, default=2.0)
ap.add_argument('--verbose', action='store_true')
args = ap.parse_args()

dd = np.load('out/top_dec100k.npz'); V2, T2, src = dd['V'], dd['T'], dd['src']
tr = np.load('out/top_dec100k_truth.npz'); M2, A2 = tr['M'], tr['areas']
X_all = np.load('out/top_dec100k_features.npz')['X']
d = np.load('out/top_truth.npz'); A, V, T = d['areas'], d['V'], d['T']; C_full = V[T].mean(1)
rg = np.load('out/top_regions.npz'); comp, hops, table = rg['comp'], rg['hops'], rg['table']
tbl = [dict(id=int(a), label=int(b), area=float(c), n_faces=int(d)) for a, b, c, d in table]
types = part_types(tbl)
tcomp = comp[src]
k = M2.shape[1]
names = [f'F{i + 1}' for i in range(k)]
graph = FaceGraph(V2, T2)
graph.through_links(V2, T2)
hcol = NAMES.index('height')
variants = {'with': list(range(len(NAMES))), 'without': [j for j in range(len(NAMES)) if j != hcol]}
if args.height != 'both':
    variants = {args.height: variants[args.height]}

def score_fn(lab):
    res = score(M2, np.eye(k)[lab - 1] * A2[:, None], names, mode='identity')
    return f'overall {100 * res["overall"]:.2f} IoU ' + ' '.join(f'{100 * p["iou"]:.1f}' for p in res['per_section'])


results = {}
best = {}
segs = {}
if args.method == 'seg':
    for hv, cols in variants.items():
        nm = [NAMES[j] for j in cols]
        t0 = time.time()
        _, Z = prep_features(X_all[:, cols], nm)
        segs[hv] = oversegment(graph, Z, nm, n_clusters=args.clusters, lam=args.seg_lam)
        print(f'oversegmentation ({hv} height): {segs[hv].max() + 1} segments in {time.time() - t0:.0f}s')
for cond in args.conditions.split(','):
    for hv, cols in variants.items():
        rows = []
        for draw in range(args.draws):
            rng = np.random.default_rng(100 + draw)
            clicks = sample_clicks_condition(comp, tbl, hops, C_full, A, cond, rng)
            clicks2 = clicks_to_mesh(clicks, C_full, graph)
            t0 = time.time()
            if args.method == 'seg':
                lab, P = seeded_segments(graph, X_all[:, cols], [NAMES[j] for j in cols], clicks2, k, segs[hv],
                                         lam=args.lam, rounds=args.rounds, sigma_feat=args.sigma_feat,
                                         verbose=args.verbose, score_fn=score_fn if args.verbose else None)
            else:
                lab, P = seeded_labels(graph, X_all[:, cols], [NAMES[j] for j in cols], clicks2, k,
                                       patch_mm=args.patch, lam=args.lam, beta=args.beta, rounds=args.rounds,
                                       sigma_feat=args.sigma_feat, tau=args.tau, kappa=args.kappa,
                                       prior=args.prior, h=args.h, balance=args.balance, lam_through=args.lam_through,
                                       verbose=args.verbose, score_fn=score_fn if args.verbose else None)
            cand = np.eye(k)[lab - 1] * A2[:, None]
            res = score(M2, cand, names, mode='identity')
            ious = [p['iou'] for p in res['per_section']]
            rows.append([res['overall']] + ious)
            # per-region recall (area fraction of each full-mesh region labelled right)
            reg_rec = {}
            for r in tbl:
                m = tcomp == r['id']
                reg_rec[r['id']] = (A2[m] * (lab[m] == r['label'])).sum() / max(A2[m].sum(), 1e-9)
            if args.verbose or draw == 0:
                print(f'{cond:14s} height={hv:7s} draw {draw}: overall {100 * res["overall"]:.2f} IoU '
                      + ' '.join(f'{100 * v:.1f}' for v in ious) + f'  ({time.time() - t0:.0f}s, {len(clicks)} clicks)')
            key = (cond, hv)
            if key not in best or res['overall'] > best[key][0]:
                best[key] = (res['overall'], lab, clicks2, reg_rec)
        rows = np.array(rows) * 100
        results[(cond, hv)] = rows

print('\n=== decimated-space scores, mean / worst over draws (ceilings: overall 98.3, F1 97.9, F2 97.1, F3 86.3, F4 97.0)')
print(f"{'condition':14s}{'height':8s}{'overall':>14s}" + ''.join(f'{n:>14s}' for n in names))
for (cond, hv), rows in results.items():
    mean, worst = rows.mean(0), rows.min(0)
    print(f'{cond:14s}{hv:8s}' + ''.join(f'{m:7.2f}/{w:6.2f}' for m, w in zip(mean, worst)))

print('\n=== per_type best draw: recall of each region by part type (does one click label all of its kind?)')
for hv in variants:
    if ('per_type', hv) not in best:
        continue
    _, lab, clicks2, reg_rec = best[('per_type', hv)]
    clicked = {int(tcomp[f]) for f, _ in clicks2}
    by_type = {}
    for r in tbl:
        by_type.setdefault(types[r['id']], []).append((reg_rec[r['id']], r['id'] in clicked))
    print(f'height={hv}: ' + '; '.join(f"{t}: " + ','.join(f"{100 * v:.0f}{'*' if c else ''}" for v, c in vs)
                                      for t, vs in sorted(by_type.items())) + '   (* = clicked region)')

if args.save_best:
    for (cond, hv), (ov, lab, clicks2, _) in best.items():
        np.savez_compressed(f'out/seeded_{cond}_{hv}.npz', labels=lab, clicks=np.array(clicks2), overall=ov)
    print('saved best labels per condition to out/seeded_*.npz')
