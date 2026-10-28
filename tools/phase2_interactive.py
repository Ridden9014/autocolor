"""Phase 2 / C-interactive: click-and-correct curve. Start from per_type (6
clicks); each round add one click at the centre of the largest wrongly
labelled region; up to --max-clicks. Scores with and without the bottom-
surface rule, timings per update.

usage: python tools/phase2_interactive.py [--draws 5] [--max-clicks 40]
"""
import argparse
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autocolor.features import NAMES  # noqa: E402
from autocolor.clicks import sample_clicks_condition  # noqa: E402
from autocolor.seeded import FaceGraph, clicks_to_mesh  # noqa: E402
from autocolor.interactive import InteractiveSession, largest_wrong_region  # noqa: E402
from autocolor.rules import bottom_surface, apply_bottom_rule, above_through_solid  # noqa: E402
from autocolor.score import score  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--draws', type=int, default=5)
ap.add_argument('--max-clicks', type=int, default=40)
ap.add_argument('--start', default='per_type')
ap.add_argument('--no-height', action='store_true')
ap.add_argument('--save', action='store_true')
ap.add_argument('--reach', type=float, default=80.0, help='geodesic reach (mm) of corrective clicks; 0 = global')
ap.add_argument('--gate', type=float, default=1.5, help='feature gate (std) for a corrective click to claim look-alike faces within reach; 0 = off')
args = ap.parse_args()

dd = np.load('out/top_dec100k.npz'); V2, T2 = dd['V'], dd['T']
tr = np.load('out/top_dec100k_truth.npz'); M2, A2 = tr['M'], tr['areas']
truth2 = M2.argmax(1) + 1
X = np.load('out/top_dec100k_features.npz')['X']
d = np.load('out/top_truth.npz'); A, V, T = d['areas'], d['V'], d['T']; C_full = V[T].mean(1)
rg = np.load('out/top_regions.npz'); comp, hops, table = rg['comp'], rg['hops'], rg['table']
tbl = [dict(id=int(a), label=int(b), area=float(c), n_faces=int(d)) for a, b, c, d in table]
k = M2.shape[1]; names = [f'F{i + 1}' for i in range(k)]
cols = [j for j in range(len(NAMES)) if not (args.no_height and NAMES[j] == 'height')]
nm = [NAMES[j] for j in cols]

def sc(lab):
    res = score(M2, np.eye(k)[lab - 1] * A2[:, None], names, mode='identity')
    return [100 * res['overall']] + [100 * p['iou'] for p in res['per_section']]


t0 = time.time()
graph = FaceGraph(V2, T2); graph.through_links(V2, T2)
bottom = bottom_surface(graph, V2, T2)
above = above_through_solid(graph, V2, T2, bottom)
print(f'setup {time.time() - t0:.0f}s; bottom surface: {bottom.sum()} faces, {100 * A2[bottom].sum() / A2.sum():.1f}% of area, '
      f'truth there: ' + ' '.join(f'F{f}:{100 * A2[bottom & (truth2 == f)].sum() / A2[bottom].sum():.0f}%' for f in range(1, k + 1))
      + f'; with a surface above within 12 mm: {100 * A2[bottom & (above >= 0)].sum() / A2[bottom].sum():.0f}% of bottom area')
rule_oracle = apply_bottom_rule(graph, truth2, bottom, above)
print(f'rule applied to the TRUTH itself (its cost at best): overall {sc(rule_oracle)[0]:.2f}, IoU ' + ' '.join(f'{v:.1f}' for v in sc(rule_oracle)[1:]))



curves = {'raw': [], 'rule': []}
timings = []
for draw in range(args.draws):
    rng = np.random.default_rng(100 + draw)
    clicks2 = clicks_to_mesh(sample_clicks_condition(comp, tbl, hops, C_full, A, args.start, rng), C_full, graph)
    sess = InteractiveSession(graph, X[:, cols], nm, k)
    for f, l in clicks2:
        sess.add_click(f, l)
    rows_raw, rows_rule = [], []
    for n_clicks in range(len(clicks2), args.max_clicks + 1):
        lab = sess.solve()
        lab_rule = apply_bottom_rule(graph, lab, bottom, above)
        rows_raw.append(sc(lab)); rows_rule.append(sc(lab_rule))
        if n_clicks == args.max_clicks:
            break
        nxt = largest_wrong_region(graph, lab_rule, truth2, A2, clicked=[f for f, _ in sess.clicks])
        if nxt is None:
            break
        sess.add_click(nxt[0], nxt[1], reach=args.reach if args.reach > 0 else None, gate=args.gate if args.gate > 0 else None)
    stuck = largest_wrong_region(graph, lab_rule, truth2, A2, clicked=[f for f, _ in sess.clicks])
    stuck_area = stuck[3] if stuck else 0.0
    tc = [t for kind, t in sess.timings if kind == 'click']; ts = [t for kind, t in sess.timings if kind == 'solve']
    timings.append((np.mean(tc), np.mean(ts)))
    print(f'draw {draw}: {len(rows_raw)} rounds; raw overall {rows_raw[0][0]:.1f} -> {rows_raw[-1][0]:.1f}; '
          f'with rule {rows_rule[0][0]:.1f} -> {rows_rule[-1][0]:.1f}; per update: click {np.mean(tc):.2f}s + solve {np.mean(ts):.2f}s; '
          f'wrong area still in clicked regions {100 * stuck_area / A2.sum():.1f}%')
    curves['raw'].append(np.array(rows_raw)); curves['rule'].append(np.array(rows_rule))
    if args.save:
        np.savez_compressed(f'out/interactive_draw{draw}.npz', labels=lab, labels_rule=lab_rule, clicks=np.array(sess.seeds), click_labels=np.array(sess.seed_lab))

start_n = len(clicks2)
for tag in ('raw', 'rule'):
    L = min(len(c) for c in curves[tag])
    arr = np.stack([c[:L] for c in curves[tag]])         # draws x rounds x (overall, F1..Fk)
    mean, worst = arr.mean(0), arr.min(0)
    print(f'\n=== curve ({tag}): clicks -> overall mean/worst | per-section IoU mean (F1..F{k}) | min section IoU mean/worst')
    for r in range(L):
        if r % 2 == 0 or r == L - 1:
            print(f'{start_n + r:3d} clicks: {mean[r, 0]:6.2f}/{worst[r, 0]:6.2f} | ' + ' '.join(f'{v:5.1f}' for v in mean[r, 1:])
                  + f' | {mean[r, 1:].min():5.1f}/{worst[r, 1:].min():5.1f}')
    for thr in (90, 95):
        per_draw = []
        for c in arr:
            hit = np.where(c[:, 0] >= thr)[0]
            per_draw.append(start_n + hit[0] if len(hit) else None)
        reached = [p for p in per_draw if p is not None]
        print(f'clicks to overall >= {thr}: per draw {per_draw}; mean {np.mean(reached) if reached else "n/a"}, worst {max(reached) if reached else "n/a"}'
              f' ({len(reached)}/{len(per_draw)} draws reach it)')
    per_draw = []
    for c in arr:
        hit = np.where(c[:, 1:].min(1) >= 90)[0]
        per_draw.append(start_n + hit[0] if len(hit) else None)
    print(f'clicks to every section IoU >= 90: {per_draw}')
tc, ts = np.mean([t[0] for t in timings]), np.mean([t[1] for t in timings])
print(f'\ntime per click update on 100k faces: {tc + ts:.2f}s (patch+density {tc:.2f}s, graph cut {ts:.2f}s)')
