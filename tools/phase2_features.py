"""Phase 2 / B: compute per-face features on the decimated copy and print a
separability table: area-weighted ROC AUC of each single feature for each pair
of sections and for the part-type pairs that matter (rails vs deck, posts vs
deck, loops vs rails, caps vs F1)."""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autocolor.features import compute_features, NAMES  # noqa: E402
from autocolor.clicks import part_types  # noqa: E402

dd = np.load('out/top_dec100k.npz'); V2, T2, src = dd['V'], dd['T'], dd['src']
tr = np.load('out/top_dec100k_truth.npz'); M2, A2 = tr['M'], tr['areas']
lab = M2.argmax(1) + 1

t0 = time.time()
if os.path.exists('out/top_dec100k_features.npz') and '--recompute' not in sys.argv:
    X = np.load('out/top_dec100k_features.npz')['X']
else:
    X = compute_features(V2, T2)
    np.savez_compressed('out/top_dec100k_features.npz', X=X, names=np.array(NAMES))
print(f'features {X.shape} in {time.time() - t0:.0f}s')

# part types on the decimated mesh via the full-mesh regions
rg = np.load('out/top_regions.npz')
comp_full, table = rg['comp'], rg['table']
tbl = [dict(id=int(a), label=int(b), area=float(c), n_faces=int(d)) for a, b, c, d in table]
types = part_types(tbl)
tcomp = comp_full[src]
ptype = np.array([types.get(int(c), 'none') for c in tcomp])


def auc(x, pos, neg, w):
    """Area-weighted ROC AUC of feature x for pos vs neg masks; reported as
    max(auc, 1-auc) so direction does not matter."""
    xs = np.r_[x[pos], x[neg]]
    ws = np.r_[w[pos], w[neg]]
    ys = np.r_[np.ones(pos.sum()), np.zeros(neg.sum())]
    o = np.argsort(xs, kind='stable')
    xs, ws, ys = xs[o], ws[o], ys[o]
    # weighted rank-sum with ties averaged
    cum_neg = np.cumsum(ws * (ys == 0))
    # for each positive, weight of negatives strictly below + half of ties
    uniq, inv = np.unique(xs, return_inverse=True)
    tot_neg_at = np.bincount(inv, weights=ws * (ys == 0))
    below = np.cumsum(tot_neg_at) - tot_neg_at
    score = (below[inv] + 0.5 * tot_neg_at[inv])
    a = (score * ws * (ys == 1)).sum() / (ws[ys == 1].sum() * ws[ys == 0].sum())
    return max(a, 1 - a)


pairs = [('F2 loops', lab == 2, 'F3 rails', lab == 3), ('F3 rails', lab == 3, 'F4 deck', lab == 4),
         ('F1 all', lab == 1, 'F4 deck', lab == 4), ('F1 all', lab == 1, 'F2 loops', lab == 2),
         ('F1 all', lab == 1, 'F3 rails', lab == 3), ('F2 loops', lab == 2, 'F4 deck', lab == 4),
         ('posts', ptype == 'F1:major', 'deck', ptype == 'F4:main'),
         ('posts', ptype == 'F1:major', 'rails', ptype == 'F3:major'),
         ('posts', ptype == 'F1:major', 'tree', ptype == 'F1:main'),
         ('caps', ptype == 'F4:major', 'F1 all', lab == 1),
         ('caps', ptype == 'F4:major', 'deck', ptype == 'F4:main')]
print('\npart-type face counts:', {str(t): int((ptype == t).sum()) for t in np.unique(ptype)})
hdr = f"{'feature':<11}" + ''.join(f"{a[:5]}|{b[:5]:<6}" for a, _, b, _ in pairs)
print('\nSingle-feature separability, area-weighted ROC AUC (0.5 = none, 1.0 = perfect)')
print(hdr)
rows = []
for j, name in enumerate(NAMES):
    vals = [auc(X[:, j], p, q, A2) for _, p, _, q in pairs]
    rows.append(vals)
    print(f"{name:<11}" + ''.join(f"{v:6.3f}      " for v in vals))
best = np.array(rows).max(0)
print(f"{'BEST single':<11}" + ''.join(f"{v:6.3f}      " for v in best))

# combined features: gradient boosting, trained on half of the REGIONS of each
# side and tested on the other half (so adjacent faces never leak), both ways.
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402


def combined_auc(cols, p, q):
    regs = tcomp
    y = np.where(p, 1, 0)
    sel = p | q
    rid = regs[sel]
    fold = np.zeros(sel.sum(), int)
    C2 = V2[T2].mean(1)[sel]
    for cls in (p, q):
        rc = np.unique(regs[cls])
        if len(rc) >= 2:
            grp = {r: i % 2 for i, r in enumerate(rc)}   # alternate regions into 2 folds
            m = cls[sel]
            fold[m] = [grp[r] for r in rid[m]]
        else:  # single region (deck, tree): split it spatially at the x median (adjacent-face leakage only at the cut)
            m = cls[sel]
            fold[m] = (C2[m, 0] > np.median(C2[m, 0])).astype(int)
    Xs, ys, ws = X[sel][:, cols], y[sel], A2[sel]
    aucs = []
    for f in (0, 1):
        tr, te = fold != f, fold == f
        if len(np.unique(ys[tr])) < 2 or len(np.unique(ys[te])) < 2:
            continue
        clf = HistGradientBoostingClassifier(max_iter=150, max_depth=4).fit(Xs[tr], ys[tr], sample_weight=ws[tr])
        pr = clf.predict_proba(Xs[te])[:, 1]
        aucs.append(auc(pr, ys[te] == 1, ys[te] == 0, ws[te]))
    return np.mean(aucs) if aucs else np.nan


all_cols = list(range(len(NAMES)))
no_height = [j for j in all_cols if NAMES[j] != 'height']
for tag, cols in (('ALL (GBM)', all_cols), ('ALL-height', no_height)):
    vals = [combined_auc(cols, p, q) for _, p, _, q in pairs]
    print(f"{tag:<11}" + ''.join(f"{v:6.3f}      " for v in vals))
print('(combined rows: trained on half the regions of each side, tested on the other half, averaged both ways)')
print('\nfeature medians by section:')
print(f"{'feature':<11}" + ''.join(f"{s:>9}" for s in ['F1', 'F2', 'F3', 'F4', 'posts', 'caps']))
for j, name in enumerate(NAMES):
    meds = [np.median(X[lab == f, j]) for f in (1, 2, 3, 4)] + [np.median(X[ptype == 'F1:major', j]), np.median(X[ptype == 'F4:major', j])]
    print(f"{name:<11}" + ''.join(f"{v:9.2f}" for v in meds))
