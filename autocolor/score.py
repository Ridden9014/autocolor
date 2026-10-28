"""Area-weighted agreement between a reference paint matrix and a candidate.

Both are (N, k) matrices of area per label for the same triangle list.
Overlap O[i, j] = area where reference says i and candidate says j. When
a triangle is fractional on both sides the sub-triangle geometry is unknown,
so we use the product coupling ref_i * cand_j / area_t; for whole-triangle
candidates (the Phase 1/2 case) this is exact.
"""
import numpy as np
from functools import lru_cache
from scipy.optimize import linear_sum_assignment
from .paint import decode_tree, leaf_states_at, sample_points

_PTS = sample_points(64)


@lru_cache(maxsize=None)
def _states_at_samples(s):
    """States of the 64 fixed sample points for one paint string (0 = unpainted)."""
    if s == '':
        return np.zeros(len(_PTS), dtype=np.int64)
    t = decode_tree(s)
    if isinstance(t, int):
        return np.full(len(_PTS), t)
    return leaf_states_at(t, _PTS)


def overlap_matrix_exact(ref_paint, cand_paint, areas, k, ref_default=1, cand_default=1):
    """Overlap O (k,k) between two per-triangle paint-string lists over the
    same triangles, by point sampling inside split triangles. Unsplit pairs
    are handled in bulk. State 0 maps to the given default filament."""
    ref_paint, cand_paint = list(ref_paint), list(cand_paint)
    O = np.zeros((k, k))
    split = np.array([len(a) > 2 or len(b) > 2 for a, b in zip(ref_paint, cand_paint)])
    # bulk: whole-triangle on both sides
    rs = np.array([_states_at_samples(s)[0] if not sp else 0 for s, sp in zip(ref_paint, split)])
    cs = np.array([_states_at_samples(s)[0] if not sp else 0 for s, sp in zip(cand_paint, split)])
    rs = np.where(rs == 0, ref_default, rs) - 1
    cs = np.where(cs == 0, cand_default, cs) - 1
    w = np.where(split, 0.0, areas)
    np.add.at(O, (rs, cs), w)
    # sampled: anything split
    n = len(_PTS)
    for i in np.where(split)[0]:
        r = _states_at_samples(ref_paint[i]); c = _states_at_samples(cand_paint[i])
        r = np.where(r == 0, ref_default, r) - 1
        c = np.where(c == 0, cand_default, c) - 1
        np.add.at(O, (r, c), areas[i] / n)
    return O


def overlap_matrix(ref, cand):
    areas = ref.sum(1)
    w = np.where(areas > 0, 1.0 / np.maximum(areas, 1e-300), 0.0)
    return (ref * w[:, None]).T @ cand


def map_labels(O, mode='best'):
    """Map candidate columns to reference rows. Returns dict cand_col -> ref_row.
    mode='best': each candidate label goes to the ref section it overlaps most
      (many-to-one; several candidate labels may merge into one section, and a
      section may get nothing).
    mode='one2one': Hungarian assignment maximizing total overlap; leftover
      candidate columns fall back to best overlap."""
    if mode == 'identity':
        return {j: j for j in range(O.shape[1])}
    if mode == 'best':
        return {j: int(O[:, j].argmax()) for j in range(O.shape[1])}
    rows, cols = linear_sum_assignment(O, maximize=True)
    m = dict(zip(cols.tolist(), rows.tolist()))
    for j in range(O.shape[1]):
        if j not in m:
            m[j] = int(O[:, j].argmax())
    return m


def remap(cand, mapping, k):
    out = np.zeros((cand.shape[0], k))
    for j, i in mapping.items():
        out[:, i] += cand[:, j]
    return out


def score_from_overlap(O, names=None, mode='identity'):
    """Scores from an overlap matrix O (ref rows x cand cols, areas)."""
    k = O.shape[0]
    mapping = map_labels(O, mode)
    Om = np.zeros((k, k))
    for j, i in mapping.items():
        Om[:, i] += O[:, j]
    diag = np.diag(Om)
    ra, ca = Om.sum(1), Om.sum(0)
    total = O.sum()
    per = []
    for i in range(k):
        union = ra[i] + ca[i] - diag[i]
        per.append(dict(section=names[i] if names else i, ref_area_pct=100 * ra[i] / total,
                        iou=diag[i] / union if union > 0 else float('nan'),
                        recall=diag[i] / ra[i] if ra[i] > 0 else float('nan'),
                        precision=diag[i] / ca[i] if ca[i] > 0 else float('nan')))
    return dict(overall=diag.sum() / total, mapping=mapping, overlap=O, per_section=per)


def score(ref, cand, names=None, mode='best'):
    """Score from (N,k) area matrices (product coupling on fractional pairs)."""
    return score_from_overlap(overlap_matrix(ref, cand), names, mode)


def score_paint(ref_paint, cand_paint, areas, k, names=None, mode='identity', ref_default=1, cand_default=1):
    """Exact score from paint strings (point-sampled inside split triangles).
    Use areas=0 for faces excluded from scoring (e.g. invisible)."""
    O = overlap_matrix_exact(ref_paint, cand_paint, areas, k, ref_default, cand_default)
    return score_from_overlap(O, names, mode)


def format_score(res, label_names=None):
    lines = [f"overall agreement: {100 * res['overall']:.2f}%  (area-weighted)",
             f"candidate->ref mapping: {res['mapping']}",
             f"{'section':<28}{'area%':>7}{'IoU':>8}{'recall':>8}{'prec':>8}"]
    for p in res['per_section']:
        lines.append(f"{str(p['section']):<28}{p['ref_area_pct']:7.2f}{100 * p['iou']:8.2f}{100 * p['recall']:8.2f}{100 * p['precision']:8.2f}")
    return '\n'.join(lines)
