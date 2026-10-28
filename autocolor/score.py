"""Area-weighted agreement between a reference paint matrix and a candidate.

Both are (N, k) matrices of area per label for the same triangle list.
Overlap O[i, j] = area where reference says i and candidate says j. When
a triangle is fractional on both sides the sub-triangle geometry is unknown,
so we use the product coupling ref_i * cand_j / area_t; for whole-triangle
candidates (the Phase 1/2 case) this is exact.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


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


def score(ref, cand, names=None, mode='best'):
    """Return dict with overall agreement and per-section IoU/recall/precision.
    cand columns are first mapped onto ref columns (see map_labels)."""
    k = ref.shape[1]
    O = overlap_matrix(ref, cand)
    mapping = map_labels(O, mode)
    C = remap(cand, mapping, k)
    Om = overlap_matrix(ref, C)
    diag = np.diag(Om)
    ra, ca = ref.sum(0), C.sum(0)
    total = ref.sum()
    per = []
    for i in range(k):
        union = ra[i] + ca[i] - diag[i]
        per.append(dict(section=names[i] if names else i, ref_area_pct=100 * ra[i] / total,
                        iou=diag[i] / union if union > 0 else float('nan'),
                        recall=diag[i] / ra[i] if ra[i] > 0 else float('nan'),
                        precision=diag[i] / ca[i] if ca[i] > 0 else float('nan')))
    return dict(overall=diag.sum() / total, mapping=mapping, overlap=O, per_section=per)


def format_score(res, label_names=None):
    lines = [f"overall agreement: {100 * res['overall']:.2f}%  (area-weighted)",
             f"candidate->ref mapping: {res['mapping']}",
             f"{'section':<28}{'area%':>7}{'IoU':>8}{'recall':>8}{'prec':>8}"]
    for p in res['per_section']:
        lines.append(f"{str(p['section']):<28}{p['ref_area_pct']:7.2f}{100 * p['iou']:8.2f}{100 * p['recall']:8.2f}{100 * p['precision']:8.2f}")
    return '\n'.join(lines)
