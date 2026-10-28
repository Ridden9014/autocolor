"""Encode/decode the per-triangle `paint_color` strings used by Bambu Studio,
Orca and PrusaSlicer (slic3rpe:mmu_segmentation).

Format (from BambuStudio TriangleSelector::serialize / FacetsAnnotation):
  * The string is a sequence of hex nibbles, written LAST nibble FIRST
    (FacetsAnnotation::get_triangle_as_string inserts at the front), so we
    reverse it before reading.
  * Each node is one nibble: bits 0-1 = number of split sides (0 = leaf),
    bits 2-3 = leaf state, or the "special side" for a split node.
  * Leaf state 3 (bits 2-3 == 11) is an escape: following nibbles hold
    state-3; every 0xF nibble adds 15 and continues.
  * A split node with s sides has s+1 children, serialized in REVERSE
    child order (child s first, child 0 last).
  * State 0 = unpainted (object default), state k = filament k (1-based).

Child area fractions (TriangleSelector::perform_split), in child order:
  1 side : [1/2, 1/2]
  2 sides: [1/4, 1/4, 1/2]
  3 sides: [1/4, 1/4, 1/4, 1/4]
Midpoint subdivision makes these independent of triangle shape.

Tree representation used here:
  leaf  -> int state
  split -> (sides, special_side, [child_0, ..., child_sides])
"""
from functools import lru_cache
import numpy as np

CHILD_AREAS = {1: (0.5, 0.5), 2: (0.25, 0.25, 0.5), 3: (0.25,) * 4}


def _nibbles(s):
    return [int(c, 16) for c in reversed(s)]


def decode_tree(s):
    """Parse a paint_color string into a tree. Raises ValueError on leftover
    or truncated nibbles."""
    nib = _nibbles(s)
    pos = 0

    def read():
        nonlocal pos
        if pos >= len(nib):
            raise ValueError(f"truncated paint string {s!r}")
        v = nib[pos]
        pos += 1
        return v

    def node():
        code = read()
        sides = code & 0b11
        if sides == 0:
            if (code & 0b1100) == 0b1100:
                n, v = 0, read()
                while v == 0b1111:
                    n += 1
                    v = read()
                return v + 15 * n + 3
            return code >> 2
        special = code >> 2
        kids = [node() for _ in range(sides + 1)]
        kids.reverse()  # stream holds child sides..0
        return (sides, special, kids)

    t = node()
    if pos != len(nib):
        raise ValueError(f"{len(nib) - pos} leftover nibbles in {s!r}")
    return t


def encode_tree(t):
    """Inverse of decode_tree."""
    out = []

    def leaf(state):
        if state < 3:
            out.append(state << 2)
        else:
            out.append(0b1100)
            n = state - 3
            while n >= 15:
                out.append(0b1111)
                n -= 15
            out.append(n)

    def node(t):
        if isinstance(t, int):
            leaf(t)
            return
        sides, special, kids = t
        assert sides in (1, 2, 3) and len(kids) == sides + 1
        out.append(sides | (special << 2))
        for k in reversed(kids):
            node(k)

    node(t)
    return ''.join('0123456789ABCDEF'[v] for v in reversed(out))


def encode_state(state):
    """paint_color for a whole (unsplit) triangle. state 0 -> '' (unpainted)."""
    return '' if state == 0 else encode_tree(state)


def tree_fractions(t, n_states):
    """Area fraction per state (index 0 = unpainted) summing to 1."""
    f = np.zeros(n_states)
    if isinstance(t, int):
        f[t] = 1.0
        return f
    sides, _, kids = t
    for a, k in zip(CHILD_AREAS[sides], kids):
        f += a * tree_fractions(k, n_states)
    return f


# --- split geometry (port of TriangleSelector::perform_split) -------------
# Work in barycentric coords of the root triangle: corner i = unit vector e_i.
# A node is a (3,3) array of its corner barycentrics in PrusaSlicer vertex order.

def split_children(P, sides, special):
    """Child corner arrays for a node with corners P (3,3), in child order
    (matching CHILD_AREAS). Vertex rotation and midpoint rules follow
    perform_split exactly."""
    a, b, c = (P[special % 3], P[(special + 1) % 3], P[(special + 2) % 3])
    if sides == 1:
        m = (b + c) / 2
        return [np.array([a, b, m]), np.array([m, c, a])]
    if sides == 2:
        mab, mac = (a + b) / 2, (a + c) / 2
        return [np.array([a, mab, mac]), np.array([mab, b, mac]), np.array([b, c, mac])]
    if sides == 3:
        a, b, c = P
        mab, mbc, mac = (a + b) / 2, (b + c) / 2, (a + c) / 2
        return [np.array([a, mab, mac]), np.array([mab, b, mbc]),
                np.array([mbc, c, mac]), np.array([mab, mbc, mac])]
    raise ValueError(sides)


ROOT = np.eye(3)


def tree_leaves(t, P=ROOT):
    """List of (corners (3,3) barycentric, state) for every leaf."""
    if isinstance(t, int):
        return [(P, t)]
    sides, special, kids = t
    out = []
    for cp, k in zip(split_children(P, sides, special), kids):
        out.extend(tree_leaves(k, cp))
    return out


def _inside(P, x):
    """Point-in-triangle for barycentric points x (n,3) against corners P (3,3):
    express x in the child's own barycentric coordinates (2D in the root
    triangle's plane) and check all >= -eps."""
    A = np.vstack([P[:, :2].T, np.ones(3)])  # maps child bary -> root (u, v, 1)
    B = np.c_[x[:, :2], np.ones(len(x))]
    lam = np.linalg.solve(A, B.T).T         # (n,3) child barycentrics
    return (lam >= -1e-9).all(1)


def leaf_states_at(t, x, P=ROOT):
    """State of the leaf containing each barycentric point x (n,3)."""
    if isinstance(t, int):
        return np.full(len(x), t)
    out = np.full(len(x), -1)
    sides, special, kids = t
    todo = np.ones(len(x), bool)
    for cp, k in zip(split_children(P, sides, special), kids):
        if not todo.any():
            break
        ins = todo & _inside(cp, x)
        if ins.any():
            out[ins] = leaf_states_at(k, x[ins], cp)
            todo &= ~ins
    if todo.any():  # numerical edge case: assign nearest child by centroid
        cents = np.array([cp.mean(0) for cp in split_children(P, sides, special)])
        j = np.argmin(((x[todo][:, None, :] - cents[None]) ** 2).sum(2), 1)
        for ji in np.unique(j):
            sel = np.where(todo)[0][j == ji]
            out[sel] = leaf_states_at(kids[ji], x[sel], split_children(P, sides, special)[ji])
    return out


def sample_points(n=64, seed=0):
    """Fixed barycentric samples inside a triangle. For n = m*m the points are
    the centroids of the m*m sub-triangles of a uniform m-subdivision (exactly
    area-stratified, and exact for any midpoint split of depth <= log2 m);
    otherwise uniform random."""
    m = int(round(np.sqrt(n)))
    if m * m == n:
        pts = []
        for i in range(m):
            for j in range(m - i):
                # upright cell with corner at (i, j) in lattice coords
                pts.append(((i + 1 / 3), (j + 1 / 3)))
                if j < m - i - 1:
                    pts.append(((i + 2 / 3), (j + 2 / 3)))
        p = np.array(pts) / m
        return np.c_[1 - p.sum(1), p]
    rng = np.random.default_rng(seed)
    r = rng.random((n, 2))
    flip = r.sum(1) > 1
    r[flip] = 1 - r[flip]
    return np.c_[1 - r.sum(1), r]


def build_tree(label_fn, max_depth=3, P=ROOT, depth=0):
    """Split a triangle until every leaf is uniform in label_fn (barycentric
    points (n,3) -> int labels) or max_depth is reached. Splits follow the
    boundary: 2 differing edges -> 2-side split at their shared corner,
    otherwise 3-side split. Leaves at max_depth take the majority label
    over sampled points. Returns a tree (int or tuple)."""
    corners = label_fn(P)
    mids = label_fn(np.array([(P[1] + P[2]) / 2, (P[0] + P[2]) / 2, (P[0] + P[1]) / 2]))
    cen = label_fn(P.mean(0, keepdims=True))
    allv = np.concatenate([corners, mids, cen])
    if (allv == allv[0]).all():
        return int(allv[0])
    if depth >= max_depth:
        lab = label_fn(sample_points(32) @ P)
        return int(np.bincount(lab).argmax())
    diff = [corners[1] != corners[2], corners[0] != corners[2], corners[0] != corners[1]]  # edge opposite corner i
    if sum(diff) == 2:
        special = diff.index(False)  # corner shared by the two differing edges is opposite the equal edge
        sides = 2
    else:
        sides, special = 3, 0
    kids = [build_tree(label_fn, max_depth, cp, depth + 1) for cp in split_children(P, sides, special)]
    if all(isinstance(k, int) for k in kids) and len(set(kids)) == 1:
        return kids[0]
    return (sides, special, kids)


def tree_depth(t):
    return 0 if isinstance(t, int) else 1 + max(tree_depth(k) for k in t[2])


def tree_max_state(t):
    return t if isinstance(t, int) else max(tree_max_state(k) for k in t[2])


@lru_cache(maxsize=None)
def fractions(s, n_states):
    """Cached decode of one string to an area-fraction tuple."""
    if s == '':
        f = np.zeros(n_states)
        f[0] = 1.0
        return tuple(f)
    return tuple(tree_fractions(decode_tree(s), n_states))


def fractions_matrix(strings, n_states):
    """(N, n_states) area fractions for a sequence of paint strings."""
    uniq = {}
    idx = np.empty(len(strings), dtype=np.int64)
    for i, s in enumerate(strings):
        j = uniq.get(s)
        if j is None:
            j = uniq[s] = len(uniq)
        idx[i] = j
    table = np.array([fractions(s, n_states) for s in uniq])
    return table[idx]
