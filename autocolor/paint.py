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
