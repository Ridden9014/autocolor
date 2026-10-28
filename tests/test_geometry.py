import random
import numpy as np
from autocolor.paint import (split_children, tree_leaves, leaf_states_at, sample_points,
                             build_tree, encode_tree, decode_tree, tree_fractions, CHILD_AREAS, ROOT)
from autocolor.score import overlap_matrix_exact


def bary_area(P):
    # area of a barycentric triangle relative to the root (root area = 1)
    u, v = P[:, 1], P[:, 2]
    return abs((u[1] - u[0]) * (v[2] - v[0]) - (u[2] - u[0]) * (v[1] - v[0]))


def test_child_areas_match_table():
    for sides in (1, 2, 3):
        for special in range(3 if sides < 3 else 1):
            areas = [bary_area(c) for c in split_children(ROOT, sides, special)]
            assert np.allclose(areas, CHILD_AREAS[sides]), (sides, special, areas)


def rand_tree(rng, depth, n_states):
    if depth == 0 or rng.random() < 0.4:
        return rng.randrange(n_states)
    sides = rng.choice([1, 2, 3])
    special = 0 if sides == 3 else rng.randrange(3)
    return (sides, special, [rand_tree(rng, depth - 1, n_states) for _ in range(sides + 1)])


def test_leaves_tile_root_and_match_fractions():
    rng = random.Random(3)
    pts = sample_points(500, seed=5)
    for _ in range(50):
        t = rand_tree(rng, 4, 5)
        leaves = tree_leaves(t)
        assert abs(sum(bary_area(P) for P, _ in leaves) - 1) < 1e-9
        # every sample point in exactly one leaf
        from autocolor.paint import _inside
        counts = sum(_inside(P, pts).astype(int) for P, _ in leaves)
        assert (counts >= 1).all()
        # sampled fractions agree with analytic ones
        st = leaf_states_at(t, pts)
        f = np.bincount(st, minlength=5) / len(pts)
        assert np.abs(f - tree_fractions(t, 5)).max() < 0.06


def test_build_tree_follows_straight_boundary():
    # label 1 where u > 0.3 (u = bary coord of corner 1), else 2
    fn = lambda x: np.where(x[:, 1] > 0.3, 1, 2)
    t = build_tree(fn, max_depth=4)
    s = encode_tree(t)
    assert decode_tree(s) == t
    pts = sample_points(2000, seed=9)
    agree = (leaf_states_at(t, pts) == fn(pts)).mean()
    assert agree > 0.97, agree
    assert build_tree(lambda x: np.ones(len(x), int)) == 1


def test_exact_overlap_whole_vs_split():
    fn = lambda x: np.where(x[:, 1] > 0.5, 1, 2)
    t = build_tree(fn, max_depth=4)
    ref = [encode_tree(t)]
    cand = [encode_tree(1)]
    O = overlap_matrix_exact(ref, cand, np.array([1.0]), 2)
    # region bary1 > 0.5 is the corner triangle of area 1/4; the tree approximates
    # it and the scorer samples 64 points (+-5%), so compare to the tree's own fraction
    f = tree_fractions(t, 3)
    assert abs(f[1] - 0.25) < 0.03
    assert abs(O[0, 0] - f[1]) < 0.06 and abs(O[1, 0] - f[2]) < 0.06
