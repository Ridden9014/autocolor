import random
import numpy as np
import pytest
from autocolor.paint import (decode_tree, encode_tree, encode_state,
                             tree_fractions, fractions, CHILD_AREAS)

# Whole-triangle codes seen in the Bambu-painted reference file.
KNOWN = {'4': 1, '8': 2, '0C': 3, '1C': 4}


def test_known_bambu_codes():
    for s, state in KNOWN.items():
        assert decode_tree(s) == state
        assert encode_state(state) == s


@pytest.mark.parametrize('state', list(range(0, 10)) + [17, 18, 19, 33, 40])
def test_state_roundtrip(state):
    s = encode_tree(state)
    assert decode_tree(s) == state
    assert len(s) <= 1 + 1 + (state - 3) // 15 + 1


def test_unpainted_is_empty():
    assert encode_state(0) == ''
    assert fractions('', 5)[0] == 1.0


def rand_tree(rng, depth, n_states):
    if depth == 0 or rng.random() < 0.4:
        return rng.randrange(n_states)
    sides = rng.choice([1, 2, 3])
    special = 0 if sides == 3 else rng.randrange(3)
    return (sides, special, [rand_tree(rng, depth - 1, n_states) for _ in range(sides + 1)])


def test_tree_roundtrip_up_to_9_filaments():
    rng = random.Random(0)
    for _ in range(500):
        t = rand_tree(rng, 5, 10)  # states 0..9
        s = encode_tree(t)
        assert decode_tree(s) == t, s


def test_fractions_sum_to_one():
    rng = random.Random(1)
    for _ in range(200):
        t = rand_tree(rng, 4, 10)
        f = tree_fractions(t, 10)
        assert abs(f.sum() - 1) < 1e-12


def test_two_side_split_half_child_is_serialized_first():
    # child order [1/4, 1/4, 1/2]; stream is reversed so the 1/2 leaf comes first
    t = (2, 0, [1, 1, 2])
    f = tree_fractions(t, 3)
    assert f[2] == 0.5 and f[1] == 0.5
    nib = [int(c, 16) for c in reversed(encode_tree(t))]
    assert nib[0] & 3 == 2 and nib[1] >> 2 == 2  # first child read has state 2


def test_leftover_nibbles_rejected():
    with pytest.raises(ValueError):
        decode_tree('44')
    with pytest.raises(ValueError):
        decode_tree('1')  # split with no children
