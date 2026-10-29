"""Plain mesh loaders (no paint): binary STL with vertex merging."""
import struct
import numpy as np


def load_stl(path):
    """Binary STL -> (V, T) with exactly-equal vertices merged, degenerate
    faces dropped. Triangle order is the file's order (needed to write paint
    back against the same triangles)."""
    with open(path, 'rb') as f:
        f.read(80)
        n = struct.unpack('<I', f.read(4))[0]
        d = np.fromfile(f, dtype=np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')]), count=n)
    P = d['v'].reshape(-1, 3).astype(np.float64)
    V, inv = np.unique(P, axis=0, return_inverse=True)
    T = inv.reshape(-1, 3)
    good = (T[:, 0] != T[:, 1]) & (T[:, 1] != T[:, 2]) & (T[:, 0] != T[:, 2])
    return V, T, good
