"""Simulate user clicks: connected regions of each filament in the hand paint,
and 1..n click points per region, biased toward the region interior."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def face_graph(T, n):
    """Symmetric sparse face adjacency from shared edges."""
    import trimesh
    adj = trimesh.Trimesh(np.zeros((T.max() + 1, 3)), T, process=False).face_adjacency
    return adj


def regions(T, labels, areas, min_area=20.0):
    """Connected components of equal label by face adjacency.
    Returns comp (N,) component id or -1 for dropped faces, and a table of
    kept components: list of dict(id, label, area, n_faces)."""
    adj = face_graph(T, len(T))
    same = labels[adj[:, 0]] == labels[adj[:, 1]]
    g = coo_matrix((np.ones(same.sum()), (adj[same, 0], adj[same, 1])), shape=(len(T), len(T)))
    _, comp = connected_components(g, directed=False)
    carea = np.bincount(comp, weights=areas)
    keep = carea >= min_area
    table = [dict(id=int(c), label=int(labels[np.argmax(comp == c)]), area=float(carea[c]),
                  n_faces=int((comp == c).sum())) for c in np.where(keep)[0]]
    comp = np.where(keep[comp], comp, -1)
    return comp, table, adj


def boundary_hops(adj, labels, n):
    """Per face: BFS hop distance to the nearest face with a different label."""
    diff = labels[adj[:, 0]] != labels[adj[:, 1]]
    dist = np.full(n, np.iinfo(np.int32).max, dtype=np.int64)
    frontier = np.unique(adj[diff].ravel())
    dist[frontier] = 0
    same = adj[~diff]
    nbr = coo_matrix((np.ones(len(same) * 2), (np.r_[same[:, 0], same[:, 1]], np.r_[same[:, 1], same[:, 0]])),
                     shape=(n, n)).tocsr()
    d = 0
    while len(frontier):
        d += 1
        nxt = np.unique(nbr[frontier].indices)
        nxt = nxt[dist[nxt] > d]
        dist[nxt] = d
        frontier = nxt
    return dist


def part_types(table, main_ratio=2.0, big_ratio=0.25):
    """Assign each region a part-type string, used only by the click simulator
    to express 'one click per kind of part'. FLAGGED model-specific grouping
    by area: within a filament, the largest region is 'main' if it is >=
    main_ratio x the second largest; of the remaining regions, those >=
    big_ratio x the largest remaining are 'major', the rest 'minor' (slivers).
    On Top: F1 main (tree) + 8 major (posts); F2 4 major (loop sides) + 4 minor;
    F3 8 major (rails) + 3 minor; F4 main (deck) + 6 major (caps)."""
    out = {}
    for f in sorted({r['label'] for r in table}):
        rs = sorted([r for r in table if r['label'] == f], key=lambda r: -r['area'])
        if len(rs) == 1 or rs[0]['area'] >= main_ratio * rs[1]['area']:
            out[rs[0]['id']] = f'F{f}:main'
            rs = rs[1:]
        if rs:
            top = rs[0]['area']
            for r in rs:
                out[r['id']] = f'F{f}:major' if r['area'] >= big_ratio * top else f'F{f}:minor'
    return out


def _one_click(faces, hops, areas, rng):
    w = areas[faces] * (hops[faces].astype(float) + 0.5) ** 2
    return int(rng.choice(faces, p=w / w.sum()))


def sample_clicks_condition(comp, table, hops, centers, areas, condition, rng):
    """condition: 'per_region_1/2/3', 'per_filament' (one click in the largest
    region of each filament), 'per_type' (one click in one randomly chosen
    region of each part type, minor types excluded -- the question being
    whether one post / one loop side / one cap labels all of them)."""
    if condition.startswith('per_region_'):
        return sample_clicks(comp, table, hops, centers, areas, int(condition[-1]), rng)
    out = []
    if condition == 'per_filament':
        for f in sorted({r['label'] for r in table}):
            r = max([r for r in table if r['label'] == f], key=lambda r: r['area'])
            out.append((_one_click(np.where(comp == r['id'])[0], hops, areas, rng), f))
    elif condition == 'per_type':
        types = part_types(table)
        for t in sorted(set(types.values())):
            if t.endswith(':minor'):
                continue  # slivers
            rs = [r for r in table if types[r['id']] == t]
            r = rs[rng.integers(len(rs))]
            out.append((_one_click(np.where(comp == r['id'])[0], hops, areas, rng), r['label']))
    else:
        raise ValueError(condition)
    return out


def sample_clicks(comp, table, hops, centers, areas, n_clicks, rng, n_cand=64):
    """One draw: for each kept region, n_clicks faces. First click is drawn
    with weight area * hops^2 (interior bias); later clicks additionally
    prefer faces far from the clicks already placed (farthest of n_cand
    weighted candidates). Returns list of (face, label)."""
    out = []
    for r in table:
        faces = np.where(comp == r['id'])[0]
        w = areas[faces] * (hops[faces].astype(float) + 0.5) ** 2
        w /= w.sum()
        chosen = [rng.choice(faces, p=w)]
        for _ in range(n_clicks - 1):
            cand = rng.choice(faces, size=min(n_cand, len(faces)), p=w)
            dmin = np.min(np.linalg.norm(centers[cand][:, None] - centers[chosen][None], axis=2), axis=1)
            chosen.append(cand[np.argmax(dmin)])
        out.extend((int(f), r['label']) for f in chosen)
    return out
