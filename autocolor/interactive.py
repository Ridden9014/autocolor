"""Click-and-correct session: the seeded method of seeded.py kept as state so
that adding one click costs one patch + one kernel density + one warm-started
graph cut, not a full recompute."""
import time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from .seeded import (prep_features, feature_affinity, pairwise_weights, alpha_expansion)


class InteractiveSession:
    def __init__(self, graph, X, names, k, lam=3.0, lam_through=3.0, h=1.0, patch_mm=6.0,
                 sigma_deg=35.0, concave_bonus=0.8, sigma_feat=1.0, kappa=10.0, cycles=3):
        self.graph, self.k, self.h, self.patch_mm, self.kappa, self.cycles = graph, k, h, patch_mm, kappa, cycles
        _, self.Z = prep_features(X, names)
        self.aff = feature_affinity(graph, self.Z, names, sigma_feat)
        W = pairwise_weights(graph, self.Z, names, lam, sigma_deg, concave_bonus, sigma_feat)
        self.adj = graph.adj
        self.W = W
        if lam_through > 0 and getattr(graph, 'links', None) is not None and len(graph.links):
            self.adj = np.r_[graph.adj, graph.links]
            self.W = np.r_[W, np.full(len(graph.links), lam_through * np.median(graph.elen))]
        self.best = np.full((graph.n, k), -np.inf)   # max over clicks of each label's log density
        self.seeds, self.seed_lab, self.clicks = [], [], []
        self.labels = None
        self.timings = []
        self.prims = None

    def set_primitives(self, prims, owner, strength=4.0, propagate_kinds=('cyl', 'rail'), mode='seed',
                       link_w=2.0, link_k=3, seed=0):
        """Fitted primitives (primitives.fit_all).
        mode='seed': a click inside a primitive labels the whole primitive
          (majority of core clicks); unlabeled primitives of a kind in
          propagate_kinds take the label of clicked ones of that kind; claimed
          faces get a unary bonus `strength` (50 = hard).
        mode='link': no voting. Each claimed face is linked to link_k random
          other faces of the same primitive with Potts weight link_w x median
          edge length, so a primitive strongly prefers one label; which label
          is decided by the cut from clicks and prototypes as usual."""
        core_owner = prims[0]['core_owner'] if prims and 'core_owner' in prims[0] else owner
        self.prims = dict(kinds=[p['kind'] for p in prims], owner=owner, core_owner=core_owner,
                          strength=strength, propagate=propagate_kinds, mode=mode)
        if mode == 'link':
            rng = np.random.default_rng(seed)
            pairs = []
            for i in range(len(prims)):
                f = np.where(owner == i)[0]
                if len(f) < 2:
                    continue
                for _ in range(link_k):
                    pairs.append(np.c_[f, rng.permutation(f)])
            pairs = np.concatenate(pairs)
            pairs = pairs[pairs[:, 0] != pairs[:, 1]]
            self.adj = np.r_[self.adj, pairs]
            self.W = np.r_[self.W, np.full(len(pairs), link_w * np.median(self.graph.elen))]

    def primitive_labels(self):
        """A primitive's label is the majority of the clicks on its CORE faces
        (its own surface, not the fringe of its claim), earliest click breaking
        ties. A click on the fringe, or a disagreeing correction, acts only
        through its own hard patch."""
        kinds, core_owner = self.prims['kinds'], self.prims['core_owner']
        votes = {}
        for f, l in self.clicks:
            if core_owner[f] >= 0:
                votes.setdefault(int(core_owner[f]), []).append(l)
        lab = {i: max(dict.fromkeys(v), key=v.count) for i, v in votes.items()}
        for kind in self.prims['propagate']:
            got = [lab[i] for i in lab if kinds[i] == kind]
            if got:
                maj = max(set(got), key=got.count)
                for i, kd in enumerate(kinds):
                    if kd == kind and i not in lab:
                        lab[i] = maj
        return lab

    def add_click(self, face, label, reach=None, gate=None):
        """reach: geodesic radius (mm) within which this click's prototype
        applies; None = whole mesh (a part-level click), a number = a local
        correction."""
        t0 = time.time()
        g = self.graph
        dgeo = g.geodesic([face], aff=self.aff, kappa=self.kappa)
        near = np.where(dgeo <= self.patch_mm)[0]
        S = self.Z[near]
        out = np.empty(g.n)
        for s in range(0, g.n, 20000):
            m = -((self.Z[s:s + 20000, None, :] - S[None]) ** 2).sum(2) / (2 * self.h * self.h)
            mx = m.max(1, keepdims=True)
            out[s:s + 20000] = mx[:, 0] + np.log(np.exp(m - mx).mean(1))
        if reach is not None:
            out[dgeo > reach] = -np.inf
        if gate is not None:
            # extended support: faces within `reach` that look like the clicked
            # patch (feature distance to the patch mean <= gate std) are pulled
            # strongly to this label -- a click claims its part, not just a dab
            fd = np.linalg.norm(self.Z - S.mean(0), axis=1)
            sup = (fd <= gate) & (dgeo <= (reach if reach is not None else np.inf))
            out[sup] = np.maximum(out[sup], 0.0)
        self.best[:, label - 1] = np.maximum(self.best[:, label - 1], out)
        # the whole patch is hard-seeded (a click is a small brush dab)
        self.seeds.extend(int(f) for f in near)
        self.seed_lab.extend([label] * len(near))
        self.clicks.append((int(face), int(label)))
        self.timings.append(('click', time.time() - t0))

    def solve(self):
        t0 = time.time()
        present = sorted(set(self.seed_lab))
        lp = self.best - self.best.max(1, keepdims=True)
        P = np.exp(lp)
        P[:, [c for c in range(self.k) if c + 1 not in present]] = 1e-3
        P /= P.sum(1, keepdims=True)
        U = -np.log(np.clip(P, 1e-4, 1))
        U[:, [c for c in range(self.k) if c + 1 not in present]] = 50.0
        if self.prims is not None and self.prims['mode'] == 'seed':
            owner, s = self.prims['owner'], self.prims['strength']
            for i, l in self.primitive_labels().items():
                m = owner == i
                U[m, :] += s
                U[m, l - 1] -= s
        seeds, sl = np.array(self.seeds), np.array(self.seed_lab)
        U[seeds, :] = 50.0
        U[seeds, sl - 1] = 0.0
        init = None if self.labels is None else self.labels - 1
        self.labels = alpha_expansion(U, self.adj, self.W, init=init, cycles=self.cycles) + 1
        self.timings.append(('solve', time.time() - t0))
        return self.labels


def largest_wrong_region(graph, labels, truth, areas, clicked=()):
    """Largest connected component (by area) of faces whose label differs from
    the truth, within equal truth label, skipping components that already
    contain a click (a click that did not fix its region is not repeated).
    Returns (face at its centre, truth label, area, skipped_area) or None."""
    wrong = labels != truth
    if not wrong.any():
        return None
    i, j = graph.adj[:, 0], graph.adj[:, 1]
    keep = wrong[i] & wrong[j] & (truth[i] == truth[j])
    g = coo_matrix((np.ones(keep.sum()), (i[keep], j[keep])), shape=(graph.n, graph.n))
    _, comp = connected_components(g, directed=False)
    comp = np.where(wrong, comp, -1)
    ca = np.bincount(comp[wrong], weights=areas[wrong], minlength=comp.max() + 1)
    skipped = 0.0
    for f in clicked:
        if comp[f] >= 0:
            skipped += ca[comp[f]]
            ca[comp[f]] = 0
    if ca.max() <= 0:
        return None
    c = int(ca.argmax())
    faces = np.where(comp == c)[0]
    cen = (graph.C[faces] * areas[faces, None]).sum(0) / areas[faces].sum()
    f = faces[np.argmin(((graph.C[faces] - cen) ** 2).sum(1))]
    return int(f), int(truth[f]), float(ca[c]), skipped
