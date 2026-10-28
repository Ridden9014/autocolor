"""Seeded sectioning: user clicks -> per-face labels.

Pipeline (all on one mesh, normally the decimated copy):
  1. Each click (face, filament) becomes a training patch: faces within
     `patch_mm` geodesic distance of the click, all given the click's label.
  2. A gradient-boosted classifier on the per-face features (features.py)
     fit on the patches gives p(label | shape) for every face.
  3. Unary cost = -log p  - beta * exp(-geodesic_to_nearest_seed_of_label / tau)
     (the second term lets a click claim its surroundings even where the
     classifier is unsure). Seed faces are hard-constrained.
  4. Pairwise Potts cost on face adjacency: lam * edge_length * g(dihedral),
     g ~ 1 on smooth edges, small on sharp (especially concave) ones, so
     boundaries prefer creases. Minimised by alpha-expansion (PyMaxflow).

Hyper-parameters (patch_mm, lam, beta, tau, sigma_deg) are tuned on Top:
flagged as such, but they are physical/energy scales, not coordinates.
"""
import numpy as np
import maxflow
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingClassifier


class FaceGraph:
    def __init__(self, V, T):
        m = trimesh.Trimesh(V, T, process=False)
        self.C = m.triangles_center
        self.A = m.area_faces
        self.N = m.face_normals
        self.adj = m.face_adjacency                     # (E,2)
        self.angle = m.face_adjacency_angles            # unsigned dihedral, radians
        self.convex = m.face_adjacency_convex
        e = m.face_adjacency_edges
        self.elen = np.linalg.norm(V[e[:, 0]] - V[e[:, 1]], axis=1)
        self.cdist = np.linalg.norm(self.C[self.adj[:, 0]] - self.C[self.adj[:, 1]], axis=1)
        self.n = len(T)
        self.tree = cKDTree(self.C)

    def through_links(self, V, T, max_dist=12.0, cos_max=-0.7, mutual_tol=2.0):
        """Pairs (face, face hit by its inward normal ray) for thin parts: hit
        within max_dist, the hit face's normal anti-parallel (cos < cos_max),
        and MUTUAL: the hit face's own inward ray lands within mutual_tol mm
        of the source (so a rail fused into a loop lattice is not linked to
        the loop below it). Links the two sides of a slab, bar or ring so
        labels agree across the thickness (deck top <-> underside)."""
        from .features import Caster
        cast = Caster(V, T)
        N = cast.N
        C = self.C.astype(np.float32)
        t, p = cast.cast(C - 1e-3 * N, -N)
        ok = np.isfinite(t) & (t < max_dist) & (p < self.n)
        ok[ok] &= np.einsum('ij,ij->i', N[ok], N[p[ok]]) < cos_max
        src, dst = np.arange(self.n)[ok], p[ok]
        if mutual_tol is not None:
            back_ok = ok[dst]
            hitpt = C[dst] - 1e-3 * N[dst] - N[dst] * t[dst][:, None]
            mut = back_ok & (np.linalg.norm(hitpt - C[src], axis=1) < mutual_tol)
            src, dst = src[mut], dst[mut]
        links = np.c_[src, dst]
        self.links = links[links[:, 0] != links[:, 1]]
        return self.links

    def sparse(self, w):
        i, j = self.adj[:, 0], self.adj[:, 1]
        return coo_matrix((np.r_[w, w], (np.r_[i, j], np.r_[j, i])), shape=(self.n, self.n)).tocsr()

    def geodesic(self, sources, crease_penalty=3.0, sigma_deg=35.0, aff=None, kappa=10.0):
        """Multi-source Dijkstra on centroid distances; crossing a sharp
        concave edge costs (1 + crease_penalty) times more, and with an
        affinity aff (E,) in 0..1, crossing a shape change costs up to
        (1 + kappa) times more."""
        sharp = np.exp(-(np.degrees(self.angle) / sigma_deg) ** 2)
        w = self.cdist * (1 + crease_penalty * (1 - sharp) * (~self.convex))
        if aff is not None:
            w = w * (1 + kappa * (1 - aff))
        g = self.sparse(w)
        return dijkstra(g, directed=False, indices=np.asarray(sources), min_only=True)

    def faces_near(self, face):
        return self.tree.query(self.C[face])[1]


def clicks_to_mesh(clicks, C_full, graph):
    """Map (full-mesh face, label) clicks to this mesh's faces by centroid."""
    return [(int(graph.tree.query(C_full[f])[1]), lab) for f, lab in clicks]


LOG_FEATURES = ['thick_sdf', 'thick_min', 'tan_min', 'tan_med', 'tan_max', 'tan_aniso', 'height']
AFFINITY_FEATURES = ['thick_sdf', 'thick_min', 'tan_min', 'tan_med', 'tan_max', 'curv_3', 'plan_12', 'sph_12']


def prep_features(X, names):
    """log1p the length-like columns; return (Xp, standardized copy)."""
    Xp = X.copy()
    for nm in LOG_FEATURES:
        if nm in names:
            Xp[:, names.index(nm)] = np.log1p(np.maximum(X[:, names.index(nm)], 0))
    mu, sd = Xp.mean(0), Xp.std(0) + 1e-9
    return Xp, (Xp - mu) / sd


def pairwise_weights(graph, Z, names, lam, sigma_deg, concave_bonus, sigma_feat):
    """lam * edge_length * g(dihedral) * exp(-|dZ|^2 / 2 sigma_feat^2) over the
    affinity features: cheap to cut across creases and across shape changes."""
    sharp = np.exp(-(np.degrees(graph.angle) / sigma_deg) ** 2)
    g = sharp + (1 - sharp) * np.where(graph.convex, 1 - concave_bonus * 0.5, 1 - concave_bonus)
    aff = feature_affinity(graph, Z, names, sigma_feat)
    return lam * graph.elen * g * aff


def feature_affinity(graph, Z, names, sigma_feat):
    cols = [names.index(nm) for nm in AFFINITY_FEATURES if nm in names]
    dz = Z[graph.adj[:, 0]][:, cols] - Z[graph.adj[:, 1]][:, cols]
    return np.exp(-(dz ** 2).sum(1) / (2 * sigma_feat ** 2 * len(cols)))


def _balanced(lab, w=None):
    """Sample weights so every label carries the same total weight."""
    w = np.ones(len(lab)) if w is None else np.asarray(w, float)
    tot = {l: w[lab == l].sum() for l in np.unique(lab)}
    return np.array([w[i] / tot[l] for i, l in enumerate(lab)]) * len(lab) / len(tot)


def _fit_predict(Xz, idx, lab, w, k, kind, balance=True, groups=None, h=1.0):
    """P(label | features). kind: 'gbm', 'logreg', 'kde' (per-label Gaussian
    kernel density, mean over the label's samples) or 'proto' (per-group
    kernel density, P(l|x) ~ max over groups of label l; groups = click id per
    sample, so a label made of unlike parts is not diluted). balance=True
    gives every label the same total weight."""
    present = np.unique(lab)
    P = np.full((len(Xz), k), 1e-3)
    wb = _balanced(lab, w) if balance else (np.ones(len(lab)) if w is None else np.asarray(w, float))
    if kind == 'proto':
        best = np.full((len(Xz), k), -np.inf)
        for g in np.unique(groups):
            S = Xz[idx[groups == g]]
            l = lab[groups == g][0]
            out = np.zeros(len(Xz))
            for s in range(0, len(Xz), 10000):
                m = -((Xz[s:s + 10000, None, :] - S[None]) ** 2).sum(2) / (2 * h * h)
                mx = m.max(1, keepdims=True)
                out[s:s + 10000] = mx[:, 0] + np.log(np.exp(m - mx).mean(1))
            best[:, l - 1] = np.maximum(best[:, l - 1], out)
        best -= best.max(1, keepdims=True)
        P = np.exp(best)
        P[:, [c for c in range(k) if c + 1 not in present]] = 1e-3
    elif kind == 'kde':
        from scipy.spatial import cKDTree
        from sklearn.neighbors import NearestNeighbors
        h = np.median(NearestNeighbors(n_neighbors=6).fit(Xz[idx]).kneighbors(Xz[idx])[0][:, 1:]) + 1e-6
        dens = np.zeros((len(Xz), len(present)))
        for c, l in enumerate(present):
            S = Xz[idx[lab == l]]
            d2 = ((Xz[:, None, :] - S[None]) ** 2).sum(2) if len(S) * len(Xz) < 4e7 else None
            if d2 is None:  # chunk
                out = np.zeros(len(Xz))
                for s in range(0, len(Xz), 20000):
                    out[s:s + 20000] = np.exp(-((Xz[s:s + 20000, None, :] - S[None]) ** 2).sum(2) / (2 * h * h)).mean(1)
                dens[:, c] = out
            else:
                dens[:, c] = np.exp(-d2 / (2 * h * h)).mean(1)
        P[:, present - 1] = dens / np.maximum(dens.sum(1, keepdims=True), 1e-300)
    elif kind == 'gbm':
        clf = HistGradientBoostingClassifier(max_iter=150, max_depth=4, min_samples_leaf=10)
        clf.fit(Xz[idx], lab, sample_weight=wb)
        P[:, present - 1] = clf.predict_proba(Xz)
    else:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(C=1.0, max_iter=3000).fit(Xz[idx], lab, sample_weight=wb)
        P[:, present - 1] = clf.predict_proba(Xz)
    return P / P.sum(1, keepdims=True), present


def seeded_labels(graph, X, names, clicks, k, lam=3.0, beta=0.0, tau=15.0, sigma_deg=35.0,
                  concave_bonus=0.8, sigma_feat=1.0, rounds=2, patch_mm=6.0, cycles=3,
                  kappa=10.0, prior='proto', conf=0.0, balance=False, h=1.0, lam_through=3.0,
                  verbose=False, score_fn=None):
    """Seeds -> labels (N,) in 1..k.
    Round 0: unary = -log P from a per-label kernel density over the click
      patches (equal class priors, so click counts per label do not matter);
      optional geodesic pull (beta); feature-aware Potts pairwise;
      alpha-expansion.
    Rounds 1..: retrain a class-balanced GBM on the previous labeling
      (confident faces away from label boundaries), rebuild the unary from its
      probabilities, cut again. This carries a label from a clicked part to
      unclicked parts of the same kind."""
    seeds = np.array([f for f, _ in clicks])
    seed_lab = np.array([l for _, l in clicks])
    n = graph.n
    Xp, Z = prep_features(X, names)
    aff = feature_affinity(graph, Z, names, sigma_feat)
    geo = np.full((n, k), np.inf)
    idx0, lab0, grp0 = [], [], []
    for l in range(1, k + 1):
        s = seeds[seed_lab == l]
        if len(s) == 0:
            continue
        geo[:, l - 1] = graph.geodesic(s, aff=aff, kappa=kappa)
    for ci, (f, l) in enumerate(clicks):   # one patch (group) per click
        near = np.where(graph.geodesic([f], aff=aff, kappa=kappa) <= patch_mm)[0]
        idx0.append(near)
        lab0.append(np.full(len(near), l))
        grp0.append(np.full(len(near), ci))
    idx0, lab0, grp0 = np.concatenate(idx0), np.concatenate(lab0), np.concatenate(grp0)
    present = np.unique(lab0)
    W = pairwise_weights(graph, Z, names, lam, sigma_deg, concave_bonus, sigma_feat)
    adj = graph.adj
    if lam_through > 0 and getattr(graph, 'links', None) is not None and len(graph.links):
        adj = np.r_[graph.adj, graph.links]
        W = np.r_[W, np.full(len(graph.links), lam_through * np.median(graph.elen))]
    pull = -beta * np.exp(-np.minimum(geo, 1e6) / tau)

    def cut(P, init=None):
        U = -np.log(np.clip(P, 1e-4, 1)) + pull
        U[:, [l - 1 for l in range(1, k + 1) if l not in present]] = 50.0
        U[seeds, :] = 50.0
        U[seeds, seed_lab - 1] = 0.0
        return alpha_expansion(U, adj, W, init=init, cycles=cycles, verbose=verbose) + 1

    P, _ = _fit_predict(Z, idx0, lab0, np.ones(len(idx0)), k, prior, groups=grp0, h=h)
    labels = cut(P)
    if score_fn:
        print('  round 0:', score_fn(labels))
    for r in range(rounds):
        # training set: confident faces at least 2 edges away from any label change
        i, j = graph.adj[:, 0], graph.adj[:, 1]
        bnd = np.zeros(n, bool)
        bnd[i[labels[i] != labels[j]]] = True
        bnd[j[labels[i] != labels[j]]] = True
        nb = graph.sparse(np.ones(len(i)))
        near_b = bnd | (nb @ bnd.astype(float) > 0)
        near_b = near_b | (nb @ near_b.astype(float) > 0)
        confident = P[np.arange(n), labels - 1] >= conf
        tr = np.where(~near_b & confident)[0]
        P, _ = _fit_predict(Z, tr, labels[tr], graph.A[tr], k, 'gbm', balance=balance)
        new = cut(P, init=labels - 1)
        if verbose:
            print(f'  round {r + 1}: {100 * (new != labels).mean():.2f}% faces changed')
        labels = new
        if score_fn:
            print(f'  round {r + 1}:', score_fn(labels))
    return labels, P


def oversegment(graph, Z, names, n_clusters=16, lam=2.0, sigma_deg=35.0, concave_bonus=0.8,
                sigma_feat=1.0, seed=0):
    """Unsupervised geometric parts: GMM on the standardized features ->
    cluster posteriors as unary -> feature-aware Potts graph cut -> connected
    components of equal cluster = segments. Returns seg (N,) ids."""
    from sklearn.mixture import GaussianMixture
    from scipy.sparse.csgraph import connected_components
    gmm = GaussianMixture(n_clusters, covariance_type='full', random_state=seed, max_iter=200).fit(Z[::5])
    U = -np.log(np.clip(gmm.predict_proba(Z), 1e-6, 1))
    W = pairwise_weights(graph, Z, names, lam, sigma_deg, concave_bonus, sigma_feat)
    cl = alpha_expansion(U, graph.adj, W, cycles=2)
    i, j = graph.adj[:, 0], graph.adj[:, 1]
    same = cl[i] == cl[j]
    g = coo_matrix((np.ones(same.sum()), (i[same], j[same])), shape=(graph.n, graph.n))
    return connected_components(g, directed=False)[1]


def seeded_segments(graph, X, names, clicks, k, seg, lam=3.0, sigma_deg=35.0, concave_bonus=0.8,
                    sigma_feat=1.0, rounds=1, cycles=3, verbose=False, score_fn=None):
    """Seeds + over-segmentation -> labels (N,) in 1..k.
    1. Every segment that contains a click takes the click's label.
    2. A class-balanced GBM on the faces of labeled segments predicts every
       face; unlabeled segments take the area-weighted argmax.
    3. Face-level graph cut (unary = -log GBM probabilities, seeds hard)
       refines boundaries beyond the segment borders. Optional extra rounds
       retrain the GBM on the result."""
    seeds = np.array([f for f, _ in clicks])
    seed_lab = np.array([l for _, l in clicks])
    n = graph.n
    Xp, Z = prep_features(X, names)
    nseg = seg.max() + 1
    seg_lab = np.zeros(nseg, int)
    for f, l in clicks:
        seg_lab[seg[f]] = l           # last click wins on conflicts (rare)
    lab_faces = seg_lab[seg] > 0
    idx = np.where(lab_faces)[0]
    P, present = _fit_predict(Z, idx, seg_lab[seg][idx], graph.A[idx], k, 'gbm')
    # segment-level decision for unlabeled segments
    SP = np.zeros((nseg, k))
    for c in range(k):
        SP[:, c] = np.bincount(seg, weights=graph.A * P[:, c], minlength=nseg)
    seg_pred = SP.argmax(1) + 1
    seg_pred[seg_lab > 0] = seg_lab[seg_lab > 0]
    labels = seg_pred[seg]
    if score_fn:
        print('  segments labeled:', score_fn(labels))
    W = pairwise_weights(graph, Z, names, lam, sigma_deg, concave_bonus, sigma_feat)

    def cut(P, init):
        U = -np.log(np.clip(P, 1e-4, 1))
        U[:, [l - 1 for l in range(1, k + 1) if l not in present]] = 50.0
        U[seeds, :] = 50.0
        U[seeds, seed_lab - 1] = 0.0
        return alpha_expansion(U, graph.adj, W, init=init, cycles=cycles) + 1

    for r in range(rounds):
        tr = np.arange(n)
        P, present = _fit_predict(Z, tr, labels, graph.A, k, 'gbm')
        labels = cut(P, labels - 1)
        if score_fn:
            print(f'  round {r + 1} (gbm on all + cut):', score_fn(labels))
    return labels, P


def alpha_expansion(U, adj, W, init=None, cycles=3, verbose=False):
    """Potts multi-label minimisation. U (N,k) unary, adj (E,2), W (E,) >= 0.
    Returns labels in 0..k-1."""
    n, k = U.shape
    lab = U.argmin(1) if init is None else init.copy()
    i, j = adj[:, 0], adj[:, 1]

    def energy(l):
        return U[np.arange(n), l].sum() + (W * (l[i] != l[j])).sum()

    E = energy(lab)
    for c in range(cycles):
        changed = False
        for a in range(k):
            g = maxflow.Graph[float](n, len(adj))
            nodes = g.add_nodes(n)
            # x=0 (source side) keep current label, x=1 (sink side) switch to a
            stay = U[np.arange(n), lab].copy()
            switch = U[:, a].copy()
            same = lab[i] == lab[j]
            diff = ~same
            # diff edges: cost w if p stays (unary on p) + w if p switches and q stays
            np.add.at(stay, i[diff], W[diff])
            g.add_grid_tedges(nodes, switch, stay)   # (source cap cut when x=1, sink cap cut when x=0)
            # same: Potts both directions
            g.add_edges(nodes[i[same]], nodes[j[same]], W[same], W[same])
            # diff: cost when p switches (x_p=1) and q stays (x_q=0): edge q->p
            g.add_edges(nodes[i[diff]], nodes[j[diff]], np.zeros(diff.sum()), W[diff])
            g.maxflow()
            sw = g.get_grid_segments(nodes)          # True = sink side = switch
            new = np.where(sw, a, lab)
            En = energy(new)
            if En < E - 1e-9:
                lab, E, changed = new, En, True
        if verbose:
            print(f'  cycle {c}: energy {E:.1f}')
        if not changed:
            break
    return lab
