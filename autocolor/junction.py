"""Junction-aware scoring. Parts come from the hand paint (connected regions
of each filament, typed by clicks.part_types); a junction is the set of mesh
edges where two parts of different type meet. For each junction type we
report the wrongly labelled area within `zone_mm` of the junction and the
misplaced boundary length (truth boundary with no predicted boundary within
`tol_mm`, plus predicted boundary in the zone with no truth boundary within
`tol_mm`)."""
import numpy as np
from scipy.spatial import cKDTree

TYPE_NAMES = {'F1:main': 'tree', 'F1:major': 'post', 'F2:major': 'loop', 'F3:major': 'rail',
              'F4:main': 'deck', 'F4:major': 'cap'}


def part_of_face(comp, types, names=None):
    """Per face: part type name or '' for slivers/dropped. names maps the
    simulator's type keys ('F1:main', ...) to display names; the Top names
    (tree, post, loop, rail, deck, cap) are the default and only make sense
    on Top -- on another model pass names={} to keep the raw keys. Minor
    (sliver) types are always ''."""
    names = TYPE_NAMES if names is None else names
    out = []
    for c in comp:
        t = types.get(int(c), '')
        out.append('' if (t == '' or t.endswith(':minor')) else names.get(t, t))
    return np.array(out, dtype=object)


class Junctions:
    def __init__(self, graph, V, part, truth, zone_mm=3.0, tol_mm=1.0):
        """graph: FaceGraph of the mesh being scored; part: per-face part name;
        truth: per-face truth label (dominant)."""
        self.graph, self.zone_mm, self.tol_mm = graph, zone_mm, tol_mm
        adj = graph.adj
        i, j = adj[:, 0], adj[:, 1]
        self.emid = (graph.C[i] + graph.C[j]) / 2
        self.elen = graph.elen
        pa, pb = part[i], part[j]
        self.pair = np.array([''] * len(adj), dtype=object)
        m = (pa != pb) & (pa != '') & (pb != '')
        self.pair[m] = [' - '.join(sorted((a, b))) for a, b in zip(pa[m], pb[m])]
        self.truth_bnd = truth[i] != truth[j]
        self.types = sorted(set(self.pair[m]))
        # zone faces per junction type: faces within zone_mm of a junction edge midpoint
        self.zone = {}
        self.jtree = {}
        for t in self.types:
            e = np.where(self.pair == t)[0]
            tree = cKDTree(self.emid[e])
            d, _ = tree.query(graph.C, distance_upper_bound=zone_mm)
            self.zone[t] = np.isfinite(d)
            self.jtree[t] = (tree, e)
        self.truth = truth

    def score(self, labels):
        """Dict junction type -> dict(zone_area, wrong_area, wrong_pct,
        truth_len, missed_len, spurious_len, misplaced_len)."""
        g = self.graph
        i, j = g.adj[:, 0], g.adj[:, 1]
        pred_bnd = labels[i] != labels[j]
        out = {}
        for t in self.types:
            z = self.zone[t]
            za = g.A[z].sum()
            wrong = g.A[z & (labels != self.truth)].sum()
            tree, e = self.jtree[t]
            # truth junction edges with no predicted boundary edge within tol
            pb = np.where(pred_bnd)[0]
            if len(pb):
                ptree = cKDTree(self.emid[pb])
                d, _ = ptree.query(self.emid[e], distance_upper_bound=self.tol_mm)
                missed = self.elen[e][~np.isfinite(d)].sum()
            else:
                missed = self.elen[e].sum()
            # predicted boundary edges inside the zone (either face) with no truth boundary within tol
            inzone = pb[z[i[pb]] | z[j[pb]]]
            tb = np.where(self.truth_bnd)[0]
            if len(inzone):
                ttree = cKDTree(self.emid[tb])
                d, _ = ttree.query(self.emid[inzone], distance_upper_bound=self.tol_mm)
                spurious = self.elen[inzone][~np.isfinite(d)].sum()
            else:
                spurious = 0.0
            out[t] = dict(zone_area=za, wrong_area=wrong, wrong_pct=100 * wrong / max(za, 1e-9),
                          truth_len=self.elen[e].sum(), missed_len=missed, spurious_len=spurious,
                          misplaced_len=missed + spurious)
        return out


def format_junctions(res, focus=('post - rail', 'loop - post', 'loop - rail')):
    lines = [f"{'junction':<16}{'zone mm2':>10}{'wrong mm2':>11}{'wrong%':>8}{'truth mm':>10}{'missed':>8}{'spurious':>10}{'misplaced':>11}"]
    order = [t for t in focus if t in res] + [t for t in sorted(res) if t not in focus]
    for t in order:
        r = res[t]
        lines.append(f"{t:<16}{r['zone_area']:10.0f}{r['wrong_area']:11.0f}{r['wrong_pct']:8.1f}{r['truth_len']:10.0f}"
                     f"{r['missed_len']:8.0f}{r['spurious_len']:10.0f}{r['misplaced_len']:11.0f}")
    return '\n'.join(lines)
