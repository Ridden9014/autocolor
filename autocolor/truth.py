"""Load a Bambu Studio project 3MF: filament list, objects, per-triangle
paint, and build the (N, n_filaments) ground-truth area matrix."""
import json
import re
import zipfile
import numpy as np
from .paint import fractions_matrix, decode_tree, tree_depth

_VERT = re.compile(r'<vertex x="([^"]+)" y="([^"]+)" z="([^"]+)"')
_TRI = re.compile(r'<triangle v1="(\d+)" v2="(\d+)" v3="(\d+)"([^/]*)/>')
_PAINT = re.compile(r'paint_color="([^"]*)"')


class Project:
    def __init__(self, path):
        self.path = path
        self.z = zipfile.ZipFile(path)
        names = self.z.namelist()
        cfg = {}
        if 'Metadata/project_settings.config' in names:
            cfg = json.loads(self.z.read('Metadata/project_settings.config'))
        self.filament_colours = cfg.get('filament_colour', [])
        self.n_filaments = len(self.filament_colours)
        self.objects = self._read_objects(names)

    def _read_objects(self, names):
        """Map build object id -> dict(name, extruder, part_id, part_file, offset)."""
        objs = {}
        if 'Metadata/model_settings.config' in names:
            ms = self.z.read('Metadata/model_settings.config').decode()
            for m in re.finditer(r'<object id="(\d+)">(.*?)</object>', ms, re.S):
                oid, body = int(m.group(1)), m.group(2)
                md = dict(re.findall(r'<metadata key="([^"]+)" value="([^"]*)"', body))
                part = re.search(r'<part id="(\d+)"', body)
                off = [float(md.get(f'source_offset_{a}', 0)) for a in 'xyz']
                objs[oid] = dict(name=md.get('name'), extruder=int(md.get('extruder', 0)),
                                 part_id=int(part.group(1)) if part else None,
                                 offset=np.array(off), part_file=None)
        model = self.z.read('3D/3dmodel.model').decode()
        for m in re.finditer(r'<object id="(\d+)"[^>]*>\s*<components>\s*<component p:path="([^"]+)" objectid="(\d+)"', model):
            oid = int(m.group(1))
            objs.setdefault(oid, dict(name=None, extruder=0, offset=np.zeros(3)))
            objs[oid]['part_file'] = m.group(2).lstrip('/')
            objs[oid]['part_id'] = int(m.group(3))
        return objs

    def default_extruder(self, oid):
        """Bambu stores 0 for 'Default', which resolves to filament 1."""
        e = self.objects[oid]['extruder']
        return e if e > 0 else 1

    def load_mesh(self, oid, world=True):
        """Return V (M,3), T (N,3), paint (list of N strings) for object oid."""
        o = self.objects[oid]
        s = self.z.read(o['part_file']).decode()
        V = np.array(_VERT.findall(s), dtype=np.float64)
        tris = _TRI.findall(s)
        T = np.array([t[:3] for t in tris], dtype=np.int64)
        paint = []
        for t in tris:
            m = _PAINT.search(t[3])
            paint.append(m.group(1) if m else '')
        if world:
            V = V + o['offset']
        return V, T, paint


def load_paint_3mf(path):
    """Load the first mesh object in any .model entry of a plain 3MF (no Bambu
    project metadata), e.g. our own exports. Returns V, T, paint."""
    z = zipfile.ZipFile(path)
    for n in z.namelist():
        if not n.endswith('.model'):
            continue
        s = z.read(n).decode()
        if '<mesh>' not in s:
            continue
        V = np.array(_VERT.findall(s), dtype=np.float64)
        tris = _TRI.findall(s)
        T = np.array([t[:3] for t in tris], dtype=np.int64)
        paint = [(_PAINT.search(t[3]) or [None, ''])[1] for t in tris]
        return V, T, paint
    raise ValueError(f'no mesh in {path}')


def tri_areas(V, T):
    a = V[T[:, 1]] - V[T[:, 0]]
    b = V[T[:, 2]] - V[T[:, 0]]
    return 0.5 * np.linalg.norm(np.cross(a, b), axis=1)


def truth_matrix(paint, areas, n_filaments, default_extruder):
    """(N, n_filaments) area per filament (0-based column = filament-1).
    Unpainted area (state 0) is assigned to default_extruder."""
    F = fractions_matrix(paint, n_filaments + 1)       # col 0 = unpainted
    F[:, default_extruder] += F[:, 0]
    return F[:, 1:] * areas[:, None]


def paint_stats(paint):
    """Parse every distinct string; report counts, depth, max state."""
    uniq = set(paint)
    bad, max_depth, max_state, split = 0, 0, 0, 0
    for s in uniq:
        if s == '':
            continue
        try:
            t = decode_tree(s)
        except ValueError:
            bad += 1
            continue
        d = tree_depth(t)
        max_depth = max(max_depth, d)
        if d:
            split += 1
        from .paint import tree_max_state
        max_state = max(max_state, tree_max_state(t))
    n_split = sum(1 for s in paint if len(s) > 2)
    return dict(n=len(paint), distinct=len(uniq), unparsable=bad, painted=sum(1 for s in paint if s),
                split_tris=n_split, max_depth=max_depth, max_state=max_state)
