"""Phase 0: report OBJ stats and list objects in a 3MF (counts, bounds, paint_color presence)."""
import sys, zipfile, re
import numpy as np
import trimesh
import xml.etree.ElementTree as ET

def obj_stats(path):
    m = trimesh.load(path, process=False, force='mesh')
    print(f"OBJ {path}: verts={len(m.vertices)} tris={len(m.faces)}")
    print("  bounds min", m.bounds[0].round(4), "max", m.bounds[1].round(4), "extents", m.extents.round(4))
    return m

def strip(t): return t.split('}')[-1]

def inspect_3mf(path):
    z = zipfile.ZipFile(path)
    names = z.namelist()
    print(f"\n3MF {path}: {len(names)} entries")
    for n in names:
        print(f"  {n}  {z.getinfo(n).file_size}")
    for n in names:
        if not n.endswith('.model'): continue
        root = ET.fromstring(z.read(n))
        print(f"\n== {n}")
        for obj in root.iter():
            if strip(obj.tag) != 'object': continue
            mesh = next((c for c in obj if strip(c.tag) == 'mesh'), None)
            oid, name = obj.get('id'), obj.get('name')
            if mesh is None:
                comps = [c.get('objectid') for c in obj.iter() if strip(c.tag)=='component']
                print(f"object id={oid} name={name!r} type={obj.get('type')} components={comps}")
                continue
            V = np.array([[float(v.get(k)) for k in 'xyz'] for v in mesh.iter() if strip(v.tag)=='vertex'])
            tris = [t for t in mesh.iter() if strip(t.tag)=='triangle']
            attrs = set(); painted = 0
            for t in tris:
                for a in t.attrib:
                    attrs.add(a)
                if t.get('paint_color'): painted += 1
            print(f"object id={oid} name={name!r} verts={len(V)} tris={len(tris)}")
            print(f"  bounds min {V.min(0).round(4)} max {V.max(0).round(4)} extents {(V.max(0)-V.min(0)).round(4)}")
            print(f"  triangle attrs={sorted(attrs)} paint_color tris={painted} ({100*painted/len(tris):.1f}%)")
        # build/item transforms
        for it in root.iter():
            if strip(it.tag)=='item':
                print(f"build item objectid={it.get('objectid')} transform={it.get('transform')}")

if __name__ == '__main__':
    obj_stats(sys.argv[1])
    for p in sys.argv[2:]: inspect_3mf(p)
