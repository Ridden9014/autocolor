import numpy as np, zipfile, io
V=np.load('V.npy');F=np.load('F.npy');lab=np.load('lab.npy')
ext=np.array([1,2,2,3,4])[lab]       # deck=1, frame+posts=2, railing=3, tree=4
code={1:'4',2:'8',3:'0C',4:'1C'}
vs='\n'.join(f'<vertex x="{a:.5f}" y="{b:.5f}" z="{c:.5f}"/>' for a,b,c in V)
ts='\n'.join(f'<triangle v1="{i}" v2="{j}" v3="{k}" paint_color="{code[e]}" slic3rpe:mmu_segmentation="{code[e]}"/>' for (i,j,k),e in zip(F,ext))
model=f'''<?xml version="1.0" encoding="UTF-8"?>
<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" xmlns:slic3rpe="http://schemas.slic3r.org/3mf/2017/06">
<resources><object id="1" type="model" name="TreetopDruidHaven-Top"><mesh>
<vertices>
{vs}
</vertices>
<triangles>
{ts}
</triangles>
</mesh></object></resources>
<build><item objectid="1"/></build>
</model>'''
ct='<?xml version="1.0" encoding="UTF-8"?>\n<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/></Types>'
rels='<?xml version="1.0" encoding="UTF-8"?>\n<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>'
out='/mnt/user-data/outputs/TreetopDruidHaven-Top_autocolor.3mf'
import os; os.makedirs('/mnt/user-data/outputs',exist_ok=True)
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
    z.writestr('[Content_Types].xml',ct); z.writestr('_rels/.rels',rels); z.writestr('3D/3dmodel.model',model)
print(os.path.getsize(out)/1e6,'MB')
