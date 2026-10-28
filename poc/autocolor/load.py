import numpy as np
F='/mnt/user-data/uploads/SD_RPG_TreetopDruidHavenDiceTower-Top.obj'
V=[];Fc=[]
with open(F) as f:
    for line in f:
        if line.startswith('v '): V.append(line[2:])
        elif line.startswith('f '): Fc.append(line[2:])
V=np.loadtxt(V,dtype=np.float64)
Fc=np.array([[int(t.split('/')[0]) for t in l.split()] for l in Fc])-1
np.save('V.npy',V);np.save('F.npy',Fc)
print(V.shape,Fc.shape,V.min(0),V.max(0))
