import numpy as np, scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
V=np.load('V.npy');F=np.load('F.npy')
nf=len(F); nv=len(V)
# vertex-face incidence -> face components via vertex components
rows=np.concatenate([F[:,0],F[:,1],F[:,2]]); cols=np.concatenate([F[:,1],F[:,2],F[:,0]])
A=sp.coo_matrix((np.ones(len(rows)),(rows,cols)),shape=(nv,nv))
n,lab=connected_components(A,directed=False)
flab=lab[F[:,0]]
cnt=np.bincount(flab)
order=np.argsort(-cnt)
print('components',n)
for i in order[:15]:
    m=flab==i; vv=V[np.unique(F[m])]
    print(i,cnt[i],vv.min(0).round(1),vv.max(0).round(1))
np.save('flab.npy',flab)
# normals & area
e1=V[F[:,1]]-V[F[:,0]]; e2=V[F[:,2]]-V[F[:,0]]
N=np.cross(e1,e2); A2=np.linalg.norm(N,axis=1); N=N/np.maximum(A2,1e-12)[:,None]
np.save('N.npy',N); np.save('area.npy',A2/2)
print('total area',A2.sum()/2, 'mean edge', np.linalg.norm(e1,axis=1).mean())
