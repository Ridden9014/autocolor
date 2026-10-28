import numpy as np, scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
V=np.load('V.npy');F=np.load('F.npy'); nf=len(F); nv=len(V)
# edges
E=np.concatenate([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]]); fid=np.tile(np.arange(nf),3)
E.sort(1); key=E[:,0].astype(np.int64)*nv+E[:,1]
o=np.argsort(key); k=key[o]; f=fid[o]
same=k[1:]==k[:-1]
pairs=np.stack([f[:-1][same],f[1:][same]],1)
ek=np.stack([E[o][:-1][same]],0)[0]
print('edges w/2 faces',len(pairs),'nonmanifold-ish', np.sum(np.bincount(np.unique(k,return_inverse=True)[1])!=2))
np.save('pairs.npy',pairs)
# vertex adjacency & taubin smoothing
r=np.concatenate([F[:,0],F[:,1],F[:,2],F[:,1],F[:,2],F[:,0]]); c=np.concatenate([F[:,1],F[:,2],F[:,0],F[:,0],F[:,1],F[:,2]])
A=sp.csr_matrix((np.ones(len(r)),(r,c)),shape=(nv,nv)); A.data[:]=1
deg=np.asarray(A.sum(1)).ravel(); L=sp.diags(1/deg)@A
def taubin(X,it,lam=0.5,mu=-0.53):
    for i in range(it):
        X=X+lam*(L@X-X); X=X+mu*(L@X-X)
    return X
for it in [10,40,120]:
    Vs=taubin(V.copy(),it); np.save(f'Vs{it}.npy',Vs)
print('done')
