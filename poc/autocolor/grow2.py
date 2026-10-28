import numpy as np, scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
V=np.load('V.npy');F=np.load('F.npy');N=np.load('N.npy');pairs=np.load('pairs.npy')
g=np.load('geo.npz'); rpost,drail,Ns=g['rpost'],g['drail'],g['Ns']
C=V[F].mean(1); x,y,z=C.T; nf=len(F)
X0,X1,Y0,Y1=97.4,225.6,91.6,221.2; BX0,BX1,BY0,BY1=87.5,235.7,81.9,232.2
inb=(x>BX0+3)&(x<BX1-3)&(y>BY0+3)&(y<BY1-3)
dout=np.minimum.reduce([x-BX0,BX1-x,y-BY0,BY1-y])  # >0 inside
# rail normal
railn=np.where(np.minimum(np.abs(x-X0),np.abs(x-X1))<np.minimum(np.abs(y-Y0),np.abs(y-Y1)),0,1)
nal=np.abs(Ns[np.arange(nf),railn])
seed=-np.ones(nf,int)
seed[(N[:,2]>0.85)&(z>3)&(z<8)&(drail>7)&(rpost>12)&inb]=0          # deck
seed[((z<0.6)&(N[:,2]<-0.9))|((np.abs(N[:,2])<0.3)&(z<4)&(np.abs(dout)<1.5))]=1  # frame
seed[(rpost<7.5)&(z>15)&(z<58)&(np.abs(Ns[:,2])<0.5)]=2                # posts
seed[(drail<3)&(rpost>11)&(z>12)&(z<36)&(nal>0.95)]=3                 # railing
seed[((z>46)&(rpost>10))|((dout<-3)&(z>3))|(inb&(z>13)&(drail>6)&(rpost>10))]=4                          # tree
names=['deck','frame','posts','railing','tree']
for i in range(5): print(names[i],'seeds',(seed==i).sum())
# edge costs
a,b=pairs.T
d=np.linalg.norm(C[a]-C[b],axis=1)
ang=np.degrees(np.arccos(np.clip((Ns[a]*Ns[b]).sum(1),-1,1)))
conc=((C[b]-C[a])*Ns[a]).sum(1)>0
w=d*(1+np.where(conc,(ang/8)**2,0))+1e-6
G=sp.coo_matrix((w,(a,b)),shape=(nf,nf)).tocsr()
src=np.where(seed>=0)[0]
dist,pred,sources=dijkstra(G,directed=False,indices=src,min_only=True,return_predecessors=True)
lab=seed[sources]; lab[sources<0]=4

# cleanup: small islands merge into the neighbour label they touch most
from scipy.sparse.csgraph import connected_components
for it in range(3):
    same=lab[a]==lab[b]
    Gs=sp.coo_matrix((np.ones(same.sum()),(a[same],b[same])),shape=(nf,nf))
    n,cl=connected_components(Gs,directed=False)
    A=np.load('area.npy'); car=np.bincount(cl,weights=A)
    small=car<400
    diff=~same
    ea,eb=a[diff],b[diff]
    # votes: for islands, count boundary edges to each neighbor label
    rows=np.r_[cl[ea],cl[eb]]; nb=np.r_[lab[eb],lab[ea]]
    keep=small[rows]
    M=sp.coo_matrix((np.ones(keep.sum()),(rows[keep],nb[keep])),shape=(n,5)).toarray()
    newl=np.where(small&(M.sum(1)>0),M.argmax(1),-1)
    ch=newl[cl]>=0
    print('iter',it,'islands',int((small&(M.sum(1)>0)).sum()),'faces relabeled',int(ch.sum()))
    lab[ch]=newl[cl][ch]
np.save('lab.npy',lab)
A=np.load('area.npy')
for i in range(5): print(names[i],'area %',(A[lab==i].sum()/A.sum()*100).round(1))
