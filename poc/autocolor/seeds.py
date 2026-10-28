import numpy as np
V=np.load('V.npy');F=np.load('F.npy');N=np.load('N.npy')
Vs=np.load('Vs10.npy'); e1=Vs[F[:,1]]-Vs[F[:,0]]; e2=Vs[F[:,2]]-Vs[F[:,0]]
Ns=np.cross(e1,e2); Ns/=np.maximum(np.linalg.norm(Ns,axis=1),1e-12)[:,None]
C=V[F].mean(1); x,y,z=C.T
X0,X1,Y0,Y1=97.4,225.6,91.6,221.2
# deck bounds from low faces
low=(z>1)&(z<7.5)&(N[:,2]>0.8)
print('deck up-face xy bounds',np.percentile(x[low],[0.2,99.8]).round(1),np.percentile(y[low],[0.2,99.8]).round(1))
corners=np.array([[X0,Y0],[X0,Y1],[X1,Y0],[X1,Y1]])
rpost=np.min(np.hypot(x[:,None]-corners[:,0],y[:,None]-corners[:,1]),1)
# distance to rail lines (segments between posts)
def segd(px,py,ax,ay,bx,by):
    t=np.clip(((px-ax)*(bx-ax)+(py-ay)*(by-ay))/((bx-ax)**2+(by-ay)**2),0,1)
    return np.hypot(px-(ax+t*(bx-ax)),py-(ay+t*(by-ay)))
drail=np.min(np.stack([segd(x,y,X0,Y0,X0,Y1),segd(x,y,X1,Y0,X1,Y1),segd(x,y,X0,Y0,X1,Y0),segd(x,y,X0,Y1,X1,Y1)]),0)
np.savez('geo.npz',rpost=rpost,drail=drail,Ns=Ns)
for zz in [6,8,10,12,15,40,45,50,55]:
    m=(drail<4)&(rpost>12)&(np.abs(z-zz)<1)
    print('z',zz,'rail-band faces',m.sum())
