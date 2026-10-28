import numpy as np
from PIL import Image
V=np.load('V.npy');F=np.load('F.npy');N=np.load('N.npy')
C=(V[F[:,0]]+V[F[:,1]]+V[F[:,2]])/3
# extra samples: per-face 3 points near vertices for coverage
S=np.concatenate([C, (C+V[F[:,0]])/2,(C+V[F[:,1]])/2,(C+V[F[:,2]])/2])
SI=np.tile(np.arange(len(F)),4)
def rot(az,el):
    az,el=np.radians(az),np.radians(el)
    Rz=np.array([[np.cos(az),-np.sin(az),0],[np.sin(az),np.cos(az),0],[0,0,1]])
    Rx=np.array([[1,0,0],[0,np.cos(el),-np.sin(el)],[0,np.sin(el),np.cos(el)]])
    return Rx@Rz
def render(colors,az,el,res=0.3,fn='out.png'):
    R=rot(az,el); P=(S-S.mean(0))@R.T  # x right, y depth?, use: x=P0, up=P2, depth=P1
    x=P[:,0]; y=P[:,2]; d=P[:,1]
    W=int((x.max()-x.min())/res)+3; H=int((y.max()-y.min())/res)+3
    ix=((x-x.min())/res).astype(int)+1; iy=(H-2-((y-y.min())/res)).astype(int)
    pix=iy*W+ix
    o=np.lexsort((d,pix)) # smallest depth first (closest = smaller d? choose camera at -y)
    pix_s=pix[o]; first=np.r_[True,pix_s[1:]!=pix_s[:-1]]
    win=o[first]; img=np.ones((H*W,3))*0.95
    nrm=N[SI[win]]@R.T
    light=np.array([-0.4,-0.8,0.5]);light/=np.linalg.norm(light)
    sh=np.clip(nrm@light,0,1)*0.75+0.25
    img[pix_s[first]]=colors[SI[win]]*sh[:,None]
    img=(img.reshape(H,W,3)*255).astype(np.uint8)
    Image.fromarray(img).save(fn); return fn
if __name__=='__main__':
    col=np.ones((len(F),3))*0.8
    for az,el,n in [(0,15,'front'),(90,15,'right'),(180,15,'back'),(270,15,'left'),(0,70,'top')]:
        render(col,az,el,fn=f'v_{n}.png')
