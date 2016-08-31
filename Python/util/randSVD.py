import numpy as np

def svd(input,d):
    trans=False
    if input.shape[1]>input.shape[0]:
        trans=True
        input=input.transpose()
    omega=np.random.random([input.shape[1],d])
    Q,R=np.linalg.qr(np.dot(input,omega))
    B=np.dot(Q.transpose(),input)
    U,s,V=np.linalg.svd(B)
    V=V[:,0:d]
    U=np.dot(Q,U)
    if trans:
        return V,s,U
    else:
        return U,s,V