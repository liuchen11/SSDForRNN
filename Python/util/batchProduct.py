import numpy as np

def oneXn(m1,m2):
    batch=m2.shape[0]
    assert(m1.shape[1]==m2.shape[1])
    ret=np.zeros([m2.shape[0],m1.shape[0],m2.shape[2]],dtype=np.float32)
    for i in xrange(batch):
        ret[i]=np.dot(m1,m2[i])
    return ret

def nXone(m1,m2):
    batch=m1.shape[0]
    assert(m1.shape[2]==m2.shape[0])
    ret=np.zeros([m1.shape[0],m1.shape[1],m2.shape[1]],dtype=np.float32)
    for i in xrange(batch):
        ret[i]=np.dot(m1[i],m2)
    return ret